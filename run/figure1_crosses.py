"""Generate, train, audit and plot the small cross-classification experiment.

Run from the package root with PYTHONPATH=src. Training defaults to CUDA.
All large/generated artifacts are written under the ignored artifacts tree.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from qcnn.figure1_crosses import (SIZE, CrossClassifier, audit_dataset, diagnostics,
    digest, make_dataset, source_hashes, verify_model, write_json)


@torch.no_grad()
def metrics(model, encoded, labels, weights, pixels, batch_size):
    logits = torch.cat([model.logits_encoded(x) for x in encoded.split(batch_size)])
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    correct = (logits >= 0) == labels.bool()
    return {'loss': float((weights*losses).sum()),
            'accuracy': float((weights*correct).sum()),
            'strata': {f'{count}_pixels_label{label}': float(correct[(pixels == count) & (labels == label)].float().mean())
                       for count in (5, 10) for label in (0, 1)},
            'logit_min': float(logits.min()), 'logit_max': float(logits.max())}


def train(args, kind, seed, data):
    target = args.output_root/f'{kind}_seed{seed}'
    target.mkdir(parents=True, exist_ok=True)
    config = dict(kind=kind, seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                  learning_rate=args.lr, qcs_depth=args.qcs_depth, dataset_sha256=digest(args.output_root/'dataset.npz'),
                  source_sha256=source_hashes(), runner_sha256=digest(__file__),
                  dtype='float32', initialization_std=0.3, head_initialization_std=0.1,
                  optimizer='Adam', head='256 times blank-centered summed probabilities; affine 4 to 1')
    checkpoint_path = target/'checkpoint.pt'
    if (target/'result.json').exists():
        previous = json.loads((target/'result.json').read_text())
        if previous['config'] != config:
            raise ValueError(f'Incompatible completed run: {target}; use a new output root')
        print(f'Reuse {target}', flush=True)
        return
    model = CrossClassifier(kind, seed, qcs_depth=args.qcs_depth).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, history, elapsed_before = 0, [], 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError(f'Incompatible checkpoint: {checkpoint_path}')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch, history = checkpoint['epoch'], checkpoint['history']
        elapsed_before = checkpoint['elapsed_seconds']
    images = torch.tensor(data['images'], device=args.device)
    labels = torch.tensor(data['labels'], device=args.device)
    weights = torch.tensor(data['weights'], dtype=torch.float32, device=args.device)
    pixels = torch.tensor(data['pixels'], device=args.device)
    encoded = model.encode(images).detach()
    started = time.monotonic()
    if start_epoch == 0:
        history.append(dict(epoch=0, **metrics(model, encoded, labels, weights, pixels, args.batch_size)))
    for epoch in range(start_epoch+1, args.epochs+1):
        generator = torch.Generator().manual_seed(seed*1000000+epoch)
        order = torch.randperm(len(labels), generator=generator).to(args.device)
        for selected in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model.logits_encoded(encoded[selected])
            loss_values = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[selected], reduction='none')
            # Uniform minibatches estimate the explicitly balanced full objective.
            loss = (loss_values * weights[selected]).sum() * len(labels) / len(selected)
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            record = dict(epoch=epoch, **metrics(model, encoded, labels, weights, pixels, args.batch_size))
            history.append(record)
            seconds = elapsed_before + time.monotonic()-started
            print(f'{kind} seed={seed} epoch={epoch} loss={record["loss"]:.5f} acc={record["accuracy"]:.4f} seconds={seconds:.1f}', flush=True)
            checkpoint = dict(config=config, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                              epoch=epoch, history=history, elapsed_seconds=seconds)
            temporary = target/'checkpoint.tmp'
            torch.save(checkpoint, temporary); temporary.replace(checkpoint_path)
            write_json(target/'history.json', history)
    result = dict(config=config, parameters=model.parameter_counts(), final=history[-1],
                  elapsed_seconds=elapsed_before+time.monotonic()-started,
                  checkpoint_sha256=digest(checkpoint_path), torch_version=torch.__version__,
                  device=(torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu'))
    write_json(target/'result.json', result)


def load_model(args, kind, seed):
    path = args.output_root/f'{kind}_seed{seed}'/'checkpoint.pt'
    checkpoint = torch.load(path, map_location=args.device, weights_only=False)
    if checkpoint['config']['source_sha256'] != source_hashes():
        raise ValueError('Source code differs from training checkpoint')
    model = CrossClassifier(kind, seed, qcs_depth=checkpoint['config']['qcs_depth']).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'pdf.fonttype': 42})
    output = args.output_root/'figures'; output.mkdir(exist_ok=True)
    seed = min(args.seeds)  # Deterministic example, never selected for appearance.
    arrays = {kind: np.load(args.output_root/f'{kind}_seed{seed}'/'states.npz') for kind in args.models}
    limit = max(float(np.abs(d['contributions']).max()) for d in arrays.values())
    limit = max(limit, 1e-12)
    def tile(ax, values, *, grayscale=False):
        cmap = plt.get_cmap('gray' if grayscale else 'RdBu_r')
        norm = matplotlib.colors.Normalize(0, 1) if grayscale else TwoSlopeNorm(0, -limit, limit)
        colors = cmap(norm(values.T))
        for v, u in np.ndindex(SIZE, SIZE):
            ax.add_patch(Rectangle((u-.5, v-.5), 1, 1, facecolor=colors[v,u], edgecolor='none'))
        edges = np.arange(SIZE-1)+.5
        lines = [((i,-.5),(i,SIZE-.5)) for i in edges] + [((-.5,i),(SIZE-.5,i)) for i in edges]
        ax.add_collection(LineCollection(lines, colors='#a8a8a8', linewidths=.2))
        ax.set(xlim=(-.5,SIZE-.5), ylim=(SIZE-.5,-.5), aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_linewidth(.4); spine.set_color('#808080')
    names = ['pair', 'T3_pair', 'S3_pair']
    titles = [r'$x$', r'$T_x^3x$', r'$S_x^3x$']
    fig, axes = plt.subplots(1+len(arrays), 3, figsize=(5.15, 2.0*(1+len(arrays))), squeeze=False)
    first = next(iter(arrays.values()))
    for j, name in enumerate(names):
        index = list(first['names']).index(name)
        tile(axes[0,j], first['images'][index], grayscale=True)
        axes[0,j].set_title(titles[j])
    axes[0,0].set_ylabel('Input')
    for row, (kind, data) in enumerate(arrays.items(), 1):
        for j, name in enumerate(names):
            index = list(data['names']).index(name)
            tile(axes[row,j], data['contributions'][index])
            probability = 1/(1+np.exp(-data['logits'][index]))
            axes[row,j].set_title(f'p(cross) = {probability:.3f}')
        axes[row,0].set_ylabel(kind.upper())
    fig.subplots_adjust(left=.1,right=.90,bottom=.05,top=.95,hspace=.3,wspace=.2)
    color_ax = fig.add_axes([.93,.08,.02,.5])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=TwoSlopeNorm(0,-limit,limit),cmap='RdBu_r'), cax=color_ax, label='Contribution to logit relative to blank')
    fig.savefig(output/'feature_maps.pdf'); fig.savefig(output/'feature_maps.png',dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(5.15,2.1))
    for kind in args.models:
        for seed in args.seeds:
            history = json.loads((args.output_root/f'{kind}_seed{seed}'/'history.json').read_text())
            for ax, key in zip(axes, ('loss','accuracy')):
                ax.plot([r['epoch'] for r in history],[r[key] for r in history],
                        label=f'{kind.upper()} {seed}',color=('#17658b' if kind=='pcs' else '#884366'),alpha=.65)
    axes[0].set_ylabel('Balanced training loss'); axes[1].set_ylabel('Balanced training accuracy')
    axes[1].set_ylim(0,1.02)
    for ax in axes: ax.set_xlabel('Epoch'); ax.spines[['top','right']].set_visible(False)
    axes[1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(output/'training.pdf'); fig.savefig(output/'training.png',dpi=200)
    plt.close(fig)
    summary = {f'{kind}_seed{seed}': json.loads((args.output_root/f'{kind}_seed{seed}'/'result.json').read_text())
               for kind in args.models for seed in args.seeds}
    write_json(args.output_root/'summary.json', summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','data','verify','train','diagnose','plot'],default='all')
    parser.add_argument('--output-root',type=Path,default=Path('artifacts/figure1_crosses/full_batch'))
    parser.add_argument('--models',nargs='+',choices=['pcs','qcs'],default=['pcs','qcs'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--epochs',type=int,default=2000)
    parser.add_argument('--batch-size',type=int,default=4864)
    parser.add_argument('--log-every',type=int,default=25)
    parser.add_argument('--lr',type=float,default=.01)
    parser.add_argument('--qcs-depth',type=int,default=2)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if min(args.epochs,args.batch_size,args.log_every,args.qcs_depth) < 1: parser.error('Counts must be positive')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    args.output_root.mkdir(parents=True,exist_ok=True)
    stages=['data','verify','train','diagnose','plot'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage=='data':
            data=make_dataset()
            path=args.output_root/'dataset.npz'
            if path.exists():
                with np.load(path) as stored:
                    for key,value in data.items():
                        if not np.array_equal(value,stored[key]): raise ValueError('Existing dataset differs')
            else: np.savez_compressed(path,**data)
            audit=audit_dataset(data); write_json(args.output_root/'dataset_audit.json',audit)
            print(json.dumps(audit),flush=True)
        elif stage=='verify':
            result={kind:verify_model(CrossClassifier(kind,seed=17,qcs_depth=args.qcs_depth).to(args.device)) for kind in args.models}
            write_json(args.output_root/'architecture_checks.json',result); print(json.dumps(result),flush=True)
        elif stage=='train':
            with np.load(args.output_root/'dataset.npz') as stored: data=dict(stored)
            for kind in args.models:
                for seed in args.seeds: train(args,kind,seed,data)
        elif stage=='diagnose':
            for kind in args.models:
                for seed in args.seeds:
                    result=diagnostics(load_model(args,kind,seed),args.output_root/f'{kind}_seed{seed}')
                    print(kind,seed,json.dumps(result),flush=True)
        elif stage=='plot': plot(args)


if __name__=='__main__': main()
