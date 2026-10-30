"""Render the trained Figure 1 components, with fixed symmetric probe positions.

Read archived states or checkpoints without training or selecting a seed by appearance.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO/'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import torch

from qcnn.figure1_crosses import (CrossClassifier, CROSS, SIZE, bit_cycle_numpy,
    cross_count, digest, image_at, source_hashes, transform, write_json)


def verify_model_sources(archived, archive):
    current = source_hashes()
    if archived == current:
        return
    # Explicitly audited source revisions; never accept an arbitrary
    # mismatch or rewrite the hashes of the sources used for training.
    compatibility = json.loads((archive / 'source_compatibility.json').read_text())
    if archived != compatibility['archived'] or current != compatibility['compatible']:
        raise ValueError('Archived model source differs')


def tile(path, colors):
    fig=plt.figure(figsize=(1,1)); ax=fig.add_axes([0,0,1,1])
    for u,v in np.ndindex(SIZE,SIZE):
        ax.add_patch(Rectangle((u,SIZE-1-v),1,1,facecolor=colors[u,v],edgecolor=colors[u,v],linewidth=.12))
    edges=np.arange(1,SIZE)
    lines=[((i,0),(i,SIZE)) for i in edges]+[((0,i),(SIZE,i)) for i in edges]
    ax.add_collection(LineCollection(lines,colors='#b3b6b5',linewidths=.3,snap=False,capstyle='butt'))
    ax.set(xlim=(0,SIZE),ylim=(0,SIZE)); ax.axis('off')
    fig.savefig(path,transparent=True); plt.close(fig)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=REPO/'artifacts/figure1_crosses/full_batch')
    parser.add_argument('--prepared-root',type=Path,help='Rebuild from compact archived states, without checkpoints')
    args=parser.parse_args()
    out=ROOT/'build/crosses'; out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    top=image_at(CROSS,(3,3)); bottom=image_at(CROSS,(4,12)); pair=np.maximum(top,bottom)
    names=['blank','top','bottom','pair','T_pair','S_pair']
    images=np.stack([np.zeros_like(pair),top,bottom,pair,np.roll(pair,8,axis=0),bit_cycle_numpy(pair,0,3)])
    images_tensor=torch.tensor(images,dtype=torch.float64)
    data={}; evidence={'positions':[[3,3],[4,12]],'translation':8,'bit_cycle_power':3,
                      'seed':0,'source_sha256':source_hashes(),'runs':{}}
    if args.prepared_root is not None:
        archive=args.prepared_root
        evidence=json.loads((archive/'figure_evidence.json').read_text())
        verify_model_sources(evidence['source_sha256'], archive)
        with np.load(archive/'figure_data.npz') as stored: data=dict(stored)
        if not np.array_equal(data['images'],images): raise ValueError('Archived probe positions differ')
        if evidence['translation']!=8 or evidence['bit_cycle_power']!=3: raise ValueError('Archived transforms differ')
    else:
        for kind in ('pcs','qcs'):
            for seed in (0,1,2):
                path=args.data_root/f'{kind}_seed{seed}'/'checkpoint.pt'
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if saved['config']['source_sha256'] != source_hashes(): raise ValueError('Model source differs from checkpoint')
                model=CrossClassifier(kind,seed,qcs_depth=saved['config']['qcs_depth'])
                model.load_state_dict(saved['state_dict']); model.double()
                inputs=model.encode(images_tensor); states=model.core(inputs)
                probabilities=model.probabilities(states)
                h=SIZE**2*torch.einsum('bxyf,f->bxy',probabilities-probabilities[:1],model.head.weight[0])
                logits=h.sum((1,2))+model.head.bias[0]
                checks={'checkpoint_sha256':digest(path),'examples':{name:{'p_cross':float(logits[i].sigmoid()),
                        'actual_crosses':cross_count(images[i])} for i,name in enumerate(names)}}
                for action,index,power in [('Tx',4,8),('Sx',5,3)]:
                    expected_state=transform(states[3:4],action,power)[0]
                    expected_map=transform(h[3:4],action,power)[0]
                    checks[action+'_state_linf']=float((states[index]-expected_state).abs().max())
                    checks[action+'_map_linf']=float((h[index]-expected_map).abs().max())
                    if seed==0:
                        data[kind+'_'+action+'_transported']=expected_map.numpy()
                if checks[('Tx' if kind=='pcs' else 'Sx')+'_state_linf']>1e-10: raise AssertionError(checks)
                if checks[('Tx' if kind=='pcs' else 'Sx')+'_map_linf']>1e-9: raise AssertionError(checks)
                evidence['runs'][f'{kind}_seed{seed}']=checks
                if seed==0:
                    data[kind+'_states']=states.numpy(); data[kind+'_inputs']=inputs.numpy()
                    data[kind+'_probabilities']=probabilities.numpy(); data[kind+'_h']=h.numpy()
                    data[kind+'_logits']=logits.numpy()
    data['images']=images; data['names']=np.asarray(names)
    # The same physical projection is used before and after both networks.
    # Feature order is (auxiliary, color), so 0 and 2 have color c=0.
    def color_probability(states):
        probabilities = np.abs(states)**2
        address_mass = probabilities.sum(axis=-1)
        if np.any(address_mass <= 0):
            raise ValueError('Conditional color is undefined at a zero-mass address')
        values = probabilities[..., 0::2].sum(axis=-1) / address_mass
        if values.min() < -1e-12 or values.max() > 1+1e-12:
            raise AssertionError('Conditional probabilities must lie in [0,1]')
        return values, address_mass

    # Historical archive keys ending in _C are retained; the figure uses p_0.
    projection_checks = {}
    for kind in ('pcs', 'qcs'):
        values, mass = color_probability(data[kind+'_states'])
        data[kind+'_C'] = values
        data[kind+'_address_mass'] = mass
        projection_checks[kind] = {'minimum_address_mass': float(mass.min()),
            'state_norm_error': float(np.abs(mass.sum((1,2))-1).max())}
        if projection_checks[kind]['state_norm_error'] > 1e-10:
            raise AssertionError('Archived states are not normalized')
        for action,index,power in [('Tx',4,8),('Sx',5,3)]:
            transported = (np.roll(values[3], power, axis=0) if action=='Tx'
                           else bit_cycle_numpy(values[3], 0, power))
            data[kind+'_'+action+'_C_transported'] = transported
            discrepancy = float(np.abs(values[index]-transported).max())
            projection_checks[kind][action+'_conditional_linf'] = discrepancy
            transported_state = (np.roll(data[kind+'_states'][3], power, axis=0) if action=='Tx'
                                 else bit_cycle_numpy(data[kind+'_states'][3], 0, power))
            state_error = float(np.abs(data[kind+'_states'][index]-transported_state).max())
            projection_checks[kind][action+'_state_linf'] = state_error
            if action == ('Tx' if kind=='pcs' else 'Sx') and state_error > 1e-10:
                raise AssertionError('Full-state symmetry failed')
            if action == ('Tx' if kind=='pcs' else 'Sx') and discrepancy > 1e-10:
                raise AssertionError(projection_checks)
    if projection_checks['qcs']['Tx_conditional_linf'] < 1e-3:
        raise AssertionError('The central noncommuting paths must differ visibly')
    inputs, input_mass = color_probability(data['pcs_inputs'])
    if not np.allclose(inputs, np.sin(np.pi*images)**2, atol=1e-14):
        raise AssertionError('Input projection differs from FRQI sin^2(vartheta)')
    if not np.allclose(input_mass, 1/SIZE**2, atol=1e-14):
        raise AssertionError('FRQI addresses must have uniform mass')
    data['input_C'] = inputs

    # Coordinated sequential palettes distinguish encoded inputs from outputs.
    # All maps retain the same physical projection and the fixed [0,1] range.
    palettes = {
        'input': ['#f4f7f0', '#d0e3d9', '#9dc5b8', '#67a496', '#397e78', '#18515a'],
        'output': ['#fff7e9', '#fdd7b0', '#f5a071', '#dc655c', '#a53654', '#511b48'],
    }
    cmaps = {}
    for name, colors in palettes.items():
        cmap = LinearSegmentedColormap.from_list(name+'_color_probability', colors, N=1024)
        rgb = cmap(np.linspace(0,1,1024))[:, :3]
        linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        luminance = linear @ np.array([.2126, .7152, .0722])
        if np.max(np.diff(luminance)) > 1e-12:
            raise AssertionError(name+' palette luminance must decrease monotonically')
        cmaps[name] = cmap
    for name,index in [('psi',3),('Tpsi',4),('Spsi',5)]:
        tile(out/f'{name}.pdf',cmaps['input'](inputs[index]))
    for kind in ('pcs','qcs'):
        for name,index in [('base',3),('T',4),('S',5)]:
            tile(out/f'{kind}_{name}.pdf',cmaps['output'](data[kind+'_C'][index]))
        tile(out/f'{kind}_Tx_transport.pdf',cmaps['output'](data[kind+'_Tx_C_transported']))
    for name,angles in [('colorbar',False),('anglebar',True)]:
        cmap = cmaps['input' if angles else 'output']
        fig=plt.figure(figsize=(.08,1)); ax=fig.add_axes([0,0,1,1])
        boundaries=np.linspace(0,1,257)
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            midpoint=(lo+hi)/2
            value=np.sin(np.pi/2*midpoint)**2 if angles else midpoint
            ax.add_patch(Rectangle((0,lo),1,hi-lo,facecolor=cmap(value),edgecolor='none'))
        ax.set(xlim=(0,1),ylim=(1,0)); ax.axis('off')
        fig.savefig(out/f'{name}.pdf',transparent=True); plt.close(fig)
    np.savez_compressed(out/'states.npz',**data)
    evidence['visualization'] = {
        'quantity': 'P(color=0 | address), marginalized over auxiliary qubit',
        'symbol': 'p_0(r)', 'common_linear_range': [0,1], 'input_relation': 'sin(vartheta)^2',
        'palettes': palettes,
        'head_weights_used': False, 'projection_checks': projection_checks,
        'scope': 'Conditional color only; address mass, phases, and auxiliary correlations are not shown.'}
    # Old archived evidence retains the original logit-map settings; do not
    # present those as the scale of the new figure.
    evidence.pop('color_limit', None); evidence.pop('color_scale', None)
    write_json(out/'evidence.json',evidence)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__': main()
