"""Reproduce four PCS translation benchmarks, individually or in one command.

General multiplexers: 2000 epochs. Local multiplexers: 3 x 2000 epochs with
fresh Adam between stages. Every stage is retained, including epoch 2000.
An optional reuse map supplies existing checkpoints and result files; missing
entries are trained. Each seed has its own resumable output directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import platform

import torch

from qcnn.automation import run_mnist_auto_training
from qcnn.serialization import load_model_checkpoint
from prepare_local_pcsqcnn_translated_mnist_data import (
    LocalPcsqcnnTranslatedMnistDefaults, iter_local_pcsqcnn_training_configs,
)
from evaluate_summed_translation import file_hash, write_json


ROOT = Path(__file__).resolve().parents[1]
MODELS = {"general_full": ("general", False, 2000),
          "general_summed": ("general", True, 2000),
          "local_full": ("local_3x3", False, 6000),
          "local_summed": ("local_3x3", True, 6000)}


def read_plain(path):
    """Read both the original typed JSON exports and the new plain results."""
    def plain(obj):
        if isinstance(obj, list):
            return [plain(x) for x in obj]
        if isinstance(obj, dict):
            if "__dataclass__" in obj and "fields" in obj:
                return plain(obj["fields"])
            return {k: plain(v) for k, v in obj.items()}
        return obj
    return plain(json.loads(Path(path).read_text()))


def training_configs(name, seed, data_root, output_root, device, initial_paths=None):
    kind, summed, final_epoch = MODELS[name]
    defaults = LocalPcsqcnnTranslatedMnistDefaults(
        data_root=data_root, artifacts_root=output_root, device=device,
        download=False, seed_count=1, base_seed=seed,
        torch_matmul_precision="highest" if kind == "local_3x3" else "high")
    run = "pcsqcnn_local_fq2_ql3_reduced_train8" if summed else "pcsqcnn_local_fq2_ql3_train8"
    base, = iter_local_pcsqcnn_training_configs(defaults, runs=[run])
    base = replace(base, model=replace(base.model, constructor_kwargs={
        **base.model.constructor_kwargs, "multiplexer_kind": kind}))
    stages = []
    for total in range(2000, final_epoch+1, 2000):
        previous = total-2000
        initial = None
        if previous:
            initial = str((initial_paths or {}).get(previous,
                output_root/f"training/{name}/e{previous}/seed{seed}/checkpoint_final_seed{seed}.pt"))
        stages.append(replace(base,
            training=replace(base.training, init_checkpoint=initial,
                             snapshot_epochs=(1, *range(100, 2001, 100))),
            output=replace(base.output, directory_name=f"training/{name}/e{total}/seed{seed}")))
    return stages


def provenance(threads):
    files = sorted((ROOT/"src/qcnn").rglob("*.py"))
    files += [ROOT/n for n in ("pyproject.toml", "uv.lock",
              "run/benchmark_pcs_translation.py", "run/prepare_local_pcsqcnn_translated_mnist_data.py",
              "run/evaluate_summed_translation.py")]
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "threads": threads,
            "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in files}}


def reuse_entries(path):
    if path is None:
        return {}
    result = read_plain(path)
    if set(result)-set(MODELS):
        raise ValueError("Unknown model in reuse map")
    for name, epochs in result.items():
        for epoch, entry in epochs.items():
            if int(epoch) not in range(2000, MODELS[name][2]+1, 2000):
                raise ValueError(f"Unexpected reuse epoch: {name}/{epoch}")
            for key in ("checkpoint", "result"):
                p = Path(entry[key]).expanduser()
                entry[key] = str(p if p.is_absolute() else path.resolve().parent/p)
    return result


def validate_entry(entry, name, seed, total):
    for key in ("checkpoint", "result"):
        p = Path(entry[key])
        if not p.is_file():
            raise FileNotFoundError(p)
        digest = file_hash(p)
        if key+"_sha256" in entry and digest != entry[key+"_sha256"]:
            raise ValueError(f"Input changed: {p}")
        entry[key+"_sha256"] = digest
    result = read_plain(entry["result"])
    history = result["training_history"]
    if result["seed"] != seed or history["train_epoch"] != list(range(1, 2001)):
        raise ValueError(f"Wrong seed or incomplete history: {entry['result']}")
    if result.get("epoch_offset", total-2000) != total-2000:
        raise ValueError("Unexpected epoch offset")
    config = result["resolved_config"]
    data = config["dataset"]
    if any(data[k] != v for k, v in {"image_size": 32, "scaled_image_size": 16,
                                    "max_offset": 8, "samples_per_class": 1000}.items()):
        raise ValueError("Reused training data do not match the benchmark")
    if data.get("test_max_offset") not in (None, 8):
        raise ValueError("Unexpected test offsets")
    loaded = load_model_checkpoint(entry["checkpoint"], map_location="cpu")
    model = loaded.model
    kind, summed, _ = MODELS[name]
    if (model.multiplexer_kind != kind or model.reduce_readout_to_feature_distribution != summed
            or model.image_size != 32 or model.feature_qubits != 2 or model.quantum_layers != 3
            or model.encoding != "frqi" or model.shot_budget is not None):
        raise ValueError(f"Checkpoint architecture mismatch: {entry['checkpoint']}")
    entry.update(model=name, seed=seed, total_epoch=total,
                 parameters=sum(p.numel() for p in model.parameters()))
    return entry


def resolve_runs(args, record, status):
    root = args.output_root
    index_path = root/"runs.json"
    index = read_plain(index_path) if index_path.exists() else {"runs": {}}
    reused = reuse_entries(args.reuse_map)
    def retain_record(entry):
        relative = f"records/{entry['model']}/e{entry['total_epoch']}/seed{entry['seed']}.json"
        write_json(root/relative, read_plain(entry["result"]))
        entry.update(record_file=relative, record_sha256=file_hash(root/relative))
        return entry
    # Validate all supplied inputs before starting a potentially long run.
    prepared = {}
    for name in args.models:
        for total, spec in reused.get(name, {}).items():
            for seed in args.seeds:
                entry = {k: v.format(seed=seed, epoch=total) for k, v in spec.items()}
                entry["origin"] = "existing checkpoint"
                prepared[name, seed, int(total)] = validate_entry(entry, name, seed, int(total))
    for name in args.models:
        index["runs"].setdefault(name, {})
        for seed in args.seeds:
            entries = index["runs"][name].setdefault(str(seed), {})
            initial_paths = {total: Path(entry["checkpoint"]) for (n, s, total), entry in prepared.items()
                             if (n, s) == (name, seed)}
            for total, entry in entries.items():
                initial_paths[int(total)] = Path(entry["checkpoint"])
            configs = training_configs(name, seed, args.data_root, root, args.device, initial_paths)
            for stage, config in enumerate(configs):
                total = (stage+1)*2000
                if str(total) in entries:
                    validate_entry(entries[str(total)], name, seed, total)
                    if (name, seed, total) in prepared:
                        candidate = prepared[name, seed, total]
                        if candidate["checkpoint_sha256"] != entries[str(total)]["checkpoint_sha256"]:
                            raise ValueError("Reuse map conflicts with an indexed checkpoint")
                    continue
                if (name, seed, total) in prepared:
                    entries[str(total)] = retain_record(prepared[name, seed, total])
                    write_json(index_path, index)
                    continue
                if args.stage not in ("all", "train"):
                    raise FileNotFoundError(f"Missing {name}, seed {seed}, epoch {total}; run train first")
                directory = root/config.output.directory_name
                def report(run_seed, epoch, train_summary, test_summary, history):
                    if epoch == 1 or test_summary is not None:
                        values = {"status": "training", "model": name, "seed": run_seed,
                                  "epoch": total-2000+epoch,
                                  "train_accuracy": train_summary.metrics["accuracy"],
                                  "test_accuracy": None if test_summary is None else test_summary.metrics["accuracy"]}
                        status(**values)
                        write_json(directory/f"history_seed{run_seed}.json",
                                   {"epoch_offset": total-2000, "training_history": asdict(history)})
                        print(json.dumps(values), flush=True)
                status(status="training", model=name, seed=seed, epoch=total-2000)
                trained = run_mnist_auto_training(config, resume=directory.exists(), save_training_state=True,
                                                 implementation=record, seed_epoch_end_callback=report)
                run, = trained.runs
                result_path = directory/f"result_seed{seed}.json"
                write_json(result_path, {"seed": seed, "epoch_offset": total-2000,
                    "resolved_config": run.resolved_config, "training_history": asdict(run.history),
                    "final_summaries": {"train": asdict(run.final_train_summary),
                                        "test": asdict(run.final_test_summary)}})
                entry = {"checkpoint": str(directory/f"checkpoint_final_seed{seed}.pt"),
                         "result": str(result_path), "origin": "new training run"}
                entries[str(total)] = retain_record(validate_entry(entry, name, seed, total))
                write_json(index_path, index)
                del trained, run
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    return index


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "train", "evaluate", "verify", "plot"), default="all")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/pcs_translation"))
    parser.add_argument("--reuse-map", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("Seeds must be unique nonnegative integers")
    if len(set(args.models)) != len(args.models) or args.threads < 1:
        parser.error("Models must be unique and threads positive")
    args.data_root, args.output_root = args.data_root.resolve(), args.output_root.resolve()
    return args


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps({name: {str(seed): [asdict(c) for c in training_configs(
            name, seed, args.data_root, args.output_root, args.device)] for seed in args.seeds}
            for name in args.models}, indent=2, default=str))
        return
    torch.set_num_threads(args.threads)
    if args.stage in ("all", "train", "evaluate") and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU fallback is disabled")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root/"benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = provenance(args.threads)
        state = {"pid": os.getpid(), "requested_stage": args.stage,
                 "models": args.models, "seeds": args.seeds}
        def status(**values):
            state.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_root/"status.json", state)
        try:
            status(status="starting")
            if args.stage in ("plot", "verify"):
                index = read_plain(args.output_root/"runs.json")
            else:
                index = resolve_runs(args, record, status)
            if args.stage in ("all", "evaluate"):
                from evaluate_pcs_translation import evaluate_checkpoint
                for name in args.models:
                    for seed in args.seeds:
                        # Preserve checkpoints at every stage; calculate the
                        # full diagnostic maps only at the specified final epoch.
                        entry = index["runs"][name][str(seed)][str(MODELS[name][2])]
                        status(status="evaluating", model=name, seed=seed)
                        output = args.output_root/"diagnostics"/f"{name}_seed{seed}.json"
                        evaluate_checkpoint(entry["checkpoint"], output, data_root=args.data_root, device=args.device)
            if args.stage in ("all", "verify"):
                from verify_pcs_translation import verify
                status(status="verifying")
                verify(args.output_root, args.models, args.seeds)
            if args.stage in ("all", "plot"):
                from plot_pcs_translation import plot
                status(status="plotting")
                plot(args.output_root, args.models, args.seeds)
            status(status="complete")
        except BaseException as error:
            status(status="failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
