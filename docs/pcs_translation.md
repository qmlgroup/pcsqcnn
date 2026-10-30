# Reproducible PCS translation benchmarks

This experiment compares general and radius-one PCS layers with full and
position-summed readout. Training and per-image evaluation use a GPU; plotting
the compact results requires only a CPU. Generated weights and arrays are
stored under `artifacts/`.

## Protocol

| Model key | Multiplexers | Readout | Final epoch | Seeds |
| --- | --- | --- | ---: | --- |
| `general_full` | general | full | 2000 | 0, 1, 2 |
| `general_summed` | general | summed | 2000 | 0, 1, 2 |
| `local_full` | radius one | full | 6000 | 0, 1, 2 |
| `local_summed` | radius one | summed | 6000 | 0, 1, 2 |

All models use FRQI, three quantum layers, two feature qubits, and the original
translated-MNIST data: 1000 training examples per class, 10000 test examples,
16x16 digits on a 32x32 canvas, fixed seeded offsets in [-8,8]. Adam uses
learning rate 0.03; batch sizes are 256/1600; tests run every ten epochs.
General runs preserve the original `high` matmul setting; local runs use
`highest`. Translation diagnostics use `highest`, double precision, and exact probabilities
at the unchanged trained weights. The original checkpoint dtype is retained in
a separate audit of normalization, readout/logit differences, and all predicted
classes. Large trained generators can accumulate single-precision matrix-
exponential error; readouts are never renormalized to hide this error.

Local training consists of three 2000-epoch stages. Each stage retains its own
final checkpoint, JSON history, and metrics: the 2000-epoch result is never
replaced by the 6000-epoch result. Adam and the seeded loader restart at stage
boundaries, as in the existing continuation protocol. Checkpoints within a
stage include Adam, RNG, data fingerprints, and source hashes for exact resume.
Snapshots are saved after epoch 1 and every 100 epochs. Final-epoch results are
used regardless of accuracy; no best-test checkpoint selection is performed.

## Commands

From the package root, using the project's environment:

```sh
PYTHONPATH=src python run/benchmark_pcs_translation.py --dry-run
PYTHONPATH=src python run/benchmark_pcs_translation.py --data-root data
```

The second command trains all four variants from scratch, evaluates final
checkpoints, verifies saved outputs, and generates figures. Existing compatible
results under the same output root are reused. After interruption, run the
same command to resume; incompatible sources or configurations are rejected.
Use a different `--output-root` for an intentional fresh repetition.

The stages and models can be selected independently:

```sh
PYTHONPATH=src python run/benchmark_pcs_translation.py --stage train --models local_full --seeds 0
PYTHONPATH=src python run/benchmark_pcs_translation.py --stage evaluate --models local_full --seeds 0
PYTHONPATH=src python run/benchmark_pcs_translation.py --stage verify --models local_full --seeds 0
PYTHONPATH=src python run/benchmark_pcs_translation.py --stage plot --models local_full --seeds 0
```

Each seed has its own training directories. A root-level lock prevents two
controllers from writing the same index simultaneously; use separate output
roots for concurrent independent runs. To evaluate an individual checkpoint
without the training orchestrator:

```sh
PYTHONPATH=src python run/evaluate_pcs_translation.py \
  --checkpoint /path/to/checkpoint.pt --data-root data --output artifacts/one_map.json
```

## Reusing completed runs

`--reuse-map` takes a JSON mapping from model and cumulative epoch to checkpoint
and JSON result paths. Templates accept `{seed}` and `{epoch}`. Relative paths
are resolved relative to the map file. Every supplied file is checked before
training begins. Supply all three stages of a reused local run to retain its
convergence history and 2000-epoch comparison. Example structure:

```json
{
  "general_full": {
    "2000": {
      "checkpoint": "existing/general_full/checkpoint_final_seed{seed}.pt",
      "result": "existing/general_full/result_seed{seed}.json"
    }
  },
  "local_summed": {
    "2000": {
      "checkpoint": "existing/local_summed_e2000/checkpoint_final_seed{seed}.pt",
      "result": "existing/local_summed_e2000/result_seed{seed}.json"
    },
    "4000": {
      "checkpoint": "existing/local_summed_e4000/checkpoint_final_seed{seed}.pt",
      "result": "existing/local_summed_e4000/result_seed{seed}.json"
    },
    "6000": {
      "checkpoint": "existing/local_summed_e6000/checkpoint_final_seed{seed}.pt",
      "result": "existing/local_summed_e6000/result_seed{seed}.json"
    }
  }
}
```

## Translation maps and interpretation

Every displacement uses the same 10000 **centered** test digits. This scan is
distinct from the fixed random offsets used for the training-accuracy table.
Rows denote delta_y, columns delta_x (image tensor axes 1 and 2). Full grids
contain 32x32 distinct cyclic shifts; plots show -16..15, including wrapping.

Three panels report accuracy, the fraction of labels changed from the origin,
and mean per-image total variation of features. The latter removes only the
known coarse permutation: for displacement `(4a+r,4b+s)`, compare the aligned
readout with the original readout. Thus it needs only a 4x4 cell even for full
readout. Alignment is never applied to the classifier input. For summed
readout all three maps are 4x4. Cell annotations show means across seeds.
Sample standard deviations for all maps remain in the CSV.

Sixteen direct quantum evaluations supply the fundamental readouts. Known
coarse spatial permutations and the original head reconstruct full maps.
Separate direct simulations on 200 images cover every residue and six coarse
translations, including boundary/corner crossings. Numerical residuals and
label agreement validate reconstruction independently of the reconstructed
maps. This is not a claim of invariance under every one-pixel translation.

Summed readout demonstrates invariance of the complete classifier under the
guaranteed subgroup. It compresses 256 features to four; neither this invariant
representation nor its head is optimized for maximum classification accuracy.
Accuracy, per-image decision stability, and feature equivariance are distinct
properties and are reported separately.

## Saved outputs

- `training/`: new per-seed stage checkpoints, manifests, histories, resume states.
- `runs.json`: source paths, checkpoint/result hashes, and origin of every run.
- `records/`: portable normalized JSON copies of all reused/new histories.
- `diagnostics/`: per-seed JSON metrics and compressed arrays containing the
  sixteen quantum readouts, predictions, and labels. These bulk arrays are
  distributed separately from Git.
- `verification.json`: independent recomputation of metrics from saved arrays.
- `plots/`: PDF/PNG figures, CSV tables, final results at 2000/4000/6000, and
  unsmoothed convergence curves with sample-standard-deviation bands.
- `status.json`: progress or a recorded failure; `benchmark.lock`: process lock.

`plot` needs only `runs.json`, `records/`, and diagnostic JSON files; it does not
load weights, require CUDA, or simulate a quantum circuit. `verify` additionally
needs the NPZ arrays. `evaluate` needs the referenced weights and MNIST cache.

## Recorded results

The [numerical bundle](../results/pcs_translation_2026-10-25/README.md)
contains all four variants and their stage histories. From this package root:

```sh
uv run python run/plot_pcs_translation.py --output-root results/pcs_translation_2026-10-25 --figures-dir figs
```

Bulk arrays are needed only for independent per-image verification;
checkpoints are needed for new state simulation or continued training.

The combined `pcs_readout_full.pdf` and `pcs_readout_summed.pdf` use six
uniquely labeled panels: general layers in (a)-(c), radius-one layers in
(d)-(f). Each column shares a color scale across both rows and both figures.
Axes labeled modulo four show the 16 representative displacements, not
spatial pixels of an output image. Feature distance is the per-image total
variation from the unshifted readout, averaged over test images. For full
readout, the known coarse spatial permutation is undone first; summed
readout has no remaining spatial coordinates. Classification always uses
the unaligned readout. Plotting uses the saved diagnostic JSON data only.
