# `qcnn` Build and Run Guide

This document covers environment setup, artifact generation, figure rendering,
inspection notebooks, and other operational commands.

For architecture, model families, tensor/state conventions, and common
repository-wide concepts, see `INFO.md`.
For the detailed description of generated artifacts, payloads, and rendered
figure outputs, see `ARTIFACTS.md`.

## Environment Setup

The project targets Python 3.11+ and uses `uv` for environment management.

Install the default build and notebook extras:

```bash
uv sync --extra test --extra notebook
```

## Optional Extras

If you also want the optional PennyLane reference backend, install that extra
too:

```bash
uv sync --extra test --extra notebook --extra pennylane
```

## Hardware Guidance

- Full rebuilds from raw MNIST data are compute-heavy. A CUDA-capable NVIDIA GPU is strongly recommended.
- Figure-only rendering from already prepared artifacts is much lighter and is usually practical on CPU.
- The most GPU-oriented stages are the checkpoint reevaluation scripts, especially finite-shot loss sampling and readout-landscape evaluation.
- Exact runtime depends strongly on hardware and on whether the required artifacts already exist.

## Full One-Shot Build

To run the full currently wired pipeline from raw inputs:

```bash
sh build_all_figures.sh
```

That one-shot shell entrypoint runs every currently wired figure-generation
step in dependency order, but by default it does not force a rebuild of
existing saved artifacts.

This writes rendered PDFs to `figs/`, stores generated intermediate results in
`artifacts/`, and downloads MNIST data into `data/` when needed.

## Rebuild From Scratch

If you want to discard completed outputs and recompute all prep/evaluation
steps from scratch, use:

```bash
sh build_all_figures.sh --rebuild
```

Passing `--rebuild` forces every prep/evaluation step to forward its existing
`--rebuild` flag before the rendering stage runs.

## Using Precomputed Artifacts

The Git repository is intended to stay lightweight, so the full artifact tree
should be distributed separately as a companion artifact archive or release
asset.

After downloading that archive, unpack it into the repository-root
`artifacts/` directory so that paths such as
`artifacts/pcsqcnn_image_size_sweep/16on16/` exist locally.

With precomputed artifacts in place, you can either:

- rerun only the plotting scripts in `run/plot_*.py`, or
- run `sh build_all_figures.sh` and let the prep/evaluation steps reuse the completed outputs already present in `artifacts/`

## Generate Training Artifacts

Run the training/data-preparation entrypoints you need first.

- the four fixed translated MNIST runs used by Figure 2
- the full-MNIST classical control runs used by the supplementary classical comparison
- the PCS-QCNN brightness sweep artifacts
- the translated MNIST PCS-QCNN architecture-sweep artifacts used by Figure 5a
- the full-MNIST pairwise size sweep artifacts used by Figure 5b
- the canonical `16on16` subrun inside that same Figure 5b family, used as the shared reference source for Figures 6, 7, S2, S3, and S4

```bash
uv run python run/prepare_translated_mnist_baselines_data.py
uv run python run/prepare_full_mnist_classical_baselines_data.py
uv run python run/prepare_brightness_sweep_data.py
uv run python run/prepare_pcsqcnn_architecture_sweep_data.py
uv run python run/prepare_pcsqcnn_image_size_sweep_data.py
```

That `prepare_pcsqcnn_image_size_sweep_data.py` step also creates the canonical
reference subrun `pcsqcnn_image_size_sweep/16on16`, which is reused by Figures
6, 7, S2, S3, and S4. The diagnostic scripts that operate on a single saved
run default to the first saved seed from that manifest; pass `--seed` to
override it.

If you need to discard already completed outputs and recompute them from
scratch, rerun the relevant prep script with `--rebuild`:

```bash
uv run python run/prepare_pcsqcnn_image_size_sweep_data.py --rebuild
```

## Generate Derived Evaluation Artifacts

Figure 6 needs a separate finite-shot reevaluation pass on the saved reference
checkpoints from the canonical `16on16` run:

```bash
uv run python run/evaluate_finite_shot_snapshots.py
```

Figure S2a evaluates initialization-time RMS quantum gradients in a
depth-scaling PCS-QCNN family. The scientific configuration is fixed at the
top of `run/evaluate_gradient_norms.py`: `n_i=1`, `n_f=3`, `Q=1..8`,
`12` parameter seeds, and a deterministic class-balanced `256`-sample MNIST
test subset. The script also refuses to start when those explicit constants are
raised beyond the local safety guard for this machine:

```bash
uv run python run/evaluate_gradient_norms.py
```

Figure S2b uses the same reference checkpoints and computes per-sample Shannon
entropy of one sampled full-readout histogram for each shot budget:

```bash
uv run python run/evaluate_readout_entropy.py
```

Figure 7 uses the same reference checkpoints but needs repeated finite-shot
batch-loss sampling. By default the script uses `cuda` when available:

```bash
uv run python run/evaluate_finite_shot_loss_sampling.py
```

Figure S3 also uses the same reference checkpoints, but computes a GPU-optimized
loss landscape over exact readout distributions perturbed along epoch-local
shot-noise PCA directions:

```bash
uv run python run/evaluate_readout_landscape.py
```

Figure S4 uses the saved Figure 5b pairwise size sweep and extracts an
error-analysis payload from the default `16on16` run:

```bash
uv run python run/evaluate_error_structure.py
```

## Render Figures

All current figure scripts write PDFs into `figs/` by default.

Figure S1:

```bash
uv run python run/plot_classical_baseline_architectures.py
```

Figure 2 and the supplementary full-MNIST classical control:

```bash
uv run python run/plot_translated_mnist_baselines.py
```

Brightness-sweep helper figures:

```bash
uv run python run/plot_brightness_sweep.py
```

Figure 5a:

```bash
uv run python run/plot_pcsqcnn_architecture_sweep.py
```

Figure 5b:

```bash
uv run python run/plot_pcsqcnn_image_size_sweep.py
```

Figure S2a:

```bash
uv run python run/plot_gradient_norms.py
```

Figure S2b:

```bash
uv run python run/plot_readout_entropy.py
```

Figure 6:

```bash
uv run python run/plot_finite_shot_accuracy.py
```

Figure 7:

```bash
uv run python run/plot_finite_shot_loss_histograms.py
```

Figure S3:

```bash
uv run python run/plot_readout_landscape.py
```

Figure S4:

```bash
uv run python run/plot_error_structure.py
```

## Inspect Saved Artifacts in `marimo`

The repository includes small `marimo` notebooks for interactive inspection of
saved runs.

To inspect a saved Torch run, open:

```bash
uv run marimo edit notebooks/inspect_artifact.py
```

Before running the notebook, set `run_directory` and `seed` in the
configuration cell to point to the saved run you want to inspect under
`artifacts/`. The notebook reconstructs the saved run, renders its convergence
plots, re-evaluates it on the saved MNIST test configuration, and shows
article-style error analysis.

For model development and end-to-end experimentation, you can also open:

```bash
uv run marimo edit notebooks/train_qcnn.py
```

That notebook is oriented toward interactive training and pipeline exploration
rather than post-hoc artifact inspection.

## Run the PennyLane Reference

The PennyLane path is optional and inference-only. It reconstructs a readable
PennyLane reference model from a saved Torch run rather than providing a
separate training stack.

First install the optional extra if needed:

```bash
uv sync --extra test --extra notebook --extra pennylane
```

Then open the PennyLane inspection notebook:

```bash
uv run marimo edit pennylane/inspect_pennylane_artifact.py
```

Set `run_directory` and `seed` in the notebook to the saved run you want to
inspect. The notebook rebuilds `PennyLanePCSQCNN` from the Torch weights,
reconstructs the saved MNIST test loader, evaluates predictions, and displays
the resulting confusion matrix.

## Optional Development Commands

Run the test suite:

```bash
uv run --extra test pytest
```
