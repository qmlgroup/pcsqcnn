# `qcnn`

Companion code for *Pixel-Translation-Equivariant Quantum Convolutional Neural Networks via Fourier Multiplexers* by Dmitry Chirkov and Igor Lobanov ([arXiv:2604.06094](https://arxiv.org/abs/2604.06094)).

This repository contains the Torch-first PCS-QCNN implementation, the figure-generation pipeline used for the paper, and an optional PennyLane reference backend for readable inference-only circuit inspection.

## Additional Documentation

The files below provide more detailed project-specific documentation beyond this landing page:

- [`INFO.md`](INFO.md): architecture, model families, data/state representation, and other repository-wide concepts
- [`BUILD.md`](BUILD.md): detailed build instructions, manual rebuild workflows, and script-level commands
- [`ARTIFACTS.md`](ARTIFACTS.md): figure map, artifact contracts, and notes on which data are computed for each figure

## What Is In This Repository

- `src/qcnn/`: core library code for the Torch implementation
- `run/`: data-preparation, evaluation, and plotting entrypoints for the paper figures
- `build_all_figures.sh`: one-shot entrypoint that runs the currently wired figure pipeline in dependency order
- `pennylane/`: optional readable reference implementation for full-readout inference

## Installation

The project targets Python 3.11+ and uses `uv` for environment management.

```bash
uv sync --extra test --extra notebook
```

If you also want the optional PennyLane reference backend, install that extra too:

```bash
uv sync --extra test --extra notebook --extra pennylane
```

## Build All Figures

To run the full currently wired pipeline from raw inputs:

```bash
sh build_all_figures.sh
```

This writes rendered PDFs to `figs/`, stores generated intermediate results in `artifacts/`, and downloads MNIST data into `data/` when needed.

## Hardware Guidance

- Full rebuilds from raw MNIST data are compute-heavy. A CUDA-capable NVIDIA GPU is strongly recommended.
- Figure-only rendering from already prepared artifacts is much lighter and is usually practical on CPU.
- The most GPU-oriented stages are the checkpoint reevaluation scripts, especially finite-shot loss sampling and readout-landscape evaluation.
- Exact runtime depends strongly on hardware and on whether the required artifacts already exist.

## Using Precomputed Artifacts

The Git repository is intended to stay lightweight, so the full artifact tree should be distributed separately as a companion artifact archive or release asset.

After downloading that archive, unpack it into the repository-root `artifacts/` directory so that paths such as
`artifacts/pcsqcnn_image_size_sweep/16on16/` exist locally.

With precomputed artifacts in place, you can either:

- rerun only the plotting scripts in `run/plot_*.py`, or
- run `sh build_all_figures.sh` and let the prep/evaluation steps reuse the completed outputs already present in `artifacts/`

## Inspect Saved Artifacts in Marimo

The repository includes small `marimo` notebooks for interactive inspection of saved runs.

To inspect a saved Torch run, open:

```bash
uv run marimo edit notebooks/inspect_artifact.py
```

Before running the notebook, set `run_directory` and `seed` in the configuration cell to point to the saved run you want to inspect under `artifacts/`. The notebook reconstructs the saved run, renders its convergence plots, re-evaluates it on the saved MNIST test configuration, and shows article-style error analysis.

For model development and end-to-end experimentation, you can also open:

```bash
uv run marimo edit notebooks/train_qcnn.py
```

That notebook is oriented toward interactive training and pipeline exploration rather than post-hoc artifact inspection.

## Run the PennyLane Reference

The PennyLane path is optional and inference-only. It reconstructs a readable PennyLane reference model from a saved Torch run rather than providing a separate training stack.

First install the optional extra:

```bash
uv sync --extra test --extra notebook --extra pennylane
```

Then open the PennyLane inspection notebook:

```bash
uv run marimo edit pennylane/inspect_pennylane_artifact.py
```

Set `run_directory` and `seed` in the notebook to the saved run you want to inspect. The notebook rebuilds `PennyLanePCSQCNN` from the Torch weights, reconstructs the saved MNIST test loader, evaluates predictions, and displays the resulting confusion matrix.

## Notes

- The main runtime is Torch-based.
- PennyLane support is optional and inference-only.
- The repository contains code and documentation for reproducing paper artifacts and figures, not just the final plotting scripts.
