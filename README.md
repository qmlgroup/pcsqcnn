# PCS-QCNN

Companion code for *Pixel-Translation-Equivariant Quantum Convolutional Neural
Networks via Fourier Multiplexers* by Dmitry Chirkov and Igor Lobanov
([arXiv:2604.06094](https://arxiv.org/abs/2604.06094)).

The package provides the PyTorch PCS-QCNN, classical CNN/MLP and quantum
controls, the FRQI-adapted HKP and sp-QCNN comparators, and the PCA-based MFA
pipeline. PCS supports general and radius-one Fourier multiplexers, with full
or position-summed readout. An optional PennyLane backend checks the general
PCS circuit at inference.

## Install

Python 3.11+ and `uv` are required. A CUDA-capable GPU is recommended for
training; saved numerical results can be plotted on CPU.

```sh
uv sync --locked --extra test --extra notebook
```

Add `--extra mfa` for the MFA experiment or `--extra pennylane` for the
reference backend. The implementation is MIT-licensed; dependencies retain
their own licenses.

## Reproduce all paper figures on CPU

Download `pcsqcnn-source.zip`, `pcsqcnn-plot-data.zip` and `SHA256SUMS`
from the same [release](https://github.com/qmlgroup/pcsqcnn/releases).
The [quick start](REPRODUCING.md#quick-start-all-figures-on-cpu) gives the
complete checksum, extraction and installation commands. These two archives
contain everything needed to render the main-text and supplementary figures;
CUDA, the v2 archives, MNIST downloads and retraining are not required.

After extracting the data archive at the package root and installing the
Python and TeX dependencies, run:

```sh
sh build_all_figures.sh --plots-only
```

This also builds the quantum diagrams and Figure 1. Their TeX dependencies
are pdfLaTeX with TikZ/PGF, Quantikz, `standalone`, `braket`, Helvetica and
`sansmath`. The output is written to `figs/`.

The paired main-text plots can also be rebuilt together, or individually with
`--figure sweeps`, `--figure baselines`, or `--figure histograms`:

```sh
uv run python run/plot_main_panels.py --figure all
```

The optional `--verify` mode first reconstructs PCS translation diagnostics
from the supplied weights. Running the script without a mode starts the
main MNIST and FRQI experiment workflow, which may require training.
See [reproduction modes](REPRODUCING.md#reproduction-modes) before choosing
these alternatives. Generated data and weights belong in `artifacts/`.

## Numerical results

- [PCS readout and cyclic translations](results/pcs_translation_2026-10-25/README.md):
  four architectures, three seeds, convergence histories and translation maps.
- [MFA comparison](results/mfa_translated/README.md): three-seed results and
  learning histories on the translated-MNIST task.
- [Cross-pattern illustration](results/figure1_crosses/README.md): trained
  states, classification results and full-state symmetry checks for Figure 1.

The shared FRQI comparison uses the original translated-MNIST dataset:
16×16 digits on a 32×32 canvas, offsets ±8, 1000 training images per class,
and the standard 10000-image test set. It compares full-readout PCS and the
adapted HKP, sp-QCNN and address-QCS models. The sp-QCNN result uses seeds 0
and 1; the other models use seeds 0–2. MFA is reported separately because
its PCA and amplitude encoding define a different input pipeline.

## Guides

- [Paper figures, tables and release data](REPRODUCING.md)
- [Build and run commands](BUILD.md)
- [Architecture and data representation](INFO.md)
- [Artifact formats and figure inputs](ARTIFACTS.md)
- [MFA model and benchmark](MFA_BENCHMARK.md)
- [PCS readout and translation experiment](docs/pcs_translation.md)
- [Cross-pattern experiment](docs/figure1_crosses.md)

`src/qcnn/` contains the library; `run/` contains training, evaluation and
plotting entry points; `tests/` contains model and pipeline checks. Optional
inspection notebooks are in `notebooks/` and `pennylane/`.

## Artifact storage

The [release downloads](https://github.com/qmlgroup/pcsqcnn/releases) provide
the source snapshot, a compact data archive and SHA-256 checksums. All model
implementations and training scripts are included.

Extract `pcsqcnn-plot-data.zip` at the package root to create `artifacts/`.
It contains the numerical records for all plots and twelve final PCS
checkpoints for the translation study. Compact results under `results/` are
included in Git. Per-image translation arrays are regenerated from these
weights when needed instead of being distributed as a large download.

`sh build_all_figures.sh --verify` reconstructs missing translation arrays,
checks the reported statistics and renders all figures, without training.
`--plots-only` renders directly from compact records. See
[REPRODUCING.md](REPRODUCING.md) for installation and extraction.

The earlier MNIST records and checkpoints remain available in the
[`artifacts-v2` release](https://github.com/qmlgroup/pcsqcnn/releases/tag/artifacts-v2).
That archive predates the additional comparator and translation experiments.

To export the MNIST/FRQI figure inputs and comparator-table evidence from
a complete local artifact tree:

```sh
uv run python run/export_plot_artifacts.py --source FULL_ARTIFACTS --destination EXPORTED_ARTIFACTS --check
```

The exporter retains numerical values and training histories, converts local
data paths to package-relative paths, and records file checksums. It checks
that every selected plot renders identically from the export. Model weights
are excluded; the comparison table's untuned address-QCS runs and validation
search summary are included.
