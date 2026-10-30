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

## Reproduce figures

With the MNIST artifact inputs under `artifacts/`, run:

```sh
sh build_all_figures.sh --plots-only
uv run python run/render_paper_diagrams.py
```

The diagram command requires pdfLaTeX with TikZ, Quantikz, `standalone`,
`braket`, Helvetica and `sansmath`.

The paired main-text plots can also be rebuilt together, or individually with
`--figure sweeps`, `--figure baselines`, or `--figure histograms`:

```sh
uv run python run/plot_main_panels.py --figure all
```

The same entry point without `--plots-only` prepares the main MNIST and FRQI
comparison inputs before rendering. Training for the PCS readout experiment,
MFA and the cross-pattern illustration has separate commands in the guides
below. Generated data and weights belong in `artifacts/`, and figures in `figs/`.

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

The [artifacts-v3 release](https://github.com/qmlgroup/pcsqcnn/releases/tag/artifacts-v3)
contains the source snapshot, numerical data and SHA-256 checksums for this
revision. All model implementations and training scripts are included;
trained weights from the new experiments are not part of this release.

Numerical records sufficient for all plots are supplied as a separate
`pcsqcnn-plot-data.zip` archive, extracted at the package root to create
`artifacts/`. The compact result bundles under `results/` are included in Git.
The optional `pcsqcnn-translation-arrays.zip` supplies the per-image arrays
used to verify the cyclic-translation statistics independently. Neither
archive is needed for training from scratch. See [REPRODUCING.md](REPRODUCING.md)
for extraction, checksums and the distinction between plotting, verification
and rerunning an experiment.

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
