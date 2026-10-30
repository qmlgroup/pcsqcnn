# Reproducing the paper

## Quick start: all figures on CPU

The paired `pcsqcnn-source.zip` and `pcsqcnn-plot-data.zip` archives are
sufficient for every main-text and supplementary figure. CUDA, older artifact
releases, MNIST downloads and retraining are not required for this workflow.

Install Python 3.11+, `uv`, and pdfLaTeX with TikZ/PGF, Quantikz, `standalone`,
`braket`, Helvetica and `sansmath` (available in a full TeX Live installation).
The commands below use a POSIX shell, `unzip` and `sha256sum`.

Download both archives and `SHA256SUMS` from the same
[release](https://github.com/qmlgroup/pcsqcnn/releases) into one directory.
Starting in that directory, run:

```sh
sha256sum --check SHA256SUMS
unzip pcsqcnn-source.zip
cd pcsqcnn
unzip ../pcsqcnn-plot-data.zip
uv sync --locked --extra notebook --extra test
sh build_all_figures.sh --plots-only
```

Both archive checksums must pass before extraction. On macOS, use
`shasum -a 256 -c SHA256SUMS` in place of `sha256sum --check SHA256SUMS`.
The source archive creates `pcsqcnn/`; the data archive creates `artifacts/`
inside it. All subsequent commands run from this package root.

The plotting command reads the bundled numerical records and saved states,
and writes the figures to `figs/`, including architecture diagrams, Figure 1
and the complete cross-model layers. Figure 1 uses the saved complex quantum
states. The original multiplexer drawing is supplied as a vector PDF.
No model training or inference is performed by `--plots-only`.

## Reproduction modes

Choose one mode; these commands are alternatives, not sequential steps.

| Command | Work performed |
| --- | --- |
| `sh build_all_figures.sh --plots-only` | Render all figures from saved results on CPU. |
| `sh build_all_figures.sh --verify` | Reconstruct and verify PCS translation diagnostics from final weights, then render all figures; no training. |
| `sh build_all_figures.sh` | Run the main MNIST and common-FRQI experiment workflow, reusing completed runs where possible, then plot. Missing training outputs require training. |
| `sh build_all_figures.sh --rebuild` | Discard existing outputs of that workflow and recompute them, including training. |

The full PCS readout, MFA and cross-pattern training workflows are documented
under [Recompute experiments](#recompute-experiments). They are separate from
the quick plotting workflow.

## Data packages

`artifacts/` contains the original MNIST and common-FRQI plot inputs,
complete per-seed histories, untuned address-QCS runs and the
validation-search summary. `artifacts/INDEX.json` maps each renderer to its
input files and hashes. Compact MFA, PCS-translation and cross-pattern
results are already in `results/` in the source archive.

`artifacts/pcs_translation_weights/` contains the twelve exact final PCS
models used in the translation study, with their architecture and weights.
Its index links the exported checkpoints to the original run hashes. These
inference checkpoints omit optimizer state and intermediate training snapshots.
The much larger per-image arrays are regenerated on demand into an ignored
cache; they are not included in the data archive or needed for plotting.

Other original MNIST checkpoints remain in
[artifacts-v2](https://github.com/qmlgroup/pcsqcnn/releases/tag/artifacts-v2).
They are optional historical data, not a dependency of this package's
plotting or translation-verification workflows. The package does not include
weights for the other comparison experiments: their figures and summary
statistics use recorded results, whereas new inference requires separately
available weights or training.

## Optional: reconstruct translation diagnostics

```sh
sh build_all_figures.sh --verify
```

This command verifies the training histories, reconstructs missing translation
arrays from final weights, compares all map statistics with the published
values, and renders the paper figures. It downloads MNIST if needed and
selects CUDA when available, otherwise CPU. No training is started.
The translation calculation took about three minutes on the experiment GPU;
CPU execution can take longer. Generated arrays are cached under
`artifacts/pcs_translation_recomputed/`, separately from the published results.
Compatible cached results are reused; changing weights, implementation or
runtime selects a different cache. Deleting the cache only requires inference
again. CUDA/CPU and library differences can affect numerical results; the
verification reports disagreements rather than overwriting the reference data.

## Figure inputs and individual renderers

| Paper figure or topic | Renderer (under `run/`) | Numerical input |
| --- | --- | --- |
| Figure 1, crosses and symmetry | `render_paper_diagrams.py` | `results/figure1_crosses/` |
| Figures 2-3, PCS and hybrid architecture | `render_paper_diagrams.py` | TeX sources and one bundled MNIST image |
| Figure 4, width/depth and resolution | `plot_main_panels.py --figure sweeps` | PCS architecture and image-size sweeps |
| Figure 5, classical and quantum controls | `plot_main_panels.py --figure baselines` | translated-MNIST and architecture runs |
| Figure 6, gradient diagnostics | `plot_gradient_norms.py` | `depth_scaling_gradient_norms/` |
| Figure 7, finite-shot accuracy | `plot_finite_shot_accuracy.py` | image-size sweep snapshot evaluations |
| Figure 8, shot-loss distributions | `plot_main_panels.py --figure histograms` | image-size sweep loss-sampling evaluations |
| FRQI angle range | `plot_brightness_sweep.py` | `pcsqcnn_brightness_sweep/` |
| Classical architecture and centered MNIST | `plot_classical_baseline_architectures.py`, `plot_translated_mnist_baselines.py` | classical models and full-MNIST runs |
| Readout entropy and landscape | `plot_readout_entropy.py`, `plot_readout_landscape.py` | image-size sweep evaluations |
| Confusion matrix and errors | `plot_error_structure.py` | image-size sweep error records |
| Common-FRQI convergence | `plot_frqi_comparison.py` | general/local PCS, HKP, sp-QCNN, tuned address-QCS |
| PCS translation maps and convergence | `plot_pcs_translation.py --output-root results/pcs_translation_2026-10-25 --figures-dir figs` | four-model compact bundle |
| MFA comparison | `plot_mfa_pcs_comparison.py --curves results/mfa_translated/comparison/comparison_learning_curves.csv --output figs/pcs_mfa_translated_comparison.pdf` | MFA and general PCS CSV histories |
| Multiplexer, Fourier junctions, local circuit, complete cross layers | `render_paper_diagrams.py` | bundled diagram sources |

## Tables and independent checks

`run/count_pcs_resources.py --json` computes the single-qubit and CNOT counts
for the specified Fourier-based implementation. It does not estimate hardware
routing or wall-clock time.

The common-FRQI table uses final per-seed test accuracies from the runs in
`plot_frqi_comparison.py`, together with the untuned address-QCS K=1 and K=8
records indexed in `artifacts/INDEX.json`. The table reports mean and sample
standard deviation across seeds, with two seeds for sp-QCNN and three for
the other models. The convergence bands use the same unsmoothed aggregation.
MFA is separate because its PCA/amplitude input differs from FRQI.

The translation table and maps come from
`results/pcs_translation_2026-10-25/summary.json` and `diagnostics/*.json`;
their convergence histories are in `records/`. Reconstruct missing arrays, verify the histories and independently recalculate
all saved translation statistics with:

```sh
uv run python run/verify_pcs_translation.py
```

Select an individual model and seed with `--models local_full --seeds 0`.
Use `--device cpu` or `--device cuda` to override automatic device selection.
The command handles MNIST preparation, inference, direct symmetry checks and
comparison with the reference statistics in that order.

For MFA, `results/mfa_translated/comparison/` contains the numerical summary
and full learning curves. `SHA256SUMS` in that bundle verifies its files.
For Figure 1, `manifest.json` verifies the compact states, histories and
checks; `figure_evidence.json` records the full-state commutators. The
renderer checks model-source hashes before using the saved states, including
the explicitly audited source revisions recorded in `source_compatibility.json`.

Run the model, dataset, reference-backend and plotting tests with:

```sh
uv run --extra test --extra mfa pytest -q
```

The optional comparison with the original MFA authors' code is documented
in [MFA_BENCHMARK.md](MFA_BENCHMARK.md); it requires that separately downloaded
source. CUDA-specific checks skip when no CUDA device is available.

## Recompute experiments

[BUILD.md](BUILD.md) gives the individual training and evaluation commands,
including warm restarts, the address-QCS validation search and table-only
controls. `sh build_all_figures.sh` runs the original MNIST and common-FRQI
experiments before plotting. The four-model PCS readout study, MFA and the
cross illustration have their own complete workflows:

```sh
uv run python run/benchmark_pcs_translation.py --data-root data
uv run --extra mfa python run/run_mfa_full_matrix.py --device cuda
uv run python run/figure1_crosses.py --device cuda
```

Manifest-based preparation commands skip completed runs and preserve
nonempty directories from interrupted runs. They stop with an explanatory
error instead of deleting partial results. Resume through the experiment's
resume command where supported, choose a new directory, or use `--rebuild`
to explicitly discard the existing outputs and restart.

MNIST is downloaded by the dataset preparation code into `data/`. The
translated datasets are regenerated with the recorded seeds and preprocessing;
the cross dataset is generated in code. These release archives do not
redistribute the original MNIST dataset. See the experiment guides for split
definitions, epoch counts and precision settings.
