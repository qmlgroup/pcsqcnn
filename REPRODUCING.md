# Reproducing the paper

Commands below run from the repository root. Install the locked environment
with `uv sync --locked --extra notebook --extra test` (add `--extra mfa` for MFA).
Rendering and numerical verification run on CPU. Training is a separate,
much longer operation; CUDA is recommended.

## Data packages

Download the required files from the
[artifacts-v3 release](https://github.com/qmlgroup/pcsqcnn/releases/tag/artifacts-v3):

- [Plot data](https://github.com/qmlgroup/pcsqcnn/releases/download/artifacts-v3/pcsqcnn-plot-data.zip)
  for all recorded-result figures and comparison tables.
- [Translation arrays](https://github.com/qmlgroup/pcsqcnn/releases/download/artifacts-v3/pcsqcnn-translation-arrays.zip)
  for the optional independent per-image checks.
- [SHA256SUMS](https://github.com/qmlgroup/pcsqcnn/releases/download/artifacts-v3/SHA256SUMS)
  to verify the downloaded archives.

Use the code at tag `artifacts-v3`, or extract
[pcsqcnn-source.zip](https://github.com/qmlgroup/pcsqcnn/releases/download/artifacts-v3/pcsqcnn-source.zip)
and work inside its `pcsqcnn/` directory. Place the data archives and
`SHA256SUMS` at that repository root, then verify and extract them:

```sh
sha256sum --check --ignore-missing SHA256SUMS
unzip pcsqcnn-plot-data.zip
# Optional, for checking the translation statistics from per-image data:
unzip pcsqcnn-translation-arrays.zip
```

`pcsqcnn-plot-data.zip` creates `artifacts/` with the numerical inputs for
the original MNIST figures and the common-FRQI comparison. It includes complete
per-seed histories, untuned address-QCS runs and the validation-search summary.
It does not include model weights. `artifacts/INDEX.json` maps each renderer
to its input files and their SHA-256 digests. Compact MFA, PCS-translation and
cross-pattern results are already in `results/`.

`pcsqcnn-translation-arrays.zip` adds twelve NPZ files beside the diagnostic
JSON records in `results/pcs_translation_2026-10-25/diagnostics/`. These hold
per-image predictions and feature probabilities, not trained weights.
Archive checksums are in the accompanying `SHA256SUMS`; the diagnostic JSON
files also record each NPZ checksum.

The earlier [artifacts-v2 release](https://github.com/qmlgroup/pcsqcnn/releases/tag/artifacts-v2)
contains checkpoints for the original MNIST experiments. The new plots do
not require them. Re-evaluating a model on new images requires its weights
or rerunning its training. Seeds and configuration records make reruns
reproducible, but numerical libraries and devices can affect the fitted weights.
The compact archives preserve the exact reported measurements.

## All figures from recorded results

```sh
sh build_all_figures.sh --plots-only
uv run python run/render_paper_diagrams.py
```

The first command renders all numerical plots and the classical architecture
diagrams. The second renders Figure 1, the complete cross-model layers and
the quantum diagrams. It requires pdfLaTeX with TikZ/PGF, Quantikz, `standalone`,
`braket`, Helvetica and `sansmath` (available in a full TeX Live installation).
All paper assets are written to `figs/`. Quantum diagram sources are in
`figsrc/`; the original multiplexer drawing is supplied as a vector PDF.
Figure 1 is reconstructed from the saved complex states, not schematic
feature maps or selected training reruns.

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
their convergence histories are in `records/`. Verify the large arrays,
histories and all saved translation statistics with:

```sh
uv run python run/verify_pcs_translation.py --output-root results/pcs_translation_2026-10-25
```

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
