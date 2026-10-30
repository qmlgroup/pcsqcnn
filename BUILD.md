# Build and run guide

Run all commands from this package root. Python 3.11+ and `uv` are required.

```sh
uv sync --locked --extra test --extra notebook
```

Add `--extra mfa` for MFA training and `--extra pennylane` for the PCS
reference circuit. Training and checkpoint evaluation benefit from CUDA;
plotting saved results runs on CPU. See [artifact formats](ARTIFACTS.md) for
the data expected by each plotter and [model definitions](INFO.md) for the API.

## Figure pipeline

For all paper figures on CPU, follow the
[archive quick start](REPRODUCING.md#quick-start-all-figures-on-cpu), then use:

```sh
sh build_all_figures.sh --plots-only
```

This reads saved results and renders all numerical plots, quantum diagrams
and Figure 1 into `figs/`. It does not train, perform inference, download
MNIST or require CUDA or older artifact releases. The PCS translation
renderer also writes individual panels under
`results/pcs_translation_2026-10-25/plots/`.

Other modes are optional alternatives:

| Mode | Behavior |
| --- | --- |
| `--verify` | First reconstruct missing PCS translation arrays from supplied weights, prepare MNIST if needed, and verify the published statistics; then plot. No training. |
| No option | Prepare the main MNIST and common-FRQI comparison inputs before plotting; completed runs are reused. Missing training outputs require training. |
| `--rebuild` | Discard existing outputs of that workflow and recompute them, including training. |

To render only the quantum diagrams and Figure 1:

```sh
uv run python run/render_paper_diagrams.py
```

This requires pdfLaTeX with TikZ/PGF, Quantikz, `standalone`, `braket`, Helvetica
and `sansmath`; it reads the bundled states without training or checkpoints.
See [REPRODUCING.md](REPRODUCING.md) for the figure and data inventory.

The pipeline uses recorded PCS-translation and MFA bundles for those plots.
Their complete training/evaluation workflows and the Figure 1 illustration
are independent commands, documented below. Add `--extra mfa` when installing
the environment for MFA training.

## Main MNIST experiments

```sh
uv run python run/prepare_translated_mnist_baselines_data.py
uv run python run/prepare_full_mnist_classical_baselines_data.py
uv run python run/prepare_brightness_sweep_data.py
uv run python run/prepare_pcsqcnn_architecture_sweep_data.py
uv run python run/prepare_pcsqcnn_image_size_sweep_data.py
```

The image-size sweep supplies the `16on16` reference used for finite-shot,
readout and error diagnostics. Evaluation scripts reuse its saved checkpoints:

```sh
uv run python run/evaluate_gradient_norms.py
uv run python run/evaluate_readout_entropy.py
uv run python run/evaluate_finite_shot_snapshots.py
uv run python run/evaluate_finite_shot_loss_sampling.py
uv run python run/evaluate_readout_landscape.py
uv run python run/evaluate_error_structure.py
```

Render the corresponding figures:

```sh
uv run python run/plot_classical_baseline_architectures.py
uv run python run/plot_translated_mnist_baselines.py
uv run python run/plot_brightness_sweep.py
uv run python run/plot_pcsqcnn_architecture_sweep.py
uv run python run/plot_pcsqcnn_image_size_sweep.py
uv run python run/plot_gradient_norms.py
uv run python run/plot_readout_entropy.py
uv run python run/plot_finite_shot_accuracy.py
uv run python run/plot_finite_shot_loss_histograms.py
uv run python run/plot_readout_landscape.py
uv run python run/plot_error_structure.py
```

`plot_translated_mnist_baselines.py` also produces
`full_mnist_classical_baselines.pdf`. Test bands for the original plots follow
the aggregation rules in `ARTIFACTS.md`.

## Common FRQI comparison

The benchmark uses offsets ±8 in training and testing. The published sp-QCNN
curve has two seeds; the other curves have three. Run the continuation stages
in the stated order:

```sh
uv run python run/prepare_qcsqcnn_translated_mnist_data.py --runs fq2_ql3_sl8 --learning-rate 0.001 --generator-init-scale 0.5 --root-directory qcsqcnn_translated_mnist_tuned
uv run python run/prepare_encoding_swap_data.py --runs hur_su4_l1_frqi_train8 hur_su4_l1_frqi_train8_e6000_warm
uv run python run/prepare_sp_qcnn_translated_mnist_data.py --runs sp_qcnn_train8 --seed-count 2
uv run python run/prepare_local_pcsqcnn_translated_mnist_data.py --runs pcsqcnn_local_fq2_ql3_train8 pcsqcnn_local_fq2_ql3_train8_e4000_warm pcsqcnn_local_fq2_ql3_train8_e6000_warm
uv run python run/plot_frqi_comparison.py --output-dir figs
```

The table also includes untuned address-QCS controls; their parameters and
the validation search can be reproduced independently:

```sh
uv run python run/prepare_qcsqcnn_translated_mnist_data.py --runs fq2_ql3_sl1 fq2_ql3_sl8
uv run python run/tune_qcsqcnn_translated_mnist.py
uv run python run/count_pcs_resources.py --json
```

Continuation stages load the preceding weights and restart Adam. The plot
joins stages by seed identity, retains the actual completed epoch ranges and
uses unsmoothed means with sample-standard-deviation bands. Missing or
inconsistent histories are errors. General PCS uses the architecture-sweep
`fq2_ql3` run. All five curves use FRQI; MFA is a separate comparison.

For local PCS, use `torch.set_float32_matmul_precision("highest")` in custom
training/inference loops, as the supplied benchmark scripts do. Lower matrix
product precision can introduce unitarity errors in the local factorization.

## PCS readout and cyclic translations

```sh
uv run python run/benchmark_pcs_translation.py --data-root data
```

This trains, evaluates, verifies and plots general/local × full/summed PCS.
Use `--stage`, `--models` and `--seeds` for individual operations. The
[protocol guide](docs/pcs_translation.md) specifies all stage checkpoints,
precision settings, cyclic shifts and interpretation of the three metrics.

To re-evaluate the supplied final models, check the diagnostic statistics and
avoid retraining:

```sh
uv run python run/verify_pcs_translation.py
```

Use `--models` and `--seeds` to select individual runs. Missing inference arrays
are generated automatically from the packaged weights. Plot the compact
recorded data without training:

```sh
uv run python run/plot_pcs_translation.py --output-root results/pcs_translation_2026-10-25
```

## MFA and Figure 1

```sh
uv run --extra mfa python run/run_mfa_full_matrix.py --device cuda
uv run python run/figure1_crosses.py --device cuda
```

See [MFA](MFA_BENCHMARK.md) and [cross-pattern classification](docs/figure1_crosses.md)
for model definitions, training and independent verification. The MFA curves
can be rendered directly from the recorded CSV:

```sh
uv run python run/plot_mfa_pcs_comparison.py --curves results/mfa_translated/comparison/comparison_learning_curves.csv --output figs/pcs_mfa_translated_comparison.pdf
```

## Export compact plot inputs

```sh
uv run python run/export_plot_artifacts.py --source FULL_ARTIFACTS --destination artifacts --check
```

The exporter records the files read by the main MNIST and common FRQI
plotters, converts serialized metric histories to lossless JSON and verifies
the rendered outputs. It does not export weights. The separate PCS, MFA and
Figure 1 result bundles under `results/` describe their own required inputs.

## Inspect saved runs

```sh
uv run marimo edit notebooks/inspect_artifact.py
uv run --extra pennylane marimo edit pennylane/inspect_pennylane_artifact.py
```

Set the run directory and seed in the notebook configuration. The PennyLane
backend reconstructs the general PCS circuit from a Torch checkpoint and is
intended for inference checks rather than training.
