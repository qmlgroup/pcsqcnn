# PCS readout and cyclic translations

Four PCS variants, seeds 0–2, on the FRQI translated-MNIST task. General
layers are trained for 2000 epochs and radius-one layers for 6000, retaining
the 2000- and 4000-epoch stage results. Full readout has 256 features and
2570 head parameters; summed readout has four features and 50 head parameters.

`records/` contains complete stage histories; `runs.json` indexes them with
record and checkpoint hashes. `diagnostics/` contains 12 final-checkpoint
translation maps, independent reconstruction checks and precision audits.
`summary.json` and `report.md` give the final metrics. Portable paths are
relative to the package root; records are relative to this bundle.

Each map uses the same 10000 centered test digits. Full-readout classification
maps cover all 1024 cyclic shifts. Summed-readout maps and aligned-feature
distances use the 4×4 fundamental cell. Direct simulations on 200 digits
check reconstruction, including shifts through the boundary. Classification
always uses the unaligned readout. The summed classifier is invariant under
shifts divisible by four; it need not be invariant under a one-pixel shift.

Diagnostics evaluate the trained weights in float64 without renormalization.
The maximum observed norm residual is 1.37e-13; the native-float32 audit
differs in at most 0.01875% of predictions. Training histories retain their
original float32 results. The local/full cohort is an independent repetition
of the model used in the common QCS comparison, rather than that same run.

From the package root, recreate figures without weights or CUDA:

```sh
uv run python run/plot_pcs_translation.py --output-root results/pcs_translation_2026-10-25
```

Figures and CSVs are written to `plots/`. For independent per-image verification,
extract the supplied data archive at the repository root and run:

```sh
uv run python run/verify_pcs_translation.py
```

The data archive supplies twelve final checkpoints, with model tensors and
architecture only. The verifier prepares MNIST, reconstructs missing arrays
from these weights, checks direct equivariance and compares the reconstructed
statistics with the reference JSON. Generated arrays are cached separately under
`artifacts/pcs_translation_recomputed/`; plotting uses only this compact bundle.
Each JSON retains the original array SHA-256 for verification of older copies;
recomputed arrays are checked numerically and carry their own hashes. The complete reproduction
protocol is in [the benchmark guide](../../docs/pcs_translation.md).
