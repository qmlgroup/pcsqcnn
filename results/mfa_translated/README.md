# MFA comparison on translated MNIST

Three 2000-epoch runs per model, seeds 0, 1, 2. Mean final test accuracy and
sample standard deviation: **MFA 16.57 ± 1.35%**, **PCS 75.89 ± 0.12%**.
Both receive the same prepared images within each seed, but use different
encodings, parameter counts and objectives.

`comparison/` contains the numerical comparison and complete train/test
learning curves. `mfa/` and `pcs/` contain configurations and per-seed metric
histories. Checkpoints and execution logs are separate from this plotting
bundle. Data fingerprints and saved evaluation checks are retained.

From the package root, regenerate the comparison and figure without training:

```sh
uv run python run/summarize_pcs_mfa_comparison.py --pcs-artifacts-root results/mfa_translated/pcs --mfa-root results/mfa_translated/mfa --output-dir /tmp/pcs-mfa-report
uv run python run/plot_mfa_pcs_comparison.py --curves /tmp/pcs-mfa-report/comparison_learning_curves.csv --output /tmp/pcs-mfa-report/comparison.pdf
```

For the circuit, training and checkpoint-evaluation commands, see
[the MFA protocol](../../MFA_BENCHMARK.md). Run `sha256sum -c SHA256SUMS`
from this directory to check the compact files.
