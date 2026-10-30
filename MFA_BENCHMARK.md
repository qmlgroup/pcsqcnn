# MFA benchmark

This implementation reproduces the ten-class circuit of Mordacci, Ferrari
and Amoretti, *Multi-Class Quantum Convolutional Neural Networks*, QUASAR 2024,
DOI 10.1145/3660318.3660326. The reference implementation is
[`qslab-unipr/qcnn`](https://github.com/qslab-unipr/qcnn), revision
`5c320d93664aa02125f18eaab6b33119f08da39a`. Its source is not included here.

The circuit has eight amplitude-encoded qubits, three convolution blocks,
two pooling blocks and 105 trainable parameters. Four measured qubits give
16 probabilities; prediction selects the largest of the first ten. The loss
is the negative log of the raw true-class probability, clipped to
`[1e-15, 1-1e-15]`. No ten-class renormalization or classical head is added.

PCA with 256 components is fitted separately on each seed's training set.
The batched PennyLane circuit uses PyTorch backpropagation and `MFAAdam`,
which reproduces the reference optimizer's update rule. CPU and CUDA use
the same circuit and objective.

## Dataset and training

The benchmark uses 1000 training images per digit and all 10000 MNIST test
images. Digits are resized to 16×16 and placed on a 32×32 canvas with offsets
up to eight pixels. Seeds are 0, 1, 2; train/test batches are 256/1600;
training lasts 2000 epochs with Adam at learning rate 0.01 and no weight
decay. Tests run every ten epochs and at the final epoch. The comparison
uses final checkpoints, not the best test epoch.

Image/label fingerprints verify that MFA and the general PCS reference
receive the same prepared images within each seed. Their representations,
parameter counts and objectives differ: this comparison assesses complete
classification pipelines rather than isolating quantum symmetry.

```sh
uv sync --extra mfa --extra test
uv run --extra mfa python run/run_mfa_full_matrix.py --device cuda
```

Runs are resumable under `artifacts/mfa/`. To train one seed:

```sh
uv run --extra mfa python run/prepare_mfa_baseline_data.py --regime translated --seed-count 1 --base-seed 0 --directory-name translated_seed0
```

Repeat the same command with `--resume` after interruption. PCA, optimizer,
RNG state and data fingerprints are stored with the generated checkpoint.

## Results and verification

The [numerical bundle](results/mfa_translated/README.md) contains final
metrics and complete learning histories. Mean test accuracy and sample
standard deviation are **16.57 ± 1.35%** for MFA and **75.89 ± 0.12%** for PCS.

`tests/test_mfa.py` checks probabilities, loss, gradients and optimizer steps.
Set `MFA_AUTHOR_SOURCE` to the reference source directory to enable the
independent author-circuit comparison. With generated weights and the MNIST
cache, verify predictions on all test images using:

```sh
uv run --extra mfa python run/verify_mfa_benchmarks.py --root artifacts/mfa --data-root data --device cuda
```

Weights and PCA buffers are generated artifacts, separate from the compact
plotting bundle. Plotting the recorded comparison needs neither weights nor
a new training run.
