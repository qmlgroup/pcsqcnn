# Cross-pattern experiment for Figure 1

This binary classification experiment supplies the trained states for Figure 1.
No feature-location target is used in training. Generated checkpoints and dense
state arrays are stored under `artifacts/`.

## Protocol

Images have shape 16 x 16. Array axes are `(x, y)`; plotting transposes them.
The background is 0 and the five pixels of a 3 x 3 plus sign have brightness
1/2. The manuscript denotes the image by calligraphic I and its pixel intensity
by I(r), where r=(x,y), with x rightward and y downward. The encoder is
`sin(vartheta_r)|0> + cos(vartheta_r)|1>`, with vartheta_r=pi*I(r) and an
extra feature qubit in zero. The state is normalized by 1/16: 8 address and
2 feature qubits in total.

There are 19 base patterns: one isolated cross, four rotated broken crosses,
and seven two-pattern arrangements for each displacement `(0,7)` and `(6,5)`.
A broken cross moves one arm pixel to the adjacent corner. Positive pairs
contain two crosses or one cross and one broken cross; negative pairs contain
two broken crosses. Every base pattern is shifted by all 256 toroidal offsets,
giving 4864 unique images. Labels mean that at least one complete cross is
present. The four strata `(5 or 10 occupied pixels) x (label 0 or 1)` carry
equal total weight. Thus raw summed FRQI probabilities cannot exceed 50%
weighted accuracy. The generator checks labels and all C4 x C4 symmetry
orbits for contradictory labels. There are no such contradictions in this set.

This finite training task is intentionally exhausted over all positions.
Its accuracy is training accuracy, not a held-out translation test. The
illustrated pair at `(3,3)` and `(4,12)` has a different relative displacement
from the training pairs. This one example is not an independent statistical
test set. All transformed images and their actual cross counts are retained.

PCS uses `LocalModeMultiplexer2D`: two U(4) coins, 32 real coefficients,
exact radius-one support. No pooling. QCS uses two joint blocks from the
existing sp-QCNN adaptation, followed by separate RX-RZ-RX rotations on the
two feature qubits: 46 coefficients. The address bits of x and y rotate
independently; all gates commute with C4 x C4. This short block is not the
full published sp-QCNN. Both have four feature probabilities summed over all
addresses and the same affine 4-to-1 binary head (five coefficients). Nominal
parameter counts do not imply matched expressivity; phase/readout redundancies
are not removed.

For conditioning, the four summed probabilities are centered on the empty
image and multiplied by 256 before the affine head. This is the same affine
function family on the probabilities, since the head has a free bias. Quantum
coefficients start as Normal(0,0.3); head weights as Normal(0,0.1), bias zero.
Both heads have identical initial values for a given seed. Adam uses lr 0.01,
full batches of 4864 images and 2000 updates; seeds 0,1,2. No best-accuracy checkpoint selection.
The full gradient uses the explicitly balanced objective.
Training uses float32 with highest matmul precision; audits use the unchanged
trained coefficients converted to float64.

The manuscript figure shows the same conditional color probability at every
address: `p_0(r) = sum_f |Psi(r,f,0)|^2 / sum_fc |Psi(r,f,c)|^2`.
It uses no head weights, blank subtraction or per-panel normalization.
All states use one linear probability range [0,1]. Cool teal inputs and warm
coral/berry outputs use distinct sequential palettes, both with monotonically
decreasing luminance. The input satisfies p_0(r)=sin(vartheta_r)^2.
Address probabilities, phases and auxiliary correlations are not represented by
this scalar; small-probability addresses are not suppressed. Full-state symmetry
checks are performed independently, so the display is not used as a proof of
state equality. All ten tiles have the same size and thin gray vector pixel grids.

Seed 0 is the predefined illustration seed, never selected for appearance.
The final diagram uses T_x^8 and S_x^3. The two crosses are offset horizontally
by one pixel, centered as a pair in the left half and translated into the right
half. Since S_x^8 is the identity, its exponent is intentionally different.

## Reproduce

From the package root, using its environment:

```sh
PYTHONPATH=src python run/figure1_crosses.py --device cuda
```

Each stage can run separately with `--stage data`, `verify`, `train`,
`diagnose`, or `plot`; select models/seeds with `--models pcs` or `--seeds 0`.
`--output-root` selects a fresh artifact directory. Resuming uses the same
command and restores the optimizer at the last saved epoch. Batch ordering
is fixed by seed and epoch. Incompatible configuration/source hashes are
rejected. Diagnostics refuse weights trained with different model sources.
Training defaults to CUDA; use `--device cpu` for a CPU run.

Verification checks full-state commutators, norm preservation, PCS impulse
support and an independent spatial shift/coin implementation. Tests additionally
compare classification gradients to finite differences. Trained audits check
the commutative diagrams, the exact logit decomposition and locality by
removing the distant cross. Nonmatching symmetries are measured, not required
to fail. Diagnostic plots and the final state projection are distinct outputs.

The compact [Figure 1 data](../results/figure1_crosses/README.md) provide
the final displayed states and checks without loading checkpoints. The stored
probe uses crosses at (3,3) and (4,12), T_x^8 and S_x^3. The runner also supports
the classification and diagnostic stages independently.
