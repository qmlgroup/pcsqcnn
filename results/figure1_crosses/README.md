# Trained cross example for Figure 1

The protocol uses 4864 images, 2000
full-batch Adam updates (lr 0.01), and seeds 0, 1, 2. These accuracies fit a
finite synthetic training task; they are not held-out MNIST results.

| Model | Quantum + head parameters | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: | ---: |
| Local PCS | 32 + 5 | 100.00% | 100.00% | 100.00% |
| Small QCS | 46 + 5 | 67.68% | 78.50% | 68.30% |

QCS mean and sample standard deviation: 71.49 +/- 6.08 percentage points.
The baseline based only on raw summed FRQI probabilities has optimal weighted
accuracy 50%; there are no contradictory labels in the C4 x C4 training orbits.

The actual figure uses the predefined seed 0, crosses at (3,3) and (4,12),
T_x^8 and S_x^3. `figure_data.npz` contains the normalized complex states,
probabilities, and the earlier classifier-contribution diagnostics; it contains
no model checkpoint. `figure_evidence.json` records all six trained models'
checks at these positions and the checkpoint hashes. Its `visualization` field
records the current figure projection p_0(r) = P(color=0 | address), with the auxiliary
qubit marginalized, a shared linear [0,1] range, and no classifier weights.
Array names ending in `_C` retain the historical archive spelling of this quantity.
The cool input and warm output palettes both have monotone luminance; full-state checks are
independent of this incomplete projection of the state. Matching full-state
commutators are at round-off (below 1.2e-16 for the displayed probes).

For the pictured pair, PCS gives p(cross)=0.99785 before and after translation,
and below 1e-12 after the bit cycle. QCS gives 0.95797, 0.96546, and 0.95797,
respectively. Its summed response survives the bit cycle even though no local
cross remains. Its spatial contribution map does not follow pixel translation.
The figure illustrates this distinction, not a universal performance ordering.

`pcs_seed*/` and `qcs_seed*/` contain the configurations, final metrics and
complete learning histories. `architecture_checks.json` and `dataset_audit.json`
record model and data checks. `manifest.json` hashes the compact numerical
files. Model weights are generated under `artifacts/` and stored separately.

See [the experiment guide](../../docs/figure1_crosses.md) for layer definitions,
training and reproduction commands. The displayed probe and its full-state
checks are in `figure_evidence.json`; `figure_data.npz` provides the complex
states needed to reproduce the input and output probability maps.

`source_compatibility.json` records audited docstring revisions and guards
that keep cached constants usable for training after inference. The numerical
operations are unchanged. States, logits and parameter gradients from all six
saved models agree exactly before and after the guards on the Figure 1 probes.
The original training hashes are preserved; rendering accepts only the
recorded source hashes.
