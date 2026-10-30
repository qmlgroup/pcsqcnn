# PCS translation benchmark

The same 10000 centered test digits are used at every displacement and for every model.
Heatmap rows are delta_y; columns are delta_x. Full maps use offsets -16..15, modulo 32.
The white square marks the bounded offsets [-8,8]^2; outside it the canvas wraps.
Cell annotations are means and sample standard deviations across seeds.
Changed predictions are measured against each model's own unshifted predictions.
Feature TV averages per-image distances after undoing the known coarse spatial permutation.
Only this feature metric is aligned; classification always uses the unaligned readout.
16 quantum readouts reconstruct the full grid. Separate direct simulations validate this method.
The summed classifier is invariant under translations divisible by four, not every pixel shift.
Summed readout compresses 256 features to four and is a symmetry demonstration, not an optimized invariant classifier.
Histories use the original fixed randomly translated test split, distinct from the centered shift scans.
Training accuracy is accumulated within an epoch; test accuracy uses the final weights of that epoch.
Local runs restart Adam at epochs 2000 and 4000. All curves are unsmoothed.

| Model | Epochs | Parameters | Fixed-test accuracy (%) |
| --- | ---: | ---: | ---: |
| General PCS, full | 2000 | 39434 | 75.89 ± 0.12 |
| General PCS, summed | 2000 | 36914 | 67.34 ± 0.24 |
| Local PCS, full | 2000 | 2858 | 62.00 ± 4.08 |
| Local PCS, full | 4000 | 2858 | 66.54 ± 2.17 |
| Local PCS, full | 6000 | 2858 | 67.57 ± 2.04 |
| Local PCS, summed | 2000 | 338 | 47.86 ± 6.85 |
| Local PCS, summed | 4000 | 338 | 51.76 ± 6.02 |
| Local PCS, summed | 6000 | 338 | 49.80 ± 4.46 |

## Translation diagnostics

Ranges below are extrema of the seed-mean map, not extrema over individual seeds.
Full-map bounded averages use 289 offsets in [-8,8]^2; wrapping averages use the other 735 offsets.
The coarse subgroup consists of the 64 offsets divisible by four (including the origin).
For summed readout, every whole-grid mean averages the 16 residues equally.
Means and sample SD of spatial averages are computed after averaging within each seed.

| Model | Accuracy range (%) | Max changed predictions (%) | Max aligned feature TV (%) |
| --- | ---: | ---: | ---: |
| General PCS, full | 8.13–81.82 | 90.73 | 10.6760 |
| General PCS, summed | 67.09–68.01 | 8.81 | 0.1669 |
| Local PCS, full | 2.77–76.38 | 96.38 | 8.3031 |
| Local PCS, summed | 47.98–50.93 | 33.72 | 0.3087 |

Fixed weights are evaluated in double precision without renormalization.
Per-seed direct residuals and native-precision prediction audits are retained in the diagnostic JSON files.
The local full model is a separate repetition; it does not replace the earlier QCS-comparison cohort.
