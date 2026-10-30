# PCS-QCNN and MFA benchmark

Final test accuracy on the same translated-MNIST images. Values are
mean ± sample standard deviation across seeds 0, 1, 2.
The final column is the difference of means in percentage points.

| Regime | PCS-QCNN | MFA | MFA − PCS |
| --- | ---: | ---: | ---: |
| translated | 75.89 ± 0.12% | 16.57 ± 1.35% | -59.32 pp |

Translated trainable parameters: PCS-QCNN 36,864 quantum + 2,570 classical; MFA 105 quantum + 0 classical.

The translated PCS reference is the full-readout `fq2_ql3` model,
not the `pcsqcnn_no_qft_spc1000` ablation. The two models use the
same MNIST image/label split and geometry within each seed.
The models' input encodings, parameter counts, and objectives differ, so
accuracy differences alone cannot isolate quantum symmetry.
Their loss values are not directly comparable.
MFA wall times are reported in `benchmark_summary.json`; matching
PCS wall times are not available in the saved artifacts.
Three seeds do not support a strong significance claim.
