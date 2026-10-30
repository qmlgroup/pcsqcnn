from __future__ import annotations

import torch

from qcnn import PCSQCNN
from qcnn.baselines.sp_qcnn import SpQCNN
from tests.script_loading import load_run_script


alignment = load_run_script("evaluate_readout_alignment_module", "evaluate_readout_alignment.py")


def test_row_bit_rotation_permutes_rows_by_rotating_their_index_bits() -> None:
    images = torch.arange(8, dtype=torch.float32)[None, :, None].expand(1, 8, 8)

    rotated = alignment.rotate_row_bits(images)

    # Row r = (b2 b1 b0) moves to (b1 b0 b2): 1 -> 2, 3 -> 6, 4 -> 1, 7 -> 7.
    assert [int(rotated[0, row, 0]) for row in (2, 6, 1, 7)] == [1, 3, 4, 7]
    assert sorted(rotated[0, :, 0].tolist()) == list(range(8))


def test_sp_qcnn_is_invariant_and_pcs_qcnn_is_not_equivariant_under_the_bit_rotation() -> None:
    torch.manual_seed(0)
    images = torch.rand(4, 8, 8)
    sp_qcnn = SpQCNN(image_size=8, depth=2, dtype="float64").eval()
    pcs_qcnn = PCSQCNN(image_size=8, num_classes=10, feature_qubits=2, quantum_layers=2).eval()

    with torch.no_grad():
        sp_residual = alignment.bit_rotation_residual(sp_qcnn, images.double())
        pcs_bound = alignment.bit_rotation_relabel_residual(pcs_qcnn, images)

    assert sp_residual < 1e-10
    assert pcs_bound > 1e-3
