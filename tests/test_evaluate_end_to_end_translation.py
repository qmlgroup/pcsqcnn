from __future__ import annotations

import torch

from qcnn import PCSQCNN
from qcnn.hur import HurQCNN
from tests.script_loading import load_run_script


end_to_end = load_run_script("evaluate_end_to_end_translation_module", "evaluate_end_to_end_translation.py")


class _BrightnessClassifier(torch.nn.Module):
    """Predicts from the total brightness, which no cyclic shift changes."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(images.sum(dim=(1, 2)).round().long() % 3, 3).float()


def test_shift_invariant_classifier_keeps_every_label() -> None:
    torch.manual_seed(0)
    images = torch.rand(6, 32, 32)
    labels = _BrightnessClassifier()(images).argmax(dim=1)

    run = end_to_end.evaluate_seed(_BrightnessClassifier(), images, labels)

    assert len(run["grid_acc"]) == 17 and len(run["cyclic_acc"]) == 32
    assert run["grid_acc_min"] == 1.0 and run["grid_cons_mean"] == 1.0 and run["wrapping_acc_mean"] == 1.0


def test_pcs_readout_follows_the_coarse_lattice_and_hur_readout_does_not() -> None:
    torch.manual_seed(0)
    images = torch.rand(4, 8, 8)
    pcs_qcnn = PCSQCNN(image_size=8, num_classes=10, feature_qubits=2, quantum_layers=2).eval()
    hur_qcnn = HurQCNN(image_size=8, num_classes=10).eval()

    with torch.no_grad():
        pcs = end_to_end.readout_errors(pcs_qcnn, images, family="lattice")
        hur = end_to_end.readout_errors(hur_qcnn, images, family="hur")

    assert max(value for name, value in pcs["readout_l1_max"].items() if "vs" not in name) < 1e-5
    assert "readout_l1_relabel_max" not in pcs
    for name, bound in hur["readout_l1_relabel_max"].items():
        assert 1e-3 < bound <= hur["readout_l1_max"][name] + 1e-6
