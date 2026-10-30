"""Checks for the independently implemented MFA literature baseline."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

qml = pytest.importorskip("pennylane")

from qcnn.baselines.mfa import MFAClassifier, MFAProbabilityNLLLoss, mfa_circuit
from qcnn.baselines.mfa_optimizer import MFAAdam
from qcnn.model import ImageClassifierRunner
from qcnn.model_spec import ModelSpec
from qcnn.model_stats import format_trainable_parameter_stats_line
from qcnn.serialization import load_model_checkpoint, save_model_checkpoint


def test_mfa_probability_loss_keeps_unused_outcomes() -> None:
    probabilities = torch.tensor([[0.1, 0.2, *([0.0] * 8)]], dtype=torch.float64)
    labels = torch.tensor([1])
    assert torch.allclose(MFAProbabilityNLLLoss()(probabilities, labels), -probabilities[0, 1].log())


def test_mfa_pca_checkpoint_and_runner(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    torch.manual_seed(4)
    training_images = torch.rand(257, 16, 16)
    model = MFAClassifier(image_size=16)
    assert format_trainable_parameter_stats_line(model) == "q105 Q105 C0"
    model.fit_preprocessing(training_images, seed=4)
    images = torch.rand(2, 16, 16)
    before = model(images)
    assert before.shape == (2, 10)
    assert isinstance(ImageClassifierRunner(model=model).loss_collector.loss_fn, MFAProbabilityNLLLoss)

    spec = ModelSpec("qcnn.baselines.mfa", "MFAClassifier", {"image_size": 16})
    path = tmp_path / "mfa.pt"
    save_model_checkpoint(model, spec, path)
    restored = load_model_checkpoint(path).model
    torch.testing.assert_close(restored.pca_mean, model.pca_mean)
    torch.testing.assert_close(restored.pca_components, model.pca_components)
    torch.testing.assert_close(restored(images), before)


def test_mfa_adam_matches_pennylane_steps() -> None:
    initial = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    gradients = (
        np.array([0.4, -0.1, 0.7]),
        np.array([-0.2, 0.5, 0.3]),
        np.array([0.1, 0.2, -0.4]),
    )
    parameter = torch.nn.Parameter(torch.tensor(initial.copy()))
    torch_optimizer = MFAAdam([parameter], lr=0.01)
    reference = qml.AdamOptimizer(stepsize=0.01)
    reference_parameters = qml.numpy.array(initial, requires_grad=True)
    for gradient in gradients:
        parameter.grad = torch.tensor(gradient)
        torch_optimizer.step()
        reference_parameters = reference.apply_grad((gradient,), (reference_parameters,))[0]
        np.testing.assert_allclose(parameter.detach().numpy(), reference_parameters, atol=1e-14, rtol=1e-13)


def test_mfa_gates_and_gradients_match_author_reference() -> None:
    source = os.environ.get("MFA_AUTHOR_SOURCE")
    if not source:
        pytest.skip("Set MFA_AUTHOR_SOURCE to the separately downloaded author python_files directory")
    reference_dir = Path(source)
    if not (reference_dir / "QCNN_circuit.py").is_file():
        pytest.skip("Author reference source is not available")

    sys.path.insert(0, str(reference_dir))
    try:
        original = importlib.import_module("QCNN_circuit")
        author_qnode = qml.QNode(
            original.QCNN.func, qml.device("default.qubit", wires=8), interface="torch", diff_method="backprop"
        )
        own_qnode = qml.QNode(
            mfa_circuit, qml.device("default.qubit", wires=8), interface="torch", diff_method="backprop"
        )
        generator = torch.Generator().manual_seed(23)
        features = torch.randn(3, 256, generator=generator, dtype=torch.float64)
        own_angles = torch.randn(105, generator=generator, dtype=torch.float64, requires_grad=True)
        author_angles = own_angles.detach().clone().requires_grad_(True)
        own = own_qnode(features, own_angles)
        author = author_qnode(features, author_angles, 15, "amplitude", num_classes=10, num_layer=1)
        torch.testing.assert_close(own, author, atol=1e-12, rtol=1e-11)
        labels = torch.tensor([0, 7, 9])
        objective = MFAProbabilityNLLLoss()
        own_loss = objective(own[:, :10], labels)
        author_loss = objective(author[:, :10], labels)
        torch.testing.assert_close(own_loss, author_loss, atol=1e-12, rtol=1e-11)
        own_loss.backward()
        author_loss.backward()
        torch.testing.assert_close(own_angles.grad, author_angles.grad, atol=1e-11, rtol=1e-10)
    finally:
        sys.path.remove(str(reference_dir))
