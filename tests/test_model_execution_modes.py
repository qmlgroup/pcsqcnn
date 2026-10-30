"""Exercise inference-first execution and precision changes through full models."""

import pytest
import torch
from torch import nn

from qcnn import PCSQCNN, PCSQCNNNoQFT, QCSQCNN
from qcnn import quantum, qcs

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available.",
))]


@pytest.fixture
def cold_quantum_caches():
    def clear():
        quantum._pauli_basis_matrices.cache_clear()
        quantum._PAULI_BASIS_DEVICE_DTYPE_CACHE.clear()
        quantum._FOURIER_JUNCTION_PHASE_CACHE.clear()
        qcs._local_term_placements.cache_clear()
        qcs.cyclic_shift_eigenbasis.cache_clear()
        qcs._symmetry_block_pauli_terms.cache_clear()
        qcs._DEVICE_TENSOR_CACHE.clear()

    clear()
    yield
    clear()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model_class", [PCSQCNN, PCSQCNNNoQFT, QCSQCNN])
def test_inference_first_does_not_break_later_training(model_class, cold_quantum_caches, device):
    torch.manual_seed(4)
    images = torch.rand(2, 8, 8, device=device)
    labels = torch.tensor([0, 1], device=device)
    model = model_class(image_size=8, num_classes=2, feature_qubits=1, quantum_layers=2).to(device)
    with torch.inference_mode():
        expected = model.eval()(images)

    # Caches are shared with fresh instances, not just the original model.
    fresh = model_class(image_size=8, num_classes=2, feature_qubits=1, quantum_layers=2).to(device)
    fresh.load_state_dict(model.state_dict())
    for candidate in [model, fresh]:
        logits = candidate.train()(images)
        torch.testing.assert_close(logits, expected)
        nn.functional.cross_entropy(logits, labels).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters())
        assert any(p.grad.abs().sum() > 0 for name, p in candidate.named_parameters() if not name.startswith("classifier."))


@pytest.mark.parametrize("method", ["constructor", "double", "to"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model_class", [PCSQCNN, PCSQCNNNoQFT, QCSQCNN])
def test_double_precision_forward_and_backward(model_class, method, device):
    torch.manual_seed(5)
    dtype = torch.float64 if method == "constructor" else torch.float32
    model = model_class(image_size=8, num_classes=2, feature_qubits=1, quantum_layers=2, dtype=dtype).to(device)
    images = torch.rand(2, 8, 8, dtype=torch.float64, device=device)
    with torch.no_grad():
        before = model(images).clone()
    unitaries = {
        name: tensor.clone() for name, tensor in model.named_buffers() if tensor.is_complex()
    }
    if method == "double":
        model.double()
    elif method == "to":
        model.to(dtype=torch.float64)

    logits = model(images)
    assert logits.dtype == torch.float64
    assert model.encoder(images).dtype == torch.complex128
    torch.testing.assert_close(logits, before.to(torch.float64), atol=2e-5, rtol=2e-5)
    for name, original in unitaries.items():
        converted = model.get_buffer(name)
        assert converted.dtype == torch.complex128
        torch.testing.assert_close(converted, original.to(torch.complex128), atol=0, rtol=0)

    nn.functional.cross_entropy(logits, torch.tensor([0, 1], device=device)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.float()
    assert model(images).dtype == torch.float32
    assert model.encoder(images).dtype == torch.complex64
    for name in unitaries:
        assert model.get_buffer(name).dtype == torch.complex64
