"""Physics and integration checks for the explicitly adapted sp-QCNN."""

import math

import pytest
import torch

from qcnn.baselines.sp_qcnn import SpQCNN, binary_rotation_orbits
from qcnn.model_stats import format_trainable_parameter_stats_line


def shift_wires(state, bits, axis):
    order = list(range(2 * bits + 2))
    start = axis * bits
    order[start:start + bits] = order[start + 1:start + bits] + [order[start]]
    return state.reshape(-1, *([2] * (2 * bits + 2))).permute(0, *(i + 1 for i in order)).reshape_as(state)


def test_main_architecture_and_normalization():
    model = SpQCNN()
    assert [sum(p.numel() for p in s.parameters()) for s in model.stages] == [200, 150, 12]
    assert sum(p.numel() for p in model.classifier.parameters()) == 2570
    assert format_trainable_parameter_stats_line(model) == "q200 q150 q12 c2570 Q362 C2570"
    orbits = binary_rotation_orbits(5)
    assert [min(o) for o in orbits] == [0, 1, 3, 5, 7, 11, 15, 31]
    assert sorted(v for orbit in orbits for v in orbit) == list(range(32))
    images = torch.rand(3, 32, 32)
    encoded = model.encode(images)
    torch.testing.assert_close(encoded, model.encoder(images).reshape(3, -1) / 32)
    p = model.exact_quantum_readout_probabilities(images)
    assert p.shape == (3, 8, 8, 4)
    torch.testing.assert_close(p.sum((1, 2, 3)), torch.ones(3), atol=2e-5, rtol=0)


@pytest.mark.parametrize("axis", [0, 1])
def test_every_stage_and_classifier_respect_address_qcs(axis):
    torch.manual_seed(41)
    model = SpQCNN(dtype="float64")
    state = torch.randn(2, 4096, dtype=torch.complex128)
    state = state / state.norm(dim=1, keepdim=True)
    rotated = shift_wires(state, 5, axis)
    for stage in model.stages:
        torch.testing.assert_close(stage(rotated), shift_wires(stage(state), 5, axis), atol=2e-13, rtol=2e-12)
    output = model.quantum_core(state)
    output_rotated = model.quantum_core(rotated)
    torch.testing.assert_close(output_rotated, shift_wires(output, 5, axis), atol=2e-13, rtol=2e-12)
    torch.testing.assert_close(output.norm(dim=1), state.norm(dim=1), atol=2e-13, rtol=2e-12)
    h, hr = model.readout_from_state(output), model.readout_from_state(output_rotated)
    torch.testing.assert_close(h, hr, atol=2e-13, rtol=2e-12)
    torch.testing.assert_close(model.classify_readout_histogram(h), model.classify_readout_histogram(hr))
    images = torch.rand(2, 32, 32, dtype=torch.float64)
    order = list(range(10))
    start = axis * 5
    order[start:start + 5] = order[start + 1:start + 5] + [order[start]]
    shifted_images = images.reshape(2, *([2] * 10)).permute(0, *(i + 1 for i in order)).reshape_as(images)
    torch.testing.assert_close(model(images), model(shifted_images), atol=2e-13, rtol=2e-12)
    # This contract must not accidentally be advertised as pixel invariance.
    assert (model(images) - model(images.roll(1, axis + 1))).abs().max() > 1e-8


@pytest.mark.parametrize(("size", "depth"), [(4, 2), (32, 1)])
def test_probabilities_and_parameter_gradients_against_independent_pennylane(size, depth):
    qml = pytest.importorskip("pennylane")
    torch.manual_seed(7)
    model = SpQCNN(image_size=size, depth=depth, dtype="float64")
    bits = size.bit_length() - 1
    qubits = 2 * bits + 2
    state = torch.randn(1, 1 << qubits, dtype=torch.complex128)
    state /= state.norm(dim=1, keepdim=True)
    weights = torch.randn(1 << qubits, dtype=torch.float64)
    reference_angles = [s.angles.detach().clone().requires_grad_() for s in model.stages]

    @qml.qnode(qml.device("default.qubit", wires=qubits), interface="torch", diff_method="backprop")
    def reference(*angles):
        qml.StatePrep(state[0], wires=range(qubits))
        groups = (tuple(range(bits)), tuple(range(bits, 2 * bits)), (2 * bits,), (2 * bits + 1,))
        for stage, blocks in enumerate(angles):
            for block in blocks:
                for group, wires in enumerate(groups):
                    for wire in wires:
                        qml.RX(2 * block[3 * group], wires=wire)
                        qml.RZ(2 * block[3 * group + 1], wires=wire)
                        qml.RX(2 * block[3 * group + 2], wires=wire)
                if stage == 2:
                    continue
                for group in (0, 1):
                    wires = groups[group]
                    # n=2 has one undirected edge, not a doubled edge.
                    for j in range(bits if bits > 2 else 1):
                        qml.IsingZZ(2 * block[12 + group], wires=(wires[j], wires[(j + 1) % bits]))
                if stage == 0:
                    for x in groups[0]:
                        for y in groups[1]:
                            qml.IsingZZ(2 * block[14], wires=(x, y))
                    for group, feature, index in ((0, 2 * bits, 15), (0, 2 * bits + 1, 16),
                                                   (1, 2 * bits, 17), (1, 2 * bits + 1, 18)):
                        for wire in groups[group]:
                            qml.IsingZZ(2 * block[index], wires=(wire, feature))
                qml.IsingZZ(2 * block[-1], wires=(2 * bits, 2 * bits + 1))
        return qml.probs(wires=range(qubits))

    expected = reference(*reference_angles)
    actual = model.quantum_core(state).abs().square()[0]
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-12)
    reference_gradients = torch.autograd.grad(expected @ weights, reference_angles)
    actual_gradients = torch.autograd.grad(actual @ weights, [s.angles for s in model.stages])
    for actual_grad, expected_grad in zip(actual_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-12, rtol=2e-10)

    # Independent grouping: rotate binary strings and directly sum outcomes.
    canonical = lambda v: min((s := f"{v:0{bits}b}")[k:] + s[:k] for k in range(bits))
    reps = sorted({canonical(v) for v in range(size)})
    reference_hist = torch.stack([
        expected.reshape(size, size, 4)[[v for v in range(size) if canonical(v) == x]][:, [v for v in range(size) if canonical(v) == y], f].sum()
        for x in reps for y in reps for f in range(4)
    ])
    torch.testing.assert_close(model.readout_from_state(model.quantum_core(state)).flatten(), reference_hist)


def test_dtype_conversion_and_input_validation():
    model = SpQCNN(image_size=4, depth=1).double()
    assert model.encode(torch.rand(2, 4, 4)).dtype == torch.complex128
    assert model(torch.rand(2, 4, 4)).dtype == torch.float64
    with pytest.raises(ValueError, match="expects"):
        model(torch.rand(2, 8, 8))
    with pytest.raises(ValueError, match="power of two"):
        SpQCNN(image_size=31)
    with pytest.raises(ValueError, match="positive"):
        SpQCNN(depth=0)
