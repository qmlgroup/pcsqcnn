from __future__ import annotations

import math

import pytest
import torch

from qcnn import HurQCNN
from qcnn.article_training import (
    ArticleTrainingDefaults,
    build_hur_qcnn_directory_name,
    build_hur_qcnn_model_spec,
)
from qcnn.hur import build_gate_phases, build_hur_layer_schedule, pooling_gate, su4_gate
from qcnn.model_spec import instantiate_model
from qcnn.model_stats import format_trainable_parameter_stats_line


def test_eight_qubit_binary_schedule_follows_the_authors_wiring() -> None:
    schedule = build_hur_layer_schedule(8, output_qubits=1)

    # QCNN_circuit.py: conv_layer1-3 and pooling_layer1-2 of github.com/takh04/QCNN.
    assert schedule.convolutions == (
        ((0, 7), (0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6)),
        ((0, 6), (0, 2), (4, 6), (2, 4)),
        ((0, 4),),
    )
    assert schedule.poolings[:2] == (((1, 0), (3, 2), (5, 4), (7, 6)), ((2, 0), (6, 4)))
    # The authors' last pooling uses control 0 and target 4; the SU(4) in front absorbs the swap.
    assert schedule.poolings[2] == ((4, 0),)
    assert schedule.measured_wires == (0,)


def test_ten_class_schedule_keeps_five_measured_qubits() -> None:
    schedule = build_hur_layer_schedule(10, output_qubits=4)

    assert len(schedule.convolutions) == 2
    assert len(schedule.convolutions[0]) == 10
    assert schedule.poolings == (((1, 0), (3, 2), (5, 4), (7, 6), (9, 8)),)
    assert schedule.convolutions[1] == ((0, 8), (0, 2), (4, 6), (2, 4), (6, 8))
    assert schedule.measured_wires == (0, 2, 4, 6, 8)


def test_odd_registers_keep_the_unpaired_qubit() -> None:
    schedule = build_hur_layer_schedule(6, output_qubits=2)

    assert schedule.poolings == (((1, 0), (3, 2), (5, 4)), ((2, 0),))
    assert schedule.convolutions[-1] == ((0, 4),)
    assert schedule.measured_wires == (0, 4)


def test_gate_phases_group_disjoint_gates_around_a_split() -> None:
    schedule = build_hur_layer_schedule(10, output_qubits=4)

    closing, even, odd = build_gate_phases(schedule.convolutions[0], qubits=10)
    (pooling,) = build_gate_phases(schedule.poolings[0], qubits=10)

    assert (closing.left, closing.right, closing.straddling) == ((), (), ((0, 9),))
    assert (even.split, even.left, even.right, even.straddling) == (4, ((0, 1), (2, 3)), ((4, 5), (6, 7), (8, 9)), ())
    assert (odd.split, odd.left, odd.right, odd.straddling) == (5, ((1, 2), (3, 4)), ((5, 6), (7, 8)), ())
    assert pooling.straddling == () and len(pooling.left) + len(pooling.right) == 5


def test_gates_are_unitary() -> None:
    torch.manual_seed(0)
    identity = torch.eye(4, dtype=torch.complex128)
    for gate in (su4_gate(torch.rand(15, dtype=torch.float64) * 2 * math.pi), pooling_gate(torch.rand(2, dtype=torch.float64))):
        assert torch.allclose(gate.conj().T @ gate, identity, atol=1e-12)


def test_pooling_gate_rotates_the_target_by_the_control_value() -> None:
    angles = torch.tensor([0.7, 1.3], dtype=torch.float64)
    gate = pooling_gate(angles).reshape(2, 2, 2, 2)
    rz = torch.diag(torch.exp(torch.tensor([-0.35j, 0.35j], dtype=torch.complex128)))
    cos, sin = math.cos(0.65), math.sin(0.65)
    rx = torch.tensor([[cos, -1j * sin], [-1j * sin, cos]], dtype=torch.complex128)

    # Control |1> -> RZ(theta_1), control |0> -> RX(theta_2); the X flips the traced-out control.
    assert torch.allclose(gate[0, :, 1, :], rz)
    assert torch.allclose(gate[1, :, 0, :], rx)
    assert torch.allclose(gate[0, :, 0, :], torch.zeros(2, 2, dtype=torch.complex128))


@pytest.mark.parametrize(
    ("image_size", "num_classes", "filters", "channels", "encoding"),
    [
        (4, 4, 1, 1, "amplitude"),
        (8, 4, 2, 1, "amplitude"),
        (16, 2, 1, 1, "amplitude"),
        (8, 4, 1, 3, "amplitude"),
        (4, 4, 1, 1, "frqi"),
        (8, 4, 1, 2, "frqi"),
    ],
)
def test_model_matches_a_pennylane_reference(
    image_size: int, num_classes: int, filters: int, channels: int, encoding: str
) -> None:
    qml = pytest.importorskip("pennylane")
    torch.manual_seed(1)
    model = HurQCNN(image_size, num_classes, filters, channels=channels, encoding=encoding, dtype=torch.float64)
    images = torch.rand(3, image_size, image_size, dtype=torch.float64)
    schedule = model.schedule

    def embed(pixels):
        if encoding == "amplitude":
            qml.AmplitudeEmbedding(pixels, wires=range(schedule.qubits), normalize=True)
            return
        # FRQI: uniform address superposition, then RY(pi - 2p)|0> = sin(p)|0> + cos(p)|1> on the last wire.
        address = list(range(schedule.qubits - 1))
        for wire in address:
            qml.Hadamard(wires=wire)
        for index, value in enumerate(pixels):
            bits = [int(bit) for bit in format(index, f"0{len(address)}b")]
            qml.ctrl(qml.RY, control=address, control_values=bits)(math.pi - 2 * math.pi * value, wires=address[-1] + 1)

    def u_su4(params, wires):  # unitary.U_SU4
        qml.U3(params[0], params[1], params[2], wires=wires[0])
        qml.U3(params[3], params[4], params[5], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(params[6], wires=wires[0])
        qml.RZ(params[7], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RY(params[8], wires=wires[0])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.U3(params[9], params[10], params[11], wires=wires[0])
        qml.U3(params[12], params[13], params[14], wires=wires[1])

    def pooling(params, wires):  # unitary.Pooling_ansatz1
        qml.CRZ(params[0], wires=[wires[0], wires[1]])
        qml.PauliX(wires=wires[0])
        qml.CRX(params[1], wires=[wires[0], wires[1]])

    device = qml.device("default.qubit", wires=schedule.qubits)

    @qml.qnode(device)
    def circuit(features, conv_angles, pool_angles):
        embed(features)
        for index, pairs in enumerate(schedule.convolutions):
            for filter_angles in conv_angles[index]:
                for wires in pairs:
                    u_su4(filter_angles, wires)
            if index < len(schedule.poolings):
                for wires in schedule.poolings[index]:
                    pooling(pool_angles[index], wires)
        return qml.probs(wires=list(schedule.measured_wires))

    actual = model.exact_quantum_readout_probabilities(images)
    assert actual.shape == (3, channels, 2 ** len(schedule.measured_wires))
    for channel in range(channels):
        conv_angles = [layer.angles[channel].detach().numpy() for layer in model.layers[0::2]]
        pool_angles = [layer.angles[channel].detach().numpy() for layer in model.layers[1::2]]
        expected = torch.stack(
            [torch.as_tensor(circuit(image.reshape(-1).numpy(), conv_angles, pool_angles)) for image in images]
        )
        assert torch.allclose(actual[:, channel], expected, atol=1e-10)


def test_channel_chunks_with_checkpointing_match_one_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    import qcnn.hur

    torch.manual_seed(3)
    model = HurQCNN(8, 4, channels=5, dtype=torch.float64)
    images = torch.rand(6, 8, 8, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])

    def logits_and_gradients():
        model.zero_grad()
        logits = model(images)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        return logits.detach(), [parameter.grad.clone() for parameter in model.parameters()]

    whole_logits, whole_gradients = logits_and_gradients()
    # 6 images x 64 amplitudes x 2 channels per chunk -> chunks of 2, 2, 1 channels.
    monkeypatch.setattr(qcnn.hur, "MAX_CHUNK_AMPLITUDES", 6 * 64 * 2)
    chunked_logits, chunked_gradients = logits_and_gradients()

    assert torch.allclose(chunked_logits, whole_logits, atol=1e-12)
    for chunked, whole in zip(chunked_gradients, whole_gradients, strict=True):
        assert torch.allclose(chunked, whole, atol=1e-12)


def test_channel_readouts_are_averaged_before_the_head() -> None:
    torch.manual_seed(4)
    model = HurQCNN(8, 4, channels=3, dtype=torch.float64)
    images = torch.rand(2, 8, 8, dtype=torch.float64)

    readout = model.exact_quantum_readout_probabilities(images)
    expected = model.classifier(readout.reshape(2, -1) / 3)

    assert torch.allclose(readout.sum(dim=-1), torch.ones(2, 3, dtype=torch.float64))
    assert torch.allclose(model(images), expected)


def test_frqi_encoding_adds_the_colour_qubit_as_the_last_ring_wire() -> None:
    model = HurQCNN(32, 10, encoding="frqi")
    images = torch.rand(4, 32, 32)
    images[0] = 0.0

    state = model.encode(images)

    assert model.qubits == 11
    assert model.schedule.convolutions[0][0] == (0, 10)
    assert model.schedule.measured_wires == (0, 2, 4, 6, 8, 10)
    assert torch.allclose(state.abs().square().sum(dim=1), torch.ones(4))
    # A blank image is |+>^10 |1>: the colour qubit is the least significant bit.
    assert torch.allclose(state[0].real.reshape(1024, 2), torch.tensor([0.0, 1.0 / 32]).expand(1024, 2))


def test_benchmark_parameter_counts() -> None:
    assert format_trainable_parameter_stats_line(HurQCNN(32, 10)) == "q15 q2 q15 c330 Q32 C330"
    assert format_trainable_parameter_stats_line(HurQCNN(32, 10, 4)) == "q60 q2 q60 c330 Q122 C330"
    assert format_trainable_parameter_stats_line(HurQCNN(16, 2)) == "q15 q2 q15 q2 q15 q2 c6 Q51 C6"
    # FRQI adds the colour qubit: the same gates, 64 readout probabilities.
    assert format_trainable_parameter_stats_line(HurQCNN(32, 10, encoding="frqi")) == "q15 q2 q15 c650 Q32 C650"
    # Eight channels match the PCS-QCNN head (2,570), 112 channels its 39,434 parameters in total.
    assert format_trainable_parameter_stats_line(HurQCNN(32, 10, channels=8)) == "q120 q16 q120 c2570 Q256 C2570"
    assert format_trainable_parameter_stats_line(HurQCNN(32, 10, channels=112)) == (
        "q1680 q224 q1680 c35850 Q3584 C35850"
    )


def test_gradients_reach_every_parameter() -> None:
    torch.manual_seed(2)
    model = HurQCNN(8, 4)
    loss = torch.nn.functional.cross_entropy(model(torch.rand(5, 8, 8)), torch.tensor([0, 1, 2, 3, 0]))
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def test_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="power of two"):
        HurQCNN(12, 10)
    with pytest.raises(ValueError, match="spatial shape"):
        HurQCNN(8, 4)(torch.rand(2, 4, 4))
    with pytest.raises(ValueError, match="encoding must be"):
        HurQCNN(8, 4, encoding="basis")


def test_model_spec_round_trip() -> None:
    spec = build_hur_qcnn_model_spec(ArticleTrainingDefaults(), image_size=32, filters=1)
    model = instantiate_model(spec)

    assert isinstance(model, HurQCNN)
    assert model.angle_init_scale == 2 * math.pi
    assert all(0 <= value < 2 * math.pi for layer in model.layers for value in layer.angles.flatten().tolist())
    assert build_hur_qcnn_directory_name(filters=1) == "hur_qcnn_translated_mnist/su4_l1"
    assert "encoding" not in spec.constructor_kwargs

    frqi_spec = build_hur_qcnn_model_spec(ArticleTrainingDefaults(), image_size=32, filters=1, encoding="frqi")
    frqi_model = instantiate_model(frqi_spec)
    assert frqi_spec.constructor_kwargs["brightness_range"] == (0.0, math.pi)
    assert frqi_model.encoding == "frqi" and frqi_model.qubits == 11


def test_shared_hur_convolution_is_not_exact_one_wire_shift_equivariant() -> None:
    torch.manual_seed(17)
    model = HurQCNN(32, 10, dtype=torch.float64)
    state = torch.randn(1, 1, 1024, dtype=torch.complex128)
    state = state / state.norm(dim=-1, keepdim=True)

    def shift(value):
        return value.reshape(1, 1, *([2] * 10)).permute(0, 1, *range(3, 12), 2).reshape_as(value)

    layer = model.layers[0]
    assert (layer(shift(state)) - shift(layer(state))).norm() > 0.1


def test_article_hur_factory_round_trip() -> None:
    from qcnn import HurQCNN as ExportedHurQCNN
    from qcnn.article_training import (
        ArticleTrainingDefaults, build_hur_qcnn_directory_name, build_hur_qcnn_model_spec,
    )

    spec = build_hur_qcnn_model_spec(ArticleTrainingDefaults(), image_size=32, filters=1)
    model = instantiate_model(spec)
    assert isinstance(model, ExportedHurQCNN)
    assert model.angle_init_scale == 2 * math.pi
    assert build_hur_qcnn_directory_name(filters=1) == "hur_qcnn_translated_mnist/su4_l1"
