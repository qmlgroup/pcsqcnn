import math

import pytest
import torch

from qcnn import HendersonQCNN
from qcnn.article_training import (
    ArticleTrainingDefaults,
    build_henderson_qcnn_directory_name,
    build_henderson_qcnn_model_spec,
    resolve_article_learning_rate,
)
from qcnn.model_spec import instantiate_model
from qcnn.model_stats import format_trainable_parameter_stats_line
from qcnn.henderson import (
    _pauli_string_coefficients,
    use_compiled_block_transfer,
)
from qcnn.quantum import _HermitianExponential, _pauli_basis_matrices, pauli_coefficients_to_unitary
from qcnn.serialization import load_model_checkpoint, save_model_checkpoint


def _ry_density_matrix(theta: float) -> torch.Tensor:
    state = torch.tensor([math.cos(theta / 2), math.sin(theta / 2)], dtype=torch.complex128)
    return torch.outer(state, state.conj())


def _kron(*matrices: torch.Tensor) -> torch.Tensor:
    result = matrices[0]
    for matrix in matrices[1:]:
        result = torch.kron(result, matrix)
    return result


def _brute_force_block_distributions(model: HendersonQCNN, image: torch.Tensor) -> torch.Tensor:
    """Simulate every block with explicit density matrices, independently of the transfer tensors."""

    start, end = model.brightness_range
    size = image.shape[0]
    channel_outputs = []
    for channel in range(model.channels):
        lattice = [
            [_ry_density_matrix(start + (end - start) * float(image[u, v])) for v in range(size)]
            for u in range(size)
        ]
        for layer in model.layers:
            unitary = layer.block_unitaries()[channel].detach()
            corrections = layer.pooling_unitaries()[channel].detach() if layer.pool else None
            half = len(lattice) // 2
            next_lattice = []
            for i in range(half):
                row = []
                for j in range(half):
                    rho = _kron(
                        lattice[2 * i][2 * j],
                        lattice[2 * i][2 * j + 1],
                        lattice[2 * i + 1][2 * j],
                        lattice[2 * i + 1][2 * j + 1],
                    )
                    rho = unitary @ rho @ unitary.conj().T
                    if corrections is None:
                        row.append(torch.diagonal(rho).real)
                    else:
                        blocks = rho.reshape(2, 8, 2, 8)
                        row.append(
                            sum(
                                corrections[m] @ blocks[:, m, :, m] @ corrections[m].conj().T
                                for m in range(8)
                            )
                        )
                next_lattice.append(row)
            lattice = next_lattice
        channel_outputs.append(torch.stack([torch.stack(row) for row in lattice]))
    return torch.stack(channel_outputs)


@pytest.mark.parametrize(
    ("quantum_layers", "image_size", "channels"),
    [(1, 4, 2), (2, 4, 2), (3, 8, 2), (2, 8, 1)],
)
def test_transfer_simulation_matches_brute_force_density_matrices(
    quantum_layers: int,
    image_size: int,
    channels: int,
) -> None:
    torch.manual_seed(1)
    model = HendersonQCNN(
        image_size=image_size,
        num_classes=10,
        channels=channels,
        quantum_layers=quantum_layers,
        dtype=torch.float64,
    )
    images = torch.rand(2, image_size, image_size, dtype=torch.float64)

    distributions = model.block_readout_distributions(images).detach()

    for index in range(images.shape[0]):
        expected = _brute_force_block_distributions(model, images[index])
        torch.testing.assert_close(distributions[index], expected, atol=1e-10, rtol=0.0)


def test_readout_is_a_normalized_probability_vector() -> None:
    torch.manual_seed(2)
    model = HendersonQCNN(image_size=16, num_classes=10, channels=3, quantum_layers=2, dtype=torch.float64)
    images = torch.rand(4, 16, 16, dtype=torch.float64)

    distributions = model.block_readout_distributions(images)
    readout = model.exact_quantum_readout_probabilities(images)

    assert readout.shape == (4, 3, 4, 4, 16)
    assert model.readout_shape == (3, 4, 4, 16)
    assert torch.all(distributions > -1e-12)
    torch.testing.assert_close(distributions.sum(dim=-1), torch.ones(4, 3, 4, 4, dtype=torch.float64))
    torch.testing.assert_close(readout.sum(dim=(1, 2, 3, 4)), torch.ones(4, dtype=torch.float64))
    assert model(images).shape == (4, 10)


def test_readout_rolls_with_cyclic_image_shifts_by_the_coarse_stride() -> None:
    torch.manual_seed(3)
    quantum_layers = 2
    model = HendersonQCNN(image_size=16, num_classes=10, channels=2, quantum_layers=quantum_layers, dtype=torch.float64)
    images = torch.rand(2, 16, 16, dtype=torch.float64)
    stride = 2**quantum_layers

    readout = model.block_readout_distributions(images)
    shifted_x = model.block_readout_distributions(torch.roll(images, shifts=stride, dims=1))
    shifted_y = model.block_readout_distributions(torch.roll(images, shifts=-stride, dims=2))

    torch.testing.assert_close(shifted_x, torch.roll(readout, shifts=1, dims=2))
    torch.testing.assert_close(shifted_y, torch.roll(readout, shifts=-1, dims=3))


def test_chunked_inference_matches_the_autograd_path() -> None:
    torch.manual_seed(4)
    model = HendersonQCNN(image_size=16, num_classes=10, channels=2, quantum_layers=2, dtype=torch.float64)
    images = torch.rand(600, 16, 16, dtype=torch.float64)

    with torch.no_grad():
        chunked = model(images)
    unchunked = model(images)

    # Different batch shapes can select different floating-point GEMM kernels.
    torch.testing.assert_close(chunked, unchunked.detach(), atol=1e-14, rtol=1e-13)


def test_parameter_counts_match_the_documented_matched_baselines() -> None:
    expectations = {
        (12, 3): ("q3456 q3456 q3072 c30730 Q9984 C30730", 40714),
        (1, 1): ("q256 c40970 Q256 C40970", 41226),
        (1, 3): ("q288 q288 q256 c2570 Q832 C2570", 3402),
    }
    for (channels, quantum_layers), (stats_line, total) in expectations.items():
        model = HendersonQCNN(image_size=32, num_classes=10, channels=channels, quantum_layers=quantum_layers)

        assert format_trainable_parameter_stats_line(model) == stats_line
        assert sum(parameter.numel() for parameter in model.parameters()) == total


def test_pooling_transfer_preserves_the_trace() -> None:
    torch.manual_seed(5)
    layer = HendersonQCNN(image_size=4, num_classes=10, channels=3, quantum_layers=2, dtype=torch.float64).layers[0]

    transfer = layer.transfer_tensor()

    expected = torch.zeros(3, 256, dtype=torch.float64)
    expected[:, 0] = 1.0
    torch.testing.assert_close(transfer[:, 0, :], expected)


def test_pauli_string_coefficients_match_the_dense_basis() -> None:
    torch.manual_seed(6)
    matrices = torch.randn(5, 16, 16, dtype=torch.complex128)

    expected = torch.einsum("bij,nji->nb", _pauli_basis_matrices(4), matrices)

    torch.testing.assert_close(_pauli_string_coefficients(matrices), expected)


@pytest.mark.parametrize("num_qubits", [1, 4])
def test_hermitian_exponential_matches_matrix_exp_values_and_gradients(num_qubits: int) -> None:
    torch.manual_seed(7)
    basis = _pauli_basis_matrices(num_qubits)
    parameters = (torch.rand(3, 4**num_qubits, dtype=torch.float64) * 2 * math.pi).requires_grad_()
    weights = torch.randn(3, 2**num_qubits, 2**num_qubits, dtype=torch.complex128)

    spectral = _HermitianExponential.apply(torch.einsum("np,pij->nij", parameters.to(torch.complex128), basis))
    reference = pauli_coefficients_to_unitary(parameters)
    (spectral_grad,) = torch.autograd.grad((spectral * weights).real.sum(), parameters)
    (reference_grad,) = torch.autograd.grad((reference * weights).real.sum(), parameters)

    torch.testing.assert_close(spectral, reference, atol=1e-11, rtol=0.0)
    torch.testing.assert_close(spectral_grad, reference_grad, atol=1e-9, rtol=1e-9)


def test_hermitian_exponential_gradient_is_finite_for_degenerate_spectra() -> None:
    parameters = torch.zeros(1, 256, dtype=torch.float64)
    parameters[0, 0] = 0.7
    parameters.requires_grad_()
    basis = _pauli_basis_matrices(4)

    unitary = _HermitianExponential.apply(torch.einsum("np,pij->nij", parameters.to(torch.complex128), basis))
    (gradient,) = torch.autograd.grad(unitary.real.sum(), parameters)
    (reference,) = torch.autograd.grad(pauli_coefficients_to_unitary(parameters).real.sum(), parameters)

    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, reference, atol=1e-12, rtol=0.0)


def test_gradients_reach_every_quantum_parameter() -> None:
    torch.manual_seed(8)
    model = HendersonQCNN(image_size=8, num_classes=10, channels=2, quantum_layers=3)
    images = torch.rand(5, 8, 8)
    labels = torch.randint(0, 10, (5,))

    torch.nn.functional.cross_entropy(model(images), labels).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_compiled_block_transfer_switch_restores_the_eager_path() -> None:
    import qcnn.henderson as henderson

    use_compiled_block_transfer(True)
    assert henderson._COMPILED_CONTRACTIONS is not None
    use_compiled_block_transfer(False)
    assert henderson._COMPILED_CONTRACTIONS is None


def test_henderson_qcnn_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="power of two"):
        HendersonQCNN(image_size=12, num_classes=10)
    with pytest.raises(ValueError, match="at least 2\\*\\*quantum_layers"):
        HendersonQCNN(image_size=4, num_classes=10, quantum_layers=3)
    model = HendersonQCNN(image_size=8, num_classes=10)
    with pytest.raises(ValueError, match="spatial shape"):
        model(torch.rand(2, 4, 4))


def test_article_model_spec_uses_hybrid_learning_rate_and_round_trips(tmp_path) -> None:
    defaults = ArticleTrainingDefaults()
    spec = build_henderson_qcnn_model_spec(
        defaults,
        image_size=8,
        channels=2,
        quantum_layers=2,
        brightness_range=(0.0, math.pi),
    )

    assert resolve_article_learning_rate(defaults=defaults, model_spec=spec) == defaults.hybrid_learning_rate
    assert spec.constructor_kwargs["block_init_scale"] == defaults.multiplexer_init_scale
    assert build_henderson_qcnn_directory_name(channels=12, quantum_layers=3) == "henderson_qcnn_translated_mnist/ch12_ql3"

    model = instantiate_model(spec)
    path = tmp_path / "checkpoint.pt"
    save_model_checkpoint(model, spec, path)
    restored = load_model_checkpoint(path).model
    images = torch.rand(3, 8, 8)

    torch.testing.assert_close(restored(images), model(images))


def test_pennylane_circuit_parity_with_deferred_measurement_pooling() -> None:
    qml = pytest.importorskip("pennylane")
    import numpy as np

    torch.manual_seed(9)
    model = HendersonQCNN(image_size=4, num_classes=10, channels=2, quantum_layers=2, dtype=torch.float64)
    image = torch.rand(1, 4, 4, dtype=torch.float64)
    ours = model.block_readout_distributions(image)[0].detach().numpy()
    device = qml.device("default.qubit", wires=16)

    for channel in range(model.channels):
        first = model.layers[0].block_unitaries()[channel].detach().numpy()
        corrections = model.layers[0].pooling_unitaries()[channel].detach().numpy()
        second = model.layers[1].block_unitaries()[channel].detach().numpy()

        @qml.qnode(device)
        def circuit(pixels):
            for u in range(4):
                for v in range(4):
                    qml.RY(np.pi * pixels[u, v], wires=4 * u + v)
            kept = []
            for i in range(2):
                for j in range(2):
                    wires = [8 * i + 2 * j, 8 * i + 2 * j + 1, 8 * i + 4 + 2 * j, 8 * i + 4 + 2 * j + 1]
                    qml.QubitUnitary(first, wires=wires)
                    for outcome in range(8):
                        bits = [(outcome >> 2) & 1, (outcome >> 1) & 1, outcome & 1]
                        qml.ctrl(
                            qml.QubitUnitary(corrections[outcome], wires=wires[0]),
                            control=wires[1:],
                            control_values=bits,
                        )
                    kept.append(wires[0])
            qml.QubitUnitary(second, wires=kept)
            return qml.probs(wires=kept)

        np.testing.assert_allclose(ours[channel, 0, 0], circuit(image[0].numpy()), atol=1e-10)


def test_single_layer_reproduces_the_pennylane_quanvolution_demo_kernel() -> None:
    qml = pytest.importorskip("pennylane")
    scipy_linalg = pytest.importorskip("scipy.linalg")
    import numpy as np

    rng = np.random.default_rng(0)
    random_parameters = rng.uniform(high=2 * np.pi, size=(1, 4))
    device = qml.device("default.qubit", wires=4)

    @qml.qnode(device)
    def demo_circuit(phi):
        for wire in range(4):
            qml.RY(np.pi * phi[wire], wires=wire)
        qml.RandomLayers(random_parameters, wires=list(range(4)))
        return [qml.expval(qml.PauliZ(wire)) for wire in range(4)]

    demo_unitary = qml.matrix(qml.RandomLayers(random_parameters, wires=list(range(4))), wire_order=range(4))
    generator = -1j * scipy_linalg.logm(demo_unitary)
    coefficients = np.real(np.einsum("bij,ji->b", _pauli_basis_matrices(4).numpy(), generator)) / 16
    model = HendersonQCNN(image_size=2, num_classes=10, channels=1, quantum_layers=1, dtype=torch.float64)
    with torch.no_grad():
        model.layers[0].block_parameters.copy_(torch.from_numpy(coefficients)[None])
    patches = torch.rand(4, 2, 2, dtype=torch.float64)

    probabilities = model.block_readout_distributions(patches)[:, 0, 0, 0].detach().numpy()
    signs = np.array([[1 - 2 * ((outcome >> (3 - wire)) & 1) for outcome in range(16)] for wire in range(4)])
    expected = np.array([demo_circuit(patch.flatten().numpy()) for patch in patches])

    np.testing.assert_allclose(probabilities @ signs.T, expected, atol=1e-10)
