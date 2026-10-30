import math

import pytest
import torch

from qcnn import QCSQCNN
from qcnn.article_training import (
    ArticleTrainingDefaults,
    build_qcsqcnn_directory_name,
    build_qcsqcnn_model_spec,
    resolve_article_learning_rate,
)
from qcnn.layout import move_active_qubit_to_condition
from qcnn.model_spec import instantiate_model
from qcnn.model_stats import format_trainable_parameter_stats_line
from qcnn.qcs import (
    block_exponentials,
    cyclic_shift_eigenbasis,
    cyclic_shift_matrix,
    local_term_parameter_count,
    symmetry_block_generators,
    translation_invariant_generator,
)
from qcnn.quantum import _pauli_basis_matrices, hermitian_exponential
from qcnn.serialization import load_model_checkpoint, save_model_checkpoint


def _local_terms(parameters: torch.Tensor, feature_qubits: int) -> torch.Tensor:
    return torch.einsum("...p,pij->...ij", parameters.to(torch.complex128), _pauli_basis_matrices(2 + feature_qubits))


def _pixel_shift_matrix(index_qubits: int, feature_qubits: int) -> torch.Tensor:
    size = 2**index_qubits
    shift = torch.zeros(size, size, dtype=torch.complex128)
    for index in range(size):
        shift[(index + 1) % size, index] = 1.0
    return torch.kron(shift, torch.eye(2**feature_qubits, dtype=torch.complex128))


def _explicit_placed_term(local_term: torch.Tensor, *, first: int, index_qubits: int, feature_qubits: int) -> torch.Tensor:
    """Place a local term on index qubits (first, first + 1 mod n) by permuting tensor factors."""

    second = (first + 1) % index_qubits
    rest = [qubit for qubit in range(index_qubits) if qubit not in (first, second)]
    full = torch.kron(local_term, torch.eye(2 ** len(rest), dtype=torch.complex128))
    # canonical factor order: (first, second, features..., rest...) -> register order (index..., features...)
    canonical = [first, second, *[index_qubits + f for f in range(feature_qubits)], *rest]
    order = [canonical.index(qubit) for qubit in range(index_qubits + feature_qubits)]
    qubits = index_qubits + feature_qubits
    tensor = full.reshape((2,) * (2 * qubits))
    tensor = tensor.permute(*order, *[qubits + position for position in order])
    return tensor.reshape(2**qubits, 2**qubits)


@pytest.mark.parametrize(("index_qubits", "feature_qubits"), [(2, 1), (3, 1), (3, 2), (4, 1)])
def test_generator_sums_one_local_term_over_all_cyclic_pairs(index_qubits: int, feature_qubits: int) -> None:
    torch.manual_seed(0)
    local_term = _local_terms(torch.rand(local_term_parameter_count(feature_qubits), dtype=torch.float64), feature_qubits)

    generator = translation_invariant_generator(local_term, index_qubits=index_qubits, feature_qubits=feature_qubits)
    expected = sum(
        _explicit_placed_term(local_term, first=first, index_qubits=index_qubits, feature_qubits=feature_qubits)
        for first in range(index_qubits)
    )

    torch.testing.assert_close(generator, expected)


@pytest.mark.parametrize(("index_qubits", "feature_qubits"), [(2, 1), (3, 2), (5, 2)])
def test_layers_commute_with_qubit_cyclic_shifts_but_not_with_pixel_shifts(index_qubits: int, feature_qubits: int) -> None:
    torch.manual_seed(1)
    parameters = torch.rand(2, local_term_parameter_count(feature_qubits), dtype=torch.float64) * 2 * math.pi
    generator = translation_invariant_generator(
        _local_terms(parameters, feature_qubits),
        index_qubits=index_qubits,
        feature_qubits=feature_qubits,
    )
    unitary = hermitian_exponential(generator)
    qubit_shift = cyclic_shift_matrix(index_qubits, feature_qubits).to(torch.complex128)
    pixel_shift = _pixel_shift_matrix(index_qubits, feature_qubits)

    torch.testing.assert_close(qubit_shift @ unitary, unitary @ qubit_shift)
    assert (pixel_shift @ unitary - unitary @ pixel_shift).abs().max() > 0.1


@pytest.mark.parametrize("index_qubits", [2, 3, 4, 5])
def test_cyclic_shift_eigenbasis_is_unitary_and_block_diagonalizes_the_shift(index_qubits: int) -> None:
    basis, sizes = cyclic_shift_eigenbasis(index_qubits)
    shift = cyclic_shift_matrix(index_qubits).to(torch.complex128)

    torch.testing.assert_close(basis.conj().T @ basis, torch.eye(2**index_qubits, dtype=torch.complex128))
    eigenvalues = torch.cat(
        [
            torch.full((size,), complex(math.cos(2 * math.pi * k / index_qubits), math.sin(2 * math.pi * k / index_qubits)))
            for k, size in enumerate(sizes)
        ]
    ).to(torch.complex128)
    torch.testing.assert_close(basis.conj().T @ shift @ basis, torch.diag(eigenvalues))


def test_symmetry_blocks_reproduce_the_direct_exponential() -> None:
    torch.manual_seed(2)
    index_qubits, feature_qubits = 4, 2
    parameters = torch.rand(3, local_term_parameter_count(feature_qubits), dtype=torch.float64) * 2 * math.pi
    direct = hermitian_exponential(
        translation_invariant_generator(
            _local_terms(parameters, feature_qubits),
            index_qubits=index_qubits,
            feature_qubits=feature_qubits,
        )
    )
    register_basis = torch.kron(
        cyclic_shift_eigenbasis(index_qubits)[0],
        torch.eye(2**feature_qubits, dtype=torch.complex128),
    )

    blocks = block_exponentials(
        symmetry_block_generators(parameters, index_qubits=index_qubits, feature_qubits=feature_qubits)
    )
    assembled = torch.stack([torch.block_diag(*[block[index] for block in blocks]) for index in range(3)])

    torch.testing.assert_close(assembled, register_basis.conj().T @ direct @ register_basis)


def _brute_force_readout(model: QCSQCNN, images: torch.Tensor) -> torch.Tensor:
    state = model.encoder(images)
    layout = model.layer_layouts[0]
    for layer_idx, layer in enumerate(model.layers):
        unitaries = hermitian_exponential(
            translation_invariant_generator(
                _local_terms(layer.local_parameters.detach(), layer.feature_qubits),
                index_qubits=layer.index_qubits,
                feature_qubits=layer.feature_qubits,
            )
        )
        batch, x_dim, cx_dim, y_dim, cy_dim, feature_dim = state.shape
        us, vs = unitaries.shape[:2]
        split = state.reshape(batch, x_dim, us, cx_dim // us, y_dim, vs, cy_dim // vs, feature_dim)
        for sublayer in range(layer.sublayers):
            for axis in range(2):
                updated = torch.zeros_like(split)
                for u in range(us):
                    for v in range(vs):
                        block = split[:, :, u, :, :, v, :, :]
                        matrix = unitaries[u, v, sublayer, axis].reshape(x_dim, feature_dim, x_dim, feature_dim)
                        if axis == 0:
                            updated[:, :, u, :, :, v, :, :] = torch.einsum("xfgh,bgrywh->bxrywf", matrix, block)
                        else:
                            updated[:, :, u, :, :, v, :, :] = torch.einsum("yfgh,bxrgwh->bxrywf", matrix, block)
                split = updated
        state = split.reshape(batch, x_dim, cx_dim, y_dim, cy_dim, feature_dim)
        if layer_idx < model.quantum_layers - 1:
            state, layout = move_active_qubit_to_condition(state, layout, x_qubits_to_condition=1, y_qubits_to_condition=1)
    return model.measurement(state)


def test_model_matches_brute_force_evolution_in_the_computational_basis() -> None:
    torch.manual_seed(3)
    model = QCSQCNN(image_size=16, num_classes=10, feature_qubits=2, quantum_layers=3, sublayers=2, dtype=torch.float64)
    images = torch.rand(2, 16, 16, dtype=torch.float64)

    readout = model.exact_quantum_readout_probabilities(images)

    torch.testing.assert_close(readout, _brute_force_readout(model, images), atol=1e-12, rtol=0.0)
    torch.testing.assert_close(readout.sum(dim=(1, 2, 3)), torch.ones(2, dtype=torch.float64))
    assert model(images).shape == (2, 10)


def test_parameter_counts_match_the_pcsqcnn_head_and_quantum_core() -> None:
    head_matched = QCSQCNN(image_size=32, num_classes=10, feature_qubits=2, quantum_layers=3, sublayers=1)
    fully_matched = QCSQCNN(image_size=32, num_classes=10, feature_qubits=2, quantum_layers=3, sublayers=8)

    assert format_trainable_parameter_stats_line(head_matched) == "q512 q2048 q2048 c2570 Q4608 C2570"
    assert format_trainable_parameter_stats_line(fully_matched) == "q4096 q16384 q16384 c2570 Q36864 C2570"
    assert sum(parameter.numel() for parameter in fully_matched.parameters()) == 39434


def test_gradients_reach_every_quantum_parameter() -> None:
    torch.manual_seed(4)
    model = QCSQCNN(image_size=8, num_classes=10, feature_qubits=1, quantum_layers=2, sublayers=2)
    torch.nn.functional.cross_entropy(model(torch.rand(4, 8, 8)), torch.randint(0, 10, (4,))).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_qcsqcnn_rejects_layers_with_fewer_than_two_index_qubits() -> None:
    with pytest.raises(ValueError, match="at least two active index qubits"):
        QCSQCNN(image_size=8, num_classes=10, quantum_layers=3)


def test_article_model_spec_uses_hybrid_learning_rate_and_round_trips(tmp_path) -> None:
    defaults = ArticleTrainingDefaults()
    spec = build_qcsqcnn_model_spec(
        defaults,
        image_size=8,
        feature_qubits=1,
        quantum_layers=2,
        sublayers=2,
        brightness_range=(0.0, math.pi),
    )

    assert resolve_article_learning_rate(defaults=defaults, model_spec=spec) == defaults.hybrid_learning_rate
    assert spec.constructor_kwargs["generator_init_scale"] == defaults.multiplexer_init_scale
    assert (
        build_qcsqcnn_directory_name(feature_qubits=2, quantum_layers=3, sublayers=8)
        == "qcsqcnn_translated_mnist/fq2_ql3_sl8"
    )

    model = instantiate_model(spec)
    path = tmp_path / "checkpoint.pt"
    save_model_checkpoint(model, spec, path)
    restored = load_model_checkpoint(path).model
    images = torch.rand(3, 8, 8)

    torch.testing.assert_close(restored(images), model(images))
