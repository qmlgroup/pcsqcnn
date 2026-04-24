import math

import pytest
import torch

from qcnn import (
    FiniteShotHistogram2D,
    FrqiEncoder2D,
    MarginalMeasurement2D,
    ModeMultiplexer2D,
    RegisterLayout2D,
    move_active_qubit_to_condition,
    pauli_coefficients_to_unitary,
)
from qcnn.quantum import (
    _apply_fourier_junction_1d,
    _apply_fourier_junction_2d,
    _apply_iqft_2d,
    _apply_qft_2d,
    _apply_unitary_to_active_axis_1d,
    _materialized_fourier_junction_phase,
    _materialized_pauli_basis_matrices,
)


def make_encoder(
    feature_qubits: int = 1,
    *,
    brightness_range: tuple[float, float] = (0.0, math.pi),
) -> FrqiEncoder2D:
    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        return FrqiEncoder2D(feature_qubits=feature_qubits, brightness_range=brightness_range)


def make_measurement() -> MarginalMeasurement2D:
    return MarginalMeasurement2D()


def make_histogram(shot_budget: int | None = None) -> FiniteShotHistogram2D:
    return FiniteShotHistogram2D(shot_budget=shot_budget)


def make_multiplexer(
    layout: RegisterLayout2D,
    *,
    x_condition_qubits_to_use: int = 0,
    y_condition_qubits_to_use: int = 0,
    multiplexer_init_scale: float = 0.05,
    dtype: torch.dtype = torch.float32,
) -> ModeMultiplexer2D:
    return ModeMultiplexer2D(
        layout,
        x_condition_qubits_to_use=x_condition_qubits_to_use,
        y_condition_qubits_to_use=y_condition_qubits_to_use,
        multiplexer_init_scale=multiplexer_init_scale,
        dtype=dtype,
    )


def test_encoder_rejects_non_bxy_inputs() -> None:
    encoder = make_encoder()

    with pytest.raises(ValueError, match=r"\[B, X, Y\]"):
        encoder(torch.zeros(2, 1, 4, 4))


def test_encoder_rejects_non_square_inputs() -> None:
    encoder = make_encoder()

    with pytest.raises(ValueError, match="square"):
        encoder(torch.zeros(2, 4, 8))


def test_encoder_rejects_non_power_of_two_inputs() -> None:
    encoder = make_encoder()

    with pytest.raises(ValueError, match="power of two"):
        encoder(torch.zeros(2, 6, 6))


def test_encoder_rejects_invalid_brightness_range() -> None:
    with pytest.raises(ValueError, match="a < b"):
        FrqiEncoder2D(brightness_range=(1.0, 1.0))


def test_encoder_maps_normalized_images_into_configured_angle_range() -> None:
    with pytest.warns(UserWarning) as caught:
        encoder = FrqiEncoder2D(feature_qubits=1, brightness_range=(-2.0, 3.0))
    images = torch.tensor([[[0.0, 0.5], [1.0, 0.25]]], dtype=torch.float32)
    mapped = -2.0 + 5.0 * images

    encoded = encoder(images)
    expected = torch.zeros((1, 2, 1, 2, 1, 2), dtype=torch.complex64)
    expected[:, :, 0, :, 0, 0] = torch.sin(mapped).to(torch.complex64)
    expected[:, :, 0, :, 0, 1] = torch.cos(mapped).to(torch.complex64)

    assert encoded.shape == (1, 2, 1, 2, 1, 2)
    assert torch.allclose(encoded, expected)
    messages = [str(warning.message) for warning in caught]
    assert any("brightness_range deviates" in message for message in messages)
    assert any("1/sqrt(XY)" in message for message in messages)


def test_encoder_only_populates_color_subspace_when_auxiliary_features_exist() -> None:
    encoder = make_encoder(feature_qubits=3)
    images = torch.rand(2, 4, 4)

    encoded = encoder(images)

    assert encoded.shape == (2, 4, 1, 4, 1, 8)
    assert torch.allclose(encoded[..., 2:], torch.zeros_like(encoded[..., 2:]))


def test_encoder_output_norm_is_sqrt_hw_without_global_normalization() -> None:
    encoder = make_encoder()
    images = torch.rand(3, 8, 8)

    encoded = encoder(images)
    norms = encoded.reshape(3, -1).norm(dim=1)

    assert torch.allclose(norms, torch.full((3,), math.sqrt(64.0)))


def test_encoder_supports_backpropagation_to_input_images() -> None:
    encoder = make_encoder()
    images = torch.rand(2, 4, 4, requires_grad=True)

    encoded = encoder(images)
    loss = encoded.real[..., 0].sum()
    loss.backward()

    assert images.grad is not None
    assert images.grad.shape == images.shape
    assert torch.count_nonzero(images.grad) > 0


def test_encoder_is_vmap_compatible() -> None:
    encoder = make_encoder()
    images = torch.rand(3, 4, 4)

    encoded = torch.func.vmap(lambda image: encoder(image.unsqueeze(0)).squeeze(0))(images)
    expected = torch.stack([encoder(image.unsqueeze(0)).squeeze(0) for image in images], dim=0)

    assert encoded.shape == (3, 4, 1, 4, 1, 2)
    assert torch.allclose(encoded, expected)


def test_pauli_coefficients_to_unitary_preserves_leading_axes_elementwise() -> None:
    angle = torch.tensor(0.25, dtype=torch.float64)
    params = torch.zeros((2, 3, 3), dtype=torch.float64)
    params[1, 2, 0] = angle

    unitaries = pauli_coefficients_to_unitary(params)

    eye = torch.eye(2, dtype=torch.complex128)
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    expected = eye.repeat(2, 3, 1, 1)
    expected[1, 2] = torch.cos(angle) * eye + 1j * torch.sin(angle) * pauli_x

    assert unitaries.shape == (2, 3, 2, 2)
    assert unitaries.dtype == torch.complex128
    assert torch.allclose(unitaries, expected)


def test_materialized_pauli_basis_cache_reuses_same_device_dtype_tensor() -> None:
    first = _materialized_pauli_basis_matrices(1, device=torch.device("cpu"), dtype=torch.complex64)
    second = _materialized_pauli_basis_matrices(1, device=torch.device("cpu"), dtype=torch.complex64)

    assert first is second
    assert first.dtype == torch.complex64
    assert first.device.type == "cpu"


def test_pauli_coefficients_to_unitary_uses_xyz_order_for_one_qubit() -> None:
    angle = torch.tensor(0.37, dtype=torch.float64)
    params = torch.eye(3, dtype=torch.float64) * angle

    unitaries = pauli_coefficients_to_unitary(params)

    eye = torch.eye(2, dtype=torch.complex128).unsqueeze(0)
    basis = torch.tensor(
        [
            [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]],
            [[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]],
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]],
        ],
        dtype=torch.complex128,
    )
    expected = torch.cos(angle) * eye + 1j * torch.sin(angle) * basis

    assert unitaries.shape == (3, 2, 2)
    assert torch.allclose(unitaries, expected)


def test_pauli_coefficients_to_unitary_returns_batched_unitaries() -> None:
    torch.manual_seed(0)
    params = torch.randn((2, 3, 15), dtype=torch.float32)

    unitaries = pauli_coefficients_to_unitary(params)
    gram = unitaries @ unitaries.conj().transpose(-1, -2)
    identity = torch.eye(4, dtype=torch.complex64).expand(2, 3, 4, 4)

    assert unitaries.shape == (2, 3, 4, 4)
    assert unitaries.dtype == torch.complex64
    assert torch.allclose(gram, identity, atol=1e-5, rtol=1e-5)


def test_pauli_coefficients_to_unitary_supports_backpropagation() -> None:
    params = torch.randn((2, 15), dtype=torch.float64, requires_grad=True)

    unitaries = pauli_coefficients_to_unitary(params)
    loss = unitaries.real.sum() + unitaries.imag.sum()
    loss.backward()

    assert params.grad is not None
    assert params.grad.shape == params.shape
    assert torch.count_nonzero(params.grad.abs()) > 0


def test_pauli_coefficients_to_unitary_rejects_invalid_parameter_count() -> None:
    with pytest.raises(ValueError, match=r"4\*\*f - 1"):
        pauli_coefficients_to_unitary(torch.zeros((2, 5), dtype=torch.float32))


@pytest.mark.parametrize(
    "params",
    [
        torch.zeros(3, dtype=torch.int64),
        torch.zeros(3, dtype=torch.complex64),
    ],
)
def test_pauli_coefficients_to_unitary_rejects_non_real_floating_inputs(
    params: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="real floating"):
        pauli_coefficients_to_unitary(params)


def test_fourier_helpers_roundtrip_canonical_states() -> None:
    state = torch.randn((2, 4, 2, 4, 3, 2), dtype=torch.complex128)

    restored = _apply_iqft_2d(_apply_qft_2d(state))

    assert restored.shape == state.shape
    assert torch.allclose(restored, state, atol=1e-10, rtol=1e-10)


def test_fourier_helpers_only_transform_active_index_axes() -> None:
    state = torch.zeros((2, 2, 2, 2, 2, 2), dtype=torch.complex128)
    state[0, :, 1, :, 0, 1] = torch.tensor(
        [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
        dtype=torch.complex128,
    )

    transformed = _apply_qft_2d(state)
    expected_slice = torch.fft.fft2(state[0, :, 1, :, 0, 1], dim=(0, 1))

    assert torch.allclose(transformed[0, :, 1, :, 0, 1], expected_slice)
    assert torch.allclose(transformed[0, :, 0, :, 0, 1], torch.zeros((2, 2), dtype=torch.complex128))
    assert torch.allclose(transformed[1], torch.zeros_like(transformed[1]))


def test_apply_unitary_to_active_axis_roundtrips_with_adjoint() -> None:
    torch.manual_seed(0)
    state = torch.randn((2, 4, 3, 2, 5, 2), dtype=torch.complex128)
    params = torch.randn(15, dtype=torch.float64)
    unitary = pauli_coefficients_to_unitary(params)

    transformed = _apply_unitary_to_active_axis_1d(state, unitary, active_axis=1)
    restored = _apply_unitary_to_active_axis_1d(transformed, unitary.conj().transpose(-1, -2), active_axis=1)

    assert transformed.shape == state.shape
    assert torch.allclose(restored, state, atol=1e-10, rtol=1e-10)


def test_apply_unitary_to_active_axis_rejects_incompatible_inputs() -> None:
    state = torch.randn((2, 4, 1, 4, 1, 2), dtype=torch.complex64)
    unitary = torch.eye(4, dtype=torch.complex64)

    with pytest.raises(ValueError, match="active_axis"):
        _apply_unitary_to_active_axis_1d(state, unitary, active_axis=2)

    with pytest.raises(ValueError, match="complex unitary matrix"):
        _apply_unitary_to_active_axis_1d(state, torch.eye(4, dtype=torch.float32), active_axis=1)

    with pytest.raises(ValueError, match="must match the selected active axis"):
        _apply_unitary_to_active_axis_1d(state, torch.eye(2, dtype=torch.complex64), active_axis=1)


def test_fourier_junction_1d_matches_explicit_x_axis_composition() -> None:
    layout = RegisterLayout2D(image_size=8, feature_qubits=2, x_condition_qubits=1, y_condition_qubits=1)
    state = torch.randn(layout.state_shape(batch_size=2), dtype=torch.complex128)

    explicit = torch.fft.ifft(state, dim=1)
    explicit, explicit_layout = move_active_qubit_to_condition(
        explicit,
        layout,
        x_qubits_to_condition=1,
        y_qubits_to_condition=0,
    )
    explicit = torch.fft.fft(explicit, dim=1)

    junction = _apply_fourier_junction_1d(state, active_axis=1, condition_axis=2)

    assert explicit_layout.state_shape(batch_size=2) == junction.shape
    assert torch.allclose(junction, explicit, atol=1e-10, rtol=1e-10)


def test_materialized_fourier_junction_phase_reuses_cached_tensor() -> None:
    first = _materialized_fourier_junction_phase(
        8,
        device=torch.device("cpu"),
        dtype=torch.complex64,
    )
    second = _materialized_fourier_junction_phase(
        8,
        device=torch.device("cpu"),
        dtype=torch.complex64,
    )

    assert first is second
    assert first.shape == (4,)
    assert first.dtype == torch.complex64
    assert first.device.type == "cpu"


def test_materialized_fourier_junction_phase_rejects_too_small_dimension() -> None:
    with pytest.raises(ValueError, match="at least two active basis states"):
        _materialized_fourier_junction_phase(
            1,
            device=torch.device("cpu"),
            dtype=torch.complex64,
        )


def test_materialized_fourier_junction_phase_rejects_odd_dimension() -> None:
    with pytest.raises(ValueError, match="even active dimension"):
        _materialized_fourier_junction_phase(
            3,
            device=torch.device("cpu"),
            dtype=torch.complex64,
        )


def test_fourier_junction_2d_matches_explicit_fourier_pooling_composition() -> None:
    layout = RegisterLayout2D(image_size=8, feature_qubits=2, x_condition_qubits=1, y_condition_qubits=1)
    state = torch.randn(layout.state_shape(batch_size=2), dtype=torch.complex128)

    explicit = _apply_iqft_2d(state)
    explicit, explicit_layout = move_active_qubit_to_condition(
        explicit,
        layout,
        x_qubits_to_condition=1,
        y_qubits_to_condition=1,
    )
    explicit = _apply_qft_2d(explicit)

    junction, junction_layout = _apply_fourier_junction_2d(state, layout)

    assert junction_layout == explicit_layout
    assert junction.shape == explicit.shape
    assert torch.allclose(junction, explicit, atol=1e-10, rtol=1e-10)


def test_fourier_junction_2d_rejects_non_complex_inputs() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=1)

    with pytest.raises(ValueError, match="complex"):
        _apply_fourier_junction_2d(torch.zeros(layout.state_shape(batch_size=1), dtype=torch.float32), layout)


def test_fourier_junction_2d_rejects_shape_mismatch() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=1, x_condition_qubits=1)

    with pytest.raises(ValueError, match="does not match the layout contract"):
        _apply_fourier_junction_2d(torch.zeros((1, 4, 1, 4, 1, 2), dtype=torch.complex64), layout)


def test_fourier_junction_2d_rejects_exhausted_active_axes() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=1, x_condition_qubits=2, y_condition_qubits=1)
    state = torch.zeros(layout.state_shape(batch_size=1), dtype=torch.complex64)

    with pytest.raises(ValueError, match="at least two active x-register basis states"):
        _apply_fourier_junction_2d(state, layout)


def test_mode_multiplexer_acts_as_identity_when_block_parameters_are_zero() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=2, x_condition_qubits=1, y_condition_qubits=1)
    multiplexer = make_multiplexer(
        layout,
        x_condition_qubits_to_use=1,
        y_condition_qubits_to_use=1,
    )
    with torch.no_grad():
        multiplexer.block_parameters.zero_()
    state = torch.randn(layout.state_shape(batch_size=2), dtype=torch.complex64)

    transformed = multiplexer(state)

    assert transformed.shape == state.shape
    assert torch.allclose(transformed, state)


def test_mode_multiplexer_randomly_initializes_block_parameters_by_default() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=2, x_condition_qubits=1, y_condition_qubits=1)
    torch.manual_seed(123)
    multiplexer = make_multiplexer(layout)

    assert multiplexer.multiplexer_init_scale == pytest.approx(0.05)
    assert torch.count_nonzero(multiplexer.block_parameters) > 0


def test_mode_multiplexer_initialization_follows_manual_seed_deterministically() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=2, x_condition_qubits=1, y_condition_qubits=1)
    torch.manual_seed(123)
    first = make_multiplexer(layout)
    torch.manual_seed(123)
    second = make_multiplexer(layout)
    torch.manual_seed(124)
    third = make_multiplexer(layout)

    assert torch.allclose(first.block_parameters, second.block_parameters)
    assert not torch.allclose(first.block_parameters, third.block_parameters)


def test_mode_multiplexer_applies_mode_dependent_feature_blocks_elementwise_over_batch() -> None:
    layout = RegisterLayout2D(image_size=2, feature_qubits=1)
    multiplexer = make_multiplexer(layout, dtype=torch.float64)
    params = torch.zeros_like(multiplexer.block_parameters)
    params[0, 0, 0, 0, 0] = 0.2
    params[0, 0, 1, 1, 2] = -0.35
    with torch.no_grad():
        multiplexer.block_parameters.copy_(params)

    state = torch.zeros(layout.state_shape(batch_size=2), dtype=torch.complex128)
    state[0, 0, 0, 0, 0] = torch.tensor([1.0 + 0.0j, 0.5 + 0.25j], dtype=torch.complex128)
    state[0, 1, 0, 1, 0] = torch.tensor([0.2 - 0.1j, -0.3 + 0.4j], dtype=torch.complex128)
    state[1, 0, 0, 0, 0] = torch.tensor([-0.7 + 0.2j, 0.6 + 0.1j], dtype=torch.complex128)
    state[1, 1, 0, 1, 0] = torch.tensor([0.3 + 0.8j, -0.2 + 0.5j], dtype=torch.complex128)

    transformed = multiplexer(state)
    blocks = pauli_coefficients_to_unitary(params).to(dtype=torch.complex128)
    expected = state.clone()
    expected[:, 0, 0, 0, 0] = torch.einsum("ij,bj->bi", blocks[0, 0, 0, 0], state[:, 0, 0, 0, 0])
    expected[:, 1, 0, 1, 0] = torch.einsum("ij,bj->bi", blocks[0, 0, 1, 1], state[:, 1, 0, 1, 0])

    assert torch.allclose(transformed, expected)


def test_mode_multiplexer_partial_conditioning_ignores_older_condition_bits() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=1, x_condition_qubits=2)
    multiplexer = make_multiplexer(layout, x_condition_qubits_to_use=1, dtype=torch.float64)
    params = torch.zeros_like(multiplexer.block_parameters)
    params[0, 0, 0, 0, 0] = 0.15
    params[1, 0, 0, 0, 0] = -0.25
    with torch.no_grad():
        multiplexer.block_parameters.copy_(params)

    state = torch.zeros(layout.state_shape(batch_size=1), dtype=torch.complex128)
    state[0, 0, 0, 0, 0] = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128)
    state[0, 0, 1, 0, 0] = torch.tensor([0.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex128)
    state[0, 0, 2, 0, 0] = torch.tensor([0.5 + 0.0j, 0.5 + 0.0j], dtype=torch.complex128)
    state[0, 0, 3, 0, 0] = torch.tensor([0.25 + 0.0j, -0.75 + 0.0j], dtype=torch.complex128)

    transformed = multiplexer(state)
    branch_blocks = pauli_coefficients_to_unitary(params[:, 0, 0, 0]).to(dtype=torch.complex128)

    assert torch.allclose(
        transformed[0, 0, 0, 0, 0],
        branch_blocks[0] @ state[0, 0, 0, 0, 0],
    )
    assert torch.allclose(
        transformed[0, 0, 1, 0, 0],
        branch_blocks[0] @ state[0, 0, 1, 0, 0],
    )
    assert torch.allclose(
        transformed[0, 0, 2, 0, 0],
        branch_blocks[1] @ state[0, 0, 2, 0, 0],
    )
    assert torch.allclose(
        transformed[0, 0, 3, 0, 0],
        branch_blocks[1] @ state[0, 0, 3, 0, 0],
    )


def test_mode_multiplexer_uses_newest_condition_bits_after_multiple_pooling_steps() -> None:
    base_layout = RegisterLayout2D(image_size=8, feature_qubits=1)
    state = torch.zeros(base_layout.state_shape(batch_size=1), dtype=torch.complex128)
    state[0, 7, 0, 0, 0] = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128)
    state[0, 5, 0, 0, 0] = torch.tensor([0.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex128)

    moved_state, moved_layout = move_active_qubit_to_condition(state, base_layout, x_qubits_to_condition=2)
    multiplexer = make_multiplexer(moved_layout, x_condition_qubits_to_use=1, dtype=torch.float64)
    params = torch.zeros_like(multiplexer.block_parameters)
    params[0, 0, 1, 0, 0] = 0.1
    params[1, 0, 1, 0, 0] = -0.2
    with torch.no_grad():
        multiplexer.block_parameters.copy_(params)

    transformed = multiplexer(moved_state)
    branch_blocks = pauli_coefficients_to_unitary(params[:, 0, 1, 0]).to(dtype=torch.complex128)

    assert torch.allclose(
        transformed[0, 1, 1, 0, 0],
        branch_blocks[0] @ moved_state[0, 1, 1, 0, 0],
    )
    assert torch.allclose(
        transformed[0, 1, 3, 0, 0],
        branch_blocks[1] @ moved_state[0, 1, 3, 0, 0],
    )


def test_mode_multiplexer_supports_backpropagation_to_block_parameters() -> None:
    layout = RegisterLayout2D(image_size=2, feature_qubits=2, x_condition_qubits=1, y_condition_qubits=1)
    multiplexer = make_multiplexer(
        layout,
        x_condition_qubits_to_use=1,
        y_condition_qubits_to_use=1,
        dtype=torch.float64,
    )
    with torch.no_grad():
        multiplexer.block_parameters.normal_()
    state = torch.randn(layout.state_shape(batch_size=2), dtype=torch.complex128)

    transformed = multiplexer(state)
    loss = transformed.real.sum() + transformed.imag.sum()
    loss.backward()

    assert multiplexer.block_parameters.grad is not None
    assert multiplexer.block_parameters.grad.shape == multiplexer.block_parameters.shape
    assert torch.count_nonzero(multiplexer.block_parameters.grad.abs()) > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x_condition_qubits_to_use": -1}, "x_condition_qubits_to_use must be non-negative"),
        ({"y_condition_qubits_to_use": -1}, "y_condition_qubits_to_use must be non-negative"),
        ({"multiplexer_init_scale": 0.0}, "multiplexer_init_scale must be positive"),
        ({"multiplexer_init_scale": -0.1}, "multiplexer_init_scale must be positive"),
        (
            {"x_condition_qubits_to_use": 2},
            "x_condition_qubits_to_use cannot exceed layout.x_condition_qubits",
        ),
        (
            {"y_condition_qubits_to_use": 2},
            "y_condition_qubits_to_use cannot exceed layout.y_condition_qubits",
        ),
    ],
)
def test_mode_multiplexer_rejects_invalid_condition_selection_counts(
    kwargs: dict[str, int],
    message: str,
) -> None:
    layout = RegisterLayout2D(image_size=2, feature_qubits=1, x_condition_qubits=1, y_condition_qubits=1)

    with pytest.raises(ValueError, match=message):
        make_multiplexer(layout, **kwargs)


def test_mode_multiplexer_rejects_non_complex_inputs() -> None:
    layout = RegisterLayout2D(image_size=2, feature_qubits=1)
    multiplexer = make_multiplexer(layout)

    with pytest.raises(ValueError, match="complex"):
        multiplexer(torch.zeros(layout.state_shape(batch_size=1), dtype=torch.float32))


def test_mode_multiplexer_rejects_shape_mismatch() -> None:
    layout = RegisterLayout2D(image_size=2, feature_qubits=1, x_condition_qubits=1)
    multiplexer = make_multiplexer(layout, x_condition_qubits_to_use=1)

    with pytest.raises(ValueError, match="does not match the layout contract"):
        multiplexer(torch.zeros((1, 2, 1, 2, 1, 2), dtype=torch.complex64))


def test_marginal_measurement_returns_real_bxyf_tensor() -> None:
    measurement = make_measurement()
    state = torch.ones((2, 1, 3, 1, 2, 4), dtype=torch.complex128)

    measured = measurement(state)

    assert measured.shape == (2, 1, 1, 4)
    assert measured.dtype == torch.float64
    assert torch.allclose(measured, torch.ones((2, 1, 1, 4), dtype=torch.float64))


def test_marginal_measurement_sums_over_condition_axes_only() -> None:
    measurement = make_measurement()
    state = torch.zeros((1, 2, 2, 2, 2, 2), dtype=torch.complex64)
    state[0, 0, 0, 0, 0, 0] = 1.0 + 0.0j
    state[0, 0, 1, 0, 0, 0] = 0.0 + 2.0j
    state[0, 0, 1, 0, 1, 1] = 3.0 + 4.0j
    state[0, 1, 0, 1, 1, 1] = 0.0 - 1.0j

    measured = measurement(state)
    expected = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    expected[0, 0, 0, 0] = 5.0 / 16.0
    expected[0, 0, 0, 1] = 25.0 / 16.0
    expected[0, 1, 1, 1] = 1.0 / 16.0

    assert torch.allclose(measured, expected)


def test_marginal_measurement_normalizes_encoder_style_states_to_unit_mass() -> None:
    encoder = make_encoder(feature_qubits=1)
    measurement = make_measurement()
    images = torch.randn(3, 8, 8)

    measured = measurement(encoder(images))

    assert measured.shape == (3, 8, 8, 2)
    assert torch.allclose(measured.sum(dim=(1, 2, 3)), torch.ones(3))


def test_marginal_measurement_preserves_active_indices_and_feature_axes() -> None:
    measurement = make_measurement()
    weights = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=torch.float32,
    )
    state = (
        weights.sqrt()
        .unsqueeze(0)
        .unsqueeze(2)
        .unsqueeze(4)
        .expand(1, 2, 2, 2, 2, 3)
        .to(torch.complex64)
    )

    measured = measurement(state)

    assert measured.shape == (1, 2, 2, 3)
    assert torch.allclose(measured, weights.unsqueeze(0) / 4.0)


def test_marginal_measurement_supports_backpropagation() -> None:
    measurement = make_measurement()
    state = torch.randn(2, 4, 1, 4, 1, 2, dtype=torch.complex64, requires_grad=True)

    measured = measurement(state)
    loss = measured[..., 0].sum()
    loss.backward()

    assert state.grad is not None
    assert state.grad.shape == state.shape
    assert torch.count_nonzero(state.grad.abs()) > 0


def test_marginal_measurement_rejects_non_rank6_inputs() -> None:
    measurement = make_measurement()

    with pytest.raises(ValueError, match=r"\[B, X, CX, Y, CY, F\]"):
        measurement(torch.zeros(2, 4, 4, dtype=torch.complex64))


def test_marginal_measurement_rejects_non_complex_inputs() -> None:
    measurement = make_measurement()

    with pytest.raises(ValueError, match="complex"):
        measurement(torch.zeros(1, 2, 1, 2, 1, 2, dtype=torch.float32))


@pytest.mark.parametrize("shot_budget", [None, 0])
def test_finite_shot_histogram_passes_through_infinite_shot_modes(shot_budget: int | None) -> None:
    histogram = make_histogram(shot_budget=shot_budget)
    histogram.eval()
    probabilities = torch.tensor(
        [[[[0.2, 0.3], [0.1, 0.4]]]],
        dtype=torch.float64,
    )

    measured = histogram(probabilities)

    assert measured.shape == probabilities.shape
    assert measured.dtype == probabilities.dtype
    assert measured.device == probabilities.device
    assert torch.allclose(measured, probabilities)


def test_finite_shot_histogram_passes_through_while_training_and_preserves_gradients() -> None:
    histogram = make_histogram(shot_budget=8)
    histogram.train()
    probabilities = torch.tensor(
        [[[[0.2, 0.3], [0.1, 0.4]]]],
        dtype=torch.float32,
        requires_grad=True,
    )

    measured = histogram(probabilities)
    measured.sum().backward()

    assert torch.allclose(measured, probabilities)
    assert probabilities.grad is not None
    assert torch.allclose(probabilities.grad, torch.ones_like(probabilities))


def test_finite_shot_histogram_pass_through_mode_is_vmap_compatible() -> None:
    histogram = make_histogram(shot_budget=None)
    histogram.eval()
    probabilities = torch.rand(3, 2, 2, 2, dtype=torch.float32)
    probabilities = probabilities / probabilities.sum(dim=(1, 2, 3), keepdim=True)

    measured = torch.func.vmap(
        lambda probability: histogram(probability.unsqueeze(0)).squeeze(0)
    )(probabilities)

    assert measured.shape == probabilities.shape
    assert torch.allclose(measured, probabilities)


@pytest.mark.parametrize("shot_budget", [-1, 1.5, True])
def test_finite_shot_histogram_rejects_invalid_shot_budgets(shot_budget: object) -> None:
    with pytest.raises(ValueError, match="shot_budget"):
        FiniteShotHistogram2D(shot_budget=shot_budget)  # type: ignore[arg-type]


def test_finite_shot_histogram_samples_normalized_frequencies_reproducibly() -> None:
    histogram = make_histogram(shot_budget=8)
    histogram.eval()
    probabilities = torch.tensor(
        [
            [[[0.10, 0.20], [0.30, 0.40]]],
            [[[0.55, 0.15], [0.20, 0.10]]],
        ],
        dtype=torch.float64,
    )

    torch.manual_seed(7)
    measured_a = histogram(probabilities)
    torch.manual_seed(7)
    measured_b = histogram(probabilities)
    torch.manual_seed(8)
    measured_c = histogram(probabilities)

    assert measured_a.shape == probabilities.shape
    assert measured_a.dtype == probabilities.dtype
    assert measured_a.device == probabilities.device
    assert torch.allclose(measured_a, measured_b)
    assert not torch.allclose(measured_a, measured_c)
    assert torch.allclose(measured_a.sum(dim=(1, 2, 3)), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(measured_a * 8.0, torch.round(measured_a * 8.0))


def test_finite_shot_histogram_sample_repeated_histograms_returns_expected_shape() -> None:
    histogram = make_histogram(shot_budget=8)
    histogram.eval()
    probabilities = torch.tensor(
        [
            [[[0.10, 0.20], [0.30, 0.40]]],
            [[[0.55, 0.15], [0.20, 0.10]]],
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(7)
    repeated = histogram.sample_repeated_histograms(probabilities, repetitions=5, block_size=2)

    assert repeated.shape == (5, 2, 1, 2, 2)
    assert repeated.dtype == probabilities.dtype
    assert torch.allclose(repeated.sum(dim=(2, 3, 4)), torch.ones((5, 2), dtype=torch.float32))
    assert torch.allclose(repeated * 8.0, torch.round(repeated * 8.0))


def test_finite_shot_histogram_sample_repeated_histograms_matches_chunked_sampling() -> None:
    histogram = make_histogram(shot_budget=8)
    histogram.eval()
    probabilities = torch.tensor(
        [
            [[[0.10, 0.20], [0.30, 0.40]]],
            [[[0.55, 0.15], [0.20, 0.10]]],
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(11)
    repeated_a = histogram.sample_repeated_histograms(probabilities, repetitions=5, block_size=5)
    torch.manual_seed(11)
    repeated_b = histogram.sample_repeated_histograms(probabilities, repetitions=5, block_size=2)

    assert torch.allclose(repeated_a, repeated_b)
