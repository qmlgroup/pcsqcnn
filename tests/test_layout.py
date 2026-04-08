import pytest
import torch

from qcnn import RegisterLayout2D, move_active_qubit_to_condition


def test_register_layout_reports_expected_dimensions() -> None:
    layout = RegisterLayout2D(image_size=8, feature_qubits=3)

    assert layout.index_qubits_per_axis == 3
    assert layout.feature_dim == 8
    assert layout.x_condition_dim == 1
    assert layout.y_condition_dim == 1
    assert layout.state_shape(batch_size=4) == (4, 8, 1, 8, 1, 8)


def test_move_active_x_qubit_to_condition_preserves_msb_left_order() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=1)
    state = torch.arange(1 * 4 * 1 * 4 * 1 * 2, dtype=torch.float32).reshape(1, 4, 1, 4, 1, 2)
    state = state.to(torch.complex64)

    moved_state, moved_layout = move_active_qubit_to_condition(
        state,
        layout,
        x_qubits_to_condition=1,
    )

    assert moved_layout.state_shape(batch_size=1) == (1, 2, 2, 4, 1, 2)
    assert moved_state[0, 1, 1, 3, 0, 1] == state[0, 3, 0, 3, 0, 1]
    assert torch.allclose(moved_state.abs().sum(), state.abs().sum())


def test_move_active_y_qubit_to_condition_preserves_msb_left_order() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=1)
    state = torch.arange(1 * 4 * 1 * 4 * 1 * 2, dtype=torch.float32).reshape(1, 4, 1, 4, 1, 2)
    state = state.to(torch.complex64)

    moved_state, moved_layout = move_active_qubit_to_condition(
        state,
        layout,
        y_qubits_to_condition=1,
    )

    assert moved_layout.state_shape(batch_size=1) == (1, 4, 1, 2, 2, 2)
    assert moved_state[0, 2, 0, 1, 1, 0] == state[0, 2, 0, 3, 0, 0]
    assert torch.allclose(moved_state.abs().sum(), state.abs().sum())


def test_move_multiple_x_qubits_matches_repeated_single_qubit_moves() -> None:
    layout = RegisterLayout2D(image_size=8, feature_qubits=1)
    state = torch.arange(1 * 8 * 1 * 8 * 1 * 2, dtype=torch.float32).reshape(1, 8, 1, 8, 1, 2)
    state = state.to(torch.complex64)

    moved_once, layout_once = move_active_qubit_to_condition(
        state,
        layout,
        x_qubits_to_condition=1,
    )
    moved_twice, layout_twice = move_active_qubit_to_condition(
        moved_once,
        layout_once,
        x_qubits_to_condition=1,
    )
    moved_bulk, layout_bulk = move_active_qubit_to_condition(
        state,
        layout,
        x_qubits_to_condition=2,
    )

    assert layout_bulk == layout_twice
    assert torch.equal(moved_bulk, moved_twice)


def test_move_newer_x_condition_bit_becomes_msb() -> None:
    layout = RegisterLayout2D(image_size=8, feature_qubits=1)
    state = torch.arange(1 * 8 * 1 * 8 * 1 * 2, dtype=torch.float32).reshape(1, 8, 1, 8, 1, 2)
    state = state.to(torch.complex64)

    moved_state, moved_layout = move_active_qubit_to_condition(state, layout, x_qubits_to_condition=2)

    assert moved_layout.state_shape(batch_size=1) == (1, 2, 4, 8, 1, 2)
    assert moved_state[0, 1, 1, 3, 0, 0] == state[0, 5, 0, 3, 0, 0]
    assert moved_state[0, 1, 2, 3, 0, 0] == state[0, 6, 0, 3, 0, 0]


def test_move_newer_y_condition_bit_becomes_msb() -> None:
    layout = RegisterLayout2D(image_size=8, feature_qubits=1)
    state = torch.arange(1 * 8 * 1 * 8 * 1 * 2, dtype=torch.float32).reshape(1, 8, 1, 8, 1, 2)
    state = state.to(torch.complex64)

    moved_state, moved_layout = move_active_qubit_to_condition(state, layout, y_qubits_to_condition=2)

    assert moved_layout.state_shape(batch_size=1) == (1, 8, 1, 2, 4, 2)
    assert moved_state[0, 3, 0, 1, 1, 0] == state[0, 3, 0, 5, 0, 0]
    assert moved_state[0, 3, 0, 1, 2, 0] == state[0, 3, 0, 6, 0, 0]


def test_move_x_then_y_matches_sequential_calls() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=1)
    state = torch.arange(1 * 4 * 1 * 4 * 1 * 2, dtype=torch.float32).reshape(1, 4, 1, 4, 1, 2)
    state = state.to(torch.complex64)

    moved_seq, layout_seq = move_active_qubit_to_condition(state, layout, x_qubits_to_condition=1)
    moved_seq, layout_seq = move_active_qubit_to_condition(
        moved_seq,
        layout_seq,
        y_qubits_to_condition=1,
    )
    moved_bulk, layout_bulk = move_active_qubit_to_condition(
        state,
        layout,
        x_qubits_to_condition=1,
        y_qubits_to_condition=1,
    )

    assert layout_bulk == layout_seq
    assert torch.equal(moved_bulk, moved_seq)


def test_move_active_qubit_to_condition_noop_returns_original_objects() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=1)
    state = torch.arange(1 * 4 * 1 * 4 * 1 * 2, dtype=torch.float32).reshape(1, 4, 1, 4, 1, 2)
    state = state.to(torch.complex64)

    moved_state, moved_layout = move_active_qubit_to_condition(state, layout)

    assert moved_state is state
    assert moved_layout is layout


def test_move_active_qubit_to_condition_rejects_negative_move_counts() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=1)
    state = torch.zeros((1, 4, 1, 4, 1, 2), dtype=torch.complex64)

    with pytest.raises(ValueError, match="x_qubits_to_condition must be non-negative"):
        move_active_qubit_to_condition(state, layout, x_qubits_to_condition=-1)

    with pytest.raises(ValueError, match="y_qubits_to_condition must be non-negative"):
        move_active_qubit_to_condition(state, layout, y_qubits_to_condition=-1)


def test_move_active_qubit_to_condition_rejects_requests_exceeding_active_qubits() -> None:
    layout = RegisterLayout2D(image_size=4, feature_qubits=1, x_condition_qubits=2)
    state = torch.zeros(layout.state_shape(batch_size=1), dtype=torch.complex64)

    with pytest.raises(ValueError, match="only 0 active x-register qubits remain"):
        move_active_qubit_to_condition(state, layout, x_qubits_to_condition=1)

    layout = RegisterLayout2D(image_size=4, feature_qubits=1, y_condition_qubits=1)
    state = torch.zeros(layout.state_shape(batch_size=1), dtype=torch.complex64)

    with pytest.raises(ValueError, match="only 1 active y-register qubits remain"):
        move_active_qubit_to_condition(state, layout, y_qubits_to_condition=2)
