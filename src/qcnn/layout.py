"""Canonical tensor-layout utilities for Torch-based QCNN state tensors.

The authoritative state contract is ``[B, X, CX, Y, CY, F]``:

- ``B``: batch axis
- ``X``: active ``x``-register basis states
- ``CX``: explicit ``x``-condition register basis states
- ``Y``: active ``y``-register basis states
- ``CY``: explicit ``y``-condition register basis states
- ``F``: feature-register basis states

Bit order is MSB-left inside every logical register. Deferred-measurement
pooling preserves this outer axis order and updates only the register sizes.

If an active index qubit is moved into a condition register, the active
dimension halves and the condition dimension doubles:

- ``X -> X / 2`` and ``CX -> 2 * CX`` for the x-register
- ``Y -> Y / 2`` and ``CY -> 2 * CY`` for the y-register

Each newly moved bit becomes the most-significant bit of its condition
register, so the most recently measured outcomes appear first.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

CANONICAL_TENSOR_CONTRACT = (
    "[B, X, CX, Y, CY, F] with axis semantics "
    "(batch, x-register, x-condition, y-register, y-condition, features)."
)


def is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two.

    Args:
        value: Integer to test.

    Returns:
        ``True`` exactly when ``value > 0`` and ``value`` has a single set bit.

    Notes:
        Zero and negative integers return ``False``.
    """

    return value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class RegisterLayout2D:
    """Describe the canonical register layout for a 2D QCNN state tensor.

    Args:
        image_size: Square image side length. Must be a positive power of two.
        feature_qubits: Number of qubits in the feature register.
        x_condition_qubits: Number of x-index qubits already deferred into the
            explicit x-condition register.
        y_condition_qubits: Number of y-index qubits already deferred into the
            explicit y-condition register.

    Contract:
        States governed by this layout have shape ``[B, X, CX, Y, CY, F]`` with
        ``X = image_size / 2**x_condition_qubits``,
        ``CX = 2**x_condition_qubits``,
        ``Y = image_size / 2**y_condition_qubits``,
        ``CY = 2**y_condition_qubits``, and
        ``F = 2**feature_qubits``.

    Notes:
        The explicit condition axes allow pooling by pure reshape/reindexing
        without changing the outer tensor order.
    """

    image_size: int
    feature_qubits: int
    x_condition_qubits: int = 0
    y_condition_qubits: int = 0

    def __post_init__(self) -> None:
        if not is_power_of_two(self.image_size):
            raise ValueError(f"image_size must be a positive power of two, got {self.image_size}.")
        if self.feature_qubits < 1:
            raise ValueError(f"feature_qubits must be at least 1, got {self.feature_qubits}.")
        if self.x_condition_qubits < 0 or self.y_condition_qubits < 0:
            raise ValueError("Condition-register qubit counts must be non-negative.")
        if self.x_condition_qubits > self.index_qubits_per_axis:
            raise ValueError("x_condition_qubits cannot exceed the available x-register qubits.")
        if self.y_condition_qubits > self.index_qubits_per_axis:
            raise ValueError("y_condition_qubits cannot exceed the available y-register qubits.")

    @property
    def index_qubits_per_axis(self) -> int:
        """Return the total number of index qubits on one image axis."""

        return int(math.log2(self.image_size))

    @property
    def x_active_dim(self) -> int:
        """Return ``image_size / 2**x_condition_qubits``."""

        return self.image_size // (2 ** self.x_condition_qubits)

    @property
    def y_active_dim(self) -> int:
        """Return ``image_size / 2**y_condition_qubits``."""

        return self.image_size // (2 ** self.y_condition_qubits)

    @property
    def x_condition_dim(self) -> int:
        """Return ``2**x_condition_qubits``."""

        return 2 ** self.x_condition_qubits

    @property
    def y_condition_dim(self) -> int:
        """Return ``2**y_condition_qubits``."""

        return 2 ** self.y_condition_qubits

    @property
    def feature_dim(self) -> int:
        """Return ``2**feature_qubits``."""

        return 2 ** self.feature_qubits

    def state_shape(self, batch_size: int) -> tuple[int, int, int, int, int, int]:
        """Return the canonical tensor shape for a given batch size.

        Args:
            batch_size: Number of samples on the leading axis.

        Returns:
            The tuple ``(B, X, CX, Y, CY, F)`` implied by the layout.

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """

        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        return (
            batch_size,
            self.x_active_dim,
            self.x_condition_dim,
            self.y_active_dim,
            self.y_condition_dim,
            self.feature_dim,
        )

    def validate_state_shape(self, state: torch.Tensor) -> None:
        """Validate that ``state`` matches this layout exactly.

        Args:
            state: Candidate tensor expected to satisfy the canonical contract
                ``[B, X, CX, Y, CY, F]``.

        Raises:
            ValueError: If the tensor rank is not 6 or if any axis size differs
                from the layout-implied shape for the given batch size.
        """

        if state.ndim != 6:
            raise ValueError(
                "State tensors must have rank 6 and canonical layout [B, X, CX, Y, CY, F], "
                f"got shape {tuple(state.shape)}."
            )

        expected = self.state_shape(batch_size=int(state.shape[0]))
        if tuple(state.shape) != expected:
            raise ValueError(
                "State tensor shape does not match the layout contract: "
                f"expected {expected}, got {tuple(state.shape)}."
            )


def _move_one_active_x_qubit_to_condition(
    state: torch.Tensor,
    layout: RegisterLayout2D,
) -> tuple[torch.Tensor, RegisterLayout2D]:
    """Move one active x-index qubit into the explicit x-condition register.

    Contract:
        The input state must satisfy ``layout``. The returned state keeps axis
        order ``[B, X, CX, Y, CY, F]`` but changes dimensions as
        ``X -> X / 2`` and ``CX -> 2 * CX``.

    Formula:
        This is an amplitude-preserving reshape, not a numerical transform:
        ``state[B, X, CX, Y, CY, F] -> state[B, X/2, 2*CX, Y, CY, F]``.

    Notes:
        The new condition bit is inserted as the most-significant x-condition
        bit, so the most recently pooled x outcome is selected first.
    """

    batch, x_dim, cx_dim, y_dim, cy_dim, feature_dim = state.shape
    moved = state.reshape(batch, x_dim // 2, cx_dim * 2, y_dim, cy_dim, feature_dim)
    new_layout = RegisterLayout2D(
        image_size=layout.image_size,
        feature_qubits=layout.feature_qubits,
        x_condition_qubits=layout.x_condition_qubits + 1,
        y_condition_qubits=layout.y_condition_qubits,
    )
    return moved, new_layout


def _move_one_active_y_qubit_to_condition(
    state: torch.Tensor,
    layout: RegisterLayout2D,
) -> tuple[torch.Tensor, RegisterLayout2D]:
    """Move one active y-index qubit into the explicit y-condition register.

    Contract:
        The input state must satisfy ``layout``. The returned state keeps axis
        order ``[B, X, CX, Y, CY, F]`` but changes dimensions as
        ``Y -> Y / 2`` and ``CY -> 2 * CY``.

    Formula:
        This is an amplitude-preserving reshape:
        ``state[B, X, CX, Y, CY, F] -> state[B, X, CX, Y/2, 2*CY, F]``.

    Notes:
        The new condition bit becomes the most-significant y-condition bit.
    """

    batch, x_dim, cx_dim, y_dim, cy_dim, feature_dim = state.shape
    moved = state.reshape(batch, x_dim, cx_dim, y_dim // 2, cy_dim * 2, feature_dim)
    new_layout = RegisterLayout2D(
        image_size=layout.image_size,
        feature_qubits=layout.feature_qubits,
        x_condition_qubits=layout.x_condition_qubits,
        y_condition_qubits=layout.y_condition_qubits + 1,
    )
    return moved, new_layout


def move_active_qubit_to_condition(
    state: torch.Tensor,
    layout: RegisterLayout2D,
    *,
    x_qubits_to_condition: int = 0,
    y_qubits_to_condition: int = 0,
) -> tuple[torch.Tensor, RegisterLayout2D]:
    """Move least-significant active index qubits into the condition registers.

    Args:
        state: Complex or real tensor with canonical axis order
            ``[B, X, CX, Y, CY, F]``.
        layout: Register layout describing ``state``.
        x_qubits_to_condition: Number of active x qubits to move.
        y_qubits_to_condition: Number of active y qubits to move.

    Returns:
        A pair ``(moved_state, moved_layout)`` where the outer axis order stays
        ``[B, X, CX, Y, CY, F]`` and only the active/condition dimensions are
        updated.

    Raises:
        ValueError: If the requested move counts are negative, exceed the number
            of remaining active qubits, or if ``state`` does not match
            ``layout``.

    Notes:
        The operation preserves amplitudes exactly. It is implemented by a
        sequence of reshapes and does not mix batch, feature, or unaffected
        register axes.
    """

    layout.validate_state_shape(state)

    if x_qubits_to_condition < 0:
        raise ValueError(
            f"x_qubits_to_condition must be non-negative, got {x_qubits_to_condition}."
        )
    if y_qubits_to_condition < 0:
        raise ValueError(
            f"y_qubits_to_condition must be non-negative, got {y_qubits_to_condition}."
        )

    available_x_qubits = layout.index_qubits_per_axis - layout.x_condition_qubits
    available_y_qubits = layout.index_qubits_per_axis - layout.y_condition_qubits
    if x_qubits_to_condition > available_x_qubits:
        raise ValueError(
            "Cannot move "
            f"{x_qubits_to_condition} x-register qubits to the condition register; "
            f"only {available_x_qubits} active x-register qubits remain."
        )
    if y_qubits_to_condition > available_y_qubits:
        raise ValueError(
            "Cannot move "
            f"{y_qubits_to_condition} y-register qubits to the condition register; "
            f"only {available_y_qubits} active y-register qubits remain."
        )

    if x_qubits_to_condition == 0 and y_qubits_to_condition == 0:
        return state, layout

    moved_state = state
    moved_layout = layout
    for _ in range(x_qubits_to_condition):
        moved_state, moved_layout = _move_one_active_x_qubit_to_condition(moved_state, moved_layout)
    for _ in range(y_qubits_to_condition):
        moved_state, moved_layout = _move_one_active_y_qubit_to_condition(moved_state, moved_layout)
    return moved_state, moved_layout
