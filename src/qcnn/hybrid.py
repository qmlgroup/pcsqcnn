"""Hybrid PCS-QCNN models built on the canonical Torch state contract.

`PCSQCNN` implements the article-style layer stack

`encoder -> [QFT -> multiplexer -> IQFT -> pooling]^(Q-1) -> QFT -> multiplexer
-> IQFT -> marginal measurement -> finite-shot histogram -> linear classifier`.

`PCSQCNNNoQFT` implements the low-data ablation used for Figure 2(b):

`encoder -> [R -> multiplexer -> R* -> pooling]^(Q-1) -> R -> multiplexer
-> R* -> marginal measurement -> finite-shot histogram -> linear classifier`.

Pooling is handled through deferred measurement by moving one active qubit per
index register into the corresponding condition register after each non-final
layer. The Fourier model can optionally collapse adjacent `IQFT -> pooling ->
QFT` segments into the reduced Fourier junction helper from `qcnn.quantum`.

The models always return logits. Softmax and the training loss remain outside
the module.
"""

from __future__ import annotations

import math
import torch
from torch import nn

from qcnn.layout import RegisterLayout2D, move_active_qubit_to_condition
from qcnn.quantum import (
    FiniteShotHistogram2D,
    FrqiEncoder2D,
    MarginalMeasurement2D,
    ModeMultiplexer2D,
    _apply_fourier_junction_2d,
    _apply_iqft_2d,
    _apply_qft_2d,
    _apply_unitary_to_active_axis_1d,
    pauli_coefficients_to_unitary,
)


def _build_layer_layouts(
    *,
    image_size: int,
    feature_qubits: int,
    quantum_layers: int,
) -> list[RegisterLayout2D]:
    initial_layout = RegisterLayout2D(image_size=image_size, feature_qubits=feature_qubits)
    if quantum_layers < 1:
        raise ValueError(f"quantum_layers must be positive, got {quantum_layers}.")
    if quantum_layers > initial_layout.index_qubits_per_axis:
        raise ValueError(
            "quantum_layers cannot exceed the number of index qubits per axis: "
            f"got {quantum_layers} > {initial_layout.index_qubits_per_axis}."
        )

    return [
        RegisterLayout2D(
            image_size=image_size,
            feature_qubits=feature_qubits,
            x_condition_qubits=layer_idx,
            y_condition_qubits=layer_idx,
        )
        for layer_idx in range(quantum_layers)
    ]


def _make_multiplexers(
    layer_layouts: list[RegisterLayout2D],
    *,
    multiplexer_init_scale: float,
    dtype: torch.dtype,
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            ModeMultiplexer2D(
                layout,
                x_condition_qubits_to_use=0 if layer_idx == 0 else 1,
                y_condition_qubits_to_use=0 if layer_idx == 0 else 1,
                multiplexer_init_scale=multiplexer_init_scale,
                dtype=dtype,
            )
            for layer_idx, layout in enumerate(layer_layouts)
        ]
    )


def _apply_shared_spatial_unitary_2d(state: torch.Tensor, unitary: torch.Tensor) -> torch.Tensor:
    state = _apply_unitary_to_active_axis_1d(state, unitary, active_axis=1)
    return _apply_unitary_to_active_axis_1d(state, unitary, active_axis=3)


def _validate_image_spatial_shape(images: torch.Tensor, *, image_size: int, model_name: str) -> None:
    if images.ndim == 3 and tuple(images.shape[1:]) != (image_size, image_size):
        raise ValueError(
            f"{model_name} expects image tensors with spatial shape "
            f"({image_size}, {image_size}), got {tuple(images.shape[1:])}."
        )


def _flatten_readout_histogram(
    readout_histogram: torch.Tensor,
    *,
    expected_spatial_shape: tuple[int, int, int],
    model_name: str,
) -> torch.Tensor:
    if readout_histogram.ndim < 4:
        raise ValueError(
            f"{model_name} expects readout histograms with shape [..., X, Y, F], "
            f"got {tuple(readout_histogram.shape)}."
        )
    if tuple(readout_histogram.shape[-3:]) != expected_spatial_shape:
        raise ValueError(
            f"{model_name} expects readout histograms ending with {expected_spatial_shape}, "
            f"got trailing shape {tuple(readout_histogram.shape[-3:])}."
        )
    return readout_histogram.reshape(*readout_histogram.shape[:-3], -1)


def _reduce_readout_to_feature_distribution(
    readout_histogram: torch.Tensor,
    *,
    model_name: str,
) -> torch.Tensor:
    if readout_histogram.ndim < 4:
        raise ValueError(
            f"{model_name} expects readout histograms with shape [..., X, Y, F], "
            f"got {tuple(readout_histogram.shape)}."
        )
    return readout_histogram.sum(dim=(-3, -2), keepdim=True)


class PCSQCNN(nn.Module):
    """A multilayer PCS-QCNN with Fourier-domain multiplexers and logits output.

    Args:
        image_size: Square image side length. Must be a power of two because the
            model pools one index qubit per layer.
        num_classes: Number of logits produced by the final linear head.
        feature_qubits: Number of feature-register qubits.
        quantum_layers: Number of PCS layers. This cannot exceed the number of
            index qubits per spatial axis.
        brightness_range: Encoder-side angle interval ``(a, b)`` used to map
            normalized grayscale inputs into FRQI arguments before the quantum
            stack.
        shot_budget: Optional finite number of measurement shots used to sample
            readout histograms during evaluation. ``None`` keeps the current
            exact-probability readout; ``0`` is accepted as an alias for
            ``None``.
        reduce_readout_to_feature_distribution: Whether to marginalize the
            measured readout over the active spatial axes and keep only the
            feature-register distribution with trailing shape ``[1, 1, F]``.
            This Torch-only mode is outside the PennyLane/article reference
            contract, which supports full readout only.
        use_reduced_fourier_junction: Whether to replace each explicit
            ``IQFT -> move_active_qubit_to_condition -> QFT`` triple with the
            analytically equivalent reduced junction helper.
        multiplexer_init_scale: Upper endpoint of the uniform
            Pauli-coefficient initialization interval used by every trainable
            multiplexer block. Each coefficient is sampled from
            ``Uniform(0, multiplexer_init_scale)``.
        dtype: Real dtype used by the encoder and multiplexer parameters.

    Formula:
        For ``Q = quantum_layers``, the quantum stack is
        ``[QFT -> MUX -> IQFT -> pooling]^(Q-1) -> QFT -> MUX -> IQFT``.
        The first multiplexer is unconditional. Every later multiplexer is
        conditioned on one most-recently pooled bit from each index register.

    Returns:
        ``forward(images)`` returns logits of shape ``[B, num_classes]``.

    Notes:
        After the final IQFT the model does not pool again. Instead it performs
        ``MarginalMeasurement2D`` over the remaining active state, optionally
        samples a finite-shot histogram during evaluation, flattens the
        resulting ``[B, X_active, Y_active, F]`` tensor, and applies a linear
        classifier. The reduced Fourier junction is enabled by default because
        it is exactly equivalent to the explicit path and avoids an unnecessary
        intermediate transform pair between adjacent layers. The hybrid model
        owns the encoder-side ``brightness_range`` and ``shot_budget``
        configuration and forwards them to its internal readout modules.
    """

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        feature_qubits: int = 1,
        quantum_layers: int = 1,
        *,
        brightness_range: tuple[float, float] = (0.0, math.pi),
        shot_budget: int | None = None,
        reduce_readout_to_feature_distribution: bool = False,
        use_reduced_fourier_junction: bool = True,
        multiplexer_init_scale: float = 2.0 * math.pi,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if feature_qubits < 1:
            raise ValueError(f"feature_qubits must be at least 1, got {feature_qubits}.")

        self.layer_layouts = _build_layer_layouts(
            image_size=image_size,
            feature_qubits=feature_qubits,
            quantum_layers=quantum_layers,
        )
        final_layout = self.layer_layouts[-1]
        self.image_size = image_size
        self.num_classes = num_classes
        self.feature_qubits = feature_qubits
        self.quantum_layers = quantum_layers
        self.reduce_readout_to_feature_distribution = bool(reduce_readout_to_feature_distribution)
        self.use_reduced_fourier_junction = use_reduced_fourier_junction
        self.multiplexer_init_scale = float(multiplexer_init_scale)
        self.encoder = FrqiEncoder2D(
            feature_qubits=feature_qubits,
            brightness_range=brightness_range,
            dtype=dtype,
        )
        self.brightness_range = self.encoder.brightness_range
        self.measurement = MarginalMeasurement2D()
        self.readout_histogram = FiniteShotHistogram2D(shot_budget=shot_budget)
        self.shot_budget = self.readout_histogram.shot_budget
        self.multiplexers = _make_multiplexers(
            self.layer_layouts,
            multiplexer_init_scale=self.multiplexer_init_scale,
            dtype=dtype,
        )
        self.readout_spatial_shape = (
            (1, 1, final_layout.feature_dim)
            if self.reduce_readout_to_feature_distribution
            else (final_layout.x_active_dim, final_layout.y_active_dim, final_layout.feature_dim)
        )
        self.classifier = nn.Linear(
            self.readout_spatial_shape[0] * self.readout_spatial_shape[1] * self.readout_spatial_shape[2],
            num_classes,
        )

    def exact_quantum_readout_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        """Return exact measured readout probabilities before finite-shot sampling."""

        _validate_image_spatial_shape(images, image_size=self.image_size, model_name="PCSQCNN")

        state = self.encoder(images)
        current_layout = self.layer_layouts[0]

        if self.use_reduced_fourier_junction:
            state = _apply_qft_2d(state)

            for layer_idx, multiplexer in enumerate(self.multiplexers):
                state = multiplexer(state)

                if layer_idx < self.quantum_layers - 1:
                    state, current_layout = _apply_fourier_junction_2d(state, current_layout)
                else:
                    state = _apply_iqft_2d(state)
        else:
            for layer_idx, multiplexer in enumerate(self.multiplexers):
                state = _apply_qft_2d(state)
                state = multiplexer(state)
                state = _apply_iqft_2d(state)

                if layer_idx < self.quantum_layers - 1:
                    state, current_layout = move_active_qubit_to_condition(
                        state,
                        current_layout,
                        x_qubits_to_condition=1,
                        y_qubits_to_condition=1,
                    )

        measured = self.measurement(state)
        if self.reduce_readout_to_feature_distribution:
            measured = _reduce_readout_to_feature_distribution(
                measured,
                model_name="PCSQCNN",
            )
        return measured

    def classify_readout_histogram(self, readout_histogram: torch.Tensor) -> torch.Tensor:
        """Map measured readout histograms into logits through the linear head."""

        flattened = _flatten_readout_histogram(
            readout_histogram,
            expected_spatial_shape=self.readout_spatial_shape,
            model_name="PCSQCNN",
        )
        return self.classifier(flattened)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the full PCS-QCNN pipeline on a batch of images.

        Args:
            images: Real tensor with shape ``[B, image_size, image_size]``
                containing normalized grayscale values.

        Returns:
            Logits with shape ``[B, num_classes]``.

        Raises:
            ValueError: If the input spatial shape does not match the configured
                ``image_size``.
        """
        measured = self.exact_quantum_readout_probabilities(images)
        measured = self.readout_histogram(measured)
        return self.classify_readout_histogram(measured)


class PCSQCNNNoQFT(nn.Module):
    """A PCS-QCNN ablation that replaces QFT/IQFT with fixed random unitaries.

    Args:
        image_size: Square image side length. Must be a power of two because the
            model pools one index qubit per layer.
        num_classes: Number of logits produced by the final linear head.
        feature_qubits: Number of feature-register qubits.
        quantum_layers: Number of quantum layers. This cannot exceed the number
            of index qubits per spatial axis.
        brightness_range: Encoder-side angle interval ``(a, b)`` used to map
            normalized grayscale inputs into FRQI arguments before the quantum
            stack.
        shot_budget: Optional finite number of measurement shots used to sample
            readout histograms during evaluation. ``None`` keeps the current
            exact-probability readout; ``0`` is accepted as an alias for
            ``None``.
        reduce_readout_to_feature_distribution: Whether to marginalize the
            measured readout over the active spatial axes and keep only the
            feature-register distribution with trailing shape ``[1, 1, F]``.
            This Torch-only mode is outside the PennyLane/article reference
            contract, which supports full readout only.
        multiplexer_init_scale: Upper endpoint of the uniform
            Pauli-coefficient initialization interval used by every trainable
            multiplexer block. Each coefficient is sampled from
            ``Uniform(0, multiplexer_init_scale)``.
        dtype: Real dtype used by the encoder, multiplexer parameters, and the
            fixed spatial-unitary initialization.

    Formula:
        For ``Q = quantum_layers``, the quantum stack is
        ``[R -> MUX -> R* -> pooling]^(Q-1) -> R -> MUX -> R*`` where each
        layer samples one Gaussian vector in the Pauli basis, builds a unitary
        ``R`` with ``pauli_coefficients_to_unitary(...)``, and keeps that
        matrix fixed for the lifetime of the module.

    Returns:
        ``forward(images)`` returns logits of shape ``[B, num_classes]``.

    Notes:
        The same fixed layer-specific unitary ``R`` is applied to both active
        spatial axes. ``R*`` is implemented as the conjugate transpose of that
        same layer-specific unitary. Both are stored as registered buffers, so
        they move with the module and appear in ``state_dict()`` but are not
        optimized. Pooling uses the explicit
        ``move_active_qubit_to_condition(...)`` path between layers.
    """

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        feature_qubits: int = 1,
        quantum_layers: int = 1,
        *,
        brightness_range: tuple[float, float] = (0.0, math.pi),
        shot_budget: int | None = None,
        reduce_readout_to_feature_distribution: bool = False,
        multiplexer_init_scale: float = 2.0 * math.pi,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if feature_qubits < 1:
            raise ValueError(f"feature_qubits must be at least 1, got {feature_qubits}.")

        self.layer_layouts = _build_layer_layouts(
            image_size=image_size,
            feature_qubits=feature_qubits,
            quantum_layers=quantum_layers,
        )
        final_layout = self.layer_layouts[-1]
        self.image_size = image_size
        self.num_classes = num_classes
        self.feature_qubits = feature_qubits
        self.quantum_layers = quantum_layers
        self.reduce_readout_to_feature_distribution = bool(reduce_readout_to_feature_distribution)
        self.multiplexer_init_scale = float(multiplexer_init_scale)
        self.encoder = FrqiEncoder2D(
            feature_qubits=feature_qubits,
            brightness_range=brightness_range,
            dtype=dtype,
        )
        self.brightness_range = self.encoder.brightness_range
        self.measurement = MarginalMeasurement2D()
        self.readout_histogram = FiniteShotHistogram2D(shot_budget=shot_budget)
        self.shot_budget = self.readout_histogram.shot_budget
        self.multiplexers = _make_multiplexers(
            self.layer_layouts,
            multiplexer_init_scale=self.multiplexer_init_scale,
            dtype=dtype,
        )
        self.readout_spatial_shape = (
            (1, 1, final_layout.feature_dim)
            if self.reduce_readout_to_feature_distribution
            else (final_layout.x_active_dim, final_layout.y_active_dim, final_layout.feature_dim)
        )
        self.classifier = nn.Linear(
            self.readout_spatial_shape[0] * self.readout_spatial_shape[1] * self.readout_spatial_shape[2],
            num_classes,
        )

        r_unitary_buffer_names: list[str] = []
        r_adjoint_buffer_names: list[str] = []
        for layer_idx, layout in enumerate(self.layer_layouts):
            parameter_count = layout.x_active_dim * layout.x_active_dim
            coefficients = torch.randn(parameter_count, dtype=dtype)
            unitary = pauli_coefficients_to_unitary(coefficients).to(dtype=self.encoder.complex_dtype)
            adjoint = unitary.conj().transpose(-1, -2)

            unitary_name = f"r_unitary_layer_{layer_idx}"
            adjoint_name = f"r_adjoint_layer_{layer_idx}"
            self.register_buffer(unitary_name, unitary)
            self.register_buffer(adjoint_name, adjoint)
            r_unitary_buffer_names.append(unitary_name)
            r_adjoint_buffer_names.append(adjoint_name)
        self.r_unitary_buffer_names = tuple(r_unitary_buffer_names)
        self.r_adjoint_buffer_names = tuple(r_adjoint_buffer_names)

    def _layer_unitary_pair(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            getattr(self, self.r_unitary_buffer_names[layer_idx]),
            getattr(self, self.r_adjoint_buffer_names[layer_idx]),
        )

    def exact_quantum_readout_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        """Return exact measured readout probabilities before finite-shot sampling."""

        _validate_image_spatial_shape(images, image_size=self.image_size, model_name="PCSQCNNNoQFT")

        state = self.encoder(images)
        current_layout = self.layer_layouts[0]

        for layer_idx, multiplexer in enumerate(self.multiplexers):
            unitary, adjoint = self._layer_unitary_pair(layer_idx)
            state = _apply_shared_spatial_unitary_2d(state, unitary)
            state = multiplexer(state)
            state = _apply_shared_spatial_unitary_2d(state, adjoint)

            if layer_idx < self.quantum_layers - 1:
                state, current_layout = move_active_qubit_to_condition(
                    state,
                    current_layout,
                    x_qubits_to_condition=1,
                    y_qubits_to_condition=1,
                )

        measured = self.measurement(state)
        if self.reduce_readout_to_feature_distribution:
            measured = _reduce_readout_to_feature_distribution(
                measured,
                model_name="PCSQCNNNoQFT",
            )
        return measured

    def classify_readout_histogram(self, readout_histogram: torch.Tensor) -> torch.Tensor:
        """Map measured readout histograms into logits through the linear head."""

        flattened = _flatten_readout_histogram(
            readout_histogram,
            expected_spatial_shape=self.readout_spatial_shape,
            model_name="PCSQCNNNoQFT",
        )
        return self.classifier(flattened)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the no-QFT PCS-QCNN ablation on a batch of images.

        Args:
            images: Real tensor with shape ``[B, image_size, image_size]``
                containing normalized grayscale values.

        Returns:
            Logits with shape ``[B, num_classes]``.

        Raises:
            ValueError: If the input spatial shape does not match the configured
                ``image_size``.
        """
        measured = self.exact_quantum_readout_probabilities(images)
        measured = self.readout_histogram(measured)
        return self.classify_readout_histogram(measured)
