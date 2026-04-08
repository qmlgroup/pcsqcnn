import pytest
import torch
from torch import nn

from qcnn import PCSQCNN, PCSQCNNNoQFT
from qcnn import hybrid as hybrid_module
from qcnn.layout import move_active_qubit_to_condition
from qcnn.quantum import _apply_fourier_junction_2d, _apply_iqft_2d, _apply_qft_2d


def make_model(
    *,
    image_size: int = 4,
    num_classes: int = 3,
    feature_qubits: int = 1,
    quantum_layers: int = 1,
    brightness_range: tuple[float, float] = (-1.0, 1.0),
    shot_budget: int | None = None,
    reduce_readout_to_feature_distribution: bool = False,
    use_reduced_fourier_junction: bool = True,
    multiplexer_init_scale: float = 0.05,
) -> PCSQCNN:
    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        return PCSQCNN(
            image_size=image_size,
            num_classes=num_classes,
            feature_qubits=feature_qubits,
            quantum_layers=quantum_layers,
            brightness_range=brightness_range,
            shot_budget=shot_budget,
            reduce_readout_to_feature_distribution=reduce_readout_to_feature_distribution,
            use_reduced_fourier_junction=use_reduced_fourier_junction,
            multiplexer_init_scale=multiplexer_init_scale,
        )


def make_no_qft_model(
    *,
    image_size: int = 4,
    num_classes: int = 3,
    feature_qubits: int = 1,
    quantum_layers: int = 1,
    brightness_range: tuple[float, float] = (-1.0, 1.0),
    shot_budget: int | None = None,
    reduce_readout_to_feature_distribution: bool = False,
    multiplexer_init_scale: float = 0.05,
) -> PCSQCNNNoQFT:
    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        return PCSQCNNNoQFT(
            image_size=image_size,
            num_classes=num_classes,
            feature_qubits=feature_qubits,
            quantum_layers=quantum_layers,
            brightness_range=brightness_range,
            shot_budget=shot_budget,
            reduce_readout_to_feature_distribution=reduce_readout_to_feature_distribution,
            multiplexer_init_scale=multiplexer_init_scale,
        )


def _quantum_readout(model: PCSQCNN, images: torch.Tensor) -> torch.Tensor:
    state = model.encoder(images)
    current_layout = model.layer_layouts[0]

    if model.use_reduced_fourier_junction:
        state = _apply_qft_2d(state)
        for layer_idx, multiplexer in enumerate(model.multiplexers):
            state = multiplexer(state)
            if layer_idx < model.quantum_layers - 1:
                state, current_layout = _apply_fourier_junction_2d(state, current_layout)
            else:
                state = _apply_iqft_2d(state)
    else:
        for layer_idx, multiplexer in enumerate(model.multiplexers):
            state = _apply_qft_2d(state)
            state = multiplexer(state)
            state = _apply_iqft_2d(state)
            if layer_idx < model.quantum_layers - 1:
                state, current_layout = move_active_qubit_to_condition(
                    state,
                    current_layout,
                    x_qubits_to_condition=1,
                    y_qubits_to_condition=1,
                )

    return model.measurement(state)


@pytest.mark.parametrize("quantum_layers", [1, 2, 3])
def test_pcsqcnn_returns_logits_with_expected_shape(quantum_layers: int) -> None:
    model = make_model(image_size=8, num_classes=5, feature_qubits=2, quantum_layers=quantum_layers)
    images = torch.randn(3, 8, 8)

    logits = model(images)

    assert logits.shape == (3, 5)


def test_pcsqcnn_classifier_input_size_matches_flattened_measurement_shape_after_pooling() -> None:
    model = make_model(image_size=8, num_classes=10, feature_qubits=3, quantum_layers=3)

    assert model.classifier.in_features == 2 * 2 * (2 ** 3)


def test_pcsqcnn_no_qft_returns_logits_with_expected_shape() -> None:
    model = make_no_qft_model(image_size=8, num_classes=5, feature_qubits=2, quantum_layers=3)
    images = torch.randn(3, 8, 8)

    logits = model(images)

    assert logits.shape == (3, 5)
    assert model.classifier.in_features == 2 * 2 * (2 ** 2)
    assert model.multiplexer_init_scale == pytest.approx(0.05)


def test_pcsqcnn_feature_only_readout_uses_feature_distribution_shape() -> None:
    model = make_model(
        image_size=8,
        num_classes=4,
        feature_qubits=2,
        quantum_layers=3,
        reduce_readout_to_feature_distribution=True,
        shot_budget=None,
    )
    images = torch.rand(3, 8, 8)

    direct_logits = model(images)
    exact_readout = model.exact_quantum_readout_probabilities(images)
    split_logits = model.classify_readout_histogram(exact_readout)

    assert model.classifier.in_features == 2 ** 2
    assert model.readout_spatial_shape == (1, 1, 2 ** 2)
    assert exact_readout.shape == (3, 1, 1, 2 ** 2)
    assert torch.allclose(split_logits, direct_logits, atol=1e-6, rtol=1e-6)


def test_pcsqcnn_no_qft_feature_only_readout_uses_feature_distribution_shape() -> None:
    model = make_no_qft_model(
        image_size=8,
        num_classes=4,
        feature_qubits=2,
        quantum_layers=3,
        reduce_readout_to_feature_distribution=True,
        shot_budget=None,
    )
    images = torch.rand(3, 8, 8)

    direct_logits = model(images)
    exact_readout = model.exact_quantum_readout_probabilities(images)
    split_logits = model.classify_readout_histogram(exact_readout)

    assert model.classifier.in_features == 2 ** 2
    assert model.readout_spatial_shape == (1, 1, 2 ** 2)
    assert exact_readout.shape == (3, 1, 1, 2 ** 2)
    assert torch.allclose(split_logits, direct_logits, atol=1e-6, rtol=1e-6)


def test_pcsqcnn_split_stages_match_deterministic_forward() -> None:
    model = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2, shot_budget=None)
    images = torch.rand(3, 8, 8)

    direct_logits = model(images)
    split_logits = model.classify_readout_histogram(model.exact_quantum_readout_probabilities(images))

    assert torch.allclose(split_logits, direct_logits, atol=1e-6, rtol=1e-6)


def test_pcsqcnn_no_qft_split_stages_match_deterministic_forward() -> None:
    model = make_no_qft_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2, shot_budget=None)
    images = torch.rand(3, 8, 8)

    direct_logits = model(images)
    split_logits = model.classify_readout_histogram(model.exact_quantum_readout_probabilities(images))

    assert torch.allclose(split_logits, direct_logits, atol=1e-6, rtol=1e-6)


def test_pcsqcnn_classifier_stage_supports_repeated_leading_axes() -> None:
    model = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2, shot_budget=None)
    repeated_histograms = torch.rand(5, 3, *model.readout_spatial_shape)

    logits = model.classify_readout_histogram(repeated_histograms)

    assert logits.shape == (5, 3, 4)


def test_pcsqcnn_no_qft_registers_fixed_r_buffers_and_excludes_them_from_parameters() -> None:
    model = make_no_qft_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=3)
    buffer_names = set(dict(model.named_buffers()))
    parameter_names = set(dict(model.named_parameters()))

    assert model.r_unitary_buffer_names == ("r_unitary_layer_0", "r_unitary_layer_1", "r_unitary_layer_2")
    assert model.r_adjoint_buffer_names == ("r_adjoint_layer_0", "r_adjoint_layer_1", "r_adjoint_layer_2")
    assert set(model.r_unitary_buffer_names).issubset(buffer_names)
    assert set(model.r_adjoint_buffer_names).issubset(buffer_names)
    assert not any(name.startswith("r_unitary_layer_") for name in parameter_names)
    assert not any(name.startswith("r_adjoint_layer_") for name in parameter_names)
    assert getattr(model, "r_unitary_layer_0").shape == (8, 8)
    assert getattr(model, "r_unitary_layer_1").shape == (4, 4)
    assert getattr(model, "r_unitary_layer_2").shape == (2, 2)
    assert "r_unitary_layer_0" in model.state_dict()
    assert "r_adjoint_layer_2" in model.state_dict()


def test_pcsqcnn_no_qft_fixed_r_buffers_follow_manual_seed_deterministically() -> None:
    torch.manual_seed(123)
    first = make_no_qft_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=3)
    torch.manual_seed(123)
    second = make_no_qft_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=3)
    torch.manual_seed(124)
    third = make_no_qft_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=3)

    for layer_idx in range(3):
        assert torch.allclose(
            getattr(first, f"r_unitary_layer_{layer_idx}"),
            getattr(second, f"r_unitary_layer_{layer_idx}"),
        )
        assert torch.allclose(
            getattr(first, f"r_adjoint_layer_{layer_idx}"),
            getattr(second, f"r_adjoint_layer_{layer_idx}"),
        )

    assert not torch.allclose(first.r_unitary_layer_0, third.r_unitary_layer_0)


def test_pcsqcnn_and_no_qft_share_trainable_initialization_under_same_seed() -> None:
    torch.manual_seed(123)
    full_model = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2)
    torch.manual_seed(123)
    no_qft_model = make_no_qft_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2)

    for full_multiplexer, no_qft_multiplexer in zip(
        full_model.multiplexers,
        no_qft_model.multiplexers,
        strict=True,
    ):
        assert torch.allclose(full_multiplexer.block_parameters, no_qft_multiplexer.block_parameters)

    assert torch.allclose(full_model.classifier.weight, no_qft_model.classifier.weight)
    assert torch.allclose(full_model.classifier.bias, no_qft_model.classifier.bias)


def test_pcsqcnn_builds_expected_conditioning_schedule() -> None:
    model = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=3)

    assert model.use_reduced_fourier_junction is True
    assert model.shot_budget is None
    assert model.multiplexer_init_scale == pytest.approx(0.05)
    assert len(model.multiplexers) == 3
    assert model.multiplexers[0].x_condition_qubits_to_use == 0
    assert model.multiplexers[0].y_condition_qubits_to_use == 0
    assert model.multiplexers[1].x_condition_qubits_to_use == 1
    assert model.multiplexers[1].y_condition_qubits_to_use == 1
    assert model.multiplexers[2].x_condition_qubits_to_use == 1
    assert model.multiplexers[2].y_condition_qubits_to_use == 1


def test_pcsqcnn_no_qft_builds_expected_conditioning_schedule() -> None:
    model = make_no_qft_model(
        image_size=8,
        num_classes=4,
        feature_qubits=2,
        quantum_layers=3,
        multiplexer_init_scale=0.07,
    )

    assert model.shot_budget is None
    assert model.multiplexer_init_scale == pytest.approx(0.07)
    assert len(model.multiplexers) == 3
    assert model.multiplexers[0].x_condition_qubits_to_use == 0
    assert model.multiplexers[0].y_condition_qubits_to_use == 0
    assert model.multiplexers[1].x_condition_qubits_to_use == 1
    assert model.multiplexers[1].y_condition_qubits_to_use == 1
    assert model.multiplexers[2].x_condition_qubits_to_use == 1
    assert model.multiplexers[2].y_condition_qubits_to_use == 1


def test_pcsqcnn_no_qft_reuses_same_layer_unitary_for_x_and_y_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_no_qft_model(image_size=4, num_classes=3, feature_qubits=1, quantum_layers=2)
    images = torch.rand(2, 4, 4)
    captured: list[tuple[int, torch.Tensor]] = []
    original = hybrid_module._apply_unitary_to_active_axis_1d

    def wrapped(state: torch.Tensor, unitary: torch.Tensor, *, active_axis: int) -> torch.Tensor:
        captured.append((active_axis, unitary.detach().clone()))
        return original(state, unitary, active_axis=active_axis)

    monkeypatch.setattr(hybrid_module, "_apply_unitary_to_active_axis_1d", wrapped)

    _ = model(images)

    assert [active_axis for active_axis, _ in captured] == [1, 3, 1, 3, 1, 3, 1, 3]
    assert torch.allclose(captured[0][1], model.r_unitary_layer_0)
    assert torch.allclose(captured[1][1], model.r_unitary_layer_0)
    assert torch.allclose(captured[2][1], model.r_adjoint_layer_0)
    assert torch.allclose(captured[3][1], model.r_adjoint_layer_0)
    assert torch.allclose(captured[4][1], model.r_unitary_layer_1)
    assert torch.allclose(captured[5][1], model.r_unitary_layer_1)
    assert torch.allclose(captured[6][1], model.r_adjoint_layer_1)
    assert torch.allclose(captured[7][1], model.r_adjoint_layer_1)


def test_pcsqcnn_normalizes_zero_shot_budget_to_infinite_shot_mode() -> None:
    model = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2, shot_budget=0)

    assert model.shot_budget is None
    assert model.readout_histogram.shot_budget is None


def test_pcsqcnn_no_qft_normalizes_zero_shot_budget_to_infinite_shot_mode() -> None:
    model = make_no_qft_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2, shot_budget=0)

    assert model.shot_budget is None
    assert model.readout_histogram.shot_budget is None


def test_pcsqcnn_passes_brightness_range_to_internal_encoder() -> None:
    with pytest.warns(UserWarning) as caught:
        model = PCSQCNN(
            image_size=8,
            num_classes=4,
            feature_qubits=2,
            quantum_layers=2,
            brightness_range=(-0.5, 0.5),
        )

    assert model.brightness_range == (-0.5, 0.5)
    assert model.encoder.brightness_range == (-0.5, 0.5)
    messages = [str(warning.message) for warning in caught]
    assert any("brightness_range deviates" in message for message in messages)
    assert any("1/sqrt(XY)" in message for message in messages)


def test_pcsqcnn_no_qft_passes_brightness_range_to_internal_encoder() -> None:
    with pytest.warns(UserWarning) as caught:
        model = PCSQCNNNoQFT(
            image_size=8,
            num_classes=4,
            feature_qubits=2,
            quantum_layers=2,
            brightness_range=(-0.5, 0.5),
        )

    assert model.brightness_range == (-0.5, 0.5)
    assert model.encoder.brightness_range == (-0.5, 0.5)
    messages = [str(warning.message) for warning in caught]
    assert any("brightness_range deviates" in message for message in messages)
    assert any("1/sqrt(XY)" in message for message in messages)


def test_pcsqcnn_random_multiplexer_initialization_follows_manual_seed() -> None:
    torch.manual_seed(123)
    first = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2)
    torch.manual_seed(123)
    second = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2)
    torch.manual_seed(124)
    third = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2)

    for first_multiplexer, second_multiplexer, third_multiplexer in zip(
        first.multiplexers,
        second.multiplexers,
        third.multiplexers,
        strict=True,
    ):
        assert torch.allclose(first_multiplexer.block_parameters, second_multiplexer.block_parameters)
        assert not torch.allclose(first_multiplexer.block_parameters, third_multiplexer.block_parameters)


def test_pcsqcnn_no_qft_random_multiplexer_initialization_follows_manual_seed() -> None:
    torch.manual_seed(123)
    first = make_no_qft_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2)
    torch.manual_seed(123)
    second = make_no_qft_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2)
    torch.manual_seed(124)
    third = make_no_qft_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2)

    for first_multiplexer, second_multiplexer, third_multiplexer in zip(
        first.multiplexers,
        second.multiplexers,
        third.multiplexers,
        strict=True,
    ):
        assert torch.allclose(first_multiplexer.block_parameters, second_multiplexer.block_parameters)
        assert not torch.allclose(first_multiplexer.block_parameters, third_multiplexer.block_parameters)


def test_pcsqcnn_random_multiplexer_init_populates_higher_feature_channels() -> None:
    torch.manual_seed(0)
    model = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=1, shot_budget=None)
    images = torch.rand(4, 8, 8)

    readout = model.exact_quantum_readout_probabilities(images)
    higher_feature_mass = readout[..., 2:].sum()

    assert higher_feature_mass.item() > 0.0


def test_pcsqcnn_random_multiplexer_init_allows_gradient_from_higher_feature_channels() -> None:
    torch.manual_seed(0)
    model = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=1, shot_budget=None)
    images = torch.rand(4, 8, 8)

    readout = model.exact_quantum_readout_probabilities(images)
    objective = readout[..., 2:].sum()
    objective.backward()

    assert any(
        multiplexer.block_parameters.grad is not None
        and torch.count_nonzero(multiplexer.block_parameters.grad) > 0
        for multiplexer in model.multiplexers
    )


def test_pcsqcnn_supports_cross_entropy_backward_through_quantum_layers() -> None:
    torch.manual_seed(0)
    model = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=3)
    images = torch.rand(2, 8, 8)
    labels = torch.tensor([0, 3], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()

    parameter_grad_count = sum(
        int(torch.count_nonzero(multiplexer.block_parameters.grad.abs()).item())
        for multiplexer in model.multiplexers
        if multiplexer.block_parameters.grad is not None
    )

    assert model.classifier.weight.grad is not None
    assert model.classifier.bias.grad is not None
    assert torch.count_nonzero(model.classifier.weight.grad) > 0
    assert parameter_grad_count > 0


def test_pcsqcnn_no_qft_supports_cross_entropy_backward_through_trainable_parts_only() -> None:
    torch.manual_seed(0)
    model = make_no_qft_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=3)
    images = torch.rand(2, 8, 8)
    labels = torch.tensor([0, 3], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()

    parameter_grad_count = sum(
        int(torch.count_nonzero(multiplexer.block_parameters.grad.abs()).item())
        for multiplexer in model.multiplexers
        if multiplexer.block_parameters.grad is not None
    )

    assert model.classifier.weight.grad is not None
    assert torch.count_nonzero(model.classifier.weight.grad) > 0
    assert parameter_grad_count > 0
    assert model.r_unitary_layer_0.requires_grad is False
    assert model.r_unitary_layer_0.grad is None
    assert model.r_adjoint_layer_0.requires_grad is False
    assert model.r_adjoint_layer_0.grad is None


def test_pcsqcnn_finite_shot_mode_matches_exact_readout_during_training() -> None:
    torch.manual_seed(0)
    exact_model = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2)
    sampled_model = make_model(
        image_size=8,
        num_classes=4,
        feature_qubits=2,
        quantum_layers=2,
        shot_budget=8,
    )
    sampled_model.load_state_dict(exact_model.state_dict())
    exact_model.train()
    sampled_model.train()
    images = torch.rand(2, 8, 8)
    labels = torch.tensor([0, 3], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    exact_logits = exact_model(images)
    sampled_logits = sampled_model(images)
    loss = criterion(sampled_logits, labels)
    loss.backward()

    assert torch.allclose(sampled_logits, exact_logits, atol=1e-6, rtol=1e-6)
    assert sampled_model.classifier.weight.grad is not None
    assert torch.count_nonzero(sampled_model.classifier.weight.grad) > 0
    assert any(
        multiplexer.block_parameters.grad is not None
        and torch.count_nonzero(multiplexer.block_parameters.grad) > 0
        for multiplexer in sampled_model.multiplexers
    )


def test_pcsqcnn_finite_shot_mode_samples_stochastic_eval_readout() -> None:
    torch.manual_seed(0)
    exact_model = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2)
    sampled_model = make_model(
        image_size=8,
        num_classes=4,
        feature_qubits=2,
        quantum_layers=2,
        shot_budget=7,
    )
    sampled_model.load_state_dict(exact_model.state_dict())
    images = torch.rand(4, 8, 8)
    exact_model.eval()
    sampled_model.eval()

    exact_readout = _quantum_readout(exact_model, images)
    torch.manual_seed(11)
    sampled_readout_a = sampled_model.readout_histogram(exact_readout)
    torch.manual_seed(11)
    sampled_readout_b = sampled_model.readout_histogram(exact_readout)
    torch.manual_seed(12)
    sampled_readout_c = sampled_model.readout_histogram(exact_readout)
    torch.manual_seed(11)
    sampled_logits = sampled_model(images)

    assert sampled_readout_a.shape == exact_readout.shape
    assert torch.allclose(sampled_readout_a, sampled_readout_b)
    assert not torch.allclose(sampled_readout_a, exact_readout)
    assert not torch.allclose(sampled_readout_a, sampled_readout_c)
    assert torch.allclose(sampled_readout_a.sum(dim=(1, 2, 3)), torch.ones(images.shape[0]))
    assert sampled_logits.shape == (4, 4)


def test_pcsqcnn_reduced_fourier_junction_matches_explicit_path() -> None:
    torch.manual_seed(0)
    reduced_model = make_model(
        image_size=8,
        num_classes=4,
        feature_qubits=2,
        quantum_layers=3,
        use_reduced_fourier_junction=True,
    )
    explicit_model = make_model(
        image_size=8,
        num_classes=4,
        feature_qubits=2,
        quantum_layers=3,
        use_reduced_fourier_junction=False,
    )
    explicit_model.load_state_dict(reduced_model.state_dict())
    images = torch.rand(2, 8, 8)

    reduced_logits = reduced_model(images)
    explicit_logits = explicit_model(images)

    assert torch.allclose(reduced_logits, explicit_logits, atol=1e-6, rtol=1e-6)


def test_pcsqcnn_rejects_invalid_quantum_layer_counts() -> None:
    with pytest.raises(ValueError, match="quantum_layers must be positive"):
        PCSQCNN(image_size=4, num_classes=2, feature_qubits=1, quantum_layers=0)

    with pytest.raises(ValueError, match="cannot exceed the number of index qubits per axis"):
        PCSQCNN(image_size=4, num_classes=2, feature_qubits=1, quantum_layers=3)


def test_pcsqcnn_rejects_non_positive_multiplexer_init_scale() -> None:
    with pytest.raises(ValueError, match="multiplexer_init_scale must be positive"):
        PCSQCNN(image_size=4, num_classes=2, feature_qubits=1, quantum_layers=1, multiplexer_init_scale=0.0)


def test_pcsqcnn_no_qft_rejects_non_positive_multiplexer_init_scale() -> None:
    with pytest.raises(ValueError, match="multiplexer_init_scale must be positive"):
        PCSQCNNNoQFT(
            image_size=4,
            num_classes=2,
            feature_qubits=1,
            quantum_layers=1,
            multiplexer_init_scale=0.0,
        )


def test_pcsqcnn_surfaces_encoder_input_validation() -> None:
    model = make_model(image_size=4, num_classes=2, feature_qubits=1, quantum_layers=2)

    with pytest.raises(ValueError, match=r"\[B, X, Y\]"):
        model(torch.zeros(2, 1, 4, 4))


def test_pcsqcnn_one_layer_configuration_matches_previous_logit_shape() -> None:
    model = make_model(image_size=4, num_classes=5, feature_qubits=2, quantum_layers=1)
    images = torch.rand(3, 4, 4)

    logits = model(images)

    assert logits.shape == (3, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_pcsqcnn_runs_forward_and_backward_on_cuda() -> None:
    torch.manual_seed(0)
    model = make_model(image_size=8, num_classes=4, feature_qubits=2, quantum_layers=2).to("cuda")
    images = torch.rand(2, 8, 8, device="cuda")
    labels = torch.tensor([0, 3], dtype=torch.long, device="cuda")
    criterion = nn.CrossEntropyLoss()

    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()

    multiplexer_grad = next(
        multiplexer.block_parameters.grad
        for multiplexer in model.multiplexers
        if multiplexer.block_parameters.grad is not None
    )

    assert logits.is_cuda
    assert model.classifier.weight.grad is not None
    assert model.classifier.weight.grad.is_cuda
    assert multiplexer_grad is not None
    assert multiplexer_grad.is_cuda
