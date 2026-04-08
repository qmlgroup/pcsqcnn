from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from qcnn import (
    AutoTrainingConfig,
    AutoTrainingSnapshotBatchwiseLossSamplingEvaluation,
    AutoTrainingSnapshotReadoutLandscapeEvaluation,
    AutoTrainingSnapshotLayerGradientNormsEvaluation,
    AutoTrainingSnapshotReadoutEntropyEvaluation,
    AutoTrainingSnapshotRepeatedEvaluation,
    AutoTrainingSnapshotTestEvaluation,
    ImageClassifierRunner,
    MnistDatasetConfig,
    ModelSpec,
    OptimizerConfig,
    OutputConfig,
    PCSQCNN,
    PreparedMnistSplits,
    SeedConfig,
    TensorImageDataset,
    TrainingConfig,
    compute_histogram_shannon_entropy,
    evaluate_auto_training_snapshot_layer_gradient_norms_on_saved_mnist_test,
    evaluate_auto_training_snapshot_batchwise_loss_sampling_on_saved_mnist_test,
    evaluate_auto_training_snapshot_readout_entropy_on_saved_mnist_test,
    evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test,
    evaluate_auto_training_snapshot_repeated_on_saved_mnist_test,
    evaluate_auto_training_snapshot_on_saved_mnist_test,
    evaluate_loaded_auto_training_run_on_saved_mnist_test,
    format_nested_mapping_markdown,
    load_auto_training_run,
    load_model_checkpoint,
    reconstruct_run_runner_and_test_loader,
    reconstruct_saved_mnist_splits_from_run,
    resolve_snapshot_trainable_layer_blocks,
    save_model_checkpoint,
    run_mnist_auto_training,
)
from qcnn import automation as automation_module
from qcnn import serialization as serialization_module


def make_pcsqcnn_runner(
    *,
    fit: bool,
    shot_budget: int | None = None,
    reduce_readout_to_feature_distribution: bool = False,
) -> tuple[ImageClassifierRunner, DataLoader, DataLoader]:
    torch.manual_seed(0)
    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        model = PCSQCNN(
            image_size=4,
            num_classes=2,
            feature_qubits=1,
            quantum_layers=2,
            shot_budget=shot_budget,
            reduce_readout_to_feature_distribution=reduce_readout_to_feature_distribution,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    runner = ImageClassifierRunner(model=model, optimizer=optimizer)

    train_images = torch.tensor(
        [
            [[0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]],
            [[0.6, 0.5, 0.4, 0.3], [0.5, 0.4, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1], [0.3, 0.2, 0.1, 0.0]],
            [[0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]],
            [[0.8, 0.8, 0.8, 0.8], [0.8, 0.8, 0.8, 0.8], [0.8, 0.8, 0.8, 0.8], [0.8, 0.8, 0.8, 0.8]],
        ],
        dtype=torch.float32,
    )
    test_images = torch.tensor(
        [
            [[0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05]],
            [[0.95, 0.95, 0.95, 0.95], [0.95, 0.95, 0.95, 0.95], [0.95, 0.95, 0.95, 0.95], [0.95, 0.95, 0.95, 0.95]],
        ],
        dtype=torch.float32,
    )
    train_dataset = TensorImageDataset(
        train_images,
        torch.tensor([0, 1, 0, 1], dtype=torch.long),
        metadata={
            "dataset_name": "MNIST",
            "split": "train",
            "image_size": 4,
            "scaled_image_size": 4,
            "max_offset": 0,
            "samples_per_class": 2,
            "seed": 7,
        },
    )
    test_dataset = TensorImageDataset(
        test_images,
        torch.tensor([0, 1], dtype=torch.long),
        metadata={
            "dataset_name": "MNIST",
            "split": "test",
            "image_size": 4,
            "scaled_image_size": 4,
            "max_offset": 0,
            "test_split": "standard",
        },
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if fit:
        runner.fit(train_loader, test_loader, num_epochs=2)

    return runner, train_loader, test_loader

def make_fake_mnist_splits(*, samples_per_class: int | None = 2) -> PreparedMnistSplits:
    train_dataset = TensorImageDataset(
        torch.tensor(
            [
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([0, 1], dtype=torch.long),
        metadata={
            "dataset_name": "MNIST",
            "split": "train",
            "image_size": 4,
            "scaled_image_size": 4,
            "max_offset": 0,
            "samples_per_class": samples_per_class,
            "seed": 7,
        },
    )
    test_dataset = TensorImageDataset(
        torch.tensor(
            [
                [[0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05]],
                [[0.95, 0.95, 0.95, 0.95], [0.95, 0.95, 0.95, 0.95], [0.95, 0.95, 0.95, 0.95], [0.95, 0.95, 0.95, 0.95]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([0, 1], dtype=torch.long),
        metadata={
            "dataset_name": "MNIST",
            "split": "test",
            "image_size": 4,
            "scaled_image_size": 4,
            "max_offset": 0,
            "test_split": "standard",
        },
    )
    return PreparedMnistSplits(train=train_dataset, test=test_dataset)


def make_auto_training_config(
    tmp_path: Path,
    *,
    model_spec: ModelSpec | None = None,
    samples_per_class: int | None = 2,
    num_epochs: int = 2,
    snapshot_epochs: tuple[int, ...] = (),
) -> AutoTrainingConfig:
    return AutoTrainingConfig(
        dataset=MnistDatasetConfig(
            root=tmp_path / "data",
            samples_per_class=samples_per_class,
            image_size=4,
            train_batch_size=2,
            test_batch_size=4,
            download=False,
        ),
        model=(
            model_spec
            if model_spec is not None
            else ModelSpec(
                module="qcnn.hybrid",
                class_name="PCSQCNN",
                constructor_kwargs={
                    "image_size": 4,
                    "num_classes": 2,
                    "feature_qubits": 1,
                    "quantum_layers": 2,
                },
            )
        ),
        optimizer=OptimizerConfig(
            kind="adam",
            learning_rate=1e-2,
        ),
        training=TrainingConfig(
            num_epochs=num_epochs,
            snapshot_epochs=snapshot_epochs,
            device="cpu",
            torch_matmul_precision=None,
        ),
        seeds=SeedConfig(
            base_seed=7,
            seed_count=1,
        ),
        output=OutputConfig(
            root=tmp_path / "outputs",
            use_timestamp_dir=False,
        ),
    )


def test_format_nested_mapping_markdown_handles_nested_mappings_and_mapping_sequences() -> None:
    markdown = format_nested_mapping_markdown(
        {
            "artifact": {
                "enabled": True,
                "metrics": ["accuracy", "f1"],
            },
            "optimizer": {
                "defaults": {"lr": 0.01, "weight_decay": 0.0},
                "param_groups": [
                    {"lr": 0.01, "betas": [0.9, 0.999]},
                    {"lr": 0.001, "betas": [0.8, 0.95]},
                ],
            },
        },
        title="Example",
    )

    assert "## Example" in markdown
    assert "### Artifact" in markdown
    assert "- Enabled: `True`" in markdown
    assert "- Metrics: `[accuracy, f1]`" in markdown
    assert "### Optimizer" in markdown
    assert "#### Defaults" in markdown
    assert "#### Param Groups 1" in markdown
    assert "#### Param Groups 2" in markdown
    assert "- Betas: `[0.900000, 0.999000]`" in markdown


def test_save_and_load_model_checkpoint_round_trip_preserves_logits_state(tmp_path: Path) -> None:
    runner, _, test_loader = make_pcsqcnn_runner(fit=True)
    checkpoint_path = tmp_path / "checkpoint.pt"
    model_spec = ModelSpec(
        module="qcnn.hybrid",
        class_name="PCSQCNN",
        constructor_kwargs={
            "image_size": 4,
            "num_classes": 2,
            "feature_qubits": 1,
            "quantum_layers": 2,
        },
    )

    save_model_checkpoint(runner.model, model_spec, checkpoint_path)
    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        loaded_checkpoint = load_model_checkpoint(checkpoint_path)

    assert loaded_checkpoint.checkpoint_path == checkpoint_path
    assert loaded_checkpoint.model_spec == model_spec
    with torch.no_grad():
        reference_images = test_loader.dataset.images
        assert torch.allclose(
            loaded_checkpoint.model(reference_images),
            runner.model(reference_images),
            atol=1e-6,
            rtol=1e-6,
        )


def test_load_model_checkpoint_rejects_wrong_checkpoint_type(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "bad_checkpoint.pt"
    torch.save({"checkpoint_type": "other", "format_version": 1}, checkpoint_path)

    with pytest.raises(ValueError, match="checkpoint_type"):
        load_model_checkpoint(checkpoint_path)


def test_load_auto_training_run_and_reconstruct_saved_mnist_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_splits = make_fake_mnist_splits()

    def fake_prepare_mnist_splits(**kwargs):
        return fake_splits

    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    monkeypatch.setattr(serialization_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        auto_training_result = run_mnist_auto_training(make_auto_training_config(tmp_path))

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        loaded_run = load_auto_training_run(auto_training_result.output_directory, seed=7)

    assert loaded_run.training_history == auto_training_result.runs[0].history
    assert loaded_run.loss_name == "CrossEntropyLoss"
    assert loaded_run.model.__class__.__name__ == "PCSQCNN"
    assert loaded_run.model_spec.class_name == "PCSQCNN"
    assert loaded_run.saved_mnist_test_config() == {
        "dataset_name": "MNIST",
        "image_size": 4,
        "scaled_image_size": 4,
        "max_offset": 0,
        "samples_per_class": 2,
        "seed": 7,
        "test_loader": {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
        },
    }

    restored = reconstruct_saved_mnist_splits_from_run(
        loaded_run,
        root=tmp_path / "data",
        download=False,
    )
    assert restored is fake_splits

    context = reconstruct_run_runner_and_test_loader(
        loaded_run,
        fake_splits,
        device="cpu",
    )
    assert context.device == torch.device("cpu")
    assert context.runner.model is loaded_run.model
    assert context.test_loader.batch_size == 4
    assert context.notes == []

    evaluation = evaluate_loaded_auto_training_run_on_saved_mnist_test(
        loaded_run,
        root=tmp_path / "data",
        device="cpu",
        download=False,
    )
    assert evaluation.context.device == torch.device("cpu")
    assert evaluation.summary.metrics["accuracy"] == pytest.approx(
        context.runner.evaluate_loader(context.test_loader).metrics["accuracy"]
    )
    assert evaluation.report_markdown.startswith("## Recomputed Test Report")


def test_load_auto_training_run_reconstructs_full_train_split_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_splits = make_fake_mnist_splits(samples_per_class=None)

    def fake_prepare_mnist_splits(**kwargs):
        return fake_splits

    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    monkeypatch.setattr(serialization_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        auto_training_result = run_mnist_auto_training(
            make_auto_training_config(tmp_path, samples_per_class=None)
        )

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        loaded_run = load_auto_training_run(auto_training_result.output_directory, seed=7)

    assert loaded_run.saved_mnist_test_config() == {
        "dataset_name": "MNIST",
        "image_size": 4,
        "scaled_image_size": 4,
        "max_offset": 0,
        "samples_per_class": None,
        "seed": 7,
        "test_loader": {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
        },
    }

    restored = reconstruct_saved_mnist_splits_from_run(
        loaded_run,
        root=tmp_path / "data",
        download=False,
    )
    assert restored is fake_splits


def test_loaded_run_saved_mnist_test_config_rejects_missing_translation_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_splits = make_fake_mnist_splits()

    def fake_prepare_mnist_splits(**kwargs):
        return fake_splits

    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    monkeypatch.setattr(serialization_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        auto_training_result = run_mnist_auto_training(make_auto_training_config(tmp_path))

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        loaded_run = load_auto_training_run(auto_training_result.output_directory, seed=7)

    dataset_config = loaded_run.resolved_config["dataset"]
    dataset_config.pop("scaled_image_size")
    dataset_config.pop("max_offset")

    with pytest.raises(ValueError, match="scaled_image_size, max_offset"):
        loaded_run.saved_mnist_test_config()


def test_loaded_run_saved_mnist_test_config_rejects_missing_test_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_splits = make_fake_mnist_splits()

    def fake_prepare_mnist_splits(**kwargs):
        return fake_splits

    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    monkeypatch.setattr(serialization_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        auto_training_result = run_mnist_auto_training(make_auto_training_config(tmp_path))

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        loaded_run = load_auto_training_run(auto_training_result.output_directory, seed=7)

    dataset_config = loaded_run.resolved_config["dataset"]
    dataset_config.pop("test_batch_size")

    with pytest.raises(ValueError, match="test_batch_size"):
        loaded_run.saved_mnist_test_config()


def test_load_auto_training_run_reconstructs_classical_model_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_splits = make_fake_mnist_splits()

    def fake_prepare_mnist_splits(**kwargs):
        return fake_splits

    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    auto_training_result = run_mnist_auto_training(
        make_auto_training_config(
            tmp_path,
            model_spec=ModelSpec(
                module="qcnn.classic",
                class_name="ClassicalMLP",
                constructor_kwargs={"image_size": 4, "num_classes": 2},
            ),
        )
    )

    loaded_run = load_auto_training_run(auto_training_result.output_directory, seed=7)

    assert loaded_run.model.__class__.__name__ == "ClassicalMLP"
    assert loaded_run.training_history == auto_training_result.runs[0].history


def test_evaluate_auto_training_snapshot_on_saved_mnist_test_overrides_shot_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_splits = make_fake_mnist_splits()
    batch_progress_updates: list[tuple[int, int]] = []

    def fake_prepare_mnist_splits(**kwargs):
        return fake_splits

    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    monkeypatch.setattr(serialization_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        auto_training_result = run_mnist_auto_training(
            make_auto_training_config(tmp_path, num_epochs=2, snapshot_epochs=(1, 2))
        )

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        evaluation = evaluate_auto_training_snapshot_on_saved_mnist_test(
            auto_training_result.output_directory,
            seed=7,
            epoch=1,
            shot_budget=5,
            root=tmp_path / "data",
            device="cpu",
            download=False,
            batch_progress_callback=lambda completed, total: batch_progress_updates.append((completed, total)),
        )

    assert isinstance(evaluation, AutoTrainingSnapshotTestEvaluation)
    assert evaluation.checkpoint_path.name == "checkpoint_epoch1_seed7.pt"
    assert evaluation.model_spec.class_name == "PCSQCNN"
    assert evaluation.model_spec.constructor_kwargs["shot_budget"] == 5
    assert evaluation.epoch == 1
    assert evaluation.shot_budget == 5
    assert evaluation.predictions.dtype == torch.long
    assert evaluation.targets.dtype == torch.long
    assert evaluation.predictions.shape == evaluation.targets.shape == (2,)
    assert evaluation.targets.tolist() == [0, 1]
    assert 0.0 <= evaluation.summary.metrics["accuracy"] <= 1.0
    assert batch_progress_updates == [(1, 1)]


def test_evaluate_auto_training_snapshot_repeated_on_saved_mnist_test_returns_repetition_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_splits = make_fake_mnist_splits()

    def fake_prepare_mnist_splits(**kwargs):
        return fake_splits

    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    monkeypatch.setattr(serialization_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        auto_training_result = run_mnist_auto_training(
            make_auto_training_config(tmp_path, num_epochs=2, snapshot_epochs=(1, 2))
        )

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        evaluation = evaluate_auto_training_snapshot_repeated_on_saved_mnist_test(
            auto_training_result.output_directory,
            seed=7,
            epoch=1,
            shot_budget=5,
            repetitions=4,
            root=tmp_path / "data",
            device="cpu",
            download=False,
        )

    assert isinstance(evaluation, AutoTrainingSnapshotRepeatedEvaluation)
    assert evaluation.context.checkpoint_path.name == "checkpoint_epoch1_seed7.pt"
    assert evaluation.context.model_spec.constructor_kwargs["shot_budget"] == 5
    assert evaluation.context.epoch == 1
    assert evaluation.context.shot_budget == 5
    assert evaluation.mean_loss.shape == (4,)
    assert evaluation.mean_accuracy.shape == (4,)
    assert evaluation.mean_loss.dtype == torch.float32
    assert evaluation.mean_accuracy.dtype == torch.float32
    assert torch.all((0.0 <= evaluation.mean_accuracy) & (evaluation.mean_accuracy <= 1.0))


def test_resolve_snapshot_trainable_layer_blocks_returns_quantum_layers_then_classifier() -> None:
    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        model = PCSQCNN(
            image_size=4,
            num_classes=2,
            feature_qubits=1,
            quantum_layers=2,
            reduce_readout_to_feature_distribution=False,
        )

    layer_blocks = resolve_snapshot_trainable_layer_blocks(model)

    assert [layer_key for layer_key, _, _ in layer_blocks] == [
        "multiplexers.0",
        "multiplexers.1",
        "classifier",
    ]
    assert [layer_label for _, layer_label, _ in layer_blocks] == [
        "Quantum 1",
        "Quantum 2",
        "Classifier",
    ]
    assert [name for name, _ in layer_blocks[-1][2]] == [
        "classifier.weight",
        "classifier.bias",
    ]


def test_compute_histogram_shannon_entropy_handles_zeros_without_nan() -> None:
    histograms = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    entropy = compute_histogram_shannon_entropy(histograms)

    assert entropy.shape == (2,)
    assert entropy[0].item() == pytest.approx(0.0)
    assert entropy[1].item() == pytest.approx(float(torch.log(torch.tensor(2.0)).item()))


def test_evaluate_auto_training_snapshot_layer_gradient_norms_matches_full_dataset_mean_gradient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_splits = make_fake_mnist_splits()

    def fake_prepare_mnist_splits(**kwargs):
        return fake_splits

    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    monkeypatch.setattr(serialization_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        auto_training_result = run_mnist_auto_training(
            make_auto_training_config(tmp_path, num_epochs=1, snapshot_epochs=(1,))
        )

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        evaluation = evaluate_auto_training_snapshot_layer_gradient_norms_on_saved_mnist_test(
            auto_training_result.output_directory,
            seed=7,
            epoch=1,
            root=tmp_path / "data",
            device="cpu",
            batch_size=1,
            download=False,
        )

    assert isinstance(evaluation, AutoTrainingSnapshotLayerGradientNormsEvaluation)
    assert evaluation.layer_labels[-1] == "Classifier"

    context = serialization_module.reconstruct_auto_training_snapshot_runner_and_test_loader(
        auto_training_result.output_directory,
        seed=7,
        epoch=1,
        shot_budget=None,
        root=tmp_path / "data",
        device="cpu",
        download=False,
    )
    layer_blocks = resolve_snapshot_trainable_layer_blocks(context.runner.model)
    images = context.test_loader.dataset.images.to(context.device)
    labels = context.test_loader.dataset.labels.to(device=context.device, dtype=torch.long)
    logits = context.runner.model(images)
    loss = context.runner.loss_collector.compute_loss(logits, labels)
    parameters = [
        parameter
        for _, _, group_parameters in layer_blocks
        for _, parameter in group_parameters
    ]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)

    expected_mean_gradients: list[torch.Tensor] = []
    gradient_offset = 0
    for _, _, group_parameters in layer_blocks:
        group_gradients: list[torch.Tensor] = []
        for _, parameter in group_parameters:
            gradient = gradients[gradient_offset]
            gradient_offset += 1
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            group_gradients.append(gradient.detach().cpu().reshape(-1))
        expected_mean_gradients.append(torch.cat(group_gradients, dim=0))
    expected_gradient_norms = torch.stack(
        [gradient.norm() for gradient in expected_mean_gradients]
    ).to(dtype=torch.float32)

    assert list(evaluation.layer_keys) == [layer_key for layer_key, _, _ in layer_blocks]
    assert list(evaluation.layer_labels) == [layer_label for _, layer_label, _ in layer_blocks]
    assert torch.allclose(evaluation.gradient_norms, expected_gradient_norms, atol=1e-6, rtol=1e-6)
    for actual, expected in zip(evaluation.mean_gradients, expected_mean_gradients, strict=True):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_evaluate_auto_training_snapshot_readout_entropy_returns_per_sample_tensor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_splits = make_fake_mnist_splits()

    def fake_prepare_mnist_splits(**kwargs):
        return fake_splits

    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    monkeypatch.setattr(serialization_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        auto_training_result = run_mnist_auto_training(
            make_auto_training_config(
                tmp_path,
                num_epochs=1,
                snapshot_epochs=(1,),
                model_spec=ModelSpec(
                    module="qcnn.hybrid",
                    class_name="PCSQCNN",
                    constructor_kwargs={
                        "image_size": 4,
                        "num_classes": 2,
                        "feature_qubits": 1,
                        "quantum_layers": 2,
                        "reduce_readout_to_feature_distribution": False,
                    },
                ),
            )
        )

    torch.manual_seed(0)
    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        evaluation = evaluate_auto_training_snapshot_readout_entropy_on_saved_mnist_test(
            auto_training_result.output_directory,
            seed=7,
            epoch=1,
            shot_budget=5,
            root=tmp_path / "data",
            device="cpu",
            batch_size=1,
            download=False,
        )

    assert isinstance(evaluation, AutoTrainingSnapshotReadoutEntropyEvaluation)
    assert evaluation.shot_budget == 5
    assert evaluation.entropy.shape == (2,)
    assert evaluation.entropy.dtype == torch.float32
    assert torch.isfinite(evaluation.entropy).all()
    assert torch.all(evaluation.entropy >= 0.0)


def test_evaluate_auto_training_snapshot_batchwise_loss_sampling_returns_batch_aggregates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_splits = make_fake_mnist_splits()
    batch_progress_updates: list[tuple[int, int]] = []

    def fake_prepare_mnist_splits(**kwargs):
        return fake_splits

    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    monkeypatch.setattr(serialization_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        auto_training_result = run_mnist_auto_training(
            make_auto_training_config(tmp_path, num_epochs=2, snapshot_epochs=(1, 2))
        )

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        evaluation = evaluate_auto_training_snapshot_batchwise_loss_sampling_on_saved_mnist_test(
            auto_training_result.output_directory,
            seed=7,
            epoch=1,
            shot_budgets=(3, None),
            repetitions=4,
            batch_size=1,
            repetition_block_size=2,
            root=tmp_path / "data",
            device="cpu",
            download=False,
            batch_progress_callback=lambda completed, total: batch_progress_updates.append((completed, total)),
        )

    assert isinstance(evaluation, AutoTrainingSnapshotBatchwiseLossSamplingEvaluation)
    assert evaluation.context.checkpoint_path.name == "checkpoint_epoch1_seed7.pt"
    assert evaluation.context.model_spec.constructor_kwargs["shot_budget"] is None
    assert evaluation.repetitions == 4
    assert evaluation.batch_size == 1
    assert evaluation.repetition_block_size == 2
    assert [entry.shot_budget for entry in evaluation.evaluations] == [3, None]
    first = evaluation.evaluations[0]
    assert first.batch_sizes.shape == (2,)
    assert first.batch_sizes.tolist() == [1, 1]
    assert first.batch_loss_sum.shape == (4, 2)
    assert first.batch_correct_count.shape == (4, 2)
    assert first.num_draws == 4
    assert first.batch_loss_sum.dtype == torch.float32
    assert first.batch_correct_count.dtype == torch.int64
    second = evaluation.evaluations[1]
    assert second.num_draws == 1
    assert second.batch_loss_sum.shape == (1, 2)
    assert second.batch_correct_count.shape == (1, 2)
    assert batch_progress_updates == [(1, 2), (2, 2)]


def test_evaluate_auto_training_snapshot_readout_landscape_returns_expected_grid_and_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_splits = make_fake_mnist_splits()
    batch_progress_updates: list[tuple[int, int]] = []

    def fake_prepare_mnist_splits(**kwargs):
        return fake_splits

    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    monkeypatch.setattr(serialization_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        auto_training_result = run_mnist_auto_training(
            make_auto_training_config(tmp_path, num_epochs=2, snapshot_epochs=(1, 2))
        )

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        evaluation = evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test(
            auto_training_result.output_directory,
            seed=7,
            epoch=1,
            root=tmp_path / "data",
            device="cpu",
            alpha_beta_points=5,
            sample_batch_size=1,
            grid_chunk_size=3,
            download=False,
            batch_progress_callback=lambda completed, total: batch_progress_updates.append((completed, total)),
        )

    assert isinstance(evaluation, AutoTrainingSnapshotReadoutLandscapeEvaluation)
    assert evaluation.context.checkpoint_path.name == "checkpoint_epoch1_seed7.pt"
    assert evaluation.context.model_spec.constructor_kwargs["shot_budget"] is None
    assert evaluation.context.epoch == 1
    assert evaluation.alpha.shape == (5,)
    assert evaluation.beta.shape == (5,)
    assert evaluation.alpha[2].item() == pytest.approx(0.0)
    assert evaluation.beta[2].item() == pytest.approx(0.0)
    assert evaluation.mean_loss.shape == (5, 5)
    assert evaluation.valid_count.shape == (5, 5)
    assert evaluation.valid_fraction.shape == (5, 5)
    assert evaluation.total_samples == 2
    assert 0 < evaluation.eligible_sample_count <= evaluation.total_samples
    assert torch.allclose(
        evaluation.valid_fraction,
        evaluation.valid_count.to(dtype=torch.float32) / float(evaluation.total_samples),
    )
    assert (evaluation.valid_count >= 0).all()
    assert batch_progress_updates == [(1, 2), (2, 2)]

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        repeated = evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test(
            auto_training_result.output_directory,
            seed=7,
            epoch=1,
            root=tmp_path / "data",
            device="cpu",
            alpha_beta_points=5,
            sample_batch_size=1,
            grid_chunk_size=3,
            download=False,
        )

    assert evaluation.eligible_sample_count == repeated.eligible_sample_count
    assert torch.allclose(evaluation.mean_loss, repeated.mean_loss)
    assert torch.equal(evaluation.valid_count, repeated.valid_count)
    assert torch.allclose(evaluation.valid_fraction, repeated.valid_fraction)


def test_estimate_local_readout_landscape_basis_tracks_dense_local_covariance_modes() -> None:
    probabilities = torch.tensor(
        [
            [0.70, 0.20, 0.10],
            [0.55, 0.30, 0.15],
        ],
        dtype=torch.float32,
    )

    direction_u, direction_v, valid = serialization_module._estimate_local_readout_landscape_basis(
        probabilities,
        sigma_shot_budget=1024,
        iteration_count=12,
    )
    repeated_u, repeated_v, repeated_valid = serialization_module._estimate_local_readout_landscape_basis(
        probabilities,
        sigma_shot_budget=1024,
        iteration_count=12,
    )

    assert valid.tolist() == [True, True]
    assert torch.equal(valid, repeated_valid)
    assert torch.allclose(direction_u, repeated_u)
    assert torch.allclose(direction_v, repeated_v)

    for sample_idx, probability in enumerate(probabilities.to(dtype=torch.float64)):
        covariance = (torch.diag(probability) - torch.outer(probability, probability)) / 1024.0
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        leading_eigenvalues = eigenvalues[order[:2]]
        leading_eigenvectors = eigenvectors[:, order[:2]].to(dtype=torch.float32)
        observed_u = direction_u[sample_idx]
        observed_v = direction_v[sample_idx]
        cosine_u = torch.dot(
            observed_u / torch.norm(observed_u),
            leading_eigenvectors[:, 0] / torch.norm(leading_eigenvectors[:, 0]),
        ).abs()
        cosine_v = torch.dot(
            observed_v / torch.norm(observed_v),
            leading_eigenvectors[:, 1] / torch.norm(leading_eigenvectors[:, 1]),
        ).abs()

        assert float(torch.norm(observed_u).item()) == pytest.approx(float(torch.sqrt(leading_eigenvalues[0]).item()))
        assert float(torch.norm(observed_v).item()) == pytest.approx(float(torch.sqrt(leading_eigenvalues[1]).item()))
        assert float(cosine_u.item()) == pytest.approx(1.0, abs=1e-4)
        assert float(cosine_v.item()) == pytest.approx(1.0, abs=1e-4)
        assert float(observed_u.sum().item()) == pytest.approx(0.0, abs=1e-5)
        assert float(observed_v.sum().item()) == pytest.approx(0.0, abs=1e-5)


def test_evaluate_auto_training_snapshot_readout_landscape_skips_low_rank_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LookupLandscapeModel(torch.nn.Module):
        def __init__(self, readout_by_code: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("readout_by_code", readout_by_code.to(dtype=torch.float32))
            self.readout_spatial_shape = (3, 1, 1)

        def exact_quantum_readout_probabilities(self, images: torch.Tensor) -> torch.Tensor:
            sample_codes = images[:, 0, 0].round().to(dtype=torch.long)
            return self.readout_by_code[sample_codes][:, :, None, None]

        def classify_readout_histogram(self, readout_histogram: torch.Tensor) -> torch.Tensor:
            return readout_histogram.reshape(readout_histogram.shape[0], -1)

    readouts = torch.tensor(
        [
            [0.60, 0.30, 0.10],
            [0.50, 0.50, 0.00],
        ],
        dtype=torch.float32,
    )
    dataset = TensorImageDataset(
        torch.arange(2, dtype=torch.float32)[:, None, None].expand(-1, 2, 2).clone(),
        torch.tensor([0, 1], dtype=torch.long),
        metadata={
            "dataset_name": "MNIST",
            "split": "test",
            "image_size": 2,
            "test_split": "standard",
        },
    )
    fake_context = serialization_module.AutoTrainingSnapshotEvaluationContext(
        checkpoint_path=Path("checkpoint_epoch1_seed7.pt"),
        model_spec=ModelSpec(
            module="qcnn.hybrid",
            class_name="PCSQCNN",
            constructor_kwargs={"image_size": 2, "num_classes": 3, "feature_qubits": 1, "quantum_layers": 1},
        ),
        epoch=1,
        shot_budget=None,
        runner=ImageClassifierRunner(model=LookupLandscapeModel(readouts), device="cpu"),
        test_loader=DataLoader(dataset, batch_size=2, shuffle=False),
        device=torch.device("cpu"),
    )

    monkeypatch.setattr(
        serialization_module,
        "reconstruct_auto_training_snapshot_runner_and_test_loader",
        lambda *args, **kwargs: fake_context,
    )

    evaluation = evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test(
        "unused",
        seed=7,
        epoch=1,
        root="unused",
        device="cpu",
        alpha_beta_points=3,
        sample_batch_size=2,
        grid_chunk_size=2,
        download=False,
    )

    assert evaluation.total_samples == 2
    assert evaluation.eligible_sample_count == 1
    assert evaluation.valid_count[1, 1].item() == 1
    assert evaluation.valid_fraction[1, 1].item() == pytest.approx(0.5)


def test_evaluate_auto_training_snapshot_readout_landscape_filters_invalid_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLandscapeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.readout_spatial_shape = (3, 1, 1)

        def exact_quantum_readout_probabilities(self, images: torch.Tensor) -> torch.Tensor:
            batch_size = images.shape[0]
            return torch.tensor(
                [[[[0.6]], [[0.3]], [[0.1]]]],
                dtype=torch.float32,
                device=images.device,
            ).expand(batch_size, -1, -1, -1)

        def classify_readout_histogram(self, readout_histogram: torch.Tensor) -> torch.Tensor:
            flattened = readout_histogram.reshape(readout_histogram.shape[0], -1)
            return flattened

    dataset = TensorImageDataset(
        torch.zeros((1, 2, 2), dtype=torch.float32),
        torch.tensor([0], dtype=torch.long),
        metadata={
            "dataset_name": "MNIST",
            "split": "test",
            "image_size": 2,
            "test_split": "standard",
        },
    )
    fake_context = serialization_module.AutoTrainingSnapshotEvaluationContext(
        checkpoint_path=Path("checkpoint_epoch1_seed7.pt"),
        model_spec=ModelSpec(
            module="qcnn.hybrid",
            class_name="PCSQCNN",
            constructor_kwargs={"image_size": 2, "num_classes": 3, "feature_qubits": 1, "quantum_layers": 1},
        ),
        epoch=1,
        shot_budget=None,
        runner=ImageClassifierRunner(model=FakeLandscapeModel(), device="cpu"),
        test_loader=DataLoader(dataset, batch_size=1, shuffle=False),
        device=torch.device("cpu"),
    )

    monkeypatch.setattr(
        serialization_module,
        "reconstruct_auto_training_snapshot_runner_and_test_loader",
        lambda *args, **kwargs: fake_context,
    )
    monkeypatch.setattr(
        serialization_module,
        "_estimate_local_readout_landscape_basis",
        lambda probabilities, **kwargs: (
            torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32, device=probabilities.device).expand_as(probabilities),
            torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float32, device=probabilities.device).expand_as(probabilities),
            torch.ones(probabilities.shape[0], dtype=torch.bool, device=probabilities.device),
        ),
    )

    evaluation = evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test(
        "unused",
        seed=7,
        epoch=1,
        root="unused",
        device="cpu",
        alpha_beta_points=3,
        sample_batch_size=1,
        grid_chunk_size=2,
        download=False,
    )

    assert evaluation.valid_count.shape == (3, 3)
    # beta = +1 pushes one component negative; alpha = +1 makes the total probability sum exceed 1.
    assert evaluation.valid_count[2, 1].item() == 0
    assert evaluation.valid_count[1, 2].item() == 0
    assert evaluation.valid_count[1, 1].item() == 1


def test_evaluate_auto_training_snapshot_readout_landscape_supports_custom_axis_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLandscapeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.readout_spatial_shape = (3, 1, 1)

        def exact_quantum_readout_probabilities(self, images: torch.Tensor) -> torch.Tensor:
            batch_size = images.shape[0]
            return torch.tensor(
                [[[[0.6]], [[0.3]], [[0.1]]]],
                dtype=torch.float32,
                device=images.device,
            ).expand(batch_size, -1, -1, -1)

        def classify_readout_histogram(self, readout_histogram: torch.Tensor) -> torch.Tensor:
            flattened = readout_histogram.reshape(readout_histogram.shape[0], -1)
            return flattened

    dataset = TensorImageDataset(
        torch.zeros((1, 2, 2), dtype=torch.float32),
        torch.tensor([0], dtype=torch.long),
        metadata={
            "dataset_name": "MNIST",
            "split": "test",
            "image_size": 2,
            "test_split": "standard",
        },
    )
    fake_context = serialization_module.AutoTrainingSnapshotEvaluationContext(
        checkpoint_path=Path("checkpoint_epoch1_seed7.pt"),
        model_spec=ModelSpec(
            module="qcnn.hybrid",
            class_name="PCSQCNN",
            constructor_kwargs={"image_size": 2, "num_classes": 3, "feature_qubits": 1, "quantum_layers": 1},
        ),
        epoch=1,
        shot_budget=None,
        runner=ImageClassifierRunner(model=FakeLandscapeModel(), device="cpu"),
        test_loader=DataLoader(dataset, batch_size=1, shuffle=False),
        device=torch.device("cpu"),
    )

    monkeypatch.setattr(
        serialization_module,
        "reconstruct_auto_training_snapshot_runner_and_test_loader",
        lambda *args, **kwargs: fake_context,
    )
    monkeypatch.setattr(
        serialization_module,
        "_estimate_local_readout_landscape_basis",
        lambda probabilities, **kwargs: (
            torch.zeros_like(probabilities),
            torch.zeros_like(probabilities),
            torch.ones(probabilities.shape[0], dtype=torch.bool, device=probabilities.device),
        ),
    )

    evaluation = evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test(
        "unused",
        seed=7,
        epoch=1,
        root="unused",
        device="cpu",
        alpha_beta_points=5,
        axis_limit=3.0,
        sample_batch_size=1,
        grid_chunk_size=2,
        download=False,
    )

    assert evaluation.alpha.tolist() == pytest.approx([-3.0, -1.5, 0.0, 1.5, 3.0])
    assert evaluation.beta.tolist() == pytest.approx([-3.0, -1.5, 0.0, 1.5, 3.0])


def test_evaluate_auto_training_snapshot_readout_landscape_accepts_small_probability_sum_overshoot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLandscapeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.readout_spatial_shape = (3, 1, 1)

        def exact_quantum_readout_probabilities(self, images: torch.Tensor) -> torch.Tensor:
            batch_size = images.shape[0]
            return torch.tensor(
                [[[[0.6000010]], [[0.3000005]], [[0.1000005]]]],
                dtype=torch.float32,
                device=images.device,
            ).expand(batch_size, -1, -1, -1)

        def classify_readout_histogram(self, readout_histogram: torch.Tensor) -> torch.Tensor:
            flattened = readout_histogram.reshape(readout_histogram.shape[0], -1)
            return flattened

    dataset = TensorImageDataset(
        torch.zeros((1, 2, 2), dtype=torch.float32),
        torch.tensor([0], dtype=torch.long),
        metadata={
            "dataset_name": "MNIST",
            "split": "test",
            "image_size": 2,
            "test_split": "standard",
        },
    )
    fake_context = serialization_module.AutoTrainingSnapshotEvaluationContext(
        checkpoint_path=Path("checkpoint_epoch1_seed7.pt"),
        model_spec=ModelSpec(
            module="qcnn.hybrid",
            class_name="PCSQCNN",
            constructor_kwargs={"image_size": 2, "num_classes": 3, "feature_qubits": 1, "quantum_layers": 1},
        ),
        epoch=1,
        shot_budget=None,
        runner=ImageClassifierRunner(model=FakeLandscapeModel(), device="cpu"),
        test_loader=DataLoader(dataset, batch_size=1, shuffle=False),
        device=torch.device("cpu"),
    )

    monkeypatch.setattr(
        serialization_module,
        "reconstruct_auto_training_snapshot_runner_and_test_loader",
        lambda *args, **kwargs: fake_context,
    )
    monkeypatch.setattr(
        serialization_module,
        "_estimate_local_readout_landscape_basis",
        lambda probabilities, **kwargs: (
            torch.zeros_like(probabilities),
            torch.zeros_like(probabilities),
            torch.ones(probabilities.shape[0], dtype=torch.bool, device=probabilities.device),
        ),
    )

    evaluation = evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test(
        "unused",
        seed=7,
        epoch=1,
        root="unused",
        device="cpu",
        alpha_beta_points=3,
        sample_batch_size=1,
        grid_chunk_size=2,
        download=False,
    )

    assert torch.equal(evaluation.valid_count, torch.ones((3, 3), dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this smoke test.")
def test_evaluate_auto_training_snapshot_readout_landscape_supports_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_splits = make_fake_mnist_splits()

    def fake_prepare_mnist_splits(**kwargs):
        return fake_splits

    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    monkeypatch.setattr(serialization_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        auto_training_result = run_mnist_auto_training(
            make_auto_training_config(tmp_path, num_epochs=2, snapshot_epochs=(1, 2))
        )

    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        evaluation = evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test(
            auto_training_result.output_directory,
            seed=7,
            epoch=1,
            root=tmp_path / "data",
            device="cuda",
            alpha_beta_points=5,
            sample_batch_size=1,
            grid_chunk_size=3,
            download=False,
        )

    assert evaluation.context.device.type == "cuda"
