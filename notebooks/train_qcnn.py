import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch

    from qcnn import (
        AutoTrainingConfig,
        ModelSpec,
        MnistDatasetConfig,
        OptimizerConfig,
        OutputConfig,
        SeedConfig,
        TrainingConfig,
        plot_convergence,
        plot_error_analysis,
        run_mnist_auto_training,
    )

    return (
        AutoTrainingConfig,
        ModelSpec,
        MnistDatasetConfig,
        OptimizerConfig,
        OutputConfig,
        Path,
        SeedConfig,
        TrainingConfig,
        mo,
        np,
        plot_convergence,
        plot_error_analysis,
        run_mnist_auto_training,
        torch,
    )


@app.cell
def _(Path, torch):
    "Notebook configuration"
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    artifacts_root = project_root / "artifacts"

    samples_per_class = 1000
    image_size = 16
    batch_size = 64
    seed = 0
    num_classes = 10

    feature_qubits = 1
    quantum_layers = 1
    shot_budget = None
    brightness_range = (0.0 * torch.pi, 1.0 * torch.pi)

    learning_rate = 1e-2
    num_epochs = 100

    if not torch.cuda.is_available():
        raise RuntimeError(
            "train_qcnn.py requires CUDA, but torch.cuda.is_available() is False."
        )

    torch.set_float32_matmul_precision("high")
    execution_device = torch.device("cuda")
    return (
        artifacts_root,
        batch_size,
        brightness_range,
        data_root,
        execution_device,
        feature_qubits,
        image_size,
        learning_rate,
        num_classes,
        num_epochs,
        project_root,
        quantum_layers,
        samples_per_class,
        seed,
        shot_budget,
    )


@app.cell
def _(
    AutoTrainingConfig,
    ModelSpec,
    MnistDatasetConfig,
    OptimizerConfig,
    OutputConfig,
    SeedConfig,
    TrainingConfig,
    artifacts_root,
    batch_size,
    brightness_range,
    data_root,
    execution_device,
    feature_qubits,
    image_size,
    learning_rate,
    num_classes,
    num_epochs,
    quantum_layers,
    samples_per_class,
    seed,
    shot_budget,
):
    "Automatic training config"
    auto_training_config = AutoTrainingConfig(
        dataset=MnistDatasetConfig(
            root=data_root,
            samples_per_class=samples_per_class,
            image_size=image_size,
            train_batch_size=batch_size,
            test_batch_size=16000,
            pin_memory=True,
            download=True,
        ),
        model=ModelSpec(
            module="qcnn.hybrid",
            class_name="PCSQCNN",
            constructor_kwargs={
                "image_size": image_size,
                "num_classes": num_classes,
                "feature_qubits": feature_qubits,
                "quantum_layers": quantum_layers,
                "brightness_range": brightness_range,
                "shot_budget": shot_budget,
                "multiplexer_init_scale": 0.05,
            },
        ),
        optimizer=OptimizerConfig(
            kind="adam",
            learning_rate=learning_rate,
        ),
        training=TrainingConfig(
            num_epochs=num_epochs,
            device=execution_device,
        ),
        seeds=SeedConfig(
            base_seed=seed,
            seed_count=1,
        ),
        output=OutputConfig(
            root=artifacts_root,
        ),
    )
    model_label = "PCS-QCNN"
    return auto_training_config, model_label


@app.cell
def _():
    # "Classical CNN model"
    # ModelSpec(module="qcnn.classic", class_name="ClassicalCNN", constructor_kwargs={"image_size": image_size, "num_classes": num_classes})
    return


@app.cell
def _():
    # "Classical MLP model"
    # ModelSpec(module="qcnn.classic", class_name="ClassicalMLP", constructor_kwargs={"image_size": image_size, "num_classes": num_classes})
    return


@app.cell
def _(auto_training_config, mo, model_label, run_mnist_auto_training):
    "Training"

    def progress_factory(epoch_range):
        return mo.status.progress_bar(epoch_range, title=f"Training {model_label}")

    auto_training_result = run_mnist_auto_training(
        auto_training_config,
        progress_factory=progress_factory,
    )
    seed_run = auto_training_result.runs[0]
    history = seed_run.history
    runner = seed_run.runner
    test_loader = seed_run.test_loader
    return auto_training_result, history, runner, seed_run, test_loader


@app.cell
def _(history, mo, model_label, plot_convergence):
    "Results"
    figure1, axes1 = plot_convergence(history)

    summary = (
        f"{model_label}: train loss {history.train_loss[-1]:.6f}, "
        f"test loss {history.test_loss[-1]:.6f}, "
        f'train acc {history.train_metrics["accuracy"][-1]:.4%}, '
        f'test acc {history.test_metrics["accuracy"][-1]:.4%}'
    )
    results_view = mo.vstack([mo.md(summary), mo.ui.matplotlib(axes1[0])])
    results_view
    return


@app.cell
def _(mo, plot_error_analysis, runner, test_loader):
    "Error analysis"
    error_ax = plot_error_analysis(runner, test_loader)
    error_view = mo.ui.matplotlib(error_ax)
    error_view
    return


@app.cell
def _(auto_training_result, mo, seed_run):
    "Saved files"
    checkpoint_path = seed_run.artifacts.final_checkpoint_path
    result_path = seed_run.artifacts.result_path
    results_dir = auto_training_result.output_directory
    mo.md(
        f"""
        ## Saved Outputs

        Results directory: `{results_dir}`

        Final checkpoint: `{checkpoint_path}`

        Result payload: `{result_path}`
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
