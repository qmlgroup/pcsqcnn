import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import sys

    import marimo as mo
    import torch

    from qcnn import load_auto_training_run

    return Path, load_auto_training_run, mo, sys, torch


@app.cell
def _(Path, mo, torch):
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"
    run_directory = artifacts_dir / "pcsqcnn_reference_training"
    seed = 0
    execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download = True

    mo.md(
        f"""
        # Inspect PennyLane PCS-QCNN Run

        Run directory: `{run_directory}`
        Seed: `{seed}`
        Execution device: `{execution_device}`

        Download MNIST if missing: `{download}`
        """
    )
    return download, execution_device, project_root, run_directory, seed


@app.cell
def _(Path, sys):
    notebook_dir = Path(__file__).resolve().parent
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))

    from pcs_qcnn_reference import (
        PennyLanePCSQCNN,
        build_mnist_test_loader,
        evaluate_predictions,
        plot_confusion_matrix,
        prepare_mnist_test_dataset,
    )

    return (
        PennyLanePCSQCNN,
        build_mnist_test_loader,
        evaluate_predictions,
        plot_confusion_matrix,
        prepare_mnist_test_dataset,
    )


@app.cell
def _(execution_device, load_auto_training_run, mo, run_directory, seed):
    loaded_run = load_auto_training_run(
        run_directory,
        seed,
        map_location=execution_device,
    )
    mo.md(loaded_run.to_markdown())
    return (loaded_run,)


@app.cell
def _(PennyLanePCSQCNN, loaded_run):
    torch_model = loaded_run.model
    model = PennyLanePCSQCNN.from_torch_state(
        loaded_run.model_spec.constructor_kwargs,
        torch_model.state_dict(),
    )
    return (model,)


@app.cell
def _(
    build_mnist_test_loader,
    download,
    loaded_run,
    prepare_mnist_test_dataset,
    project_root,
):
    mnist_test_config = loaded_run.saved_mnist_test_config()
    test_loader_config = mnist_test_config["test_loader"]
    test_dataset = prepare_mnist_test_dataset(
        root=project_root / "data",
        image_size=mnist_test_config["image_size"],
        scaled_image_size=mnist_test_config["scaled_image_size"],
        max_offset=mnist_test_config["max_offset"],
        seed=mnist_test_config["seed"],
        download=download,
    )
    test_loader = build_mnist_test_loader(
        dataset=test_dataset,
        batch_size=test_loader_config["batch_size"],
        num_workers=test_loader_config["num_workers"],
        pin_memory=test_loader_config["pin_memory"],
        drop_last=test_loader_config["drop_last"],
    )
    return (test_loader,)


@app.cell
def _(evaluate_predictions, mo, model, test_loader):
    def progress_factory(batch_iterator):
        return mo.status.progress_bar(
            batch_iterator,
            title="Predicting labels",
            remove_on_exit=False,
        )

    evaluation_result = evaluate_predictions(
        model,
        test_loader,
        progress_factory=progress_factory,
    )
    accuracy_view = mo.md(f'Accuracy: `{evaluation_result["accuracy"]:.4%}`')
    accuracy_view
    return (evaluation_result,)


@app.cell
def _(evaluation_result, mo, plot_confusion_matrix):
    confusion_ax = plot_confusion_matrix(evaluation_result["confusion_matrix"])
    mo.md("## Confusion Matrix")
    confusion_view = mo.ui.matplotlib(confusion_ax)
    confusion_view
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
