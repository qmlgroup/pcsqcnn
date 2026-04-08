import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import torch

    from qcnn import (
        evaluate_loaded_auto_training_run_on_saved_mnist_test,
        load_auto_training_run,
        plot_convergence,
        plot_error_analysis,
    )

    return (
        Path,
        evaluate_loaded_auto_training_run_on_saved_mnist_test,
        load_auto_training_run,
        mo,
        plot_convergence,
        plot_error_analysis,
        torch,
    )


@app.cell
def _(Path, mo, torch):
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"
    run_directory = artifacts_dir / "run_YYYYMMDD_HHMMSS"
    seed = 0
    execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configuration_view = mo.md(
        f"""
        # Inspect Auto-Training Run
        Run directory: `{run_directory}`
        Seed: `{seed}`
        Execution device: `{execution_device}`
        """
    )
    configuration_view
    return execution_device, project_root, run_directory, seed


@app.cell
def _(execution_device, load_auto_training_run, mo, run_directory, seed):
    loaded_run = load_auto_training_run(
        run_directory,
        seed,
        map_location=execution_device,
    )
    status_view = mo.md(
        f"Auto-training files found in `{run_directory}` for seed `{seed}`."
    )
    status_view
    return (loaded_run,)


@app.cell
def _(loaded_run, mo, plot_convergence):
    overview_blocks = [mo.md(loaded_run.to_markdown())]
    convergence_figure, convergence_axes = plot_convergence(loaded_run.training_history)
    overview_blocks.append(mo.ui.matplotlib(convergence_axes[0]))
    run_overview_view = mo.vstack(overview_blocks)
    run_overview_view
    return


@app.cell
def _(
    evaluate_loaded_auto_training_run_on_saved_mnist_test,
    execution_device,
    loaded_run,
    mo,
    plot_error_analysis,
    project_root,
):
    loaded_run_evaluation = evaluate_loaded_auto_training_run_on_saved_mnist_test(
        loaded_run,
        root=project_root / "data",
        device=execution_device,
        download=True,
    )

    evaluation_blocks = []
    if loaded_run_evaluation.context.notes:
        evaluation_blocks.append(
            mo.md(
                "## Re-evaluation Notes"
                + "\n".join(
                    f"- {note}"
                    for note in loaded_run_evaluation.context.notes
                )
            )
        )
    evaluation_blocks.append(mo.md(loaded_run_evaluation.report_markdown))
    error_analysis_ax = plot_error_analysis(
        loaded_run_evaluation.context.runner,
        loaded_run_evaluation.context.test_loader,
    )
    evaluation_blocks.append(mo.ui.matplotlib(error_analysis_ax))
    reconstructed_evaluation_view = mo.vstack(evaluation_blocks)

    reconstructed_evaluation_view
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
