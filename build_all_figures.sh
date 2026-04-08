#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

REBUILD_FLAG=

usage() {
    printf 'Usage: sh build_all_figures.sh [--rebuild]\n' >&2
}

case $# in
    0)
        ;;
    1)
        if [ "$1" = "--rebuild" ]; then
            REBUILD_FLAG=--rebuild
        else
            usage
            exit 1
        fi
        ;;
    *)
        usage
        exit 1
        ;;
esac

run_step() {
    description=$1
    shift
    printf '==> %s\n' "$description"
    "$@"
}

run_python_step() {
    description=$1
    script_path=$2
    if [ -n "$REBUILD_FLAG" ]; then
        run_step "$description" uv run python "$script_path" "$REBUILD_FLAG"
    else
        run_step "$description" uv run python "$script_path"
    fi
}

printf '==> Starting full figure build in %s\n' "$SCRIPT_DIR"

printf '==> Data and artifact generation\n'
run_python_step "Prepare translated-MNIST baselines data" run/prepare_translated_mnist_baselines_data.py
run_python_step "Prepare full-MNIST classical baselines data" run/prepare_full_mnist_classical_baselines_data.py
run_python_step "Prepare brightness sweep data" run/prepare_brightness_sweep_data.py
run_python_step "Prepare PCS-QCNN architecture sweep data" run/prepare_pcsqcnn_architecture_sweep_data.py
run_python_step "Prepare PCS-QCNN image-size sweep data" run/prepare_pcsqcnn_image_size_sweep_data.py
run_python_step "Evaluate gradient norms" run/evaluate_gradient_norms.py
run_python_step "Evaluate readout entropy" run/evaluate_readout_entropy.py
run_python_step "Evaluate finite-shot accuracy snapshots" run/evaluate_finite_shot_snapshots.py
run_python_step "Evaluate finite-shot loss sampling" run/evaluate_finite_shot_loss_sampling.py
run_python_step "Evaluate readout landscape" run/evaluate_readout_landscape.py
run_python_step "Evaluate error structure" run/evaluate_error_structure.py

printf '==> Figure rendering\n'
run_step "Render classical baseline architectures" uv run python run/plot_classical_baseline_architectures.py
run_step "Render translated-MNIST baselines" uv run python run/plot_translated_mnist_baselines.py
run_step "Render brightness sweep helper figures" uv run python run/plot_brightness_sweep.py
run_step "Render PCS-QCNN architecture sweep" uv run python run/plot_pcsqcnn_architecture_sweep.py
run_step "Render PCS-QCNN image-size sweep" uv run python run/plot_pcsqcnn_image_size_sweep.py
run_step "Render gradient norms" uv run python run/plot_gradient_norms.py
run_step "Render readout entropy" uv run python run/plot_readout_entropy.py
run_step "Render finite-shot accuracy" uv run python run/plot_finite_shot_accuracy.py
run_step "Render finite-shot loss histograms" uv run python run/plot_finite_shot_loss_histograms.py
run_step "Render readout landscape" uv run python run/plot_readout_landscape.py
run_step "Render error structure" uv run python run/plot_error_structure.py

printf '==> Full figure build complete\n'
