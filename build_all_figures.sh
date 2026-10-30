#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

REBUILD_FLAG=
PLOTS_ONLY=
VERIFY=

usage() {
    printf 'Usage: sh build_all_figures.sh [--rebuild | --plots-only | --verify]\n' >&2
}

case $# in
    0)
        ;;
    1)
        if [ "$1" = "--rebuild" ]; then
            REBUILD_FLAG=--rebuild
        elif [ "$1" = "--plots-only" ]; then
            PLOTS_ONLY=1
        elif [ "$1" = "--verify" ]; then
            PLOTS_ONLY=1
            VERIFY=1
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
    shift 2
    if [ -n "$REBUILD_FLAG" ]; then
        run_step "$description" uv run python "$script_path" "$@" "$REBUILD_FLAG"
    else
        run_step "$description" uv run python "$script_path" "$@"
    fi
}

printf '==> Starting full figure build in %s\n' "$SCRIPT_DIR"

if [ -n "$VERIFY" ]; then
    run_step "Recompute and verify PCS translation diagnostics from final weights" \
        uv run python run/verify_pcs_translation.py
fi

if [ -z "$PLOTS_ONLY" ]; then
printf '==> Data and artifact generation\n'
run_python_step "Prepare translated-MNIST baselines data" run/prepare_translated_mnist_baselines_data.py
run_python_step "Prepare full-MNIST classical baselines data" run/prepare_full_mnist_classical_baselines_data.py
run_python_step "Prepare brightness sweep data" run/prepare_brightness_sweep_data.py
run_python_step "Prepare PCS-QCNN architecture sweep data" run/prepare_pcsqcnn_architecture_sweep_data.py
run_python_step "Prepare PCS-QCNN image-size sweep data" run/prepare_pcsqcnn_image_size_sweep_data.py
# Published learning rate and initialization scale.
run_python_step "Prepare tuned QCS-QCNN data" run/prepare_qcsqcnn_translated_mnist_data.py \
    --runs fq2_ql3_sl8 --learning-rate 0.001 --generator-init-scale 0.5 --root-directory qcsqcnn_translated_mnist_tuned
run_python_step "Continue HKP x1 with FRQI to 6000 epochs" run/prepare_encoding_swap_data.py --runs hur_su4_l1_frqi_train8 hur_su4_l1_frqi_train8_e6000_warm
run_python_step "Prepare sp-QCNN data" run/prepare_sp_qcnn_translated_mnist_data.py --runs sp_qcnn_train8 --seed-count 2
# The warm-restart stages continue the 2000-epoch run and must follow it.
run_python_step "Prepare local PCS-QCNN data" run/prepare_local_pcsqcnn_translated_mnist_data.py \
    --runs pcsqcnn_local_fq2_ql3_train8 pcsqcnn_local_fq2_ql3_train8_e4000_warm pcsqcnn_local_fq2_ql3_train8_e6000_warm
run_python_step "Evaluate gradient norms" run/evaluate_gradient_norms.py
run_python_step "Evaluate readout entropy" run/evaluate_readout_entropy.py
run_python_step "Evaluate finite-shot accuracy snapshots" run/evaluate_finite_shot_snapshots.py
run_python_step "Evaluate finite-shot loss sampling" run/evaluate_finite_shot_loss_sampling.py
run_python_step "Evaluate readout landscape" run/evaluate_readout_landscape.py
run_python_step "Evaluate error structure" run/evaluate_error_structure.py
fi

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
run_step "Render compact main-text panel figures" uv run python run/plot_main_panels.py
run_step "Render readout landscape" uv run python run/plot_readout_landscape.py
run_step "Render error structure" uv run python run/plot_error_structure.py

run_step "Render common FRQI comparison" uv run python run/plot_frqi_comparison.py
run_step "Render PCS translation results" uv run python run/plot_pcs_translation.py --output-root results/pcs_translation_2026-10-25 --figures-dir figs
run_step "Render MFA comparison" uv run python run/plot_mfa_pcs_comparison.py \
    --curves results/mfa_translated/comparison/comparison_learning_curves.csv --output figs/pcs_mfa_translated_comparison.pdf

run_step "Render quantum diagrams and the cross illustration" uv run python run/render_paper_diagrams.py

printf '==> Full figure build complete\n'
