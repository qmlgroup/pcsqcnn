import matplotlib

matplotlib.use("Agg")

import qcnn.visualization as visualization_module

from qcnn import configure_matplotlib_pdf_fonts

from .script_loading import load_run_script

PLOT_SCRIPT_FILENAMES = (
    "plot_brightness_sweep.py",
    "plot_classical_baseline_architectures.py",
    "plot_error_structure.py",
    "plot_finite_shot_accuracy.py",
    "plot_finite_shot_loss_histograms.py",
    "plot_gradient_norms.py",
    "plot_pcsqcnn_architecture_sweep.py",
    "plot_pcsqcnn_image_size_sweep.py",
    "plot_readout_entropy.py",
    "plot_readout_landscape.py",
    "plot_translated_mnist_baselines.py",
)


def _assert_type42_fonts_configured() -> None:
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42


def test_configure_matplotlib_pdf_fonts_sets_type42_font_embedding() -> None:
    previous_pdf_fonttype = matplotlib.rcParams["pdf.fonttype"]
    previous_ps_fonttype = matplotlib.rcParams["ps.fonttype"]
    try:
        matplotlib.rcParams["pdf.fonttype"] = 3
        matplotlib.rcParams["ps.fonttype"] = 3

        configure_matplotlib_pdf_fonts(matplotlib)

        _assert_type42_fonts_configured()
    finally:
        matplotlib.rcParams["pdf.fonttype"] = previous_pdf_fonttype
        matplotlib.rcParams["ps.fonttype"] = previous_ps_fonttype


def test_plot_entrypoints_configure_type42_font_embedding_before_pyplot() -> None:
    previous_pdf_fonttype = matplotlib.rcParams["pdf.fonttype"]
    previous_ps_fonttype = matplotlib.rcParams["ps.fonttype"]
    try:
        for index, filename in enumerate(PLOT_SCRIPT_FILENAMES):
            module = load_run_script(f"plot_font_config_{index}", filename)
            matplotlib.rcParams["pdf.fonttype"] = 3
            matplotlib.rcParams["ps.fonttype"] = 3

            module._require_matplotlib()

            _assert_type42_fonts_configured()

        matplotlib.rcParams["pdf.fonttype"] = 3
        matplotlib.rcParams["ps.fonttype"] = 3

        visualization_module._require_matplotlib()

        _assert_type42_fonts_configured()
    finally:
        matplotlib.rcParams["pdf.fonttype"] = previous_pdf_fonttype
        matplotlib.rcParams["ps.fonttype"] = previous_ps_fonttype
