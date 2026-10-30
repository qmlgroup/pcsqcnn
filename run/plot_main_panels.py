"""Render compact main-text panel figures from existing saved summaries.

Component plots remain available through their original standalone scripts.
Shared canvases control gutters without changing numerical aggregation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import plot_finite_shot_loss_histograms as shots
import plot_pcsqcnn_architecture_sweep as architecture
import plot_pcsqcnn_image_size_sweep as sizes
import plot_translated_mnist_baselines as baselines
from qcnn.matplotlib_config import MANUSCRIPT_TEXT_WIDTH_IN, configure_manuscript_plot_fonts

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _canvas(*, height: float, bottom: float, top: float = 0.92):
    import matplotlib
    configure_manuscript_plot_fonts(matplotlib)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(MANUSCRIPT_TEXT_WIDTH_IN, height))
    figure.subplots_adjust(left=0.115, right=0.99, bottom=bottom, top=top, wspace=0.34)
    return figure, axes


def _label_panels(axes) -> None:
    import matplotlib
    for letter, ax in zip('ab', axes, strict=True):
        ax.annotate(f'({letter})', (0, 1), xycoords='axes fraction',
                    xytext=(-27, 0), textcoords='offset points',
                    ha='right', va='top', fontweight='bold',
                    fontsize=1.125 * matplotlib.rcParams['font.size'], clip_on=False)
        ax.set_xlabel('')


def _shared_xlabel(figure, text: str, y: float) -> None:
    import matplotlib
    figure.text(0.5525, y, text, ha='center', va='bottom',
                fontsize=matplotlib.rcParams['font.size'])


def _top_legend(figure, axes, handles, labels=None, **kwargs):
    from matplotlib.transforms import ScaledTranslation

    # One header row, three points above the axes. Panel labels stay beside them.
    transform = figure.transFigure + ScaledTranslation(0, 3 / 72, figure.dpi_scale_trans)
    return figure.legend(
        handles=handles, labels=labels, loc='lower center',
        bbox_to_anchor=(0.5525, axes[0].get_position().y1), bbox_transform=transform,
        frameon=False, borderaxespad=0, borderpad=0, **kwargs)


def plot_sweeps(artifacts_root: Path):
    figure, axes = _canvas(height=2.2, bottom=0.18, top=0.97)
    architecture.plot_article_figure_5a(
        sweep_root=artifacts_root / architecture.build_sweep_directory_name(), ax=axes[0])
    sizes.plot_article_figure_5b(
        sweep_root=artifacts_root / sizes.DEFAULT_SWEEP_DIRECTORY_NAME, ax=axes[1])
    _label_panels(axes)
    _shared_xlabel(figure, 'Epoch', 0.025)
    from matplotlib.legend import Legend
    # Stack depth (color) and width (line style) in the unused lower area.
    for ax in axes:
        legends = [child for child in ax.get_children() if isinstance(child, Legend)]
        entries = [(old.legend_handles, [text.get_text() for text in old.get_texts()])
                   for old in legends]
        for old in legends:
            old.remove()
        for index, (handles, labels) in enumerate(entries):
            feature_key = labels[0].startswith('$n_f')
            legend = ax.legend(
                handles, labels, loc='lower right',
                bbox_to_anchor=(0.98, 0 if feature_key else (0.12 if ax is axes[0] else 0)),
                ncols=3 if ax is axes[0] else 2,
                frameon=False, handlelength=1.4, handletextpad=0.4,
                columnspacing=0.7, labelspacing=0.25, borderaxespad=0.5)
            if index < len(entries) - 1:
                ax.add_artist(legend)
    return figure


def plot_baselines(artifacts_root: Path):
    classical, _ = baselines._load_available_classical_series_configs(
        artifacts_root=artifacts_root,
        scaled_image_size=baselines.DEFAULT_TRANSLATED_CLASSICAL_SCALED_IMAGE_SIZE)
    quantum, _ = baselines._load_available_quantum_series_configs(
        pcsqcnn_run_directory=baselines._resolve_architecture_sweep_run_directory(artifacts_root),
        pcsqcnn_no_qft_run_directory=baselines._resolve_fixed_run_directory(
            artifacts_root, base_name='pcsqcnn_no_qft',
            samples_per_class=baselines.DEFAULT_SAMPLES_PER_CLASS))
    figure, axes = _canvas(height=2.1, bottom=0.18, top=0.90)
    baselines.plot_article_figure_2a(classical_series_configs=classical, ax=axes[0])
    baselines.plot_article_figure_2b(quantum_series_configs=quantum, ax=axes[1])
    _label_panels(axes)
    _shared_xlabel(figure, 'Epoch', 0.025)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        model_keys = [Patch(facecolor=handle.get_color(), label=label)
                      for handle, label in zip(handles, labels, strict=True)
                      if not label.startswith('Train ')]
        ax.legend(handles=model_keys, loc='center right', bbox_to_anchor=(0.98, 0.62),
                  ncols=2, frameon=False, handlelength=0.8, handletextpad=0.4,
                  columnspacing=0.8, borderaxespad=0)
    _top_legend(figure, axes,
                [Line2D([], [], color='0.25', linestyle='-', label='Test'),
                 Line2D([], [], color='0.25', linestyle='--', label='Train')],
                ncols=2, handlelength=2, columnspacing=1.4)
    return figure


def plot_histograms(artifacts_root: Path):
    panels = shots.summarize_finite_shot_loss_sampling_payload(
        artifacts_root / shots.DEFAULT_REFERENCE_DIRECTORY_NAME / shots.DEFAULT_PAYLOAD_FILENAME)
    panels = sorted(panels, key=lambda panel: panel.epoch)
    if [panel.epoch for panel in panels] != list(shots.DEFAULT_REFERENCE_EPOCHS):
        raise ValueError('The paired histogram requires the 100- and 800-epoch panels.')
    # A common legend requires identical shot budgets and color assignments.
    budgets = [[series.shot_budget for series in panel.series_by_budget] for panel in panels]
    if budgets[0] != budgets[1]:
        raise ValueError('Histogram panels must have the same ordered shot budgets.')
    figure, axes = _canvas(height=2.1, bottom=0.18, top=0.90)
    for ax, panel in zip(axes, panels, strict=True):
        shots.plot_article_figure_7_panel(panel=panel, ax=ax)
    _label_panels(axes)
    _shared_xlabel(figure, 'Batch-mean test cross-entropy (nats/sample)', 0.025)
    handles, labels = axes[0].get_legend_handles_labels()
    _top_legend(figure, axes, handles, labels,
                ncols=len(handles), handlelength=1.4,
                handletextpad=0.4, columnspacing=1.0)
    return figure


PLOTS = {
    'sweeps': (plot_sweeps, 'pcsqcnn_sweeps.pdf'),
    'baselines': (plot_baselines, 'translated_mnist_comparison.pdf'),
    'histograms': (plot_histograms, 'finite_shot_loss_histograms.pdf'),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root', type=Path, default=PROJECT_ROOT / 'artifacts')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'figs')
    parser.add_argument('--figure', choices=['all', *PLOTS], default='all')
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = PLOTS if args.figure == 'all' else {args.figure: PLOTS[args.figure]}
    for plot, filename in selected.values():
        figure = plot(args.artifacts_root.expanduser().resolve())
        path = output_dir / filename
        figure.savefig(path)
        plt.close(figure)
        print(path)


if __name__ == '__main__':
    main()
