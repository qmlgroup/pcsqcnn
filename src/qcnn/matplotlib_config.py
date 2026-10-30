"""Matplotlib configuration helpers for publication-ready vector outputs."""

from __future__ import annotations

from typing import Any

TRUETYPE_FONT_TYPE = 42


def configure_matplotlib_pdf_fonts(matplotlib_module: Any | None = None) -> None:
    """Configure Matplotlib PDF/PS backends to avoid Type 3 font embedding."""

    if matplotlib_module is None:
        import matplotlib as matplotlib_module

    matplotlib_module.rcParams["pdf.fonttype"] = TRUETYPE_FONT_TYPE
    matplotlib_module.rcParams["ps.fonttype"] = TRUETYPE_FONT_TYPE


# Match the 8-pt figure captions at the final manuscript size (31-pica text).
# Matplotlib uses PostScript points; TeX uses 72.27 points per inch.
MANUSCRIPT_TEXT_WIDTH_IN = 372.0 / 72.27
MANUSCRIPT_HALF_FIGSIZE = (0.49 * MANUSCRIPT_TEXT_WIDTH_IN, 2.6)
MANUSCRIPT_COMPACT_FIGSIZE = (0.82 * MANUSCRIPT_TEXT_WIDTH_IN, 2.8)


def configure_manuscript_plot_fonts(matplotlib_module: Any) -> None:
    """Use caption-size text on figures exported at their insertion width."""
    configure_matplotlib_pdf_fonts(matplotlib_module)
    size = 8.0 * 72.0 / 72.27
    matplotlib_module.rcParams.update({
        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
        "legend.title_fontsize": size,
    })
