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
