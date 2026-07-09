"""Smoke tests for the plotting layer (headless Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xasbatch.io import load_combined_bcr  # noqa: E402
from xasbatch.model import Params  # noqa: E402
from xasbatch.plotting import figure_report  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "sample_small.bcr.combined"
N_SCANS = 2  # fixture has 2 scans


def test_figure_report_builds_four_figures():
    bcr = load_combined_bcr(FIXTURE)
    figs = dict(figure_report(bcr, Params(), kweight=3))
    try:
        assert list(figs) == ["1_raw", "2_norm_fits", "3_flat", "4_exafs"]
        # per-scan normalization grid: 2 columns (fit | flattened) per scan
        assert len(figs["2_norm_fits"].axes) == 2 * N_SCANS
        # exafs view: normalized-μ+spline and kⁿ·χ(k)
        assert len(figs["4_exafs"].axes) == 2
        # raw / flat overlays are single-axes
        assert len(figs["1_raw"].axes) == 1
        assert len(figs["3_flat"].axes) == 1
    finally:
        plt.close("all")
