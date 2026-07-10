"""Smoke + behavior tests for the Plotly rendering layer.

Skipped entirely when plotly isn't installed (it lives in the optional ``viewer``
extra, not in the default CI sync). Larch/numpy come in via the core deps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("plotly")  # optional dep; skip the whole module without it

from xasbatch.io import load_combined_bcr  # noqa: E402
from xasbatch.model import Params  # noqa: E402
from xasbatch.plotlyplots import fig_norm_fits, figure_report_plotly  # noqa: E402
from xasbatch.process import process_scans  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "sample_small.bcr.combined"
N_SCANS = 2  # fixture has 2 scans


@pytest.fixture(scope="module")
def scans():
    bcr = load_combined_bcr(FIXTURE)
    e0m, names, scan_mu, groups, e0s, merged, scan_pass = process_scans(bcr, Params())
    return dict(bcr=bcr, e0m=e0m, names=names, scan_mu=scan_mu, groups=groups,
                e0s=e0s, merged=merged, scan_pass=scan_pass)


def _xaxis_count(fig) -> int:
    return sum(k.startswith("xaxis") for k in fig.layout)


def test_report_builds_four_figures(scans):
    figs = dict(figure_report_plotly(scans["bcr"], Params(), kweight=3))
    assert list(figs) == ["1_raw", "2_norm_fits", "3_flat", "4_exafs"]
    # norm-fits grid: 2 columns (fit | flattened) per scan
    assert _xaxis_count(figs["2_norm_fits"]) == 2 * N_SCANS
    # exafs view: μ+spline and kⁿ·χ(k)
    assert _xaxis_count(figs["4_exafs"]) == 2
    # overlays are single-axes
    assert _xaxis_count(figs["1_raw"]) == 1
    assert _xaxis_count(figs["3_flat"]) == 1


def test_norm_fits_guards_missing_name(scans):
    """A None member name must never reach an annotation (renders as bold 'undefined')."""
    names = [None] + list(scans["names"][1:])
    fig = fig_norm_fits(scans["bcr"].energy, scans["groups"], names, scans["e0s"],
                        e0_merged=scans["e0m"], scan_pass=scans["scan_pass"])
    texts = [a.text for a in fig.layout.annotations]
    assert None not in texts
    assert "(unnamed)" in texts  # the guarded label
    assert "undefined" not in fig.to_json().lower()


def test_exafs_legend_left_and_follows_swing(scans):
    figs = dict(figure_report_plotly(scans["bcr"], Params(), kweight=3))
    fig = figs["4_exafs"]

    # χ(k) legend is anchored to the LEFT of the right subplot
    leg2 = fig.layout.legend2
    assert leg2.xanchor == "left"
    assert leg2.x == pytest.approx(fig.layout.xaxis2.domain[0] + 0.015, abs=1e-6)

    # upper vs lower tracks the swing: |max| > |min| -> upper (top), else lower (bottom)
    kw = 3
    allv = np.concatenate(
        [np.ravel(g.k**kw * g.chi) for g in scans["groups"]]
        + [np.ravel(scans["merged"].k**kw * scans["merged"].chi)]
    )
    expected = "top" if abs(np.nanmax(allv)) > abs(np.nanmin(allv)) else "bottom"
    assert leg2.yanchor == expected


def test_exafs_no_duplicate_scan_legend(scans):
    """Scan/excluded entries appear once (right subplot only), not once per subplot."""
    fig = dict(figure_report_plotly(scans["bcr"], Params(), kweight=3))["4_exafs"]
    scan_entries = [t.name for t in fig.data
                    if t.showlegend and (t.name or "").startswith("scans (n=")]
    assert len(scan_entries) == 1
