"""Plotly renderers for the per-scan processing pipeline (interactive twin of ``plotting``).

Mirrors the four matplotlib figures in :mod:`xasbatch.plotting` but returns
Plotly ``go.Figure`` objects so a Streamlit/browser front-end gets native
zoom / pan / hover. Both modules call the same :func:`xasbatch.process.process_scans`,
so the interactive view is byte-for-byte the same processing as the PNGs — only the
drawing differs. Kept separate so nothing here imports matplotlib.

Palette is intentionally identical to ``plotting`` (Okabe–Ito accents; muted gray
scans, bold black merge, faint-red QC-excluded). Keep the two in sync if you retune.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- palette (mirror of plotting.py; strings only, no matplotlib) -------------
_SCAN = "#999999"   # muted gray — individual scans
_EXCL = "#C0392B"   # faint red — QC-excluded scans
_MERGED = "#000000"  # bold black — the E-space merge
_C_PRE, _C_POST, _C_FLAT, _C_BKG = "#0072B2", "#D55E00", "#13396B", "#CC3311"
_GRID = "#B8B8B8"   # reference guide lines (e0, y=1, y=0)


def _pass_mask(n, scan_pass):
    if scan_pass is None:
        return np.ones(n, dtype=bool)
    return np.asarray(scan_pass, dtype=bool)


def _axref(idx: int, axis: str) -> str:
    """Domain-relative axis ref for subplot ``idx`` (1-based), e.g. ``'x domain'``.

    make_subplots numbers axes in row-major order; subplot 1 uses the bare
    ``x``/``y`` names, the rest are suffixed.
    """
    base = axis if idx == 1 else f"{axis}{idx}"
    return f"{base} domain"


def _overlay(fig, x_cols, y_cols, keep, names, *, merged_xy=None,
             merged_name="merged (avg)", row=None, col=None,
             legend="legend", show_scan_legend=True):
    """Add a bundle of per-scan line traces (one legend entry each for pass/excl/merge).

    ``x_cols``/``y_cols`` are lists of 1-D arrays (one per scan). Scans share a
    single legend group so N scans collapse to one legend row. ``legend`` picks
    which legend the entries attach to (``"legend"``/``"legend2"``) — for subplots
    with their own legends; ``show_scan_legend=False`` suppresses the scan/excluded
    entries entirely (e.g. when a sibling subplot already labels them).
    """
    n_pass = int(keep.sum())
    n_excl = int((~keep).sum())
    first_pass = first_excl = True
    for j, (x, y, nm) in enumerate(zip(x_cols, y_cols, names)):
        if keep[j]:
            show, first_pass = first_pass, False
            color, grp, label = _SCAN, "scan", f"scans (n={n_pass})"
        else:
            show, first_excl = first_excl, False
            color, grp, label = _EXCL, "excl", f"excluded (n={n_excl})"
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines", name=label, legendgroup=grp, legend=legend,
                showlegend=bool(show and show_scan_legend),
                line=dict(color=color, width=1.0),
                opacity=0.75 if keep[j] else 0.6, hovertext=nm, hoverinfo="text+x+y",
            ),
            row=row, col=col,
        )
    if merged_xy is not None:
        mx, my = merged_xy
        fig.add_trace(
            go.Scatter(x=mx, y=my, mode="lines", name=merged_name, legendgroup="merged",
                       legend=legend, line=dict(color=_MERGED, width=2.4)),
            row=row, col=col,
        )


# --------------------------------------------------------------------------- 1
def fig_raw(energy, scan_mu, names, merged_mu=None, scan_pass=None) -> go.Figure:
    """Raw summed μ(E) (= Σ FF/I0) per scan, with the E-space merge (mean μ)."""
    keep = _pass_mask(scan_mu.shape[1], scan_pass)
    fig = go.Figure()
    _overlay(
        fig,
        [energy] * scan_mu.shape[1],
        [scan_mu[:, j] for j in range(scan_mu.shape[1])],
        keep, names,
        merged_xy=(energy, merged_mu) if merged_mu is not None else None,
    )
    fig.update_layout(
        title="Raw summed scans",
        xaxis_title="Energy (eV)", yaxis_title="μ (summed FF/I0)",
    )
    return _style(fig)


# --------------------------------------------------------------------------- 2
def fig_norm_fits(energy, groups, names, e0s, e0_merged=None, scan_pass=None) -> go.Figure:
    """Grid: each scan's μ(E) with pre/post-edge fits (left) and flattened μ (right)."""
    n = len(groups)
    e0s = np.atleast_1d(np.asarray(e0s, dtype=float))
    keep = _pass_mask(n, scan_pass)
    # Size rows in pixels with a small fixed inter-row gap. plotly's vertical_spacing
    # is a fraction of the *total* height per gap, so deriving it from pixel targets
    # keeps the gap tight (and constant) no matter how many scans there are. The header
    # band (top_px) is reserved in the top margin for the title + E0 stats + column heads.
    row_px, gap_px, top_px = 320, 46, 132
    height = n * row_px + max(n - 1, 0) * gap_px + top_px
    fig = make_subplots(
        rows=n, cols=2, shared_xaxes=False, horizontal_spacing=0.09,
        vertical_spacing=(gap_px / height) if n > 1 else 0.0,
    )
    for j, (g, name) in enumerate(zip(groups, names)):
        r = j + 1
        e0j = float(e0s[j])
        left_idx = (j * 2) + 1  # subplot index of this row's left panel (annotations)
        fig.add_trace(go.Scatter(x=energy, y=g.mu, mode="lines", name="μ(E)",
                                 line=dict(color=_MERGED, width=1.3), showlegend=False),
                      row=r, col=1)
        fig.add_trace(go.Scatter(x=energy, y=g.pre_edge, mode="lines", name="pre-edge",
                                 line=dict(color=_C_PRE, width=1.3, dash="dash"), showlegend=False),
                      row=r, col=1)
        fig.add_trace(go.Scatter(x=energy, y=g.post_edge, mode="lines", name="post-edge",
                                 line=dict(color=_C_POST, width=1.3, dash="dash"), showlegend=False),
                      row=r, col=1)
        fig.add_vline(x=e0j, line=dict(color="#A6A6A6", width=1.0, dash="dot"), row=r, col=1)
        d = getattr(g, "pre_edge_details", None)
        if d is not None:
            for x in (e0j + d.pre1, e0j + d.pre2):
                fig.add_vline(x=x, line=dict(color=_C_PRE, width=0.8), opacity=0.5, row=r, col=1)
            for x in (e0j + d.norm1, e0j + d.norm2):
                fig.add_vline(x=x, line=dict(color=_C_POST, width=0.8), opacity=0.5, row=r, col=1)
        # guard against a missing member name — a None annotation renders as "undefined"
        nm = name or "(unnamed)"
        label = nm if keep[j] else f"{nm} — EXCLUDED"
        fig.add_annotation(
            text=label, xref=_axref(left_idx, "x"), yref=_axref(left_idx, "y"),
            x=0.02, y=0.97, xanchor="left", yanchor="top", showarrow=False,
            font=dict(size=12, color=_MERGED if keep[j] else _EXCL),
        )
        fig.add_annotation(
            text=f"E₀ = {e0j:.2f} eV<br>Δμ₀ = {g.edge_step:.3f}",
            xref=_axref(left_idx, "x"), yref=_axref(left_idx, "y"),
            x=0.66, y=0.30, xanchor="left", yanchor="middle", showarrow=False,
            font=dict(size=11),
        )
        fig.add_trace(go.Scatter(x=energy, y=g.flat, mode="lines", name="flattened μ(E)",
                                 line=dict(color=_C_FLAT, width=1.4), showlegend=False),
                      row=r, col=2)
        fig.add_hline(y=1.0, line=dict(color=_GRID, width=0.8, dash="dot"), row=r, col=2)
        if r == n:
            fig.update_xaxes(title_text="Energy (eV)", row=r, col=1)
            fig.update_xaxes(title_text="Energy (eV)", row=r, col=2)

    kept = e0s[keep]
    n_excl = int((~keep).sum())
    stats = (f"⟨E₀⟩ = {kept.mean():.2f} ± {kept.std():.2f} eV  "
             f"(per scan, n={int(keep.sum())}" + (f", {n_excl} excl.)" if n_excl else ")"))
    if e0_merged is not None:
        stats += f"      E₀ merged = {e0_merged:.2f} eV"

    fig = _style(fig, grid=True)
    H = max(height, 380)

    def _above(px: float) -> float:  # paper-y this many px above the plotting area
        return 1.0 + px / H

    fig.update_layout(height=H, margin=dict(l=60, r=20, t=top_px, b=50))
    fig.add_annotation(text="Normalization per scan", xref="paper", yref="paper",
                       x=0.5, y=_above(104), xanchor="center", yanchor="middle",
                       showarrow=False, font=dict(size=17))
    fig.add_annotation(text=stats, xref="paper", yref="paper",
                       x=0.5, y=_above(64), xanchor="center", yanchor="middle",
                       showarrow=False, font=dict(size=14, color=_MERGED))
    fig.add_annotation(text="pre / post-edge fit", xref="paper", yref="paper",
                       x=0.21, y=_above(20), xanchor="center", yanchor="middle",
                       showarrow=False, font=dict(size=13))
    fig.add_annotation(text="flattened μ(E)", xref="paper", yref="paper",
                       x=0.79, y=_above(20), xanchor="center", yanchor="middle",
                       showarrow=False, font=dict(size=13))
    return fig


# --------------------------------------------------------------------------- 3
def fig_flat_overlay(energy, groups, names, merged=None, scan_pass=None) -> go.Figure:
    """All flattened/normalized scans overlaid, with the merged (E-space) spectrum."""
    keep = _pass_mask(len(groups), scan_pass)
    fig = go.Figure()
    _overlay(
        fig, [energy] * len(groups), [g.flat for g in groups], keep, names,
        merged_xy=(energy, merged.flat) if merged is not None else None,
    )
    fig.add_hline(y=1.0, line=dict(color=_GRID, width=0.8, dash="dot"))
    fig.update_layout(title="Flattened scans",
                      xaxis_title="Energy (eV)", yaxis_title="flattened μ(E)")
    return _style(fig)


# --------------------------------------------------------------------------- 4
def fig_exafs(energy, groups, names, e0, merged=None, kweight=3, scan_pass=None) -> go.Figure:
    """Left: normalized μ + AUTOBK background per scan. Right: kⁿ·χ(k) per scan."""
    keep = _pass_mask(len(groups), scan_pass)
    above = energy >= e0
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.1,
        subplot_titles=["Normalized μ + AUTOBK spline", f"EXAFS  k^{kweight}·χ(k)"],
    )

    def _norm_bkg(g):
        return (g.bkg - g.pre_edge) / g.edge_step

    # Left subplot has its own legend (merged μ + bkg); the shared scan/excluded
    # entries are shown only by the right subplot's legend to avoid duplicates.
    _overlay(
        fig, [energy[above]] * len(groups), [g.norm[above] for g in groups], keep, names,
        merged_xy=(energy[above], merged.norm[above]) if merged is not None else None,
        merged_name="merged μ", row=1, col=1, legend="legend", show_scan_legend=False,
    )
    if merged is not None:
        fig.add_trace(go.Scatter(x=energy[above], y=_norm_bkg(merged)[above], mode="lines",
                                 name="merged bkg", legend="legend",
                                 line=dict(color=_C_BKG, width=1.8, dash="dash")),
                      row=1, col=1)
    fig.update_yaxes(range=[0.6, None], title_text="normalized μ", row=1, col=1)
    fig.update_xaxes(title_text="Energy (eV)", row=1, col=1)

    chi_cols = [g.k**kweight * g.chi for g in groups]
    merged_chi = merged.k**kweight * merged.chi if merged is not None else None
    _overlay(
        fig, [g.k for g in groups], chi_cols, keep, names,
        merged_xy=(merged.k, merged_chi) if merged is not None else None,
        row=1, col=2, legend="legend2",
    )
    fig.add_hline(y=0.0, line=dict(color=_GRID, width=0.8, dash="dot"), row=1, col=2)
    fig.update_xaxes(title_text="k (Å⁻¹)", row=1, col=2)
    fig.update_yaxes(title_text=f"k^{kweight}·χ(k)", row=1, col=2)

    # χ(k) legend goes on the LEFT of the right panel (near k=0, where the curve is
    # small). Upper vs lower left tracks the swing: |max| > |min| → data leans up →
    # upper-left, else lower-left.
    stacked = chi_cols + ([merged_chi] if merged_chi is not None else [])
    allv = (np.concatenate([np.asarray(v, float).ravel() for v in stacked])
            if stacked else np.array([0.0]))
    leans_up = abs(np.nanmax(allv)) > abs(np.nanmin(allv))
    leg2_y, leg2_yanchor = (0.98, "top") if leans_up else (0.02, "bottom")
    x0_right = fig.layout.xaxis2.domain[0]  # left edge of the right subplot (paper x)

    legbg = "rgba(255,255,255,0.7)"
    fig.update_layout(
        title="",
        legend=dict(x=0.45, y=0.02, xanchor="right", yanchor="bottom",
                    bgcolor=legbg, font=dict(size=11)),
        legend2=dict(x=x0_right + 0.015, y=leg2_y, xanchor="left", yanchor=leg2_yanchor,
                     bgcolor=legbg, font=dict(size=11)),
    )
    return _style(fig, grid=True)


# --------------------------------------------------------------------------- style
def _style(fig: go.Figure, grid: bool = False) -> go.Figure:
    """Shared cosmetic pass: serif font, clean axes, comfortable margins."""
    fig.update_layout(
        template="simple_white",
        font=dict(family="Times New Roman, Times, serif", size=13),
        margin=dict(l=60, r=20, t=70, b=50),
        hovermode="closest",
        legend=dict(bgcolor="rgba(255,255,255,0.6)"),
    )
    fig.update_xaxes(showgrid=grid, gridcolor="#EDEDED", zeroline=False, ticks="outside")
    fig.update_yaxes(showgrid=grid, gridcolor="#EDEDED", zeroline=False, ticks="outside")
    return fig


def figure_report_plotly(bcr, params, kweight=3) -> list[tuple[str, go.Figure]]:
    """Build all four interactive figures for one file. Returns [(label, Figure), ...].

    Same processing as :func:`xasbatch.plotting.figure_report` (via ``process_scans``),
    rendered with Plotly for in-browser zoom/pan/hover.
    """
    from xasbatch.process import process_scans

    e0_merged, names, scan_mu, groups, scan_e0s, merged, scan_pass = process_scans(bcr, params)
    energy = bcr.energy
    return [
        ("1_raw", fig_raw(energy, scan_mu, names, merged_mu=merged.mu, scan_pass=scan_pass)),
        ("2_norm_fits", fig_norm_fits(energy, groups, names, scan_e0s,
                                      e0_merged=e0_merged, scan_pass=scan_pass)),
        ("3_flat", fig_flat_overlay(energy, groups, names, merged=merged, scan_pass=scan_pass)),
        ("4_exafs", fig_exafs(energy, groups, names, e0_merged, merged=merged,
                              kweight=kweight, scan_pass=scan_pass)),
    ]
