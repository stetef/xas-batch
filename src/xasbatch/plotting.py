"""Matplotlib helpers to visualize the per-scan processing pipeline.

Kept separate from the core so `io`/`process` never import matplotlib. These
functions take the processed per-scan Larch groups (from
:func:`xasbatch.process.process_scans`) and draw:

1. raw summed μ(E) per scan + average          (`plot_raw`)
2. per-scan pre/post-edge fits + flattened      (`plot_norm_fits`)
3. all flattened scans + average                (`plot_flat_overlay`)
4. normalized μ + AUTOBK splines, and kⁿ·χ(k)   (`plot_exafs`)

`figure_report` builds all four as Figures. The functions are backend-agnostic,
so a Streamlit/notebook front-end can reuse them; only the CLI picks a backend.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

_AVG_KW = dict(color="black", lw=2.2, zorder=5)  # style for the "average" trace


def _scan_colors(n: int):
    return plt.get_cmap("viridis")(np.linspace(0.0, 0.92, max(n, 1)))


def plot_raw(energy, scan_mu, names, ax=None):
    """Raw summed μ(E) (= Σ FF/I0) for each scan, with the average across scans."""
    ax = ax or plt.gca()
    colors = _scan_colors(len(names))
    for j, name in enumerate(names):
        ax.plot(energy, scan_mu[:, j], color=colors[j], lw=0.9, alpha=0.85, label=name)
    ax.plot(energy, scan_mu.mean(axis=1), label="average", **_AVG_KW)
    ax.set(xlabel="Energy (eV)", ylabel="μ (summed FF/I0)", title="Raw summed scans")
    return ax


def plot_norm_fits(energy, groups, names, e0, fig=None):
    """Grid: each scan's μ(E) with pre/post-edge fits (left) and flattened μ (right)."""
    n = len(groups)
    fig = fig or plt.figure(figsize=(9.5, 2.1 * n + 0.5))
    axes = fig.subplots(n, 2, squeeze=False)
    for j, (g, name) in enumerate(zip(groups, names)):
        left, right = axes[j]
        left.plot(energy, g.mu, color="0.2", lw=1.0, label="μ(E)")
        left.plot(energy, g.pre_edge, "--", color="tab:blue", lw=1.0, label="pre-edge")
        left.plot(energy, g.post_edge, "--", color="tab:red", lw=1.0, label="post-edge")
        left.axvline(e0, color="0.6", lw=0.8, ls=":")
        left.set_ylabel(name, fontsize=8)
        right.plot(energy, g.flat, color="tab:green", lw=1.0)
        right.axhline(1.0, color="0.7", lw=0.8, ls=":")
        if j == 0:
            left.set_title("pre/post-edge fit")
            right.set_title("flattened μ(E)")
            left.legend(fontsize=7, loc="lower right")
        if j == n - 1:
            left.set_xlabel("Energy (eV)")
            right.set_xlabel("Energy (eV)")
    fig.suptitle("Normalization per scan", y=0.999)
    fig.tight_layout()
    return fig


def plot_flat_overlay(energy, groups, names, ax=None):
    """All flattened/normalized scans overlaid, with the average."""
    ax = ax or plt.gca()
    colors = _scan_colors(len(names))
    flat = np.column_stack([g.flat for g in groups])
    for j, name in enumerate(names):
        ax.plot(energy, flat[:, j], color=colors[j], lw=0.9, alpha=0.85, label=name)
    ax.plot(energy, flat.mean(axis=1), label="average", **_AVG_KW)
    ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
    ax.set(xlabel="Energy (eV)", ylabel="flattened μ(E)", title="Flattened scans")
    return ax


def plot_exafs(energy, groups, names, e0, kweight=3, fig=None):
    """Left: normalized μ + AUTOBK background per scan. Right: kⁿ·χ(k) + average."""
    fig = fig or plt.figure(figsize=(11, 4.2))
    ax_bkg, ax_chi = fig.subplots(1, 2)
    colors = _scan_colors(len(names))

    above = energy >= e0  # the region the spline actually models
    for j, (g, name) in enumerate(zip(groups, names)):
        norm_bkg = (g.bkg - g.pre_edge) / g.edge_step  # background in normalized units
        ax_bkg.plot(energy[above], g.norm[above], color=colors[j], lw=0.8, alpha=0.8)
        ax_bkg.plot(energy[above], norm_bkg[above], color=colors[j], lw=0.9, ls="--", alpha=0.9)
    ax_bkg.set(xlabel="Energy (eV)", ylabel="normalized μ", title="Normalized μ + AUTOBK spline (– –)")

    k = groups[0].k
    chi = np.column_stack([g.chi for g in groups])
    kw = k**kweight
    for j, name in enumerate(names):
        ax_chi.plot(k, kw * chi[:, j], color=colors[j], lw=0.9, alpha=0.85, label=name)
    ax_chi.plot(k, kw * chi.mean(axis=1), label="average", **_AVG_KW)
    ax_chi.set(xlabel="k (Å⁻¹)", ylabel=f"k$^{kweight}$·χ(k)", title=f"EXAFS  k$^{kweight}$·χ(k)")
    fig.tight_layout()
    return fig


def figure_report(bcr, params, kweight=3) -> list[tuple[str, Figure]]:
    """Build all four processing figures for one file. Returns [(label, Figure), ...]."""
    from xasbatch.process import process_scans

    e0, names, scan_mu, groups = process_scans(bcr, params)
    energy = bcr.energy

    f_raw = plt.figure(figsize=(8, 4.5))
    plot_raw(energy, scan_mu, names, ax=f_raw.gca())
    _compact_legend(f_raw.gca(), names)

    f_norm = plot_norm_fits(energy, groups, names, e0)

    f_flat = plt.figure(figsize=(8, 4.5))
    plot_flat_overlay(energy, groups, names, ax=f_flat.gca())
    _compact_legend(f_flat.gca(), names)

    f_exafs = plot_exafs(energy, groups, names, e0, kweight=kweight)
    _compact_legend(f_exafs.axes[1], names)

    for f in (f_raw, f_flat):
        f.tight_layout()
    return [("1_raw", f_raw), ("2_norm_fits", f_norm), ("3_flat", f_flat), ("4_exafs", f_exafs)]


def _compact_legend(ax, names, max_entries=16):
    """Show a legend only when it won't overwhelm the plot (many scans -> skip)."""
    if len(names) <= max_entries:
        ax.legend(fontsize=7, ncol=2, loc="best")
