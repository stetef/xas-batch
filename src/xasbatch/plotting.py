"""Matplotlib helpers to visualize the per-scan processing pipeline.

Kept separate from the core so `io`/`process` never import matplotlib. These
functions take the processed per-scan Larch groups (from
:func:`xasbatch.process.process_scans`) plus the E-space *merged* group, and draw:

1. raw summed μ(E) per scan + merged           (`plot_raw`)
2. per-scan pre/post-edge fits + flattened      (`plot_norm_fits`)
3. all flattened scans + merged                 (`plot_flat_overlay`)
4. normalized μ + AUTOBK splines, and kⁿ·χ(k)   (`plot_exafs`)

The functions are backend-agnostic; `figure_report` applies a Times/LaTeX-style
rc context (no TeX install required) so a Streamlit/notebook front-end can reuse the
raw builders without inheriting the styling.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# Publication-ish styling: Times text + STIX math (LaTeX look, no TeX dependency).
_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.1,
    "legend.frameon": False,
    "figure.dpi": 120,
}

# Individual scans share one muted color (their identity isn't the point — the
# spread and the merge are); the E-space merge is the bold trace.
_SCAN = dict(color="0.60", lw=0.8, alpha=0.75)
_MERGED = dict(color="black", lw=2.0, zorder=6)
# Okabe–Ito colorblind-safe accents for the per-scan normalization fits.
_C_MU, _C_PRE, _C_POST, _C_FLAT, _C_BKG = "black", "#0072B2", "#D55E00", "#13396B", "#CC3311"


def _finish(ax):
    ax.tick_params(direction="out", length=4)
    return ax


def _overlay_legend(ax, n, merged=True):
    """Two-entry legend for the overlay panels (scans share a color)."""
    handles = [Line2D([0], [0], color=_SCAN["color"], lw=1.4, label=f"scans (n={n})")]
    if merged:
        handles.append(Line2D([0], [0], **_MERGED, label="merged (avg)"))
    ax.legend(handles=handles, loc="best")


def plot_raw(energy, scan_mu, names, merged_mu=None, ax=None):
    """Raw summed μ(E) (= Σ FF/I0) per scan, with the E-space merge (mean μ)."""
    ax = ax or plt.gca()
    for j in range(scan_mu.shape[1]):
        ax.plot(energy, scan_mu[:, j], **_SCAN)
    if merged_mu is not None:
        ax.plot(energy, merged_mu, **_MERGED)
    ax.set(xlabel="Energy (eV)", ylabel=r"$\mu$ (summed FF/I0)", title="Raw summed scans")
    _overlay_legend(ax, len(names), merged=merged_mu is not None)
    return _finish(ax)


def plot_norm_fits(energy, groups, names, e0, fig=None):
    """Grid: each scan's μ(E) with pre/post-edge fits (left) and flattened μ (right)."""
    n = len(groups)
    header = 0.9  # inches reserved at the top for the title + shared legend row
    height = 2.2 * n + header
    fig = fig or plt.figure(figsize=(9.5, height))
    axes = fig.subplots(n, 2, squeeze=False)
    for j, (g, name) in enumerate(zip(groups, names)):
        left, right = axes[j]
        left.plot(energy, g.mu, color=_C_MU, lw=1.3)
        left.plot(energy, g.pre_edge, color=_C_PRE, lw=1.3, ls="--")
        left.plot(energy, g.post_edge, color=_C_POST, lw=1.3, ls="--")
        left.axvline(e0, color="0.55", lw=0.9, ls=":")
        left.text(0.02, 0.95, name, transform=left.transAxes, ha="left", va="top",
                  fontsize=12, color="black")
        # E0 + edge step reported per scan (no box), so a future per-scan e0 shows here
        left.text(0.55, 0.32, rf"$E_0 = {e0:.1f}$ eV" "\n" rf"$\Delta\mu_0 = {g.edge_step:.3f}$",
                  transform=left.transAxes, ha="left", va="center", fontsize=12)
        right.plot(energy, g.flat, color=_C_FLAT, lw=1.4)
        right.axhline(1.0, color="0.7", lw=0.8, ls=":")
        _finish(left)
        _finish(right)
        if j == 0:
            left.set_title("pre / post-edge fit")
            right.set_title(r"flattened $\mu(E)$")
        if j == n - 1:
            left.set_xlabel("Energy (eV)")
            right.set_xlabel("Energy (eV)")

    fig.tight_layout(rect=(0, 0, 1, 1 - header / height))
    fig.suptitle("Normalization per scan", y=1 - 0.22 * header / height, fontsize=15)
    handles = [
        Line2D([0], [0], color=_C_MU, lw=1.5, label=r"$\mu(E)$"),
        Line2D([0], [0], color=_C_PRE, lw=1.5, ls="--", label="pre-edge"),
        Line2D([0], [0], color=_C_POST, lw=1.5, ls="--", label="post-edge"),
        Line2D([0], [0], color=_C_FLAT, lw=1.6, label=r"flattened $\mu(E)$"),
    ]
    fig.legend(handles=handles, loc="center", ncol=4, bbox_to_anchor=(0.5, 1 - 0.62 * header / height))
    return fig


def plot_flat_overlay(energy, groups, names, merged=None, ax=None):
    """All flattened/normalized scans overlaid, with the merged (E-space) spectrum."""
    ax = ax or plt.gca()
    for g in groups:
        ax.plot(energy, g.flat, **_SCAN)
    if merged is not None:
        ax.plot(energy, merged.flat, **_MERGED)
    ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
    ax.set(xlabel="Energy (eV)", ylabel=r"flattened $\mu(E)$", title="Flattened scans")
    _overlay_legend(ax, len(names), merged=merged is not None)
    return _finish(ax)


def plot_exafs(energy, groups, names, e0, merged=None, kweight=3, fig=None):
    """Left: normalized μ + AUTOBK background per scan. Right: kⁿ·χ(k) per scan.

    The bold ``merged`` trace is the E-space merge carried through (pre_edge + AUTOBK
    on the mean μ), so its χ(k) is *not* the mean of the per-scan χ(k).
    """
    fig = fig or plt.figure(figsize=(10.5, 3.8))
    ax_bkg, ax_chi = fig.subplots(1, 2)

    def _norm_bkg(g):  # background in normalized-μ units
        return (g.bkg - g.pre_edge) / g.edge_step

    above = energy >= e0  # the region the spline actually models
    for g in groups:
        ax_bkg.plot(energy[above], g.norm[above], **_SCAN)
        ax_bkg.plot(energy[above], _norm_bkg(g)[above], color=_SCAN["color"], lw=0.8, ls="--", alpha=0.55)
    if merged is not None:
        ax_bkg.plot(energy[above], merged.norm[above], label=r"merged $\mu$", **_MERGED)
        ax_bkg.plot(energy[above], _norm_bkg(merged)[above], color=_C_BKG, lw=1.8, ls="--", zorder=6, label="merged bkg")
    ax_bkg.set_ylim(bottom=0)  # normalized μ starts at 0
    ax_bkg.set(xlabel="Energy (eV)", ylabel=r"normalized $\mu$", title=r"Normalized $\mu$ + AUTOBK spline")
    ax_bkg.legend(loc="lower right")
    _finish(ax_bkg)

    for g in groups:
        ax_chi.plot(g.k, g.k**kweight * g.chi, **_SCAN)
    if merged is not None:
        ax_chi.plot(merged.k, merged.k**kweight * merged.chi, **_MERGED)
    ax_chi.axhline(0.0, color="0.7", lw=0.8, ls=":")
    ax_chi.set(xlabel=r"$k$ (Å$^{-1}$)", ylabel=rf"$k^{kweight}\,\chi(k)$", title=rf"EXAFS   $k^{kweight}\chi(k)$")
    _overlay_legend(ax_chi, len(names), merged=merged is not None)
    _finish(ax_chi)
    fig.tight_layout()
    return fig


def figure_report(bcr, params, kweight=3) -> list[tuple[str, Figure]]:
    """Build all four processing figures for one file. Returns [(label, Figure), ...]."""
    from xasbatch.process import process_channel, process_scans

    e0, names, scan_mu, groups = process_scans(bcr, params)
    energy = bcr.energy

    # Merge in E space, then carry it through: average μ(E), run the SAME pipeline once.
    merged_mu = scan_mu.mean(axis=1)
    merged = process_channel(energy, merged_mu, params, e0)

    with plt.rc_context(_RC):
        f_raw = plt.figure(figsize=(8, 4.8))
        plot_raw(energy, scan_mu, names, merged_mu=merged_mu, ax=f_raw.gca())

        f_norm = plot_norm_fits(energy, groups, names, e0)

        f_flat = plt.figure(figsize=(6.2, 4.2))
        plot_flat_overlay(energy, groups, names, merged=merged, ax=f_flat.gca())

        f_exafs = plot_exafs(energy, groups, names, e0, merged=merged, kweight=kweight)

        for f in (f_raw, f_flat):
            f.tight_layout()
    return [("1_raw", f_raw), ("2_norm_fits", f_norm), ("3_flat", f_flat), ("4_exafs", f_exafs)]
