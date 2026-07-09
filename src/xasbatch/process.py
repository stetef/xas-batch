"""The Larch layer: normalization, AUTOBK background spline, χ(k), optional FT.

All Larch imports live here (and in ``plotting``). Everything is a thin call into
``larch.xafs.*`` — the value catXAS's wrappers added (delE bookkeeping, Experiment
param bundling) does not apply to these already-calibrated, I0-divided files.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from larch import Group
from larch.xafs import autobk, find_e0, pre_edge, xftf
from scipy.signal import savgol_filter

from xasbatch.io import scan_groups
from xasbatch.model import BatchResult, BcrData, Params, ProcessBlock

# k [Å⁻¹] = sqrt(ETOK * (E - e0) [eV]); matches Larch's photoelectron-wavenumber constant.
ETOK = 0.2624682843


def _shared_kmax(energy: np.ndarray, kstep: float, e0_hi: float) -> float:
    """Largest k (floored to a kstep multiple) the data covers for the highest e0.

    Using the *highest* e0 (whose data spans the smallest k range) guarantees no
    spectrum extrapolates past its data when this single kmax is applied to all.
    """
    kmax_full = float(np.sqrt(ETOK * (float(np.max(energy)) - e0_hi)))
    return float(np.floor(kmax_full / kstep) * kstep)


def _effective_params(energy: np.ndarray, params: Params, e0_values) -> Params:
    """Params with ``kmax`` resolved once per file so every spectrum shares one k-grid.

    When ``params.kmax`` is None (auto), derive a single kmax from the highest e0 in use;
    an explicit ``params.kmax`` is passed through unchanged.
    """
    if params.kmax is not None:
        return params
    kmax = _shared_kmax(energy, params.kstep, float(np.max(e0_values)))
    return replace(params, kmax=kmax)


def build_group(energy: np.ndarray, mu: np.ndarray) -> Group:
    """Wrap one energy/mu column in a Larch group."""
    return Group(energy=np.asarray(energy, dtype=float), mu=np.asarray(mu, dtype=float))


def find_edge(
    energy: np.ndarray,
    mu: np.ndarray,
    e0_guess: float | None = None,
    search_window: float = 25.0,
    smooth: bool = True,
    sg_window: int = 7,
    sg_polyorder: int = 2,
) -> float:
    """Detect the edge energy e0 (derivative-max) via ``larch.xafs.find_e0``.

    Noise guards (mirroring catXAS's ``calculate_spectrum_e0``): when ``e0_guess`` is
    given (we pass the header ``E0_tab``), the search is restricted to
    ``e0_guess ± search_window`` so a glitch or far feature can't hijack it, and μ is
    lightly Savitzky-Golay smoothed within that window first. Both are no-ops on a clean
    high-SNR spectrum but protect noisy or glitchy ones. Falls back to the full range if
    no guess is given or the window holds too few points.
    """
    energy = np.asarray(energy, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if e0_guess is not None:
        m = (energy >= e0_guess - search_window) & (energy <= e0_guess + search_window)
        if int(m.sum()) >= max(sg_window, 5):
            energy, mu = energy[m], mu[m]
    if smooth and mu.size > sg_window:
        mu = savgol_filter(mu, sg_window, sg_polyorder)
    return float(find_e0(energy, mu=mu, group=None))


def normalize(group: Group, params: Params, e0: float) -> Group:
    """Pre/post-edge normalization → sets ``.flat``, ``.norm``, ``.edge_step``."""
    pre_edge(
        group.energy,
        group.mu,
        group=group,
        e0=e0,
        pre1=params.pre1,
        pre2=params.pre2,
        norm1=params.norm1,
        norm2=params.norm2,
        nnorm=params.nnorm,
    )
    return group


def extract_exafs(group: Group, params: Params, e0: float) -> Group:
    """AUTOBK background spline + E→k + spline subtraction → sets ``.k``, ``.chi``, ``.bkg``."""
    autobk(
        group.energy,
        group.mu,
        group=group,
        ek0=e0,
        rbkg=params.rbkg,
        kmin=params.kmin,
        kmax=params.kmax,
        kweight=params.kweight,
        kstep=params.kstep,
    )
    return group


def forward_ft(group: Group, params: Params) -> Group:
    """Optional forward FT χ(k)→χ(R) → sets ``.r``, ``.chir_mag``."""
    xftf(
        group.k,
        group.chi,
        group=group,
        kmin=params.ft_kmin,
        kmax=params.ft_kmax,
        kweight=params.ft_kweight,
        dk=params.ft_dk,
    )
    return group


def process_channel(energy: np.ndarray, mu: np.ndarray, params: Params, e0: float) -> Group:
    """Full single-column pipeline: normalize → extract χ(k) → (optional) FT."""
    group = build_group(energy, mu)
    normalize(group, params, e0)
    extract_exafs(group, params, e0)
    if params.ft:
        forward_ft(group, params)
    return group


def _resolve_e0_for_mu(bcr: BcrData, params: Params, mu: np.ndarray) -> float:
    """Edge energy for a given μ: explicit ``params.e0`` > find_e0 (default) > header."""
    if params.e0 is not None:
        return float(params.e0)
    header_e0 = bcr.meta.get("e0_tab")
    if params.auto_e0 or header_e0 is None:
        return find_edge(bcr.energy, mu, e0_guess=header_e0)
    return float(header_e0)


def resolve_e0(bcr: BcrData, params: Params) -> float:
    """Resolve the representative edge energy on the mean-of-channels μ (high-SNR)."""
    return _resolve_e0_for_mu(bcr, params, bcr.mu.mean(axis=1))


def resolve_scan_e0s(bcr: BcrData, params: Params, scan_mu: np.ndarray) -> np.ndarray:
    """Per-scan edge energies (one per summed scan).

    In auto mode each scan gets its own ``find_e0`` (scans are high-SNR sums of ~30
    channels, so this is stable — unlike per-channel). An explicit ``params.e0`` or the
    header value is broadcast to every scan instead.
    """
    n = scan_mu.shape[1]
    if params.e0 is not None:
        return np.full(n, float(params.e0))
    header_e0 = bcr.meta.get("e0_tab")
    if not (params.auto_e0 or header_e0 is None):
        return np.full(n, float(header_e0))
    return np.array(
        [find_edge(bcr.energy, scan_mu[:, j], e0_guess=header_e0) for j in range(n)], dtype=float
    )


def _stack_groups(names: list[str], groups: list[Group], e0s: np.ndarray, ft: bool) -> ProcessBlock:
    """Stack already-processed Larch groups (shared k-grid) into a :class:`ProcessBlock`."""
    k_ref = np.asarray(groups[0].k, dtype=float)
    for g, name in zip(groups, names):
        if np.asarray(g.k).shape != k_ref.shape:
            raise ValueError(
                f"{name!r} returned k of length {np.asarray(g.k).shape[0]}, expected "
                f"{k_ref.shape[0]}; shared-grid assumption violated."
            )
    return ProcessBlock(
        names=list(names),
        flat=np.column_stack([np.asarray(g.flat, dtype=float) for g in groups]),
        k=k_ref,
        chi=np.column_stack([np.asarray(g.chi, dtype=float) for g in groups]),
        edge_step=np.asarray([float(g.edge_step) for g in groups], dtype=float),
        e0=np.asarray(e0s, dtype=float).copy(),
        r=np.asarray(groups[0].r, dtype=float) if ft else None,
        chir_mag=np.column_stack([np.asarray(g.chir_mag, dtype=float) for g in groups]) if ft else None,
    )


def _process_matrix(
    energy: np.ndarray, mu_matrix: np.ndarray, names: list[str], params: Params, e0s
) -> ProcessBlock:
    """Process each column of ``mu_matrix`` (each with its own ``e0s[j]``) and stack."""
    e0s = np.atleast_1d(np.asarray(e0s, dtype=float))
    if e0s.size == 1:
        e0s = np.full(mu_matrix.shape[1], float(e0s[0]))
    groups = [process_channel(energy, mu_matrix[:, j], params, float(e0s[j]))
              for j in range(mu_matrix.shape[1])]
    return _stack_groups(names, groups, e0s, params.ft)


def _sum_scans(bcr: BcrData) -> tuple[list[str], np.ndarray, list[dict]]:
    """Sum each original file's channels into one μ(E) per scan (total fluorescence).

    ``nansum`` so a missing detector element doesn't poison the scan sum.
    Returns (scan names, μ matrix (nE, nScans), scan-member metadata).
    """
    groups = scan_groups(bcr.meta)
    names, cols, members = [], [], []
    for name, start, stop in groups:
        names.append(name)
        cols.append(np.nansum(bcr.mu[:, start:stop], axis=1))
        members.append({"name": name, "n_channels": stop - start})
    return names, np.column_stack(cols), members


class SkipFile(Exception):
    """Raised when a file cannot be sensibly processed (e.g. too little post-edge range).

    The batch runner records these as ``skipped`` rather than ``error`` — a clean,
    expected outcome, not a crash.
    """


# QC thresholds for the robust per-scan e0 outlier test.
_E0_MAD_K = 5.0  # keep within this many robust-σ of the median ...
_E0_FLOOR = 2.0  # ... but never flag anything within this many eV (σ can be ~0.08)
_RANGE_MARGIN = 20.0  # required post-edge span beyond norm1 (eV) for a usable fit


def _e0_outlier_mask(e0s: np.ndarray) -> np.ndarray:
    """Robust keep-mask: True where e0 is within max(K·MAD, floor) of the median.

    Robust (median/MAD + an absolute floor), not 3σ — the per-scan σ is ~0.08 eV, so a
    plain 3σ would flag normal scatter; only genuinely mis-picked edges should drop.
    """
    med = float(np.median(e0s))
    mad = float(np.median(np.abs(e0s - med)))
    thresh = max(_E0_MAD_K * 1.4826 * mad, _E0_FLOOR)
    return np.abs(e0s - med) <= thresh


def _scan_finite_ok(group: Group) -> bool:
    """Deterministic gate: normalization/spline produced finite, real results."""
    return (
        np.isfinite(group.edge_step)
        and float(group.edge_step) > 0.0
        and np.isfinite(group.flat).all()
        and np.isfinite(group.chi).all()
    )


def _process_scan_set(bcr: BcrData, params: Params):
    """Core per-scan pipeline with QC + E-space merge over the passing scans.

    Returns a dict with: names, scan_mu, members, scan_e0s, groups, scan_pass (bool),
    reasons (list[str] per scan), merged (Group), e0_merged, eff (resolved Params).
    Raises :class:`SkipFile` if the file lacks enough post-edge range to normalize.
    """
    names, scan_mu, members = _sum_scans(bcr)
    n = len(names)
    scan_e0s = resolve_scan_e0s(bcr, params, scan_mu)

    # file-level range gate: need a usable post-edge window above the edge
    e0_ref = float(np.median(scan_e0s))
    span = float(bcr.energy.max()) - e0_ref
    if params.qc and span < params.norm1 + _RANGE_MARGIN:
        raise SkipFile(
            f"insufficient post-edge range: {span:.0f} eV above e0≈{e0_ref:.0f} "
            f"(need ≥ norm1+{_RANGE_MARGIN:.0f} = {params.norm1 + _RANGE_MARGIN:.0f} eV)"
        )

    reasons: list[list[str]] = [[] for _ in range(n)]
    e0_keep = _e0_outlier_mask(scan_e0s) if params.qc else np.ones(n, dtype=bool)
    for j in np.where(~e0_keep)[0]:
        reasons[j].append("e0_outlier")

    # one shared k-grid, then process every scan (failing ones kept + flagged, not dropped)
    eff = _effective_params(bcr.energy, params, scan_e0s)
    groups = [process_channel(bcr.energy, scan_mu[:, j], eff, float(scan_e0s[j])) for j in range(n)]
    finite_keep = np.array([_scan_finite_ok(g) for g in groups]) if params.qc else np.ones(n, bool)
    for j in np.where(~finite_keep)[0]:
        reasons[j].append("nonfinite_or_bad_edge_step")

    scan_pass = e0_keep & finite_keep
    use = scan_pass if scan_pass.any() else np.ones(n, dtype=bool)  # fallback: none passed
    merged_mu = scan_mu[:, use].mean(axis=1)
    e0_merged = _resolve_e0_for_mu(bcr, params, merged_mu)
    merged = process_channel(bcr.energy, merged_mu, eff, e0_merged)

    return {
        "names": names, "scan_mu": scan_mu, "members": members, "scan_e0s": scan_e0s,
        "groups": groups, "scan_pass": scan_pass, "reasons": reasons,
        "merged": merged, "e0_merged": e0_merged, "eff": eff,
    }


def process_scans(bcr: BcrData, params: Params):
    """Per-scan pipeline for the plotting layer, keeping the full Larch groups.

    Returns ``(e0_merged, names, scan_mu, groups, scan_e0s, merged, scan_pass)``. See
    :func:`_process_scan_set`; ``merged`` is the mean-of-passing-scans μ processed on the
    shared k-grid, and ``scan_pass`` flags which scans were included in that merge.
    """
    ss = _process_scan_set(bcr, params)
    return (ss["e0_merged"], ss["names"], ss["scan_mu"], ss["groups"],
            ss["scan_e0s"], ss["merged"], ss["scan_pass"])


def process_batch(bcr: BcrData, params: Params) -> BatchResult:
    """Process one file into ``scan`` / ``channel`` / ``merged`` blocks.

    ``params.mode`` selects which per-spectrum blocks are computed ("scan" default,
    "channel", or "both"). In scan/both mode a ``merged`` block (mean of the QC-passing
    scans, processed through the same pipeline) is always produced — the clean target.

    e0: the ``scan`` block gets a **per-scan** e0; the ``channel`` block uses the
    **merged** e0 (per-channel e0 is biased). With ``params.qc`` (default), scans failing
    QC (robust e0 outlier, or non-finite normalization/spline) are excluded from the merge
    but still stored and flagged in ``scan_pass``; a file with too little post-edge range
    raises :class:`SkipFile`.
    """
    if params.mode not in ("scan", "channel", "both"):
        raise ValueError(f"invalid mode {params.mode!r}; expected scan|channel|both.")

    scan_block = channel_block = merged_block = None
    scan_members = scan_e0s = scan_pass = None
    reasons = None

    if params.mode in ("scan", "both"):
        ss = _process_scan_set(bcr, params)
        scan_members, scan_e0s, scan_pass = ss["members"], ss["scan_e0s"], ss["scan_pass"]
        reasons = ss["reasons"]
        eff, e0_merged = ss["eff"], ss["e0_merged"]
        scan_block = _stack_groups(ss["names"], ss["groups"], scan_e0s, params.ft)
        merged_block = _stack_groups(["merged"], [ss["merged"]], np.array([e0_merged]), params.ft)
    else:
        e0_merged = resolve_e0(bcr, params)
        eff = _effective_params(bcr.energy, params, [e0_merged])

    if params.mode in ("channel", "both"):
        channel_block = _process_matrix(bcr.energy, bcr.mu, bcr.channel_names, eff, e0_merged)

    meta = dict(bcr.meta)
    meta["e0_used"] = e0_merged
    meta["e0_merged"] = e0_merged
    meta["kmax_used"] = eff.kmax
    meta["e0_source"] = (
        "explicit"
        if params.e0 is not None
        else ("find_e0" if (params.auto_e0 or bcr.meta.get("e0_tab") is None) else "header_e0_tab")
    )
    meta["mode"] = params.mode
    meta["n_channels_raw"] = bcr.n_channels
    meta["modes_present"] = [
        name
        for name, blk in (("scan", scan_block), ("channel", channel_block), ("merged", merged_block))
        if blk is not None
    ]
    if scan_e0s is not None:
        kept = scan_e0s[scan_pass] if scan_pass.any() else scan_e0s
        meta["e0_scan_mean"] = float(np.mean(kept))
        meta["e0_scan_std"] = float(np.std(kept))
        meta["n_scans_total"] = int(len(scan_e0s))
        meta["n_scans_used"] = int(scan_pass.sum())
        meta["n_scans_excluded"] = int((~scan_pass).sum())
        meta["scan_qc_reasons"] = {
            scan_members[j]["name"]: reasons[j] for j in range(len(reasons)) if reasons[j]
        }
    if scan_members is not None:
        meta["scan_members"] = scan_members

    return BatchResult(
        energy=bcr.energy, e0=e0_merged, scan=scan_block, channel=channel_block,
        merged=merged_block, scan_pass=scan_pass, meta=meta,
    )
