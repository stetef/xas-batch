"""The Larch layer: normalization, AUTOBK background spline, χ(k), optional FT.

All Larch imports live here (and in ``plotting``). Everything is a thin call into
``larch.xafs.*`` — the value catXAS's wrappers added (delE bookkeeping, Experiment
param bundling) does not apply to these already-calibrated, I0-divided files.
"""

from __future__ import annotations

import numpy as np
from larch import Group
from larch.xafs import autobk, find_e0, pre_edge, xftf

from xasbatch.io import scan_groups
from xasbatch.model import BatchResult, BcrData, Params, ProcessBlock


def build_group(energy: np.ndarray, mu: np.ndarray) -> Group:
    """Wrap one energy/mu column in a Larch group."""
    return Group(energy=np.asarray(energy, dtype=float), mu=np.asarray(mu, dtype=float))


def find_edge(energy: np.ndarray, mu: np.ndarray) -> float:
    """Detect the edge energy e0 via ``larch.xafs.find_e0``."""
    group = build_group(energy, mu)
    return float(find_e0(group.energy, group.mu, group=group))


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


def resolve_e0(bcr: BcrData, params: Params) -> float:
    """Resolve the edge energy once: explicit ``params.e0`` > header ``E0_tab`` > find_e0.

    ``find_e0`` is used only when ``params.auto_e0`` is set, or as a last resort when
    the header carries no tabulated edge.
    """
    if params.e0 is not None:
        return float(params.e0)

    header_e0 = bcr.meta.get("e0_tab")
    if params.auto_e0 or header_e0 is None:
        # detect from a representative (mean) column so a single noisy channel can't skew it
        return find_edge(bcr.energy, bcr.mu.mean(axis=1))
    return float(header_e0)


def _process_matrix(
    energy: np.ndarray, mu_matrix: np.ndarray, names: list[str], params: Params, e0: float
) -> ProcessBlock:
    """Process each column of ``mu_matrix`` and stack onto a shared k-grid."""
    flat_cols, chi_cols, edge_steps = [], [], []
    k_ref = r_ref = None
    chir_cols = [] if params.ft else None

    for j in range(mu_matrix.shape[1]):
        group = process_channel(energy, mu_matrix[:, j], params, e0)

        if k_ref is None:
            k_ref = np.asarray(group.k, dtype=float)
        elif group.k.shape != k_ref.shape:
            raise ValueError(
                f"{names[j]!r} returned k of length {group.k.shape[0]}, expected "
                f"{k_ref.shape[0]}; shared-grid assumption violated."
            )

        flat_cols.append(np.asarray(group.flat, dtype=float))
        chi_cols.append(np.asarray(group.chi, dtype=float))
        edge_steps.append(float(group.edge_step))

        if params.ft:
            if r_ref is None:
                r_ref = np.asarray(group.r, dtype=float)
            chir_cols.append(np.asarray(group.chir_mag, dtype=float))

    return ProcessBlock(
        names=list(names),
        flat=np.column_stack(flat_cols),
        k=k_ref,
        chi=np.column_stack(chi_cols),
        edge_step=np.asarray(edge_steps, dtype=float),
        r=r_ref,
        chir_mag=np.column_stack(chir_cols) if params.ft else None,
    )


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


def process_scans(bcr: BcrData, params: Params):
    """Process the per-scan summed spectra, keeping the full Larch groups.

    Returns ``(e0, names, scan_mu, groups)`` where ``scan_mu`` is the raw summed
    μ(E) per scan (nE×nScans) and ``groups`` are the processed Larch groups (with
    ``.pre_edge/.post_edge/.norm/.flat/.bkg/.k/.chi``). Used by the plotting layer,
    which needs the intermediate fit curves that ``process_batch`` discards.
    """
    e0 = resolve_e0(bcr, params)
    names, scan_mu, _ = _sum_scans(bcr)
    groups = [process_channel(bcr.energy, scan_mu[:, j], params, e0) for j in range(scan_mu.shape[1])]
    return e0, names, scan_mu, groups


def process_batch(bcr: BcrData, params: Params) -> BatchResult:
    """Process one file into a ``scan`` and/or ``channel`` block on a shared e0.

    ``params.mode`` selects which:
      - ``"scan"`` (default): sum each original file's channels → one μ(E) per scan.
      - ``"channel"``: process every μ column individually.
      - ``"both"``: compute and store both blocks in the one result.

    e0 is resolved once (see :func:`resolve_e0`) and reused across every column of
    every block, so all k-grids align (asserted in :func:`_process_matrix`).
    """
    if params.mode not in ("scan", "channel", "both"):
        raise ValueError(f"invalid mode {params.mode!r}; expected scan|channel|both.")

    e0 = resolve_e0(bcr, params)

    scan_block = channel_block = None
    scan_members = None
    if params.mode in ("scan", "both"):
        names, scan_mu, scan_members = _sum_scans(bcr)
        scan_block = _process_matrix(bcr.energy, scan_mu, names, params, e0)
    if params.mode in ("channel", "both"):
        channel_block = _process_matrix(bcr.energy, bcr.mu, bcr.channel_names, params, e0)

    meta = dict(bcr.meta)
    meta["e0_used"] = e0
    meta["e0_source"] = (
        "explicit"
        if params.e0 is not None
        else ("find_e0" if (params.auto_e0 or bcr.meta.get("e0_tab") is None) else "header_e0_tab")
    )
    meta["mode"] = params.mode
    meta["n_channels_raw"] = bcr.n_channels
    meta["modes_present"] = [
        name for name, blk in (("scan", scan_block), ("channel", channel_block)) if blk is not None
    ]
    if scan_members is not None:
        meta["scan_members"] = scan_members

    return BatchResult(
        energy=bcr.energy, e0=e0, scan=scan_block, channel=channel_block, meta=meta
    )
