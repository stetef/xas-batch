"""The Larch layer: normalization, AUTOBK background spline, χ(k), optional FT.

All Larch imports live here (and in ``plotting``). Everything is a thin call into
``larch.xafs.*`` — the value catXAS's wrappers added (delE bookkeeping, Experiment
param bundling) does not apply to these already-calibrated, I0-divided files.
"""

from __future__ import annotations

import numpy as np
from larch import Group
from larch.xafs import autobk, find_e0, pre_edge, xftf

from xasbatch.model import BatchResult, BcrData, Params


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


def process_batch(bcr: BcrData, params: Params) -> BatchResult:
    """Process every μ channel of one file onto a shared k-grid and stack the results.

    e0 is resolved once (see :func:`resolve_e0`) and reused across channels, so
    identical energy+e0+kstep yields an aligned k-grid for every column — which we
    assert, then store a single ``k`` alongside the ``chi`` matrix.
    """
    e0 = resolve_e0(bcr, params)

    flat_cols, chi_cols, edge_steps = [], [], []
    k_ref = None
    r_ref = None
    chir_cols = [] if params.ft else None

    for j in range(bcr.n_channels):
        group = process_channel(bcr.energy, bcr.mu[:, j], params, e0)

        if k_ref is None:
            k_ref = np.asarray(group.k, dtype=float)
        elif group.k.shape != k_ref.shape:
            raise ValueError(
                f"channel {bcr.channel_names[j]!r} returned k of length "
                f"{group.k.shape[0]}, expected {k_ref.shape[0]}; shared-grid "
                "assumption violated."
            )

        flat_cols.append(np.asarray(group.flat, dtype=float))
        chi_cols.append(np.asarray(group.chi, dtype=float))
        edge_steps.append(float(group.edge_step))

        if params.ft:
            if r_ref is None:
                r_ref = np.asarray(group.r, dtype=float)
            chir_cols.append(np.asarray(group.chir_mag, dtype=float))

    meta = dict(bcr.meta)
    meta["e0_used"] = e0
    meta["e0_source"] = (
        "explicit"
        if params.e0 is not None
        else ("find_e0" if (params.auto_e0 or bcr.meta.get("e0_tab") is None) else "header_e0_tab")
    )

    return BatchResult(
        energy=bcr.energy,
        flat=np.column_stack(flat_cols),
        k=k_ref,
        chi=np.column_stack(chi_cols),
        e0=e0,
        edge_step=np.asarray(edge_steps, dtype=float),
        channel_names=list(bcr.channel_names),
        meta=meta,
        r=r_ref,
        chir_mag=np.column_stack(chir_cols) if params.ft else None,
    )
