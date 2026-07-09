"""Data model for xasbatch.

Pure numpy dataclasses — no Larch import here, so I/O and the data model stay
unit-testable without the heavy Larch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BcrData:
    """One combined-BCR file: a shared energy grid + N pre-computed mu channels."""

    energy: np.ndarray  # (nE,) ascending
    mu: np.ndarray  # (nE, nFF) the FF*/I0 channels
    channel_names: list[str]  # len nFF, e.g. ["FF1/I0", ...]
    rtc: np.ndarray | None = None  # (nE, nRTC) ignored for processing, kept for provenance
    rtc_names: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n_energy(self) -> int:
        return self.energy.shape[0]

    @property
    def n_channels(self) -> int:
        return self.mu.shape[1]


@dataclass
class Params:
    """Processing knobs, with sensible Co-K defaults.

    e0 resolution order at batch time: explicit ``e0`` > header ``E0_tab`` >
    (only when ``auto_e0``) Larch ``find_e0``.
    """

    mode: str = "scan"  # "scan" (sum each original file's channels) | "channel" | "both"
    e0: float | None = None  # explicit override; None -> use header E0_tab (or auto if auto_e0)
    auto_e0: bool = False  # when True and e0 is None, detect once via find_e0
    # pre-edge / normalization (eV relative to e0). The spans default to the data
    # extremes (None); pinning only the offsets pre2/norm1 keeps the post-edge
    # polynomial fit across the whole range, which flattens far better than a
    # narrow near-edge window extrapolated outward.
    pre1: float | None = None  # pre-edge fit start; None -> file start
    pre2: float = -50.0  # pre-edge fit end (stay below the edge onset)
    norm1: float = 150.0  # post-edge fit start (above the XANES; Athena convention)
    norm2: float | None = None  # post-edge fit end; None -> file end
    nnorm: int = 2
    # autobk -> chi(k)
    rbkg: float = 1.0
    kmin: float = 0.0
    kmax: float | None = None  # None -> Larch default (uses full k range)
    kweight: int = 1
    kstep: float = 0.05
    # optional forward FT
    ft: bool = False
    ft_kmin: float = 3.0
    ft_kmax: float = 12.0
    ft_kweight: int = 2
    ft_dk: float = 5.0


@dataclass
class ProcessBlock:
    """Stacked results for one set of spectra (all scans, or all channels).

    Every column shares the same k-grid (guaranteed by a shared energy+e0+kstep).
    """

    names: list[str]  # column labels (scan/member names, or channel names)
    flat: np.ndarray  # (nE, n) flattened mu(E) per column
    k: np.ndarray  # (nk,) shared across columns
    chi: np.ndarray  # (nk, n)
    edge_step: np.ndarray  # (n,) per-column edge jump
    # optional forward FT (populated only when Params.ft is True)
    r: np.ndarray | None = None  # (nR,)
    chir_mag: np.ndarray | None = None  # (nR, n)

    @property
    def n(self) -> int:
        return self.chi.shape[1]


@dataclass
class BatchResult:
    """Results for one file. Holds a ``scan`` and/or ``channel`` block on a shared e0.

    Which blocks are present is set by ``Params.mode`` ("scan", "channel", "both").
    """

    energy: np.ndarray  # (nE,)
    e0: float
    scan: ProcessBlock | None = None  # summed-per-original-file spectra (mode scan/both)
    channel: ProcessBlock | None = None  # per-column spectra (mode channel/both)
    meta: dict = field(default_factory=dict)

    @property
    def n_scans(self) -> int:
        return self.scan.n if self.scan is not None else 0

    @property
    def n_channels(self) -> int:
        return self.channel.n if self.channel is not None else 0
