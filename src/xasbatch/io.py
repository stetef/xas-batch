"""File parsing + writers for combined-BCR XAS files.

Pure numpy — NO Larch import. This isolates the custom parser (the highest-bug-risk
code) so it can be unit-tested without the heavy Larch dependency.

Input format (one file = one sample)::

    # Combined per-sample XAS spectrum
    # Session: 2017_7-3_Apr
    # Sample: Co3NK_s
    # Element: Co  (reference edge: K @ 7709.0000 eV, from xraydb)
    # k_max calculated (1/Å): 15.0200  (... E0_tab=7709.0000 eV ...)
    # ...
    # Columns: Energy FF1/I0 FF2/I0 ... FF448/I0 RTC_1 ... RTC_15
    7389.354000 0.001137 0.001471 ...
    ...

The ``# Columns:`` line is authoritative for the column split: index 0 is energy,
any column whose name starts with ``RTC`` is provenance (ignored for processing),
everything else is a μ channel.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from xasbatch.model import BatchResult, BcrData

COLUMNS_PREFIX = "# Columns:"


def _read_header_lines(path: Path) -> list[str]:
    """Return the leading ``#`` comment lines (stops at the first data row)."""
    lines: list[str] = []
    with open(path) as fh:
        for raw in fh:
            if raw.startswith("#"):
                lines.append(raw.rstrip("\n"))
            elif raw.strip() == "":
                continue
            else:
                break
    return lines


def _find(pattern: str, text: str, cast=str, group: int = 1):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return cast(m.group(group))
    except (ValueError, IndexError):
        return None


def parse_header(path: str | Path) -> dict:
    """Parse the ``#``-comment header block.

    Returns a dict with parsed metadata plus the column split:
    ``column_names`` (all, in order), ``energy_col`` (int index),
    ``mu_cols`` / ``rtc_cols`` (index lists) and ``channel_names`` / ``rtc_names``.
    Raises ``ValueError`` if no ``# Columns:`` line is found — we fail loudly rather
    than guess the layout.
    """
    path = Path(path)
    header_lines = _read_header_lines(path)
    header_text = "\n".join(header_lines)

    columns_line = next((ln for ln in header_lines if ln.startswith(COLUMNS_PREFIX)), None)
    if columns_line is None:
        raise ValueError(
            f"{path.name}: no '{COLUMNS_PREFIX}' line in header; refusing to guess the "
            "column layout."
        )

    column_names = columns_line[len(COLUMNS_PREFIX) :].split()
    if not column_names:
        raise ValueError(f"{path.name}: '{COLUMNS_PREFIX}' line is empty.")

    energy_col = 0  # by construction the first column is Energy
    mu_cols, rtc_cols = [], []
    channel_names, rtc_names = [], []
    for idx, name in enumerate(column_names):
        if idx == energy_col:
            continue
        if name.upper().startswith("RTC"):
            rtc_cols.append(idx)
            rtc_names.append(name)
        else:
            mu_cols.append(idx)
            channel_names.append(name)

    meta = {
        "source_path": str(path),
        "session": _find(r"# Session:\s*(.+)", header_text),
        "sample": _find(r"# Sample:\s*(.+)", header_text),
        "depositor": _find(r"# Depositor:\s*(.+)", header_text),
        "element": _find(r"# Element:\s*(\S+)", header_text),
        "edge": _find(r"reference edge:\s*(\S+)\s*@", header_text),
        "e0_tab": _find(r"reference edge:.*?@\s*([\d.]+)\s*eV", header_text, float),
        "k_max": _find(r"# k_max[^:]*:\s*([\d.]+)", header_text, float),
        "n_files": _find(r"# Number of files in group:\s*(\d+)", header_text, int),
        "n_channels_header": _find(
            r"# Total channels \(summed across files\):\s*(\d+)", header_text, int
        ),
        "last_energy": _find(r"# Last energy \(eV\):\s*([\d.]+)", header_text, float),
        "column_names": column_names,
        "energy_col": energy_col,
        "mu_cols": mu_cols,
        "rtc_cols": rtc_cols,
        "channel_names": channel_names,
        "rtc_names": rtc_names,
    }
    # e0_tab can also appear as "E0_tab=7709.0000" inside the k_max annotation
    if meta["e0_tab"] is None:
        meta["e0_tab"] = _find(r"E0_tab=([\d.]+)", header_text, float)
    return meta


def load_combined_bcr(path: str | Path) -> BcrData:
    """Load a combined-BCR file into a :class:`BcrData`.

    Slices columns using the parsed header, flips to ascending energy if the file
    stores it descending, and stashes (but does not process) the RTC columns.
    """
    path = Path(path)
    meta = parse_header(path)

    data = np.loadtxt(path, comments="#", ndmin=2)
    n_expected = len(meta["column_names"])
    if data.shape[1] != n_expected:
        raise ValueError(
            f"{path.name}: header declares {n_expected} columns but data has "
            f"{data.shape[1]}."
        )

    energy = data[:, meta["energy_col"]]
    mu = data[:, meta["mu_cols"]]
    rtc = data[:, meta["rtc_cols"]] if meta["rtc_cols"] else None

    # Energy may be stored descending -> flip to ascending.
    if energy.size > 1 and energy[0] > energy[-1]:
        order = np.argsort(energy)
        energy = energy[order]
        mu = mu[order, :]
        if rtc is not None:
            rtc = rtc[order, :]

    return BcrData(
        energy=energy,
        mu=mu,
        channel_names=list(meta["channel_names"]),
        rtc=rtc,
        rtc_names=list(meta["rtc_names"]),
        meta=meta,
    )


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_result(result: BatchResult, outdir: str | Path) -> Path:
    """Write a :class:`BatchResult` to ``<outdir>/<sample>.npz``; returns the path.

    ``meta`` is JSON-encoded into a 0-d string array so it survives the npz round-trip.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sample = result.meta.get("sample") or Path(result.meta.get("source_path", "result")).stem
    out_path = outdir / f"{sample}.npz"

    arrays = {
        "energy": result.energy,
        "flat": result.flat,
        "k": result.k,
        "chi": result.chi,
        "e0": np.asarray(result.e0),
        "edge_step": result.edge_step,
        "channel_names": np.asarray(result.channel_names, dtype=object),
        "meta_json": np.asarray(json.dumps(_json_safe(result.meta))),
    }
    if result.r is not None and result.chir_mag is not None:
        arrays["r"] = result.r
        arrays["chir_mag"] = result.chir_mag

    np.savez(out_path, **arrays)
    return out_path
