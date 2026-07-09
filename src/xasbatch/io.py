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
COMBINED_SUFFIX = ".bcr.combined"


def combined_stem(path: str | Path) -> str:
    """Basename with the full ``.bcr.combined`` suffix stripped.

    ``Co3NK_s.bcr.combined`` -> ``Co3NK_s`` (``Path.stem`` would leave ``Co3NK_s.bcr``).
    """
    name = Path(path).name
    if name.endswith(COMBINED_SUFFIX):
        return name[: -len(COMBINED_SUFFIX)]
    return Path(name).stem


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

    # Members block: each original file (=one scan) and how many channels it contributed.
    #   "#   BCR_Co3NK_s_043_A.001  → 30 channel(s), shift=+0.6480 eV, ..."
    # Channels sit in this order, so the counts slice the mu matrix into scans.
    members = []
    for ln in header_lines:
        m = re.match(r"#\s+(\S+)\s*(?:→|->)\s*(\d+)\s*channel", ln)
        if m:
            members.append({"name": m.group(1), "n_channels": int(m.group(2))})
    meta["members"] = members
    return meta


def scan_groups(meta: dict) -> list[tuple[str, int, int]]:
    """Column ranges ``(member_name, start, stop)`` into the μ matrix, one per scan.

    Derived from the header ``members`` block. Fails loudly if the block is missing
    or its channel counts don't sum to the number of FF columns — we refuse to guess
    the scan→channel mapping.
    """
    members = meta.get("members")
    if not members:
        raise ValueError(
            "no 'Members (kept)' block in header; cannot group channels into scans "
            "(use --mode channel to process columns individually)."
        )
    n_ff = len(meta["channel_names"])
    total = sum(m["n_channels"] for m in members)
    if total != n_ff:
        raise ValueError(
            f"member channel counts sum to {total} but there are {n_ff} FF columns; "
            "header members block is inconsistent with the data."
        )
    groups, start = [], 0
    for m in members:
        stop = start + m["n_channels"]
        groups.append((m["name"], start, stop))
        start = stop
    return groups


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


def _block_arrays(prefix: str, block) -> dict:
    """Namespaced arrays for one :class:`~xasbatch.model.ProcessBlock`."""
    arrays = {
        f"{prefix}_names": np.asarray(block.names, dtype=object),
        f"{prefix}_flat": block.flat,
        f"{prefix}_k": block.k,
        f"{prefix}_chi": block.chi,
        f"{prefix}_edge_step": block.edge_step,
    }
    if block.e0 is not None:
        arrays[f"{prefix}_e0"] = block.e0
    if block.r is not None and block.chir_mag is not None:
        arrays[f"{prefix}_r"] = block.r
        arrays[f"{prefix}_chir_mag"] = block.chir_mag
    return arrays


def _npz_arrays(result: BatchResult) -> dict:
    """Build the array dict written to an ``.npz``.

    Shared ``energy``/``e0``/``meta_json`` plus a ``scan_*`` and/or ``channel_*``
    block, depending on which were computed (``meta["modes_present"]`` lists them).
    """
    arrays = {
        "energy": result.energy,
        "e0": np.asarray(result.e0),
        "meta_json": np.asarray(json.dumps(_json_safe(result.meta))),
    }
    if result.scan is not None:
        arrays.update(_block_arrays("scan", result.scan))
    if result.channel is not None:
        arrays.update(_block_arrays("channel", result.channel))
    if result.merged is not None:
        arrays.update(_block_arrays("merged", result.merged))
    if result.scan_pass is not None:
        arrays["scan_pass"] = np.asarray(result.scan_pass, dtype=bool)
    return arrays


def save_npz(result: BatchResult, out_path: str | Path) -> Path:
    """Write a :class:`BatchResult` to an exact ``.npz`` path (parents created)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **_npz_arrays(result))
    return out_path


def save_result(result: BatchResult, outdir: str | Path) -> Path:
    """Write a :class:`BatchResult` to ``<outdir>/<sample>.npz``; returns the path.

    Filename is derived from ``meta["sample"]`` (falls back to the source stem).
    For exact-basename / tree-mirrored output use :func:`save_npz` directly.
    """
    sample = result.meta.get("sample") or combined_stem(result.meta.get("source_path", "result"))
    return save_npz(result, Path(outdir) / f"{sample}.npz")
