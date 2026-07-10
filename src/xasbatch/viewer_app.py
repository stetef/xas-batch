"""``xas-batch-view`` — interactive fit viewer (Streamlit + Plotly).

Reads the SQLite catalog written by ``xas-batch-tree``, lets you search / filter
the processed samples, and renders the four processing figures **live** from the
source ``.bcr.combined`` — the same pipeline as the batch, so the view always
matches what was catalogued. Plotly gives native zoom / pan / hover.

Launch it with the console script (recommended)::

    xas-batch-view                 # discovers ./out/xas_catalog.sqlite or the .env
    xas-batch-view --db path.sqlite

or directly::

    uv run streamlit run src/xasbatch/viewer_app.py

Nothing here is imported by the batch pipeline, so Streamlit/Plotly stay optional.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

import streamlit as st

from xasbatch.io import combined_stem
from xasbatch.model import Params
from xasbatch.tree import env_get, load_env

st.set_page_config(page_title="XAS fit viewer", page_icon="📈", layout="wide")

# Controlled vocabulary of common processing problems — keeps exported logs
# aggregatable (free text still available via the note field).
QC_ISSUES = [
    "Bad E₀ / edge misidentified",
    "Poor pre-edge fit",
    "Poor post-edge / normalization",
    "AUTOBK background over/under-subtracted",
    "Glitch / spike in μ(E)",
    "Truncated / insufficient k-range",
    "Noisy / low SNR",
    "Scan(s) should be excluded (QC missed a bad scan)",
    "Scan(s) wrongly excluded (QC too aggressive)",
    "Energy grid / calibration issue",
    "Wrong element / multiple edges",
    "Other (see note)",
]
_FLAG_LABELS = {"— none —": None, "🟡 yellow": "yellow", "🔴 red": "red"}


# ── catalog discovery ─────────────────────────────────────────────────────────
def discover_db() -> Path | None:
    """Best-effort default catalog path: launcher --db → ?db → .env → ./out fallback."""
    import os

    launcher_db = os.environ.get("XAS_VIEW_DB")
    if launcher_db and Path(launcher_db).expanduser().exists():
        return Path(launcher_db).expanduser()

    qp = st.query_params.get("db")
    if qp and Path(qp).expanduser().exists():
        return Path(qp).expanduser()

    env = load_env(".env")
    db_val = env_get(env, "XAS_DB_PATH")
    if db_val and Path(db_val).expanduser().exists():
        return Path(db_val).expanduser()

    base = env_get(env, "XAS_OUTPUT_DIR") or env_get(env, "XAS_INPUT_ROOT")
    if base:
        cand = Path(base).expanduser() / "xas_catalog.sqlite"
        if cand.exists():
            return cand

    for cand in (Path("out/xas_catalog.sqlite"), Path("xas_catalog.sqlite")):
        if cand.exists():
            return cand.resolve()
    return None


def input_root() -> Path | None:
    """The ``XAS_INPUT_ROOT`` from ``.env`` (the run folder that holds the sources)."""
    val = env_get(load_env(".env"), "XAS_INPUT_ROOT")
    return Path(val).expanduser() if val else None


def display_path(source_path: str, root: Path | None) -> str:
    """Path relative to the run folder's parent — e.g. ``combined-files-…/session/file``.

    Keeps the run-folder name as the prefix and never leaks the absolute local path.
    Falls back to the bare filename if the source lives outside the root.
    """
    p = Path(source_path)
    if root is not None:
        anchor = root.parent  # so the run-folder name stays in the shown path
        try:
            return str(p.resolve().relative_to(anchor.resolve()))
        except ValueError:
            pass
    return p.name


# ── DB access (cached) ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_rows(db_path: str, mtime: float) -> list[dict]:
    """All catalog rows, enriched with derived ``sample`` and ``session``.

    ``mtime`` is part of the cache key so the list refreshes when the catalog is
    rewritten by a new batch run.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT source_path, source_mtime, status, mode, e0, e0_source, "
        "n_scans, n_channels, element, edge, params_json, error "
        "FROM files ORDER BY source_path"
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["sample"] = combined_stem(d["source_path"])
        d["session"] = Path(d["source_path"]).parent.name
        out.append(d)
    return out


def params_from_row(row: dict) -> Params:
    """Reconstruct the exact batch Params from the catalog's ``params_json``.

    Falls back to scan-mode defaults if absent or from an older schema. Always
    renders in scan mode — the figures are per-scan regardless of batch mode.
    """
    try:
        data = json.loads(row.get("params_json") or "{}")
        fields = set(Params.__dataclass_fields__)
        params = Params(**{k: v for k, v in data.items() if k in fields})
    except Exception:  # noqa: BLE001 — any bad/old JSON → sane defaults
        params = Params()
    params.mode = "scan"
    return params


@st.cache_data(show_spinner="Processing scans…")
def build_figures(source_path: str, source_mtime: float, params_json: str, kweight: int,
                  render_mtime: float):
    """Load one file and build its four Plotly figures (cached per file+params).

    ``source_mtime`` and ``params_json`` invalidate the entry when a file is
    reprocessed or params change; ``render_mtime`` (the mtime of ``plotlyplots.py``)
    invalidates it whenever the plotting code itself is edited — otherwise stale
    figures would survive a layout change. Returns picklable ``go.Figure``s.
    """
    from xasbatch.io import load_combined_bcr
    from xasbatch.plotlyplots import figure_report_plotly

    bcr = load_combined_bcr(source_path)
    figs = figure_report_plotly(bcr, params_from_row({"params_json": params_json}), kweight=kweight)
    return figs, dict(bcr.meta)


def _render_mtime() -> float:
    """mtime of the Plotly rendering module — used to bust the figure cache on edits."""
    import xasbatch.plotlyplots as pp

    return Path(pp.__file__).stat().st_mtime


# ── QC review (session-scoped flags → downloadable log) ───────────────────────────
def _qc_records() -> dict:
    """The session's QC store: ``{source_path: record}`` (survives reruns, not reload)."""
    return st.session_state.setdefault("qc_records", {})


def build_qc_text(recs: dict, exported: str) -> str:
    """Human-readable QC log, red flags first, then yellow, then note-only."""
    order = {"red": 0, "yellow": 1, None: 2}
    items = sorted(recs.values(), key=lambda r: (order.get(r["flag"], 3), r["sample"] or ""))
    lines = [
        "# XAS QC flags",
        f"# exported: {exported}",
        f"# {len(items)} sample(s) flagged  "
        f"(🔴 {sum(r['flag'] == 'red' for r in items)}, "
        f"🟡 {sum(r['flag'] == 'yellow' for r in items)})",
        "# per entry: [FLAG] sample (element edge) / path / issues / note",
        "",
    ]
    for r in items:
        el = f"{r['element'] or '—'} {r['edge'] or ''}".strip()
        lines += [
            f"[{(r['flag'] or 'note').upper()}] {r['sample']}  ({el})",
            f"  path:   {r['path']}",
            f"  issues: {'; '.join(r['issues']) if r['issues'] else '—'}",
            f"  note:   {r['note'] or '—'}",
            "",
        ]
    return "\n".join(lines)


def build_qc_jsonl(recs: dict, exported: str) -> str:
    """One JSON object per flagged sample — machine-readable for driving reprocessing."""
    return "".join(
        json.dumps({"exported": exported, **r}, ensure_ascii=False) + "\n"
        for r in recs.values()
    )


def render_qc_sidebar(chosen: dict, root: Path | None) -> None:
    """Sidebar QC panel for the current sample; accumulates flags + offers downloads."""
    sp = chosen["source_path"]
    with st.sidebar:
        st.divider()
        st.header("🏷️ QC review")
        st.caption(f"Tagging: **{chosen['sample']}**")

        # qc_records is the source of truth. Streamlit purges an unrendered widget's
        # key, so we can't lean on per-sample keys alone — instead we seed each widget
        # from the saved record, so revisiting a sample restores its flag/issues/note.
        recs = _qc_records()
        saved = recs.get(sp, {})
        labels = list(_FLAG_LABELS)
        inv = {v: k for k, v in _FLAG_LABELS.items()}
        flag_label = st.radio("Flag", labels, index=labels.index(inv[saved.get("flag")]),
                              horizontal=True, key=f"qc_flag::{sp}")
        issues = st.multiselect("Issue(s)", QC_ISSUES, default=saved.get("issues", []),
                                key=f"qc_issues::{sp}",
                                help="Pick any that apply; combine with a free note below.")
        note = st.text_area("Note", value=saved.get("note", ""), key=f"qc_note::{sp}",
                            placeholder="what looks wrong / how to fix it")

        flag = _FLAG_LABELS[flag_label]
        if flag or issues or note.strip():
            recs[sp] = {
                "flag": flag, "issues": issues, "note": note.strip(),
                "sample": chosen["sample"], "session": chosen["session"],
                "element": chosen["element"], "edge": chosen["edge"],
                "path": display_path(sp, root),
            }
        else:  # cleared back to nothing → drop from the log
            recs.pop(sp, None)

        n_red = sum(r["flag"] == "red" for r in recs.values())
        n_yel = sum(r["flag"] == "yellow" for r in recs.values())
        st.caption(f"{len(recs)} flagged this session · 🔴 {n_red} · 🟡 {n_yel}")

        if recs:
            exported = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            st.download_button("⬇️ QC log (.txt)", data=build_qc_text(recs, exported),
                               file_name="xas_qc_flags.txt", mime="text/plain", width="stretch")
            st.download_button("⬇️ QC log (.jsonl)", data=build_qc_jsonl(recs, exported),
                               file_name="xas_qc_flags.jsonl", mime="application/x-ndjson",
                               width="stretch")
            if st.button("Clear all flags", width="stretch"):
                recs.clear()
                st.rerun()


# ── UI ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("📈 XAS fit viewer")

    with st.sidebar:
        st.header("Catalog")
        default_db = discover_db()
        db_str = st.text_input(
            "Catalog path (.sqlite)",
            value=str(default_db) if default_db else "",
            help="Written by xas-batch-tree. Auto-discovered from .env / ./out.",
        )
        kweight = st.radio("χ(k) k-weight", options=[1, 2, 3], index=2, horizontal=True)

    if not db_str or not Path(db_str).expanduser().exists():
        st.info("Point the sidebar at a `xas_catalog.sqlite` to begin.")
        st.stop()

    db_path = str(Path(db_str).expanduser())
    rows = load_rows(db_path, Path(db_path).stat().st_mtime)
    if not rows:
        st.warning("Catalog is empty.")
        st.stop()

    # ── filters ────────────────────────────────────────────────────────────────
    st.caption(f"{len(rows)} files catalogued · {sum(r['status']=='ok' for r in rows)} ok")
    elements = sorted({r["element"] for r in rows if r["element"]})
    sessions = sorted({r["session"] for r in rows if r["session"]})

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    query = fc1.text_input("🔍 Search sample", placeholder="e.g. Co3NK").strip().lower()
    sel_el = fc2.multiselect("Element", elements)
    sel_ses = fc3.multiselect("Session", sessions)

    def keep(r: dict) -> bool:
        if query and query not in r["sample"].lower():
            return False
        if sel_el and r["element"] not in sel_el:
            return False
        if sel_ses and r["session"] not in sel_ses:
            return False
        return True

    matches = [r for r in rows if keep(r)]
    if not matches:
        st.warning("No samples match the current filters.")
        st.stop()

    def label(r: dict) -> str:
        flag = "" if r["status"] == "ok" else f"  ⚠️ {r['status']}"
        return f"{r['sample']}  ·  {r['session']}  ·  {r['element'] or '—'} {r['edge'] or ''}{flag}"

    st.caption(f"{len(matches)} match(es)")
    chosen = st.selectbox("Sample", matches, format_func=label)

    # ── selected sample ─────────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Element / edge", f"{chosen['element'] or '—'} {chosen['edge'] or ''}")
    m2.metric("E₀ (merged)", f"{chosen['e0']:.2f} eV" if chosen["e0"] is not None else "—")
    m3.metric("Scans", chosen["n_scans"] if chosen["n_scans"] is not None else "—")
    m4.metric("e₀ source", chosen["e0_source"] or "—")
    root = input_root()
    st.caption(f"`{display_path(chosen['source_path'], root)}`")

    # QC panel renders for any chosen sample (including error/skipped ones you'd flag).
    render_qc_sidebar(chosen, root)

    if chosen["status"] != "ok":
        st.error(f"This file was **{chosen['status']}** during batch: {chosen['error'] or ''}")
        if not Path(chosen["source_path"]).exists():
            st.stop()
        st.info("Attempting to render anyway from the source file…")

    if not Path(chosen["source_path"]).exists():
        st.error("Source `.bcr.combined` not found on disk — cannot render live.")
        st.stop()

    try:
        figs, meta = build_figures(
            chosen["source_path"],
            Path(chosen["source_path"]).stat().st_mtime,
            chosen.get("params_json") or "{}",
            kweight,
            _render_mtime(),
        )
    except Exception as exc:  # noqa: BLE001 — surface processing failures in the UI
        st.exception(exc)
        st.stop()

    titles = {
        "1_raw": "1 · Raw summed scans",
        "2_norm_fits": "2 · Per-scan normalization fits",
        "3_flat": "3 · Flattened overlay",
        "4_exafs": "4 · EXAFS: μ + AUTOBK spline and kⁿ·χ(k)",
    }
    tabs = st.tabs([titles.get(lbl, lbl) for lbl, _ in figs])
    for tab, (lbl, fig) in zip(tabs, figs):
        with tab:
            st.plotly_chart(fig, width="stretch", key=f"chart_{lbl}")

    with st.expander("Header metadata"):
        shown = {k: v for k, v in meta.items()
                 if k not in ("members", "column_names", "channel_names", "rtc_names",
                              "mu_cols", "rtc_cols", "scan_members")}
        if "source_path" in shown:
            shown["source_path"] = display_path(str(shown["source_path"]), root)
        st.json(shown)


if __name__ == "__main__":
    main()
