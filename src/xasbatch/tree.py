"""``xas-batch-tree`` — mass-process a whole directory tree of combined-BCR files.

Reads an ``.env`` for the input root (and optional output dir + catalog path),
recursively finds every ``*.bcr.combined``, and processes each to an ``.npz`` —
either as a sister file next to the source, or mirrored into an output dir. A
SQLite catalog records every file (resume ledger + queryable provenance index),
and files can be processed in parallel across a process pool.

Design: workers do all the heavy, independent work (load → process → save npz)
and return a small record; only this main process writes the catalog, so there is
no DB write-contention. Larch is imported lazily inside the worker, keeping the
orchestrator light (fast ``--help``, fast resume-skip).

.env keys (CLI flags override):
    XAS_INPUT_ROOT   required — root dir to scan recursively
    XAS_OUTPUT_DIR   optional — mirror outputs here; unset -> sister .npz files
    XAS_DB_PATH      optional — catalog path; default <out-or-root>/xas_catalog.sqlite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from xasbatch import catalog
from xasbatch.cli import GLOB, add_param_args, params_from_args
from xasbatch.io import combined_stem
from xasbatch.model import Params


# --------------------------------------------------------------------------- env
def load_env(env_path: str | Path) -> dict:
    """Return values from the ``.env`` file (empty if it does not exist)."""
    from dotenv import dotenv_values

    p = Path(env_path)
    return dict(dotenv_values(p)) if p.exists() else {}


def env_get(values: dict, key: str, default=None):
    """Prefer the .env file, then the real environment, then default."""
    val = values.get(key)
    if val is None or val == "":
        val = os.environ.get(key)
    return val if (val is not None and val != "") else default


# ---------------------------------------------------------------------- format
def fmt_duration(seconds: float) -> str:
    """Human-readable elapsed time, e.g. ``1h 23m 45s`` / ``2m 03s`` / ``12.4s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


# ----------------------------------------------------------------- output paths
def output_path_for(src: Path, input_root: Path, output_dir: Path | None) -> Path:
    """Target ``.npz`` path: sister file, or mirrored under ``output_dir``."""
    stem = combined_stem(src)
    if output_dir is None:
        return src.with_name(stem + ".npz")
    rel = src.resolve().relative_to(input_root.resolve()).parent
    return output_dir / rel / (stem + ".npz")


def plot_dir_for(src: Path, input_root: Path, plots_root: Path) -> Path:
    """Per-sample PNG directory: ``<plots_root>/<mirrored rel>/<sample>/``.

    Mirrors the input tree (like :func:`output_path_for`) so same-named samples in
    different sessions don't collide.
    """
    rel = src.resolve().relative_to(input_root.resolve()).parent
    return plots_root / rel / combined_stem(src)


# ---------------------------------------------------------------------- worker
def _save_plots(bcr, params: Params, plot_dir: str) -> None:
    """Render the four PNGs for one file into ``plot_dir`` (headless Agg)."""
    import matplotlib

    matplotlib.use("Agg")  # headless save; safe to call per worker
    import matplotlib.pyplot as plt

    from xasbatch.plotting import figure_report

    out = Path(plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    for label, fig in figure_report(bcr, params):
        fig.savefig(out / f"{label}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def _process_one(task: tuple[str, str, Params, str | None]) -> dict:
    """Load → process → save one file; return a catalog record. Never raises."""
    src, out_path, params, plot_dir = task
    src_p = Path(src)
    rec = {
        "source_path": str(src_p),
        "source_mtime": src_p.stat().st_mtime,
        "output_path": str(out_path),
        "params_json": json.dumps(asdict(params)),
    }
    try:
        # imported here so Larch loads in the worker, not the orchestrator
        from xasbatch.io import load_combined_bcr, save_npz
        from xasbatch.process import SkipFile, process_batch

        bcr = load_combined_bcr(src_p)
        try:
            result = process_batch(bcr, params)
        except SkipFile as exc:  # expected, clean skip (e.g. too little post-edge range)
            rec.update(status="skipped", element=bcr.meta.get("element"),
                       edge=bcr.meta.get("edge"), error=str(exc))
            return rec
        save_npz(result, out_path)
        if plot_dir is not None:
            # opt-in PNGs; a plotting failure must not undo the successful npz
            try:
                _save_plots(bcr, params, plot_dir)
            except Exception as exc:  # noqa: BLE001
                rec["plot_error"] = f"{type(exc).__name__}: {exc}"
        rec.update(
            status="ok",
            mode=result.meta.get("mode"),
            e0=result.e0,
            e0_source=result.meta.get("e0_source"),
            n_scans=result.n_scans,
            n_channels=result.meta.get("n_channels_raw"),
            element=bcr.meta.get("element"),
            edge=bcr.meta.get("edge"),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - record and keep the batch going
        rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return rec


# ------------------------------------------------------------------------ cli
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xas-batch-tree",
        description="Recursively process a tree of .bcr.combined files (parallel + SQLite catalog).",
    )
    p.add_argument("--env", type=Path, default=Path(".env"), help="path to .env (default: ./.env)")
    p.add_argument("--input-root", type=Path, default=None, help="override XAS_INPUT_ROOT")
    p.add_argument("--output-dir", type=Path, default=None, help="override XAS_OUTPUT_DIR")
    p.add_argument("--db", type=Path, default=None, help="override catalog path (XAS_DB_PATH)")
    p.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="worker processes (default: cpu_count-1; 1 = serial)",
    )
    p.add_argument("--force", action="store_true", help="reprocess even files the catalog marks done")
    p.add_argument("--limit", type=int, default=None, help="process at most N files (for testing)")
    p.add_argument("--plots", action="store_true",
                   help="also render the 4 processing PNGs per sample (needs matplotlib)")
    p.add_argument("--plots-dir", type=Path, default=None,
                   help="where PNGs go (default: XAS_PLOTS_DIR, else <out-or-root>/plots)")
    add_param_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    from tqdm import tqdm

    args = build_parser().parse_args(argv)
    env = load_env(args.env)

    root_val = args.input_root or env_get(env, "XAS_INPUT_ROOT")
    if not root_val:
        print("no input root: set XAS_INPUT_ROOT in .env or pass --input-root", file=sys.stderr)
        return 2
    input_root = Path(root_val).expanduser()
    if not input_root.is_dir():
        print(f"input root is not a directory: {input_root}", file=sys.stderr)
        return 2

    out_val = args.output_dir or env_get(env, "XAS_OUTPUT_DIR")
    output_dir = Path(out_val).expanduser() if out_val else None

    db_val = args.db or env_get(env, "XAS_DB_PATH")
    if db_val:
        db_path = Path(db_val).expanduser()
    else:
        db_path = (output_dir or input_root) / "xas_catalog.sqlite"

    params = params_from_args(args)

    plots_root = None
    if args.plots:
        try:
            import matplotlib  # noqa: F401
        except ModuleNotFoundError:
            print("--plots needs matplotlib: install the plot extra "
                  "(uv pip install -e '.[plot]')", file=sys.stderr)
            return 2
        plots_val = args.plots_dir or env_get(env, "XAS_PLOTS_DIR")
        plots_root = (Path(plots_val).expanduser() if plots_val
                      else (output_dir or input_root) / "plots")

    files = sorted(input_root.rglob(GLOB))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print(f"no {GLOB} files found under {input_root}", file=sys.stderr)
        return 1

    conn = catalog.connect(db_path)

    tasks, skipped = [], 0
    for src in files:
        mtime = src.stat().st_mtime
        if not args.force and catalog.is_done(conn, str(src), mtime):
            skipped += 1
            continue
        plot_dir = str(plot_dir_for(src, input_root, plots_root)) if plots_root else None
        tasks.append((str(src), str(output_path_for(src, input_root, output_dir)), params, plot_dir))

    dest = "sister files" if output_dir is None else str(output_dir)
    plots_note = f" | plots -> {plots_root}" if plots_root else ""
    print(
        f"{len(files)} files under {input_root} | {skipped} already done, "
        f"{len(tasks)} to process | jobs={args.jobs} | -> {dest}{plots_note} | catalog: {db_path}"
    )

    ok = err = qc_skipped = 0

    def handle(rec: dict) -> None:
        nonlocal ok, err, qc_skipped
        catalog.record(conn, rec)
        name = Path(rec["source_path"]).name
        if rec.get("plot_error"):
            tqdm.write(f"PLOT  {name}: {rec['plot_error']} (npz still saved)")
        if rec["status"] == "ok":
            ok += 1
        elif rec["status"] == "skipped":
            qc_skipped += 1
            tqdm.write(f"SKIP  {name}: {rec['error']}")
        else:
            err += 1
            tqdm.write(f"FAIL  {name}: {rec['error']}")

    t0 = time.perf_counter()
    if not tasks:
        print("nothing to do.")
    elif args.jobs > 1 and len(tasks) > 1:
        import multiprocessing as mp

        with mp.Pool(args.jobs) as pool:
            for rec in tqdm(pool.imap_unordered(_process_one, tasks), total=len(tasks), desc="XAS"):
                handle(rec)
    else:
        for task in tqdm(tasks, desc="XAS"):
            handle(_process_one(task))
    elapsed = time.perf_counter() - t0

    total = catalog.add_elapsed(conn, elapsed) if tasks else catalog.total_elapsed(conn)
    conn.close()
    print(
        f"done: {ok} ok, {qc_skipped} skipped (QC), {err} error, "
        f"{skipped} already-done ({len(files)} total)."
    )
    print(f"time: {fmt_duration(elapsed)} this run | {fmt_duration(total)} cumulative (all runs).")
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
