"""SQLite catalog for the tree-processing run.

One row per source file — a resume ledger *and* a queryable provenance index for
downstream (denoising/ML) work. The spectra themselves live in the ``.npz`` files;
this table only records where they are and how they were made.

Only the main process writes here (workers return records), so there is no write
contention; WAL is still enabled so external readers can query mid-run.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    source_path  TEXT PRIMARY KEY,
    source_mtime REAL,
    output_path  TEXT,
    status       TEXT,          -- 'ok' | 'error'
    e0           REAL,
    e0_source    TEXT,
    n_channels   INTEGER,
    element      TEXT,
    edge         TEXT,
    params_json  TEXT,
    error        TEXT,
    updated_at   TEXT
);
"""

COLUMNS = (
    "source_path",
    "source_mtime",
    "output_path",
    "status",
    "e0",
    "e0_source",
    "n_channels",
    "element",
    "edge",
    "params_json",
    "error",
    "updated_at",
)


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the catalog DB and ensure the schema exists."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def is_done(conn: sqlite3.Connection, source_path: str, source_mtime: float) -> bool:
    """True if this exact source (path + mtime) already has a successful row."""
    row = conn.execute(
        "SELECT status, source_mtime FROM files WHERE source_path = ?", (str(source_path),)
    ).fetchone()
    if row is None or row["status"] != "ok":
        return False
    # float mtime equality with a tiny tolerance for filesystem rounding
    return abs((row["source_mtime"] or -1.0) - float(source_mtime)) < 1e-6


def record(conn: sqlite3.Connection, rec: dict) -> None:
    """Upsert one file's result. Missing keys default to NULL; ``params`` is JSON-encoded."""
    row = dict(rec)
    if "params_json" not in row and "params" in row:
        row["params_json"] = json.dumps(row.pop("params"))
    row.setdefault("updated_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
    values = [row.get(c) for c in COLUMNS]
    placeholders = ",".join("?" for _ in COLUMNS)
    updates = ",".join(f"{c}=excluded.{c}" for c in COLUMNS if c != "source_path")
    conn.execute(
        f"INSERT INTO files ({','.join(COLUMNS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(source_path) DO UPDATE SET {updates}",
        values,
    )
    conn.commit()


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Return {status: count} across the whole catalog."""
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM files GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}
