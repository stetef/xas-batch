"""Tree-runner path logic + SQLite catalog — pure, no Larch."""

from __future__ import annotations

from pathlib import Path

from xasbatch import catalog
from xasbatch.io import combined_stem
from xasbatch.tree import output_path_for


def test_combined_stem_strips_full_suffix():
    assert combined_stem("/a/b/Co3NK_s.bcr.combined") == "Co3NK_s"
    assert combined_stem(Path("x/y.bcr.combined")) == "y"


def test_output_path_sister(tmp_path):
    src = tmp_path / "sub" / "Sample_A.bcr.combined"
    src.parent.mkdir(parents=True)
    src.touch()
    out = output_path_for(src, tmp_path, None)
    assert out == src.with_name("Sample_A.npz")  # next to the source


def test_output_path_mirrors_tree(tmp_path):
    root = tmp_path / "in"
    src = root / "2018_May" / "deep" / "Sample_B.bcr.combined"
    src.parent.mkdir(parents=True)
    src.touch()
    outdir = tmp_path / "out"
    out = output_path_for(src, root, outdir)
    assert out == outdir / "2018_May" / "deep" / "Sample_B.npz"  # structure preserved


def test_catalog_record_and_resume(tmp_path):
    conn = catalog.connect(tmp_path / "cat.sqlite")
    src = "/data/Sample_C.bcr.combined"

    assert catalog.is_done(conn, src, 100.0) is False
    catalog.record(
        conn,
        {
            "source_path": src,
            "source_mtime": 100.0,
            "output_path": "/out/Sample_C.npz",
            "status": "ok",
            "e0": 7709.0,
            "e0_source": "header_e0_tab",
            "n_channels": 448,
            "element": "Co",
            "edge": "K",
            "params": {"kweight": 1},
        },
    )
    # same path + mtime -> done; changed mtime (edited source) -> not done
    assert catalog.is_done(conn, src, 100.0) is True
    assert catalog.is_done(conn, src, 200.0) is False
    assert catalog.counts(conn) == {"ok": 1}
    conn.close()


def test_catalog_error_row_not_done(tmp_path):
    conn = catalog.connect(tmp_path / "cat.sqlite")
    src = "/data/bad.bcr.combined"
    catalog.record(
        conn, {"source_path": src, "source_mtime": 5.0, "status": "error", "error": "boom"}
    )
    assert catalog.is_done(conn, src, 5.0) is False  # errors are retried on resume
    conn.close()
