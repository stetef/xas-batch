"""Tests for the viewer's pure helpers (path display, params, QC log builders).

Skipped when streamlit isn't installed (optional ``viewer`` extra). Importing the
module runs ``st.set_page_config`` in bare mode — harmless (warns, doesn't raise).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("streamlit")  # optional dep; skip the whole module without it

from xasbatch.model import Params  # noqa: E402
from xasbatch.viewer_app import (  # noqa: E402
    build_qc_jsonl,
    build_qc_text,
    display_path,
    params_from_row,
)


def test_display_path_relative_to_run_folder(tmp_path):
    root = tmp_path / "data" / "combined-files-2026-05-29"
    src = root / "2018_May" / "S.bcr.combined"
    src.parent.mkdir(parents=True)
    src.touch()
    # relativized against the run folder's parent, keeping the run-folder prefix
    assert display_path(str(src), root) == "combined-files-2026-05-29/2018_May/S.bcr.combined"


def test_display_path_outside_root_falls_back_to_name(tmp_path):
    root = tmp_path / "data" / "run"
    root.mkdir(parents=True)
    assert display_path("/somewhere/else/Other.bcr.combined", root) == "Other.bcr.combined"
    assert display_path("/x/y/Z.bcr.combined", None) == "Z.bcr.combined"


def test_params_from_row_roundtrip_and_forces_scan():
    p = Params(mode="channel", rbkg=1.4, kmin=2.0, nnorm=3)
    row = {"params_json": json.dumps(p.__dict__)}
    got = params_from_row(row)
    assert got.rbkg == 1.4 and got.kmin == 2.0 and got.nnorm == 3
    assert got.mode == "scan"  # always render per-scan


def test_params_from_row_bad_json_defaults():
    got = params_from_row({"params_json": "not json{"})
    assert got.mode == "scan"
    assert got == Params(mode="scan")  # otherwise all defaults


def _recs():
    return {
        "/a/combined-files-X/s1/CoFoo.bcr.combined": {
            "flag": "red", "issues": ["Poor pre-edge fit", "Glitch / spike in μ(E)"],
            "note": "spike ~7900 eV", "sample": "CoFoo", "session": "s1",
            "element": "Co", "edge": "K", "path": "combined-files-X/s1/CoFoo.bcr.combined",
        },
        "/a/combined-files-X/s2/CuBar.bcr.combined": {
            "flag": "yellow", "issues": [], "note": "noisy tail", "sample": "CuBar",
            "session": "s2", "element": "Cu", "edge": "K",
            "path": "combined-files-X/s2/CuBar.bcr.combined",
        },
    }


def test_build_qc_text_orders_red_first_and_counts():
    txt = build_qc_text(_recs(), "2026-07-09T16:30:00-07:00")
    assert "# 2 sample(s) flagged" in txt
    # red block precedes yellow block
    assert txt.index("[RED] CoFoo") < txt.index("[YELLOW] CuBar")
    assert "issues: Poor pre-edge fit; Glitch / spike in μ(E)" in txt
    assert "note:   noisy tail" in txt


def test_build_qc_jsonl_one_valid_object_per_record():
    lines = build_qc_jsonl(_recs(), "2026-07-09T16:30:00-07:00").splitlines()
    assert len(lines) == 2
    objs = [json.loads(ln) for ln in lines]
    assert {o["sample"] for o in objs} == {"CoFoo", "CuBar"}
    assert all(o["exported"] == "2026-07-09T16:30:00-07:00" for o in objs)
    assert objs[0]["path"].startswith("combined-files-X/")  # relative, no absolute leak
