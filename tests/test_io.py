"""Parser/writer tests — no Larch import, run standalone and fast."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xasbatch.io import load_combined_bcr, parse_header, save_result
from xasbatch.model import BatchResult

FIXTURE = Path(__file__).parent / "data" / "sample_small.bcr.combined"


def test_parse_header_columns_split():
    meta = parse_header(FIXTURE)
    assert meta["channel_names"] == ["FF1/I0", "FF2/I0", "FF3/I0", "FF4/I0"]
    assert meta["rtc_names"] == ["RTC_1", "RTC_2"]
    assert meta["energy_col"] == 0
    # energy + 4 FF + 2 RTC = 7 columns
    assert len(meta["column_names"]) == 7


def test_parse_header_metadata():
    meta = parse_header(FIXTURE)
    assert meta["element"] == "Co"
    assert meta["edge"] == "K"
    assert meta["e0_tab"] == pytest.approx(7709.0)
    assert meta["sample"] == "Co3NK_s"
    assert meta["session"] == "2017_7-3_Apr"
    assert meta["k_max"] == pytest.approx(15.02, abs=0.01)


def test_load_shapes_and_rtc_excluded():
    bcr = load_combined_bcr(FIXTURE)
    assert bcr.n_channels == 4
    assert bcr.mu.shape == (bcr.n_energy, 4)
    # RTC columns are stashed for provenance, never mixed into mu
    assert bcr.rtc is not None
    assert bcr.rtc.shape == (bcr.n_energy, 2)
    assert bcr.mu.shape[1] + bcr.rtc.shape[1] + 1 == len(bcr.meta["column_names"])


def test_energy_ascending():
    bcr = load_combined_bcr(FIXTURE)
    assert np.all(np.diff(bcr.energy) > 0)


def test_descending_energy_is_flipped(tmp_path):
    """A file stored with descending energy must load ascending, rows realigned."""
    lines = FIXTURE.read_text().splitlines()
    header = [ln for ln in lines if ln.startswith("#")]
    data = np.loadtxt(FIXTURE, comments="#")
    flipped = data[::-1]  # reverse row order -> descending energy

    p = tmp_path / "descending.bcr.combined"
    with p.open("w") as fh:
        fh.write("\n".join(header) + "\n")
        for row in flipped:
            fh.write(" ".join(f"{v:.6f}" for v in row) + "\n")

    bcr = load_combined_bcr(p)
    assert np.all(np.diff(bcr.energy) > 0)
    # mu must be reordered together with energy: compare against ascending reference
    ref = load_combined_bcr(FIXTURE)
    np.testing.assert_allclose(bcr.energy, ref.energy)
    np.testing.assert_allclose(bcr.mu, ref.mu)


def test_missing_columns_line_fails_loudly(tmp_path):
    p = tmp_path / "bad.bcr.combined"
    p.write_text("# Sample: nope\n7000.0 0.1\n7001.0 0.2\n")
    with pytest.raises(ValueError, match="Columns"):
        parse_header(p)


def test_save_result_roundtrip(tmp_path):
    bcr = load_combined_bcr(FIXTURE)
    nE, nFF = bcr.n_energy, bcr.n_channels
    nk = 50
    result = BatchResult(
        energy=bcr.energy,
        flat=np.zeros((nE, nFF)),
        k=np.linspace(0, 10, nk),
        chi=np.ones((nk, nFF)),
        e0=7709.0,
        edge_step=np.ones(nFF),
        channel_names=bcr.channel_names,
        meta=bcr.meta,
    )
    out = save_result(result, tmp_path)
    assert out.name == "Co3NK_s.npz"

    loaded = np.load(out, allow_pickle=True)
    assert loaded["k"].shape == (nk,)
    assert loaded["chi"].shape == (nk, nFF)
    assert list(loaded["channel_names"]) == bcr.channel_names
    import json

    meta = json.loads(str(loaded["meta_json"]))
    assert meta["sample"] == "Co3NK_s"
