"""Parser/writer tests — no Larch import, run standalone and fast."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xasbatch.io import load_combined_bcr, parse_header, save_result, scan_groups
from xasbatch.model import BatchResult, ProcessBlock

FIXTURE = Path(__file__).parent / "data" / "sample_small.bcr.combined"


def test_parse_header_columns_split():
    meta = parse_header(FIXTURE)
    assert meta["channel_names"] == ["FF%d/I0" % i for i in range(1, 7)]
    assert meta["rtc_names"] == ["RTC_1", "RTC_2"]
    assert meta["energy_col"] == 0
    # energy + 6 FF + 2 RTC = 9 columns
    assert len(meta["column_names"]) == 9


def test_parse_header_members():
    meta = parse_header(FIXTURE)
    assert [m["name"] for m in meta["members"]] == [
        "BCR_Co3NK_s_043_A.001",
        "BCR_Co3NK_s_043_A.002",
    ]
    assert [m["n_channels"] for m in meta["members"]] == [3, 3]


def test_scan_groups_slices_by_member():
    meta = parse_header(FIXTURE)
    groups = scan_groups(meta)
    assert groups == [
        ("BCR_Co3NK_s_043_A.001", 0, 3),
        ("BCR_Co3NK_s_043_A.002", 3, 6),
    ]
    # counts must cover exactly the 6 FF columns
    assert sum(stop - start for _, start, stop in groups) == len(meta["channel_names"])


def test_scan_groups_missing_members_fails(tmp_path):
    # a header with no member lines -> cannot map channels to scans
    lines = [ln for ln in FIXTURE.read_text().splitlines() if "→" not in ln]
    p = tmp_path / "no_members.bcr.combined"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="Members"):
        scan_groups(parse_header(p))


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
    assert bcr.n_channels == 6
    assert bcr.mu.shape == (bcr.n_energy, 6)
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


def test_load_sorts_scrambled_and_dedups_energy(tmp_path):
    """A scrambled/duplicated energy axis loads strictly ascending (splines need it)."""
    header = [ln for ln in FIXTURE.read_text().splitlines() if ln.startswith("#")]
    data = np.loadtxt(FIXTURE, comments="#")
    scrambled = data[::-1].copy()  # fully reversed -> non-ascending
    scrambled[1, 0] = scrambled[0, 0]  # inject a duplicate energy
    p = tmp_path / "scrambled.bcr.combined"
    with p.open("w") as fh:
        fh.write("\n".join(header) + "\n")
        for row in scrambled:
            fh.write(" ".join(f"{v:.6f}" for v in row) + "\n")

    bcr = load_combined_bcr(p)
    assert np.all(np.diff(bcr.energy) > 0)  # strictly increasing after sort + dedup
    assert bcr.meta.get("n_duplicate_energies_dropped", 0) >= 1


def test_missing_columns_line_fails_loudly(tmp_path):
    p = tmp_path / "bad.bcr.combined"
    p.write_text("# Sample: nope\n7000.0 0.1\n7001.0 0.2\n")
    with pytest.raises(ValueError, match="Columns"):
        parse_header(p)


def test_save_result_roundtrip(tmp_path):
    bcr = load_combined_bcr(FIXTURE)
    nE = bcr.n_energy
    nk, n_scan = 50, 2
    scan = ProcessBlock(
        names=["scanA", "scanB"],
        flat=np.zeros((nE, n_scan)),
        k=np.linspace(0, 10, nk),
        chi=np.ones((nk, n_scan)),
        edge_step=np.ones(n_scan),
    )
    result = BatchResult(energy=bcr.energy, e0=7709.0, scan=scan, meta=bcr.meta)
    out = save_result(result, tmp_path)
    assert out.name == "Co3NK_s.npz"

    loaded = np.load(out, allow_pickle=True)
    # namespaced scan block; no channel block was written
    assert loaded["scan_k"].shape == (nk,)
    assert loaded["scan_chi"].shape == (nk, n_scan)
    assert list(loaded["scan_names"]) == ["scanA", "scanB"]
    assert "channel_k" not in loaded

    import json

    meta = json.loads(str(loaded["meta_json"]))
    assert meta["sample"] == "Co3NK_s"
