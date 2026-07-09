"""Numerics tests on the trimmed fixture. Sane, not golden."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xasbatch.io import load_combined_bcr
from xasbatch.model import Params
from xasbatch.process import _sum_scans, find_edge, process_batch

FIXTURE = Path(__file__).parent / "data" / "sample_small.bcr.combined"
# fixture: 2 scans x 3 FF channels
N_SCANS = 2
N_CHANNELS = 6


@pytest.fixture(scope="module")
def bcr():
    return load_combined_bcr(FIXTURE)


def test_scan_mode_default(bcr):
    result = process_batch(bcr, Params())  # mode defaults to "scan"
    assert result.meta["mode"] == "scan"
    assert result.scan is not None
    assert result.channel is None
    assert result.n_scans == N_SCANS
    # one summed mu(E) per original file, flattened per scan
    assert result.scan.flat.shape == (bcr.n_energy, N_SCANS)
    assert result.scan.chi.shape[1] == N_SCANS
    assert result.scan.chi.shape[0] == result.scan.k.size
    assert result.scan.names[0].startswith("BCR_")


def test_channel_mode(bcr):
    result = process_batch(bcr, Params(mode="channel"))
    assert result.scan is None
    assert result.channel is not None
    assert result.n_channels == N_CHANNELS
    assert result.channel.chi.shape[1] == N_CHANNELS


def test_both_mode_shares_e0_and_kgrid(bcr):
    result = process_batch(bcr, Params(mode="both"))
    assert result.scan is not None and result.channel is not None
    assert result.n_scans == N_SCANS and result.n_channels == N_CHANNELS
    assert result.meta["modes_present"] == ["scan", "channel"]
    # single e0 for the file -> scan and channel blocks land on the same k-grid
    np.testing.assert_allclose(result.scan.k, result.channel.k)


def test_sum_scans_groups_and_sums(bcr):
    """Each scan μ(E) is the nansum of exactly that original file's channels."""
    names, scan_mu, members = _sum_scans(bcr)
    assert names == [m["name"] for m in members]
    assert scan_mu.shape == (bcr.n_energy, N_SCANS)
    # members block says 3 + 3 channels
    assert [m["n_channels"] for m in members] == [3, 3]
    np.testing.assert_allclose(scan_mu[:, 0], np.nansum(bcr.mu[:, 0:3], axis=1))
    np.testing.assert_allclose(scan_mu[:, 1], np.nansum(bcr.mu[:, 3:6], axis=1))


def test_flattening_flat_far_post_edge(bcr):
    """Data-spanning norm range should flatten to ~1.0 far above the edge.

    Guards against the narrow-window default (norm2=300) whose extrapolated
    polynomial droops badly at high energy.
    """
    result = process_batch(bcr, Params(mode="scan"))
    hi = result.energy > (result.e0 + 400)
    flat_hi = result.scan.flat[hi, 0]
    assert abs(float(np.nanmean(flat_hi)) - 1.0) < 0.05
    assert float(np.nanmax(flat_hi) - np.nanmin(flat_hi)) < 0.1


def test_edge_step_positive(bcr):
    result = process_batch(bcr, Params(mode="both"))
    assert np.all(result.scan.edge_step > 0)
    assert np.all(result.channel.edge_step > 0)


def test_e0_defaults_to_find_e0(bcr):
    result = process_batch(bcr, Params())  # default is find_e0 on the merged μ
    assert result.meta["e0_source"] == "find_e0"
    # find_e0 returns the derivative-max, a few eV ABOVE the tabulated edge (7709).
    assert 7709.0 <= result.e0 <= 7729.0


def test_per_scan_e0_stored_and_near_merged(bcr):
    result = process_batch(bcr, Params(mode="scan"))
    # per-scan e0 vector present, one per scan
    assert result.scan.e0 is not None
    assert result.scan.e0.shape == (N_SCANS,)
    # scans are high-SNR sums -> per-scan e0 tight and centered on the merged e0
    assert result.meta["e0_scan_std"] < 1.0
    assert abs(result.meta["e0_scan_mean"] - result.meta["e0_merged"]) < 1.0
    # channel block (shared merged e0) all-equal when present
    ch = process_batch(bcr, Params(mode="channel")).channel
    assert np.allclose(ch.e0, ch.e0[0])


def test_header_e0_when_requested(bcr):
    result = process_batch(bcr, Params(auto_e0=False))
    assert result.meta["e0_source"] == "header_e0_tab"
    assert result.e0 == pytest.approx(7709.0)


def test_find_edge_window_rejects_out_of_window_glitch():
    """A far glitch must not hijack e0 when a search guess is supplied."""
    E = np.linspace(7389, 8569, 900)
    mu = 1.0 / (1.0 + np.exp(-(E - 7714.0) / 2.0))  # smooth edge step at ~7714
    mu[700] += 50.0  # spurious spike far above the edge (~8305)
    e0 = find_edge(E, mu, e0_guess=7709.0)  # window ±25 excludes the spike
    assert 7705.0 <= e0 <= 7725.0


def test_explicit_e0_overrides(bcr):
    result = process_batch(bcr, Params(e0=7710.5))
    assert result.meta["e0_source"] == "explicit"
    assert result.e0 == pytest.approx(7710.5)


def test_invalid_mode_raises(bcr):
    with pytest.raises(ValueError, match="mode"):
        process_batch(bcr, Params(mode="nonsense"))


def test_forward_ft_optional(bcr):
    result = process_batch(bcr, Params(mode="scan", ft=True))
    assert result.scan.r is not None
    assert result.scan.chir_mag is not None
    assert result.scan.chir_mag.shape[0] == result.scan.r.size
    assert result.scan.chir_mag.shape[1] == N_SCANS
