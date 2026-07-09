"""Numerics tests on the trimmed fixture. Sane, not golden."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xasbatch.io import load_combined_bcr
from xasbatch.model import Params
from xasbatch.process import process_batch

FIXTURE = Path(__file__).parent / "data" / "sample_small.bcr.combined"


@pytest.fixture(scope="module")
def bcr():
    return load_combined_bcr(FIXTURE)


def test_batch_shapes(bcr):
    result = process_batch(bcr, Params())
    assert result.flat.shape == bcr.mu.shape  # flattened mu, one col per channel
    assert result.chi.shape[1] == bcr.n_channels  # one chi col per channel
    assert result.chi.shape[0] == result.k.size  # chi rows align with shared k
    assert result.k.ndim == 1  # a single shared k-grid


def test_edge_step_positive(bcr):
    result = process_batch(bcr, Params())
    assert np.all(result.edge_step > 0)


def test_e0_defaults_to_header(bcr):
    result = process_batch(bcr, Params())  # header E0_tab default
    assert result.meta["e0_source"] == "header_e0_tab"
    assert result.e0 == pytest.approx(7709.0)


def test_auto_e0_near_header(bcr):
    result = process_batch(bcr, Params(auto_e0=True))
    assert result.meta["e0_source"] == "find_e0"
    # find_e0 returns the derivative-max, which for Co sits a few eV ABOVE the
    # tabulated edge onset (7709) -- confirmed ~7714 on the full file. So require
    # it in a sane window near, and at/above, the tabulated edge rather than on it.
    assert 7709.0 <= result.e0 <= 7729.0


def test_explicit_e0_overrides(bcr):
    result = process_batch(bcr, Params(e0=7710.5))
    assert result.meta["e0_source"] == "explicit"
    assert result.e0 == pytest.approx(7710.5)


def test_shared_k_grid_is_identical(bcr):
    """All channels share one k-grid — the whole point of stacking into a matrix."""
    result = process_batch(bcr, Params())
    # chi is finite and the grid is monotonic increasing
    assert np.all(np.isfinite(result.chi))
    assert np.all(np.diff(result.k) > 0)


def test_forward_ft_optional(bcr):
    result = process_batch(bcr, Params(ft=True))
    assert result.r is not None
    assert result.chir_mag is not None
    assert result.chir_mag.shape[0] == result.r.size
    assert result.chir_mag.shape[1] == bcr.n_channels
