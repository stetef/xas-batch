"""Minimal end-to-end example: load one combined-BCR file, extract χ(k), save.

Run with:  uv run python examples/run_one_file.py PATH/TO/sample.bcr.combined
"""

from __future__ import annotations

import sys
from pathlib import Path

from xasbatch.io import load_combined_bcr, save_result
from xasbatch.model import Params
from xasbatch.process import process_batch

DEFAULT = Path(__file__).parent.parent / "tests" / "data" / "sample_small.bcr.combined"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    bcr = load_combined_bcr(path)
    print(f"loaded {bcr.n_channels} channels x {bcr.n_energy} energies from {path.name}")
    print(f"  element={bcr.meta['element']} edge={bcr.meta['edge']} E0_tab={bcr.meta['e0_tab']}")

    result = process_batch(bcr, Params(mode="both"))  # header E0_tab default; both blocks
    print(f"processed: e0={result.e0:.3f} eV")
    if result.scan is not None:
        s = result.scan
        print(f"  scan block:    {s.n} scans,    chi {s.chi.shape}, names={s.names}")
    if result.channel is not None:
        c = result.channel
        print(f"  channel block: {c.n} channels, chi {c.chi.shape}")

    out = save_result(result, Path("out"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
