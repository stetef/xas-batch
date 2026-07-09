"""``xas-batch-plot`` — render the per-scan processing pipeline for one file.

Loads a combined-BCR file, processes its scans, and saves four PNGs:
raw scans, per-scan normalization fits, flattened overlay, and EXAFS splines +
kⁿ·χ(k). Use ``--show`` to open them interactively instead of saving.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xasbatch.model import Params


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xas-batch-plot",
        description="Visualize per-scan normalization + EXAFS extraction for one file.",
    )
    p.add_argument("input", type=Path, help="a .bcr.combined file")
    p.add_argument("-o", "--outdir", type=Path, default=Path("plots"), help="output dir (default: ./plots)")
    p.add_argument("--chi-kweight", type=int, default=3, help="k-weight for the χ(k) display (default: 3)")
    p.add_argument("--dpi", type=int, default=120, help="PNG resolution (default: 120)")
    p.add_argument("--show", action="store_true", help="show interactively instead of saving PNGs")
    # processing knobs (edge/normalization/spline)
    p.add_argument("--e0", type=float, default=None, help="force edge energy (eV)")
    p.add_argument("--auto-e0", action="store_true", help="detect e0 via find_e0 (default: header E0_tab)")
    p.add_argument("--rbkg", type=float, default=Params().rbkg, help="AUTOBK rbkg")
    p.add_argument("--kmin", type=float, default=Params().kmin, help="χ(k) kmin")
    p.add_argument("--kmax", type=float, default=Params().kmax, help="χ(k) kmax")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")  # headless save
    import matplotlib.pyplot as plt

    from xasbatch.io import load_combined_bcr
    from xasbatch.plotting import figure_report

    params = Params(
        mode="scan", e0=args.e0, auto_e0=args.auto_e0,
        rbkg=args.rbkg, kmin=args.kmin, kmax=args.kmax,
    )
    bcr = load_combined_bcr(args.input)
    figs = figure_report(bcr, params, kweight=args.chi_kweight)

    if args.show:
        plt.show()
        return 0

    sample = bcr.meta.get("sample") or args.input.stem
    args.outdir.mkdir(parents=True, exist_ok=True)
    for label, fig in figs:
        out = args.outdir / f"{sample}_{label}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
