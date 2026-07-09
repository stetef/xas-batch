"""Command-line entry point: ``xas-batch``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xasbatch.model import Params

GLOB = "*.bcr.combined"


def add_param_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the processing-knob flags shared by ``xas-batch`` and ``xas-batch-tree``."""
    d = Params()
    p.add_argument(
        "--mode",
        choices=("scan", "channel", "both"),
        default=d.mode,
        help=f"scan=sum each file's channels; channel=per column; both (default: {d.mode})",
    )
    e = p.add_mutually_exclusive_group()
    e.add_argument("--e0", type=float, default=None, help="force edge energy (eV) for all channels")
    e.add_argument(
        "--auto-e0",
        action="store_true",
        help="detect e0 per file via find_e0 (default: trust header E0_tab)",
    )
    p.add_argument("--rbkg", type=float, default=d.rbkg, help=f"AUTOBK rbkg (default: {d.rbkg})")
    p.add_argument("--kmin", type=float, default=d.kmin, help=f"χ(k) kmin (default: {d.kmin})")
    p.add_argument("--kmax", type=float, default=d.kmax, help="χ(k) kmax (default: full range)")
    p.add_argument("--kweight", type=int, default=d.kweight, help=f"k-weight (default: {d.kweight})")
    p.add_argument("--kstep", type=float, default=d.kstep, help=f"k step (default: {d.kstep})")
    p.add_argument("--ft", action="store_true", help="also compute the forward FT χ(R)")
    return p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xas-batch",
        description="Normalize + extract EXAFS χ(k) from combined-BCR fluorescence files.",
    )
    p.add_argument("input", type=Path, help=f"a {GLOB} file, or a directory to scan for them")
    p.add_argument("-o", "--outdir", type=Path, default=Path("out"), help="output dir (default: ./out)")
    add_param_args(p)
    return p


def block_summary(result) -> str:
    """One-line description of which blocks a result carries, e.g. 'scan=15, channel=448 (nk=301)'."""
    parts = []
    for label, blk in (("scan", result.scan), ("channel", result.channel)):
        if blk is not None:
            parts.append(f"{label}={blk.n} (nk={blk.k.size})")
    return ", ".join(parts) if parts else "no blocks"


def gather_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob(GLOB))
    return [path]


def params_from_args(args: argparse.Namespace) -> Params:
    return Params(
        mode=args.mode,
        e0=args.e0,
        auto_e0=args.auto_e0,
        rbkg=args.rbkg,
        kmin=args.kmin,
        kmax=args.kmax,
        kweight=args.kweight,
        kstep=args.kstep,
        ft=args.ft,
    )


def main(argv: list[str] | None = None) -> int:
    # Larch is imported here (not at module top) so the shared arg helpers can be
    # reused by xas-batch-tree without pulling Larch into the orchestrator process.
    from xasbatch.io import load_combined_bcr, save_result
    from xasbatch.process import process_batch

    args = build_parser().parse_args(argv)
    params = params_from_args(args)

    inputs = gather_inputs(args.input)
    if not inputs:
        print(f"no {GLOB} files found under {args.input}", file=sys.stderr)
        return 1

    failures = 0
    for path in inputs:
        try:
            bcr = load_combined_bcr(path)
            result = process_batch(bcr, params)
            out_path = save_result(result, args.outdir)
        except Exception as exc:  # keep going across a batch; report at the end
            print(f"FAIL  {path.name}: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(
            f"OK    {path.name}: {block_summary(result)}, "
            f"e0={result.e0:.2f} eV ({result.meta['e0_source']}) -> {out_path}"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
