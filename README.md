# xas-batch

Batch normalization + EXAFS extraction for **combined-BCR** fluorescence XAS files.

Each input is one energy-calibrated, I0-divided file with many μ(E) columns
(`FF*/I0`) on a shared energy grid, plus trailing `RTC_*` columns that are ignored
for processing. For every channel this normalizes μ(E) (pre/post-edge → flattened),
runs an AUTOBK background spline, and extracts χ(k) on a shared k-grid — then stacks
the results and writes one `.npz` per file. Wraps [Larch](https://xraypy.github.io/xraylarch/).

Upstream steps (energy calibration, I0 division) are assumed already done.

## Install / run

```bash
uv sync
uv run xas-batch INPUT [-o OUTDIR]        # INPUT = a .bcr.combined file or a directory
uv run pytest
```

## Key options

- `--e0 FLOAT` — force the edge energy for all channels.
- `--auto-e0` — detect e0 once per file via Larch `find_e0` instead of trusting the
  header `E0_tab` (the default).
- `--kweight`, `--kmin`, `--kmax`, `--rbkg`, `--kstep` — AUTOBK / χ(k) knobs.
- `--ft` — also compute the forward FT (χ(R)).

## Output

One `<sample>.npz` per input file containing `energy`, `flat` (nE×nFF), `k`,
`chi` (nk×nFF), `e0`, `edge_step`, `channel_names`, and JSON-encoded `meta`
(optionally `r`, `chir_mag` when `--ft` is given).

## Layout

- `io.py` / `model.py` — pure numpy (no Larch); the custom parser and data model.
- `process.py` — the Larch layer (`pre_edge`, `autobk`, `xftf`).
- `cli.py` — the `xas-batch` entry point.
