# xas-batch

[![CI](https://github.com/stetef/xas-batch/actions/workflows/ci.yml/badge.svg)](https://github.com/stetef/xas-batch/actions/workflows/ci.yml)

Batch normalization + EXAFS extraction for **combined-BCR** fluorescence XAS files,
via [Larch](https://xraypy.github.io/xraylarch/).

Each input is one energy-calibrated, I0-divided file with many μ(E) columns (`FF*/I0`)
on a shared energy grid. For each spectrum it normalizes μ(E) (pre/post-edge →
flattened), runs an AUTOBK background spline, and extracts χ(k) on a shared k-grid —
then stacks the results and writes one `.npz` per file. Energy calibration and I0
division are assumed already done upstream.

See [PROCESSING.md](PROCESSING.md) for exactly what the pipeline does to the data (what
gets splined, how χ(k) is defined, what each parameter changes), and [DESIGN.md](DESIGN.md)
for the file format, architecture, and rationale.

## Setup

```bash
uv sync
uv run pytest        # 25 tests, ~3s
```

## Process one file (or a flat directory)

```bash
uv run xas-batch INPUT [-o OUTDIR]     # INPUT = a .bcr.combined file or a directory
uv run xas-batch INPUT --mode both     # store per-scan AND per-channel blocks
uv run xas-batch INPUT --ft --kweight 2
```

## Process a whole tree

Copy `.env.example` to `.env`, set `XAS_INPUT_ROOT` (and optionally `XAS_OUTPUT_DIR`),
then:

```bash
uv run xas-batch-tree              # parallel over all *.bcr.combined under XAS_INPUT_ROOT
uv run xas-batch-tree --jobs 8     # worker count (default: cpu_count-1; 1 = serial)
uv run xas-batch-tree --limit 8    # first N files only (trial run)
uv run xas-batch-tree --force      # reprocess even files the catalog marks done
```

- **Output:** mirrored under `XAS_OUTPUT_DIR` preserving the tree, or (if unset) a
  sister `.npz` next to each source file.
- **Resumable:** a SQLite catalog (`xas_catalog.sqlite`) records one row per file; a
  re-run skips files already done (unless the source changed, or `--force`).
- **Queryable:** the catalog is a provenance index for downstream work, e.g.
  ```sql
  SELECT source_path, output_path FROM files WHERE element='Co' AND status='ok';
  ```

`.env` keys: `XAS_INPUT_ROOT` (required), `XAS_OUTPUT_DIR` (optional),
`XAS_DB_PATH` (optional; defaults to `<output-or-input root>/xas_catalog.sqlite`).

## Modes (`--mode`, default `scan`)

A combined file bundles several original scans, each contributing several detector
channels (the header's `# Members (kept):` block says how many).

| mode | what it processes | Co3NK_s | file size |
|---|---|---|---|
| `scan` *(default)* | sum each scan's channels → one μ(E)/scan | 15 spectra | ~0.2 MB |
| `channel` | every μ column individually | 448 spectra | ~3.9 MB |
| `both` | both blocks in one `.npz` | 15 + 448 | ~4.0 MB |

`scan` and `channel` share one per-file e0, so their k-grids align. `both` ≈ `channel`
in size (the scan block rides along nearly free); `scan`-only is ~20× smaller.

## Visualize the processing (`xas-batch-plot`)

Render the per-scan pipeline for one file to PNGs (or `--show` to view interactively):

```bash
uv run xas-batch-plot INPUT.bcr.combined [-o plots/] [--chi-kweight 3]
```

Writes four figures into a per-sample subdir, `<outdir>/<sample>/`:

1. `1_raw.png` — summed μ(E) per scan (= Σ FF/I0) + the merged spectrum
2. `2_norm_fits.png` — per-scan grid: μ(E) with pre/post-edge fits (left), flattened μ (right)
3. `3_flat.png` — all flattened scans overlaid + the merged spectrum
4. `4_exafs.png` — normalized μ + AUTOBK splines (left), kⁿ·χ(k) per scan + merged (right)

The bold **merged** trace is an **E-space merge carried through**: the per-scan μ(E) are
averaged, then pre_edge + AUTOBK run *once* on that mean μ — so the merged χ(k) is the
spline of the averaged spectrum, not the mean of the per-scan χ(k).

The plotting functions in `plotting.py` are reusable, so a notebook/Streamlit front-end
can call them directly. (matplotlib comes in transitively via Larch; the `plot` extra
declares it explicitly: `uv sync --extra plot`.)

## Options

- `--mode {scan,channel,both}` — see above.
- `--e0 FLOAT` — force the edge energy. By default e0 is detected once per file via
  `find_e0` on the merged spectrum; `--header-e0` uses the tabulated header `E0_tab`
  instead.
- `--kweight`, `--kmin`, `--kmax`, `--rbkg`, `--kstep` — AUTOBK / χ(k) knobs.
- `--ft` — also compute the forward FT (χ(R)).

## Output layout

One `<sample>.npz` per file: shared `energy`, `e0`, JSON `meta_json`; plus a `scan_*`
and/or `channel_*` block:

    <prefix>_names, <prefix>_flat (nE×n), <prefix>_k, <prefix>_chi (nk×n), <prefix>_edge_step
    (+ <prefix>_r / <prefix>_chir_mag when --ft is given)

`meta` records `mode`, `modes_present`, `n_channels_raw`, `e0_used`/`e0_source`, and
`scan_members`.

## Package layout

| module | role |
|---|---|
| `model.py`, `io.py` | pure numpy — data model, header parser, npz I/O, scan grouping |
| `process.py` | the Larch layer (`pre_edge`, `autobk`, `xftf`) |
| `catalog.py` | SQLite catalog for tree runs |
| `plotting.py` | matplotlib panel builders (optional; separate from core) |
| `cli.py`, `tree.py`, `plotcli.py` | the `xas-batch`, `xas-batch-tree`, `xas-batch-plot` entry points |
