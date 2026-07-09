# xas-batch

Batch normalization + EXAFS extraction for **combined-BCR** fluorescence XAS files,
via [Larch](https://xraypy.github.io/xraylarch/).

Each input is one energy-calibrated, I0-divided file with many μ(E) columns (`FF*/I0`)
on a shared energy grid. For each spectrum it normalizes μ(E) (pre/post-edge →
flattened), runs an AUTOBK background spline, and extracts χ(k) on a shared k-grid —
then stacks the results and writes one `.npz` per file. Energy calibration and I0
division are assumed already done upstream.

See [DESIGN.md](DESIGN.md) for the file format, architecture, and the rationale behind
the design choices.

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

## Options

- `--mode {scan,channel,both}` — see above.
- `--e0 FLOAT` — force the edge energy; `--auto-e0` — detect via `find_e0` instead of
  the header `E0_tab` (the default).
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
| `cli.py`, `tree.py` | the `xas-batch` and `xas-batch-tree` entry points |
