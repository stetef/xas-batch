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

## Mass processing a whole tree (`xas-batch-tree`)

For processing many files across a directory tree, copy `.env.example` to `.env` and
set the input root, then:

```bash
uv run xas-batch-tree              # parallel over all *.bcr.combined under XAS_INPUT_ROOT
uv run xas-batch-tree --jobs 8     # control worker count (default: cpu_count-1; 1 = serial)
uv run xas-batch-tree --limit 8    # process only the first N (handy for a trial run)
uv run xas-batch-tree --force      # reprocess even files the catalog marks done
```

- Recursively finds every `*.bcr.combined` under `XAS_INPUT_ROOT`.
- Output: mirrored into `XAS_OUTPUT_DIR` preserving the tree, or — if that is unset —
  written as a sister `.npz` next to each source file.
- **Resumable**: a SQLite catalog (`xas_catalog.sqlite`) records one row per file, so a
  re-run skips files already processed (unless the source changed, or `--force`).
- **Provenance / query index**: the catalog stores `status, e0, e0_source, n_channels,
  element, edge, params, error` per file — query it directly for downstream work, e.g.
  ```sql
  SELECT source_path, output_path FROM files WHERE element='Co' AND status='ok';
  ```
- Processing knobs (`--auto-e0`, `--kweight`, `--ft`, …) match `xas-batch`.

`.env` keys: `XAS_INPUT_ROOT` (required), `XAS_OUTPUT_DIR` (optional), `XAS_DB_PATH`
(optional; defaults to `<output-or-input root>/xas_catalog.sqlite`).

## Processing modes (`--mode`)

Each combined file bundles several *original scans* (one per BCR member file), and
each scan contributed several detector channels — the header's `# Members (kept):`
block records how many, in column order.

- `--mode scan` **(default)** — sum each original file's channels into one total-
  fluorescence μ(E) per scan (`nansum`, so a missing element doesn't poison the sum),
  then process those. ~15 spectra for the Co3NK_s example.
- `--mode channel` — process every μ column individually (~448 for Co3NK_s).
- `--mode both` — compute and store *both* blocks in the one `.npz`.

e0 is resolved once per file and shared across every column of every block, so the
scan and channel blocks land on the **same k-grid**.

## Key options

- `--e0 FLOAT` — force the edge energy for all channels.
- `--auto-e0` — detect e0 once per file via Larch `find_e0` instead of trusting the
  header `E0_tab` (the default).
- `--kweight`, `--kmin`, `--kmax`, `--rbkg`, `--kstep` — AUTOBK / χ(k) knobs.
- `--ft` — also compute the forward FT (χ(R)).

## Output

One `<sample>.npz` per input file. Shared arrays: `energy`, `e0`, JSON-encoded `meta`.
Then a `scan_*` and/or `channel_*` block (whichever were computed):

    scan_names, scan_flat (nE×nScans), scan_k, scan_chi (nk×nScans), scan_edge_step
    channel_names, channel_flat (nE×nFF), channel_k, channel_chi (nk×nFF), channel_edge_step
    (+ scan_r / scan_chir_mag, channel_r / channel_chir_mag when --ft is given)

`meta` records `mode`, `modes_present`, `n_channels_raw`, `e0_used`/`e0_source`, and
`scan_members` (the scan→channel-count mapping).

## Layout

- `io.py` / `model.py` — pure numpy (no Larch); the custom parser and data model.
- `process.py` — the Larch layer (`pre_edge`, `autobk`, `xftf`).
- `cli.py` — the `xas-batch` entry point.
