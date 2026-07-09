# Design & architecture

As-built design notes for `xas-batch` — the *why* behind the code. For usage see the
[README](README.md). (This supersedes the original build plan.)

## Scope

**In scope:** read the combined-BCR file format; per-spectrum pre/post-edge
normalization (→ flattened μ(E)); AUTOBK background spline + E→k + spline subtraction
(→ χ(k)); optional forward FT (→ χ(R)); batch over all spectra in a file, and over a
whole directory tree; tidy `.npz` output + a SQLite catalog.

**Out of scope:** energy calibration and I0 division (done upstream); path/FEFF fitting
(where ΔE0 floats — see below); MCR/PCA/LCF; any GUI. This repo stops at χ(k)/χ(R).

We call [Larch](https://xraypy.github.io/xraylarch/) directly and keep **zero catXAS
code** — the catXAS wrappers around `larch.xafs.*` added value (delE bookkeeping,
`Experiment`-container param bundling) that doesn't apply to these already-calibrated,
I0-divided files.

## Input format: combined-BCR

One file = one sample. Header is a `#`-comment block; data is whitespace-columns.

```
# Combined per-sample XAS spectrum
# Session: 2017_7-3_Apr
# Sample: Co3NK_s
# Element: Co  (reference edge: K @ 7709.0000 eV, from xraydb)
# k_max calculated (1/Å): 15.0200  (... E0_tab=7709.0000 eV ...)
# Members (kept):
#   BCR_Co3NK_s_043_A.001  → 30 channel(s), shift=+0.6480 eV, E_exp=7709.6480 eV
#   ... (one line per original file/scan) ...
# Columns: Energy FF1/I0 FF2/I0 ... FF448/I0 RTC_1 ... RTC_15
7389.354000 0.001137 0.001471 ...
```

Facts that drive the design:

- **Column counts vary per file** — parse the `# Columns:` line; never hard-code
  448/15. Column 0 is energy; `RTC_*` columns are provenance (ignored for processing);
  everything else is a μ channel (`FF*/I0`).
- **Data is already μ = FF/I0.** No delE shift, no `calc_mu`. μ is the column verbatim.
- **All columns share one energy grid, e0, and kstep** → AUTOBK yields an identical,
  aligned k-grid for every spectrum. We compute e0 once and stack χ(k) into a matrix.
- **Energy may be stored descending** → flipped to ascending on load.
- **The `# Members (kept):` block maps channels → original scans.** Each member (one
  original BCR file = one scan) lists how many channels it contributed, and channels sit
  in that order — so cumulative counts slice the μ matrix into scans (see modes below).

## Data model (`model.py`, pure numpy)

- `BcrData` — one loaded file: `energy` (nE), `mu` (nE×nFF), `channel_names`, stashed
  `rtc`/`rtc_names`, and `meta` (parsed header incl. `members`).
- `Params` — processing knobs with Co-K defaults: `mode`, `e0`/`auto_e0`, pre-edge
  (`pre1/pre2/norm1/norm2/nnorm`), AUTOBK (`rbkg/kmin/kmax/kweight/kstep`), FT (`ft` +
  `ft_*`).
- `ProcessBlock` — one stacked set of results: `names`, `flat` (nE×n), `k` (nk),
  `chi` (nk×n), `edge_step` (n), optional `r`/`chir_mag`.
- `BatchResult` — one file's output: shared `energy` + `e0` + `meta`, and an optional
  `scan` block and/or `channel` block.

## Modules & the purity boundary

```
io.py / model.py   pure numpy, NO Larch  — parser + data model + npz/scan-grouping
process.py         the Larch layer       — pre_edge / autobk / xftf
catalog.py         stdlib sqlite3        — the tree-run catalog
plotting.py        matplotlib (optional) — per-scan pipeline figures
cli.py             xas-batch entry point
tree.py            xas-batch-tree entry point (walk + pool + catalog)
plotcli.py         xas-batch-plot entry point (per-scan figures for one file)
```

**Plotting re-runs from source, and re-uses the Larch groups.** The fit curves a
reviewer wants to see — the pre-edge line, post-edge polynomial, and AUTOBK background
— live on the Larch group but are *not* stored in the `.npz` (which keeps only stacked
arrays). So `process.process_scans()` re-runs the per-scan pipeline and hands the full
groups to `plotting.py`. This keeps the `.npz` lean and the plots always faithful to a
given `Params`, at the cost of a cheap recompute (~15 scans). matplotlib stays out of
the core (`io`/`process`); only `plotting.py`/`plotcli.py` import it.

**The parser is the only genuinely custom, bug-prone code, so it carries no Larch
dependency** — `io.py`/`model.py`/`catalog.py` are unit-testable in ~3 s without
importing Larch. Larch lives only in `process.py` (imported lazily, so `--help`,
resume-skip, and the tree orchestrator stay light).

## Processing pipeline

Per spectrum (`process.py`): `larch.Group(energy, mu)` → `pre_edge` (sets
`.flat/.norm/.edge_step`) → `autobk` (sets `.k/.chi/.bkg`) → optional `xftf` (sets
`.r/.chir_mag`). `_process_matrix` runs this over a matrix of columns and stacks the
results, asserting every column returns the same-length `k`.

> For the numerical detail — that AUTOBK splines the **raw** μ(E) (not `flat`/`norm`),
> how χ(k) is defined, and what each parameter changes — see **[PROCESSING.md](PROCESSING.md)**.

### e0 is resolved once per file

Resolution order (`resolve_e0`): explicit `Params.e0` > header `E0_tab` > `find_e0`.
`find_e0` runs only with `--auto-e0`, or as a last resort when the header has no
tabulated edge; it detects from the **mean** column so one noisy channel can't skew it.

The shared e0 is what makes the aligned k-grid a *guarantee*, not an assumption: same
energy + e0 + kstep → identical `k` for every spectrum, and even the `scan` and
`channel` blocks share it.

> **Why header `E0_tab` by default, not `find_e0`?** The files are already calibrated,
> so the tabulated edge is meaningful and deterministic. `find_e0` returns the
> derivative-max, which for Co sits ~5 eV *above* the tabulated onset (confirmed 7714.4
> vs 7709.0 on the full Co3NK_s file). That offset is not an error — it's the difference
> between "steepest point" and "tabulated edge." E0 does **not** float during splining;
> the floating ΔE0 you may be thinking of belongs to downstream FEFF/path fitting, which
> is out of scope here. If you later fit paths, the fitted ΔE0 harmlessly absorbs this
> ~5 eV extraction-E0 choice.

### Modes (`--mode`, default `scan`)

- `scan` — sum each original file's channels (`nansum`, so a missing detector element
  doesn't poison the sum) into one total-fluorescence μ(E) per scan, then process. For
  Co3NK_s: 15 scans from 448 channels.
- `channel` — process every μ column individually (448 for Co3NK_s).
- `both` — compute and store both blocks in the one `.npz`.

`scan` recomputes from the summed μ (AUTOBK on the sum), *not* by averaging per-channel
χ — that's the physically correct order. Consequently `both` costs `scan + channel`
AUTOBK calls, but the scan block is tiny (~15 cols) so the marginal cost over `channel`
alone is small.

## Output (`.npz` per file)

One `<sample>.npz` (or, in a tree run, the source basename with `.bcr.combined` →
`.npz`). Shared `energy`, `e0`, `meta_json`; then namespaced `scan_*` and/or
`channel_*` arrays (`_names/_flat/_k/_chi/_edge_step`, plus `_r/_chir_mag` with `--ft`).

**One file, namespaced blocks — not `.scan.npz`/`.channel.npz`.** Keeps one artifact
and one catalog row per source, and lets downstream code load one file and pick a block.

`meta` records `mode`, `modes_present`, `n_channels_raw`, `e0_used`/`e0_source`, and
`scan_members` (scan → channel-count mapping).

## Mass-processing architecture (`tree.py` + `catalog.py`)

`xas-batch-tree` recursively finds every `*.bcr.combined` under `XAS_INPUT_ROOT` (from
`.env`), processes each, and writes `.npz` either mirrored under `XAS_OUTPUT_DIR` or as
a sister file next to the source.

- **Parallel, but the main process is the only DB writer.** Workers do all the heavy,
  independent work (load → process → save npz) and return a small record; the main loop
  writes each record to SQLite as it arrives via `imap_unordered`. So there is **zero DB
  write-contention** and no locking dance — parallelism stays simple and correct.
- **SQLite as a *catalog*, not a data store.** The spectra stay in `.npz`; the DB holds
  one row per file: `status, mode, e0, e0_source, n_scans, n_channels, element, edge,
  params_json, error, source_mtime, output_path`. It is three things at once:
  - a **resume ledger** — a re-run skips files whose source `mtime` matches an `ok` row
    (errors are always retried; `--force` overrides);
  - a **parallel-safe status tracker** (WAL mode, single writer);
  - a **queryable provenance index** for downstream ML work — e.g. build a training set
    from `SELECT output_path FROM files WHERE element='Co' AND status='ok'` without
    cracking open hundreds of npz headers.

## Testing

- **`test_io.py` / `test_tree.py` (no Larch):** header parse (columns, members, meta),
  `scan_groups` slicing + validation, descending-energy flip, loud failure on a missing
  `# Columns:` line, npz round-trip, tree output-path mirroring/sister logic, catalog
  upsert/resume/error.
- **`test_process.py`:** shapes per mode, scan = nansum of its channels, shared k-grid
  across blocks, `edge_step > 0`, e0 source selection, invalid-mode error, optional FT.

Fixtures are trimmed from the real Co3NK_s file (2 scans × 3 channels, every-3rd row —
dense enough near the edge for `find_e0`). Numerics are asserted sane, not golden.
