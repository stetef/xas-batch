# Standalone XAS batch-processing repo — plan

A small, self-contained repo for the specific workflow: take an energy-calibrated,
I0-divided fluorescence file with **many μ(E) columns on one shared energy grid**,
normalize each column, spline it, and extract χ(k). Depends only on **Larch** (plus
numpy/scipy, which Larch pulls in), managed with **uv**.

This replaces the need to fork/modify `catXAS`. The catXAS `xas.py` functions we were
going to reuse are thin wrappers around `larch.xafs.*`; we call Larch directly and keep
only a custom I/O layer, which catXAS doesn't provide for this file format.

## Goals & scope

- **In scope:** read the combined BCR file format, per-column pre/post-edge
  normalization (→ flattened μ(E)), AUTOBK background spline + E→k + spline
  subtraction (→ χ(k)), optional forward FT (→ χ(R)), batch over all columns and all
  files, write tidy outputs.
- **Out of scope:** energy calibration and I0 division (already done upstream),
  the catXAS time-series `Experiment` container, MCR/PCA/LCF, GUI.

## Non-negotiable facts about the input (drive the design)

- One file = one shared energy grid + N pre-computed μ columns (`FFn/I0`) + M
  trailing `RTC_*` columns to ignore. Counts vary per file — **parse the header,
  don't hard-code 448/15.**
- Data is already energy-calibrated and I0-divided → μ = column verbatim. **No `delE`
  shift, no `calc_mu`.** (This is why most catXAS wrappers add nothing here.)
- All columns share energy, e0, and kstep → AUTOBK yields an **identical, aligned
  k-grid** for every column. Compute e0 once; stack χ(k) directly into a matrix.
- Energy may be stored descending → flip to ascending on load (one-liner).

## Dependencies (uv)

- `xraylarch` — provides the `larch` module (`larch.Group`, `larch.xafs.{find_e0,
  pre_edge, autobk, xftf}`). Pulls numpy/scipy transitively.
- `matplotlib` — optional plotting helpers.
- Dev: `pytest`, `ruff`.

> Note: the PyPI package is **`xraylarch`**, imported as `larch`. Larch is a large
> dependency; if install weight matters we can revisit, but there is no lighter way to
> get AUTOBK/pre_edge with the same numerics.

`pyproject.toml` (sketch):

```toml
[project]
name = "xas-batch"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["xraylarch", "matplotlib"]

[project.scripts]
xas-batch = "xasbatch.cli:main"

[dependency-groups]
dev = ["pytest", "ruff"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Workflow: `uv sync`, `uv run xas-batch <input> -o <outdir>`, `uv run pytest`.

## Repo layout (src layout)

```
xas-batch/
├── pyproject.toml
├── uv.lock
├── README.md
├── STANDALONE_REPO_PLAN.md        # this file
├── src/
│   └── xasbatch/
│       ├── __init__.py
│       ├── io.py                  # file parsing + writers  (NO larch import)
│       ├── model.py               # dataclasses: BcrData, Params, BatchResult
│       ├── process.py             # the ~5 larch.xafs calls, per-column + batch
│       ├── plotting.py            # optional matplotlib helpers
│       └── cli.py                 # argparse entry point
├── tests/
│   ├── data/
│   │   └── sample_small.combined  # trimmed fixture (few columns, few rows)
│   ├── test_io.py                 # parser: header, FF/RTC split, flip — no larch
│   └── test_process.py           # numerics on the fixture
└── examples/
    └── run_one_file.py            # minimal end-to-end script
```

**Design principle: I/O has no Larch dependency.** `io.py` + `model.py` are pure
numpy, so the parser (the only genuinely custom, bug-prone code) is unit-testable
without Larch. Larch lives only in `process.py`.

## Data model (`model.py`)

```python
@dataclass
class BcrData:
    energy: np.ndarray            # (nE,) ascending
    mu: np.ndarray                # (nE, nFF)  the FF*/I0 channels
    channel_names: list[str]      # len nFF, e.g. ["FF1/I0", ...]
    rtc: np.ndarray | None        # (nE, nRTC) ignored for processing, kept for provenance
    meta: dict                    # element, edge, e0_tab, k_max, session, sample, source_path, ...

@dataclass
class Params:                     # processing knobs, sensible Co-K defaults
    e0: float | None = None       # None -> auto-detect once
    pre1: float = -100; pre2: float = -50
    norm1: float = 75; norm2: float = 300; nnorm: int = 2
    rbkg: float = 1.0; kmin: float = 0.0; kmax: float | None = None
    kweight: int = 1; kstep: float = 0.05
    # optional FT
    ft_kmin: float = 3; ft_kmax: float = 12; ft_kweight: int = 2; ft_dk: float = 5

@dataclass
class BatchResult:
    energy: np.ndarray            # (nE,)
    flat: np.ndarray              # (nE, nFF)  flattened mu(E) per channel
    k: np.ndarray                 # (nk,)  shared across channels
    chi: np.ndarray               # (nk, nFF)
    e0: float
    channel_names: list[str]
    meta: dict
```

## Module responsibilities & signatures

### `io.py` (pure, no larch)
- `parse_header(path) -> dict` — read `#`-comment block: `# Columns:` line (split
  `Energy` / `FF*` / `RTC_*`), plus element, reference edge, `E0_tab`, `k_max`,
  session, sample. Returns `meta` + column index ranges.
- `load_combined_bcr(path) -> BcrData` — `np.loadtxt(path, comments="#")`, slice
  columns using the parsed header, flip to ascending energy, drop/stash RTC.
- `save_flat(result, outdir)` / `save_chi(result, outdir)` — tidy CSV: energy/k as
  first column, one column per channel; filenames derived from `meta["sample"]`.
- `save_result(result, outdir)` — convenience calling both + a small JSON of `meta`.

### `process.py` (the Larch layer)
- `build_group(energy, mu) -> larch.Group`
- `find_edge(energy, mu, e0_guess=None) -> float` — wraps `larch.xafs.find_e0`.
- `normalize(group, params) -> group` — `larch.xafs.pre_edge(...)`; sets `.flat`,
  `.norm`, `.edge_step`.
- `extract_exafs(group, params) -> group` — `larch.xafs.autobk(...)`; sets `.k`,
  `.chi`, `.bkg`.
- `forward_ft(group, params) -> group` — optional `larch.xafs.xftf(...)`.
- `process_channel(energy, mu, params) -> group` — full single-column pipeline.
- `process_batch(bcr, params) -> BatchResult` — detect e0 once (if `params.e0 is
  None`) from a representative/mean column, reuse across columns, loop, stack.

The Larch mapping (what each step really calls):

| Step | Larch call | Output on group |
|---|---|---|
| build group | `larch.Group(energy=, mu=)` | `.energy`, `.mu` |
| edge (once) | `larch.xafs.find_e0` | `e0` (float) |
| normalize | `larch.xafs.pre_edge` | `.flat`, `.norm`, `.edge_step` |
| spline→χ(k) | `larch.xafs.autobk` | `.k`, `.chi`, `.bkg` |
| FT (optional) | `larch.xafs.xftf` | `.r`, `.chir_mag` |

### `cli.py`
`xas-batch INPUT [-o OUTDIR] [--e0 FLOAT] [--kweight N] [--kmin ...] [--ft]`
- `INPUT` = a file or a directory (glob `*.combined`); process each, write outputs.
- Param flags override `Params` defaults; unset → defaults.
- Prints a one-line summary per file (channels processed, e0, nk).

### `plotting.py` (optional)
- `plot_flat(result, ax=None)` — overlay flattened μ(E).
- `plot_chi(result, kweight=2, ax=None)` — overlay k^w·χ(k).
- Kept separate so core processing never imports matplotlib.

## Key design decisions

1. **Call Larch directly; keep zero catXAS code.** The wrappers' value (`delE`
   bookkeeping, param bundling for the `Experiment` container) doesn't apply here.
2. **e0 once per file, not per column** — same sample across channels. Override via
   `--e0` when needed. (Structurally allows per-column later, but default is shared.)
3. **Shared k-grid is guaranteed, not assumed** — identical energy+e0+kstep → assert
   all channels return the same `k` length, then store one `k` + a `chi` matrix.
4. **Header-driven column split** — the FF/RTC counts are read from `# Columns:`, so
   files with different channel counts just work. If the header is missing/renamed,
   fail loudly rather than guess.
5. **Purity boundary** — Larch only in `process.py`/`plotting.py`; `io.py` testable
   standalone. This isolates the custom parser (highest bug risk) for fast tests.

## Testing plan

- **`test_io.py` (no larch):** commit a trimmed `sample_small.combined` (say 4 FF + 2
  RTC, ~10 rows). Assert channel count, names, energy ascending, RTC excluded from
  `mu`, meta fields parsed. Add a descending-energy fixture to test the flip.
- **`test_process.py`:** run `process_batch` on the fixture; assert shapes
  (`flat` == `mu` shape, `chi` cols == nFF, single shared `k`), `edge_step > 0`,
  e0 near the header `E0_tab`. Numerics need only be sane, not golden.

## Build order (milestones)

1. `uv init` + `pyproject.toml` + `uv sync` (get `xraylarch` importing).
2. `model.py` dataclasses.
3. `io.py` + `test_io.py` on the trimmed fixture — nail the parser first.
4. `process.py` + `test_process.py` — the ~5 Larch calls, per-channel then batch.
5. `cli.py` + `examples/run_one_file.py` — end-to-end on the real Co3NK_s file.
6. `plotting.py` + README.
7. (later) optional FT outputs, multi-file directory runs, parallelism if needed.

## Open questions for you

- **Repo/package name** — placeholder is `xas-batch` / `xasbatch`. Prefer something
  else (e.g. `bcr-xas`, `fluoro-exafs`)?
- **Output format** — one wide CSV per file (energy/k + one col per channel), or a
  single stacked/long tidy file, or `.npz`? Wide CSV is the current assumption.
- **e0 source** — auto-detect via `find_e0`, or trust the header `E0_tab`
  (7709.0 eV for Co K) as the default? (Auto-detect assumed, header as fallback.)
- **Where should this repo live** — sibling of `catXAS` under `denoising-spectra/`?
```
