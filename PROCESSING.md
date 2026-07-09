# Processing: what the numbers actually do

The precise per-spectrum pipeline — which quantity gets splined, how χ(k) is defined,
and what each parameter changes. Code lives in
[`process.py`](src/xasbatch/process.py); everything is a thin call into `larch.xafs.*`.

Input assumption: each column is already **μ(E) = FF/I0**, energy-calibrated and
I0-divided upstream. So there is no `calc_mu`, no `delE` shift — μ is the column
verbatim (energy flipped to ascending on load).

## The pipeline (per spectrum)

`process_channel` runs three Larch steps in order on one μ(E) column:

```
build_group(energy, mu)          # a larch.Group with raw .energy, .mu
  → pre_edge(...,  group.mu, ...) # normalization      → .pre_edge .post_edge .edge_step .norm .flat
  → autobk(...,    group.mu, ...) # background + χ(k)   → .bkg .k .chi
  → xftf(...,      group.k,  ...) # optional forward FT → .r .chir_mag
```

**The key fact: AUTOBK is fed the _raw_ μ(E) (`group.mu`) — not `norm`, not `flat`.**
Normalization runs first only to establish the edge step and e0; it does not change
what gets splined.

## Step 1 — `pre_edge` (normalization)

Fits two curves to raw μ(E), relative to the edge energy `e0`:

- a **pre-edge line** over `[e0+pre1, e0+pre2]`, extrapolated across the whole range;
- a **post-edge polynomial** (degree `nnorm`, default 2) over `[e0+norm1, e0+norm2]`.

**The fit ranges span the data by default.** Only the offsets are pinned — `pre2 = −50`
(stay below the edge onset) and `norm1 = +75` (start above the XANES / white line) —
while `pre1` and `norm2` default to `None`, which Larch resolves to the **file's first
and last energy**. Fitting the post-edge polynomial across the *whole* range (rather
than a narrow near-edge window extrapolated outward) is what makes the flattened μ sit
flat at ≈1.0 out to high k; a narrow `norm2` produces a drooping, unusable flat μ. All
four bounds are overridable (`--pre1/--pre2/--norm1/--norm2`, and `--nnorm`).

From these it sets:

| attribute | meaning |
|---|---|
| `edge_step` (Δμ₀) | (post-edge − pre-edge) evaluated at `e0` — the edge jump |
| `norm` | `(μ − pre_edge_line) / edge_step` — pre-edge-subtracted **and** edge-step normalized |
| `flat` | `norm` with the post-edge curvature removed above `e0`, so it sits flat ≈ 1.0 |

`flat` is a **XANES / display** cosmetic (and what LCF would use). It is **not** used
anywhere in EXAFS extraction.

## Step 2 — `autobk` (background + χ(k))

AUTOBK fits a smooth spline background **μ₀(E) to the raw μ(E)**, choosing the number
and position of knots so that the resulting χ(R) has minimal signal below `rbkg`
(default 1.0 Å) — i.e. it removes everything "below the first shell" as background. Then

$$\chi(k) = \frac{\mu(E) - \mu_0(E)}{\Delta\mu_0}, \qquad k = \sqrt{\tfrac{2m_e}{\hbar^2}\,(E - e_0)}$$

- `group.bkg` = μ₀(E), the spline, **in raw-μ units**.
- `group.chi` = χ on a uniform k-grid of step `kstep` (default 0.05 Å⁻¹), from `kmin`
  to `kmax` (default: full data range).
- The edge step Δμ₀ comes from Step 1 — this is the *only* way normalization enters χ.
- `kweight` here (default 1) weights the internal FFT used to define "below rbkg"; it
  affects the spline stiffness, **not** how you later display χ. `chi` is stored
  unweighted (k⁰), so any kⁿ weighting is a downstream choice.

Consequence: the **pre/post-edge ranges (`pre1/pre2/norm1/norm2/nnorm`) do change χ's
amplitude**, because they set `edge_step`. **Flattening does not** — it never touches χ.

## e0

Resolved once per file (`resolve_e0`), order: explicit `Params.e0` > header `E0_tab` >
`find_e0`. `find_e0` runs only with `--auto-e0` (or if the header lacks a tabulated
edge), detecting from the mean column so one noisy channel can't skew it.

- **e0 is fixed during splining — it does not float.** It sets the k-axis origin and the
  edge-step reference, and is held at the chosen value through `pre_edge` and `autobk`.
  The floating ΔE0 you may know from **FEFF/path fitting** is a *downstream* fit
  parameter (out of scope here); if you fit paths later, that ΔE0 harmlessly absorbs the
  offset between whatever e0 you extracted with and the theory alignment.
- Default is the tabulated header `E0_tab` (deterministic; the files are calibrated).
  `find_e0` returns the derivative-max, which for Co sits ~5 eV **above** the tabulated
  onset (≈7714 vs 7709 on Co3NK_s) — expected, not an error.

Because a single e0 (+ shared energy grid + `kstep`) is reused for every spectrum in a
file, all χ(k) land on an **identical k-grid** — which is what lets them stack into a
matrix (asserted in `_process_matrix`).

## Step 3 — `xftf` (optional forward FT)

With `--ft`, χ(k) is Fourier-transformed to χ(R): `k`-window `[ft_kmin, ft_kmax]`
(default 3–12 Å⁻¹), weight `ft_kweight` (default 2), taper `ft_dk` (default 5). Sets
`group.r` and `group.chir_mag`. This is magnitude only — no path fitting.

## Scan merge (E-space)

For per-scan work, each original file's channels are first summed (`nansum`, total
fluorescence) into one μ(E) per scan. The **merged** spectrum shown in the plots is an
*E-space merge carried through*: the per-scan μ(E) are averaged, then `pre_edge` +
`autobk` run **once** on that mean μ. So the merged χ(k) is the spline of the averaged
spectrum — **not** the mean of the per-scan χ(k). (These differ whenever the per-scan
backgrounds differ.)

## Parameter reference (`Params`, in `model.py`)

| param | default | step | role |
|---|---|---|---|
| `mode` | `scan` | — | `scan` (sum each file's channels) / `channel` / `both` |
| `e0` | `None` | e0 | force edge energy; `None` → header `E0_tab` |
| `auto_e0` | `False` | e0 | detect e0 via `find_e0` instead of the header |
| `pre1`, `pre2` | None(file start), −50 | pre_edge | pre-edge fit window (eV rel. e0) |
| `norm1`, `norm2` | 75, None(file end) | pre_edge | post-edge fit window (eV rel. e0) |
| `nnorm` | 2 | pre_edge | post-edge polynomial degree |
| `rbkg` | 1.0 | autobk | R below which signal is treated as background (Å) |
| `kmin`, `kmax` | 0.0, `None` | autobk | χ(k) range (`None` = full) |
| `kweight` | 1 | autobk | k-weight for the background-fit FFT |
| `kstep` | 0.05 | autobk | χ(k) k-grid step (Å⁻¹) |
| `ft`, `ft_kmin`, `ft_kmax`, `ft_kweight`, `ft_dk` | off, 3, 12, 2, 5 | xftf | optional forward FT |

## Seeing it

`uv run xas-batch-plot INPUT.bcr.combined` renders the pre/post-edge fits, the
flattened result, and the normalized-μ + AUTOBK-spline with kⁿ·χ(k) — one place to
sanity-check every step above. In the EXAFS panel the gap between the solid (`norm`)
and dashed (normalized μ₀) curves *is* χ(k) before the E→k interpolation.
