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
(stay below the edge onset) and `norm1 = +150` (start above the XANES / white line) —
while `pre1` and `norm2` default to `None`, which Larch resolves to the **file's first
and last energy**. Fitting the post-edge polynomial across the *whole* range (rather
than a narrow near-edge window extrapolated outward) is what makes the flattened μ sit
flat at ≈1.0 out to high k; a narrow `norm2` produces a drooping, unusable flat μ. All
four bounds are overridable (`--pre1/--pre2/--norm1/--norm2`, and `--nnorm`).

### Why `norm1 = 150` and not an element-specific table

`norm1` only needs to start the post-edge fit *above* the near-edge structure (XANES /
white line), which spans roughly the first ~30–50 eV above the edge. **150 eV is a fixed,
element-agnostic offset that clears it** — the long-standing ATHENA convention — and on
these data the flattening is insensitive to it (`norm1` from 75→300 all flatten to
≈1.000; only `edge_step`, i.e. χ amplitude, shifts a few %). There is **no rigorous
per-element `norm1` standard**, so a table would be false precision. Since scans reach
high k (k=8 ⇒ only ~244 eV above E₀, data here to ~860 eV) there is ample room to start
at 150.

Genuine element/edge dependence enters elsewhere:

- **E₀** — tabulated edge energies *are* element/edge-specific; taken from the header
  `E0_tab` (xraydb) by default. See [e0](#e0).
- **Usable post-edge window / `kmax`** — bounded by the *next* absorption edge (a higher
  edge of the same element, or another element in the sample). Irrelevant for an isolated
  Co K edge; it matters for L edges (L₃/L₂/L₁ are close) or multi-element samples. That
  constrains `norm2`/`kmax`, not `norm1`.

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

Resolved once per file (`resolve_e0`), order: explicit `Params.e0` > `find_e0` (default) >
header `E0_tab`. By **default** `find_e0` detects the edge from the **merged (mean)
spectrum** — the highest-SNR estimate, so one noisy channel can't skew it. `--header-e0`
switches to the tabulated value instead, and `--e0` forces a specific value.

`find_edge` adds two noise guards (following catXAS's `calculate_spectrum_e0`): the
derivative-max search is **restricted to `E0_tab ± 25 eV`** so a glitch or far feature
can't hijack it, and μ is lightly **Savitzky-Golay smoothed** in that window first. Both
are no-ops on a clean high-SNR spectrum but matter for noisy/glitchy ones.

**Granularity — per scan vs merged:**

- The **scan** block gets a **per-scan e0**: each summed scan (~30 channels) is high-SNR
  enough for its own `find_e0`. On this data the per-scan e0 is tight — ⟨E₀⟩ ≈ 7714.4 ±
  0.08 eV across 15 scans — and its mean matches the merged value to ~0.03 eV. Each
  scan's `edge_step` is likewise per scan. Both `scan_e0` and `scan_edge_step` are stored
  in the `.npz`.
- The **merged e0** (`find_e0` on the mean-of-scans μ) is the top-level representative
  value and the comparison reported in the plots.
- The **channel** block uses the **merged e0** (shared), *not* per-channel `find_e0`:
  per-channel detection scatters ~±2.7 eV **and is biased ~3 eV high** — because
  `find_e0` is the argmax of the derivative (nonlinear), noise on a single channel pulls
  the max around. You must **average the signal first, then find e0** ("average-then-find
  ≠ find-then-average"); summing into scans or the merge does exactly that.
- **Why `find_e0`, not the header?** The header `E0_tab` (7709 eV for Co K) is the
  *reference foil's* calibration energy, used upstream to align the energy axis — not
  this sample's edge. `find_e0` returns the sample's own derivative-max (≈7714), the
  more appropriate EXAFS origin. `--header-e0` opts back into the tabulated value.
- **e0 is fixed during splining — it does not float.** It sets the k-axis origin and the
  edge-step reference, held constant through `pre_edge` and `autobk`. The floating ΔE0
  you may know from **FEFF/path fitting** is a *downstream* fit parameter (out of scope
  here); if you fit paths later, that ΔE0 harmlessly absorbs the extraction-e0 choice.

A single `kmax` is resolved **once per file** — from the *highest* e0 in use (its data
covers the smallest k range), floored to a `kstep` multiple — and applied to every scan,
the merged, and every channel. So they all share an **identical** k-grid
`[kmin, kmin+kstep, …, kmax]` regardless of their individual e0; only the E→k *mapping*
shifts by the tiny per-scan e0 spread (~0.08 eV). Recorded as `kmax_used` in `meta`. (An
explicit `--kmax` is used verbatim instead.)

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
| `e0` | `None` | e0 | force edge energy; `None` → detect/tabulated per `auto_e0` |
| `auto_e0` | `True` | e0 | `find_e0` on the merged μ (default); `False` → header `E0_tab` |
| `pre1`, `pre2` | None(file start), −50 | pre_edge | pre-edge fit window (eV rel. e0) |
| `norm1`, `norm2` | 150, None(file end) | pre_edge | post-edge fit window (eV rel. e0) |
| `nnorm` | 2 | pre_edge | post-edge polynomial degree |
| `rbkg` | 1.0 | autobk | R below which signal is treated as background (Å) |
| `kmin`, `kmax` | 0.0, `None` | autobk | χ(k) range (`None` = one shared kmax/file from the data) |
| `kweight` | 1 | autobk | k-weight for the background-fit FFT |
| `kstep` | 0.05 | autobk | χ(k) k-grid step (Å⁻¹) |
| `ft`, `ft_kmin`, `ft_kmax`, `ft_kweight`, `ft_dk` | off, 3, 12, 2, 5 | xftf | optional forward FT |

## Seeing it

`uv run xas-batch-plot INPUT.bcr.combined` renders the pre/post-edge fits, the
flattened result, and the normalized-μ + AUTOBK-spline with kⁿ·χ(k) — one place to
sanity-check every step above. In the EXAFS panel the gap between the solid (`norm`)
and dashed (normalized μ₀) curves *is* χ(k) before the E→k interpolation.

## References

- M. Newville, *Fundamentals of XAFS*, Rev. Mineral. Geochem. **78**, 33–74 (2014).
  doi:[10.2138/rmg.2014.78.2](https://doi.org/10.2138/rmg.2014.78.2) — normalization,
  AUTOBK background, and χ(k) conventions used here.
- B. Ravel & M. Newville, *ATHENA, ARTEMIS, HEPHAESTUS…*, J. Synchrotron Rad. **12**,
  537–541 (2005). doi:[10.1107/S0909049505012719](https://doi.org/10.1107/S0909049505012719)
  — the widely-used defaults these choices mirror (incl. the ~150 eV post-edge start).
- S. Calvin, *XAFS for Everyone*, CRC Press (2013) — practical guidance on picking
  pre/post-edge ranges and normalization order.
- Larch `pre_edge` / `autobk` / `xftf`:
  <https://xraypy.github.io/xraylarch/xafs/preedge.html>,
  [.../autobk.html](https://xraypy.github.io/xraylarch/xafs/autobk.html).
- Tabulated edge energies E₀ (the genuinely element/edge-specific input): xraydb,
  <https://xraypy.github.io/XrayDB/>.
