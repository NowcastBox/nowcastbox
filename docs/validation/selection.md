# Selection vs `ICfactors` and `ICshocks`

Tests: `tests/reference_validation/test_r_selection.py`. Fixtures:
`scripts/reference_fixtures/r_selection.R`.

The panels are the balanced parts of `Bpanel(USGDP)` (189 series x 306 months),
`Bpanel(BRGDP)` (86 x 215) and the simulated predictors (24 x 238).

## Number of factors: Bai & Ng (2002)

`ICfactors(x, rmax, type)` is the same computation as
`select_factors(x, rmax, criterion=f"IC{type}")` on the standardised panel. R reports
r = 1..rmax; nowcastbox also reports r = 0.

| Cases | Identical r\* | Max abs error of IC_p(r) | Plan tolerance |
|---:|---:|---:|---:|
| 18 (3 panels x rmax ∈ {8, 15} x IC_p1..IC_p3) | 18 | 6.4e-15 | exact / 1e-8 |

Both libraries pick r\* = rmax on the 24-series simulated panel, a known weakness of the
criteria when N is small (lacuna 6). The behaviour is the same in both.

## Number of shocks: Bai & Ng (2007)

`ICshocks` returns only q\*. To see which statistic and bound it uses, the fixture
script bisects on `m` to find the values at which q\* changes. With the bound written as
M = m / d(N, T, δ), these thresholds equal D_k · d(N, T, δ).

- The statistic is **D1** on the eigenvalues of the VAR residual covariance, the default
  of `select_shocks`. The thresholds are proportional to nowcastbox's D1 values.
- The bound is **m / min(N, T)^{1/(2−δ)}**: this reproduces 44 thresholds over three
  panels and four values of δ to 1e-10. Bai & Ng (2007) define
  M_NT = m / min(N^{1/2−δ}, T^{1/2−δ}), which is what `shock_bound` implements.

R's bound is smaller, so R selects more shocks: q\* = r in most cases with the defaults
δ = 0.1, m = 1. With R's bound substituted, nowcastbox reproduces every q\* of the grid
(3 panels x 4 (δ, m) settings x r = 2..6 x p = 1..3). The test
`test_icshocks_bound_differs_from_paper` documents the difference
(`reference_divergence`); nowcastbox keeps the published bound.
