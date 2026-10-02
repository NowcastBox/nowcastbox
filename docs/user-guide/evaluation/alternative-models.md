# Alternative models without groups of indicators

!!! note "New in 0.2.0"
    `nowcastbox.experiment.alternative_models` and `AlternativeNowcasts`.

How much does the nowcast depend on a particular group of indicators? Linzenich &
Meunier (2024, ECB Working Paper 3004, §3.5) answer with the *range of alternative
nowcasts*: the model is estimated again after removing each group of variables, and
each pair of groups, and the spread of the resulting nowcasts is published next to the
headline number. A narrow range means that the information is spread over the panel; a
wide one that the nowcast hinges on a few series (for example, on surveys during a
turning point). The idea is close to the forecast-combination literature (Granger &
Jeon, 2004; Timmermann, 2006): the alternatives are a population of reasonable models
whose disagreement is informative.

## Usage

```python
import warnings

from nowcastbox.datasets import load_simulated_dfm
from nowcastbox.experiment import alternative_models
from nowcastbox.models import TwoStepDFM

warnings.simplefilter("ignore")
data = load_simulated_dfm().data.select(["gdp", *(f"x{i:02d}" for i in range(1, 10))])
data = data.truncate(end="2019-10")
groups = {
    "real": ["x01", "x02", "x03"],
    "surveys": ["x04", "x05", "x06"],
    "financial": ["x07", "x08", "x09"],
}

alt = alternative_models(TwoStepDFM(n_factors=2), data, "gdp", by=groups, drop=(1, 2))
alt.n_models                      # 3 single drops + 3 pairs
alt.table()                       # one row per model, one column per target period
alt.range()                       # base, min, max, median, mean, spread, extreme models
print(alt.summary())
fig = alt.plot()                  # dots = alternatives, bar = range, diamond = base
```

The arguments are:

| Argument | Meaning |
|---|---|
| `model` | an estimator (`TwoStepDFM`, `MixedFreqDFM`, `BridgeEquation`, ...); a clone is fitted on `data` as the base model. Fitted results are also accepted with `refit=False`. |
| `target` | target name or formula (`"gdp ~ . - x10"`) |
| `by` | `"category"` (`SeriesMetadata.category`: hard, soft, financial, other), `"block"` (factor blocks; a series in several blocks belongs to each) or a mapping `{group: [series]}` / `{series: group}` |
| `drop` | number(s) of groups removed together (default `(1, 2)`) |
| `refit` | `True`: re-estimate every alternative; `False`: keep the base parameters |
| `groups` | restrict the droppable groups |
| `periods` | target periods (default: every out-of-sample period of the base model) |
| `n_jobs` | parallel jobs (joblib) for `refit=True` |
| `fit_kwargs` | extra arguments of `fit` (e.g. `{"horizon": 3}`) |

The target is never dropped, and series without a group always stay in the model.
Combinations that would remove every predictor (dropping a `global` block that contains
all the series, for instance) are skipped and listed in `alt.skipped`.

## Re-estimate or re-filter

With `refit=True` (default) each alternative is a new model estimated without the series
of the dropped groups: factors, loadings and bridge coefficients adapt to the smaller
panel. This is the exercise of the ECB toolbox and the most informative one, but it costs
one estimation per combination; use `n_jobs=-1` to spread them over the cores.

With `refit=False` the parameters of the base model are kept and the Kalman smoother is
re-run with the dropped series treated as missing. The difference to the base nowcast is
then exactly the contribution of the information in those series given the base model
(the linear functional of the news decomposition, Bańbura & Modugno, 2014). It is fast
and works for the state-space models (`MixedFreqDFM`, and `TwoStepDFM` with
`aggregate="factors"` or `"variables"`); other models raise a `TypeError` asking for
`refit=True`.

Fitted results can be passed with `refit=False` and `data=None` (the stored panel) or
with a newer vintage as `data`: the base row and every alternative are then re-filtered
on that same information set with the fitted parameters, and series the fitted model does
not contain are ignored.

```python
by_category = data
for series, category in {"x01": "hard", "x02": "hard", "x04": "soft", "x05": "soft"}.items():
    by_category = by_category.with_metadata(series, category=category)

base = TwoStepDFM(n_factors=2).fit(by_category, "gdp")
quick = alternative_models(base, None, "gdp", by="category", refit=False)
quick.table(deviation=True)       # change of each nowcast with respect to the base model
```

## Reading the output

`range()` gives, per target period, the base nowcast, the minimum, maximum, median and
mean of the alternatives, the `spread` (max − min) and the labels of the alternatives at
the extremes (`min_model`, `max_model`): a `max_model` of `"-financial"` means that the
nowcast would be highest without the financial indicators, i.e. they currently pull the
nowcast down. The pipeline Excel export (`outputs.excel`) writes the table and the range
to the sheets `alternatives` and `alternatives_range` when a run carries them.

## References

- Linzenich, J. & Meunier, B. (2024). *Nowcasting made easier: a toolbox for
  economists*. ECB Working Paper No. 3004, §3.5.
- Granger, C. W. J. & Jeon, Y. (2004). Thick modeling. *Economic Modelling*, 21(2),
  323–343.
- Timmermann, A. (2006). Forecast combinations. In *Handbook of Economic Forecasting*,
  vol. 1, 135–196.
- Bańbura, M. & Modugno, M. (2014). Maximum likelihood estimation of factor models on
  datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*,
  29(1), 133–160.

## See also

- [News decomposition](../news/news-decomposition.md): the contribution of each release
  to the change of the nowcast, a complementary view of the dependence on groups.
- [HTML reports](../visualization/reports.md) (`alternatives=`) and the pipeline output
  `alternatives` ([Pipeline and CLI](../pipeline-cli.md)).
