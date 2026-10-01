# Level contributions

News explain *changes* of the nowcast. **Level contributions** explain the nowcast
itself: because the Kalman smoother is linear in the data, the nowcast is a weighted sum
of every observation in the information set plus a baseline (the contribution of the
initial state and of the constants),

$$
\mathbb{E}[y_\tau \mid \Omega] = \text{baseline} + \sum_{i} \sum_{t} \omega_{i,t}\, x_{i,t},
$$

and the contribution of series $i$ is $\sum_t \omega_{i,t} x_{i,t}$ (Koopman & Harvey,
2003). Contributions are in target units and sum exactly to the nowcast.

```python
import nowcastbox as nb

ds = nb.load_brazil_nowcast(
    columns=["pib", "ibc_br", "pim_geral", "pmc_varejo", "pms_volume", "ipca", "selic",
             "credito_concessoes_sa", "focus_pib"],
    start="2012-01",
    end="2024-12",
)
panel = nb.prepare_panel(ds.data, ds.transform, keep="pib")
vintage = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-11-15")

res = nb.MixedFreqDFM(n_factors=1, blocks="data").fit(vintage, "pib")
contrib = res.level_contributions(vintage, target_period="2024Q4")
print(contrib.summary())
contrib.check_identity()
```

```python
contrib.to_frame(by="series")
contrib.to_frame(by="category")
contrib.to_frame(by="block")
fig = contrib.plot("bar")
```

Because standardised series enter with their sample means removed, contributions measure
how far each indicator pulls the nowcast **away from the unconditional mean**, which is
part of the baseline. A positive contribution of retail sales means that recent retail
data are above average and push the GDP nowcast up.

The functional form is `nb.level_contributions(results, data, target_period, categories=...)`.
