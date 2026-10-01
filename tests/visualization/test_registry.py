import pytest

import nowcastbox.visualization as viz
from nowcastbox.core.results import available_plots


def test_registered_kinds(two_step, em):
    assert set(viz.REGISTERED_PLOTS) <= set(available_plots(em))
    assert "loglikelihood" not in available_plots(two_step)
    for kind in ("forecast", "factors", "eigenvalues", "loadings", "fan", "data_availability"):
        assert kind in available_plots(two_step)


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_results_plot_every_kind(two_step, em, backend):
    for res in (two_step, em):
        for kind in available_plots(res):
            if kind not in viz.REGISTERED_PLOTS:
                continue  # plots registered by other modules
            assert res.plot(kind, backend=backend) is not None


def test_results_plot_kwargs(two_step):
    fig = two_step.plot("loadings", style="bar", factors=["f1"])
    assert len(fig.data) == 1
    with pytest.raises(NotImplementedError):
        two_step.plot("loglikelihood")


def test_public_api():
    for name in viz.__all__:
        assert hasattr(viz, name)


# ---------------------------------------------------------------------- wave-2 objects
@pytest.fixture
def news_and_tracker(em):
    data = em.data
    news = em.news(data.truncate(end=data.index[-3]), data)
    tracker = em.nowcast_tracker(
        data, 5, news.target_period, dates=[data.index[-4].end_time, data.index[-1].end_time]
    )
    return news, tracker


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_news_objects_on_both_backends(news_and_tracker, backend):
    news, tracker = news_and_tracker
    for by in ("category", "series"):
        assert news.plot("waterfall", by=by, backend=backend) is not None
    assert tracker.plot("path", backend=backend) is not None


def test_news_objects_theme_uses_visualization(news_and_tracker):
    import matplotlib.figure

    news, tracker = news_and_tracker
    assert isinstance(news.plot("waterfall", theme="academic"), matplotlib.figure.Figure)
    assert isinstance(tracker.plot("path", theme="academic"), matplotlib.figure.Figure)
    assert isinstance(news.plot("waterfall"), matplotlib.figure.Figure)  # news default


def test_density_kind(two_step):
    fig = two_step.plot("density", n_boot=0, title="Density")
    assert fig.layout.title.text == "Density"
    assert two_step.plot("density", backend="matplotlib") is not None
