from itertools import islice

import numpy as np
import pytest
from scipy import stats
from scipy.special import logit

import bidlab as bl
from bidlab.bidders import Linear
from bidlab.config import Campaign, SimConfig
from bidlab.markets import (
    IndependentValues,
    LogisticMarket,
    MarketDraw,
    Segment,
    Segmented,
)
from bidlab.simulator import run


def pcvr_rows(market, count, steps, seed=0):
    draws = market.draws(count, np.random.default_rng(seed))
    return np.array([draw.pcvr for draw in islice(draws, steps)])


def test_logistic_probabilities_inside_unit_interval():
    rates = pcvr_rows(LogisticMarket(feature_count=4), count=3, steps=50, seed=1)
    assert rates.shape == (50, 3)
    assert np.all((rates > 0) & (rates < 1))


@pytest.mark.parametrize(("fewer", "more"), [(1, 2), (1, 3), (2, 4), (1, 8), (3, 5)])
def test_logistic_adding_bidder_keeps_draws(fewer, more):
    for seed in range(3):
        small = pcvr_rows(LogisticMarket(), fewer, steps=500, seed=seed)
        large = pcvr_rows(LogisticMarket(), more, steps=500, seed=seed)
        np.testing.assert_array_equal(small, large[:, :fewer])


def test_cvr_bias_mean_sets_typical_conversion_rate():
    rates = pcvr_rows(LogisticMarket(cvr_bias_mean=-3.0), count=20, steps=2000, seed=2)
    # σ(-3) ≈ 0.047; feature noise spreads it, so the mean sits near 5-10%.
    assert 0.03 < rates.mean() < 0.12


def segment_draws(market, count, steps, seed=0):
    return list(islice(market.draws(count, np.random.default_rng(seed)), steps))


def test_segment_shares_without_feature():
    market = Segmented((Segment("news", 0.7), Segment("sports", 0.3)))
    draws = segment_draws(market, count=2, steps=20_000)
    names = [draw.segment for draw in draws]
    assert set(names) == {"news", "sports"}
    assert all(set(draw.features) == {"vector"} for draw in draws)
    assert abs(names.count("news") / len(names) - 0.7) < 0.02


def test_lift_shifts_logit():
    lifted = Segmented((Segment("plain", 0.5), Segment("lifted", 0.5, lift=2.0)))
    flat = Segmented((Segment("plain", 0.5), Segment("lifted", 0.5)))
    lifted_draws = segment_draws(lifted, count=3, steps=4000, seed=5)
    flat_draws = segment_draws(flat, count=3, steps=4000, seed=5)
    in_lifted = np.array([draw.segment == "lifted" for draw in flat_draws])
    lifted_rates = np.array([draw.pcvr for draw in lifted_draws])
    base_rates = np.array([draw.pcvr for draw in flat_draws])
    base_logits = np.log(base_rates / (1 - base_rates))
    np.testing.assert_allclose(
        lifted_rates[in_lifted], 1 / (1 + np.exp(-(base_logits[in_lifted] + 2.0)))
    )
    np.testing.assert_array_equal(lifted_rates[~in_lifted], base_rates[~in_lifted])
    mean_lifted = lifted_rates[in_lifted].mean(axis=0)
    mean_plain = lifted_rates[~in_lifted].mean(axis=0)
    assert np.all(mean_lifted > mean_plain)


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ((), "Segmented needs at least one segment"),
        ((Segment("a", 0.0), Segment("b", 1.0)), "share must be > 0"),
        ((Segment("a", -0.5), Segment("b", 1.5)), "share must be > 0"),
        ((Segment("a", 0.7), Segment("b", 0.2)), "sum to 1"),
        ((Segment("a", 0.5), Segment("a", 0.5)), "unique"),
    ],
)
def test_segmented_rejects_bad_segments(segments, message):
    with pytest.raises(ValueError, match=message):
        Segmented(segments)


@pytest.mark.parametrize(("fewer", "more"), [(1, 2), (1, 3), (2, 4), (1, 8), (3, 5)])
def test_segmented_adding_bidder_keeps_draws(fewer, more):
    market = Segmented((Segment("news", 0.6), Segment("sports", 0.4, lift=1.0)))
    for seed in range(3):
        small = segment_draws(market, count=fewer, steps=500, seed=seed)
        large = segment_draws(market, count=more, steps=500, seed=seed)
        assert [d.segment for d in small] == [d.segment for d in large]
        np.testing.assert_array_equal(
            np.array([d.pcvr for d in small]),
            np.array([d.pcvr for d in large])[:, :fewer],
        )


def test_independent_uniform_mean_and_correlation():
    rates = pcvr_rows(IndependentValues(stats.uniform(0, 1)), 2, steps=20_000)
    assert np.all(np.abs(rates.mean(axis=0) - 0.5) < 0.01)
    assert abs(np.corrcoef(rates[:, 0], rates[:, 1])[0, 1]) < 0.03


def test_independent_uniform_values_reach_the_log():
    config = SimConfig(
        campaigns=[
            Campaign(name, Linear(), budget=1e9, event_value=1.0) for name in ("a", "b")
        ],
        market=IndependentValues(stats.uniform(0, 1)),
        episodes=2,
        steps=10_000,
    )
    values = run(config).true_values.reshape(-1, 2)
    assert np.all(np.abs(values.mean(axis=0) - 0.5) < 0.01)
    assert abs(np.corrcoef(values[:, 0], values[:, 1])[0, 1]) < 0.03


def test_independent_values_clipped():
    rates = pcvr_rows(IndependentValues(stats.norm(0.5, 1)), 3, steps=3000)
    assert rates.min() == 0.0
    assert rates.max() == 1.0
    assert np.all((rates >= 0.0) & (rates <= 1.0))


def test_independent_rejects_non_distribution():
    with pytest.raises(TypeError, match="rvs method.*got float"):
        IndependentValues(0.5)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


@pytest.mark.parametrize(("fewer", "more"), [(1, 2), (2, 5), (1, 8)])
def test_independent_adding_bidder_keeps_values(fewer, more):
    market = IndependentValues(stats.beta(2, 5))
    for seed in range(3):
        small = pcvr_rows(market, fewer, steps=3000, seed=seed)
        large = pcvr_rows(market, more, steps=3000, seed=seed)
        np.testing.assert_array_equal(small, large[:, :fewer])


def test_market_draw_stores_float_array_and_read_only_features():
    draw = MarketDraw({"hour": 3}, [0, 1])  # pyright: ignore[reportArgumentType]  Reason: the test checks that a list becomes a float array.
    assert isinstance(draw.pcvr, np.ndarray)
    assert draw.pcvr.dtype == np.float64
    assert draw.segment == "all"
    assert draw.pctr is None
    clicks = MarketDraw({}, [0, 1], pctr=[0, 1])  # pyright: ignore[reportArgumentType]  Reason: the test checks that lists become float arrays.
    assert isinstance(clicks.pctr, np.ndarray)
    assert clicks.pctr.dtype == np.float64
    with pytest.raises(TypeError):
        draw.features["hour"] = 4  # pyright: ignore[reportIndexIssue]  Reason: the test checks that features are read-only.


def test_segments_list_stored_as_tuple():
    market = Segmented([Segment("news", 0.5), Segment("sports", 0.5)])
    assert market.segments == (Segment("news", 0.5), Segment("sports", 0.5))
    assert market.segment_names == ("news", "sports")


def logistic_draws(market, count, steps, seed=0):
    return list(islice(market.draws(count, np.random.default_rng(seed)), steps))


def stacked(draws, name):
    return np.array([getattr(draw, name) for draw in draws])


def vectors(draws):
    return np.array([draw.features["vector"] for draw in draws])


def test_click_stage_keeps_pcvr_and_features():
    for seed in range(3):
        plain = logistic_draws(LogisticMarket(), 3, steps=500, seed=seed)
        clicks = logistic_draws(LogisticMarket(ctr_bias_mean=-3.0), 3, 500, seed)
        np.testing.assert_array_equal(stacked(plain, "pcvr"), stacked(clicks, "pcvr"))
        np.testing.assert_array_equal(vectors(plain), vectors(clicks))
        assert all(draw.pctr is not None for draw in clicks)


@pytest.mark.parametrize(("fewer", "more"), [(1, 2), (2, 4), (1, 8)])
def test_adding_campaign_keeps_click_draws(fewer, more):
    market = LogisticMarket(ctr_bias_mean=-3.0)
    for seed in range(3):
        small = logistic_draws(market, fewer, steps=500, seed=seed)
        large = logistic_draws(market, more, steps=500, seed=seed)
        for name in ("pctr", "pcvr"):
            np.testing.assert_array_equal(
                stacked(small, name), stacked(large, name)[:, :fewer]
            )
        np.testing.assert_array_equal(vectors(small), vectors(large))


def test_default_features_vector_read_only():
    (draw,) = logistic_draws(LogisticMarket(), 3, steps=1)
    vector = draw.features["vector"]
    assert draw.pctr is None
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (8,)
    assert vector.dtype == np.float64
    with pytest.raises(ValueError, match="read-only"):
        vector[0] = 1.0


@pytest.mark.parametrize("public", [0, 3, 8])
def test_public_features_sets_vector_length(public):
    draws = logistic_draws(LogisticMarket(public_features=public), 2, steps=5)
    assert vectors(draws).shape == (5, public)


def fit_residuals(public_vectors, rates):
    inputs = np.column_stack([public_vectors, np.ones(len(public_vectors))])
    targets = logit(rates)
    weights, *_ = np.linalg.lstsq(inputs, targets, rcond=None)
    return targets - inputs @ weights


def test_all_public_features_explain_pcvr():
    draws = logistic_draws(LogisticMarket(public_features=8), 4, steps=2000)
    residuals = fit_residuals(vectors(draws), stacked(draws, "pcvr"))
    assert np.abs(residuals).max() < 1e-9


def test_hidden_features_leave_residual():
    draws = logistic_draws(LogisticMarket(public_features=3), 4, steps=2000)
    residuals = fit_residuals(vectors(draws), stacked(draws, "pcvr"))
    assert np.all(residuals.std(axis=0) > 0.1)


@pytest.mark.parametrize(
    ("public", "error", "got"),
    [
        (-1, ValueError, "got -1"),
        (9, ValueError, "got 9"),
        (2.0, TypeError, "got float"),
        (True, TypeError, "got bool"),
    ],
)
def test_bad_public_features_rejected(public, error, got):
    with pytest.raises(error, match=f"public_features.*{got}"):
        LogisticMarket(public_features=public)


def test_ctr_bias_sets_click_rate():
    draws = logistic_draws(LogisticMarket(ctr_bias_mean=-4.0), 20, steps=2000)
    pctr = stacked(draws, "pctr")
    assert np.all((pctr > 0) & (pctr < 1))
    assert abs(logit(pctr).mean() - -4.0) < 0.35
    assert np.abs(fit_residuals(vectors(draws), pctr)).max() < 1e-9


@pytest.mark.parametrize("old", ["context_size", "bias_mean", "bias_sd"])
def test_old_market_names_removed(old):
    with pytest.raises(TypeError, match=old):
        LogisticMarket(**{old: 1.0})  # pyright: ignore[reportArgumentType]  Reason: the test passes removed names on purpose.
    with pytest.raises(AttributeError):
        _ = bl.markets.LogisticContext  # pyright: ignore[reportAttributeAccessIssue]  Reason: the test checks the old name is gone.
    assert not hasattr(MarketDraw({}, np.zeros(1)), "probabilities")


def test_segmented_passes_click_stage():
    base = LogisticMarket(ctr_bias_mean=-3.0)
    flat = Segmented((Segment("plain", 0.5), Segment("lifted", 0.5)), base)
    lifted = Segmented((Segment("plain", 0.5), Segment("lifted", 0.5, lift=1.0)), base)
    flat_draws = logistic_draws(flat, 3, steps=500)
    lifted_draws = logistic_draws(lifted, 3, steps=500)
    np.testing.assert_array_equal(
        stacked(flat_draws, "pctr"), stacked(lifted_draws, "pctr")
    )
    np.testing.assert_array_equal(vectors(flat_draws), vectors(lifted_draws))
    shift = logit(stacked(lifted_draws, "pcvr")) - logit(stacked(flat_draws, "pcvr"))
    in_lifted = np.array([draw.segment == "lifted" for draw in lifted_draws])
    np.testing.assert_allclose(shift[in_lifted], 1.0, atol=1e-9)
    assert np.all(shift[~in_lifted] == 0.0)
    assert all(set(draw.features) == {"vector"} for draw in lifted_draws)


def test_ctr_bias_sd_sets_spread_across_campaigns():
    market = LogisticMarket(ctr_bias_mean=-4.0, ctr_bias_sd=0.5)
    draws = logistic_draws(market, 60, steps=2000)
    campaign_means = logit(stacked(draws, "pctr")).mean(axis=0)
    assert abs(campaign_means.std() - 0.5) < 0.15


def test_click_and_conversion_logits_vary_alike():
    # Both stages use ad vectors of the same scale on the same features.
    draws = logistic_draws(LogisticMarket(ctr_bias_mean=-3.0), 60, steps=2000)
    ctr_spread = logit(stacked(draws, "pctr")).var(axis=0).mean()
    cvr_spread = logit(stacked(draws, "pcvr")).var(axis=0).mean()
    assert 0.7 < ctr_spread / cvr_spread < 1.4


def test_market_draw_copies_writable_arrays_only():
    writable = np.zeros(2)
    draw = MarketDraw({"v": writable}, np.zeros(1))
    writable[0] = 1.0
    stored = draw.features["v"]
    assert isinstance(stored, np.ndarray)
    assert stored[0] == 0.0
    assert not stored.flags.writeable
    frozen = np.zeros(2)
    frozen.flags.writeable = False
    assert MarketDraw({"v": frozen}, np.zeros(1)).features["v"] is frozen
