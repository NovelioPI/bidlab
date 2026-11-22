from itertools import islice
from typing import Any

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.special import expit, logit

import bidlab as bl
from bidlab.bidders import Linear, OracleValue, Stack
from bidlab.config import Campaign, SimConfig
from bidlab.markets import (
    IndependentValues,
    LogisticMarket,
    MarketDraw,
    Segment,
    Segmented,
    _solve_biases,  # pyright: ignore[reportPrivateUsage]  Reason: the solver's accuracy is checked against an exact integral.
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


def test_cvr_lift_shifts_logit():
    lifted = Segmented((Segment("plain", 0.5), Segment("lifted", 0.5, cvr_lift=2.0)))
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
    market = Segmented((Segment("news", 0.6), Segment("sports", 0.4, cvr_lift=1.0)))
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
            Campaign(name, Stack(OracleValue(), Linear()), budget=1e9, base_value=1.0)
            for name in ("a", "b")
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
    draw = MarketDraw({"hour": 3}, pcvr=[0, 1])  # pyright: ignore[reportArgumentType]  Reason: the test checks that a list becomes a float array.
    assert isinstance(draw.pcvr, np.ndarray)
    assert draw.pcvr.dtype == np.float64
    assert draw.segment == "all"
    assert draw.pctr is None
    clicks = MarketDraw({}, pctr=[0, 1], pcvr=[0, 1])  # pyright: ignore[reportArgumentType]  Reason: the test checks that lists become float arrays.
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


@pytest.mark.parametrize(
    "old", ["context_size", "bias_mean", "bias_sd", "ctr_bias_sd", "cvr_bias_sd"]
)
def test_old_market_names_removed(old):
    with pytest.raises(TypeError, match=old):
        LogisticMarket(**{old: 1.0})  # pyright: ignore[reportArgumentType]  Reason: the test passes removed names on purpose.
    with pytest.raises(AttributeError):
        _ = bl.markets.LogisticContext  # pyright: ignore[reportAttributeAccessIssue]  Reason: the test checks the old name is gone.
    assert not hasattr(MarketDraw({}, pcvr=np.zeros(1)), "probabilities")


def test_segment_lift_removed():
    with pytest.raises(TypeError, match="lift"):
        Segment("a", 1.0, lift=1.0)  # pyright: ignore[reportCallIssue]  Reason: the test passes a removed name on purpose.


def test_segmented_passes_click_stage():
    base = LogisticMarket(ctr_bias_mean=-3.0)
    flat = Segmented((Segment("plain", 0.5), Segment("lifted", 0.5)), base)
    lifted = Segmented(
        (Segment("plain", 0.5), Segment("lifted", 0.5, cvr_lift=1.0)), base
    )
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


def test_ctr_spread_across_campaigns():
    market = LogisticMarket(ctr_bias_mean=-4.0, ctr_spread=0.5)
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
    draw = MarketDraw({"v": writable}, pcvr=np.zeros(1))
    writable[0] = 1.0
    stored = draw.features["v"]
    assert isinstance(stored, np.ndarray)
    assert stored[0] == 0.0
    assert not stored.flags.writeable
    frozen = np.zeros(2)
    frozen.flags.writeable = False
    assert MarketDraw({"v": frozen}, pcvr=np.zeros(1)).features["v"] is frozen


def test_market_draw_rates_are_keyword_only():
    # pctr and pcvr share a type, so a positional swap would pass unnoticed.
    with pytest.raises(TypeError):
        MarketDraw({}, np.zeros(1))  # pyright: ignore[reportCallIssue]  Reason: the test checks positional rates are rejected.


def test_market_draw_price_defaults():
    draw = MarketDraw({}, pcvr=np.array([0.1]))
    assert draw.market_price is None
    assert draw.floor == 0.0
    priced = MarketDraw({}, pcvr=np.array([0.1]), market_price=1, floor=0)
    assert type(priced.market_price) is float and type(priced.floor) is float


PRICE_STEPS = 20_000


def prices(draws):
    return np.array([draw.market_price for draw in draws])


def floors(draws):
    return np.array([draw.floor for draw in draws])


def test_logistic_default_has_no_market_price():
    draws = logistic_draws(LogisticMarket(ctr_bias_mean=-3.0), 3, steps=200)
    assert all(draw.market_price is None for draw in draws)
    assert all(draw.floor == 0.0 for draw in draws)


def test_price_median_and_spread():
    market = LogisticMarket(price_median=0.5, price_spread=0.5, price_quality=0.0)
    price = prices(logistic_draws(market, 2, steps=PRICE_STEPS))
    assert abs(np.median(price) / 0.5 - 1) < 0.03
    assert abs(np.log(price).std() - 0.5) < 0.03


def price_fit(quality):
    market = LogisticMarket(price_median=0.5, price_spread=0.5, price_quality=quality)
    draws = logistic_draws(market, 2, steps=PRICE_STEPS)
    design = np.column_stack([np.ones(PRICE_STEPS), vectors(draws)])
    log_price = np.log(prices(draws))
    coefficients, *_ = np.linalg.lstsq(design, log_price, rcond=None)
    residuals = log_price - design @ coefficients
    return coefficients[1:], residuals.std()


def test_price_follows_quality():
    coefficients, residual_sd = price_fit(quality=1.0)
    assert abs(residual_sd - 0.5) < 0.03
    # The features explain the rest: a quality of unit variance on average.
    assert np.abs(coefficients).max() > 0.1
    flat, _ = price_fit(quality=0.0)
    assert np.abs(flat).max() < 0.03


def mean_price_value_correlation(similarity):
    market = LogisticMarket(
        price_median=0.5,
        price_spread=0.5,
        price_quality=1.0,
        rival_similarity=similarity,
    )
    draws = logistic_draws(market, 20, steps=PRICE_STEPS)
    log_price = np.log(prices(draws))
    cvr_logits = logit(stacked(draws, "pcvr"))
    return np.mean([np.corrcoef(log_price, column)[0, 1] for column in cvr_logits.T])


def test_rival_similarity_links_price_to_value():
    expected = 0.8 / np.sqrt(1 + 0.5**2)
    assert abs(mean_price_value_correlation(0.8) - expected) < 0.15
    assert abs(mean_price_value_correlation(0.0)) < 0.1


@pytest.mark.parametrize(
    "settings",
    [
        {"price_median": 0.5},
        {"price_median": 0.5, "price_spread": 1.0, "price_quality": 0.3},
        {"price_median": 0.5, "floor_share": (2.0, 5.0)},
    ],
)
def test_price_settings_keep_rates_and_features(settings):
    for seed in range(3):
        plain = logistic_draws(LogisticMarket(ctr_bias_mean=-3.0), 3, 500, seed)
        priced = logistic_draws(
            LogisticMarket(ctr_bias_mean=-3.0, **settings), 3, 500, seed
        )
        for name in ("pctr", "pcvr"):
            np.testing.assert_array_equal(stacked(plain, name), stacked(priced, name))
        np.testing.assert_array_equal(vectors(plain), vectors(priced))
        assert all(draw.market_price is not None for draw in priced)


def test_floor_share_of_market_price():
    market = LogisticMarket(price_median=0.5, floor_share=(2.0, 5.0))
    draws = logistic_draws(market, 2, steps=PRICE_STEPS)
    price, floor = prices(draws), floors(draws)
    assert np.all((floor >= 0) & (floor <= price))
    assert abs((floor / price).mean() - 2 / 7) < 0.02


@pytest.mark.parametrize(
    ("settings", "name"),
    [
        ({"floor_share": (2.0, 5.0)}, "floor_share needs price_median"),
        ({"price_median": 0.0}, "price_median"),
        ({"price_median": -1.0}, "price_median"),
        ({"price_median": 0.5, "price_spread": -0.1}, "price_spread"),
        ({"price_median": 0.5, "floor_share": (0.0, 5.0)}, "floor_share"),
        ({"price_median": 0.5, "floor_share": (2.0, -1.0)}, "floor_share"),
        ({"price_median": 0.5, "floor_share": (2.0,)}, "floor_share"),
        ({"rival_similarity": -0.1}, "rival_similarity"),
        ({"rival_similarity": 1.1}, "rival_similarity"),
    ],
)
def test_bad_price_settings_rejected(settings, name):
    with pytest.raises(ValueError, match=name):
        LogisticMarket(**settings)


def test_price_settings_at_their_limits_accepted():
    LogisticMarket(price_median=0.5, price_spread=0.0, rival_similarity=0.0)
    LogisticMarket(rival_similarity=1.0)
    LogisticMarket(price_median=0.5, floor_share=(0.5, 0.5))


@pytest.mark.parametrize(("fewer", "more"), [(1, 2), (2, 5)])
def test_adding_campaign_keeps_price_and_floor(fewer, more):
    market = LogisticMarket(
        ctr_bias_mean=-3.0,
        price_median=0.5,
        floor_share=(2.0, 5.0),
        rival_similarity=0.6,
    )
    for seed in range(3):
        small = logistic_draws(market, fewer, steps=500, seed=seed)
        large = logistic_draws(market, more, steps=500, seed=seed)
        np.testing.assert_array_equal(prices(small), prices(large))
        np.testing.assert_array_equal(floors(small), floors(large))


def test_segmented_passes_price_and_floor():
    base = LogisticMarket(price_median=0.5, floor_share=(2.0, 5.0))
    segmented = Segmented((Segment("a", 0.5), Segment("b", 0.5, cvr_lift=1.0)), base)
    draws = logistic_draws(segmented, 3, steps=500, seed=4)
    base_rng = np.random.default_rng(4).spawn(2)[0]
    base_draws = list(islice(base.draws(3, base_rng), 500))
    np.testing.assert_array_equal(prices(draws), prices(base_draws))
    np.testing.assert_array_equal(floors(draws), floors(base_draws))
    assert np.all(floors(draws) > 0)


def independent_draws(count, steps=PRICE_STEPS, seed=0, **settings):
    market = IndependentValues(stats.uniform(0, 0.1), **settings)
    return list(islice(market.draws(count, np.random.default_rng(seed)), steps))


def test_independent_values_price_and_floor():
    # uniform(-0.1, 1) puts a tenth of the prices below 0, clipped to 0.
    settings = {"market_price": stats.uniform(-0.1, 1.0), "floor_share": (2, 5)}
    draws = independent_draws(3, **settings)
    price, floor = prices(draws), floors(draws)
    assert abs((price == 0).mean() - 0.1) < 0.01
    assert abs(price[price > 0].mean() - 0.45) < 0.01
    assert price.max() <= 0.9
    assert np.all((floor >= 0) & (floor <= price))
    assert abs((floor[price > 0] / price[price > 0]).mean() - 2 / 7) < 0.02
    plain = independent_draws(3)
    np.testing.assert_array_equal(stacked(plain, "pcvr"), stacked(draws, "pcvr"))
    assert all(draw.market_price is None and draw.floor == 0 for draw in plain)
    more = independent_draws(5, **settings)
    np.testing.assert_array_equal(prices(more), price)
    np.testing.assert_array_equal(floors(more), floor)


def test_independent_price_without_floor():
    draws = independent_draws(2, steps=50, market_price=stats.uniform(0.2, 0.1))
    price = prices(draws)
    assert np.all((price >= 0.2) & (price <= 0.3))
    assert all(draw.floor == 0.0 for draw in draws)


def test_independent_bad_price_settings_rejected():
    with pytest.raises(ValueError, match="floor_share needs market_price"):
        IndependentValues(stats.uniform(0, 0.1), floor_share=(2, 5))
    with pytest.raises(ValueError, match="floor_share must be"):
        IndependentValues(
            stats.uniform(0, 0.1), market_price=stats.uniform(), floor_share=(0, 5)
        )
    with pytest.raises(TypeError, match="^distribution must have an rvs method"):
        IndependentValues(0.5)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a number.
    with pytest.raises(TypeError, match="^market_price must have an rvs method"):
        IndependentValues(stats.uniform(0, 0.1), market_price=0.5)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a number.


def test_floor_share_stored_as_float_tuple():
    logistic = LogisticMarket(price_median=0.5, floor_share=[2, 5])  # pyright: ignore[reportArgumentType]  Reason: the test checks a list is stored as a tuple.
    independent = IndependentValues(
        stats.uniform(0, 0.1),
        market_price=stats.uniform(),
        floor_share=[2, 5],  # pyright: ignore[reportArgumentType]  Reason: the test checks a list is stored as a tuple.
    )
    for market in (logistic, independent):
        assert market.floor_share == (2.0, 5.0)
        assert type(market.floor_share) is tuple
        assert all(type(parameter) is float for parameter in market.floor_share)


@pytest.mark.parametrize("similarity", [0.8, 1.0])
def test_ad_vectors_mix_market_vector(similarity):
    # Without price noise, log(m / median) is the quality q = x · ψ exactly.
    settings: dict[str, Any] = {
        "ctr_bias_mean": -3.0,
        "price_median": 0.5,
        "price_spread": 0.0,
    }
    plain = logistic_draws(LogisticMarket(**settings), 4, steps=300)
    mixed = logistic_draws(
        LogisticMarket(rival_similarity=similarity, **settings), 4, steps=300
    )
    quality = np.log(prices(plain) / 0.5)[:, None]
    own = np.sqrt(1 - similarity**2)
    for name in ("pctr", "pcvr"):
        mixed_logits = logit(stacked(mixed, name))
        plain_logits = logit(stacked(plain, name))
        rest = mixed_logits - own * plain_logits - similarity * quality
        # Each ad vector is ρ ψ + √(1 − ρ²) η, so only (1 − √(1 − ρ²)) × bias is left.
        np.testing.assert_allclose(
            rest, np.broadcast_to(rest[0], rest.shape), atol=1e-9
        )


def test_rival_similarity_without_price_keeps_rates():
    settings: dict[str, Any] = {"ctr_bias_mean": -3.0, "rival_similarity": 0.6}
    alone = logistic_draws(LogisticMarket(**settings), 3, steps=300)
    priced = logistic_draws(LogisticMarket(price_median=0.5, **settings), 3, steps=300)
    for name in ("pctr", "pcvr"):
        np.testing.assert_array_equal(stacked(alone, name), stacked(priced, name))
    plain = logistic_draws(LogisticMarket(ctr_bias_mean=-3.0), 3, steps=300)
    assert not np.array_equal(stacked(alone, "pcvr"), stacked(plain, "pcvr"))


RATE_STEPS = 20_000


def assert_campaign_means_near(rates, target):
    # Steps are independent, so 4 standard errors of the step mean rarely fail by chance.
    means = rates.mean(axis=0)
    standard_errors = rates.std(axis=0) / np.sqrt(len(rates))
    assert np.all(np.abs(means - target) < 4 * standard_errors), (means, target)


@pytest.mark.parametrize(
    "settings",
    [{}, {"rival_similarity": 0.8, "public_features": 3, "price_median": 0.2}],
    ids=["plain", "rival"],
)
@pytest.mark.parametrize(("stage", "rate"), [("ctr", 0.02), ("cvr", 0.05)])
def test_rate_sets_each_campaign_mean(stage, rate, settings):
    market = LogisticMarket(**{stage: rate, f"{stage}_spread": 0.0}, **settings)
    draws = logistic_draws(market, 5, RATE_STEPS)
    assert_campaign_means_near(stacked(draws, f"p{stage}"), rate)


def test_rate_spread_across_campaigns():
    draws = logistic_draws(LogisticMarket(ctr=0.02, cvr=0.05), 200, RATE_STEPS)
    for name, rate in (("pctr", 0.02), ("pcvr", 0.05)):
        campaign_logits = logit(stacked(draws, name).mean(axis=0))
        assert abs(np.median(campaign_logits) - logit(rate)) < 0.15
        assert abs(campaign_logits.std() - 0.5) < 0.1


def test_rate_solve_matches_integral():
    targets = np.array([1e-6, 1e-3, 0.02, 0.05, 0.5, 0.9, 0.999])
    for scale, tolerance in [
        (0.0, 1e-7),
        (0.5, 1e-7),
        (1.0, 1e-7),
        (4.0, 1e-7),
        (6.0, 1e-5),
    ]:
        biases = _solve_biases(targets, np.full(len(targets), scale))
        for bias, target in zip(biases, targets, strict=True):

            def density(z, bias=bias, scale=scale):
                return expit(bias + scale * z) * stats.norm.pdf(z)

            mean, _ = integrate.quad(
                density, -np.inf, np.inf, epsabs=0, epsrel=1e-13, limit=200
            )
            assert abs(mean - target) / target < tolerance, (scale, target)


@pytest.mark.parametrize("stage", ["ctr", "cvr"])
def test_rate_and_bias_together_rejected(stage):
    with pytest.raises(ValueError, match=f"{stage}.*{stage}_bias_mean"):
        LogisticMarket(**{stage: 0.05, f"{stage}_bias_mean": -3.0})  # pyright: ignore[reportArgumentType]  Reason: pyright cannot match keys built from a string.


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.1, 1.5, float("nan")])
@pytest.mark.parametrize("stage", ["ctr", "cvr"])
def test_bad_rate_rejected(stage, rate):
    with pytest.raises(ValueError, match=rf"{stage} must be in \(0, 1\)"):
        LogisticMarket(**{stage: rate})


def test_ctr_rate_turns_on_click_stage():
    draw = next(LogisticMarket(ctr=0.02).draws(2, np.random.default_rng(0)))
    assert draw.pctr is not None
    assert draw.pctr.shape == (2,)


@pytest.mark.parametrize(("fewer", "more"), [(1, 2), (3, 4), (2, 7)])
def test_adding_campaign_keeps_rate_draws(fewer, more):
    market = LogisticMarket(ctr=0.02, cvr=0.05, rival_similarity=0.5)
    small = logistic_draws(market, fewer, steps=300)
    large = logistic_draws(market, more, steps=300)
    for name in ("pctr", "pcvr"):
        np.testing.assert_array_equal(
            stacked(small, name), stacked(large, name)[:, :fewer]
        )


def test_rate_mode_keeps_other_draws():
    priced: dict[str, Any] = {"price_median": 0.2, "floor_share": (2, 5)}
    rate = logistic_draws(LogisticMarket(ctr=0.02, **priced), 3, steps=300)
    bias = logistic_draws(LogisticMarket(ctr_bias_mean=-3.0, **priced), 3, steps=300)
    np.testing.assert_array_equal(vectors(rate), vectors(bias))
    for name in ("pcvr", "market_price", "floor"):
        np.testing.assert_array_equal(stacked(rate, name), stacked(bias, name))
    assert not np.array_equal(stacked(rate, "pctr"), stacked(bias, "pctr"))


def segment_rows(draws, name, segment):
    in_segment = np.array([draw.segment == segment for draw in draws])
    return stacked(draws, name)[in_segment]


def rate_segments(spread, steps):
    base = LogisticMarket(ctr=0.02, cvr=0.05, ctr_spread=spread, cvr_spread=spread)
    segments = (Segment("a", 0.5, ctr=0.05, cvr=0.10), Segment("b", 0.5))
    return logistic_draws(Segmented(segments, base), 5, steps)


def test_segment_rates_replace_market_rate():
    draws = rate_segments(spread=0.0, steps=2 * RATE_STEPS)
    targets = {("a", "pctr"): 0.05, ("a", "pcvr"): 0.10, ("b", "pctr"): 0.02}
    targets[("b", "pcvr")] = 0.05
    for (segment, name), target in targets.items():
        assert_campaign_means_near(segment_rows(draws, name, segment), target)


def test_segment_rate_keeps_campaign_offsets():
    draws = rate_segments(spread=0.5, steps=3 * RATE_STEPS)
    for name, rate_a, rate_b in (("pctr", 0.05, 0.02), ("pcvr", 0.10, 0.05)):
        mean_a = segment_rows(draws, name, "a").mean(axis=0)
        mean_b = segment_rows(draws, name, "b").mean(axis=0)
        gap = logit(mean_a) - logit(mean_b)
        np.testing.assert_allclose(gap, logit(rate_a) - logit(rate_b), atol=0.05)


def test_segment_rate_on_bias_base():
    base = LogisticMarket(ctr_bias_mean=-3.0, ctr_spread=0.0)
    market = Segmented((Segment("a", 1.0, ctr=0.05),), base)
    draws = logistic_draws(market, 5, RATE_STEPS)
    assert_campaign_means_near(stacked(draws, "pctr"), 0.05)


def test_ctr_lift_shifts_click_logit():
    base = LogisticMarket(ctr_bias_mean=-3.0, price_median=0.2, floor_share=(2, 5))
    flat = Segmented((Segment("a", 0.5), Segment("b", 0.5)), base)
    lifted = Segmented((Segment("a", 0.5, ctr_lift=1.0), Segment("b", 0.5)), base)
    flat_draws = logistic_draws(flat, 3, steps=500)
    lifted_draws = logistic_draws(lifted, 3, steps=500)
    shift = logit(stacked(lifted_draws, "pctr")) - logit(stacked(flat_draws, "pctr"))
    in_lifted = np.array([draw.segment == "a" for draw in lifted_draws])
    np.testing.assert_allclose(shift[in_lifted], 1.0, atol=1e-9)
    assert np.all(shift[~in_lifted] == 0.0)
    np.testing.assert_array_equal(vectors(flat_draws), vectors(lifted_draws))
    for name in ("pcvr", "market_price", "floor"):
        np.testing.assert_array_equal(
            stacked(flat_draws, name), stacked(lifted_draws, name)
        )


@pytest.mark.parametrize("rate", [0.0, 1.0, float("nan")])
@pytest.mark.parametrize("stage", ["ctr", "cvr"])
def test_bad_segment_rate_rejected(stage, rate):
    with pytest.raises(ValueError, match=rf"{stage} must be in \(0, 1\)"):
        Segment("a", 1.0, **{stage: rate})


@pytest.mark.parametrize("stage", ["ctr", "cvr"])
def test_segment_rate_and_lift_together_rejected(stage):
    with pytest.raises(ValueError, match=f"{stage}.*{stage}_lift"):
        Segment("a", 1.0, **{stage: 0.05, f"{stage}_lift": 1.0})


@pytest.mark.parametrize("settings", [{"ctr": 0.05}, {"ctr_lift": 1.0}])
def test_segment_click_setting_needs_click_stage(settings):
    with pytest.raises(ValueError, match="click stage"):
        Segmented((Segment("a", 1.0, **settings),))
    Segmented((Segment("a", 1.0, **settings),), LogisticMarket(ctr=0.02))
