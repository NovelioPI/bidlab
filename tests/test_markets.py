from itertools import islice

import numpy as np
import pytest

from bidlab.markets import LogisticContext, Segment, Segmented


def probabilities(market, bidders, steps, seed=0):
    draws = market.draws(bidders, np.random.default_rng(seed))
    return np.array([draw.probabilities for draw in islice(draws, steps)])


def test_logistic_probabilities_inside_unit_interval():
    rates = probabilities(LogisticContext(context_dim=4), bidders=3, steps=50, seed=1)
    assert rates.shape == (50, 3)
    assert np.all((rates > 0) & (rates < 1))


@pytest.mark.parametrize(("fewer", "more"), [(1, 2), (1, 3), (2, 4), (1, 8), (3, 5)])
def test_logistic_adding_bidder_keeps_draws(fewer, more):
    for seed in range(3):
        small = probabilities(LogisticContext(), fewer, steps=500, seed=seed)
        large = probabilities(LogisticContext(), more, steps=500, seed=seed)
        np.testing.assert_array_equal(small, large[:, :fewer])


def test_bias_mean_sets_typical_conversion_rate():
    rates = probabilities(
        LogisticContext(bias_mean=-3.0), bidders=20, steps=2000, seed=2
    )
    # σ(-3) ≈ 0.047; context noise spreads it, so the mean sits near 5-10%.
    assert 0.03 < rates.mean() < 0.12


def segment_draws(market, bidders, steps, seed=0):
    return list(islice(market.draws(bidders, np.random.default_rng(seed)), steps))


def test_segment_shares_and_feature():
    market = Segmented((Segment("news", 0.7), Segment("sports", 0.3)))
    draws = segment_draws(market, bidders=2, steps=20_000)
    names = [draw.features["segment"] for draw in draws]
    assert set(names) == {"news", "sports"}
    assert [draw.segment for draw in draws] == names
    assert abs(names.count("news") / len(names) - 0.7) < 0.02


def test_lift_shifts_logit():
    lifted = Segmented((Segment("plain", 0.5), Segment("lifted", 0.5, lift=2.0)))
    flat = Segmented((Segment("plain", 0.5), Segment("lifted", 0.5)))
    lifted_draws = segment_draws(lifted, bidders=3, steps=4000, seed=5)
    flat_draws = segment_draws(flat, bidders=3, steps=4000, seed=5)
    in_lifted = np.array([draw.segment == "lifted" for draw in flat_draws])
    lifted_rates = np.array([draw.probabilities for draw in lifted_draws])
    base_rates = np.array([draw.probabilities for draw in flat_draws])
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
        ((), "at least one segment"),
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
        small = segment_draws(market, bidders=fewer, steps=500, seed=seed)
        large = segment_draws(market, bidders=more, steps=500, seed=seed)
        assert [d.segment for d in small] == [d.segment for d in large]
        np.testing.assert_array_equal(
            np.array([d.probabilities for d in small]),
            np.array([d.probabilities for d in large])[:, :fewer],
        )
