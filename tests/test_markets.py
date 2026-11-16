from itertools import islice

import numpy as np

from bidlab.markets import LogisticContext


def probabilities(market, bidders, steps, seed=0):
    draws = market.draws(bidders, np.random.default_rng(seed))
    return np.array([draw.probabilities for draw in islice(draws, steps)])


def test_logistic_probabilities_inside_unit_interval():
    rates = probabilities(LogisticContext(context_dim=4), bidders=3, steps=50, seed=1)
    assert rates.shape == (50, 3)
    assert np.all((rates > 0) & (rates < 1))


def test_bias_mean_sets_typical_conversion_rate():
    rates = probabilities(
        LogisticContext(bias_mean=-3.0), bidders=20, steps=2000, seed=2
    )
    # σ(-3) ≈ 0.047; context noise spreads it, so the mean sits near 5-10%.
    assert 0.03 < rates.mean() < 0.12
