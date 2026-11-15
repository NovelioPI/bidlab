import numpy as np

from auction_sim.world import World, WorldConfig


def test_value_equals_w_times_sigmoid():
    weights = np.array([1.0, 2.5, 10.0])
    world = World(WorldConfig(context_dim=4), weights, np.random.default_rng(1))
    for _ in range(50):
        probabilities, values = world.step()
        assert np.all((probabilities > 0) & (probabilities < 1))
        np.testing.assert_allclose(values, weights * probabilities)


def test_bias_mean_sets_typical_conversion_rate():
    world = World(WorldConfig(bias_mean=-3.0), np.ones(20), np.random.default_rng(2))
    rates = np.array([world.step()[0] for _ in range(2000)])
    # Reason: σ(-3) ≈ 0.047; context noise spreads it, so the mean sits near 5-10%.
    assert 0.03 < rates.mean() < 0.12
