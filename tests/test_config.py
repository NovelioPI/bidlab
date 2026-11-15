import numpy as np
import pytest

from bidlab import rewards
from bidlab.auctions import FirstPrice, SecondPrice
from bidlab.bidders import Bidder, PidPacing, Truthful
from bidlab.config import BidderSpec, SimConfig
from bidlab.rewards import cpa_score_of
from bidlab.simulator import run


class Recorder(Bidder):
    """Truthful bidder that records what it receives."""

    def __init__(self, params, rng):
        super().__init__(params, rng)
        self.rewards = []
        self.episode_rewards = []

    def bid(self, observation):
        return observation.value

    def update(self, feedback):
        self.rewards.append(feedback.reward)

    def on_episode_end(self, stats, episode_reward):
        self.episode_rewards.append((stats, episode_reward))


def other():
    return BidderSpec("other", Truthful, 50.0, 4.0)


def small_config(*bidders, **kwargs):
    return SimConfig(bidders=[*bidders, other()], episodes=2, steps=100, **kwargs)


def recording(bidder_class):
    """Return a factory for bidder_class and the list it fills with built bidders."""
    built = []

    def factory(params, rng):
        bidder = bidder_class(params, rng)
        built.append(bidder)
        return bidder

    return factory, built


def test_bidder_class_with_empty_params():
    factory, built = recording(PidPacing)
    run(small_config(BidderSpec("pid", factory, budget=15, value_per_conversion=5)))
    assert isinstance(built[0], PidPacing)
    assert built[0].params == {}


def test_factory_called_once_with_params_and_own_rng():
    calls = []

    def factory(params, rng):
        calls.append((params, rng))
        return Truthful(params, rng)

    spec = BidderSpec("f", factory, 50, 5, params={"multiplier": 0.5})
    run(small_config(spec))
    assert len(calls) == 1
    params, rng = calls[0]
    assert params == {"multiplier": 0.5}
    assert isinstance(rng, np.random.Generator)


def test_factory_gives_each_bidder_its_own_rng():
    rngs = []

    def factory(params, rng):
        rngs.append(rng)
        return Truthful(params, rng)

    run(small_config(BidderSpec("a", factory, 50, 5), BidderSpec("b", factory, 50, 5)))
    assert rngs[0] is not rngs[1]
    assert rngs[0].random() != rngs[1].random()


def test_factory_returning_non_bidder_fails_before_step_zero():
    steps_run = []

    class Watch(Truthful):
        def bid(self, observation):
            steps_run.append(observation.step)
            return observation.value

    spec = BidderSpec("bad", lambda params, rng: object(), 50, 5)
    config = SimConfig(
        bidders=[BidderSpec("watch", Watch, 50, 5), spec], episodes=1, steps=10
    )
    with pytest.raises(TypeError, match="'bad'.*not a Bidder"):
        run(config)

    assert steps_run == []


def test_string_bidder_rejected():
    with pytest.raises(TypeError, match="Bidder subclass or a factory"):
        BidderSpec("pid", "pid_pacing", 15, 5)


@pytest.mark.parametrize(
    "kwargs",
    [{"reward": "surplus"}, {"episode_reward": "cpa_score"}, {"bidder": 3}],
)
def test_non_callable_fields_rejected(kwargs):
    fields = {"name": "x", "bidder": Truthful, "budget": 1, "value_per_conversion": 1}
    with pytest.raises(TypeError, match="callable"):
        BidderSpec(**(fields | kwargs))


def test_reward_function_reaches_feedback():
    factory, built = recording(Recorder)
    spec = BidderSpec("r", factory, 50, 5, reward=rewards.conversions)
    log = run(small_config(spec))
    recorder = built[0]
    expected = (log.converted & (log.winner == 0)).astype(float).ravel()
    assert recorder.rewards == list(expected)

    def custom(observation, feedback, params):
        return 7.0

    built.clear()
    run(small_config(BidderSpec("r", factory, 50, 5, reward=custom)))
    assert set(built[0].rewards) == {7.0}


def test_cpa_score_episode_reward_uses_target():
    factory, built = recording(Recorder)
    spec = BidderSpec(
        "r", factory, 50, 5, cpa_target=3, episode_reward=rewards.cpa_score
    )
    run(small_config(spec))
    for stats, episode_reward in built[0].episode_rewards:
        expected = cpa_score_of(stats.value, stats.spend, stats.conversions, 3.0)
        assert episode_reward == expected


def test_numbers_stored_as_float():
    spec = BidderSpec("a", Truthful, 3, 5, cpa_target=2)
    assert all(
        type(value) is float
        for value in (spec.budget, spec.value_per_conversion, spec.cpa_target)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("budget", 0),
        ("budget", -1),
        ("value_per_conversion", 0),
        ("cpa_target", 0),
        ("cpa_target", -2),
    ],
)
def test_bidder_spec_rejects_non_positive_numbers(field, value):
    fields = {"name": "x", "bidder": Truthful, "budget": 1, "value_per_conversion": 1}
    with pytest.raises(ValueError, match=f"'x'.*{field}"):
        BidderSpec(**(fields | {field: value}))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bidders": []}, "at least one"),
        ({"bidders": [other(), other()]}, "unique"),
        ({"bidders": [other()], "episodes": 0}, "episodes"),
        ({"bidders": [other()], "steps": 0}, "steps"),
    ],
)
def test_sim_config_rejects_bad_bidders_and_lengths(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SimConfig(**kwargs)


def test_sim_config_rejects_unknown_auction():
    with pytest.raises(TypeError, match="FirstPrice or SecondPrice"):
        SimConfig(bidders=[other()], auction="second")


def test_bidders_list_stored_as_tuple():
    specs = [other(), BidderSpec("b", Truthful, 1, 1)]
    config = SimConfig(bidders=specs)
    assert config.bidders == tuple(specs)


def test_default_auction_is_second_price_no_reserve():
    assert SimConfig(bidders=[other()]).auction == SecondPrice(reserve=0.0)


def test_first_price_config_runs():
    log = run(small_config(auction=FirstPrice(0.01)))
    sold = log.winner >= 0
    np.testing.assert_array_equal(
        log.price[sold], log.bids[sold.nonzero() + (log.winner[sold],)]
    )


def test_bidder_without_bid_fails_on_creation():
    class Forgot(Bidder):
        def bids(self, observation):
            return observation.value

    with pytest.raises(TypeError, match="abstract method 'bid'"):
        Forgot({}, np.random.default_rng(0))
    with pytest.raises(TypeError, match="abstract method 'bid'"):
        run(small_config(BidderSpec("forgot", Forgot, 50, 5)))
