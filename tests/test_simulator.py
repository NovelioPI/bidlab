import dataclasses
from typing import ClassVar

import numpy as np

from auction_sim.bidders import Bidder
from auction_sim.config import AuctionConfig, BidderSpec, SimConfig
from auction_sim.simulator import run
from auction_sim.types import EpisodeStats, Feedback, Observation


class Spy(Bidder):
    """Bids `multiplier` × value and records every call it receives."""

    instances: ClassVar[list["Spy"]] = []

    def __init__(self, params, rng):
        super().__init__(params, rng)
        self.feedback: list[Feedback] = []
        self.observations: list[Observation] = []
        self.starts: list[tuple[float, int]] = []
        self.ends: list[tuple[EpisodeStats, float | None]] = []
        Spy.instances.append(self)

    def on_episode_start(self, budget, steps):
        self.starts.append((budget, steps))

    def bid(self, obs):
        self.observations.append(obs)
        return self.params.get("multiplier", 1.0) * obs.value

    def update(self, feedback):
        self.feedback.append(feedback)

    def on_episode_end(self, stats, episode_reward):
        self.ends.append((stats, episode_reward))


def spec(name, multiplier=1.0, budget=100.0, **kwargs):
    return BidderSpec(
        name=name,
        type="test_simulator:Spy",
        budget=budget,
        value_per_conversion=kwargs.pop("value_per_conversion", 5.0),
        params={"multiplier": multiplier},
        **kwargs,
    )


def config(*bidders, **kwargs):
    defaults = {"episodes": 3, "steps": 200, "seed": 7}
    return SimConfig(bidders=tuple(bidders), **(defaults | kwargs))


def run_spies(cfg):
    Spy.instances.clear()
    log = run(cfg)
    return log, list(Spy.instances)


def test_log_shapes():
    log = run(config(spec("a"), spec("b"), episodes=2, steps=50))
    assert log.bids.shape == log.values.shape == log.rewards.shape == (2, 50, 2)
    assert log.winner.shape == log.price.shape == log.converted.shape == (2, 50)
    assert log.episode_rewards.shape == (2, 2)


def test_loser_feedback_has_no_price():
    _, spies = run_spies(config(spec("a"), spec("b", multiplier=0.9)))
    losses = [fb for spy in spies for fb in spy.feedback if not fb.won]
    assert losses
    assert all(fb.price == 0.0 and not fb.converted for fb in losses)


def test_feedback_fields_are_only_own_outcome():
    fields = {f.name for f in dataclasses.fields(Feedback)}
    assert fields == {"bid", "won", "price", "converted", "reward"}
    observed = {f.name for f in dataclasses.fields(Observation)}
    assert observed == {
        "step",
        "steps_left",
        "value",
        "value_per_conversion",
        "budget_left",
    }


def test_winner_feedback_matches_log():
    log, spies = run_spies(config(spec("a"), spec("b", multiplier=0.9), episodes=1))
    for i, spy in enumerate(spies):
        for step, fb in enumerate(spy.feedback):
            assert fb.won == (log.winner[0, step] == i)
            if fb.won:
                assert fb.price == log.price[0, step]
                assert fb.converted == log.converted[0, step]


def test_spend_never_exceeds_budget():
    greedy = spec("greedy", multiplier=1e9, budget=2.0)
    log = run(config(greedy, spec("b")))
    for episode in range(log.winner.shape[0]):
        spend = log.price[episode][log.winner[episode] == 0].sum()
        assert spend <= 2.0 + 1e-12
    assert np.all(log.bids[:, :, 0] <= 2.0)


def test_budget_resets_each_episode():
    _, spies = run_spies(config(spec("greedy", multiplier=1e9, budget=2.0), spec("b")))
    greedy = spies[0]
    assert greedy.starts == [(2.0, 200)] * 3
    first_steps = [obs for obs in greedy.observations if obs.step == 0]
    assert [obs.budget_left for obs in first_steps] == [2.0] * 3


def test_same_bidder_object_and_hook_per_episode():
    _, spies = run_spies(config(spec("a"), spec("b")))
    assert len(spies) == 2
    assert all(len(spy.starts) == 3 and len(spy.ends) == 3 for spy in spies)
    assert all(len(spy.feedback) == 3 * 200 for spy in spies)


def test_same_seed_same_log():
    first = run(config(spec("a"), spec("b", multiplier=0.8)))
    second = run(config(spec("a"), spec("b", multiplier=0.8)))
    np.testing.assert_array_equal(first.bids, second.bids)
    np.testing.assert_array_equal(first.winner, second.winner)
    np.testing.assert_array_equal(first.converted, second.converted)
    other = run(config(spec("a"), spec("b", multiplier=0.8), seed=8))
    assert not np.array_equal(first.values, other.values)


def test_adding_bidder_keeps_world_draws():
    two = run(config(spec("a"), spec("b")))
    three = run(config(spec("a"), spec("b"), spec("c")))
    np.testing.assert_array_equal(two.values, three.values[:, :, :2])


def test_feedback_reward_equals_function_output():
    log, spies = run_spies(
        config(spec("a", reward="sample_plugins:double_price"), spec("b"))
    )
    rewards = [fb.reward for fb in spies[0].feedback]
    expected = [2 * fb.price for fb in spies[0].feedback]
    assert rewards == expected
    np.testing.assert_array_equal(log.rewards[:, :, 0].ravel(), expected)


def test_episode_reward_called_once_per_episode():
    log, spies = run_spies(
        config(spec("a", episode_reward="sample_plugins:total_spend"), spec("b"))
    )
    ends = spies[0].ends
    assert len(ends) == 3
    assert all(reward == stats.spend for stats, reward in ends)
    np.testing.assert_array_equal(log.episode_rewards[:, 0], [s.spend for s, _ in ends])


def test_no_episode_reward_passes_none():
    log, spies = run_spies(config(spec("a"), spec("b")))
    assert all(reward is None for _, reward in spies[0].ends)
    assert np.all(np.isnan(log.episode_rewards))


def test_cpa_target_reaches_reward_params():
    _, spies = run_spies(
        config(spec("a", cpa_target=0.5, episode_reward="cpa_score"), spec("b"))
    )
    assert all(reward is not None for _, reward in spies[0].ends)


def test_reserve_above_all_bids_gives_no_sales():
    log = run(config(spec("a"), auction=AuctionConfig(kind="first", reserve=1e6)))
    assert np.all(log.winner == -1)
    assert np.all(log.price == 0)
