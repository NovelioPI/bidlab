import dataclasses
from typing import Any, ClassVar

import numpy as np
import pytest

from bidlab import rewards
from bidlab.auctions import NO_WINNER, FirstPrice
from bidlab.bidders import Bidder
from bidlab.config import BidderSpec, SimConfig
from bidlab.markets import Market, MarketDraw, Segment, Segmented
from bidlab.simulator import run
from bidlab.types import Campaign, EpisodeStats, Feedback, Observation


class Spy(Bidder):
    """Bids `multiplier` × value and records every call it receives."""

    instances: ClassVar[list["Spy"]] = []

    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier
        self.feedback: list[Feedback] = []
        self.observations: list[Observation] = []
        self.starts: list[Campaign] = []
        self.ends: list[tuple[EpisodeStats, float | None]] = []

    def bind(self, rng):
        # Each run binds a fresh copy, so this lists the copies the run used.
        super().bind(rng)
        Spy.instances.append(self)

    def on_episode_start(self, campaign):
        self.starts.append(campaign)

    def forward(self, value, observation):
        self.observations.append(observation)
        return self.multiplier * value

    def update(self, feedback):
        self.feedback.append(feedback)

    def on_episode_end(self, stats, episode_reward):
        self.ends.append((stats, episode_reward))


def double_price(observation, feedback, params):
    return 2 * feedback.price


def total_spend(stats, params):
    return stats.spend


class BatchSpy(Spy):
    """A Spy that overrides update_batch and records each batch."""

    def __init__(self, multiplier=1.0):
        super().__init__(multiplier)
        self.batches: list[list[Feedback]] = []

    def update_batch(self, feedbacks):
        self.batches.append(list(feedbacks))


CAMPAIGN_FIELDS = ("value", "cpa_target", "segments")


def spec(name, multiplier=1.0, budget=100.0, bidder=Spy, **kwargs):
    campaign = {key: kwargs.pop(key) for key in CAMPAIGN_FIELDS if key in kwargs}
    return BidderSpec(
        name=name,
        bidder=bidder(multiplier),
        campaign=Campaign(budget, campaign.pop("value", 5.0), **campaign),
        **kwargs,
    )


def config(*bidders, **kwargs):
    defaults: dict[str, Any] = {"episodes": 3, "steps": 200, "seed": 7}
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
    fields = [f.name for f in dataclasses.fields(Feedback)]
    assert fields == ["time", "bid", "won", "price", "converted", "value", "reward"]


def test_observation_fields():
    assert [f.name for f in dataclasses.fields(Observation)] == [
        "time",
        "value",
        "features",
    ]


def test_episode_start_gets_campaign_with_schedule():
    a = spec("a", cpa_target=2.0)
    _, spies = run_spies(config(a, spec("b"), horizon_hours=12, start_hour=6))
    expected = dataclasses.replace(a.campaign, horizon_hours=12.0, start_hour=6.0)
    assert spies[0].starts == [expected] * 3


def test_observation_value_is_true_value_without_noise():
    log, spies = run_spies(config(spec("a"), spec("b", value=3.0), episodes=2))
    for i, spy in enumerate(spies):
        seen = np.array([o.value for o in spy.observations]).reshape(2, 200)
        assert seen.tobytes() == log.values[:, :, i].tobytes()


def test_observation_time_spaced_over_horizon():
    _, spies = run_spies(config(spec("a"), episodes=1, steps=1000, horizon_hours=24))
    times = np.array([o.time for o in spies[0].observations])
    assert times[0] == 0.0
    np.testing.assert_allclose(times, np.arange(1000) * 86.4, rtol=0, atol=1e-9)


def test_feedback_time_matches_observation():
    _, spies = run_spies(config(spec("a"), spec("b", multiplier=0.9)))
    for spy in spies:
        assert [f.time for f in spy.feedback] == [o.time for o in spy.observations]


def test_feedback_value_is_realized():
    log, spies = run_spies(config(spec("a", value=3.0), spec("b", multiplier=0.9)))
    for i, (spy, value) in enumerate(zip(spies, (3.0, 5.0), strict=True)):
        realized = np.where(log.converted & (log.winner == i), value, 0.0).ravel()
        assert [f.value for f in spy.feedback] == list(realized)
        assert any(f.value == value for f in spy.feedback)


def test_update_batch_gets_one_feedback_per_auction():
    _, spies = run_spies(config(spec("a", bidder=BatchSpy), spec("b")))
    batching = spies[0]
    assert isinstance(batching, BatchSpy)
    assert len(batching.batches) == 3 * 200
    assert all(len(batch) == 1 for batch in batching.batches)
    times = [batch[0].time for batch in batching.batches]
    per_episode = np.array(times).reshape(3, 200)
    assert np.all(np.diff(per_episode, axis=1) > 0)
    assert batching.feedback == []


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
    log, spies = run_spies(
        config(spec("greedy", multiplier=1e9, budget=2.0), spec("b"))
    )
    greedy = spies[0]
    assert [campaign.budget for campaign in greedy.starts] == [2.0] * 3
    # A huge bid is capped at the budget left, so each first bid shows a full budget.
    np.testing.assert_array_equal(log.bids[:, 0, 0], [2.0] * 3)


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
    log, spies = run_spies(config(spec("a", reward=double_price), spec("b")))
    rewards = [fb.reward for fb in spies[0].feedback]
    expected = [2 * fb.price for fb in spies[0].feedback]
    assert rewards == expected
    np.testing.assert_array_equal(log.rewards[:, :, 0].ravel(), expected)


def test_episode_reward_called_once_per_episode():
    log, spies = run_spies(config(spec("a", episode_reward=total_spend), spec("b")))
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
        config(spec("a", cpa_target=0.5, episode_reward=rewards.cpa_score), spec("b"))
    )
    assert all(reward is not None for _, reward in spies[0].ends)


def test_reserve_above_all_bids_gives_no_sales():
    log = run(config(spec("a"), auction=FirstPrice(reserve=1e6)))
    assert np.all(log.winner == -1)
    assert np.all(log.price == 0)


class FixedMarket(Market):
    """Yields the given probabilities and features at every step."""

    def __init__(self, probabilities, features=None):
        self.probabilities = probabilities
        self.features = features or {}

    def draws(self, bidders, rng):
        while True:
            yield MarketDraw(self.features, np.array(self.probabilities))


class CountingMarket(Market):
    """Step k gives bidder i probability (k mod 10 + i) / 20."""

    def draws(self, bidders, rng):
        step = 0
        while True:
            yield MarketDraw({}, (step % 10 + np.arange(bidders)) / 20)
            step += 1


def test_custom_market_values_are_weight_times_probability():
    cfg = config(
        spec("a", value=2.0),
        spec("b", value=4.0),
        market=CountingMarket(),
        steps=30,
    )
    log = run(cfg)
    step = np.arange(cfg.episodes * cfg.steps).reshape(cfg.episodes, cfg.steps)
    expected = (step[..., None] % 10 + np.arange(2)) / 20 * np.array([2.0, 4.0])
    np.testing.assert_array_equal(log.values, expected)


@pytest.mark.parametrize(
    "probabilities",
    [[0.5], [0.5, 1.2], [0.5, -0.1], [0.5, np.nan]],
    ids=["wrong-length", "above-one", "below-zero", "nan"],
)
def test_bad_market_draw_rejected(probabilities):
    cfg = config(spec("a"), spec("b"), market=FixedMarket(probabilities))
    with pytest.raises(ValueError, match="FixedMarket at episode 0, step 0"):
        run(cfg)


def test_market_that_stops_names_the_market():
    class Short(Market):
        def draws(self, bidders, rng):
            yield MarketDraw({}, np.full(bidders, 0.5))

    with pytest.raises(RuntimeError, match="Short at episode 0, step 1"):
        run(config(spec("a"), market=Short()))


def test_features_reach_every_bidder_read_only():
    market = FixedMarket([0.5, 0.5], features={"hour": 3})
    _, spies = run_spies(config(spec("a"), spec("b"), market=market, steps=5))
    for spy in spies:
        for observation in spy.observations:
            assert observation.features["hour"] == 3
            with pytest.raises(TypeError):
                observation.features["hour"] = 4  # pyright: ignore[reportIndexIssue]  Reason: the test checks that features are read-only.


def test_no_features_gives_empty_mapping():
    _, spies = run_spies(config(spec("a"), market=FixedMarket([0.5]), steps=5))
    for observation in spies[0].observations:
        assert dict(observation.features) == {}
        with pytest.raises(TypeError):
            observation.features["hour"] = 4  # pyright: ignore[reportIndexIssue]  Reason: the test checks that features are read-only.


NEWS_AND_SPORTS = Segmented((Segment("news", 0.5), Segment("sports", 0.5)))


def test_targeted_bidder_skips_other_segments():
    cfg = config(
        spec("news_only", segments=("news",)),
        spec("everywhere"),
        spec("news_batches", segments=("news",), bidder=BatchSpy),
        market=NEWS_AND_SPORTS,
    )
    log, (news_only, _, news_batches) = run_spies(cfg)
    sports = log.segment == NEWS_AND_SPORTS.segment_names.index("sports")
    assert sports.any() and (~sports).any()
    assert np.all(log.bids[sports][:, 0] == 0.0)
    assert not np.any(log.winner[sports] == 0)
    assert len(news_only.observations) == np.sum(~sports)
    assert len(news_only.feedback) == np.sum(~sports)
    assert isinstance(news_batches, BatchSpy)
    assert len(news_batches.batches) == np.sum(~sports)
    assert {o.features["segment"] for o in news_only.observations} == {"news"}


def test_lone_targeted_bidder_never_wins_outside_its_segments():
    cfg = config(spec("news_only", segments=("news",)), market=NEWS_AND_SPORTS)
    log = run(cfg)
    sports = log.segment == NEWS_AND_SPORTS.segment_names.index("sports")
    assert np.all(log.winner[sports] == NO_WINNER)
    assert np.all(log.price[sports] == 0.0)


def test_untargeted_bidder_takes_every_step():
    cfg = config(spec("a"), spec("b", segments=("sports",)), market=NEWS_AND_SPORTS)
    _, (everywhere, _) = run_spies(cfg)
    assert len(everywhere.observations) == cfg.episodes * cfg.steps
    assert len(everywhere.feedback) == cfg.episodes * cfg.steps


class DayAndNight(Market):
    """First 6 of every 10 steps are "day", the rest "night"."""

    @property
    def segment_names(self):
        return ("day", "night")

    def draws(self, bidders, rng):
        step = 0
        while True:
            name = "day" if step % 10 < 6 else "night"
            yield MarketDraw({"segment": name}, np.full(bidders, 0.1), name)
            step += 1


def test_custom_market_segments_drive_targeting_and_log():
    cfg = config(
        spec("night_only", segments=("night",)), spec("a"), market=DayAndNight()
    )
    log, (night_only, _) = run_spies(cfg)
    step = np.arange(cfg.episodes * cfg.steps).reshape(cfg.episodes, cfg.steps)
    np.testing.assert_array_equal(log.segment, (step % 10 >= 6).astype(int))
    assert len(night_only.observations) == np.sum(step % 10 >= 6)
    assert not np.any(log.winner[log.segment == 0] == 0)


def test_estimate_noise_has_set_spread():
    log, (spy,) = run_spies(
        config(spec("a"), episodes=1, steps=20_000, estimate_noise=0.5)
    )
    estimated = np.array([o.value for o in spy.observations]) / 5.0
    true = log.values[0, :, 0] / 5.0
    errors = np.log(estimated / (1 - estimated)) - np.log(true / (1 - true))
    assert abs(errors.mean()) < 0.02
    assert abs(errors.std() - 0.5) < 0.02


def test_estimate_noise_keeps_true_values():
    noisy = run(config(spec("a"), spec("b"), estimate_noise=0.5))
    exact = run(config(spec("a"), spec("b")))
    assert noisy.values.tobytes() == exact.values.tobytes()


def test_adding_bidder_keeps_noisy_values():
    _, two = run_spies(config(spec("a"), spec("b"), estimate_noise=0.5))
    _, three = run_spies(config(spec("a"), spec("b"), spec("c"), estimate_noise=0.5))
    for before, after in zip(two, three[:2], strict=True):
        assert [o.value for o in before.observations] == [
            o.value for o in after.observations
        ]


def test_estimate_noise_at_zero_and_one():
    market = FixedMarket([0.0, 1.0])
    _, spies = run_spies(
        config(spec("a"), spec("b"), market=market, estimate_noise=2.0)
    )
    assert {o.value for o in spies[0].observations} == {0.0}
    assert {o.value for o in spies[1].observations} == {5.0}
