import threading
from typing import Any, ClassVar

import numpy as np
import pytest

from bidlab import rewards
from bidlab.auctions import FirstPrice, SecondPrice
from bidlab.bidders import Bidder, Truthful
from bidlab.config import BidderSpec, SimConfig
from bidlab.markets import LogisticContext, Segment, Segmented
from bidlab.rewards import cpa_score_of
from bidlab.simulator import run
from bidlab.types import Campaign, Feedback, Observation


class Recorder(Bidder):
    """Truthful bidder that records what it receives."""

    built: ClassVar[list["Recorder"]] = []

    def __init__(self):
        self.rewards = []
        self.episode_rewards = []

    def bind(self, rng):
        # Each run binds a fresh copy, so this lists the copies the runs used.
        super().bind(rng)
        Recorder.built.append(self)

    def forward(self, value, observation):
        return value

    def update(self, feedback):
        self.rewards.append(feedback.reward)

    def on_episode_end(self, stats, episode_reward):
        self.episode_rewards.append((stats, episode_reward))


def other():
    return BidderSpec("other", Truthful(), Campaign(50.0, 4.0))


def small_config(*bidders, **kwargs):
    return SimConfig(bidders=[*bidders, other()], episodes=2, steps=100, **kwargs)


def built_recorders(*specs, **kwargs):
    """Run the specs with `other`, and return the log and the recorders each run used."""
    Recorder.built.clear()
    log = run(small_config(*specs, **kwargs))
    return log, list(Recorder.built)


def test_bidder_copied_and_bound_once_per_run():
    template = Recorder()
    _, built = built_recorders(BidderSpec("r", template, Campaign(15, 5)))
    assert len(built) == 1
    assert built[0] is not template
    assert isinstance(built[0].rng, np.random.Generator)
    assert not hasattr(template, "rng")
    assert template.rewards == []


def test_each_bidder_gets_its_own_rng():
    _, built = built_recorders(
        BidderSpec("a", Recorder(), Campaign(50, 5)),
        BidderSpec("b", Recorder(), Campaign(50, 5)),
    )
    assert built[0].rng is not built[1].rng
    assert built[0].rng.random() != built[1].rng.random()


def test_uncopyable_bidder_fails_before_step_zero():
    steps_run = []

    class Watch(Recorder):
        def forward(self, value, observation):
            steps_run.append(observation.time)
            return value

    class Locked(Recorder):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()

    config = SimConfig(
        bidders=[
            BidderSpec("watch", Watch(), Campaign(50, 5)),
            BidderSpec("bad", Locked(), Campaign(50, 5)),
        ],
        episodes=1,
        steps=10,
    )
    with pytest.raises(TypeError, match="'bad' cannot be copied"):
        run(config)

    assert steps_run == []


def test_non_bidder_rejected():
    with pytest.raises(TypeError, match="'bad'.*Bidder instance"):
        BidderSpec("bad", object(), Campaign(50, 5))  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_string_bidder_rejected():
    with pytest.raises(TypeError, match="Bidder instance"):
        BidderSpec("pid", "pid_pacing", Campaign(15, 5))  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


@pytest.mark.parametrize(
    "kwargs",
    [{"reward": "surplus"}, {"episode_reward": "cpa_score"}, {"bidder": 3}],
)
def test_non_callable_fields_rejected(kwargs):
    fields = {"name": "x", "bidder": Truthful(), "campaign": Campaign(1, 1)}
    with pytest.raises(TypeError, match="callable|Bidder instance"):
        BidderSpec(**(fields | kwargs))


def test_reward_function_reaches_feedback():
    spec = BidderSpec("r", Recorder(), Campaign(50, 5), reward=rewards.conversions)
    log, built = built_recorders(spec)
    recorder = built[0]
    expected = (log.converted & (log.winner == 0)).astype(float).ravel()
    assert recorder.rewards == list(expected)

    def custom(observation, feedback, params):
        return 7.0

    _, built = built_recorders(
        BidderSpec("r", Recorder(), Campaign(50, 5), reward=custom)
    )
    assert set(built[0].rewards) == {7.0}

    def realized(observation, feedback, params):
        return feedback.value

    log, built = built_recorders(
        BidderSpec("r", Recorder(), Campaign(50, 5), reward=realized)
    )
    expected = np.where(log.converted & (log.winner == 0), 5.0, 0.0).ravel()
    assert built[0].rewards == list(expected)
    np.testing.assert_array_equal(log.rewards[:, :, 0].ravel(), expected)


def test_cpa_score_episode_reward_uses_target():
    spec = BidderSpec(
        "r", Recorder(), Campaign(50, 5, cpa_target=3), episode_reward=rewards.cpa_score
    )
    _, built = built_recorders(spec)
    for stats, episode_reward in built[0].episode_rewards:
        expected = cpa_score_of(stats.value, stats.spend, stats.conversions, 3.0)
        assert episode_reward == expected


def test_numbers_stored_as_float():
    campaign = Campaign(3, 5, cpa_target=2)
    assert all(
        type(value) is float
        for value in (campaign.budget, campaign.value, campaign.cpa_target)
    )


def test_bidder_spec_holds_campaign():
    campaign = Campaign(10, 5)
    bidder = Recorder()
    spec = BidderSpec("a", bidder, campaign)
    assert spec.bidder is bidder
    assert spec.campaign is campaign
    assert spec.reward is rewards.surplus
    assert spec.episode_reward is None
    assert spec.reward_params == {}


@pytest.mark.parametrize(
    "old_field", ["budget", "value_per_conversion", "cpa_target", "segments", "params"]
)
def test_bidder_spec_rejects_old_fields_and_non_campaign(old_field):
    with pytest.raises(TypeError, match=old_field):
        BidderSpec("a", Truthful(), Campaign(10, 5), **{old_field: 1})  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a removed field.
    with pytest.raises(TypeError, match="Campaign"):
        BidderSpec("a", Truthful(), {"budget": 10, "value": 5})  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


@pytest.mark.parametrize("field_name", ["horizon_hours", "start_hour"])
def test_campaign_schedule_must_match_sim_config(field_name):
    schedule: dict[str, Any] = {field_name: 12}
    own = Campaign(10, 5, **schedule)
    spec = BidderSpec("own", Truthful(), own)
    with pytest.raises(ValueError, match=f"bidder 'own'.*{field_name}"):
        SimConfig(bidders=[spec])
    matching = SimConfig(bidders=[spec], **schedule)
    assert getattr(matching, field_name) == 12.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_hours": 0},
        {"horizon_hours": -1},
        {"start_hour": -1},
        {"start_hour": 24},
    ],
)
def test_sim_config_rejects_bad_schedule_and_noise(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        SimConfig(bidders=[other()], **kwargs)


def test_builtin_rewards_read_feedback_value():
    observation = Observation(time=0.0, value=0.1)
    won = Feedback(time=0.0, bid=0.6, won=True, price=0.5, converted=True, value=5.0)
    lost = Feedback(time=0.0, bid=0.6, won=False, price=0.0, converted=False)
    assert rewards.surplus(observation, won, {}) == 5.0 - 0.5
    assert rewards.value(observation, won, {}) == 5.0
    assert rewards.surplus(observation, lost, {}) == 0.0
    assert rewards.value(observation, lost, {}) == 0.0


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
        SimConfig(bidders=[other()], auction="second")  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_default_market_is_logistic_context():
    assert SimConfig(bidders=[other()]).market == LogisticContext()


def test_world_argument_rejected():
    with pytest.raises(TypeError, match="world"):
        SimConfig(bidders=[other()], world=LogisticContext())  # pyright: ignore[reportCallIssue]  Reason: the test checks the runtime error for a wrong type.


def test_non_market_rejected():
    with pytest.raises(TypeError, match="Market"):
        SimConfig(bidders=[other()], market="logistic")  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_bidders_list_stored_as_tuple():
    specs = [other(), BidderSpec("b", Truthful(), Campaign(1, 1))]
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


def test_bidder_without_forward_fails_on_creation():
    class Forgot(Bidder):
        def forwards(self, value, observation):
            return value

    with pytest.raises(TypeError, match="abstract method 'forward'"):
        Forgot()  # pyright: ignore[reportAbstractUsage]  Reason: the test checks the error for a missing forward method.
    with pytest.raises(TypeError, match=r"pass an instance, such as Forgot\(\)"):
        BidderSpec("forgot", Forgot, Campaign(50, 5))  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_unknown_segment_rejected():
    market = Segmented((Segment("news", 0.5), Segment("sports", 0.5)))
    spec = BidderSpec("b", Truthful(), Campaign(1, 1, segments=("news", "games")))
    with pytest.raises(ValueError, match=r"bidder 'b'.*'games'"):
        SimConfig(bidders=[spec], market=market)
    with pytest.raises(ValueError, match=r"bidder 'b'.*'news'"):
        SimConfig(
            bidders=[BidderSpec("b", Truthful(), Campaign(1, 1, segments=("news",)))]
        )


def test_segments_list_stored_as_tuple():
    assert Campaign(1, 1, segments=["news"]).segments == ("news",)


def test_campaign_stores_floats_and_defaults():
    campaign = Campaign(budget=10, value=5)
    assert (campaign.budget, campaign.value) == (10.0, 5.0)
    assert isinstance(campaign.budget, float) and isinstance(campaign.value, float)
    assert campaign.cpa_target is None
    assert campaign.segments is None
    assert campaign.horizon_hours is None
    assert campaign.start_hour is None


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"budget": 0}, "budget"),
        ({"budget": -1}, "budget"),
        ({"value": 0}, "value"),
        ({"cpa_target": 0}, "cpa_target"),
        ({"cpa_target": -2}, "cpa_target"),
        ({"horizon_hours": 0}, "horizon_hours"),
        ({"segments": ()}, "segments"),
        ({"start_hour": -0.5}, "start_hour"),
        ({"start_hour": 24}, "start_hour"),
    ],
)
def test_campaign_rejects_bad_values(kwargs, field_name):
    arguments = {"budget": 10, "value": 5} | kwargs
    with pytest.raises(ValueError, match=field_name):
        Campaign(**arguments)


def test_campaign_rejects_string_segments():
    with pytest.raises(TypeError, match="segments"):
        Campaign(10, 5, segments="news")


def test_time_left():
    campaign = Campaign(10, 5, horizon_hours=24, start_hour=0)
    assert campaign.time_left(1000.0) == 86400 - 1000.0


def test_hour_of_day_wraps():
    campaign = Campaign(10, 5, horizon_hours=24, start_hour=22)
    assert campaign.hour_of_day(3 * 3600) == 1.0


@pytest.mark.parametrize("schedule", [{}, {"horizon_hours": 24}, {"start_hour": 0}])
def test_helpers_need_schedule(schedule):
    campaign = Campaign(10, 5, **schedule)
    for helper in (campaign.time_left, campaign.hour_of_day):
        with pytest.raises(ValueError, match="simulator sets"):
            helper(0.0)
