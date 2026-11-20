import threading
from typing import ClassVar

import numpy as np
import pytest

from bidlab import rewards
from bidlab.auctions import FirstPrice, SecondPrice
from bidlab.bidders import Bidder, Linear
from bidlab.config import Campaign, SimConfig
from bidlab.markets import LogisticContext, Segment, Segmented
from bidlab.rewards import cpa_score_of
from bidlab.simulator import run
from bidlab.types import BidRequest, BidResult, Brief


class Recorder(Bidder):
    """Linear bidder that records what it receives."""

    built: ClassVar[list["Recorder"]] = []

    def __init__(self):
        self.rewards = []
        self.episode_rewards = []

    def bind(self, rng):
        # Each run binds a fresh copy, so this lists the copies the runs used.
        super().bind(rng)
        Recorder.built.append(self)

    def forward(self, value, request):
        return value

    def update(self, result):
        self.rewards.append(result.reward)

    def on_episode_end(self, report, episode_reward):
        self.episode_rewards.append((report, episode_reward))


def other():
    return Campaign("other", Linear(), budget=50.0, event_value=4.0)


def small_config(*bidders, **kwargs):
    return SimConfig(campaigns=[*bidders, other()], episodes=2, steps=100, **kwargs)


def built_recorders(*campaigns, **kwargs):
    """Run the campaigns with `other`, and return the log and the recorders each run used."""
    Recorder.built.clear()
    log = run(small_config(*campaigns, **kwargs))
    return log, list(Recorder.built)


def test_bidder_copied_and_bound_once_per_run():
    template = Recorder()
    _, built = built_recorders(Campaign("r", template, budget=15, event_value=5))
    assert len(built) == 1
    assert built[0] is not template
    assert isinstance(built[0].rng, np.random.Generator)
    assert not hasattr(template, "rng")
    assert template.rewards == []


def test_each_bidder_gets_its_own_rng():
    _, built = built_recorders(
        Campaign("a", Recorder(), budget=50, event_value=5),
        Campaign("b", Recorder(), budget=50, event_value=5),
    )
    assert built[0].rng is not built[1].rng
    assert built[0].rng.random() != built[1].rng.random()


def test_uncopyable_bidder_fails_before_step_zero():
    steps_run = []

    class Watch(Recorder):
        def forward(self, value, request):
            steps_run.append(request.time)
            return value

    class Locked(Recorder):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()

    config = SimConfig(
        campaigns=[
            Campaign("watch", Watch(), budget=50, event_value=5),
            Campaign("bad", Locked(), budget=50, event_value=5),
        ],
        episodes=1,
        steps=10,
    )
    with pytest.raises(TypeError, match="'bad' cannot be copied"):
        run(config)

    assert steps_run == []


def test_non_bidder_rejected():
    with pytest.raises(TypeError, match="'bad'.*Bidder instance"):
        Campaign("bad", object(), budget=50, event_value=5)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_string_bidder_rejected():
    with pytest.raises(TypeError, match="Bidder instance"):
        Campaign("pid", "pid_pacing", budget=15, event_value=5)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


@pytest.mark.parametrize(
    "kwargs",
    [{"step_reward": "surplus"}, {"episode_reward": "cpa_score"}, {"bidder": 3}],
)
def test_non_callable_fields_rejected(kwargs):
    fields = {"name": "x", "bidder": Linear(), "budget": 1, "event_value": 1}
    with pytest.raises(TypeError, match="callable|Bidder instance"):
        Campaign(**(fields | kwargs))


def test_reward_function_reaches_result():
    campaign = Campaign(
        "r", Recorder(), budget=50, event_value=5, step_reward=rewards.conversions
    )
    log, built = built_recorders(campaign)
    recorder = built[0]
    expected = (log.converted & (log.winner == 0)).astype(float).ravel()
    assert recorder.rewards == list(expected)

    def custom(request, result, params):
        return 7.0

    _, built = built_recorders(
        Campaign("r", Recorder(), budget=50, event_value=5, step_reward=custom)
    )
    assert set(built[0].rewards) == {7.0}

    def realized(request, result, params):
        return result.realized_value

    log, built = built_recorders(
        Campaign("r", Recorder(), budget=50, event_value=5, step_reward=realized)
    )
    expected = np.where(log.converted & (log.winner == 0), 5.0, 0.0).ravel()
    assert built[0].rewards == list(expected)
    np.testing.assert_array_equal(log.rewards[:, :, 0].ravel(), expected)


def test_cpa_score_episode_reward_uses_target():
    campaign = Campaign(
        "r",
        Recorder(),
        budget=50,
        event_value=5,
        cpa_target=3,
        episode_reward=rewards.cpa_score,
    )
    _, built = built_recorders(campaign)
    for report, episode_reward in built[0].episode_rewards:
        expected = cpa_score_of(
            report.realized_value, report.spend, report.conversions, 3.0
        )
        assert episode_reward == expected


def test_numbers_stored_as_float():
    campaign = Campaign("a", Linear(), 3, 5, cpa_target=2)
    assert all(
        type(value) is float
        for value in (campaign.budget, campaign.event_value, campaign.cpa_target)
    )


def test_campaign_holds_bidder_and_defaults():
    bidder = Recorder()
    campaign = Campaign("a", bidder, budget=10, event_value=5, segments=["news"])
    assert campaign.bidder is bidder
    assert campaign.brief() == Brief(10, 5, segments=("news",))
    assert campaign.brief(12, 6) == Brief(
        10, 5, segments=("news",), horizon_hours=12, start_hour=6
    )
    assert campaign.step_reward is rewards.surplus
    assert campaign.episode_reward is None
    assert campaign.reward_params == {}


@pytest.mark.parametrize(
    "old_field",
    ["value_per_conversion", "params", "campaign", "horizon_hours", "start_hour"],
)
def test_campaign_rejects_old_fields(old_field):
    with pytest.raises(TypeError, match=old_field):
        Campaign("a", Linear(), budget=10, event_value=5, **{old_field: 1})  # pyright: ignore[reportArgumentType, reportCallIssue]  Reason: the test checks the runtime error for a removed field.


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_hours": 0},
        {"horizon_hours": -1},
        {"start_hour": -1},
        {"start_hour": 24},
        {"estimate_noise": -0.1},
    ],
)
def test_sim_config_rejects_bad_schedule_and_noise(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        SimConfig(campaigns=[other()], **kwargs)


def test_builtin_rewards_read_result_value():
    request = BidRequest(time=0.0, value=0.1)
    won = BidResult(
        time=0.0, bid=0.6, won=True, price=0.5, converted=True, realized_value=5.0
    )
    lost = BidResult(time=0.0, bid=0.6, won=False, price=0.0, converted=False)
    assert rewards.surplus(request, won, {}) == 5.0 - 0.5
    assert rewards.realized_value(request, won, {}) == 5.0
    assert rewards.surplus(request, lost, {}) == 0.0
    assert rewards.realized_value(request, lost, {}) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"campaigns": []}, "at least one"),
        ({"campaigns": [other(), other()]}, "unique"),
        ({"campaigns": [other()], "episodes": 0}, "episodes"),
        ({"campaigns": [other()], "steps": 0}, "steps"),
    ],
)
def test_sim_config_rejects_bad_campaigns_and_lengths(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SimConfig(**kwargs)


def test_sim_config_rejects_unknown_auction():
    with pytest.raises(TypeError, match="FirstPrice or SecondPrice"):
        SimConfig(campaigns=[other()], auction="second")  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_default_market_is_logistic_context():
    assert SimConfig(campaigns=[other()]).market == LogisticContext()


def test_world_argument_rejected():
    with pytest.raises(TypeError, match="world"):
        SimConfig(campaigns=[other()], world=LogisticContext())  # pyright: ignore[reportCallIssue]  Reason: the test checks the runtime error for a wrong type.


def test_non_market_rejected():
    with pytest.raises(TypeError, match="Market"):
        SimConfig(campaigns=[other()], market="logistic")  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_bidders_list_stored_as_tuple():
    campaigns = [other(), Campaign("b", Linear(), budget=1, event_value=1)]
    config = SimConfig(campaigns=campaigns)
    assert config.campaigns == tuple(campaigns)


def test_default_auction_is_second_price_no_reserve():
    assert SimConfig(campaigns=[other()]).auction == SecondPrice(reserve=0.0)


def test_first_price_config_runs():
    log = run(small_config(auction=FirstPrice(0.01)))
    sold = log.winner >= 0
    np.testing.assert_array_equal(
        log.price[sold], log.bids[sold.nonzero() + (log.winner[sold],)]
    )


def test_bidder_without_forward_fails_on_creation():
    class Forgot(Bidder):
        def forwards(self, value, request):
            return value

    with pytest.raises(TypeError, match="abstract method 'forward'"):
        Forgot()  # pyright: ignore[reportAbstractUsage]  Reason: the test checks the error for a missing forward method.
    with pytest.raises(TypeError, match=r"pass an instance, such as Forgot\(\)"):
        Campaign("forgot", Forgot, budget=50, event_value=5)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_unknown_segment_rejected():
    market = Segmented((Segment("news", 0.5), Segment("sports", 0.5)))
    campaign = Campaign(
        "b", Linear(), budget=1, event_value=1, segments=("news", "games")
    )
    with pytest.raises(ValueError, match=r"campaign 'b'.*'games'"):
        SimConfig(campaigns=[campaign], market=market)
    with pytest.raises(ValueError, match=r"campaign 'b'.*'news'"):
        SimConfig(
            campaigns=[
                Campaign("b", Linear(), budget=1, event_value=1, segments=("news",))
            ]
        )


def test_segments_list_stored_as_tuple():
    assert Brief(1, 1, segments=["news"]).segments == ("news",)
    assert Campaign("a", Linear(), 1, 1, segments=["news"]).segments == ("news",)


def test_brief_stores_floats_and_defaults():
    brief = Brief(budget=10, event_value=5)
    assert (brief.budget, brief.event_value) == (10.0, 5.0)
    assert isinstance(brief.budget, float) and isinstance(brief.event_value, float)
    assert brief.cpa_target is None
    assert brief.segments is None
    assert brief.horizon_hours is None
    assert brief.start_hour is None


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"budget": 0}, "budget"),
        ({"budget": -1}, "budget"),
        ({"event_value": 0}, "event_value"),
        ({"cpa_target": 0}, "cpa_target"),
        ({"cpa_target": -2}, "cpa_target"),
        ({"horizon_hours": 0}, "horizon_hours"),
        ({"segments": ()}, "segments"),
        ({"start_hour": -0.5}, "start_hour"),
        ({"start_hour": 24}, "start_hour"),
    ],
)
def test_brief_and_campaign_reject_bad_values(kwargs, field_name):
    arguments = {"budget": 10, "event_value": 5} | kwargs
    with pytest.raises(ValueError, match=field_name):
        Brief(**arguments)
    if field_name in ("horizon_hours", "start_hour"):
        return
    with pytest.raises(ValueError, match=field_name):
        Campaign("a", Linear(), **arguments)


def test_brief_and_campaign_reject_string_segments():
    with pytest.raises(TypeError, match="segments"):
        Brief(10, 5, segments="news")
    with pytest.raises(TypeError, match="segments"):
        Campaign("a", Linear(), 10, 5, segments="news")


def test_time_left():
    brief = Brief(10, 5, horizon_hours=24, start_hour=0)
    assert brief.time_left(1000.0) == 86400 - 1000.0


def test_hour_of_day_wraps():
    brief = Brief(10, 5, horizon_hours=24, start_hour=22)
    assert brief.hour_of_day(3 * 3600) == 1.0


@pytest.mark.parametrize("schedule", [{}, {"horizon_hours": 24}, {"start_hour": 0}])
def test_helpers_need_schedule(schedule):
    brief = Brief(10, 5, **schedule)
    for helper in (brief.time_left, brief.hour_of_day):
        with pytest.raises(ValueError, match="simulator sets"):
            helper(0.0)
