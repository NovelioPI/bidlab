import threading
from typing import ClassVar

import numpy as np
import pytest

from bidlab.auctions import FirstPrice, SecondPrice
from bidlab.bidders import Bidder, Linear, OracleValue, Stack
from bidlab.config import Campaign, SimConfig
from bidlab.markets import LogisticMarket, Segment, Segmented
from bidlab.simulator import run


class Recorder(Bidder):
    """Linear bidder that records what it receives."""

    built: ClassVar[list["Recorder"]] = []

    def __init__(self):
        self.results = []
        self.reports = []

    def bind(self, rng):
        # Each run binds a fresh copy, so this lists the copies the runs used.
        super().bind(rng)
        Recorder.built.append(self)

    def forward(self, value):
        return value

    def update(self, result):
        self.results.append(result)

    def on_episode_end(self, report):
        self.reports.append(report)


def other():
    return Campaign(
        "other", Stack(OracleValue(), Linear()), budget=50.0, base_value=4.0
    )


def small_config(*bidders, **kwargs):
    return SimConfig(campaigns=[*bidders, other()], episodes=2, steps=100, **kwargs)


def built_recorders(*campaigns, **kwargs):
    """Run the campaigns with `other`, and return the log and the recorders each run used."""
    Recorder.built.clear()
    log = run(small_config(*campaigns, **kwargs))
    return log, list(Recorder.built)


def test_bidder_copied_and_bound_once_per_run():
    template = Recorder()
    _, built = built_recorders(
        Campaign("r", Stack(OracleValue(), template), budget=15, base_value=5)
    )
    assert len(built) == 1
    assert built[0] is not template
    assert isinstance(built[0].rng, np.random.Generator)
    assert not hasattr(template, "rng")
    assert template.results == []


def test_each_bidder_gets_its_own_rng():
    _, built = built_recorders(
        Campaign("a", Stack(OracleValue(), Recorder()), budget=50, base_value=5),
        Campaign("b", Stack(OracleValue(), Recorder()), budget=50, base_value=5),
    )
    assert built[0].rng is not built[1].rng
    assert built[0].rng.random() != built[1].rng.random()


def test_uncopyable_bidder_fails_before_step_zero():
    steps_run = []

    class Watch(Recorder):
        def forward(self, value, **inputs):
            steps_run.append(inputs["time"])
            return value

    class Locked(Recorder):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()

    config = SimConfig(
        campaigns=[
            Campaign("watch", Stack(OracleValue(), Watch()), budget=50, base_value=5),
            Campaign("bad", Stack(OracleValue(), Locked()), budget=50, base_value=5),
        ],
        episodes=1,
        steps=10,
    )
    with pytest.raises(TypeError, match="'bad' cannot be copied"):
        run(config)

    assert steps_run == []


def test_non_bidder_rejected():
    with pytest.raises(TypeError, match="'bad'.*Bidder instance"):
        Campaign("bad", object(), budget=50, base_value=5)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_string_bidder_rejected():
    with pytest.raises(TypeError, match="Bidder instance"):
        Campaign("pid", "pid_pacing", budget=15, base_value=5)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_numbers_stored_as_float():
    campaign = Campaign(
        "a", Stack(OracleValue(), Linear()), budget=3, base_value=5, cpa_target=2
    )
    assert all(
        type(value) is float
        for value in (campaign.budget, campaign.base_value, campaign.cpa_target)
    )


def test_campaign_holds_bidder_and_defaults():
    bidder = Recorder()
    campaign = Campaign("a", bidder, budget=10, base_value=5, segments=["news"])
    assert campaign.bidder is bidder
    assert campaign.segments == ("news",)
    assert campaign.cpa_target is None


@pytest.mark.parametrize(
    "old_field",
    [
        "value_per_conversion",
        "event_value",
        "params",
        "campaign",
        "horizon_hours",
        "start_hour",
        "step_reward",
        "episode_reward",
        "reward_params",
    ],
)
def test_campaign_rejects_old_fields(old_field):
    with pytest.raises(TypeError, match=old_field):
        Campaign("a", Linear(), budget=10, base_value=5, **{old_field: 1})  # pyright: ignore[reportArgumentType, reportCallIssue]  Reason: the test checks the runtime error for a removed field.


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_hours": 0},
        {"horizon_hours": -1},
        {"start_hour": -1},
        {"start_hour": 24},
    ],
)
def test_sim_config_rejects_bad_schedule(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        SimConfig(campaigns=[other()], **kwargs)


def test_estimate_noise_removed():
    with pytest.raises(TypeError, match="estimate_noise"):
        SimConfig(campaigns=[other()], estimate_noise=0.5)  # pyright: ignore[reportCallIssue]  Reason: the test checks the removed setting fails at run time.


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
    assert SimConfig(campaigns=[other()]).market == LogisticMarket()


def test_world_argument_rejected():
    with pytest.raises(TypeError, match="world"):
        SimConfig(campaigns=[other()], world=LogisticMarket())  # pyright: ignore[reportCallIssue]  Reason: the test checks the runtime error for a wrong type.


def test_non_market_rejected():
    with pytest.raises(TypeError, match="Market"):
        SimConfig(campaigns=[other()], market="logistic")  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_bidders_list_stored_as_tuple():
    campaigns = [
        other(),
        Campaign("b", Stack(OracleValue(), Linear()), budget=1, base_value=1),
    ]
    config = SimConfig(campaigns=campaigns)
    assert config.campaigns == tuple(campaigns)


def test_default_auction_is_second_price_no_reserve():
    assert SimConfig(campaigns=[other()]).auction == SecondPrice(reserve=0.0)


def test_first_price_config_runs():
    log = run(small_config(auction=FirstPrice(0.01)))
    np.testing.assert_array_equal(log.price[log.won], log.bids[log.won])


def test_bidder_without_forward_fails_on_creation():
    class Forgot(Bidder):
        def forwards(self, value):
            return value

    with pytest.raises(TypeError, match="abstract method 'forward'"):
        Forgot()  # pyright: ignore[reportAbstractUsage]  Reason: the test checks the error for a missing forward method.
    with pytest.raises(TypeError, match=r"pass an instance, such as Forgot\(\)"):
        Campaign("forgot", Forgot, budget=50, base_value=5)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_unknown_segment_rejected():
    market = Segmented((Segment("news", 0.5), Segment("sports", 0.5)))
    campaign = Campaign(
        "b",
        Stack(OracleValue(), Linear()),
        budget=1,
        base_value=1,
        segments=("news", "games"),
    )
    with pytest.raises(ValueError, match=r"campaign 'b'.*'games'"):
        SimConfig(campaigns=[campaign], market=market)
    with pytest.raises(ValueError, match=r"campaign 'b'.*'news'"):
        SimConfig(
            campaigns=[
                Campaign(
                    "b",
                    Stack(OracleValue(), Linear()),
                    budget=1,
                    base_value=1,
                    segments=("news",),
                )
            ]
        )


def test_segments_list_stored_as_tuple():
    assert Campaign(
        "a", Stack(OracleValue(), Linear()), budget=1, base_value=1, segments=["news"]
    ).segments == ("news",)


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"budget": 0}, "budget"),
        ({"budget": -1}, "budget"),
        ({"base_value": 0}, "base_value"),
        ({"cpa_target": 0}, "cpa_target"),
        ({"cpa_target": -2}, "cpa_target"),
        ({"segments": ()}, "segments"),
    ],
)
def test_campaign_rejects_bad_values(kwargs, field_name):
    arguments = {"budget": 10, "base_value": 5} | kwargs
    with pytest.raises(ValueError, match=f"campaign 'a': {field_name}"):
        Campaign("a", Stack(OracleValue(), Linear()), **arguments)


def test_campaign_rejects_string_segments():
    with pytest.raises(TypeError, match="campaign 'a': segments"):
        Campaign(
            "a",
            Stack(OracleValue(), Linear()),
            budget=10,
            base_value=5,
            segments="news",
        )
