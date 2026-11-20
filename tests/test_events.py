from typing import Any, ClassVar

import numpy as np
import pytest
from scipy.special import logit

from bidlab.bidders import Bidder, Linear
from bidlab.config import Campaign, SimConfig
from bidlab.markets import LogisticMarket, Market, MarketDraw
from bidlab.metrics import campaign_metrics
from bidlab.simulator import run
from bidlab.types import BidResult, EpisodeReport, Event, EventName

EVENT_NAMES: tuple[EventName, ...] = ("impression", "click", "conversion")


class Spy(Bidder):
    """Bids the value and records every call it receives."""

    instances: ClassVar[list["Spy"]] = []

    def __init__(self):
        self.requests: list[dict[str, Any]] = []
        self.results: list[BidResult] = []
        self.starts: list[Campaign] = []
        self.ends: list[EpisodeReport] = []

    def bind(self, rng):
        super().bind(rng)
        Spy.instances.append(self)

    def on_episode_start(self, campaign):
        self.starts.append(campaign)

    def forward(self, value, **inputs):
        self.requests.append({"value": value, **inputs})
        return value

    def update(self, result):
        self.results.append(result)

    def on_episode_end(self, report):
        self.ends.append(report)


class FixedMarket(Market):
    """Yields the given pCTR and pCVR at every step."""

    def __init__(self, pctr, pcvr):
        self.pctr = pctr
        self.pcvr = pcvr

    def draws(self, count, rng):
        pctr = None if self.pctr is None else np.array(self.pctr)
        while True:
            yield MarketDraw({}, pctr=pctr, pcvr=np.array(self.pcvr))


def event_factor(event, pctr, pcvr):
    return {"impression": 1.0, "click": pctr, "conversion": pctr * pcvr}[event]


def campaign(name, event: EventName = "conversion", base_value=5.0, **kwargs):
    return Campaign(
        name, Spy(), budget=100.0, base_value=base_value, event=event, **kwargs
    )


def run_spies(*campaigns, **kwargs):
    settings: dict[str, Any] = {"episodes": 2, "steps": 200, "seed": 7}
    Spy.instances.clear()
    log = run(SimConfig(campaigns=campaigns, **(settings | kwargs)))
    return log, list(Spy.instances)


def test_default_event_is_conversion():
    assert Campaign("a", Linear(), budget=10, base_value=5).event is Event.CONVERSION


@pytest.mark.parametrize("event", [*Event, *EVENT_NAMES])
def test_event_accepts_member_and_string(event):
    stored = Campaign("a", Linear(), budget=10, base_value=5, event=event).event
    assert stored is Event(event)
    assert stored == str(event)


@pytest.mark.parametrize(
    ("event", "error"),
    [("clicks", ValueError), ("", ValueError), (1, TypeError), (None, TypeError)],
)
def test_bad_event_rejected(event, error):
    names = "'impression', 'click', 'conversion'"
    with pytest.raises(error, match=f"event must be one of {names}, got"):
        Campaign("a", Linear(), budget=10, base_value=5, event=event)


@pytest.mark.parametrize("base_value", [0, -1.0])
def test_base_value_must_be_positive(base_value):
    with pytest.raises(ValueError, match="base_value must be > 0"):
        Campaign("a", Linear(), budget=10, base_value=base_value)


def test_bidder_sees_event_and_base_value():
    _, (spy,) = run_spies(campaign("a", event="click", base_value=0.4))
    assert {(start.event, start.base_value) for start in spy.starts} == {
        (Event.CLICK, 0.4)
    }


@pytest.mark.parametrize("pctr", [None, 0.3])
def test_true_values_follow_event(pctr):
    log, _ = run_spies(
        *(campaign(event, event=event, base_value=2.0) for event in EVENT_NAMES),
        market=FixedMarket(None if pctr is None else [pctr] * 3, [0.4] * 3),
    )
    rate = 1.0 if pctr is None else pctr
    expected = [2.0 * event_factor(event, rate, 0.4) for event in EVENT_NAMES]
    assert (log.true_values == np.array(expected)).all()


def test_value_input_follows_event():
    market = LogisticMarket(ctr_bias_mean=-2.0)
    _, spies = run_spies(
        *(campaign(event, event=event, base_value=2.0) for event in EVENT_NAMES),
        market=market,
    )
    for event, spy in zip(EVENT_NAMES, spies, strict=True):
        for request in spy.requests:
            truth = request["market"]
            assert request["value"] == 2.0 * event_factor(event, truth.pctr, truth.pcvr)


def noisy_rates(event, pctr=0.3):
    _, (spy,) = run_spies(
        campaign("a", event=event, base_value=2.0),
        market=FixedMarket([pctr], [0.2]),
        episodes=1,
        steps=20_000,
        estimate_noise=0.5,
    )
    return np.array([request["value"] for request in spy.requests]) / 2.0


def test_noise_follows_event():
    assert (noisy_rates("impression") == 1.0).all()
    click_errors = logit(noisy_rates("click")) - logit(0.3)
    assert abs(click_errors.std() - 0.5) < 0.05
    # A pCTR of 1 has an infinite logit, so noise leaves it exact and shows pCVR's error.
    conversion_errors = logit(noisy_rates("conversion", pctr=1.0)) - logit(0.2)
    assert abs(conversion_errors.std() - 0.5) < 0.05


@pytest.mark.parametrize("noise", [0.0, 0.5])
def test_event_changes_no_other_input(noise):
    def inputs(event):
        _, spies = run_spies(
            campaign("a", event=event),
            campaign("b"),
            market=LogisticMarket(ctr_bias_mean=-2.0),
            estimate_noise=noise,
        )
        markets = [[r["market"] for r in spy.requests] for spy in spies]
        return markets, [r["value"] for r in spies[1].requests]

    assert inputs("conversion") == inputs("click")


def event_mask(event, log, index):
    won = log.winner == index
    return {
        "impression": won,
        "click": log.clicked & won,
        "conversion": log.converted & won,
    }[event]


# Close expected values, so each campaign wins some auctions.
BASE_VALUES = {"impression": 0.1, "click": 0.4, "conversion": 2.0}


def three_events(**kwargs):
    return run_spies(
        *(campaign(e, event=e, base_value=BASE_VALUES[e]) for e in EVENT_NAMES),
        market=LogisticMarket(ctr_bias_mean=-1.0, cvr_bias_mean=-1.0),
        **kwargs,
    )


def test_realized_value_per_event():
    log, spies = three_events()
    for index, (event, spy) in enumerate(zip(EVENT_NAMES, spies, strict=True)):
        realized = np.array([result.realized_value for result in spy.results])
        mask = event_mask(event, log, index).ravel()
        np.testing.assert_array_equal(realized, np.where(mask, BASE_VALUES[event], 0))
        assert realized.any()


def test_episode_report_counts_event():
    log, spies = three_events()
    for index, (event, spy) in enumerate(zip(EVENT_NAMES, spies, strict=True)):
        won = log.winner == index
        for episode, report in enumerate(spy.ends):
            assert report.clicks == (log.clicked[episode] & won[episode]).sum()
            events = event_mask(event, log, index)[episode].sum()
            assert report.realized_value == events * BASE_VALUES[event]


def test_episode_report_needs_clicks():
    with pytest.raises(TypeError, match="clicks"):
        EpisodeReport(budget=1.0, spend=0.0, wins=0, conversions=0, realized_value=0.0)  # pyright: ignore[reportCallIssue]  Reason: the test checks the runtime error for a missing field.


def test_ctr_cpc_nan_without_wins_or_clicks():
    loser = Campaign("loser", Linear(multiplier=0.0), budget=10.0, base_value=5.0)
    # An impression campaign bids its base value, so it wins although pCTR is 0.
    winner = Campaign(
        "winner", Linear(), budget=10.0, base_value=0.1, event="impression"
    )
    cfg = SimConfig(
        campaigns=[loser, winner],
        episodes=1,
        steps=50,
        market=FixedMarket([0.0, 0.0], [0.5, 0.5]),
    )
    rows = {row.campaign: row for row in campaign_metrics(run(cfg))}
    assert rows["loser"].wins == 0
    assert np.isnan(rows["loser"].ctr)
    assert np.isnan(rows["loser"].cpc)
    assert rows["winner"].wins > 0
    assert rows["winner"].ctr == 0.0
    assert np.isnan(rows["winner"].cpc)
