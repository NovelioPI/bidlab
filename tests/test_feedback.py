import math
from typing import Any, ClassVar

import pytest

import bidlab as bl

SETTINGS = ("interval", "report_delay", "conversion_delay")
STEPS = 100
# One hour over 100 steps puts auction s at 36 s × s.
SECONDS_PER_STEP = 36.0
CLICKY = bl.markets.LogisticMarket(ctr=0.4, cvr=0.4)


class Probe(bl.Bidder):
    """Bids a fixed amount and records each item it gets with its delivery step."""

    instances: ClassVar[list["Probe"]] = []

    def __init__(self, bid=1.0):
        self.bid = bid
        # Per episode: (delivery step, hook, item) for every item, in arrival order.
        self.episodes: list[list[tuple[int, str, Any]]] = []
        self.budgets_seen: list[list[float]] = []

    def bind(self, rng):
        super().bind(rng)
        Probe.instances.append(self)

    def on_episode_start(self, campaign):
        self.forwards = 0
        self.episodes.append([])
        self.budgets_seen.append([])

    def forward(self, budget_left):
        self.forwards += 1
        self.budgets_seen[-1].append(budget_left)
        return self.bid

    def update_batch(self, bid_results):
        self.episodes[-1].extend((self.forwards - 1, "report", r) for r in bid_results)

    def update_conversions(self, conversions):
        self.episodes[-1].extend(
            (self.forwards - 1, "conversion", c) for c in conversions
        )


def probe_run(feedback=None, *, market=CLICKY, bid=1.0, budget=1000.0, episodes=2):
    Probe.instances.clear()
    config = bl.SimConfig(
        campaigns=[bl.Campaign("p", Probe(bid), budget=budget, base_value=5)],
        auction=bl.FirstPrice(),
        market=market,
        episodes=episodes,
        steps=STEPS,
        horizon_hours=1,
        feedback=feedback or bl.Feedback(),
        seed=3,
    )
    log = bl.run(config)
    (probe,) = Probe.instances
    return log, probe


def step_of(time):
    return round(time / SECONDS_PER_STEP)


def lone_campaign():
    return bl.Campaign(
        "c", bl.Stack(bl.OracleValue(), bl.Linear()), budget=10, base_value=5
    )


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_bad_feedback_setting_rejected(setting, bad):
    with pytest.raises(ValueError, match=setting):
        bl.Feedback(**{setting: bad})


def test_feedback_settings_are_floats():
    feedback = bl.Feedback(interval=900, report_delay=60, conversion_delay=3600)
    assert all(type(getattr(feedback, name)) is float for name in SETTINGS)
    assert bl.SimConfig(campaigns=[lone_campaign()]).feedback == bl.Feedback(0, 0, 0)


def test_feedback_must_be_feedback():
    with pytest.raises(TypeError, match="feedback"):
        bl.SimConfig(
            campaigns=[lone_campaign()],
            feedback={"interval": 900},  # pyright: ignore[reportArgumentType]  Reason: checks the run-time type check.
        )


def test_no_delay_delivers_after_each_auction():
    log, probe = probe_run()
    for episode, records in enumerate(probe.episodes):
        assert [(step, hook) for step, hook, _ in records] == [
            (step, "report") for step in range(STEPS)
        ]
        assert [step_of(r.time) for _, _, r in records] == list(range(STEPS))
        converted = [r.converted for _, _, r in records]
        assert converted == list(log.converted[episode] & (log.winner[episode] == 0))
    assert log.converted.any()
