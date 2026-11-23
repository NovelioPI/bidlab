import math
from typing import Any, ClassVar

import numpy as np
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

    def on_episode_end(self, report):
        self.episodes[-1].append((self.forwards - 1, "end", report))


class LoggingProbe(Probe):
    """A Probe that logs how many reports it has got so far in the episode."""

    def update_batch(self, bid_results):
        super().update_batch(bid_results)
        self.log("reports", len(reports(self.episodes[-1])))


def probe_run(
    feedback=None,
    *,
    market: bl.markets.Market = CLICKY,
    bid=1.0,
    budget=1000.0,
    episodes=2,
    probe=Probe,
    segments=None,
):
    Probe.instances.clear()
    campaign = bl.Campaign(
        "p", probe(bid), budget=budget, base_value=5, segments=segments
    )
    config = bl.SimConfig(
        campaigns=[campaign],
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


def reports(records):
    return [item for _, hook, item in records if hook == "report"]


def delivery_steps(records, hook="report"):
    """Map each item's auction step to the step it arrived after."""
    return {step_of(item.time): step for step, kind, item in records if kind == hook}


def expected_steps(delay, boundaries):
    """The first boundary step at or after each report's due time, else the flush."""
    flush = STEPS - 1
    return {
        s: next(
            (
                d
                for d in boundaries
                if d * SECONDS_PER_STEP >= s * SECONDS_PER_STEP + delay
            ),
            flush,
        )
        for s in range(STEPS)
    }


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
        ] + [(STEPS - 1, "end")]
        assert [step_of(r.time) for r in reports(records)] == list(range(STEPS))
        converted = [r.converted for r in reports(records)]
        assert converted == list(log.converted[episode] & (log.winner[episode] == 0))
    assert log.converted.any()


def test_report_delay_delivers_after_due_step():
    _, probe = probe_run(bl.Feedback(report_delay=100))
    for records in probe.episodes:
        steps = delivery_steps(records)
        assert steps == {s: min(s + 3, STEPS - 1) for s in range(STEPS)}
        assert steps == expected_steps(100, range(STEPS))


def test_interval_delivers_at_boundaries():
    _, probe = probe_run(bl.Feedback(interval=360))
    for records in probe.episodes:
        steps = delivery_steps(records)
        assert sorted(set(steps.values())) == [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
        assert [s for s, d in steps.items() if d == 10] == list(range(11))
        assert steps == expected_steps(0, range(10, STEPS, 10))


def test_interval_and_report_delay_together():
    _, probe = probe_run(bl.Feedback(interval=360, report_delay=100))
    for records in probe.episodes:
        steps = delivery_steps(records)
        assert [s for s, d in steps.items() if d == 20] == list(range(8, 18))
        assert steps == expected_steps(100, range(10, STEPS, 10))


FEEDBACKS = [
    bl.Feedback(),
    bl.Feedback(report_delay=100),
    bl.Feedback(interval=360, report_delay=1000),
    bl.Feedback(report_delay=10 * 3600),
]


@pytest.mark.parametrize("feedback", FEEDBACKS)
@pytest.mark.parametrize("market", ["plain", "segmented"])
def test_flush_delivers_everything(feedback, market):
    if market == "plain":
        log, probe = probe_run(feedback)
        mine = np.ones((2, STEPS), dtype=bool)
    else:
        segmented = bl.markets.Segmented(
            [bl.markets.Segment("a", 0.5), bl.markets.Segment("b", 0.5)],
            base=CLICKY,
        )
        log, probe = probe_run(feedback, market=segmented, segments=["a"])
        mine = log.segment == 0
    for episode, records in enumerate(probe.episodes):
        assert records[-1][1] == "end"
        times = [step_of(r.time) for r in reports(records)]
        assert times == list(np.flatnonzero(mine[episode]))


def test_budget_left_follows_reports():
    budget = 1000.0
    log, probe = probe_run(bl.Feedback(report_delay=600), budget=budget)
    for episode, records in enumerate(probe.episodes):
        reported = budget
        expected = []
        delivered = iter(r for r in records if r[1] == "report")
        pending = next(delivered, None)
        for step in range(STEPS):
            while pending is not None and pending[0] < step:
                if pending[2].won:
                    reported -= pending[2].price
                pending = next(delivered, None)
            expected.append(reported)
        assert probe.budgets_seen[episode] == expected
        true_left = budget - np.concatenate([[0.0], np.cumsum(log.price[episode])[:-1]])
        assert (np.array(expected) > true_left).any()


def test_cap_stays_exact_with_report_delay():
    budget = 3.0
    config = bl.SimConfig(
        campaigns=[
            bl.Campaign(
                "pid",
                bl.Stack(bl.OracleValue(), bl.PidPacing()),
                budget=budget,
                base_value=5,
            ),
            bl.Campaign(
                "greedy",
                bl.Stack(bl.OracleValue(), bl.Linear(4.0)),
                budget=budget,
                base_value=5,
            ),
        ],
        auction=bl.FirstPrice(),
        episodes=3,
        steps=STEPS,
        horizon_hours=1,
        feedback=bl.Feedback(report_delay=600),
    )
    log = bl.run(config)
    for index in range(2):
        paid = np.where(log.winner == index, log.price, 0.0)
        earlier = np.cumsum(paid, axis=1) - paid
        # The simulator subtracts prices one by one, so a sum can round 1 ulp above.
        assert (paid.sum(axis=1) <= budget + 1e-9).all()
        assert (log.bids[:, :, index] <= budget - earlier + 1e-9).all()
    assert (np.where(log.winner == 1, log.price, 0.0).sum(axis=1) > 0.9 * budget).all()


def test_feedback_draws_no_random_number():
    def fixed(feedback):
        return bl.SimConfig(
            campaigns=[
                bl.Campaign(
                    f"c{i}",
                    bl.Stack(bl.OracleValue(), bl.Linear(m)),
                    budget=2.0,
                    base_value=5,
                )
                for i, m in enumerate([0.8, 1.0, 1.3])
            ],
            market=bl.markets.LogisticMarket(ctr=0.3, cvr=0.3, price_median=0.05),
            episodes=2,
            steps=STEPS,
            horizon_hours=1,
            feedback=feedback,
        )

    plain = bl.run(fixed(bl.Feedback()))
    late = bl.run(
        fixed(bl.Feedback(interval=360, report_delay=100, conversion_delay=900))
    )
    for name in (
        "bids",
        "true_values",
        "winner",
        "price",
        "clicked",
        "converted",
        "segment",
        "market_price",
        "floor",
    ):
        assert getattr(plain, name).tobytes() == getattr(late, name).tobytes(), name


def test_logs_land_at_delivery_step():
    log, _ = probe_run(bl.Feedback(report_delay=100), probe=LoggingProbe)
    logged = log.logged["reports"]["p"]
    for episode in range(2):
        assert list(np.flatnonzero(~np.isnan(logged[episode]))) == list(range(3, STEPS))
        # The flush logs at the last step, after the regular delivery there.
        assert logged[episode, -1] == STEPS


def test_feedback_results_repeat():
    config = bl.SimConfig(
        campaigns=[
            bl.Campaign(
                f"c{i}",
                bl.Stack(bl.OracleValue(), bl.PidPacing()),
                budget=2.0,
                base_value=5,
            )
            for i in range(2)
        ],
        episodes=2,
        steps=STEPS,
        horizon_hours=1,
        feedback=bl.Feedback(interval=360, report_delay=100, conversion_delay=900),
    )
    first = bl.run_many(config, seeds=range(2))
    assert first.equals(bl.run_many(config, seeds=range(2)))
    assert first.equals(bl.run_many(config, seeds=range(2), workers=2))
