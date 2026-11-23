import dataclasses
from typing import ClassVar, override

import numpy as np
import pytest

import bidlab as bl

SEED = 5


def campaign(name: str = "a") -> bl.Campaign:
    return bl.Campaign(
        name, bl.Stack(bl.OracleValue(), bl.Linear()), budget=10.0, base_value=5.0
    )


def test_competing_defaults_to_true():
    assert bl.SimConfig(campaigns=[campaign()]).competing is True


@pytest.mark.parametrize("value", [0, 1, "no", None])
def test_competing_must_be_bool(value):
    with pytest.raises(TypeError, match="competing"):
        bl.SimConfig(campaigns=[campaign()], competing=value)  # pyright: ignore[reportArgumentType]  Reason: the test passes a wrong type on purpose.


ARRAYS = ("bids", "true_values", "won", "price", "clicked", "converted")
PRICED = bl.markets.LogisticMarket(price_median=0.2, ctr_bias_mean=-1.0)


def mixed_campaigns() -> tuple[bl.Campaign, ...]:
    # Budgets bind for the first two, so pacing and the bid cap both matter.
    return (
        bl.Campaign(
            "conv", bl.Stack(bl.OracleValue(), bl.Linear()), budget=3.0, base_value=20.0
        ),
        bl.Campaign(
            "click",
            bl.Stack(bl.OracleValue(), bl.PidPacing()),
            budget=2.0,
            base_value=2.0,
            event="click",
        ),
        bl.Campaign(
            "imp",
            bl.Stack(bl.OracleValue(), bl.Linear(multiplier=0.7)),
            budget=50.0,
            base_value=0.25,
            event="impression",
        ),
    )


def compared(campaigns, **kwargs) -> bl.SimConfig:
    defaults = {"episodes": 3, "steps": 300, "market": PRICED}
    return bl.SimConfig(campaigns=campaigns, competing=False, **(defaults | kwargs))


class Reports(bl.Callback):
    def __init__(self):
        self.reports = []

    def on_episode_end(self, episode, reports):
        self.reports.append(dict(reports))


def run_with_reports(config):
    reports = Reports()
    log = bl.run(config, callbacks=[reports], progress_bar=False, seed=SEED)
    return log, reports.reports


def assert_matches_solo_runs(config):
    log, reports = run_with_reports(config)
    for i, campaign in enumerate(config.campaigns):
        solo = dataclasses.replace(config, campaigns=(campaign,), competing=True)
        solo_log, solo_reports = run_with_reports(solo)
        for name in ARRAYS:
            column = getattr(log, name)[..., i]
            assert column.tobytes() == getattr(solo_log, name)[..., 0].tobytes(), name
        for episode in range(config.episodes):
            own = reports[episode][campaign.name]
            assert own == solo_reports[episode][campaign.name]
            assert bl.regret(log, episode, i) == bl.regret(solo_log, episode, 0)
    return log


@pytest.mark.parametrize("auction", [bl.FirstPrice(0.01), bl.SecondPrice(0.01)])
def test_each_campaign_matches_its_solo_run(auction):
    log = assert_matches_solo_runs(compared(mixed_campaigns(), auction=auction))
    spend = log.price.sum(axis=1)
    assert (spend[:, 0] > 0.99 * 3.0).any() and (spend[:, 1] > 0.99 * 2.0).any()
    assert log.clicked.any() and log.converted.any()


def test_solo_match_holds_with_feedback_delays():
    feedback = bl.Feedback(report_delay=600, conversion_delay=900, interval=360)
    assert_matches_solo_runs(compared(mixed_campaigns(), feedback=feedback))


class Probe(bl.Bidder):
    """Bids a fixed amount and records the market inputs of every call."""

    running: ClassVar[list["Probe"]] = []

    def __init__(self, bid: float = 0.3):
        self.bid = bid
        self.seen: list[tuple[bytes, float, str, bl.MarketTruth]] = []
        self.calls = 0

    @override
    def bind(self, rng):
        super().bind(rng)
        Probe.running.append(self)

    def forward(self, features, floor, segment, market):
        self.calls += 1
        self.seen.append((features["vector"].tobytes(), floor, segment, market))
        return self.bid


def run_probes(config):
    Probe.running.clear()
    log = bl.run(config, progress_bar=False, seed=SEED)
    return log, list(Probe.running)


def test_probes_see_the_same_market():
    probes = tuple(
        bl.Campaign(name, Probe(), budget=5.0, base_value=1.0) for name in ("a", "b")
    )
    _, (first, second) = run_probes(compared(probes, episodes=1, steps=50))
    assert len(first.seen) == 50
    assert first.seen == second.seen


def test_equal_campaigns_get_equal_columns():
    twins = tuple(
        bl.Campaign(
            name,
            bl.Stack(bl.OracleValue(), bl.PidPacing()),
            budget=3.0,
            base_value=20.0,
        )
        for name in ("a", "b")
    )
    log = bl.run(compared(twins), progress_bar=False, seed=SEED)
    assert log.won.any()
    for name in ARRAYS:
        column = getattr(log, name)
        np.testing.assert_array_equal(column[..., 0], column[..., 1])


@pytest.mark.parametrize("auction", [bl.FirstPrice(0.01), bl.SecondPrice(0.01)])
def test_several_winners_pay_their_own_price(auction):
    market = bl.markets.LogisticMarket(price_median=0.2, floor_share=(2, 5))
    bidders = tuple(
        bl.Campaign(name, Probe(bid), budget=1e6, base_value=1.0)
        for name, bid in (("high", 0.9), ("low", 0.5))
    )
    log = bl.run(
        compared(bidders, auction=auction, market=market), progress_bar=False, seed=SEED
    )
    both = log.won.all(axis=2)
    assert both.any()
    paid = log.price[both]
    if isinstance(auction, bl.FirstPrice):
        np.testing.assert_array_equal(paid, log.bids[both])
    else:
        lowest = np.maximum(np.maximum(auction.reserve, log.floor), log.market_price)
        np.testing.assert_array_equal(paid, np.stack([lowest[both]] * 2, axis=1))


class CountingMarket(bl.Market):
    """A priced market that records the campaign count it is asked for."""

    def __init__(self):
        self.counts: list[int] = []

    def draws(self, count, rng):
        self.counts.append(count)
        yield from PRICED.draws(count, rng)


def test_market_is_drawn_for_one_campaign():
    market = CountingMarket()
    bl.run(compared(mixed_campaigns(), market=market), progress_bar=False, seed=SEED)
    assert market.counts == [1]


def test_draw_without_market_price_raises():
    probes = (bl.Campaign("a", Probe(), budget=5.0, base_value=1.0),)
    Probe.running.clear()
    with pytest.raises(ValueError, match="competing=False") as raised:
        bl.run(
            compared(probes, market=bl.markets.LogisticMarket()),
            progress_bar=False,
            seed=SEED,
        )
    assert "market price" in str(raised.value)
    assert "price_median" in str(raised.value)
    assert [probe.calls for probe in Probe.running] == [0]


def test_segments_under_competing_false():
    market = bl.markets.Segmented(
        (bl.markets.Segment("news", 0.5), bl.markets.Segment("sports", 0.5)),
        base=bl.markets.LogisticMarket(price_median=0.2),
    )
    campaigns = (
        bl.Campaign(
            "news_only",
            bl.Stack(bl.OracleValue(), bl.Linear()),
            budget=1e6,
            base_value=20.0,
            segments=("news",),
        ),
        *mixed_campaigns()[:2],
    )
    config = compared(campaigns, market=market)
    log = assert_matches_solo_runs(config)
    sports = log.segment == market.segment_names.index("sports")
    assert sports.any() and log.won[~sports][:, 0].any()
    assert not log.won[sports][:, 0].any()
    assert (log.bids[sports][:, 0] == 0.0).all()


def test_workers_match_under_competing_false():
    config = compared(mixed_campaigns(), episodes=2, steps=100)
    serial = bl.run(config, seed=range(2), progress_bar=False)
    parallel = bl.run(config, seed=range(2), workers=2, progress_bar=False)
    assert serial.equals(parallel)


def test_platform_metrics_are_nan():
    config = compared(mixed_campaigns(), episodes=2, steps=100)
    rows = bl.platform_metrics(bl.run(config, progress_bar=False, seed=SEED))
    assert [row.episode for row in rows] == [0, 1]
    assert all(np.isnan(row.revenue) and np.isnan(row.welfare) for row in rows)
    results = bl.run(config, seed=range(2), progress_bar=False)
    assert all(
        np.isnan(row["revenue"]) and np.isnan(row["welfare"]) for row in results.rows
    )
    assert results.equals(bl.run(config, seed=range(2), progress_bar=False))
