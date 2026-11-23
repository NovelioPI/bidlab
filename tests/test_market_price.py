from itertools import cycle
from typing import Any, ClassVar

import numpy as np
import pytest

from bidlab.auctions import FirstPrice, SecondPrice
from bidlab.bidders import Bidder, Linear, OracleValue, Stack
from bidlab.config import Campaign, SimConfig
from bidlab.markets import Market, MarketDraw
from bidlab.metrics import campaign_metrics, platform_metrics, regret
from bidlab.simulator import run
from bidlab.types import BidResult


class PricedMarket(Market):
    """Yields a fixed pCVR, and each (market price, floor) pair in turn."""

    def __init__(self, pairs, pcvr=0.5, features=None):
        self.pairs = pairs
        self.pcvr = pcvr
        self.features = features or {}

    def draws(self, count, rng):
        for price, floor in cycle(self.pairs):
            yield MarketDraw(
                self.features,
                pcvr=np.full(count, self.pcvr),
                market_price=price,
                floor=floor,
            )


class Spy(Bidder):
    """Bids a fixed amount and records every input and result."""

    instances: ClassVar[list["Spy"]] = []

    def __init__(self, bid=1.0):
        self.bid = bid
        self.requests: list[dict[str, Any]] = []
        self.results: list[BidResult] = []

    def bind(self, rng):
        super().bind(rng)
        Spy.instances.append(self)

    def forward(self, **inputs):
        self.requests.append(inputs)
        return self.bid

    def update(self, result):
        self.results.append(result)


def run_market(market, *bids, auction=None, steps=4, budget=100.0):
    Spy.instances.clear()
    campaigns = [
        Campaign(f"c{i}", Spy(bid), budget=budget, base_value=1.0, event="impression")
        for i, bid in enumerate(bids)
    ]
    config = SimConfig(
        campaigns=campaigns,
        episodes=1,
        steps=steps,
        market=market,
        auction=auction or SecondPrice(),
    )
    return run(config), list(Spy.instances)


def test_run_log_records_price_and_floor():
    market = PricedMarket([(0.4, 0.1), (None, 0.0), (0.2, 0.3), (None, 0.05)])
    log, _ = run_market(market, 1.0, auction=SecondPrice(0.2))
    np.testing.assert_array_equal(log.market_price, [[0.4, np.nan, 0.2, np.nan]])
    # The market floor alone, though the reserve 0.2 is higher at steps 0, 1 and 3.
    np.testing.assert_array_equal(log.floor, [[0.1, 0.0, 0.3, 0.05]])
    for array in (log.market_price, log.floor):
        assert array.dtype == np.float64
        assert not array.flags.writeable


@pytest.mark.parametrize("field", ["market_price", "floor"])
@pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf])
def test_bad_market_price_or_floor_rejected(field, bad):
    pair = (bad, 0.0) if field == "market_price" else (None, bad)
    market = PricedMarket([(0.1, 0.0), pair])
    message = (
        rf"PricedMarket at episode 0, step 1: {field} must be finite and >= 0, got"
    )
    with pytest.raises(ValueError, match=message):
        run_market(market, 1.0)


def test_zero_market_price_and_floor_accepted():
    log, _ = run_market(PricedMarket([(0.0, 0.0)]), 1.0)
    assert log.won[:, :, 0].all()


def test_run_applies_market_price_and_floor():
    # Pairs of (market price, floor); campaign bids are 0.5 and 0.3; reserve 0.1.
    market = PricedMarket(
        [(None, 0.0), (0.6, 0.0), (0.5, 0.0), (0.4, 0.0), (None, 0.55), (None, 0.5)]
    )
    log, _ = run_market(market, 0.5, 0.3, auction=SecondPrice(0.1), steps=6)
    np.testing.assert_array_equal(log.won[0, :, 0], [1, 0, 0, 1, 0, 1])
    assert not log.won[:, :, 1].any()
    np.testing.assert_array_equal(log.price[0, :, 0], [0.3, 0.0, 0.0, 0.4, 0.0, 0.5])


@pytest.mark.parametrize("auction", [FirstPrice(), SecondPrice()])
def test_sold_outside_logs_no_winner(auction):
    log, spies = run_market(PricedMarket([(2.0, 0.0)], pcvr=1.0), 1.0, 0.5)
    assert not log.won.any()
    assert (log.price == 0).all()
    assert not log.clicked.any() and not log.converted.any()
    for spy in spies:
        assert spy.results
        for result in spy.results:
            assert (result.won, result.price, result.clicked) == (False, 0.0, False)
            assert (result.converted, result.realized_value) == (False, 0.0)


def test_no_input_holds_market_price():
    price = 0.737
    market = PricedMarket([(price, 0.0)], features={"zone": 3.0})
    _, (spy,) = run_market(market, 1.0)
    assert spy.requests
    for request in spy.requests:
        assert "market_price" not in request
        assert price not in [v for v in request.values() if isinstance(v, float)]
        assert price not in list(request["features"].values())
        truth = request["market"]
        assert price not in (truth.pctr, truth.pcvr)


@pytest.mark.parametrize(
    ("reserve", "floor", "expected"),
    [
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.2),
        (0.0, 0.3, 0.3),
        (0.2, 0.3, 0.3),
        (0.4, 0.3, 0.4),
    ],
)
def test_floor_input_is_effective_floor(reserve, floor, expected):
    _, (spy,) = run_market(
        PricedMarket([(None, floor)]), 1.0, auction=SecondPrice(reserve)
    )
    assert [request["floor"] for request in spy.requests] == [expected] * 4


class FloorReader(Bidder):
    def forward(self, floor):
        self.log("floor", floor)
        return 1.0


def test_bidder_names_floor_input():
    config = SimConfig(
        campaigns=[Campaign("a", FloorReader(), budget=10.0, base_value=1.0)],
        episodes=1,
        steps=3,
        market=PricedMarket([(None, 0.25)]),
    )
    np.testing.assert_array_equal(run(config).logged["floor"]["a"], [[0.25] * 3])


def test_feature_named_floor_rejected():
    market = PricedMarket([(None, 0.0)], features={"floor": 0.2})
    with pytest.raises(ValueError, match=r"features \['floor'\] have the names"):
        run_market(market, 1.0)


REGRET_STEPS = 50


def lone_truthful_log(auction):
    campaign = Campaign(
        "t",
        Stack(OracleValue(), Linear()),
        budget=100.0,
        base_value=1.0,
        event="impression",
    )
    config = SimConfig(
        campaigns=[campaign],
        episodes=1,
        steps=REGRET_STEPS,
        market=PricedMarket([(0.6, 0.0)]),
        auction=auction,
    )
    return run(config)


def test_regret_zero_second_price_against_market():
    log = lone_truthful_log(SecondPrice())
    assert (log.price == 0.6).all()
    assert abs(regret(log, 0, 0)) < 1e-9


def test_regret_first_price_against_market():
    # Multiplier 0.65 wins every step at 0.65; 0.6 ties the market price and loses.
    log = lone_truthful_log(FirstPrice())
    assert abs(regret(log, 0, 0) - 0.35 * REGRET_STEPS) < 1e-9


def test_regret_replay_meets_market_floor():
    # Only multipliers from 1.2 meet the floor, and they pay more than the value 1.
    campaign = Campaign(
        "t",
        Stack(OracleValue(), Linear()),
        budget=100.0,
        base_value=1.0,
        event="impression",
    )
    config = SimConfig(
        campaigns=[campaign],
        episodes=1,
        steps=REGRET_STEPS,
        market=PricedMarket([(None, 1.2)]),
    )
    log = run(config)
    assert not log.won.any()
    assert abs(regret(log, 0, 0)) < 1e-9


def test_revenue_and_welfare_count_campaign_wins():
    market = PricedMarket([(0.2, 0.0), (0.9, 0.0), (None, 0.0), (0.45, 0.1)])
    log, _ = run_market(market, 0.8, 0.4, steps=40)
    sold = log.won[0].any(axis=1)
    assert 0 < sold.sum() < sold.size
    (platform,) = platform_metrics(log)
    spend = sum(row.spend for row in campaign_metrics(log))
    assert platform.revenue == pytest.approx(spend)
    expected_welfare = log.true_values[0][log.won[0]].sum()
    assert platform.welfare == pytest.approx(expected_welfare)
    assert platform.welfare < log.true_values[0].max(axis=1).sum()
