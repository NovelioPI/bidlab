import dataclasses
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import ClassVar, override

import numpy as np
import pytest

from bidlab.auctions import FirstPrice, SecondPrice
from bidlab.bidders import Bidder, Linear, OracleValue, PidPacing, Stack
from bidlab.config import Campaign, SimConfig
from bidlab.markets import LogisticMarket, Market, MarketDraw, Segment, Segmented
from bidlab.metrics import campaign_metrics
from bidlab.simulator import RunLog, run
from bidlab.types import EpisodeReport

SEED = 4

PRICED = LogisticMarket(price_median=0.2, floor_share=(2, 5))
ARRAYS = (
    "bids",
    "true_values",
    "won",
    "price",
    "clicked",
    "converted",
    "segment",
    "market_price",
    "floor",
    "budgets",
)


def truthful():
    return Stack(OracleValue(), Linear())


def paced():
    return Stack(OracleValue(), PidPacing())


class Probe(Bidder):
    """Passes its value on and records what the simulator gives it."""

    built: ClassVar[list["Probe"]] = []

    def __init__(self):
        self.campaigns: list[Campaign] = []
        self.first_budget_left: list[float] = []
        self.reports: list[EpisodeReport] = []
        self.forward_calls = 0

    @override
    def bind(self, rng):
        super().bind(rng)
        Probe.built.append(self)

    @override
    def on_episode_start(self, campaign):
        self.campaigns.append(campaign)
        self.first_budget_left.append(math.nan)

    @override
    def forward(self, value, budget_left):
        self.forward_calls += 1
        if math.isnan(self.first_budget_left[-1]):
            self.first_budget_left[-1] = budget_left

        return value

    @override
    def on_episode_end(self, report):
        self.reports.append(report)


@pytest.fixture
def probes():
    Probe.built.clear()
    yield Probe.built
    Probe.built.clear()


def config_of(*campaigns, **settings):
    defaults = {"market": PRICED, "episodes": 3, "steps": 200}
    return SimConfig(campaigns=campaigns, **(defaults | settings))


def market_cost(log: RunLog, index: int) -> float:
    campaign = log.config.campaigns[index]
    costs = np.maximum(
        np.maximum(
            np.nan_to_num(log.market_price, nan=0.0), log.config.auction.reserve
        ),
        log.floor,
    )
    names = log.config.market.segment_names
    allowed = [
        i for i, name in enumerate(names) if name in (campaign.segments or names)
    ]
    profitable = log.true_values[:, :, index] >= costs
    total = costs[np.isin(log.segment, allowed) & profitable].sum()
    return float(total) / log.config.episodes


def twin(config: SimConfig, log: RunLog) -> SimConfig:
    campaigns = [
        dataclasses.replace(campaign, budget=float(budget), budget_fraction=1.0)
        for campaign, budget in zip(config.campaigns, log.budgets, strict=True)
    ]
    return dataclasses.replace(config, campaigns=campaigns)


def assert_same_log(first: RunLog, second: RunLog):
    for name in ARRAYS:
        assert np.array_equal(
            getattr(first, name), getattr(second, name), equal_nan=True
        ), name

    assert first.logged.keys() == second.logged.keys()
    for name, campaigns in first.logged.items():
        for campaign, values in campaigns.items():
            assert np.array_equal(
                values, second.logged[name][campaign], equal_nan=True
            ), (name, campaign)


def test_budget_fraction_defaults_to_one():
    campaign = Campaign("a", truthful(), budget=3, base_value=5)
    assert campaign.budget_fraction == 1.0
    assert type(campaign.budget_fraction) is float


@pytest.mark.parametrize("fraction", [0, -0.5, math.nan, math.inf])
def test_bad_fraction_raises(fraction):
    with pytest.raises(ValueError, match="budget_fraction"):
        Campaign("a", truthful(), budget=3, base_value=5, budget_fraction=fraction)


@pytest.mark.parametrize("budget", [0, -1])
def test_bad_budget_raises_and_none_builds(budget):
    with pytest.raises(ValueError, match="budget"):
        Campaign("a", truthful(), budget=budget, base_value=5)

    campaign = Campaign("a", truthful(), budget=None, base_value=5, budget_fraction=2)
    assert campaign.budget is None
    assert campaign.budget_fraction == 2.0
    assert type(campaign.budget_fraction) is float


def test_pid_pacing_rejects_an_unresolved_budget():
    campaign = Campaign("a", truthful(), budget=None, base_value=5)
    with pytest.raises(ValueError, match="PidPacing needs a number budget"):
        PidPacing().on_episode_start(campaign)


def test_budget_defaults_to_none_and_fields_are_keyword_only():
    assert Campaign("a", truthful(), base_value=5).budget is None
    with pytest.raises(TypeError, match="positional"):
        Campaign("a", truthful(), 3, 5)  # pyright: ignore[reportCallIssue]  Reason: the test checks the runtime error for positional fields.


def test_fraction_scales_a_set_budget():
    scaled = run(
        config_of(Campaign("p", paced(), base_value=5, budget=5, budget_fraction=0.5)),
        progress_bar=False,
        seed=SEED,
    )
    plain = run(
        config_of(Campaign("p", paced(), base_value=5, budget=2.5)),
        progress_bar=False,
        seed=SEED,
    )
    assert scaled.budgets.tolist() == [2.5]
    assert_same_log(scaled, plain)
    assert (scaled.price.sum(axis=(1, 2)) <= 2.5).all()


@pytest.mark.parametrize("budget", [3.0, None])
def test_budgets_array_is_read_only(budget):
    log = run(
        config_of(
            Campaign("a", truthful(), base_value=5, budget=budget),
            Campaign("b", truthful(), base_value=5, budget=4),
        ),
        progress_bar=False,
        seed=SEED,
    )
    assert log.budgets.shape == (2,)
    assert log.budgets.dtype == np.float64
    assert not log.budgets.flags.writeable


@pytest.mark.parametrize("budget", [4.0, None])
def test_bidder_sees_the_effective_budget(probes, budget):
    campaign = Campaign(
        "p",
        Stack(OracleValue(), Probe()),
        base_value=5,
        budget=budget,
        budget_fraction=0.5,
    )
    log = run(config_of(campaign), progress_bar=False, seed=SEED)
    (probe,) = probes
    effective = log.budgets[0]
    if budget is not None:
        assert effective == 2.0

    assert [c.budget for c in probe.campaigns] == [effective] * 3
    assert [c.budget_fraction for c in probe.campaigns] == [1.0] * 3
    assert probe.first_budget_left == [effective] * 3


@pytest.mark.parametrize("competing", [True, False])
def test_market_cost_matches_the_log(competing):
    log = run(
        config_of(
            Campaign("a", truthful(), base_value=5, budget_fraction=0.3),
            Campaign("b", paced(), base_value=5),
            competing=competing,
            auction=SecondPrice(reserve=0.01),
        ),
        progress_bar=False,
        seed=SEED,
    )
    assert log.budgets[0] == pytest.approx(0.3 * market_cost(log, 0), rel=1e-12)
    assert log.budgets[1] == pytest.approx(market_cost(log, 1), rel=1e-12)


@pytest.mark.parametrize("auction", [FirstPrice(), SecondPrice(reserve=0.02)])
def test_market_budget_run_equals_its_twin(auction):
    config = config_of(
        Campaign("t", truthful(), base_value=5, budget_fraction=0.5),
        Campaign("p", paced(), base_value=5, budget_fraction=0.25),
        Campaign("s", paced(), base_value=4, budget=3),
        auction=auction,
    )
    log = run(config, progress_bar=False, seed=SEED)
    assert_same_log(log, run(twin(config, log), progress_bar=False, seed=SEED))


def test_market_cost_counts_only_own_segments():
    market = Segmented((Segment("news", 0.7), Segment("sports", 0.3)), base=PRICED)
    log = run(
        config_of(
            Campaign("news", truthful(), base_value=5, segments=["news"]),
            Campaign("all", truthful(), base_value=5),
            market=market,
            # Shared rates give both campaigns one value per request, so only segments differ.
            competing=False,
        ),
        progress_bar=False,
        seed=SEED,
    )
    assert market_cost(log, 0) < market_cost(log, 1)
    assert log.budgets[0] == pytest.approx(market_cost(log, 0), rel=1e-12)
    assert log.budgets[1] == pytest.approx(market_cost(log, 1), rel=1e-12)


def test_market_cost_without_price_is_the_reserve():
    log = run(
        config_of(
            Campaign("a", truthful(), base_value=1, event="impression"),
            market=LogisticMarket(),
            auction=SecondPrice(reserve=0.05),
        ),
        progress_bar=False,
        seed=SEED,
    )
    assert log.budgets[0] == pytest.approx(0.05 * 200, rel=1e-12)


def test_full_fraction_is_the_unlimited_truthful_spend():
    sized = run(
        config_of(Campaign("t", truthful(), base_value=20)),
        progress_bar=False,
        seed=SEED,
    )
    unlimited = run(
        config_of(Campaign("t", truthful(), base_value=20, budget=1e9)),
        progress_bar=False,
        seed=SEED,
    )
    spend = unlimited.price.sum() / unlimited.config.episodes
    assert sized.budgets[0] == pytest.approx(spend, rel=1e-9)


class NewsOnly(Market):
    """Names a sports segment but never draws it."""

    @property
    @override
    def segment_names(self) -> tuple[str, ...]:
        return ("news", "sports")

    @override
    def draws(self, count: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        while True:
            yield MarketDraw(
                {}, pcvr=np.full(count, 0.1), segment="news", market_price=0.2
            )


@pytest.mark.parametrize(
    ("market", "segments", "base_value"),
    [(LogisticMarket(), None, 5), (NewsOnly(), ["sports"], 5), (PRICED, None, 1e-9)],
    ids=["free market", "segment never drawn", "nothing worth its cost"],
)
def test_zero_market_cost_raises(probes, market, segments, base_value):
    campaign = Campaign(
        "z", Stack(OracleValue(), Probe()), base_value=base_value, segments=segments
    )
    with pytest.raises(ValueError, match="campaign 'z'.*market cost is 0"):
        run(config_of(campaign, market=market), progress_bar=False, seed=SEED)

    assert all(probe.forward_calls == 0 for probe in probes)


def test_reports_and_metrics_use_the_effective_budget(probes):
    config = config_of(
        Campaign("p", Stack(OracleValue(), Probe()), base_value=5, budget_fraction=0.5)
    )
    log = run(config, progress_bar=False, seed=SEED)
    (probe,) = probes
    effective = log.budgets[0]
    assert [report.budget for report in probe.reports] == [effective] * 3
    twin_metrics = campaign_metrics(
        run(twin(config, log), progress_bar=False, seed=SEED)
    )
    for row, twin_row in zip(campaign_metrics(log), twin_metrics, strict=True):
        assert row.budget_used == row.spend / effective
        assert row.regret == twin_row.regret


@dataclass(frozen=True)
class CountingMarket(Market):
    """A priced market that counts how often the simulator asks for draws."""

    calls: list[int] = field(default_factory=list)

    @override
    def draws(self, count: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        self.calls.append(count)
        return PRICED.draws(count, rng)


@pytest.mark.parametrize(("budget", "calls"), [(3.0, 1), (None, 2)])
def test_replay_only_when_a_budget_is_none(budget, calls):
    market = CountingMarket()
    run(
        config_of(
            Campaign("a", truthful(), base_value=5, budget=budget),
            Campaign("b", truthful(), base_value=5, budget=3),
            market=market,
        ),
        progress_bar=False,
        seed=SEED,
    )
    assert len(market.calls) == calls


@dataclass(frozen=True)
class DriftingMarket(Market):
    """Breaks the draws contract: each call raises its price or its rate."""

    drifts: str
    calls: list[int] = field(default_factory=list)

    @override
    def draws(self, count: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        self.calls.append(count)
        rise = 0.1 * len(self.calls)
        price = rise if self.drifts == "price" else 0.1
        rate = rise if self.drifts == "rate" else 0.1
        while True:
            yield MarketDraw({}, pcvr=np.full(count, rate), market_price=price)


@pytest.mark.parametrize("drifts", ["price", "rate"])
def test_market_that_breaks_the_contract_raises(drifts):
    with pytest.raises(RuntimeError, match="Market.draws"):
        run(
            config_of(
                Campaign("a", truthful(), base_value=5), market=DriftingMarket(drifts)
            ),
            progress_bar=False,
            seed=SEED,
        )


@dataclass(frozen=True)
class FixedMarket(Market):
    """Every step has the same market price and floor."""

    price: float
    floor: float = 0.0

    @override
    def draws(self, count: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        while True:
            yield MarketDraw(
                {}, pcvr=np.full(count, 0.1), market_price=self.price, floor=self.floor
            )


@pytest.mark.parametrize(
    ("market", "base_value", "cost"),
    [(FixedMarket(0.1, floor=0.3), 1.0, 0.3), (FixedMarket(0.2), 0.2, 0.2)],
    ids=["floor above the price", "value equal to the cost"],
)
def test_request_cost_and_worth(market, base_value, cost):
    log = run(
        config_of(
            Campaign("a", truthful(), base_value=base_value, event="impression"),
            market=market,
        ),
        progress_bar=False,
        seed=SEED,
    )
    assert log.budgets[0] == pytest.approx(cost * 200, rel=1e-12)


def test_results_budget_column():
    config = config_of(
        Campaign("t", truthful(), base_value=5, budget_fraction=0.3),
        Campaign("p", paced(), base_value=5, budget=3),
    )
    results = run(config, seed=[0, 1], progress_bar=False)
    for seed in (0, 1):
        log = run(config, seed, progress_bar=False)
        rows = [row for row in results.rows if row["seed"] == seed]
        assert [row["budget"] for row in rows] == log.budgets.tolist()

    parallel = run(config, seed=[0, 1], workers=2, progress_bar=False)
    assert parallel.equals(results)
