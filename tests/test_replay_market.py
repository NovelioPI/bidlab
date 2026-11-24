import pickle
from typing import Any, ClassVar

import cloudpickle
import numpy as np
import polars as pl
import pytest
from scipy import stats

import bidlab as bl
from bidlab.markets import ReplayMarket

EPISODES = 2
STEPS = 6
ROWS = EPISODES * STEPS


class Spy(bl.Bidder):
    """Bids a fixed amount and records every input it gets."""

    instances: ClassVar[list["Spy"]] = []

    def __init__(self, bid=1.0):
        self.bid = bid
        self.requests: list[dict[str, Any]] = []

    def bind(self, rng):
        super().bind(rng)
        Spy.instances.append(self)

    def forward(self, **inputs):
        self.requests.append(inputs)
        return self.bid


def log_table(rows=ROWS):
    """A small log in which every column differs from row to row."""
    index = np.arange(rows)
    return pl.DataFrame(
        {
            "pctr": 0.1 + 0.05 * (index % 5),
            "pcvr": 0.2 + 0.03 * (index % 7),
            "price": 0.5 + 0.1 * index,
            "floor": 0.01 * index,
            "segment": ["web" if i % 3 else "app" for i in index],
            "slot": [f"s{i % 4}" for i in index],
            "score": index * 1.5,
            "vec": [[float(i), -float(i)] for i in index],
        },
        schema_overrides={"vec": pl.Array(pl.Float64, 2)},
    )


def full_market(table=None, **overrides):
    columns: dict[str, Any] = {
        "pctr": "pctr",
        "pcvr": "pcvr",
        "market_price": "price",
        "floor": "floor",
        "segment": "segment",
    }
    columns.update(overrides)
    return ReplayMarket(log_table() if table is None else table, **columns)


def replay(market, *bidders, episodes=EPISODES, steps=STEPS, seed=0, **settings):
    Spy.instances.clear()
    campaigns = [
        bl.Campaign(f"c{i}", bidder, budget=1000.0, base_value=2.0)
        for i, bidder in enumerate(bidders)
    ]
    config = bl.SimConfig(
        campaigns=campaigns, market=market, episodes=episodes, steps=steps, **settings
    )
    return bl.run(config, seed=seed, progress_bar=False)


def test_replay_logs_the_columns():
    table = log_table()
    market = full_market(table)
    log = replay(market, Spy())

    np.testing.assert_array_equal(
        log.market_price, table["price"].to_numpy().reshape(EPISODES, STEPS)
    )
    np.testing.assert_array_equal(
        log.floor, table["floor"].to_numpy().reshape(EPISODES, STEPS)
    )
    names = np.array(market.segment_names)
    np.testing.assert_array_equal(
        names[log.segment], np.array(table["segment"]).reshape(EPISODES, STEPS)
    )
    assert market.segment_names == ("app", "web")
    assert market.rows == ROWS


def test_true_values_follow_the_rates():
    table = log_table()
    log = replay(full_market(table), Spy(), Spy())

    expected = (2.0 * table["pctr"].to_numpy()) * table["pcvr"].to_numpy()
    for i in range(2):
        np.testing.assert_array_equal(
            log.true_values[..., i], expected.reshape(EPISODES, STEPS)
        )


class FeatureSpy(bl.Bidder):
    """Bids 1 and records its features input and each feature it names."""

    instances: ClassVar[list["FeatureSpy"]] = []

    def __init__(self):
        self.requests: list[dict[str, Any]] = []

    def bind(self, rng):
        super().bind(rng)
        FeatureSpy.instances.append(self)

    def forward(self, features, slot, score, vec):
        self.requests.append(
            {"features": features, "slot": slot, "score": score, "vec": vec}
        )
        return 1.0


def test_features_reach_the_bidder():
    table = log_table()
    FeatureSpy.instances.clear()
    replay(full_market(table, features=["slot", "score", "vec"]), FeatureSpy())

    requests = FeatureSpy.instances[0].requests
    assert len(requests) == ROWS
    for row, request in enumerate(requests):
        assert request["slot"] == f"s{row % 4}"
        assert isinstance(request["slot"], str)
        assert isinstance(request["score"], float)
        assert request["score"] == row * 1.5
        np.testing.assert_array_equal(request["vec"], [row, -row])
        assert not request["vec"].flags.writeable
        assert request["features"]["slot"] == request["slot"]


def test_polars_pandas_and_dict_agree():
    pd = pytest.importorskip("pandas")
    table = log_table()
    plain = {name: table[name].to_numpy() for name in table.columns}
    flat = pd.DataFrame({name: plain[name] for name in plain if name != "vec"})
    logs = [
        replay(full_market(kind, features=["slot", "score"]), Spy(0.9))
        for kind in (table, flat, plain)
    ]
    for log in logs[1:]:
        for field in (
            "market_price",
            "floor",
            "segment",
            "true_values",
            "won",
            "price",
        ):
            np.testing.assert_array_equal(getattr(log, field), getattr(logs[0], field))


@pytest.mark.parametrize("kind", ["polars", "dict"])
def test_missing_column_raises(kind):
    table = log_table()
    if kind == "dict":
        table = {name: table[name].to_numpy() for name in table.columns}

    with pytest.raises(KeyError, match=r"'nope'.*pcvr"):
        ReplayMarket(table, pcvr="pcvr", floor="nope")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("pcvr", 1.5, "pcvr"),
        ("pcvr", float("nan"), "pcvr"),
        ("pctr", -0.1, "pctr"),
        ("price", -1.0, "price"),
        ("price", float("inf"), "price"),
        ("floor", float("nan"), "floor"),
    ],
)
def test_bad_values_raise(column, value, message):
    table = log_table().with_columns(
        pl.when(pl.int_range(pl.len()) == 4)
        .then(pl.lit(value))
        .otherwise(pl.col(column))
        .alias(column)
    )
    with pytest.raises(ValueError, match=rf"{message}.*row 4"):
        full_market(table)


def test_unequal_lengths_raise():
    table = {"pcvr": np.full(5, 0.1), "price": np.ones(4)}
    with pytest.raises(ValueError, match="length"):
        ReplayMarket(table, market_price="price")


def test_empty_table_raises():
    with pytest.raises(ValueError, match="no rows"):
        ReplayMarket({"pcvr": np.array([])})


def test_object_column_raises():
    table = log_table().with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(None)
        .otherwise(pl.col("slot"))
        .alias("slot")
    )
    with pytest.raises(TypeError, match="slot"):
        full_market(table, features=["slot"])


def test_non_number_rate_raises():
    with pytest.raises(TypeError, match="pcvr"):
        ReplayMarket({"pcvr": np.array(["a", "b"])})


def test_market_keeps_no_table():
    table = log_table()
    market = full_market(table, features=["slot", "score", "vec"])
    before = replay(market, Spy())
    table[0, "price"] = 99.0
    after = replay(market, Spy())

    np.testing.assert_array_equal(before.market_price, after.market_price)
    assert b"polars" not in cloudpickle.dumps(market)
    assert b"polars" not in pickle.dumps(market)


def test_known_prices_do_not_depend_on_the_seed():
    market = full_market()
    first = replay(market, Spy(), seed=0)
    second = replay(market, Spy(), seed=1)
    for field in ("market_price", "floor", "segment", "true_values"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_outcome_rates_replay_the_log():
    index = np.arange(ROWS)
    table = {"pctr": (index % 2).astype(float), "pcvr": (index % 3 == 0).astype(float)}
    log = replay(ReplayMarket(table, pctr="pctr"), Spy(5.0))

    assert log.won.all()
    np.testing.assert_array_equal(
        log.clicked[..., 0], table["pctr"].reshape(EPISODES, STEPS).astype(bool)
    )
    np.testing.assert_array_equal(
        log.converted[..., 0],
        (table["pctr"] * table["pcvr"]).reshape(EPISODES, STEPS).astype(bool),
    )


def test_rate_list_gives_each_campaign_its_column():
    index = np.arange(ROWS)
    table = {"a": 0.01 * index, "b": 0.02 * index, "price": np.full(ROWS, 0.1)}
    market = ReplayMarket(table, pcvr=["a", "b"], market_price="price")
    log = replay(market, Spy(), Spy())

    np.testing.assert_array_equal(
        log.true_values[..., 0], (2.0 * table["a"]).reshape(EPISODES, STEPS)
    )
    np.testing.assert_array_equal(
        log.true_values[..., 1], (2.0 * table["b"]).reshape(EPISODES, STEPS)
    )


def test_market_budget_on_a_replay():
    market = full_market()
    campaign = bl.Campaign(
        "c", bl.Stack(bl.OracleValue(), bl.Linear()), base_value=50.0
    )
    config = bl.SimConfig(
        campaigns=[campaign], market=market, episodes=EPISODES, steps=STEPS
    )
    budgets = [bl.run(config, seed=seed, progress_bar=False).budgets for seed in (0, 1)]

    assert budgets[0][0] > 0
    np.testing.assert_array_equal(budgets[0], budgets[1])


def test_not_competing_equals_solo_runs():
    market = full_market()
    bidders = [
        bl.Stack(bl.OracleValue(), bl.Linear()),
        bl.Stack(bl.OracleValue(), bl.PidPacing()),
    ]
    together = replay(market, *bidders, competing=False)
    for i, bidder in enumerate(bidders):
        alone = replay(market, bidder)
        for field in ("bids", "true_values", "won", "price", "clicked", "converted"):
            np.testing.assert_array_equal(
                getattr(together, field)[..., i], getattr(alone, field)[..., 0]
            )


def test_segments_and_targeting():
    market = full_market()
    campaign = bl.Campaign(
        "app_only", Spy(5.0), budget=1000.0, base_value=2.0, segments=["app"]
    )
    other = bl.Campaign("all", Spy(0.1), budget=1000.0, base_value=2.0)
    config = bl.SimConfig(
        campaigns=[campaign, other], market=market, episodes=EPISODES, steps=STEPS
    )
    log = bl.run(config, progress_bar=False)
    app = np.array(market.segment_names)[log.segment] == "app"

    assert log.won[..., 0][app].any()
    assert not log.won[..., 0][~app].any()
    assert (log.bids[..., 0][~app] == 0).all()

    results = bl.run(config, seed=[0, 1], progress_bar=False)
    assert results.segment_rows is not None
    assert len(results.segment_rows) == 2 * 2 * 2


@pytest.mark.parametrize("rows", [ROWS - 1, ROWS + 1])
def test_row_count_must_match(rows):
    with pytest.raises(ValueError, match=rf"{rows} rows.*episodes.*steps"):
        replay(full_market(log_table(rows)), Spy())


@pytest.mark.parametrize(("campaigns", "competing"), [(3, True), (2, False)])
def test_rate_list_must_match_campaigns(campaigns, competing):
    index = np.arange(ROWS)
    table = {"a": 0.01 * index, "b": 0.02 * index, "price": np.full(ROWS, 0.1)}
    market = ReplayMarket(table, pcvr=["a", "b"], market_price="price")
    message = "competing=False" if not competing else "2 rate columns.*3 campaigns"
    with pytest.raises(ValueError, match=message):
        replay(market, *[Spy() for _ in range(campaigns)], competing=competing)


def test_workers_agree():
    market = full_market(features=["slot", "score", "vec"])
    campaign = bl.Campaign(
        "c", bl.Stack(bl.OracleValue(), bl.Linear()), budget=5.0, base_value=2.0
    )
    config = bl.SimConfig(
        campaigns=[campaign], market=market, episodes=EPISODES, steps=STEPS
    )
    one = bl.run(config, seed=[0, 1], workers=1, progress_bar=False)
    two = bl.run(config, seed=[0, 1], workers=2, progress_bar=False)
    assert one.equals(two)


def test_replay_market_is_in_markets():
    assert bl.markets.ReplayMarket is ReplayMarket


HORIZON = 24 * 3600.0


def timed_table(times=None):
    if times is None:
        # Uneven gaps, so the step times differ from step * horizon / steps.
        day = np.array([0.0, 30.0, 4000.0, 4000.0, 9000.0, 80000.0])
        times = np.concatenate([day, day + 100.0])
    return log_table().with_columns(time=pl.Series(times))


class TimeSpy(Spy):
    """Records the step time of each update_batch and each result it gets."""

    def __init__(self, bid=1.0):
        super().__init__(bid)
        self.now = -1.0
        self.batches: list[float] = []
        self.results: list[tuple[float, bl.BidResult]] = []

    def forward(self, **inputs):
        self.now = inputs["time"]
        return super().forward(**inputs)

    def update_batch(self, bid_results):
        self.batches.append(self.now)
        self.results.extend((self.now, result) for result in bid_results)


def test_row_times_reach_the_bidder():
    table = timed_table()
    replay(full_market(table, time="time"), Spy(), start_hour=5.0)

    times = table["time"].to_numpy()
    requests = Spy.instances[0].requests
    assert [request["time"] for request in requests] == times.tolist()
    for request, time in zip(requests, times, strict=True):
        assert request["time_left"] == HORIZON - time
        assert request["hour_of_day"] == (5.0 + time / 3600) % 24


def test_reports_follow_row_times():
    table = timed_table()
    replay(
        full_market(table, time="time"),
        TimeSpy(),
        feedback=bl.Feedback(report_delay=1000.0),
    )

    times = table["time"].to_numpy().reshape(EPISODES, STEPS)
    spy = Spy.instances[0]
    assert isinstance(spy, TimeSpy)
    results = iter(spy.results)
    for episode in times:
        for time in episode:
            delivered_at, result = next(results)
            assert result.time == time
            due = [t for t in episode if t >= time + 1000.0]
            # Without a later step at or after the due time, the episode-end flush delivers it.
            assert delivered_at == (due[0] if due else episode[-1])


def test_interval_follows_row_times():
    table = timed_table()
    replay(
        full_market(table, time="time"),
        TimeSpy(),
        feedback=bl.Feedback(interval=3600.0),
    )

    spy = Spy.instances[0]
    assert isinstance(spy, TimeSpy)
    expected = []
    for episode in table["time"].to_numpy().reshape(EPISODES, STEPS):
        hours = np.floor(episode / 3600.0)
        delivers = np.flatnonzero(hours[1:] > hours[:-1]) + 1
        expected.extend(episode[delivers])
        # The episode-end flush runs only for reports still waiting.
        if delivers[-1] != STEPS - 1:
            expected.append(episode[-1])
    assert len(expected) > EPISODES
    assert spy.batches == expected


@pytest.mark.parametrize(
    ("row", "value"),
    [(3, 1.0), (2, float("nan")), (4, float("inf")), (0, -1.0), (5, HORIZON)],
)
def test_bad_row_times_raise(row, value):
    times = timed_table()["time"].to_numpy().copy()
    times[row] = value
    with pytest.raises(ValueError, match=rf"time.*row {row}"):
        replay(full_market(timed_table(times), time="time"), Spy())


class TimedMarket(bl.Market):
    """Yields one draw per given time."""

    def __init__(self, times):
        self.times = times

    def draws(self, count, rng):
        for time in self.times:
            yield bl.markets.MarketDraw({}, pcvr=np.full(count, 0.1), time=time)


@pytest.mark.parametrize(
    ("times", "step"),
    [([10.0, 5.0, 20.0], 1), ([10.0, HORIZON, 20.0], 1), ([-1.0, 0.0, 1.0], 0)],
)
def test_custom_market_bad_times_raise(times, step):
    with pytest.raises(
        ValueError, match=rf"TimedMarket at episode 0, step {step}.*time"
    ):
        replay(TimedMarket(times), Spy(), episodes=1, steps=3)


def test_time_starts_again_each_episode():
    day = np.array([0.0, 30.0, 4000.0, 4000.0, 9000.0, 80000.0])
    table = timed_table(np.concatenate([day, day + 3600.0]))
    replay(full_market(table, time="time"), Spy())

    times = [request["time"] for request in Spy.instances[0].requests]
    assert times == table["time"].to_list()


LOGNORMAL = stats.lognorm(s=1.0, scale=1.0)


def bounded_table():
    """Wins (floor, bid), losses (bid, inf), and one exact row, as in a first-price log."""
    table = log_table()
    low = np.array([0.1, 0.5, 0.2, 2.0, 0.0, 1.5] * 2)
    high = np.array([0.4, np.inf, 0.2, np.inf, 0.3, 1.6] * 2)
    return table.with_columns(low=pl.Series(low), high=pl.Series(high)).drop("price")


def bounded_market(table=None, **overrides):
    columns: dict[str, Any] = {
        "pcvr": "pcvr",
        "price_low": "low",
        "price_high": "high",
        "price_distribution": LOGNORMAL,
    }
    columns.update(overrides)
    return ReplayMarket(bounded_table() if table is None else table, **columns)


def test_filled_prices_lie_in_bounds():
    table = bounded_table()
    log = replay(bounded_market(table), Spy())
    prices = log.market_price.ravel()
    low, high = table["low"].to_numpy(), table["high"].to_numpy()

    assert np.isfinite(prices).all()
    assert ((prices >= low) & (prices <= high)).all()
    exact = low == high
    np.testing.assert_array_equal(prices[exact], low[exact])


def test_filled_prices_follow_the_cut_distribution():
    rows = 20_000
    table = {
        "pcvr": np.full(rows, 0.1),
        "low": np.ones(rows),
        "high": np.full(rows, 3.0),
    }
    log = replay(bounded_market(table), Spy(0.0), episodes=1, steps=rows)
    prices = log.market_price.ravel()
    cut = LOGNORMAL.cdf(1.0), LOGNORMAL.cdf(3.0)

    def cut_cdf(x):
        return (LOGNORMAL.cdf(x) - cut[0]) / (cut[1] - cut[0])

    assert stats.kstest(prices, cut_cdf).pvalue > 0.01
    assert stats.kstest(prices, LOGNORMAL.cdf).pvalue < 0.01


def test_filled_prices_follow_the_seed():
    market = bounded_market()
    first, again, other = (replay(market, Spy(), seed=seed) for seed in (0, 0, 1))
    np.testing.assert_array_equal(first.market_price, again.market_price)
    table = bounded_table()
    open_rows = (table["low"] < table["high"]).to_numpy().reshape(EPISODES, STEPS)
    assert (first.market_price[open_rows] != other.market_price[open_rows]).any()
    for field in ("floor", "segment", "true_values"):
        np.testing.assert_array_equal(getattr(first, field), getattr(other, field))


def test_market_budget_with_filled_prices():
    campaign = bl.Campaign(
        "c", bl.Stack(bl.OracleValue(), bl.Linear()), base_value=50.0
    )
    config = bl.SimConfig(
        campaigns=[campaign], market=bounded_market(), episodes=EPISODES, steps=STEPS
    )
    assert bl.run(config, progress_bar=False).budgets[0] > 0


class NoCdf:
    def ppf(self, q):
        return q


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"row": 1, "low": 3.0, "high": 2.0}, ValueError, "low.*row 1"),
        ({"row": 2, "low": -1.0}, ValueError, "low.*row 2"),
        ({"row": 3, "low": float("nan")}, ValueError, "low.*row 3"),
        ({"row": 4, "low": float("inf")}, ValueError, "low.*row 4"),
        ({"row": 5, "high": float("nan")}, ValueError, "high.*row 5"),
        ({"market_price": "low"}, ValueError, "market_price.*price_low"),
        ({"price_distribution": None}, ValueError, "price_distribution"),
        ({"price_high": None}, ValueError, "price_high"),
        ({"price_distribution": NoCdf()}, TypeError, "cdf"),
    ],
)
def test_bad_bounds_raise(change, error, message):
    table = bounded_table()
    row = change.pop("row", None)
    for column in ("low", "high"):
        if column in change:
            values = table[column].to_numpy().copy()
            values[row] = change.pop(column)
            table = table.with_columns(pl.Series(column, values))

    with pytest.raises(error, match=message):
        bounded_market(table, **change)


def test_bounds_without_probability_raise():
    table = bounded_table()
    low = table["low"].to_numpy().copy()
    high = table["high"].to_numpy().copy()
    low[7], high[7] = 1e6, 2e6
    table = table.with_columns(pl.Series("low", low), pl.Series("high", high))
    with pytest.raises(ValueError, match="row 7"):
        bounded_market(table)
