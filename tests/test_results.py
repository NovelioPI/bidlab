import csv
import dataclasses
import sys
import threading

import numpy as np
import pytest
from scipy import stats

from bidlab.auctions import FirstPrice
from bidlab.bidders import Bidder, Linear, OracleValue, PidPacing, Stack
from bidlab.config import Campaign, SimConfig
from bidlab.markets import LogisticMarket, Market, MarketDraw, Segment, Segmented
from bidlab.results import Results
from bidlab.runner import COLUMNS, SEGMENT_COLUMNS, run_many
from bidlab.simulator import run
from bidlab.types import Event


def segments_table(results: Results) -> Results:
    assert results.segment_rows is not None
    return results.segment_rows


def config(*extra):
    campaigns = [
        Campaign("t", Stack(OracleValue(), Linear()), budget=20.0, base_value=5.0),
        Campaign("p", Stack(OracleValue(), PidPacing()), budget=8.0, base_value=5.0),
        *extra,
    ]
    return SimConfig(campaigns=campaigns, episodes=2, steps=100)


def test_rows_ordered_by_seed_then_bidder():
    results = run_many(config(), seeds=range(3))
    assert len(results) == 3 * 2
    assert [(row["seed"], row["campaign"]) for row in results.rows] == [
        (0, "t"),
        (0, "p"),
        (1, "t"),
        (1, "p"),
        (2, "t"),
        (2, "p"),
    ]
    assert all(tuple(row) == COLUMNS for row in results.rows)
    assert results.columns == COLUMNS


def test_workers_give_same_rows():
    serial = run_many(config(), seeds=range(4))
    parallel = run_many(config(), seeds=range(4), workers=2)
    assert parallel.equals(serial)


def test_lambdas_and_closures_run_in_parallel():
    fraction = 0.7

    class Closure(Bidder):
        def forward(self, value):
            return fraction * value

    extra = Campaign(
        "closure",
        Stack(OracleValue(), Closure()),
        budget=20.0,
        base_value=5.0,
    )
    serial = run_many(config(extra), seeds=range(3))
    parallel = run_many(config(extra), seeds=range(3), workers=2)
    assert parallel.equals(serial)


def test_unpicklable_spec_fails_before_runs(monkeypatch):
    class Locked(Bidder):
        def __init__(self):
            self.lock = threading.Lock()

        def forward(self, value):
            return value

    pools = []
    monkeypatch.setattr(
        "bidlab.runner.ProcessPoolExecutor", lambda *args, **kwargs: pools.append(1)
    )
    extra = Campaign("locked", Locked(), budget=20.0, base_value=5.0)
    with pytest.raises(TypeError, match="'locked'.*workers=1"):
        run_many(config(extra), seeds=range(2), workers=2)

    assert pools == []


def hand_results(spends):
    rows = []
    for seed, spend in enumerate(spends):
        row: dict[str, str | float] = {column: 0.0 for column in COLUMNS}
        row.update(seed=seed, campaign="a", spend=spend)
        rows.append(row)

    return Results(columns=COLUMNS, rows=tuple(rows))


def test_summary_interval_matches_student_t():
    summary = hand_results([1.0, 2.0, 3.0]).summary()
    # mean 2, sample std 1, n 3: half-width = t(0.975, 2) / sqrt(3)
    half_width = stats.t.ppf(0.975, 2) / np.sqrt(3)
    assert f"2.000 ± {half_width:.3f}" in summary
    assert summary.splitlines()[0].split()[:2] == ["campaign", "spend"]


def test_summary_one_seed_shows_na():
    summary = hand_results([4.0]).summary()
    row = summary.splitlines()[1]
    assert "4.000 ± n/a" in row
    assert row.count("n/a") == 5


def test_save_csv_round_trip(tmp_path):
    results = run_many(config(), seeds=range(2))
    path = tmp_path / "runs.csv"
    results.save_csv(path)
    with path.open() as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames is not None

    assert tuple(reader.fieldnames) == COLUMNS
    assert len(rows) == len(results)
    assert float(rows[1]["spend"]) == pytest.approx(results.rows[1]["spend"])


def test_to_pandas_matches_rows():
    pandas = pytest.importorskip("pandas")
    results = run_many(config(), seeds=range(2))
    frame = results.to_pandas()
    assert isinstance(frame, pandas.DataFrame)
    assert tuple(frame.columns) == COLUMNS
    # to_pandas exports rows only, so the logs come from the original.
    rows = tuple(frame.to_dict("records"))
    round_trip = Results(columns=COLUMNS, rows=rows, logged=results.logged)
    assert round_trip.equals(results)


def test_to_pandas_without_pandas_names_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match=r"pip install bidlab\[pandas\]"):
        hand_results([1.0]).to_pandas()


def test_workers_below_one_rejected():
    with pytest.raises(ValueError, match="workers"):
        run_many(config(), seeds=range(1), workers=0)


def test_nothing_printed(capsys, tmp_path):
    run(config())
    results = run_many(config(), seeds=range(2))
    results.summary()
    results.save_csv(tmp_path / "out.csv")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_equals_treats_nan_as_equal():
    first = run_many(config(), seeds=range(2))
    second = run_many(config(), seeds=range(2))
    assert first.rows != second.rows  # NaN cells make plain == fail
    assert first.equals(second)
    assert not first.equals(run_many(config(), seeds=range(1, 3)))
    assert not first.equals(Results(columns=COLUMNS[:-1], rows=first.rows))


def test_results_rows_are_read_only():
    results = run_many(config(), seeds=range(1))
    with pytest.raises(TypeError):
        results.rows[0]["spend"] = 0.0  # pyright: ignore[reportIndexIssue]  Reason: the test checks that rows are read-only.


def test_run_log_arrays_are_read_only():
    log = run(config())
    for array in (log.bids, log.true_values, log.winner, log.price, log.converted):
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 0


def test_custom_market_same_rows_in_parallel():
    class LowRates(Market):
        def draws(self, count, rng):
            while True:
                yield MarketDraw({}, pcvr=rng.uniform(0.0, 0.2, size=count))

    cfg = dataclasses.replace(config(), market=LowRates())
    serial = run_many(cfg, seeds=range(3))
    parallel = run_many(cfg, seeds=range(3), workers=2)
    assert parallel.equals(serial)
    assert segments_table(parallel).equals(segments_table(serial))


def segmented_config():
    market = Segmented((Segment("news", 0.6), Segment("sports", 0.4, lift=1.0)))
    targeted = Campaign(
        "n",
        Stack(OracleValue(), Linear()),
        budget=20.0,
        base_value=5.0,
        segments=("news",),
    )
    return SimConfig(
        campaigns=[*config().campaigns, targeted], episodes=2, steps=100, market=market
    )


def test_segments_table_columns():
    results = run_many(segmented_config(), seeds=range(2))
    assert results.columns == COLUMNS
    assert isinstance(results.segment_rows, Results)
    assert results.segment_rows.columns == SEGMENT_COLUMNS
    assert SEGMENT_COLUMNS == (
        "seed",
        "campaign",
        "segment",
        "spend",
        "wins",
        "clicks",
        "conversions",
        "surplus",
    )
    assert len(results.segment_rows) == 2 * 3 * 2


def test_segment_sums_match_main_table():
    results = run_many(segmented_config(), seeds=range(3))
    for row in results.rows:
        own = [
            segment
            for segment in segments_table(results).rows
            if segment["seed"] == row["seed"] and segment["campaign"] == row["campaign"]
        ]
        for column in ("spend", "wins", "conversions", "surplus"):
            total = sum(float(segment[column]) for segment in own)
            assert total == pytest.approx(row[column], abs=1e-9)


def test_targeted_bidder_has_nothing_in_other_segments():
    results = run_many(segmented_config(), seeds=range(2))
    sports = [
        row
        for row in segments_table(results).rows
        if row["campaign"] == "n" and row["segment"] == "sports"
    ]
    assert sports and all(row["wins"] == 0 and row["spend"] == 0 for row in sports)


def test_unsegmented_market_has_all_segment():
    results = run_many(config(), seeds=range(2))
    assert [
        (row["seed"], row["campaign"], row["segment"])
        for row in segments_table(results).rows
    ] == [
        (0, "t", "all"),
        (0, "p", "all"),
        (1, "t", "all"),
        (1, "p", "all"),
    ]


def test_segments_summary_and_csv(tmp_path):
    segments = segments_table(run_many(segmented_config(), seeds=range(3)))
    lines = segments.summary().splitlines()
    assert lines[0].split() == [
        "campaign",
        "segment",
        "spend",
        "wins",
        "clicks",
        "conversions",
        "surplus",
    ]
    labels = [tuple(line.split()[:2]) for line in lines[1:]]
    assert labels == [
        ("t", "news"),
        ("t", "sports"),
        ("p", "news"),
        ("p", "sports"),
        ("n", "news"),
        ("n", "sports"),
    ]
    path = tmp_path / "segments.csv"
    segments.save_csv(path)
    with path.open() as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames is not None
    assert tuple(reader.fieldnames) == SEGMENT_COLUMNS
    assert len(rows) == len(segments)


def test_segments_to_pandas():
    pytest.importorskip("pandas")
    segments = segments_table(run_many(segmented_config(), seeds=range(2)))
    frame = segments.to_pandas()
    assert tuple(frame.columns) == SEGMENT_COLUMNS
    round_trip = Results(columns=SEGMENT_COLUMNS, rows=tuple(frame.to_dict("records")))
    assert round_trip.equals(segments)


def test_custom_market_segments_reach_segments_table():
    class DayAndNight(Market):
        @property
        def segment_names(self):
            return ("day", "night")

        def draws(self, count, rng):
            while True:
                name = "day" if rng.random() < 0.6 else "night"
                yield MarketDraw(
                    {}, pcvr=rng.uniform(0.0, 0.2, size=count), segment=name
                )

    night = Campaign(
        "n",
        Stack(OracleValue(), Linear()),
        budget=20.0,
        base_value=5.0,
        segments=("night",),
    )
    cfg = SimConfig(
        campaigns=[*config().campaigns, night],
        episodes=2,
        steps=100,
        market=DayAndNight(),
    )
    segments = segments_table(run_many(cfg, seeds=range(2)))
    assert {row["segment"] for row in segments.rows} == {"day", "night"}
    day_rows = [
        r for r in segments.rows if r["campaign"] == "n" and r["segment"] == "day"
    ]
    assert day_rows and all(row["wins"] == 0 for row in day_rows)


def test_noisy_parallel_equals_serial():
    noisy = Campaign(
        "n", Stack(OracleValue(noise=0.5), Linear()), budget=20.0, base_value=5.0
    )
    cfg = config(noisy)
    serial = run_many(cfg, seeds=range(3))
    parallel = run_many(cfg, seeds=range(3), workers=2)
    assert parallel.equals(serial)
    exact = Campaign("n", Stack(OracleValue(), Linear()), budget=20.0, base_value=5.0)
    assert not serial.equals(run_many(config(exact), seeds=range(3)))


def test_no_reward_columns():
    results = run_many(segmented_config(), seeds=range(2))
    segments = segments_table(results)
    for columns in (results.columns, segments.columns, results.summary().split()):
        assert "reward" not in columns and "episode_reward" not in columns
    assert {"surplus", "cpa_score"} <= set(results.columns)


def test_clicks_column():
    plain = run_many(config(), seeds=range(2))
    assert plain.columns.index("clicks") == plain.columns.index("wins") + 1
    assert all(row["clicks"] == row["wins"] for row in plain.rows)

    clicking = dataclasses.replace(config(), market=LogisticMarket(ctr_bias_mean=-1.0))
    results = run_many(clicking, seeds=range(2))
    for row in results.rows:
        log = run(dataclasses.replace(clicking, seed=row["seed"]))
        index = [c.name for c in clicking.campaigns].index(str(row["campaign"]))
        clicked_wins = (log.clicked & (log.winner == index)).sum(axis=1)
        assert row["clicks"] == pytest.approx(clicked_wins.mean())
    assert any(float(row["clicks"]) < float(row["wins"]) for row in results.rows)


def test_segment_clicks_sum_to_main():
    segmented = segmented_config()
    assert isinstance(segmented.market, Segmented)
    base = LogisticMarket(ctr_bias_mean=-1.0)
    clicking = dataclasses.replace(
        segmented, market=dataclasses.replace(segmented.market, base=base)
    )
    results = run_many(clicking, seeds=range(3))
    for row in results.rows:
        own = [
            float(segment["clicks"])
            for segment in segments_table(results).rows
            if segment["seed"] == row["seed"] and segment["campaign"] == row["campaign"]
        ]
        assert len(own) == 2
        assert sum(own) == pytest.approx(row["clicks"], abs=1e-9)
    assert any(float(row["clicks"]) < float(row["wins"]) for row in results.rows)


# Close expected values, so each campaign wins some auctions.
EVENT_VALUES = {Event.IMPRESSION: 0.1, Event.CLICK: 0.4, Event.CONVERSION: 2.0}
EVENT_COUNT = {"impression": "wins", "click": "clicks", "conversion": "conversions"}


def event_config(market=None):
    campaigns = [
        Campaign(
            event,
            Stack(OracleValue(), Linear()),
            budget=20.0,
            base_value=value,
            event=event,
        )
        for event, value in EVENT_VALUES.items()
    ]
    market = market or LogisticMarket(ctr_bias_mean=-1.0, cvr_bias_mean=-1.0)
    return SimConfig(campaigns=campaigns, episodes=3, steps=200, market=market)


def episode_counts(log, index, steps=None):
    """Per-episode counts for one campaign, over all steps or the given ones."""
    won = log.winner == index
    if steps is not None:
        won = won & steps
    return {
        "wins": won.sum(axis=1),
        "clicks": (log.clicked & won).sum(axis=1),
        "conversions": (log.converted & won).sum(axis=1),
        "spend": np.where(won, log.price, 0.0).sum(axis=1),
    }


def rows_with_counts(cfg, seeds=range(2)):
    results = run_many(cfg, seeds=seeds)
    names = [campaign.name for campaign in cfg.campaigns]
    for row in results.rows:
        log = run(dataclasses.replace(cfg, seed=int(row["seed"])))
        name = str(row["campaign"])
        yield row, name, episode_counts(log, names.index(name))


def test_surplus_follows_event():
    for row, name, counts in rows_with_counts(event_config()):
        value = counts[EVENT_COUNT[name]] * EVENT_VALUES[Event(name)]
        assert row["realized_value"] == pytest.approx(value.mean())
        assert row["surplus"] == pytest.approx((value - counts["spend"]).mean())
        assert float(row["wins"]) > float(row["clicks"]) > float(row["conversions"]) > 0


def test_segment_surplus_follows_event():
    base = LogisticMarket(ctr_bias_mean=-1.0, cvr_bias_mean=-1.0)
    market = Segmented((Segment("news", 0.6), Segment("sports", 0.4, lift=1.0)), base)
    cfg = event_config(market)
    results = run_many(cfg, seeds=range(2))
    names = [campaign.name for campaign in cfg.campaigns]
    for row in segments_table(results).rows:
        log = run(dataclasses.replace(cfg, seed=int(row["seed"])))
        name = str(row["campaign"])
        here = log.segment == market.segment_names.index(str(row["segment"]))
        counts = episode_counts(log, names.index(name), here)
        value = counts[EVENT_COUNT[name]] * EVENT_VALUES[Event(name)]
        assert row["surplus"] == pytest.approx((value - counts["spend"]).mean())


def test_ctr_and_cpc_columns():
    cfg = event_config()
    columns = run_many(cfg, seeds=range(1)).columns
    assert columns.index("ctr") == columns.index("budget_used") + 1
    assert columns.index("cpc") == columns.index("ctr") + 1
    assert columns.index("cpa") == columns.index("cpc") + 1
    for row, _, counts in rows_with_counts(cfg):
        assert row["ctr"] == pytest.approx((counts["clicks"] / counts["wins"]).mean())
        assert row["cpc"] == pytest.approx((counts["spend"] / counts["clicks"]).mean())


def test_cpa_counts_conversions_for_every_event():
    target = 1.0
    cfg = event_config()
    cfg = dataclasses.replace(
        cfg,
        campaigns=[dataclasses.replace(c, cpa_target=target) for c in cfg.campaigns],
    )
    violations = []
    for row, name, counts in rows_with_counts(cfg):
        spend, conversions = counts["spend"], counts["conversions"]
        assert (spend > 0).all()
        assert (conversions > 0).all()
        cpa = spend / conversions
        value = counts[EVENT_COUNT[name]] * EVENT_VALUES[Event(name)]
        violation = np.maximum(0.0, cpa / target - 1.0)
        score = value * np.minimum((target / cpa) ** 3, 1.0)
        assert row["cpa"] == pytest.approx(cpa.mean())
        assert row["cpa_violation"] == pytest.approx(violation.mean())
        assert row["cpa_score"] == pytest.approx(score.mean())
        violations.append(float(row["cpa_violation"]))
    # Some rows above target and some below, so both sides of max and min run.
    assert min(violations) == 0 < max(violations)


class FixedRates(Market):
    """pCTR 0.3 and pCVR 0.2 for every campaign at every step."""

    def draws(self, count, rng):
        while True:
            yield MarketDraw({}, pctr=np.full(count, 0.3), pcvr=np.full(count, 0.2))


@pytest.mark.parametrize(
    ("event", "factor"),
    [(Event.IMPRESSION, 1.0), (Event.CLICK, 0.3), (Event.CONVERSION, 0.06)],
)
def test_regret_and_welfare_follow_event(event, factor):
    # Alone in a first-price auction, a truthful bidder earns nothing, while
    # bidding 0.05 × value wins every step and keeps 0.95 of each value.
    campaign = Campaign(
        "a", Stack(OracleValue(), Linear()), budget=1e9, base_value=2.0, event=event
    )
    cfg = SimConfig(
        campaigns=[campaign],
        episodes=2,
        steps=100,
        market=FixedRates(),
        auction=FirstPrice(),
    )
    (row,) = run_many(cfg, seeds=[0]).rows
    total_value = 100 * 2.0 * factor
    assert row["welfare"] == pytest.approx(total_value)
    assert row["regret"] == pytest.approx(0.95 * total_value)
