import csv
import dataclasses
import sys
import threading

import numpy as np
import pytest
from scipy import stats

from bidlab.bidders import Bidder, PidPacing, Truthful
from bidlab.config import BidderSpec, SimConfig
from bidlab.markets import Market, MarketDraw, Segment, Segmented
from bidlab.results import Results
from bidlab.runner import COLUMNS, SEGMENT_COLUMNS, run_many
from bidlab.simulator import run
from bidlab.types import Campaign


def segments_table(results: Results) -> Results:
    assert results.segments is not None
    return results.segments


def config(*extra):
    bidders = [
        BidderSpec("t", Truthful(), Campaign(20.0, 5.0)),
        BidderSpec("p", PidPacing(), Campaign(8.0, 5.0)),
        *extra,
    ]
    return SimConfig(bidders=bidders, episodes=2, steps=100)


def test_rows_ordered_by_seed_then_bidder():
    results = run_many(config(), seeds=range(3))
    assert len(results) == 3 * 2
    assert [(row["seed"], row["bidder"]) for row in results.rows] == [
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
        def forward(self, value, observation):
            return fraction * value

    extra = BidderSpec(
        "closure",
        Closure(),
        Campaign(20.0, 5.0),
        reward=lambda observation, feedback, params: 2 * feedback.price,
    )
    serial = run_many(config(extra), seeds=range(3))
    parallel = run_many(config(extra), seeds=range(3), workers=2)
    assert parallel.equals(serial)


def test_unpicklable_spec_fails_before_runs(monkeypatch):
    class Locked(Bidder):
        def __init__(self):
            self.lock = threading.Lock()

        def forward(self, value, observation):
            return value

    pools = []
    monkeypatch.setattr(
        "bidlab.runner.ProcessPoolExecutor", lambda *args, **kwargs: pools.append(1)
    )
    extra = BidderSpec("locked", Locked(), Campaign(20.0, 5.0))
    with pytest.raises(TypeError, match="'locked'.*workers=1"):
        run_many(config(extra), seeds=range(2), workers=2)

    assert pools == []


def hand_results(spends):
    rows = []
    for seed, spend in enumerate(spends):
        row: dict[str, str | float] = {column: 0.0 for column in COLUMNS}
        row.update(seed=seed, bidder="a", spend=spend)
        rows.append(row)

    return Results(columns=COLUMNS, rows=tuple(rows))


def test_summary_interval_matches_student_t():
    summary = hand_results([1.0, 2.0, 3.0]).summary()
    # mean 2, sample std 1, n 3: half-width = t(0.975, 2) / sqrt(3)
    half_width = stats.t.ppf(0.975, 2) / np.sqrt(3)
    assert f"2.000 ± {half_width:.3f}" in summary
    assert summary.splitlines()[0].split()[:2] == ["bidder", "spend"]


def test_summary_one_seed_shows_na():
    summary = hand_results([4.0]).summary()
    row = summary.splitlines()[1]
    assert "4.000 ± n/a" in row
    assert row.count("n/a") == 6


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
    round_trip = Results(columns=COLUMNS, rows=tuple(frame.to_dict("records")))
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
    for array in (log.bids, log.values, log.winner, log.price, log.converted):
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 0


def test_custom_market_same_rows_in_parallel():
    class LowRates(Market):
        def draws(self, bidders, rng):
            while True:
                yield MarketDraw({}, rng.uniform(0.0, 0.2, size=bidders))

    cfg = dataclasses.replace(config(), market=LowRates())
    serial = run_many(cfg, seeds=range(3))
    parallel = run_many(cfg, seeds=range(3), workers=2)
    assert parallel.equals(serial)
    assert segments_table(parallel).equals(segments_table(serial))


def segmented_config():
    market = Segmented((Segment("news", 0.6), Segment("sports", 0.4, lift=1.0)))
    targeted = BidderSpec("n", Truthful(), Campaign(20.0, 5.0, segments=("news",)))
    return SimConfig(
        bidders=[*config().bidders, targeted], episodes=2, steps=100, market=market
    )


def test_segments_table_columns():
    results = run_many(segmented_config(), seeds=range(2))
    assert results.columns == COLUMNS
    assert isinstance(results.segments, Results)
    assert results.segments.columns == SEGMENT_COLUMNS
    assert SEGMENT_COLUMNS == (
        "seed",
        "bidder",
        "segment",
        "spend",
        "wins",
        "conversions",
        "surplus",
    )
    assert len(results.segments) == 2 * 3 * 2


def test_segment_sums_match_main_table():
    results = run_many(segmented_config(), seeds=range(3))
    for row in results.rows:
        own = [
            segment
            for segment in segments_table(results).rows
            if segment["seed"] == row["seed"] and segment["bidder"] == row["bidder"]
        ]
        for column in ("spend", "wins", "conversions", "surplus"):
            total = sum(float(segment[column]) for segment in own)
            assert total == pytest.approx(row[column], abs=1e-9)


def test_targeted_bidder_has_nothing_in_other_segments():
    results = run_many(segmented_config(), seeds=range(2))
    sports = [
        row
        for row in segments_table(results).rows
        if row["bidder"] == "n" and row["segment"] == "sports"
    ]
    assert sports and all(row["wins"] == 0 and row["spend"] == 0 for row in sports)


def test_unsegmented_market_has_all_segment():
    results = run_many(config(), seeds=range(2))
    assert [
        (row["seed"], row["bidder"], row["segment"])
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
        "bidder",
        "segment",
        "spend",
        "wins",
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

        def draws(self, bidders, rng):
            while True:
                name = "day" if rng.random() < 0.6 else "night"
                yield MarketDraw({}, rng.uniform(0.0, 0.2, size=bidders), name)

    night = BidderSpec("n", Truthful(), Campaign(20.0, 5.0, segments=("night",)))
    cfg = SimConfig(
        bidders=[*config().bidders, night], episodes=2, steps=100, market=DayAndNight()
    )
    segments = segments_table(run_many(cfg, seeds=range(2)))
    assert {row["segment"] for row in segments.rows} == {"day", "night"}
    day_rows = [
        r for r in segments.rows if r["bidder"] == "n" and r["segment"] == "day"
    ]
    assert day_rows and all(row["wins"] == 0 for row in day_rows)
