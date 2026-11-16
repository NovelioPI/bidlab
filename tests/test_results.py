import csv
import dataclasses
import sys
import threading

import numpy as np
import pytest
from scipy import stats

from bidlab.bidders import PidPacing, Truthful
from bidlab.config import BidderSpec, SimConfig
from bidlab.markets import Market, MarketDraw
from bidlab.results import Results
from bidlab.runner import COLUMNS, run_many
from bidlab.simulator import run


def config(*extra):
    bidders = [
        BidderSpec("t", Truthful, 20.0, 5.0),
        BidderSpec("p", PidPacing, 8.0, 5.0),
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

    def make_bidder(params, rng):
        return Truthful({"multiplier": fraction}, rng)

    extra = BidderSpec(
        "closure",
        make_bidder,
        20.0,
        5.0,
        reward=lambda observation, feedback, params: 2 * feedback.price,
    )
    serial = run_many(config(extra), seeds=range(3))
    parallel = run_many(config(extra), seeds=range(3), workers=2)
    assert parallel.equals(serial)


def test_unpicklable_spec_fails_before_runs(monkeypatch):
    lock = threading.Lock()

    def make_bidder(params, rng):
        with lock:
            return Truthful(params, rng)

    pools = []
    monkeypatch.setattr(
        "bidlab.runner.ProcessPoolExecutor", lambda *args, **kwargs: pools.append(1)
    )
    extra = BidderSpec("locked", make_bidder, 20.0, 5.0)
    with pytest.raises(TypeError, match="'locked'.*workers=1"):
        run_many(config(extra), seeds=range(2), workers=2)

    assert pools == []


def hand_results(spends):
    rows = []
    for seed, spend in enumerate(spends):
        row = {column: 0.0 for column in COLUMNS}
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
        results.rows[0]["spend"] = 0.0


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
