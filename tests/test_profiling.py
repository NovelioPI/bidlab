import csv
import importlib.metadata
import re
import time
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from bidlab import profiling
from bidlab.bidders import Bidder, Linear, OracleValue, PidPacing, Stack
from bidlab.callbacks import Callback
from bidlab.config import Campaign, Feedback, SimConfig
from bidlab.markets import LogisticMarket, Segment, Segmented
from bidlab.profiling import HookTime, Profile
from bidlab.runner import run_many
from bidlab.simulator import RunLog, run
from bidlab.types import BidResult, Conversion, EpisodeReport

HOOKS = (
    "forward",
    "update_batch",
    "update",
    "update_conversions",
    "on_episode_start",
    "on_episode_end",
)
# Summing per-call differences and differencing sums round apart by far less.
TOLERANCE = 1e-9
SLEEP = 0.001
MEGABYTE = 1024 * 1024
BLOCK_BYTES = 50 * MEGABYTE
# Campaign, node and hook align left; the numbers after them align right.
LABEL_COLUMNS = 3

ROOT = Path(__file__).parent.parent
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


class Probe(Bidder):
    """Counts its own hook calls in a class-level counter, by name."""

    counts: ClassVar[Counter[tuple[str, str]]] = Counter()

    def __init__(self, name: str):
        self.name = name

    def _count(self, hook: str) -> None:
        Probe.counts[(self.name, hook)] += 1

    def on_episode_start(self, campaign: Campaign) -> None:
        self._count("on_episode_start")

    def forward(self, value: float = 1.0) -> float:
        self._count("forward")
        return value

    def update_batch(self, bid_results: Sequence[BidResult]) -> None:
        self._count("update_batch")
        super().update_batch(bid_results)

    def update(self, result: BidResult) -> None:
        self._count("update")

    def update_conversions(self, conversions: Sequence[Conversion]) -> None:
        self._count("update_conversions")

    def on_episode_end(self, report: EpisodeReport) -> None:
        self._count("on_episode_end")


class Fails(Bidder):
    """Raises in one hook."""

    def __init__(self, hook: str):
        self.hook = hook

    def _fail(self, hook: str) -> None:
        if hook == self.hook:
            raise RuntimeError(f"{hook} failed")

    def on_episode_start(self, campaign: Campaign) -> None:
        self._fail("on_episode_start")

    def forward(self, value: float = 1.0) -> float:
        self._fail("forward")
        return value

    def update(self, result: BidResult) -> None:
        self._fail("update")


class Forgiving(Bidder):
    """Calls a child that fails and bids its own value instead."""

    def __init__(self, child: Bidder):
        self.child = child

    def forward(self, value: float = 1.0) -> float:
        try:
            return self.child.forward(value=value)
        except RuntimeError:
            return value


class Sleeps(Bidder):
    """Sleeps one millisecond per forward call."""

    def forward(self, value: float = 1.0) -> float:
        time.sleep(SLEEP)
        return value


class Holds(Bidder):
    """Keeps a large array from its first episode start to the run end."""

    def __init__(self) -> None:
        self.block: np.ndarray | None = None

    def on_episode_start(self, campaign: Campaign) -> None:
        if self.block is None:
            # ones, not zeros: zeroed pages are not resident until written.
            self.block = np.ones(BLOCK_BYTES // 8)

    def forward(self, value: float = 1.0) -> float:
        return value


class Events(Bidder):
    """Writes its episode hooks into a class-level list, to order memory samples."""

    events: ClassVar[list[str]] = []

    def on_episode_start(self, campaign: Campaign) -> None:
        Events.events.append("start")

    def forward(self, value: float = 1.0) -> float:
        return value

    def on_episode_end(self, report: EpisodeReport) -> None:
        Events.events.append("end")


class Starts(Callback):
    """Counts run_many starts."""

    def __init__(self) -> None:
        self.starts = 0

    def on_run_many_start(self, config: SimConfig, seeds: Sequence[int]) -> None:
        self.starts += 1


def _profile(log: RunLog) -> Profile:
    assert log.profile is not None
    return log.profile


def _rows(
    profile: Profile | None, campaign: str = "c0"
) -> dict[tuple[str, str], HookTime]:
    assert profile is not None
    return {
        (row.path, row.hook): row for row in profile.rows if row.campaign == campaign
    }


def _names(requirements: list[str]) -> list[str]:
    return [re.split(r"[<>=!~ \[;]", requirement)[0] for requirement in requirements]


def _config(*bidders: Bidder, **settings: Any) -> SimConfig:
    campaigns = [
        Campaign(f"c{i}", bidder, budget=10, base_value=5)
        for i, bidder in enumerate(bidders)
    ]
    return SimConfig(campaigns=campaigns, **{"episodes": 2, "steps": 50, **settings})


def _plain() -> SimConfig:
    return _config(
        Stack(OracleValue(), Linear()),
        Stack(OracleValue(), PidPacing()),
        market=LogisticMarket(price_median=0.2),
    )


def test_psutil_is_a_runtime_dependency():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "psutil" in _names(pyproject["project"]["dependencies"])
    # The installed package's metadata is what a fresh `uv sync` reads.
    installed = importlib.metadata.requires("bidlab") or []
    assert "psutil" in _names([r for r in installed if "extra ==" not in r])


def test_no_profile_by_default():
    assert run(_plain(), progress_bar=False).profile is None
    assert run_many(_plain(), seeds=[0, 1], progress_bar=False).profile is None


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_profiler_must_be_bool(bad: object):
    message = f"profiler must be a bool, got {type(bad).__name__}$"
    # Any, since the check under test rejects what the annotation forbids.
    profiler: Any = bad
    with pytest.raises(TypeError, match=message):
        run(_plain(), progress_bar=False, profiler=profiler)

    # run_many checks before it starts, so no callback sees a run that cannot happen.
    starts = Starts()
    with pytest.raises(TypeError, match=message):
        run_many(
            _plain(),
            seeds=[0],
            callbacks=[starts],
            progress_bar=False,
            profiler=profiler,
        )

    assert starts.starts == 0


def test_calls_match_the_bidder_count():
    Probe.counts.clear()
    config = _config(
        Stack(OracleValue(), Stack(Linear(), Probe("inner"))),
        Probe("top"),
        steps=300,
        market=LogisticMarket(ctr=0.5, cvr=0.5),
        feedback=Feedback(conversion_delay=3600),
    )
    profile = run(config, progress_bar=False, profiler=True).profile
    assert Probe.counts[("inner", "update_conversions")] > 0
    for campaign, path, name in (
        ("c0", "bidders.1.bidders.1", "inner"),
        ("c1", "", "top"),
    ):
        rows = _rows(profile, campaign)
        for hook in HOOKS:
            calls = rows[(path, hook)].calls if (path, hook) in rows else 0
            assert calls == Probe.counts[(name, hook)], (campaign, hook)

    rows = _rows(profile)
    stack_calls = rows[("", "forward")].calls
    assert stack_calls > 0
    for path in (
        "bidders.0",
        "bidders.1",
        "bidders.1.bidders.0",
        "bidders.1.bidders.1",
    ):
        assert rows[(path, "forward")].calls == stack_calls
    # Only learners get update hooks, as without the profiler.

    assert ("", "update_batch") not in rows
    assert ("bidders.0", "update_batch") not in rows


def test_self_time_excludes_children():
    config = _config(
        Stack(OracleValue(), Stack(Linear(), Probe("inner"))),
        market=LogisticMarket(price_median=0.2),
    )
    rows = _rows(run(config, progress_bar=False, profiler=True).profile)
    trees = {
        ("", "forward"): [("bidders.0", "forward"), ("bidders.1", "forward")],
        ("bidders.1", "forward"): [
            ("bidders.1.bidders.0", "forward"),
            ("bidders.1.bidders.1", "forward"),
        ],
        # The default update_batch calls update on the same node.
        ("bidders.1.bidders.1", "update_batch"): [("bidders.1.bidders.1", "update")],
    }
    for parent, children in trees.items():
        inner = sum(rows[child].total for child in children)
        assert rows[parent].total >= inner
        assert rows[parent].self_time == pytest.approx(
            rows[parent].total - inner, abs=TOLERANCE
        )

    for key in [("bidders.0", "forward"), ("bidders.1.bidders.1", "update")]:
        assert rows[key].self_time == pytest.approx(rows[key].total, abs=TOLERANCE)


@pytest.mark.parametrize("hook", ["forward", "update", "on_episode_start"])
def test_hook_error_passes_through(hook: str):
    config = _config(Stack(OracleValue(), Fails(hook)))
    with pytest.raises(RuntimeError) as plain:
        run(config, progress_bar=False)

    with pytest.raises(RuntimeError) as profiled:
        run(config, progress_bar=False, profiler=True)

    assert type(profiled.value) is type(plain.value)
    assert str(profiled.value) == str(plain.value) == f"{hook} failed"


def test_caught_child_error_keeps_self_times():
    config = _config(Stack(OracleValue(), Forgiving(Fails("forward"))))
    rows = _rows(run(config, progress_bar=False, profiler=True).profile)
    parent, child = rows[("bidders.1", "forward")], rows[("bidders.1.child", "forward")]
    assert parent.calls == child.calls > 0
    assert parent.self_time == pytest.approx(parent.total - child.total, abs=TOLERANCE)
    top = rows[("", "forward")]
    inner = rows[("bidders.0", "forward")].total + parent.total
    assert top.self_time == pytest.approx(top.total - inner, abs=TOLERANCE)


def test_user_bidder_is_not_wrapped():
    bidder = Stack(OracleValue(), Stack(Linear(), Probe("user")))
    nodes = [node for _, node in bidder.named_bidders()]
    before = [dict(vars(node)) for node in nodes]
    methods = {
        (type(node), hook): getattr(type(node), hook)
        for node in nodes
        for hook in HOOKS
    }
    config = _config(bidder)
    first = _rows(run(config, progress_bar=False, profiler=True).profile)
    for node, state in zip(nodes, before, strict=True):
        assert not set(HOOKS) & set(vars(node))
        assert vars(node).keys() == state.keys()

    for (cls, hook), method in methods.items():
        assert getattr(cls, hook) is method
    # A second run wraps a fresh copy once, so it counts each call once.

    second = _rows(run(config, progress_bar=False, profiler=True).profile)
    assert {key: row.calls for key, row in second.items()} == {
        key: row.calls for key, row in first.items()
    }


def _arrays(log: RunLog) -> list[np.ndarray]:
    logged = [array for arrays in log.logged.values() for array in arrays.values()]
    return [getattr(log, name) for name in ARRAYS] + logged


@pytest.mark.parametrize(
    "estimator", [OracleValue(), OracleValue(noise=0.1)], ids=["plain", "noise"]
)
def test_profiled_run_keeps_every_bit(estimator: OracleValue):
    config = _config(
        Stack(estimator, Linear()),
        Stack(estimator, PidPacing()),
        market=LogisticMarket(price_median=0.2, floor_share=(2, 5)),
        steps=200,
    )
    plain = run(config, progress_bar=False)
    profiled = run(config, progress_bar=False, profiler=True)
    assert plain.logged.keys() == profiled.logged.keys()
    assert len(plain.logged) > 0
    for mine, theirs in zip(_arrays(plain), _arrays(profiled), strict=True):
        assert (mine.dtype, mine.shape) == (theirs.dtype, theirs.shape)
        assert mine.tobytes() == theirs.tobytes()


def test_slow_forward_leads_the_table():
    # Bare, since a Stack's total holds its children's, so the Stack would lead.
    config = _config(Sleeps(), Stack(OracleValue(), Linear()))
    profile = _profile(run(config, progress_bar=False, profiler=True))
    slow = _rows(profile)[("", "forward")]
    assert slow.calls == 2 * 50
    assert slow.total >= slow.calls * SLEEP
    assert slow.self_time >= slow.calls * SLEEP
    first = profile.summary().splitlines()[1].split()
    assert first[:3] == ["c0", "(top)", "forward"]


def test_run_totals():
    config = _plain()
    start = time.perf_counter()
    profile = _profile(run(config, progress_bar=False, profiler=True))
    assert 0 < profile.wall_time <= time.perf_counter() - start
    assert profile.steps == config.episodes * config.steps
    assert profile.bidder_time == pytest.approx(
        sum(row.self_time for row in profile.rows), abs=TOLERANCE
    )
    assert 0 < profile.bidder_time <= profile.wall_time
    assert profile.simulator_time == pytest.approx(
        profile.wall_time - profile.bidder_time, abs=TOLERANCE
    )


def test_memory_samples(monkeypatch: pytest.MonkeyPatch):
    profile = _profile(run(_plain(), progress_bar=False, profiler=True))
    assert isinstance(profile.memory_start, int)
    assert isinstance(profile.memory_peak, int)
    assert profile.memory_peak >= profile.memory_start > 0

    Events.events.clear()
    samples = iter([5 * MEGABYTE, 9 * MEGABYTE, 7 * MEGABYTE, 6 * MEGABYTE])

    def resident_bytes() -> int:
        Events.events.append("sample")
        return next(samples)

    monkeypatch.setattr(profiling, "_resident_bytes", resident_bytes)
    profile = _profile(run(_config(Events()), progress_bar=False, profiler=True))
    # Run start, each of the two episode ends, then run end.
    assert Events.events == [
        "sample",
        "start",
        "end",
        "sample",
        "start",
        "end",
        "sample",
        "sample",
    ]
    assert (profile.memory_start, profile.memory_peak) == (5 * MEGABYTE, 9 * MEGABYTE)


def test_memory_growth_sees_a_large_array():
    profile = _profile(run(_config(Holds()), progress_bar=False, profiler=True))
    assert profile.memory_peak - profile.memory_start >= 40 * MEGABYTE


def test_summary_table():
    profile = _profile(run(_plain(), progress_bar=False, profiler=True))
    lines = profile.summary().splitlines()
    header, body, footer = lines[0], lines[1:-1], lines[-1]
    assert header.split() == [
        "campaign",
        "node",
        "hook",
        "calls",
        "total_s",
        "self_s",
        "mean_us",
        "share",
    ]
    assert len(body) == len(profile.rows)
    by_key = {
        (row.campaign, row.path or "(top)", row.hook): row for row in profile.rows
    }
    totals = []
    for line in body:
        campaign, path, hook, calls, total, self_time, mean, share = line.split()
        row = by_key.pop((campaign, path, hook))
        assert int(calls) == row.calls
        assert float(total) == pytest.approx(row.total, abs=5e-7)
        assert float(self_time) == pytest.approx(row.self_time, abs=5e-7)
        assert float(mean) == pytest.approx(row.total / row.calls * 1e6, abs=0.05)
        assert float(share.rstrip("%")) == pytest.approx(
            row.total / profile.wall_time * 100, abs=0.05
        )
        totals.append(row.total)

    assert not by_key
    assert totals == sorted(totals, reverse=True)

    # Labels start under their heading and numbers end under theirs.
    columns = [(match.start(), match.end()) for match in re.finditer(r"\S+", header)]
    for line in body:
        assert len(line) == len(header)
        for index, (cell, (start, end)) in enumerate(
            zip(line.split(), columns, strict=True)
        ):
            if index < LABEL_COLUMNS:
                assert line[start : start + len(cell)] == cell
                assert start == 0 or line[start - 1] == " "
            else:
                assert line[end - len(cell) : end] == cell
                assert end == len(line) or line[end] == " "

    megabytes = profile.memory_peak / MEGABYTE
    growth = (profile.memory_peak - profile.memory_start) / MEGABYTE
    assert footer == (
        f"wall {profile.wall_time:.3f} s, "
        f"{profile.steps / profile.wall_time:,.0f} steps/s, "
        f"simulator {profile.simulator_time:.3f} s "
        f"({profile.simulator_time / profile.wall_time:.1%}), "
        f"memory peak {megabytes:.1f} MiB (+{growth:.1f} MiB over start)"
    )


def _calls(profile: Profile, campaign: str) -> dict[tuple[str, str], int]:
    return {key: row.calls for key, row in _rows(profile, campaign).items()}


@pytest.mark.parametrize("workers", [1, 2])
def test_run_many_profiles_in_seed_order(workers: int):
    # Each seed draws its own segments, so each seed calls forward a different number of times.
    config = SimConfig(
        campaigns=[
            Campaign(
                "news",
                Stack(OracleValue(), Linear()),
                budget=10,
                base_value=5,
                segments=("news",),
            )
        ],
        market=Segmented(
            (Segment("news", 0.5), Segment("sports", 0.5)),
            base=LogisticMarket(price_median=0.2),
        ),
        episodes=2,
        steps=50,
    )
    seeds = [3, 0, 7]
    profiled = run_many(
        config, seeds=seeds, workers=workers, progress_bar=False, profiler=True
    )
    assert profiled.profile is not None
    expected = [
        _calls(
            _profile(
                run(replace(config, seed=seed), progress_bar=False, profiler=True)
            ),
            "news",
        )
        for seed in seeds
    ]
    assert [_calls(profile, "news") for profile in profiled.profile] == expected
    assert len({calls[("", "forward")] for calls in expected}) == len(seeds)

    plain = run_many(config, seeds=seeds, workers=workers, progress_bar=False)
    assert plain.profile is None
    assert profiled.equals(plain)
    one_worker = run_many(config, seeds=seeds, progress_bar=False, profiler=True)
    assert profiled.equals(one_worker)


CSV_HEADER = [
    "seed",
    "campaign",
    "path",
    "hook",
    "calls",
    "total_seconds",
    "self_seconds",
    "wall_seconds",
    "steps",
    "memory_start",
    "memory_peak",
]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _summary_order(profile: Profile) -> list[tuple[str, str, str]]:
    keys = []
    for line in profile.summary().splitlines()[1:-1]:
        campaign, path, hook = line.split()[:3]
        keys.append((campaign, "" if path == "(top)" else path, hook))
    return keys


def _assert_lines_match(lines: list[dict[str, str]], profile: Profile) -> None:
    rows = {(row.campaign, row.path, row.hook): row for row in profile.rows}
    assert [(line["campaign"], line["path"], line["hook"]) for line in lines] == (
        _summary_order(profile)
    )
    for line in lines:
        row = rows[(line["campaign"], line["path"], line["hook"])]
        assert int(line["calls"]) == row.calls
        assert float(line["total_seconds"]) == row.total
        assert float(line["self_seconds"]) == row.self_time
        assert int(line["seed"]) == profile.seed
        assert float(line["wall_seconds"]) == profile.wall_time
        assert int(line["steps"]) == profile.steps
        assert int(line["memory_start"]) == profile.memory_start
        assert int(line["memory_peak"]) == profile.memory_peak


def test_profile_save_csv_round_trip(tmp_path: Path):
    config = replace(_plain(), seed=11)
    profile = _profile(run(config, progress_bar=False, profiler=True))
    assert profile.seed == 11
    path = tmp_path / "profile.csv"
    profile.save_csv(path)
    header, lines = _read_csv(path)
    assert header == CSV_HEADER
    assert len(lines) == len(profile.rows)
    _assert_lines_match(lines, profile)


def test_results_save_profile_csv(tmp_path: Path):
    seeds = [3, 0]
    results = run_many(_plain(), seeds=seeds, progress_bar=False, profiler=True)
    assert results.profile is not None
    assert [profile.seed for profile in results.profile] == seeds
    path = tmp_path / "profiles.csv"
    results.save_profile_csv(path)
    header, lines = _read_csv(path)
    assert header == CSV_HEADER
    start = 0
    for profile in results.profile:
        own = lines[start : start + len(profile.rows)]
        assert {line["seed"] for line in own} == {str(profile.seed)}
        _assert_lines_match(own, profile)
        start += len(profile.rows)
    assert start == len(lines)

    plain = run_many(_plain(), seeds=seeds, progress_bar=False)
    missing = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match="profiler=True"):
        plain.save_profile_csv(missing)
    assert not missing.exists()
