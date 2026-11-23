import os
import re
import threading
from collections.abc import Mapping, Sequence
from multiprocessing.context import SpawnContext
from typing import Any, ClassVar, override

import pytest

from bidlab import runner
from bidlab.bidders import Bidder, Linear, OracleValue, PidPacing, Stack
from bidlab.callbacks import Callback, ProgressBar
from bidlab.config import Campaign, SimConfig
from bidlab.results import Results
from bidlab.simulator import RunLog, run
from bidlab.types import EpisodeReport

SEED = 3


class Record(Callback):
    """Records every hook call with its arguments and the process id."""

    def __init__(self, name: str = "cb"):
        self.name = name
        self.calls: list[tuple[str, str, Any]] = []

    @override
    def on_run_start(self, config: SimConfig, seed: int) -> None:
        self.calls.append((self.name, "run_start", (config, seed)))

    @override
    def on_episode_end(
        self, episode: int, reports: Mapping[str, EpisodeReport]
    ) -> None:
        self.calls.append((self.name, "episode_end", (episode, reports)))

    @override
    def on_run_end(self, log: RunLog) -> None:
        self.calls.append((self.name, "run_end", log))

    @override
    def on_seed_end(self, seed: int, results: Results) -> None:
        self.calls.append((self.name, "seed_end", (seed, results, os.getpid())))


class Shared(Record):
    """Writes every call into one class-level list, to compare two callbacks."""

    order: ClassVar[list[tuple[str, str]]] = []

    @override
    def on_run_start(self, config: SimConfig, seed: int) -> None:
        Shared.order.append((self.name, "run_start"))

    @override
    def on_episode_end(
        self, episode: int, reports: Mapping[str, EpisodeReport]
    ) -> None:
        Shared.order.append((self.name, "episode_end"))

    @override
    def on_run_end(self, log: RunLog) -> None:
        Shared.order.append((self.name, "run_end"))


class Reports(Bidder):
    """Keeps every EpisodeReport its on_episode_end gets."""

    seen: ClassVar[list[EpisodeReport]] = []

    @override
    def forward(self, value: float) -> float:
        return value

    @override
    def on_episode_end(self, report: EpisodeReport) -> None:
        Reports.seen.append(report)


def config(*campaigns: Campaign, **kwargs: Any) -> SimConfig:
    defaults: dict[str, Any] = {"episodes": 3, "steps": 50}
    return SimConfig(campaigns=campaigns, **(defaults | kwargs))


def two() -> tuple[Campaign, ...]:
    return (
        Campaign("t", Stack(OracleValue(), Linear()), budget=1e6, base_value=5.0),
        Campaign("p", Stack(OracleValue(), PidPacing()), budget=3.0, base_value=5.0),
    )


def test_run_hook_order():
    callback = Record()
    settings = config(*two())
    log = run(settings, callbacks=[callback], seed=SEED)
    hooks = [hook for _, hook, _ in callback.calls]
    assert hooks == [
        "run_start",
        "episode_end",
        "episode_end",
        "episode_end",
        "run_end",
    ]
    assert callback.calls[0][2] == (settings, SEED)
    assert callback.calls[0][2][0] is settings
    assert [args[0] for _, hook, args in callback.calls if hook == "episode_end"] == [
        0,
        1,
        2,
    ]
    assert callback.calls[-1][2] is log


def test_episode_report_match_bidders():
    Reports.seen.clear()
    callback = Record()
    campaigns = (
        Campaign("s", Stack(OracleValue(), Reports()), budget=1e6, base_value=5.0),
        Campaign("t", Stack(OracleValue(), Linear()), budget=1e6, base_value=5.0),
    )
    run(config(*campaigns), callbacks=[callback], seed=SEED)
    reports = [args[1] for _, hook, args in callback.calls if hook == "episode_end"]
    assert [set(episode) for episode in reports] == [{"s", "t"}] * 3
    assert [episode["s"] for episode in reports] == Reports.seen
    with pytest.raises(TypeError):
        reports[0]["x"] = reports[0]["s"]  # pyright: ignore[reportIndexIssue]  Reason: the test checks the mapping is read-only.


def test_callbacks_in_list_order():
    Shared.order.clear()
    run(config(*two(), episodes=1), callbacks=[Shared("a"), Shared("b")], seed=SEED)
    assert Shared.order == [
        ("a", "run_start"),
        ("b", "run_start"),
        ("a", "episode_end"),
        ("b", "episode_end"),
        ("a", "run_end"),
        ("b", "run_end"),
    ]


@pytest.mark.parametrize("workers", [1, 2])
def test_seed_end_in_seed_order(workers: int):
    callback = Record()
    settings = config(*two())
    run(settings, seed=[4, 1, 7], workers=workers, callbacks=[callback])
    seeds = [args[0] for _, _, args in callback.calls]
    assert seeds == [4, 1, 7]
    assert {args[2] for _, _, args in callback.calls} == {os.getpid()}
    for _, _, (seed, results, _) in callback.calls:
        assert results.equals(run(settings, seed=[seed]))


def test_seed_list_skips_run_hooks():
    callback = Record()
    run(config(*two()), seed=range(2), callbacks=[callback])
    assert {hook for _, hook, _ in callback.calls} == {"seed_end"}


class Fails(Callback):
    """Raises in the hook named ``hook``."""

    def __init__(self, hook: str):
        self.hook = hook

    def fail(self, hook: str) -> None:
        if hook == self.hook:
            raise RuntimeError(f"stop in {hook}")

    @override
    def on_run_start(self, config: SimConfig, seed: int) -> None:
        self.fail("on_run_start")

    @override
    def on_episode_end(
        self, episode: int, reports: Mapping[str, EpisodeReport]
    ) -> None:
        self.fail("on_episode_end")

    @override
    def on_run_end(self, log: RunLog) -> None:
        self.fail("on_run_end")

    @override
    def on_seed_end(self, seed: int, results: Results) -> None:
        self.fail("on_seed_end")


@pytest.mark.parametrize("hook", ["on_run_start", "on_episode_end", "on_run_end"])
def test_callback_error_reaches_caller(hook: str):
    with pytest.raises(RuntimeError, match=f"stop in {hook}"):
        run(config(*two()), callbacks=[Fails(hook)], seed=SEED)

    with pytest.raises(RuntimeError, match="stop in on_seed_end"):
        run(config(*two()), seed=[0], callbacks=[Fails("on_seed_end")])


@pytest.mark.parametrize(
    "bad", [Record, "progress", print], ids=["class", "str", "function"]
)
def test_callback_type_checked(bad: object):
    instances = f"Callback instances, got {type(bad).__name__}"
    match = "pass an instance" if isinstance(bad, type) else instances
    with pytest.raises(TypeError, match=match):
        run(config(*two()), callbacks=[bad], seed=SEED)  # pyright: ignore[reportArgumentType, reportCallIssue]  Reason: the test checks the runtime error for a wrong type.

    with pytest.raises(TypeError, match=match):
        run(config(*two()), seed=[0], callbacks=[bad])  # pyright: ignore[reportArgumentType, reportCallIssue]  Reason: the test checks the runtime error for a wrong type.


class OnlyEnd(Callback):
    """Overrides one hook only."""

    def __init__(self):
        self.logged: list[RunLog] = []

    @override
    def on_run_end(self, log: RunLog) -> None:
        self.logged.append(log)


def test_partial_callback():
    callback = OnlyEnd()
    log = run(config(*two()), callbacks=[callback], seed=SEED)
    run(config(*two()), seed=[0, 1], callbacks=[callback])
    assert len(callback.logged) == 1 and callback.logged[0] is log


def test_callbacks_leave_run_unchanged():
    plain = run(config(*two()), seed=SEED)
    watched = run(config(*two()), callbacks=[Record(), OnlyEnd()], seed=SEED)
    for name in ("bids", "price", "won", "converted"):
        assert (getattr(plain, name) == getattr(watched, name)).all()


class RecordMany(Record):
    """Also records the seed list start and end hooks."""

    @override
    def on_seeds_start(self, config: SimConfig, seeds: Sequence[int]) -> None:
        self.calls.append((self.name, "seeds_start", (config, seeds)))

    @override
    def on_seeds_end(self, results: Results) -> None:
        self.calls.append((self.name, "seeds_end", results))


@pytest.mark.parametrize("workers", [1, 2])
def test_seed_list_hook_order(workers: int):
    callback = RecordMany()
    settings = config(*two())
    results = run(settings, seed=iter([4, 1, 7]), workers=workers, callbacks=[callback])
    hooks = [hook for _, hook, _ in callback.calls]
    assert hooks == [
        "seeds_start",
        "seed_end",
        "seed_end",
        "seed_end",
        "seeds_end",
    ]
    assert callback.calls[0][2] == (settings, (4, 1, 7))
    assert callback.calls[-1][2] is results


class Progress(RecordMany):
    """Also records every seed progress call and the thread it ran on."""

    @override
    def on_seed_progress(self, seed: int, episode: int, step: int) -> None:
        main = threading.current_thread() is threading.main_thread()
        self.calls.append((self.name, "progress", (seed, episode, step, main)))


def progress_steps(steps: int) -> list[int]:
    """The steps a seed reports per episode: every max(1, steps // 20), and the last."""
    every = max(1, steps // 20)
    return [
        step for step in range(steps) if (step + 1) % every == 0 or step == steps - 1
    ]


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("steps", [45, 7])
def test_seed_progress_reaches_each_episode_end(workers: int, steps: int):
    callback = Progress()
    seeds = [4, 1, 7]
    run(
        config(*two(), steps=steps),
        seed=seeds,
        workers=workers,
        callbacks=[callback],
        progress_bar=False,
    )
    progress = [args for _, hook, args in callback.calls if hook == "progress"]
    expected = [
        (episode, step) for episode in range(3) for step in progress_steps(steps)
    ]
    for seed in seeds:
        assert [(e, s) for seed_, e, s, _ in progress if seed_ == seed] == expected

    assert all(main for *_, main in progress)
    assert len(progress) == len(seeds) * len(expected)


def test_progress_steps_rule():
    assert progress_steps(45) == [*range(1, 44, 2), 44]
    assert progress_steps(2000) == list(range(99, 2000, 100))
    assert progress_steps(7) == list(range(7))


@pytest.mark.parametrize("workers", [1, 2])
def test_seed_progress_comes_before_its_seed_end(workers: int):
    callback = Progress()
    run(
        config(*two()),
        seed=[4, 1, 7],
        workers=workers,
        callbacks=[callback],
        progress_bar=False,
    )
    hooks = [(hook, args[0]) for _, hook, args in callback.calls[1:-1]]
    ends = [seed for hook, seed in hooks if hook == "seed_end"]
    assert ends == [4, 1, 7]
    for seed in ends:
        own = [i for i, (hook, s) in enumerate(hooks) if s == seed]
        assert hooks[own[-1]] == ("seed_end", seed)
        assert all(hooks[i][0] == "progress" for i in own[:-1])
        assert len(own) > 1
    assert callback.calls[0][1] == "seeds_start"
    assert callback.calls[-1][1] == "seeds_end"


@pytest.mark.parametrize("workers", [1, 2])
def test_seed_callbacks_leave_results_unchanged(workers: int):
    settings = config(*two())
    plain = run(settings, seed=[4, 1], workers=workers, progress_bar=False)
    watched = run(
        settings, seed=[4, 1], workers=workers, callbacks=[Progress(), OnlyEnd()]
    )
    assert watched.equals(plain)
    assert watched.equals(run(settings, seed=[4, 1], progress_bar=False))


def test_no_progress_work_without_callbacks(monkeypatch: pytest.MonkeyPatch):
    reports: list[tuple[int, int]] = []
    original = runner._SeedProgress.on_step_end

    def counted(self: runner._SeedProgress, episode: int, step: int) -> None:
        reports.append((episode, step))
        original(self, episode, step)

    monkeypatch.setattr(runner._SeedProgress, "on_step_end", counted)
    run(config(*two()), seed=[4], progress_bar=False)
    assert reports == []
    run(config(*two()), seed=[4], callbacks=[OnlyEnd()], progress_bar=False)
    assert len(reports) == 3 * 50


def test_no_progress_queue_without_callbacks(monkeypatch: pytest.MonkeyPatch):
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a Manager started without callbacks")

    monkeypatch.setattr(SpawnContext, "Manager", refuse)
    results = run(config(*two()), seed=[4, 1], workers=2, progress_bar=False)
    assert len(results) == 4
    with pytest.raises(AssertionError, match="Manager started"):
        run(config(*two()), seed=[4, 1], workers=2)


def test_progress_bar_counts_steps(capsys: pytest.CaptureFixture[str]):
    plain = run(config(*two()), progress_bar=False, seed=42)
    # Keep the bar referenced, so garbage collection cannot close it for the run.
    bar = ProgressBar()
    watched = run(config(*two()), callbacks=[bar], seed=42)
    out, err = capsys.readouterr()
    assert out == ""
    assert "seed 42 episode 1/3 " in err and "seed 42 episode 2/3 " in err
    assert re.search(RUN_DONE, err)
    assert "episode/s" not in err and "/episode" not in err
    assert (plain.bids == watched.bids).all()


class Steps(Record):
    """Also records every step end."""

    @override
    def on_step_end(self, episode: int, step: int) -> None:
        self.calls.append((self.name, "step_end", (episode, step)))


def test_step_end_order():
    callback = Steps()
    run(config(*two(), episodes=2, steps=3), callbacks=[callback], seed=SEED)
    calls = [
        (hook, args if hook != "episode_end" else args[0])
        for _, hook, args in callback.calls[1:-1]
    ]
    assert calls == [
        ("step_end", (0, 0)),
        ("step_end", (0, 1)),
        ("step_end", (0, 2)),
        ("episode_end", 0),
        ("step_end", (1, 0)),
        ("step_end", (1, 1)),
        ("step_end", (1, 2)),
        ("episode_end", 1),
    ]


def test_seed_list_skips_step_end():
    callback = Steps()
    run(config(*two()), seed=range(2), workers=1, callbacks=[callback])
    assert "step_end" not in {hook for _, hook, _ in callback.calls}


SEEDS_DONE = r"\rseeds   2/2 100%\|█+\| \[\d\d:\d\d<\d\d:\d\d\]\n$"
RUN_DONE = (
    r"\rseed 42 episode 3/3 100%\|█+\| 50/50 \[\d\d:\d\d<\d\d:\d\d, .* step/s\]\n$"
)
# tqdm clears a closed bar's line with spaces, then moves back up.
CLEARED_LINE = r"\r {20,}\x1b\[A"


@pytest.mark.parametrize("workers", [1, 2])
def test_progress_bar_counts_seeds(capsys: pytest.CaptureFixture[str], workers: int):
    bar = ProgressBar()
    results = run(config(*two()), seed=[4, 1], workers=workers, callbacks=[bar])
    out, err = capsys.readouterr()
    assert out == ""
    assert re.search(SEEDS_DONE, err)
    for seed in (4, 1):
        for episode in (1, 2, 3):
            assert f"seed {seed} episode {episode}/3 " in err
    assert len(re.findall(CLEARED_LINE, err)) == 2
    assert results.equals(run(config(*two()), seed=[4, 1], progress_bar=False))


def bar_text(capsys: pytest.CaptureFixture[str], *calls: tuple[str, int, int]) -> str:
    """Drive a ProgressBar for seeds 1, 2 and 3 through ``calls``, return stderr."""
    bar = ProgressBar()
    empty = Results(columns=(), rows=())
    bar.on_seeds_start(config(*two()), [1, 2, 3])
    for hook, seed, episode in calls:
        if hook == "progress":
            bar.on_seed_progress(seed, episode, 4)
        else:
            bar.on_seed_end(seed, empty)
    bar.on_seeds_end(empty)
    return capsys.readouterr().err


def test_seed_bars_reuse_freed_lines(capsys: pytest.CaptureFixture[str]):
    err = bar_text(
        capsys,
        ("progress", 1, 0),
        ("progress", 2, 0),
        ("end", 1, 0),
        ("progress", 3, 0),
        ("progress", 3, 1),
        ("end", 2, 0),
        ("end", 3, 0),
    )
    # tqdm moves down one newline per line below the seeds bar.
    assert re.search(r"(?<!\n)\n\rseed 1 episode 1/3 ", err)
    assert "\n\n\rseed 2 episode 1/3 " in err
    assert re.search(r"(?<!\n)\n\rseed 3 episode 1/3 ", err)
    assert re.search(r"(?<!\n)\n\rseed 3 episode 2/3 ", err)
    assert "episode 2/3:" not in err
    assert len(re.findall(CLEARED_LINE, err)) == 3
    assert err.endswith("\n") and "seeds   3/3 100%" in err.rsplit("\r", 1)[-1]


def test_seeds_end_clears_open_seed_bars(capsys: pytest.CaptureFixture[str]):
    err = bar_text(capsys, ("progress", 1, 0), ("progress", 2, 0))
    assert len(re.findall(CLEARED_LINE, err)) == 2
    assert "seeds   0/3" in err.rsplit("\r", 1)[-1]


def test_run_shows_a_bar_by_default(capsys: pytest.CaptureFixture[str]):
    run(config(*two()), seed=42)
    assert re.search(RUN_DONE, capsys.readouterr().err)


def test_run_without_progress_bar_prints_nothing(capsys: pytest.CaptureFixture[str]):
    run(config(*two()), progress_bar=False, seed=SEED)
    run(config(*two()), seed=range(2), progress_bar=False)
    assert capsys.readouterr() == ("", "")


def test_given_progress_bar_is_not_doubled(capsys: pytest.CaptureFixture[str]):
    bar = ProgressBar()
    run(config(*two()), callbacks=[bar], seed=SEED)
    assert capsys.readouterr().err.count(f"seed {SEED} episode 1/3   0%") == 1


def test_seed_list_shows_seed_bars_by_default(capsys: pytest.CaptureFixture[str]):
    run(config(*two()), seed=[4, 1])
    err = capsys.readouterr().err
    assert re.search(SEEDS_DONE, err)
    assert "seed 4 episode 1/3 " in err and "seed 1 episode 3/3 " in err
    assert len(re.findall(CLEARED_LINE, err)) == 2
