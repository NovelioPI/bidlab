import os
from collections.abc import Mapping
from typing import Any, ClassVar, override

import pytest

from bidlab.bidders import Bidder, Linear, PidPacing
from bidlab.callbacks import Callback
from bidlab.config import Campaign, SimConfig
from bidlab.results import Results
from bidlab.runner import run_many
from bidlab.simulator import RunLog, run
from bidlab.types import Brief, EpisodeReport

CAMPAIGN = Brief(budget=1e6, event_value=5.0)


class Record(Callback):
    """Records every hook call with its arguments and the process id."""

    def __init__(self, name: str = "cb"):
        self.name = name
        self.calls: list[tuple[str, str, Any]] = []

    @override
    def on_run_start(self, config: SimConfig) -> None:
        self.calls.append((self.name, "run_start", config))

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
    def on_run_start(self, config: SimConfig) -> None:
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
    def on_episode_end(
        self, report: EpisodeReport, episode_reward: float | None
    ) -> None:
        Reports.seen.append(report)


def config(*campaigns: Campaign, **kwargs: Any) -> SimConfig:
    defaults: dict[str, Any] = {"episodes": 3, "steps": 50, "seed": 3}
    return SimConfig(campaigns=campaigns, **(defaults | kwargs))


def two() -> tuple[Campaign, ...]:
    return (
        Campaign("t", Linear(), budget=1e6, event_value=5.0),
        Campaign("p", PidPacing(), budget=3.0, event_value=5.0),
    )


def test_run_hook_order():
    callback = Record()
    settings = config(*two())
    log = run(settings, callbacks=[callback])
    hooks = [hook for _, hook, _ in callback.calls]
    assert hooks == [
        "run_start",
        "episode_end",
        "episode_end",
        "episode_end",
        "run_end",
    ]
    assert callback.calls[0][2] is settings
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
        Campaign("s", Reports(), budget=1e6, event_value=5.0),
        Campaign("t", Linear(), budget=1e6, event_value=5.0),
    )
    run(config(*campaigns), callbacks=[callback])
    reports = [args[1] for _, hook, args in callback.calls if hook == "episode_end"]
    assert [set(episode) for episode in reports] == [{"s", "t"}] * 3
    assert [episode["s"] for episode in reports] == Reports.seen
    with pytest.raises(TypeError):
        reports[0]["x"] = reports[0]["s"]  # pyright: ignore[reportIndexIssue]  Reason: the test checks the mapping is read-only.


def test_callbacks_in_list_order():
    Shared.order.clear()
    run(config(*two(), episodes=1), callbacks=[Shared("a"), Shared("b")])
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
    run_many(settings, seeds=[4, 1, 7], workers=workers, callbacks=[callback])
    seeds = [args[0] for _, _, args in callback.calls]
    assert seeds == [4, 1, 7]
    assert {args[2] for _, _, args in callback.calls} == {os.getpid()}
    for _, _, (seed, results, _) in callback.calls:
        assert results.equals(run_many(settings, seeds=[seed]))


def test_run_many_skips_run_hooks():
    callback = Record()
    run_many(config(*two()), seeds=range(2), callbacks=[callback])
    assert {hook for _, hook, _ in callback.calls} == {"seed_end"}


class Fails(Callback):
    """Raises in the hook named ``hook``."""

    def __init__(self, hook: str):
        self.hook = hook

    def fail(self, hook: str) -> None:
        if hook == self.hook:
            raise RuntimeError(f"stop in {hook}")

    @override
    def on_run_start(self, config: SimConfig) -> None:
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
        run(config(*two()), callbacks=[Fails(hook)])

    with pytest.raises(RuntimeError, match="stop in on_seed_end"):
        run_many(config(*two()), seeds=[0], callbacks=[Fails("on_seed_end")])


@pytest.mark.parametrize(
    "bad", [Record, "progress", print], ids=["class", "str", "function"]
)
def test_callback_type_checked(bad: object):
    instances = f"Callback instances, got {type(bad).__name__}"
    match = "pass an instance" if isinstance(bad, type) else instances
    with pytest.raises(TypeError, match=match):
        run(config(*two()), callbacks=[bad])  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.

    with pytest.raises(TypeError, match=match):
        run_many(config(*two()), seeds=[0], callbacks=[bad])  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


class OnlyEnd(Callback):
    """Overrides one hook only."""

    def __init__(self):
        self.logged: list[RunLog] = []

    @override
    def on_run_end(self, log: RunLog) -> None:
        self.logged.append(log)


def test_partial_callback():
    callback = OnlyEnd()
    log = run(config(*two()), callbacks=[callback])
    run_many(config(*two()), seeds=[0, 1], callbacks=[callback])
    assert len(callback.logged) == 1 and callback.logged[0] is log


def test_callbacks_leave_run_unchanged():
    plain = run(config(*two()))
    watched = run(config(*two()), callbacks=[Record(), OnlyEnd()])
    for name in ("bids", "price", "winner", "rewards"):
        assert (getattr(plain, name) == getattr(watched, name)).all()
