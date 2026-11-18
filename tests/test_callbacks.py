import os
from collections.abc import Mapping
from typing import Any, ClassVar, override

import pytest

from bidlab.bidders import Bidder, PidPacing, Truthful
from bidlab.callbacks import Callback
from bidlab.config import BidderSpec, SimConfig
from bidlab.results import Results
from bidlab.runner import run_many
from bidlab.simulator import RunLog, run
from bidlab.types import Campaign, EpisodeStats, Observation

CAMPAIGN = Campaign(budget=1e6, value=5.0)


class Record(Callback):
    """Records every hook call with its arguments and the process id."""

    def __init__(self, name: str = "cb"):
        self.name = name
        self.calls: list[tuple[str, str, Any]] = []

    @override
    def on_run_start(self, config: SimConfig) -> None:
        self.calls.append((self.name, "run_start", config))

    @override
    def on_episode_end(self, episode: int, stats: Mapping[str, EpisodeStats]) -> None:
        self.calls.append((self.name, "episode_end", (episode, stats)))

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
    def on_episode_end(self, episode: int, stats: Mapping[str, EpisodeStats]) -> None:
        Shared.order.append((self.name, "episode_end"))

    @override
    def on_run_end(self, log: RunLog) -> None:
        Shared.order.append((self.name, "run_end"))


class Stats(Bidder):
    """Keeps every EpisodeStats its on_episode_end gets."""

    seen: ClassVar[list[EpisodeStats]] = []

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return value

    @override
    def on_episode_end(self, stats: EpisodeStats, episode_reward: float | None) -> None:
        Stats.seen.append(stats)


def config(*specs: BidderSpec, **kwargs: Any) -> SimConfig:
    defaults: dict[str, Any] = {"episodes": 3, "steps": 50, "seed": 3}
    return SimConfig(bidders=specs, **(defaults | kwargs))


def two() -> tuple[BidderSpec, ...]:
    return (
        BidderSpec("t", Truthful(), CAMPAIGN),
        BidderSpec("p", PidPacing(), Campaign(3.0, 5.0)),
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


def test_episode_stats_match_bidders():
    Stats.seen.clear()
    callback = Record()
    specs = (BidderSpec("s", Stats(), CAMPAIGN), BidderSpec("t", Truthful(), CAMPAIGN))
    run(config(*specs), callbacks=[callback])
    stats = [args[1] for _, hook, args in callback.calls if hook == "episode_end"]
    assert [set(episode) for episode in stats] == [{"s", "t"}] * 3
    assert [episode["s"] for episode in stats] == Stats.seen
    with pytest.raises(TypeError):
        stats[0]["x"] = stats[0]["s"]  # pyright: ignore[reportIndexIssue]  Reason: the test checks the mapping is read-only.


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
    def on_episode_end(self, episode: int, stats: Mapping[str, EpisodeStats]) -> None:
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
    match = "pass an instance" if isinstance(bad, type) else "Callback instances"
    with pytest.raises(TypeError, match=match):
        run(config(*two()), callbacks=[bad])  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.

    with pytest.raises(TypeError, match=match):
        run_many(config(*two()), seeds=[0], callbacks=[bad])  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


class OnlyEnd(Callback):
    """Overrides one hook only."""

    def __init__(self):
        self.logs: list[RunLog] = []

    @override
    def on_run_end(self, log: RunLog) -> None:
        self.logs.append(log)


def test_partial_callback():
    callback = OnlyEnd()
    log = run(config(*two()), callbacks=[callback])
    run_many(config(*two()), seeds=[0, 1], callbacks=[callback])
    assert len(callback.logs) == 1 and callback.logs[0] is log


def test_callbacks_leave_run_unchanged():
    plain = run(config(*two()))
    watched = run(config(*two()), callbacks=[Record(), OnlyEnd()])
    for name in ("bids", "price", "winner", "rewards"):
        assert (getattr(plain, name) == getattr(watched, name)).all()
