from typing import Any, ClassVar

import numpy as np
import pytest

from bidlab.bidders import Bidder, PidPacing, Stack, Truthful
from bidlab.config import BidderSpec, SimConfig
from bidlab.runner import run_many
from bidlab.simulator import run
from bidlab.types import Campaign, Observation

CAMPAIGN = Campaign(budget=1e6, value=5.0)


class Add(Bidder):
    """Adds a constant and records the values it gets."""

    def __init__(self, amount):
        self.amount = amount
        self.seen = []

    def forward(self, value, observation):
        self.seen.append(value)
        return value + self.amount


class Calls(Bidder):
    """Records every hook call, with its arguments, in a class-level list."""

    calls: ClassVar[list[tuple[str, str, Any]]] = []

    def __init__(self, name):
        self.name = name

    def on_episode_start(self, campaign):
        Calls.calls.append((self.name, "start", campaign))

    def forward(self, value, observation):
        Calls.calls.append((self.name, "forward", observation))
        return value

    def update_batch(self, feedbacks):
        Calls.calls.append((self.name, "batch", tuple(feedbacks)))

    def on_episode_end(self, stats, episode_reward):
        Calls.calls.append((self.name, "end", stats))


class Holds(Bidder):
    """Holds a model object and passes values through."""

    def __init__(self, model):
        self.model = model
        self.forwards = 0

    def forward(self, value, observation):
        self.forwards += 1
        return value


class Counter(Bidder):
    """Counts forward calls and reports the count at each episode end."""

    reports: ClassVar[list[int]] = []

    def __init__(self):
        self.forwards = 0

    def forward(self, value, observation):
        self.forwards += 1
        return value

    def on_episode_end(self, stats, episode_reward):
        Counter.reports.append(self.forwards)


class Draws(Bidder):
    """Draws one number per auction from its own stream, if asked to."""

    draws: ClassVar[dict[str, list[float]]] = {}

    def __init__(self, name, draw=True):
        self.name = name
        self.draw = draw

    def forward(self, value, observation):
        if self.draw:
            Draws.draws.setdefault(self.name, []).append(self.rng.random())

        return value


class Watch(Bidder):
    """Records each observation's value."""

    values: ClassVar[list[float]] = []

    def forward(self, value, observation):
        Watch.values.append(observation.value)
        return value


class Jitter(Bidder):
    """Multiplies the value by a random factor from its own stream."""

    def forward(self, value, observation):
        return value * self.rng.uniform(0.5, 1.5)


def config(*specs, **kwargs):
    defaults: dict[str, Any] = {"episodes": 2, "steps": 50, "seed": 3}
    return SimConfig(bidders=specs, **(defaults | kwargs))


def other():
    return BidderSpec("other", Truthful(multiplier=0.9), Campaign(1e6, 4.0))


def test_stages_chain_values():
    first, second = Add(1.0), Truthful(multiplier=3.0)
    stack = Stack(first, second)
    stack.bind(np.random.default_rng(0))
    assert stack.forward(2.0, Observation(time=0.0, value=2.0)) == 9.0
    assert first.seen == [2.0]


def test_stacks_nest():
    inner = Stack(Add(1.0), Truthful(multiplier=2.0))
    outer = Stack(Add(1.0), inner)
    outer.bind(np.random.default_rng(0))
    assert outer.forward(1.0, Observation(time=0.0, value=1.0)) == 6.0


def test_hooks_reach_every_stage_in_order():
    Calls.calls.clear()
    stack = Stack(Calls("a"), Calls("b"))
    run(config(BidderSpec("x", stack, CAMPAIGN), episodes=1, steps=2))
    assert [(name, hook) for name, hook, _ in Calls.calls] == [
        ("a", "start"),
        ("b", "start"),
        ("a", "forward"),
        ("b", "forward"),
        ("a", "batch"),
        ("b", "batch"),
        ("a", "forward"),
        ("b", "forward"),
        ("a", "batch"),
        ("b", "batch"),
        ("a", "end"),
        ("b", "end"),
    ]
    pairs = zip(Calls.calls[::2], Calls.calls[1::2], strict=True)
    for (_, hook_a, args_a), (_, hook_b, args_b) in pairs:
        assert hook_a == hook_b
        assert args_a is args_b or args_a == args_b


def test_stack_rejects_empty_and_non_stage():
    with pytest.raises(ValueError, match="at least one bidder"):
        Stack()

    with pytest.raises(
        TypeError, match=r"Bidder instances.*Truthful; pass Truthful\(\)"
    ):
        Stack(Truthful)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.

    with pytest.raises(TypeError, match="Bidder instances, got str"):
        Stack("pid")  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_each_run_starts_fresh():
    Counter.reports.clear()
    spec = BidderSpec("counted", Counter(), CAMPAIGN)
    run_many(config(spec, other()), seeds=range(3))
    assert Counter.reports == [50, 100] * 3


def test_user_stage_object_unchanged():
    model = {"weights": [1.0, 2.0]}
    bidder = Holds(model)
    run_many(config(BidderSpec("m", Stack(bidder), CAMPAIGN), other()), seeds=range(2))
    assert bidder.forwards == 0
    assert bidder.model is model and model == {"weights": [1.0, 2.0]}
    assert not hasattr(bidder, "rng")


def test_stage_streams_are_separate():
    Draws.draws.clear()
    drawing = Stack(Draws("first"), Draws("second"))
    log = run(config(BidderSpec("d", drawing, CAMPAIGN), other()))
    first, second = Draws.draws["first"], Draws.draws["second"]
    assert len(first) == len(second) == 100
    assert first != second

    quiet = Stack(Draws("first", draw=False), Draws("second", draw=False))
    quiet_log = run(config(BidderSpec("d", quiet, CAMPAIGN), other()))
    np.testing.assert_array_equal(log.values, quiet_log.values)
    np.testing.assert_array_equal(log.bids[:, :, 1], quiet_log.bids[:, :, 1])
    np.testing.assert_array_equal(log.converted, quiet_log.converted)


def test_single_stage_equals_stack():
    campaign = Campaign(budget=3.0, value=5.0)
    single = run(config(BidderSpec("p", PidPacing(), campaign), other()))
    stacked = run(config(BidderSpec("p", Stack(PidPacing()), campaign), other()))
    for name in ("bids", "values", "winner", "price", "converted", "rewards"):
        np.testing.assert_array_equal(getattr(single, name), getattr(stacked, name))


def test_stage_class_rejected():
    with pytest.raises(TypeError, match=r"instance, such as PidPacing\(\)"):
        BidderSpec("p", PidPacing, CAMPAIGN)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_stack_parallel_equals_serial():
    spec = BidderSpec("j", Stack(Jitter(), PidPacing()), Campaign(5.0, 5.0))
    serial = run_many(config(spec, other()), seeds=range(3))
    parallel = run_many(config(spec, other()), seeds=range(3), workers=2)
    assert parallel.equals(serial)


def test_bidder_draws_keep_noise_stream():
    seen: dict[bool, list[float]] = {}
    for draw in (True, False):
        drawing = Stack(Draws("first", draw=draw), Watch())
        Watch.values.clear()
        run(config(BidderSpec("d", drawing, CAMPAIGN), other(), estimate_noise=0.5))
        seen[draw] = list(Watch.values)
    assert seen[True] == seen[False]
