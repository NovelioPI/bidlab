from typing import Any, ClassVar

import numpy as np
import pytest

from bidlab.bidders import Bidder, Linear, PidPacing, Stack
from bidlab.config import Campaign, SimConfig
from bidlab.runner import run_many
from bidlab.simulator import run
from bidlab.types import Brief

CAMPAIGN = Brief(budget=1e6, event_value=5.0)


class Add(Bidder):
    """Adds a constant and records the values it gets."""

    def __init__(self, amount):
        self.amount = amount
        self.seen = []

    def forward(self, value):
        self.seen.append(value)
        return value + self.amount


class Calls(Bidder):
    """Records every hook call, with its arguments, in a class-level list."""

    calls: ClassVar[list[tuple[str, str, Any]]] = []

    def __init__(self, name):
        self.name = name

    def on_episode_start(self, brief):
        Calls.calls.append((self.name, "start", brief))

    def forward(self, value, **inputs):
        Calls.calls.append((self.name, "forward", inputs))
        return value

    def update_batch(self, bid_results):
        Calls.calls.append((self.name, "batch", tuple(bid_results)))

    def on_episode_end(self, report, episode_reward):
        Calls.calls.append((self.name, "end", report))


class Holds(Bidder):
    """Holds a model object and passes values through."""

    def __init__(self, model):
        self.model = model
        self.forwards = 0

    def forward(self, value):
        self.forwards += 1
        return value


class Counter(Bidder):
    """Counts forward calls and reports the count at each episode end."""

    reports: ClassVar[list[int]] = []

    def __init__(self):
        self.forwards = 0

    def forward(self, value):
        self.forwards += 1
        return value

    def on_episode_end(self, report, episode_reward):
        Counter.reports.append(self.forwards)


class Draws(Bidder):
    """Draws one number per auction from its own stream, if asked to."""

    draws: ClassVar[dict[str, list[float]]] = {}

    def __init__(self, name, draw=True):
        self.name = name
        self.draw = draw

    def forward(self, value):
        if self.draw:
            Draws.draws.setdefault(self.name, []).append(self.rng.random())

        return value


class Watch(Bidder):
    """Records each value it gets."""

    values: ClassVar[list[float]] = []

    def forward(self, value):
        Watch.values.append(value)
        return value


class Jitter(Bidder):
    """Multiplies the value by a random factor from its own stream."""

    def forward(self, value):
        return value * self.rng.uniform(0.5, 1.5)


def config(*campaigns, **kwargs):
    defaults: dict[str, Any] = {"episodes": 2, "steps": 50, "seed": 3}
    return SimConfig(campaigns=campaigns, **(defaults | kwargs))


def other():
    return Campaign("other", Linear(multiplier=0.9), budget=1e6, event_value=4.0)


def test_stages_chain_values():
    first, second = Add(1.0), Linear(multiplier=3.0)
    stack = Stack(first, second)
    stack.bind(np.random.default_rng(0))
    assert stack.forward(2.0) == 9.0
    assert first.seen == [2.0]


class AddTimeLeft(Bidder):
    def forward(self, value, time_left):
        return value + time_left


def test_stack_chains_value_and_matches_inputs():
    stack = Stack(Add(1.0), AddTimeLeft())
    assert stack.forward(2.0, time_left=10.0, segment="all") == 13.0


def test_stacks_nest():
    inner = Stack(Add(1.0), Linear(multiplier=2.0))
    outer = Stack(Add(1.0), inner)
    outer.bind(np.random.default_rng(0))
    assert outer.forward(1.0) == 6.0


def test_hooks_reach_every_stage_in_order():
    Calls.calls.clear()
    stack = Stack(Calls("a"), Calls("b"))
    run(config(Campaign("x", stack, budget=1e6, event_value=5.0), episodes=1, steps=2))
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

    with pytest.raises(TypeError, match=r"Bidder instances.*Linear; pass Linear\(\)"):
        Stack(Linear)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.

    with pytest.raises(TypeError, match="Bidder instances, got str"):
        Stack("pid")  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_stack_defines_only_init_and_forward():
    hooks = {"bind", "on_episode_start", "update_batch", "update", "on_episode_end"}
    assert {"__init__", "forward"} <= set(vars(Stack))
    assert not hooks & set(vars(Stack))


def test_each_run_starts_fresh():
    Counter.reports.clear()
    campaign = Campaign("counted", Counter(), budget=1e6, event_value=5.0)
    run_many(config(campaign, other()), seeds=range(3))
    assert Counter.reports == [50, 100] * 3


def test_user_stage_object_unchanged():
    model = {"weights": [1.0, 2.0]}
    bidder = Holds(model)
    run_many(
        config(Campaign("m", Stack(bidder), budget=1e6, event_value=5.0), other()),
        seeds=range(2),
    )
    assert bidder.forwards == 0
    assert bidder.model is model and model == {"weights": [1.0, 2.0]}
    assert not hasattr(bidder, "rng")


def test_stage_streams_are_separate():
    Draws.draws.clear()
    drawing = Stack(Draws("first"), Draws("second"))
    log = run(config(Campaign("d", drawing, budget=1e6, event_value=5.0), other()))
    first, second = Draws.draws["first"], Draws.draws["second"]
    assert len(first) == len(second) == 100
    assert first != second

    quiet = Stack(Draws("first", draw=False), Draws("second", draw=False))
    quiet_log = run(config(Campaign("d", quiet, budget=1e6, event_value=5.0), other()))
    np.testing.assert_array_equal(log.true_values, quiet_log.true_values)
    np.testing.assert_array_equal(log.bids[:, :, 1], quiet_log.bids[:, :, 1])
    np.testing.assert_array_equal(log.converted, quiet_log.converted)


def test_single_stage_equals_stack():
    single = run(
        config(Campaign("p", PidPacing(), budget=3.0, event_value=5.0), other())
    )
    stacked = run(
        config(Campaign("p", Stack(PidPacing()), budget=3.0, event_value=5.0), other())
    )
    for name in ("bids", "true_values", "winner", "price", "converted", "rewards"):
        np.testing.assert_array_equal(getattr(single, name), getattr(stacked, name))


def test_stage_class_rejected():
    with pytest.raises(TypeError, match=r"instance, such as PidPacing\(\)"):
        Campaign("p", PidPacing, budget=1e6, event_value=5.0)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_stack_parallel_equals_serial():
    campaign = Campaign("j", Stack(Jitter(), PidPacing()), budget=5.0, event_value=5.0)
    serial = run_many(config(campaign, other()), seeds=range(3))
    parallel = run_many(config(campaign, other()), seeds=range(3), workers=2)
    assert parallel.equals(serial)


def test_stack_streams_match_main():
    # Recorded on main 60c1412, before the tree walk took over binding.
    expected = {
        "a": [0.9412261510533382, 0.5963327018170845, 0.9598601871506219],
        "b": [0.9945804582113029, 0.09643783630573322, 0.941043133979684],
        "c": [0.7616405784268819, 0.2761865794161845, 0.5232990510378713],
    }
    Draws.draws.clear()
    nested = Stack(Draws("a"), Stack(Draws("b"), Draws("c")))
    run(config(Campaign("d", nested, budget=1e6, event_value=5.0), other()))
    assert {name: draws[:3] for name, draws in Draws.draws.items()} == expected


def test_bidder_draws_keep_noise_stream():
    seen: dict[bool, list[float]] = {}
    for draw in (True, False):
        drawing = Stack(Draws("first", draw=draw), Watch())
        Watch.values.clear()
        run(
            config(
                Campaign("d", drawing, budget=1e6, event_value=5.0),
                other(),
                estimate_noise=0.5,
            )
        )
        seen[draw] = list(Watch.values)

    assert seen[True] == seen[False]
