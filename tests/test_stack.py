from typing import Any, ClassVar

import numpy as np
import pytest

from bidlab import simulator
from bidlab.bidders import Bidder, Linear, OracleValue, PidPacing, Stack
from bidlab.config import Campaign, SimConfig
from bidlab.markets import LogisticMarket
from bidlab.simulator import run
from bidlab.types import MarketTruth

SEED = 3

# A first bidder in a stack gets no value, so pass-through test bidders default to one.
FIRST_VALUE = 1.0


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

    def on_episode_start(self, campaign):
        Calls.calls.append((self.name, "start", campaign))

    def forward(self, value=FIRST_VALUE, **inputs):
        Calls.calls.append((self.name, "forward", inputs))
        return value

    def update_batch(self, bid_results):
        Calls.calls.append((self.name, "batch", tuple(bid_results)))

    def on_episode_end(self, report):
        Calls.calls.append((self.name, "end", report))


class Holds(Bidder):
    """Holds a model object and passes values through."""

    def __init__(self, model):
        self.model = model
        self.forwards = 0

    def forward(self, value=FIRST_VALUE):
        self.forwards += 1
        return value


class Counter(Bidder):
    """Counts forward calls and reports the count at each episode end."""

    reports: ClassVar[list[int]] = []

    def __init__(self):
        self.forwards = 0

    def forward(self, value=FIRST_VALUE):
        self.forwards += 1
        return value

    def on_episode_end(self, report):
        Counter.reports.append(self.forwards)


class Draws(Bidder):
    """Draws one number per auction from its own stream, if asked to."""

    draws: ClassVar[dict[str, list[float]]] = {}

    def __init__(self, name, draw=True):
        self.name = name
        self.draw = draw

    def forward(self, value=FIRST_VALUE):
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

    def forward(self, value=FIRST_VALUE):
        return value * self.rng.uniform(0.5, 1.5)


def config(*campaigns, **kwargs):
    defaults: dict[str, Any] = {"episodes": 2, "steps": 50}
    return SimConfig(campaigns=campaigns, **(defaults | kwargs))


def other():
    return Campaign(
        "other",
        Stack(OracleValue(), Linear(multiplier=0.9)),
        budget=1e6,
        base_value=4.0,
    )


def test_stages_chain_values():
    first, second = Add(1.0), Linear(multiplier=3.0)
    stack = Stack(first, second)
    stack.bind(np.random.default_rng(0))
    assert stack.forward(value=2.0) == 9.0
    assert first.seen == [2.0]


class AddTimeLeft(Bidder):
    def forward(self, value, time_left):
        return value + time_left


def test_stack_chains_value_and_matches_inputs():
    stack = Stack(Add(1.0), AddTimeLeft())
    assert stack.forward(value=2.0, time_left=10.0, segment="all") == 13.0


def test_stacks_nest():
    inner = Stack(Add(1.0), Linear(multiplier=2.0))
    outer = Stack(Add(1.0), inner)
    outer.bind(np.random.default_rng(0))
    assert outer.forward(value=1.0) == 6.0


def test_hooks_reach_every_stage_in_order():
    Calls.calls.clear()
    stack = Stack(Calls("a"), Calls("b"))
    run(
        config(Campaign("x", stack, budget=1e6, base_value=5.0), episodes=1, steps=2),
        seed=SEED,
    )
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
    campaign = Campaign("counted", Counter(), budget=1e6, base_value=5.0)
    run(config(campaign, other()), seed=range(3))
    assert Counter.reports == [50, 100] * 3


def test_user_stage_object_unchanged():
    model = {"weights": [1.0, 2.0]}
    bidder = Holds(model)
    run(
        config(Campaign("m", Stack(bidder), budget=1e6, base_value=5.0), other()),
        seed=range(2),
    )
    assert bidder.forwards == 0
    assert bidder.model is model and model == {"weights": [1.0, 2.0]}
    assert not hasattr(bidder, "rng")


def test_stage_streams_are_separate():
    Draws.draws.clear()
    drawing = Stack(Draws("first"), Draws("second"))
    log = run(
        config(Campaign("d", drawing, budget=1e6, base_value=5.0), other()), seed=SEED
    )
    first, second = Draws.draws["first"], Draws.draws["second"]
    assert len(first) == len(second) == 100
    assert first != second

    quiet = Stack(Draws("first", draw=False), Draws("second", draw=False))
    quiet_log = run(
        config(Campaign("d", quiet, budget=1e6, base_value=5.0), other()), seed=SEED
    )
    np.testing.assert_array_equal(log.true_values, quiet_log.true_values)
    np.testing.assert_array_equal(log.bids[:, :, 1], quiet_log.bids[:, :, 1])
    np.testing.assert_array_equal(log.converted, quiet_log.converted)


def test_single_stage_equals_stack():
    single = run(
        config(
            Campaign(
                "p", Stack(OracleValue(), PidPacing()), budget=3.0, base_value=5.0
            ),
            other(),
        ),
        seed=SEED,
    )
    nested = Stack(Stack(OracleValue(), PidPacing()))
    stacked = run(
        config(Campaign("p", nested, budget=3.0, base_value=5.0), other()), seed=SEED
    )
    for name in ("bids", "true_values", "won", "price", "converted"):
        np.testing.assert_array_equal(getattr(single, name), getattr(stacked, name))


def test_stage_class_rejected():
    with pytest.raises(TypeError, match=r"instance, such as PidPacing\(\)"):
        Campaign("p", PidPacing, budget=1e6, base_value=5.0)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_stack_parallel_equals_serial():
    campaign = Campaign("j", Stack(Jitter(), PidPacing()), budget=5.0, base_value=5.0)
    serial = run(config(campaign, other()), seed=range(3))
    parallel = run(config(campaign, other()), seed=range(3), workers=2)
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
    run(config(Campaign("d", nested, budget=1e6, base_value=5.0), other()), seed=SEED)
    assert {name: draws[:3] for name, draws in Draws.draws.items()} == expected


def test_bidder_draws_keep_oracle_noise_stream():
    seen: dict[bool, list[float]] = {}
    for draw in (True, False):
        drawing = Stack(Draws("first", draw=draw), OracleValue(noise=0.5), Watch())
        Watch.values.clear()
        run(
            config(Campaign("d", drawing, budget=1e6, base_value=5.0), other()),
            seed=SEED,
        )
        seen[draw] = list(Watch.values)

    assert len(set(seen[True])) > 1
    assert seen[True] == seen[False]


class Records(Bidder):
    """Records the inputs of each forward call and outputs a fixed number."""

    def __init__(self, output):
        self.output = output
        self.calls = []

    def forward(self, **inputs):
        self.calls.append(inputs)
        return self.output


def test_stack_passes_value_on():
    a, b, c = Records(1.0), Records(2.0), Records(3.0)
    assert Stack(a, b, c).forward(time=0.0) == 3.0
    assert a.calls == [{"time": 0.0}]
    assert b.calls == [{"time": 0.0, "value": 1.0}]
    assert c.calls == [{"time": 0.0, "value": 2.0}]


def test_first_bidder_gets_no_value():
    campaign = Campaign("a", Stack(Linear()), budget=1e6, base_value=5.0)
    with pytest.raises(
        TypeError,
        match=r"Linear.forward needs 'value'.*bl\.Stack\(bl\.OracleValue\(\), ",
    ):
        run(config(campaign), seed=SEED)


class Fixed(Bidder):
    def forward(self):
        return 0.2


def count_market_truths(monkeypatch, bidder):
    built = []

    def counting(*args):
        built.append(args)
        return MarketTruth(*args)

    monkeypatch.setattr(simulator, "MarketTruth", counting)
    run(config(Campaign("a", bidder, budget=1e6, base_value=5.0)), seed=SEED)
    return len(built)


def test_no_market_truth_without_reader(monkeypatch):
    assert count_market_truths(monkeypatch, Stack(Fixed(), Linear())) == 0
    assert count_market_truths(monkeypatch, Stack(OracleValue(), Linear())) == 100


class ReadsMarket(Bidder):
    """Records the truth it gets, in a class-level list: runs use a copy."""

    seen: ClassVar[list[MarketTruth]] = []

    def forward(self, value, market):
        ReadsMarket.seen.append(market)
        return value


def test_deep_bidder_gets_market():
    ReadsMarket.seen.clear()
    bidder = Stack(Fixed(), Stack(Linear(), ReadsMarket()))
    campaign = Campaign("a", bidder, budget=1e6, base_value=5.0)
    log = run(config(campaign, market=LogisticMarket(ctr_bias_mean=-2.0)), seed=SEED)
    values = [5.0 * truth.pctr * truth.pcvr for truth in ReadsMarket.seen]
    assert values == log.true_values[:, :, 0].ravel().tolist()
