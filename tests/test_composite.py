from collections.abc import Sequence
from typing import Any, ClassVar, override

import numpy as np
import pytest

from bidlab.bidders import Bidder, PidPacing, Stack, Truthful
from bidlab.config import BidderSpec, SimConfig
from bidlab.markets import Segment, Segmented
from bidlab.runner import run_many
from bidlab.simulator import run
from bidlab.types import Campaign, EpisodeStats, Feedback, Observation

CAMPAIGN = Campaign(budget=1e6, value=5.0)


class Pass(Bidder):
    """Passes the value through."""

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return value


class Holder(Bidder):
    """Holds whatever it is given in one attribute, ``held``."""

    def __init__(self, held: object):
        self.held = held

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return value


class Parent(Bidder):
    """Holds a child in an attribute and children in a list among other items."""

    def __init__(self):
        self.shade = Pass()
        self.rate = 0.5
        self.parts = [Truthful(), "x", PidPacing()]

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return value


def paths(bidder: Bidder) -> list[str]:
    return [path for path, _ in bidder.named_bidders()]


def test_paths_follow_attributes_and_items():
    assert paths(Parent()) == ["", "shade", "parts.0", "parts.2"]


def test_nested_child_path():
    outer = Holder(Holder(Pass()))
    assert paths(outer) == ["", "held", "held.held"]
    assert paths(Stack(Pass(), outer)) == [
        "",
        "bidders.0",
        "bidders.1",
        "bidders.1.held",
        "bidders.1.held.held",
    ]


def test_shared_child_and_cycle_listed_once():
    shared = Pass()
    parent = Holder([shared, shared])
    assert paths(parent) == ["", "held.0"]
    assert paths(Stack(Holder(shared), shared)) == [
        "",
        "bidders.0",
        "bidders.1",
    ]

    child = Holder(None)
    loop = Holder(child)
    child.held = loop
    assert paths(loop) == ["", "held"]


@pytest.mark.parametrize(
    "held",
    [
        {"model": Pass()},
        {Pass()},
        frozenset([Pass()]),
        [[Pass()]],
        (1.0, {"model": Pass()}),
    ],
    ids=["dict", "set", "frozenset", "list-in-list", "dict-in-tuple"],
)
def test_hidden_child_raises(held: object):
    bidder = Holder(held)
    with pytest.raises(TypeError, match=r"'held(\.\d)?'.*attribute, list or tuple"):
        paths(bidder)


@pytest.mark.parametrize(
    "held", [{"model": Pass()}, [[Pass()]]], ids=["dict", "list-in-list"]
)
def test_hidden_child_raises_at_build(held: object):
    spec = BidderSpec("h", Holder(held), CAMPAIGN)
    with pytest.raises(TypeError, match="attribute, list or tuple"):
        spec.build(np.random.default_rng(0))


def test_containers_without_bidders_allowed():
    bidder = Holder([{"a": 1.0}, (2.0,), {3.0}])
    assert paths(bidder) == [""]


def test_named_bidders_binds_nothing():
    parent = Parent()
    nodes = [node for _, node in parent.named_bidders()]
    assert nodes[0] is parent
    assert nodes[1] is parent.shade
    assert not any(hasattr(node, "rng") for node in nodes)


class Log(Bidder):
    """Records every tree hook call, with its name and arguments."""

    calls: ClassVar[list[tuple[str, str, Any]]] = []

    def __init__(self, name: str):
        self.name = name

    @override
    def on_episode_start(self, campaign: Campaign) -> None:
        Log.calls.append((self.name, "start", campaign))

    @override
    def forward(self, value: float, observation: Observation) -> float:
        Log.calls.append((self.name, "forward", observation))
        return value

    @override
    def update_batch(self, feedbacks: Sequence[Feedback]) -> None:
        Log.calls.append((self.name, "batch", feedbacks))

    @override
    def on_episode_end(self, stats: EpisodeStats, episode_reward: float | None) -> None:
        Log.calls.append((self.name, "end", stats))


class LogParent(Log):
    """Logs its own hooks, never calls super(), and never calls its children."""

    def __init__(self):
        super().__init__("parent")
        self.a = Log("a")
        self.b = Log("b")


class QuietParent(Bidder):
    """Overrides every tree hook with nothing and never calls super()."""

    def __init__(self):
        self.a = Log("a")
        self.b = Log("b")

    @override
    def bind(self, rng: np.random.Generator) -> None:
        self.rng = rng

    @override
    def on_episode_start(self, campaign: Campaign) -> None:
        pass

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return self.b.forward(self.a.forward(value, observation), observation)

    @override
    def update_batch(self, feedbacks: Sequence[Feedback]) -> None:
        pass

    @override
    def on_episode_end(self, stats: EpisodeStats, episode_reward: float | None) -> None:
        pass


class Draws(Bidder):
    """Draws one number per auction from its own stream, if asked to."""

    draws: ClassVar[dict[str, list[float]]] = {}

    def __init__(self, name: str, draw: bool = True):
        self.name = name
        self.draw = draw

    @override
    def forward(self, value: float, observation: Observation) -> float:
        if self.draw:
            Draws.draws.setdefault(self.name, []).append(self.rng.random())

        return value


class Watch(Bidder):
    """Records each observation's value."""

    values: ClassVar[list[float]] = []

    @override
    def forward(self, value: float, observation: Observation) -> float:
        Watch.values.append(observation.value)
        return value


class Shade(Bidder):
    """Multiplies the value by ``factor``, which its parent may pass."""

    @override
    def forward(
        self, value: float, observation: Observation, *, factor: float = 1.0
    ) -> float:
        return value * factor


class HalfShade(Bidder):
    """Passes ``factor=0.5`` to its child on every auction."""

    def __init__(self):
        self.shade = Shade()

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return self.shade.forward(value, observation, factor=0.5)


class Fan(Bidder):
    """Calls each child's forward and passes the value through."""

    def __init__(self, children: list[Bidder]):
        self.children = children

    @override
    def forward(self, value: float, observation: Observation) -> float:
        for child in self.children:
            child.forward(value, observation)

        return value


class Jitter(Bidder):
    """Multiplies the value by a random factor from its own stream."""

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return value * self.rng.uniform(0.5, 1.5)


class Mixer(Bidder):
    """Averages two random children and counts its forward calls."""

    def __init__(self):
        self.parts = (Jitter(), Jitter())
        self.forwards = 0

    @override
    def forward(self, value: float, observation: Observation) -> float:
        self.forwards += 1
        first, second = (part.forward(value, observation) for part in self.parts)
        return (first + second) / 2


class Counts(Bidder):
    """Counts its forward and update_batch calls in a class-level dict."""

    counts: ClassVar[dict[str, list[int]]] = {}

    def __init__(self, name: str):
        self.name = name

    @override
    def forward(self, value: float, observation: Observation) -> float:
        Counts.counts.setdefault(self.name, [0, 0])[0] += 1
        return value

    @override
    def update_batch(self, feedbacks: Sequence[Feedback]) -> None:
        Counts.counts.setdefault(self.name, [0, 0])[1] += 1


class Unused(Counts):
    """Holds a child whose forward it never calls."""

    def __init__(self):
        super().__init__("parent")
        self.child = Counts("child")


class Bids(Bidder):
    """Records every feedback bid it gets."""

    bids: ClassVar[list[float]] = []

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return value

    @override
    def update(self, feedback: Feedback) -> None:
        Bids.bids.append(feedback.bid)


class HalfOfChild(Bidder):
    """Bids half of what its child outputs."""

    def __init__(self):
        self.child = Bids()

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return self.child.forward(value, observation) / 2


class Spawning(Draws):
    """Spawns two streams of its own in bind, as a user override might."""

    def __init__(self, name: str):
        super().__init__(name, draw=False)
        self.child = Draws("child")

    @override
    def bind(self, rng: np.random.Generator) -> None:
        super().bind(rng)
        self.own = rng.spawn(2)


class Grows(Bidder):
    """Adds a child bidder in the hook named ``hook``."""

    def __init__(self, hook: str):
        self.hook = hook

    def grow(self, hook: str) -> None:
        if hook == self.hook:
            self.late = Pass()

    @override
    def on_episode_start(self, campaign: Campaign) -> None:
        self.grow("on_episode_start")

    @override
    def forward(self, value: float, observation: Observation) -> float:
        self.grow("forward")
        return value

    @override
    def update_batch(self, feedbacks: Sequence[Feedback]) -> None:
        self.grow("update_batch")

    @override
    def on_episode_end(self, stats: EpisodeStats, episode_reward: float | None) -> None:
        self.grow("on_episode_end")


class Tally(Bidder):
    """Changes plain attributes in every hook."""

    def __init__(self):
        self.seen = 0
        self.history: list[float] = []

    @override
    def forward(self, value: float, observation: Observation) -> float:
        self.seen += 1
        self.history.append(value)
        self.last = observation
        return value


def other() -> BidderSpec:
    return BidderSpec("other", Truthful(multiplier=0.9), Campaign(1e6, 4.0))


def config(*specs: BidderSpec, **kwargs: Any) -> SimConfig:
    defaults: dict[str, Any] = {"episodes": 2, "steps": 50, "seed": 3}
    return SimConfig(bidders=specs, **(defaults | kwargs))


def test_every_node_has_own_stream():
    tree = Stack(Pass(), Holder([Pass(), Holder(Pass())]))
    built = BidderSpec("t", tree, CAMPAIGN).build(np.random.default_rng(0))
    draws = [tuple(node.rng.random(100)) for _, node in built.named_bidders()]
    assert len(draws) == 6
    assert len(set(draws)) == 6


def test_child_draws_leave_others_unchanged():
    logs, seen = [], []
    for draw in (True, False):
        Draws.draws.clear()
        Watch.values.clear()
        tree = Fan([Draws("child", draw=draw), Fan([Draws("deep", draw=draw)])])
        stack = Stack(tree, Watch())
        logs.append(
            run(config(BidderSpec("d", stack, CAMPAIGN), other(), estimate_noise=0.5))
        )
        seen.append(list(Watch.values))
        if draw:
            assert [len(Draws.draws[name]) for name in ("child", "deep")] == [100, 100]
    drawing, quiet = logs
    np.testing.assert_array_equal(drawing.bids[:, :, 1], quiet.bids[:, :, 1])
    np.testing.assert_array_equal(drawing.values, quiet.values)
    np.testing.assert_array_equal(drawing.converted, quiet.converted)
    assert seen[0] == seen[1]


def test_composite_parallel_equals_serial():
    spec = BidderSpec("m", Stack(Mixer(), PidPacing()), Campaign(5.0, 5.0))
    serial = run_many(config(spec, other()), seeds=range(3))
    parallel = run_many(config(spec, other()), seeds=range(3), workers=2)
    assert parallel.equals(serial)


def test_user_tree_unchanged():
    mixer = Mixer()
    run_many(config(BidderSpec("m", Stack(mixer), CAMPAIGN), other()), seeds=range(2))
    assert mixer.forwards == 0
    assert not any(hasattr(node, "rng") for _, node in mixer.named_bidders())


def test_tree_hook_order():
    Log.calls.clear()
    run(config(BidderSpec("x", LogParent(), CAMPAIGN), episodes=1, steps=2))
    auction = [
        ("parent", "forward"),
        ("parent", "batch"),
        ("a", "batch"),
        ("b", "batch"),
    ]
    assert [(name, hook) for name, hook, _ in Log.calls] == [
        ("parent", "start"),
        ("a", "start"),
        ("b", "start"),
        *auction,
        *auction,
        ("parent", "end"),
        ("a", "end"),
        ("b", "end"),
    ]
    batches = [args for _, hook, args in Log.calls if hook == "batch"]
    assert batches[0] is batches[1] is batches[2]
    assert batches[3] is batches[4] is batches[5]


def test_children_get_hooks_without_super():
    Log.calls.clear()
    run(config(BidderSpec("x", QuietParent(), CAMPAIGN), episodes=1, steps=2))
    auction = [("a", "forward"), ("b", "forward"), ("a", "batch"), ("b", "batch")]
    assert [(name, hook) for name, hook, _ in Log.calls] == [
        ("a", "start"),
        ("b", "start"),
        *auction,
        *auction,
        ("a", "end"),
        ("b", "end"),
    ]


def test_child_feedback_has_final_bid():
    Bids.bids.clear()
    log = run(config(BidderSpec("h", HalfOfChild(), CAMPAIGN), other()))
    assert Bids.bids == log.bids[:, :, 0].ravel().tolist()
    assert Bids.bids == (log.values[:, :, 0].ravel() / 2).tolist()


def test_unused_child_still_learns():
    Counts.counts.clear()
    run(config(BidderSpec("u", Unused(), CAMPAIGN), other()))
    assert Counts.counts == {"parent": [100, 100], "child": [0, 100]}


def test_tree_skipped_outside_segments():
    Counts.counts.clear()
    market = Segmented((Segment("news", 0.6), Segment("sports", 0.4, lift=1.0)))
    campaign = Campaign(1e6, 5.0, segments=["news"])
    log = run(config(BidderSpec("u", Unused(), campaign), other(), market=market))
    news = int(np.sum(log.segment == market.segment_names.index("news")))
    assert 0 < news < 100
    assert Counts.counts == {"parent": [news, news], "child": [0, news]}


def test_bind_override_keeps_child_streams():
    seen = []
    for parent in (Spawning("parent"), Holder(Draws("child"))):
        Draws.draws.clear()
        built = BidderSpec("s", parent, CAMPAIGN).build(np.random.default_rng(7))
        child = next(node for path, node in built.named_bidders() if path)
        assert isinstance(child, Draws)
        child.forward(1.0, Observation(time=0.0, value=1.0))
        seen.append(Draws.draws["child"])
    assert seen[0] == seen[1]

    rng = np.random.default_rng(7)
    built = BidderSpec("s", Spawning("parent"), CAMPAIGN).build(rng)
    assert built.rng is rng


@pytest.mark.parametrize(
    "hook", ["on_episode_start", "forward", "update_batch", "on_episode_end"]
)
def test_late_child_raises(hook: str):
    spec = BidderSpec("grower", Grows(hook), CAMPAIGN)
    with pytest.raises(ValueError, match=r"'grower'.*'late'.*__init__"):
        run(config(spec, other(), episodes=1))


def test_plain_attribute_changes_allowed():
    log = run(config(BidderSpec("t", Tally(), CAMPAIGN), other()))
    assert log.bids.shape == (2, 50, 2)


def test_parent_passes_forward_argument():
    log = run(config(BidderSpec("h", HalfShade(), CAMPAIGN), other()))
    np.testing.assert_array_equal(log.bids[:, :, 0], log.values[:, :, 0] * 0.5)
