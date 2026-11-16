from typing import override

import pytest

from bidlab.bidders import Bidder, PidPacing, Stack, Truthful
from bidlab.types import Campaign, Observation

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


def test_containers_without_bidders_allowed():
    bidder = Holder([{"a": 1.0}, (2.0,), {3.0}])
    assert paths(bidder) == [""]


def test_named_bidders_binds_nothing():
    parent = Parent()
    nodes = [node for _, node in parent.named_bidders()]
    assert nodes[0] is parent
    assert nodes[1] is parent.shade
    assert not any(hasattr(node, "rng") for node in nodes)
