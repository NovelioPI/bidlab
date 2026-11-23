from typing import Any, ClassVar, override

import numpy as np
import pytest

from bidlab.auctions import FirstPrice
from bidlab.bidders import Bidder, Linear, Stack
from bidlab.config import Campaign, SimConfig
from bidlab.markets import Market, MarketDraw
from bidlab.simulator import BUILT_IN_INPUTS, run

SEED = 1


class FeatureMarket(Market):
    """Yields probability 0.5 and the given features at every step."""

    def __init__(self, features: dict[str, str | float]):
        self.features = features

    @override
    def draws(self, count: int, rng: np.random.Generator):
        while True:
            yield MarketDraw(self.features, pcvr=np.full(count, 0.5))


BID = 0.1


class Seen(Bidder):
    """Records the keyword arguments of every forward call."""

    calls: ClassVar[list[dict[str, Any]]] = []

    @override
    def forward(self, **inputs: Any) -> float:
        Seen.calls.append(dict(inputs))
        return BID


class TimeLeftOnly(Bidder):
    calls: ClassVar[list[dict[str, Any]]] = []

    @override
    def forward(self, time_left: float) -> float:
        TimeLeftOnly.calls.append(dict(locals()))
        return BID


def run_one(bidder: Bidder, **kwargs: Any):
    campaign = Campaign("a", bidder, budget=kwargs.pop("budget", 1e6), base_value=5.0)
    defaults: dict[str, Any] = {"episodes": 2, "steps": 10}
    return run(SimConfig(campaigns=[campaign], **(defaults | kwargs)), seed=SEED)


def test_forward_gets_only_declared_inputs():
    TimeLeftOnly.calls.clear()
    run_one(TimeLeftOnly())
    assert len(TimeLeftOnly.calls) == 20
    assert all(call.keys() == {"self", "time_left"} for call in TimeLeftOnly.calls)


def test_kwargs_gets_every_input():
    Seen.calls.clear()
    run_one(Seen(), market=FeatureMarket({"country": "id", "device": 1.0}))
    assert all(
        call.keys() == BUILT_IN_INPUTS | {"country", "device"} for call in Seen.calls
    )


class Spoils(Bidder):
    """Changes every input it gets."""

    @override
    def forward(self, **inputs: Any) -> float:
        for name in inputs:
            inputs[name] = -1.0

        return BID


def test_kwargs_dict_is_per_call():
    Seen.calls.clear()
    spoiled = Campaign("spoils", Spoils(), budget=1e6, base_value=5.0)
    seen = Campaign("seen", Stack(Spoils(), Seen()), budget=1e6, base_value=5.0)
    run(SimConfig(campaigns=[spoiled, seen], episodes=1, steps=5))
    assert [call["time"] for call in Seen.calls] == [
        k * 24 * 3600 / 5 for k in range(5)
    ]


def test_clock_inputs():
    Seen.calls.clear()
    run_one(Seen(), episodes=1, steps=4, horizon_hours=24, start_hour=22)
    assert [call["time"] for call in Seen.calls] == [0.0, 21600.0, 43200.0, 64800.0]
    assert [call["time_left"] for call in Seen.calls] == [
        86400.0,
        64800.0,
        43200.0,
        21600.0,
    ]
    assert {call["horizon"] for call in Seen.calls} == {86400.0}
    assert [call["hour_of_day"] for call in Seen.calls] == [22.0, 4.0, 10.0, 16.0]


class FixedBid(Seen):
    """Records its inputs and always bids 3."""

    @override
    def forward(self, **inputs: Any) -> float:
        super().forward(**inputs)
        return 3.0


def test_budget_left_input():
    Seen.calls.clear()
    run_one(FixedBid(), budget=10.0, steps=5, auction=FirstPrice())
    budgets = [call["budget_left"] for call in Seen.calls]
    assert budgets == [10.0, 7.0, 4.0, 1.0, 0.0] * 2


def test_unsegmented_segment_is_all():
    Seen.calls.clear()
    run_one(Seen())
    assert {call["segment"] for call in Seen.calls} == {"all"}


class Country(Bidder):
    countries: ClassVar[list[str]] = []

    @override
    def forward(self, country: str) -> float:
        Country.countries.append(country)
        return BID


def test_feature_input_by_name():
    Country.countries.clear()
    run_one(Country(), episodes=1, steps=3, market=FeatureMarket({"country": "id"}))
    assert Country.countries == ["id", "id", "id"]


def test_missing_required_input_raises():
    with pytest.raises(
        TypeError, match=r"Country.forward needs 'country'.*'budget_left'"
    ):
        run_one(Country())


class Optional(Bidder):
    countries: ClassVar[list[str]] = []

    @override
    def forward(self, country: str = "none") -> float:
        Optional.countries.append(country)
        return BID


def test_missing_input_uses_default():
    Optional.countries.clear()
    run_one(Optional(), episodes=1, steps=3)
    assert Optional.countries == ["none", "none", "none"]


@pytest.mark.parametrize("name", sorted(BUILT_IN_INPUTS))
def test_feature_named_like_builtin_raises(name):
    with pytest.raises(ValueError, match=rf"features \['{name}'\]"):
        run_one(Seen(), market=FeatureMarket({name: 1.0}))


def test_feature_named_value_rejected():
    with pytest.raises(ValueError, match=r"features \['value'\]"):
        run_one(Seen(), market=FeatureMarket({"value": 1.0}))


def test_no_value_input():
    Seen.calls.clear()
    run_one(Seen())
    assert len(Seen.calls) == 20
    assert not any("value" in call for call in Seen.calls)


def test_value_at_top_names_the_oracle():
    with pytest.raises(
        TypeError,
        match=r"Linear.forward needs 'value'.*bl\.Stack\(bl\.OracleValue\(\), ",
    ):
        run_one(Linear())


class PositionalOnly(Bidder):
    @override
    def forward(self, value: float, /) -> float:
        return value


def test_positional_only_raises():
    with pytest.raises(TypeError, match="'value' is positional-only"):
        run_one(PositionalOnly())
