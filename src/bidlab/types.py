from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

# Bidders receive only these objects, so each holds one bidder's own data.


@dataclass(frozen=True)
class Observation:
    step: int
    steps_left: int
    value: float
    value_per_conversion: float
    budget_left: float
    features: Mapping[str, str | float] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class Feedback:
    bid: float
    won: bool
    price: float
    converted: bool
    reward: float = 0.0


@dataclass(frozen=True)
class EpisodeStats:
    budget: float
    spend: float
    wins: int
    conversions: int
    value: float
