from dataclasses import dataclass

# Reason: bidders receive only these objects, so each holds one bidder's own data.


@dataclass(frozen=True)
class Observation:
    step: int
    steps_left: int
    value: float
    value_per_conversion: float
    budget_left: float


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
