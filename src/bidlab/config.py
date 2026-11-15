from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from bidlab import rewards
from bidlab.auctions import Auction, FirstPrice, SecondPrice
from bidlab.bidders.base import Bidder
from bidlab.rewards import EpisodeReward, StepReward
from bidlab.world import WorldConfig

BidderFactory = Callable[[dict[str, float], np.random.Generator], Bidder]


@dataclass(frozen=True)
class BidderSpec:
    name: str
    bidder: BidderFactory
    budget: float
    value_per_conversion: float
    cpa_target: float | None = None
    params: dict[str, float] = field(default_factory=dict[str, float])
    reward: StepReward = rewards.surplus
    episode_reward: EpisodeReward | None = None
    reward_params: dict[str, float] = field(default_factory=dict[str, float])

    def __post_init__(self) -> None:
        if isinstance(self.bidder, str):
            raise TypeError(
                f"bidder {self.name!r}: got the string {self.bidder!r}; "
                "pass a Bidder subclass or a factory f(params, rng) -> Bidder"
            )

        for field_name in ("bidder", "reward"):
            if not callable(getattr(self, field_name)):
                raise TypeError(f"bidder {self.name!r}: {field_name} must be callable")

        if self.episode_reward is not None and not callable(self.episode_reward):
            raise TypeError(
                f"bidder {self.name!r}: episode_reward must be callable or None"
            )

        # An int budget would make integer numpy arrays that reject float prices.
        for field_name in ("budget", "value_per_conversion", "cpa_target"):
            number = getattr(self, field_name)
            if number is None:
                continue

            if number <= 0:
                raise ValueError(
                    f"bidder {self.name!r}: {field_name} must be > 0, got {number}"
                )

            object.__setattr__(self, field_name, float(number))

    def merged_reward_params(self) -> dict[str, float]:
        if self.cpa_target is None:
            return dict(self.reward_params)

        return {"cpa_target": self.cpa_target, **self.reward_params}


@dataclass(frozen=True)
class SimConfig:
    bidders: tuple[BidderSpec, ...]
    episodes: int = 10
    steps: int = 1000
    seed: int = 0
    auction: Auction = field(default_factory=SecondPrice)
    world: WorldConfig = field(default_factory=WorldConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bidders", tuple(self.bidders))
        if not self.bidders:
            raise ValueError("bidders must list at least one BidderSpec")

        names = [spec.name for spec in self.bidders]
        if len(set(names)) != len(names):
            raise ValueError(f"bidder names must be unique, got {names}")

        for field_name in ("episodes", "steps"):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be >= 1")

        # Callers can pass any object at run time; the type hint does not check it.
        auction: object = self.auction
        if not isinstance(auction, FirstPrice | SecondPrice):
            raise TypeError("auction must be FirstPrice or SecondPrice")
