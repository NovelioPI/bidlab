from abc import ABC, abstractmethod

import numpy as np

from bidlab.types import EpisodeStats, Feedback, Observation


class Bidder(ABC):
    """Base class: implement `bid`; override any hook a method needs."""

    def __init__(self, params: dict[str, float], rng: np.random.Generator):
        self.params = params
        self.rng = rng

    def on_episode_start(self, budget: float, steps: int) -> None:
        pass

    @abstractmethod
    def bid(self, observation: Observation) -> float: ...

    def update(self, feedback: Feedback) -> None:
        pass

    def on_episode_end(self, stats: EpisodeStats, episode_reward: float | None) -> None:
        pass
