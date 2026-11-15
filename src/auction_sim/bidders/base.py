import numpy as np

from auction_sim.types import EpisodeStats, Feedback, Observation


class Bidder:
    """Base class: override `bid`, and any hook a method needs."""

    def __init__(self, params: dict[str, float], rng: np.random.Generator):
        self.params = params
        self.rng = rng

    def on_episode_start(self, budget: float, steps: int) -> None:
        pass

    def bid(self, obs: Observation) -> float:
        raise NotImplementedError

    def update(self, feedback: Feedback) -> None:
        pass

    def on_episode_end(self, stats: EpisodeStats, episode_reward: float | None) -> None:
        pass
