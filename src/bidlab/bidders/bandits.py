"""Bandit bidders that learn a bid multiplier across episodes."""

from typing import override

import numpy as np

from bidlab.bidders.base import Bidder
from bidlab.types import EpisodeStats, Feedback, Observation

# np.arange excludes its end, so a small margin keeps max_multiplier in the grid.
ARM_GRID_END_MARGIN = 1e-9
# Keeps the exploration bonus above zero when every arm has the same mean.
MIN_REWARD_SCALE = 1e-12


class UcbMultiplier(Bidder):
    """UCB1 over bid multipliers; one arm per episode, learned across episodes.

    Each arm is a fixed multiplier m, and the bidder bids m × value for a
    whole episode. An arm's reward is the episode reward when the spec sets
    one, else the sum of step rewards. The exploration bonus is scaled by
    the spread of the arm means, so it works for any reward unit.

    Parameters
    ----------
    params : dict[str, float]
        ``min_multiplier`` (0.2), ``max_multiplier`` (2.0) and
        ``multiplier_step`` (0.2) set the arm grid. ``exploration`` (1.0)
        scales the bonus.
    rng : numpy.random.Generator
        Not used.

    Examples
    --------
    >>> import numpy as np
    >>> import bidlab as bl
    >>> bandit = bl.UcbMultiplier({"multiplier_step": 0.6}, np.random.default_rng(0))
    >>> bandit.arms.round(1).tolist()
    [0.2, 0.8, 1.4, 2.0]
    """

    def __init__(self, params: dict[str, float], rng: np.random.Generator):
        super().__init__(params, rng)
        self.arms = np.arange(
            params.get("min_multiplier", 0.2),
            params.get("max_multiplier", 2.0) + ARM_GRID_END_MARGIN,
            params.get("multiplier_step", 0.2),
        )
        self.exploration = params.get("exploration", 1.0)
        self.pulls = np.zeros(self.arms.size)
        self.reward_sums = np.zeros(self.arms.size)
        self.arm = 0
        self.episode_step_rewards = 0.0

    @override
    def on_episode_start(self, budget: float, steps: int) -> None:
        self.arm = self._choose_arm()
        self.episode_step_rewards = 0.0

    @override
    def bid(self, observation: Observation) -> float:
        return float(self.arms[self.arm]) * observation.value

    @override
    def update(self, feedback: Feedback) -> None:
        self.episode_step_rewards += feedback.reward

    @override
    def on_episode_end(self, stats: EpisodeStats, episode_reward: float | None) -> None:
        reward = self.episode_step_rewards if episode_reward is None else episode_reward
        self.pulls[self.arm] += 1
        self.reward_sums[self.arm] += reward

    def _choose_arm(self) -> int:
        untried = np.flatnonzero(self.pulls == 0)
        if untried.size:
            return int(untried[0])

        means = self.reward_sums / self.pulls
        # Scaling the bonus by the reward range makes it work for any reward unit.
        scale = max(means.max() - means.min(), MIN_REWARD_SCALE)
        bonus = scale * np.sqrt(2 * np.log(self.pulls.sum()) / self.pulls)
        return int(np.argmax(means + self.exploration * bonus))
