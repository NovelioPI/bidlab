"""Bandit bidders that learn a bid multiplier across episodes."""

from typing import override

import numpy as np

from bidlab.bidders.base import Bidder
from bidlab.types import Campaign, EpisodeStats, Feedback, Observation

# np.arange excludes its end, so a small margin keeps max_multiplier in the grid.
ARM_GRID_END_MARGIN = 1e-9
# Keeps the exploration bonus above zero when every arm has the same mean.
MIN_REWARD_SCALE = 1e-12


class UcbMultiplier(Bidder):
    """UCB1 over bid multipliers; one arm per episode, learned across episodes.

    Each arm is a fixed multiplier m, and the bidder outputs m × value for a
    whole episode. An arm's reward is the episode reward when the spec sets
    one, else the sum of step rewards. The exploration bonus is scaled by
    the spread of the arm means, so it works for any reward unit.

    Parameters
    ----------
    min_multiplier, max_multiplier : float, optional
        The arm grid's ends, with 0 <= ``min_multiplier`` <= ``max_multiplier``.
    multiplier_step : float, optional
        The grid's spacing. Must be > 0.
    exploration : float, optional
        Scales the exploration bonus. Must be >= 0.
    log_state : bool, optional
        Log the chosen arm's multiplier as ``"multiplier"`` once per episode.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> import bidlab as bl
    >>> bl.UcbMultiplier(multiplier_step=0.6).arms.round(1).tolist()
    [0.2, 0.8, 1.4, 2.0]
    """

    def __init__(
        self,
        min_multiplier: float = 0.2,
        max_multiplier: float = 2.0,
        multiplier_step: float = 0.2,
        exploration: float = 1.0,
        log_state: bool = True,
    ):
        if not 0 <= min_multiplier <= max_multiplier:
            raise ValueError(
                "need 0 <= min_multiplier <= max_multiplier, got "
                f"{min_multiplier} and {max_multiplier}"
            )

        if multiplier_step <= 0:
            raise ValueError(f"multiplier_step must be > 0, got {multiplier_step}")

        if exploration < 0:
            raise ValueError(f"exploration must be >= 0, got {exploration}")

        self.arms = np.arange(
            min_multiplier, max_multiplier + ARM_GRID_END_MARGIN, multiplier_step
        )
        self.exploration = exploration
        self.log_state = log_state
        self.pulls = np.zeros(self.arms.size)
        self.reward_sums = np.zeros(self.arms.size)
        self.arm = 0
        self.episode_step_rewards = 0.0

    @override
    def on_episode_start(self, campaign: Campaign) -> None:
        self.arm = self._choose_arm()
        self.episode_step_rewards = 0.0
        if self.log_state:
            self.log("multiplier", float(self.arms[self.arm]))

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return float(self.arms[self.arm]) * value

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
