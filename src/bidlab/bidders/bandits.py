import numpy as np

from bidlab.bidders.base import Bidder
from bidlab.types import EpisodeStats, Feedback, Observation

# np.arange excludes its end, so a small margin keeps max_multiplier in the grid.
ARM_GRID_END_MARGIN = 1e-9
# Keeps the exploration bonus above zero when every arm has the same mean.
MIN_REWARD_SCALE = 1e-12


class UcbMultiplier(Bidder):
    """UCB1 over bid multipliers; one arm per episode, learned across episodes."""

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

    def on_episode_start(self, budget: float, steps: int) -> None:
        self.arm = self._choose_arm()
        self.episode_step_rewards = 0.0

    def bid(self, observation: Observation) -> float:
        return float(self.arms[self.arm]) * observation.value

    def update(self, feedback: Feedback) -> None:
        self.episode_step_rewards += feedback.reward

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
