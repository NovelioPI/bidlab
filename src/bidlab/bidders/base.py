"""The `Bidder` base class that every bidding method subclasses."""

from abc import ABC, abstractmethod

import numpy as np

from bidlab.types import EpisodeStats, Feedback, Observation


class Bidder(ABC):
    """Base class: implement `bid`; override any hook a method needs.

    One bidder object lives for the whole run, so a learning bidder keeps
    what it learned from one episode to the next. The simulator calls, in
    order: `on_episode_start`, then `bid` and `update` once per step, then
    `on_episode_end`. Outside the spec's ``segments``, `bid` and `update`
    are not called.

    Parameters
    ----------
    params : dict[str, float]
        The spec's ``params``, for tuning values such as a multiplier.
    rng : numpy.random.Generator
        This bidder's own random stream. Use it for every random choice, so
        runs stay reproducible and other bidders' draws stay unchanged.

    See Also
    --------
    BidderSpec : Puts a bidder in a simulation.

    Examples
    --------
    >>> import bidlab as bl
    >>> class HalfValue(bl.Bidder):
    ...     def bid(self, observation):
    ...         return self.params.get("fraction", 0.5) * observation.value
    >>> config = bl.SimConfig(
    ...     bidders=[bl.BidderSpec("half", HalfValue, budget=10, value_per_conversion=5)],
    ...     episodes=1,
    ...     steps=50,
    ... )
    >>> bl.run(config).bids.shape
    (1, 50, 1)
    """

    def __init__(self, params: dict[str, float], rng: np.random.Generator):
        self.params = params
        self.rng = rng

    def on_episode_start(self, budget: float, steps: int) -> None:
        """Reset per-episode state before the first step of an episode.

        Parameters
        ----------
        budget : float
            The bidder's budget for this episode.
        steps : int
            Number of steps in the episode.
        """

    @abstractmethod
    def bid(self, observation: Observation) -> float:
        """Return the bid for one impression.

        Parameters
        ----------
        observation : Observation
            What this bidder can see about the impression.

        Returns
        -------
        float
            The bid. NaN or infinity becomes 0, and a bid above
            ``observation.budget_left`` is capped at it.
        """

    def update(self, feedback: Feedback) -> None:
        """Learn from the outcome of an auction this bidder took part in.

        Parameters
        ----------
        feedback : Feedback
            This bidder's own outcome. It holds the price only when the
            bidder won.
        """

    def on_episode_end(self, stats: EpisodeStats, episode_reward: float | None) -> None:
        """Learn from the episode's totals after its last step.

        Parameters
        ----------
        stats : EpisodeStats
            This bidder's totals for the episode.
        episode_reward : float or None
            The spec's ``episode_reward`` for this episode, or None when the
            spec sets none.
        """
