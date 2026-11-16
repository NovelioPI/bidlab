"""The `Bidder` base class that every bidding method builds on."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from bidlab.types import Campaign, EpisodeStats, Feedback, Observation


class Bidder(ABC):
    """Base class for a bidding method: implement `forward`.

    A bidder turns a value into another value. Alone, it gets the platform's
    value, `Observation.value`, and its output is the bid. In a `Stack`, each
    bidder gets the previous one's output. A value model, such as a pCVR
    model, returns its own value and may ignore the one it gets. A
    bid-control method, such as pacing or shading, turns a value into a bid.

    Take settings as constructor arguments, and keep learned state on
    ``self``. The simulator deep-copies the bidder once per run, so every
    run starts from the bidder as built, and the bidder keeps what it learns
    from one episode to the next. The simulator calls, in order:
    `on_episode_start`, then `forward` and `update_batch` once per auction,
    then `on_episode_end`. Outside the campaign's ``segments``, `forward` and
    `update_batch` are not called.

    Attributes
    ----------
    rng : numpy.random.Generator
        This bidder's own random stream, set by `bind` before the run. Use it
        for every random choice, so runs stay reproducible.

    See Also
    --------
    Stack : Chains bidders.
    BidderSpec : Puts a bidder in a simulation.

    Examples
    --------
    >>> import bidlab as bl
    >>> class Floor(bl.Bidder):
    ...     def __init__(self, minimum):
    ...         self.minimum = minimum
    ...     def forward(self, value, observation):
    ...         return max(value, self.minimum)
    >>> bidder = bl.Stack(bl.Truthful(multiplier=0.5), Floor(0.3))
    >>> spec = bl.BidderSpec("floored", bidder, bl.Campaign(budget=10, value=5))
    >>> log = bl.run(bl.SimConfig(bidders=[spec], episodes=1, steps=20))
    >>> bool((log.bids >= 0.3).all())
    True
    """

    rng: np.random.Generator

    def bind(self, rng: np.random.Generator) -> None:
        """Give this bidder its random stream; the simulator calls this per run.

        Parameters
        ----------
        rng : numpy.random.Generator
            The bidder's random stream for the run.
        """
        self.rng = rng

    def on_episode_start(self, campaign: Campaign) -> None:
        """Reset per-episode state before the first auction of an episode.

        Parameters
        ----------
        campaign : Campaign
            The bidder's fixed facts, with the schedule filled in.
        """

    @abstractmethod
    def forward(self, value: float, observation: Observation) -> float:
        """Turn the incoming value into this bidder's output.

        Parameters
        ----------
        value : float
            `Observation.value`, or the previous bidder's output in a
            `Stack`.
        observation : Observation
            What this bidder can see about the impression.

        Returns
        -------
        float
            The bid, or the value passed to the next bidder in a `Stack`. A
            final bid of NaN or infinity becomes 0, and a bid above the
            budget left is capped at it.
        """

    def update_batch(self, feedbacks: Sequence[Feedback]) -> None:
        """Learn from the outcomes of auctions this bidder took part in.

        Today each batch holds the feedback of one auction. The default calls
        `update` once per item, in time order.

        Parameters
        ----------
        feedbacks : Sequence[Feedback]
            This bidder's own outcomes. Each holds the price only when the
            bidder won. In a `Stack`, `Feedback.bid` is the final bid.
        """
        for feedback in feedbacks:
            self.update(feedback)

    def update(self, feedback: Feedback) -> None:
        """Learn from the outcome of one auction this bidder took part in.

        Parameters
        ----------
        feedback : Feedback
            This bidder's own outcome.
        """

    def on_episode_end(self, stats: EpisodeStats, episode_reward: float | None) -> None:
        """Learn from the episode's totals after its last auction.

        Parameters
        ----------
        stats : EpisodeStats
            This bidder's totals for the episode.
        episode_reward : float or None
            The spec's ``episode_reward`` for this episode, or None when the
            spec sets none.
        """
