"""Objects passed to a bidder: what it sees before a bid and what it learns after."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

# Bidders receive only these objects, so each holds one bidder's own data.


@dataclass(frozen=True)
class Observation:
    """What one bidder sees before it bids on one impression.

    Parameters
    ----------
    step : int
        Index of the step in the episode, starting at 0.
    steps_left : int
        Steps left in the episode, counting this one.
    value : float
        Expected value of the impression to this bidder:
        ``value_per_conversion`` times its conversion probability.
    value_per_conversion : float
        What one conversion is worth to this bidder.
    budget_left : float
        Budget left in this episode. A higher bid is capped at this amount.
    features : Mapping[str, str | float], optional
        Public features of the impression, the same for every bidder.
        `bidlab.markets.Segmented` puts the segment name under ``"segment"``.

    See Also
    --------
    Bidder.bid : Receives this object.

    Examples
    --------
    Build an observation by hand to unit-test a bidder:

    >>> import numpy as np
    >>> import bidlab as bl
    >>> observation = bl.Observation(
    ...     step=0, steps_left=100, value=0.5, value_per_conversion=5.0, budget_left=20.0
    ... )
    >>> bl.Truthful({"multiplier": 0.8}, np.random.default_rng(0)).bid(observation)
    0.4
    """

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
    """What one bidder learns after an auction it took part in.

    A loser never sees the winning price.

    Parameters
    ----------
    bid : float
        The bid the auction used: NaN and infinity become 0, and the bid is
        capped at the budget left.
    won : bool
        Whether this bidder won the impression.
    price : float
        Price this bidder paid; 0 when it lost.
    converted : bool
        True only when this bidder won and the impression converted.
    reward : float, optional
        This step's reward from the bidder's ``reward`` function.

    See Also
    --------
    Bidder.update : Receives this object.

    Examples
    --------
    >>> import bidlab as bl
    >>> observation = bl.Observation(
    ...     step=0, steps_left=100, value=0.5, value_per_conversion=5.0, budget_left=20.0
    ... )
    >>> feedback = bl.Feedback(bid=0.6, won=True, price=0.5, converted=True)
    >>> bl.rewards.surplus(observation, feedback, {})
    4.5
    """

    bid: float
    won: bool
    price: float
    converted: bool
    reward: float = 0.0


@dataclass(frozen=True)
class EpisodeStats:
    """One bidder's totals for one episode.

    Parameters
    ----------
    budget : float
        The bidder's budget for the episode.
    spend : float
        Sum of the prices the bidder paid.
    wins : int
        Number of impressions the bidder won.
    conversions : int
        Number of won impressions that converted.
    value : float
        ``conversions`` times the bidder's value per conversion.

    See Also
    --------
    Bidder.on_episode_end : Receives this object.

    Examples
    --------
    >>> import bidlab as bl
    >>> stats = bl.EpisodeStats(budget=20.0, spend=6.0, wins=4, conversions=2, value=10.0)
    >>> bl.rewards.total_surplus(stats, {})
    4.0
    """

    budget: float
    spend: float
    wins: int
    conversions: int
    value: float
