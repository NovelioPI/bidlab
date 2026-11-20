"""Objects passed to a bidder: what it sees before a bid and what it learns after."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0

# Bidders receive only these objects, so each holds one bidder's own data.


@dataclass(frozen=True)
class BidRequest:
    """What one bidder sees before it bids on one impression.

    Parameters
    ----------
    time : float
        Seconds since the episode start. Auctions are spaced evenly over the
        horizon, so the first is at 0.
    value : float
        The platform's value of this impression: `Campaign.event_value` times its
        estimate of the conversion probability. The estimate is exact unless
        `SimConfig.estimate_noise` is set. A bidder with its own model may
        ignore it. Not to be confused with `Campaign.event_value`, the value of one
        event.
    features : Mapping[str, str | float], optional
        Public features of the impression, the same for every bidder.
        `bidlab.markets.Segmented` puts the segment name under ``"segment"``.

    See Also
    --------
    Bidder.forward : Receives this object.

    Examples
    --------
    Build a request by hand to unit-test a bidder:

    >>> import bidlab as bl
    >>> request = bl.BidRequest(time=0.0, value=0.5)
    >>> bl.Linear(multiplier=0.8).forward(request.value, request)
    0.4
    """

    time: float
    value: float
    features: Mapping[str, str | float] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class BidResult:
    """What one bidder learns after an auction it took part in.

    A loser never sees the winning price.

    Parameters
    ----------
    time : float
        `BidRequest.time` of the auction this result answers.
    bid : float
        The bid the auction used: NaN and infinity become 0, and the bid is
        capped at the budget left.
    won : bool
        Whether this bidder won the impression.
    price : float
        Price this bidder paid; 0 when it lost.
    converted : bool
        True only when this bidder won and the impression converted.
    realized_value : float, optional
        Realized value of this auction: `Campaign.event_value` when converted,
        else 0.
    reward : float, optional
        This step's reward from the campaign's ``step_reward`` function.

    See Also
    --------
    Bidder.update_batch : Receives a list of these objects.

    Examples
    --------
    >>> import bidlab as bl
    >>> request = bl.BidRequest(time=0.0, value=0.5)
    >>> result = bl.BidResult(
    ...     time=0.0, bid=0.6, won=True, price=0.5, converted=True, realized_value=5.0
    ... )
    >>> bl.rewards.surplus(request, result, {})
    4.5
    """

    time: float
    bid: float
    won: bool
    price: float
    converted: bool
    realized_value: float = 0.0
    reward: float = 0.0


@dataclass(frozen=True)
class EpisodeReport:
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
    realized_value : float
        ``conversions`` times `Campaign.event_value`.

    See Also
    --------
    Bidder.on_episode_end : Receives this object.

    Examples
    --------
    >>> import bidlab as bl
    >>> report = bl.EpisodeReport(budget=20.0, spend=6.0, wins=4, conversions=2, realized_value=10.0)
    >>> bl.rewards.total_surplus(report, {})
    4.0
    """

    budget: float
    spend: float
    wins: int
    conversions: int
    realized_value: float
