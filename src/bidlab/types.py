"""Objects passed to a bidder: market facts, and what it learns after each auction and episode."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0


class Event(StrEnum):
    """What a campaign pays for, and so what its `Campaign.base_value` counts.

    A campaign also takes the plain string, such as ``event="click"``, and
    stores the member. A member equals its string.

    Attributes
    ----------
    IMPRESSION
        A won impression. Expected value: ``base_value``.
    CLICK
        A clicked won impression. Expected value: ``base_value`` × pCTR.
    CONVERSION
        A converted won impression, the default. Expected value:
        ``base_value`` × pCTR × pCVR.

    Examples
    --------
    >>> import bidlab as bl
    >>> campaign = bl.Campaign(
    ...     "traffic",
    ...     bl.Stack(bl.OracleValue(), bl.Linear()),
    ...     budget=10,
    ...     base_value=0.4,
    ...     event="click",
    ... )
    >>> campaign.event is bl.Event.CLICK, campaign.event == "click"
    (True, True)
    """

    IMPRESSION = "impression"
    CLICK = "click"
    CONVERSION = "conversion"


type EventName = Event | Literal["impression", "click", "conversion"]


# Bidders receive only these objects, so each holds one bidder's own data.


@dataclass(frozen=True)
class MarketTruth:
    """The market's hidden facts for one campaign at one auction.

    A bidder gets it as the ``market`` input of `Bidder.forward`, at every
    auction. A fair benchmark bidder does not read it; an oracle bidder or
    a test of an estimator can.

    Parameters
    ----------
    pctr : float
        True chance that this campaign's impression is clicked if won; 1.0
        when the market has no click stage.
    pcvr : float
        True chance that a click converts.

    Examples
    --------
    Pass one by hand to unit-test a bidder that reads ``market``:

    >>> import bidlab as bl
    >>> class Oracle(bl.Bidder):
    ...     def forward(self, market):
    ...         return 5.0 * market.pctr * market.pcvr
    >>> Oracle().forward(bl.MarketTruth(pctr=0.2, pcvr=0.5))
    0.5
    """

    pctr: float
    pcvr: float


@dataclass(frozen=True)
class BidResult:
    """What one bidder learns after an auction it took part in.

    A loser never sees the winning price.

    Parameters
    ----------
    time : float
        The ``time`` input of the auction this result answers.
    bid : float
        The bid the auction used: NaN and infinity become 0, and the bid is
        capped at the budget left.
    won : bool
        Whether this bidder won the impression.
    price : float
        Price this bidder paid; 0 when it lost.
    clicked : bool
        True only when this bidder won and the impression was clicked. In a
        market without a click stage, every won impression is clicked.
    converted : bool
        True only when this bidder won and the impression converted. A
        conversion needs a click.
    realized_value : float, optional
        Realized value of this auction: `Campaign.base_value` when the
        campaign's event happened, else 0. A won impression, a clicked one or
        a converted one, by `Campaign.event`.

    See Also
    --------
    Bidder.update_batch : Receives a list of these objects.

    Examples
    --------
    A bidder computes its own reward from a result, such as the surplus:

    >>> import bidlab as bl
    >>> result = bl.BidResult(
    ...     time=0.0,
    ...     bid=0.6,
    ...     won=True,
    ...     price=0.5,
    ...     clicked=True,
    ...     converted=True,
    ...     realized_value=5.0,
    ... )
    >>> result.realized_value - result.price
    4.5
    """

    time: float
    bid: float
    won: bool
    price: float
    clicked: bool
    converted: bool
    realized_value: float = 0.0


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
    clicks : int
        Number of won impressions that were clicked.
    conversions : int
        Number of won impressions that converted.
    realized_value : float
        The count of the campaign's event times `Campaign.base_value`: wins,
        ``clicks`` or ``conversions``, by `Campaign.event`.

    See Also
    --------
    Bidder.on_episode_end : Receives this object.

    Examples
    --------
    >>> import bidlab as bl
    >>> report = bl.EpisodeReport(
    ...     budget=20.0, spend=6.0, wins=4, clicks=3, conversions=2, realized_value=10.0
    ... )
    >>> report.realized_value - report.spend
    4.0
    """

    budget: float
    spend: float
    wins: int
    clicks: int
    conversions: int
    realized_value: float
