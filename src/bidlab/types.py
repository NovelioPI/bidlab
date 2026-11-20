"""Objects passed to a bidder: what it sees before a bid and what it learns after."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0

# Bidders receive only these objects, so each holds one bidder's own data.


@dataclass(frozen=True)
class Brief:
    """A campaign's facts that its bidder sees, given once per episode.

    The simulator builds it with `Campaign.brief`, with ``horizon_hours`` and
    ``start_hour`` from `SimConfig`, and passes it to
    `Bidder.on_episode_start`. Build one by hand only to unit-test a bidder.

    Parameters
    ----------
    budget : float
        Budget per episode. Must be > 0.
    event_value : float
        What one event is worth to the advertiser. The only event today is a
        conversion. Must be > 0. Not to be confused with `BidRequest.value`,
        the value of one impression.
    cpa_target : float, optional
        Highest acceptable cost per conversion. Reward functions get it as
        ``params["cpa_target"]``. Must be > 0 when set.
    segments : Sequence[str], optional
        Market segments the campaign bids in, stored as a tuple. None means
        every segment.
    horizon_hours : float, optional
        Episode length in hours. Leave it None; the simulator sets it.
    start_hour : float, optional
        Hour of day when the episode starts, in [0, 24). Leave it None; the
        simulator sets it.

    Raises
    ------
    TypeError
        If ``segments`` is a string.
    ValueError
        If a number is out of range or ``segments`` is empty.

    Examples
    --------
    >>> import bidlab as bl
    >>> brief = bl.Brief(budget=15, event_value=5, horizon_hours=24, start_hour=0)
    >>> brief.budget, brief.time_left(0.0)
    (15.0, 86400.0)
    """

    budget: float
    event_value: float
    cpa_target: float | None = None
    segments: Sequence[str] | None = None
    horizon_hours: float | None = None
    start_hour: float | None = None

    def __post_init__(self) -> None:
        # An int budget would make integer numpy arrays that reject float prices.
        for field_name in ("budget", "event_value", "cpa_target", "horizon_hours"):
            number = getattr(self, field_name)
            if number is None:
                continue

            if number <= 0:
                raise ValueError(f"{field_name} must be > 0, got {number}")

            object.__setattr__(self, field_name, float(number))

        if self.start_hour is not None:
            if not 0 <= self.start_hour < HOURS_PER_DAY:
                raise ValueError(
                    f"start_hour must be in [0, 24), got {self.start_hour}"
                )

            object.__setattr__(self, "start_hour", float(self.start_hour))

        self._check_segments()

    def _check_segments(self) -> None:
        if self.segments is None:
            return

        if isinstance(self.segments, str):
            raise TypeError(
                f"segments must be a tuple of names, got the string {self.segments!r}"
            )

        object.__setattr__(self, "segments", tuple(self.segments))
        if not self.segments:
            raise ValueError(
                "segments must list at least one segment; use None to bid everywhere"
            )

    def time_left(self, time: float) -> float:
        """Return the seconds left in the episode at ``time``.

        Parameters
        ----------
        time : float
            Seconds since the episode start, as in `BidRequest.time`.

        Raises
        ------
        ValueError
            If the simulator has not set the schedule yet.

        Examples
        --------
        >>> import bidlab as bl
        >>> bl.Brief(10, 5, horizon_hours=24, start_hour=0).time_left(3600.0)
        82800.0
        """
        horizon_hours, _ = self._schedule()
        return horizon_hours * SECONDS_PER_HOUR - time

    def hour_of_day(self, time: float) -> float:
        """Return the hour of day at ``time``, in [0, 24).

        Parameters
        ----------
        time : float
            Seconds since the episode start, as in `BidRequest.time`.

        Raises
        ------
        ValueError
            If the simulator has not set the schedule yet.

        Examples
        --------
        >>> import bidlab as bl
        >>> bl.Brief(10, 5, horizon_hours=24, start_hour=22).hour_of_day(3 * 3600.0)
        1.0
        """
        _, start_hour = self._schedule()
        return (start_hour + time / SECONDS_PER_HOUR) % HOURS_PER_DAY

    def _schedule(self) -> tuple[float, float]:
        if self.horizon_hours is None or self.start_hour is None:
            raise ValueError(
                "the simulator sets horizon_hours and start_hour; call this on "
                "the brief passed to on_episode_start"
            )

        return self.horizon_hours, self.start_hour


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
