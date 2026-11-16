"""Objects passed to a bidder: what it sees before a bid and what it learns after."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0

# Bidders receive only these objects, so each holds one bidder's own data.


@dataclass(frozen=True)
class Campaign:
    """An advertiser's fixed facts, given to its bidder once per episode.

    The simulator fills ``horizon_hours`` and ``start_hour`` from `SimConfig`
    before it passes the campaign to `Bidder.on_episode_start`.

    Parameters
    ----------
    budget : float
        Budget per episode. Must be > 0.
    value : float
        What one event is worth to the advertiser. The only event today is a
        conversion. Must be > 0. Not to be confused with `Observation.value`,
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
    >>> campaign = bl.Campaign(budget=15, value=5, cpa_target=3)
    >>> campaign.budget, campaign.horizon_hours
    (15.0, None)
    """

    budget: float
    value: float
    cpa_target: float | None = None
    segments: Sequence[str] | None = None
    horizon_hours: float | None = None
    start_hour: float | None = None

    def __post_init__(self) -> None:
        # An int budget would make integer numpy arrays that reject float prices.
        for field_name in ("budget", "value", "cpa_target", "horizon_hours"):
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
            Seconds since the episode start, as in `Observation.time`.

        Raises
        ------
        ValueError
            If the simulator has not set the schedule yet.

        Examples
        --------
        >>> import bidlab as bl
        >>> bl.Campaign(10, 5, horizon_hours=24, start_hour=0).time_left(3600.0)
        82800.0
        """
        horizon_hours, _ = self._schedule()
        return horizon_hours * SECONDS_PER_HOUR - time

    def hour_of_day(self, time: float) -> float:
        """Return the hour of day at ``time``, in [0, 24).

        Parameters
        ----------
        time : float
            Seconds since the episode start, as in `Observation.time`.

        Raises
        ------
        ValueError
            If the simulator has not set the schedule yet.

        Examples
        --------
        >>> import bidlab as bl
        >>> bl.Campaign(10, 5, horizon_hours=24, start_hour=22).hour_of_day(3 * 3600.0)
        1.0
        """
        _, start_hour = self._schedule()
        return (start_hour + time / SECONDS_PER_HOUR) % HOURS_PER_DAY

    def _schedule(self) -> tuple[float, float]:
        if self.horizon_hours is None or self.start_hour is None:
            raise ValueError(
                "the simulator sets horizon_hours and start_hour; call this on "
                "the campaign passed to on_episode_start"
            )

        return self.horizon_hours, self.start_hour


@dataclass(frozen=True)
class Observation:
    """What one bidder sees before it bids on one impression.

    Parameters
    ----------
    time : float
        Seconds since the episode start. Auctions are spaced evenly over the
        horizon, so the first is at 0.
    value : float
        The platform's value of this impression: `Campaign.value` times its
        estimate of the conversion probability. The estimate is exact unless
        `SimConfig.estimate_noise` is set. A bidder with its own model may
        ignore it. Not to be confused with `Campaign.value`, the value of one
        event.
    features : Mapping[str, str | float], optional
        Public features of the impression, the same for every bidder.
        `bidlab.markets.Segmented` puts the segment name under ``"segment"``.

    See Also
    --------
    Bidder.forward : Receives this object.

    Examples
    --------
    Build an observation by hand to unit-test a bidder:

    >>> import bidlab as bl
    >>> observation = bl.Observation(time=0.0, value=0.5)
    >>> bl.Truthful(multiplier=0.8).forward(observation.value, observation)
    0.4
    """

    time: float
    value: float
    features: Mapping[str, str | float] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class Feedback:
    """What one bidder learns after an auction it took part in.

    A loser never sees the winning price.

    Parameters
    ----------
    time : float
        `Observation.time` of the auction this feedback answers.
    bid : float
        The bid the auction used: NaN and infinity become 0, and the bid is
        capped at the budget left.
    won : bool
        Whether this bidder won the impression.
    price : float
        Price this bidder paid; 0 when it lost.
    converted : bool
        True only when this bidder won and the impression converted.
    value : float, optional
        Realized value of this auction: `Campaign.value` when converted,
        else 0.
    reward : float, optional
        This step's reward from the bidder's ``reward`` function.

    See Also
    --------
    Bidder.update_batch : Receives a list of these objects.

    Examples
    --------
    >>> import bidlab as bl
    >>> observation = bl.Observation(time=0.0, value=0.5)
    >>> feedback = bl.Feedback(
    ...     time=0.0, bid=0.6, won=True, price=0.5, converted=True, value=5.0
    ... )
    >>> bl.rewards.surplus(observation, feedback, {})
    4.5
    """

    time: float
    bid: float
    won: bool
    price: float
    converted: bool
    value: float = 0.0
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
        ``conversions`` times `Campaign.value`.

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
