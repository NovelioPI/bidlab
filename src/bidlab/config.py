"""Frozen, validated settings for one simulation."""

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from bidlab import rewards
from bidlab.auctions import Auction, FirstPrice, SecondPrice
from bidlab.bidders.base import Bidder, bind_tree
from bidlab.markets import LogisticContext, Market
from bidlab.rewards import EpisodeReward, StepReward
from bidlab.types import HOURS_PER_DAY, Brief


@dataclass(frozen=True)
class Campaign:
    """One advertiser's campaign: its bidder, budget, goals and rewards.

    Parameters
    ----------
    name : str
        Label used in results; unique within a `SimConfig`.
    bidder : Bidder
        A bidder instance, such as ``bl.PidPacing()`` or a `Stack`. The
        simulator deep-copies it once per run, so the object you pass never
        changes and every run starts from it.
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
    step_reward : StepReward, optional
        Called every step, ``f(request, result, params) -> float``.
        Defaults to `bidlab.rewards.surplus`.
    episode_reward : EpisodeReward, optional
        Called at the end of each episode, ``g(report, params) -> float``.
    reward_params : dict[str, float], optional
        Passed to both reward functions.

    Raises
    ------
    TypeError
        If ``bidder`` is not a `Bidder` instance, ``segments`` is a string,
        or a reward is not callable.
    ValueError
        If a number is out of range or ``segments`` is empty.

    Examples
    --------
    >>> import bidlab as bl
    >>> campaign = bl.Campaign(
    ...     "pid", bl.PidPacing(), budget=15, event_value=5, cpa_target=3,
    ...     episode_reward=bl.rewards.cpa_score,
    ... )
    >>> campaign.budget
    15.0
    """

    name: str
    bidder: Bidder
    budget: float
    event_value: float
    cpa_target: float | None = None
    segments: Sequence[str] | None = None
    step_reward: StepReward = rewards.surplus
    episode_reward: EpisodeReward | None = None
    reward_params: dict[str, float] = field(default_factory=dict[str, float])

    def __post_init__(self) -> None:
        # Callers can pass any object at run time; the type hint does not check it.
        bidder: object = self.bidder
        if isinstance(bidder, type):
            raise TypeError(
                f"campaign {self.name!r}: got the class {bidder.__name__}; "
                f"pass an instance, such as {bidder.__name__}()"
            )

        if not isinstance(bidder, Bidder):
            raise TypeError(
                f"campaign {self.name!r}: bidder must be a Bidder instance, "
                f"got {type(bidder).__name__}"
            )

        # Brief owns the checks, so a campaign and its brief cannot disagree.
        brief = self.brief()
        for field_name in ("budget", "event_value", "cpa_target", "segments"):
            object.__setattr__(self, field_name, getattr(brief, field_name))

        if not callable(self.step_reward):
            raise TypeError(f"campaign {self.name!r}: step_reward must be callable")

        if self.episode_reward is not None and not callable(self.episode_reward):
            raise TypeError(
                f"campaign {self.name!r}: episode_reward must be callable or None"
            )

    def brief(
        self, horizon_hours: float | None = None, start_hour: float | None = None
    ) -> Brief:
        """Return the facts the bidder gets in `Bidder.on_episode_start`.

        Parameters
        ----------
        horizon_hours, start_hour : float, optional
            The schedule; the simulator passes `SimConfig`'s.

        Returns
        -------
        Brief
            This campaign's budget, value, CPA target and segments, with the
            schedule.
        """
        return Brief(
            budget=self.budget,
            event_value=self.event_value,
            cpa_target=self.cpa_target,
            segments=self.segments,
            horizon_hours=horizon_hours,
            start_hour=start_hour,
        )

    def build(self, rng: np.random.Generator) -> Bidder:
        """Return a fresh copy of the bidder for one run; the simulator calls this.

        Parameters
        ----------
        rng : numpy.random.Generator
            The bidder's random stream for the run.

        Returns
        -------
        Bidder
            A deep copy of ``bidder``. It is bound to ``rng``, and each
            bidder in its tree to its own stream split from it.

        Raises
        ------
        TypeError
            If the bidder holds an object that cannot be copied.
        """
        try:
            bidder = copy.deepcopy(self.bidder)
        except Exception as error:
            raise TypeError(
                f"campaign {self.name!r} cannot be copied for a fresh run: {error}. "
                "Create objects such as open files or locks in on_episode_start"
            ) from error

        bind_tree(bidder, rng)
        return bidder

    def merged_reward_params(self) -> dict[str, float]:
        """Return the ``params`` the reward functions get.

        Returns
        -------
        dict[str, float]
            ``reward_params``, plus the campaign's ``cpa_target`` when set. A
            ``cpa_target`` key in ``reward_params`` wins.
        """
        if self.cpa_target is None:
            return dict(self.reward_params)

        return {"cpa_target": self.cpa_target, **self.reward_params}


@dataclass(frozen=True)
class SimConfig:
    """Everything one simulation needs; the same config and seed give the same run.

    Parameters
    ----------
    campaigns : Sequence[Campaign]
        At least one bidder, stored as a tuple; names must be unique.
    episodes : int, optional
        Episodes per run. Budgets reset each episode; bidders keep learning.
    steps : int, optional
        Auctions per episode.
    seed : int, optional
        Seed for every random stream. `run_many` replaces it per seed.
    auction : FirstPrice or SecondPrice, optional
        The auction rule. Defaults to `SecondPrice` with no reserve.
    market : bidlab.markets.Market, optional
        Draws features and conversion probabilities. Defaults to
        `bidlab.markets.LogisticContext`.
    horizon_hours : float, optional
        Episode length in hours. Auctions are spaced evenly over it. Must be
        > 0.
    start_hour : float, optional
        Hour of day when each episode starts, in [0, 24).
    estimate_noise : float, optional
        How wrong the platform's estimate p̂ is: the standard deviation of
        normal noise added to logit(p). 0, the default, gives the exact p.
        `BidRequest.value` uses p̂; `RunLog.true_values` keeps the true p.

    Raises
    ------
    TypeError
        If ``auction`` or ``market`` has the wrong type.
    ValueError
        If ``campaigns`` is empty, names repeat, ``episodes`` or ``steps`` is
        below 1, ``estimate_noise`` is negative, the schedule is out of
        range, or a campaign names a segment the market does not have.

    Examples
    --------
    >>> import bidlab as bl
    >>> config = bl.SimConfig(
    ...     campaigns=[
    ...         bl.Campaign("truthful", bl.Linear(), budget=50, event_value=5),
    ...         bl.Campaign("pid", bl.PidPacing(), budget=15, event_value=5),
    ...     ],
    ...     auction=bl.FirstPrice(reserve=0.01),
    ...     episodes=2,
    ...     steps=100,
    ... )
    >>> bl.run(config).bids.shape
    (2, 100, 2)
    """

    campaigns: Sequence[Campaign]
    episodes: int = 10
    steps: int = 1000
    seed: int = 0
    auction: Auction = field(default_factory=SecondPrice)
    market: Market = field(default_factory=LogisticContext)
    horizon_hours: float = 24.0
    start_hour: float = 0.0
    estimate_noise: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaigns", tuple(self.campaigns))
        if not self.campaigns:
            raise ValueError("campaigns must list at least one Campaign")

        names = [campaign.name for campaign in self.campaigns]
        if len(set(names)) != len(names):
            raise ValueError(f"campaign names must be unique, got {names}")

        for field_name in ("episodes", "steps"):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be >= 1")

        if self.horizon_hours <= 0:
            raise ValueError(f"horizon_hours must be > 0, got {self.horizon_hours}")

        if not 0 <= self.start_hour < HOURS_PER_DAY:
            raise ValueError(f"start_hour must be in [0, 24), got {self.start_hour}")

        if self.estimate_noise < 0:
            raise ValueError(f"estimate_noise must be >= 0, got {self.estimate_noise}")

        object.__setattr__(self, "estimate_noise", float(self.estimate_noise))
        object.__setattr__(self, "horizon_hours", float(self.horizon_hours))
        object.__setattr__(self, "start_hour", float(self.start_hour))

        # Callers can pass any object at run time; the type hint does not check it.
        auction: object = self.auction
        if not isinstance(auction, FirstPrice | SecondPrice):
            raise TypeError("auction must be FirstPrice or SecondPrice")

        market: object = self.market
        if not isinstance(market, Market):
            raise TypeError(
                f"market must be a bidlab.markets.Market, got {type(market).__name__}"
            )

        known = set(self.market.segment_names)
        for campaign in self.campaigns:
            unknown = sorted(set(campaign.segments or ()) - known)
            if unknown:
                raise ValueError(
                    f"campaign {campaign.name!r}: segments {unknown} are not in the "
                    f"market's segment_names {self.market.segment_names}"
                )
