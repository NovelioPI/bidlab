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
from bidlab.types import HOURS_PER_DAY, Campaign


@dataclass(frozen=True)
class BidderSpec:
    """One advertiser in a simulation: its bidder, campaign and rewards.

    Parameters
    ----------
    name : str
        Label used in results; unique within a `SimConfig`.
    bidder : Bidder
        A bidder instance, such as ``bl.PidPacing()`` or a `Stack`. The
        simulator deep-copies it once per run, so the object you pass never
        changes and every run starts from it.
    campaign : Campaign
        The advertiser's budget, value and targets.
    reward : StepReward, optional
        Called every step, ``f(observation, feedback, params) -> float``.
        Defaults to `bidlab.rewards.surplus`.
    episode_reward : EpisodeReward, optional
        Called at the end of each episode, ``g(stats, params) -> float``.
    reward_params : dict[str, float], optional
        Passed to both reward functions.

    Raises
    ------
    TypeError
        If ``bidder`` is not a `Bidder` instance, ``campaign`` is not a
        `Campaign`, or a reward is not callable.

    Examples
    --------
    >>> import bidlab as bl
    >>> spec = bl.BidderSpec(
    ...     "pid", bl.PidPacing(), bl.Campaign(budget=15, value=5, cpa_target=3),
    ...     episode_reward=bl.rewards.cpa_score,
    ... )
    >>> spec.campaign.budget
    15.0
    """

    name: str
    bidder: Bidder
    campaign: Campaign
    reward: StepReward = rewards.surplus
    episode_reward: EpisodeReward | None = None
    reward_params: dict[str, float] = field(default_factory=dict[str, float])

    def __post_init__(self) -> None:
        # Callers can pass any object at run time; the type hint does not check it.
        bidder: object = self.bidder
        if isinstance(bidder, type):
            raise TypeError(
                f"bidder {self.name!r}: got the class {bidder.__name__}; "
                f"pass an instance, such as {bidder.__name__}()"
            )

        if not isinstance(bidder, Bidder):
            raise TypeError(
                f"bidder {self.name!r}: bidder must be a Bidder instance, "
                f"got {type(bidder).__name__}"
            )

        campaign: object = self.campaign
        if not isinstance(campaign, Campaign):
            raise TypeError(
                f"bidder {self.name!r}: campaign must be a Campaign, "
                f"got {type(campaign).__name__}"
            )

        if not callable(self.reward):
            raise TypeError(f"bidder {self.name!r}: reward must be callable")

        if self.episode_reward is not None and not callable(self.episode_reward):
            raise TypeError(
                f"bidder {self.name!r}: episode_reward must be callable or None"
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
                f"bidder {self.name!r} cannot be copied for a fresh run: {error}. "
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
        if self.campaign.cpa_target is None:
            return dict(self.reward_params)

        return {"cpa_target": self.campaign.cpa_target, **self.reward_params}


@dataclass(frozen=True)
class SimConfig:
    """Everything one simulation needs; the same config and seed give the same run.

    Parameters
    ----------
    bidders : Sequence[BidderSpec]
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
        `Observation.value` uses p̂; `RunLog.values` keeps the true p.

    Raises
    ------
    TypeError
        If ``auction`` or ``market`` has the wrong type.
    ValueError
        If ``bidders`` is empty, names repeat, ``episodes`` or ``steps`` is
        below 1, ``estimate_noise`` is negative, the schedule is out of range
        or differs from a campaign's,
        or a campaign names a segment the market does not have.

    Examples
    --------
    >>> import bidlab as bl
    >>> config = bl.SimConfig(
    ...     bidders=[
    ...         bl.BidderSpec("truthful", bl.Truthful(), bl.Campaign(budget=50, value=5)),
    ...         bl.BidderSpec("pid", bl.PidPacing(), bl.Campaign(budget=15, value=5)),
    ...     ],
    ...     auction=bl.FirstPrice(reserve=0.01),
    ...     episodes=2,
    ...     steps=100,
    ... )
    >>> bl.run(config).bids.shape
    (2, 100, 2)
    """

    bidders: Sequence[BidderSpec]
    episodes: int = 10
    steps: int = 1000
    seed: int = 0
    auction: Auction = field(default_factory=SecondPrice)
    market: Market = field(default_factory=LogisticContext)
    horizon_hours: float = 24.0
    start_hour: float = 0.0
    estimate_noise: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bidders", tuple(self.bidders))
        if not self.bidders:
            raise ValueError("bidders must list at least one BidderSpec")

        names = [spec.name for spec in self.bidders]
        if len(set(names)) != len(names):
            raise ValueError(f"bidder names must be unique, got {names}")

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
        self._check_schedules()

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
        for spec in self.bidders:
            unknown = sorted(set(spec.campaign.segments or ()) - known)
            if unknown:
                raise ValueError(
                    f"bidder {spec.name!r}: segments {unknown} are not in the "
                    f"market's segment_names {self.market.segment_names}"
                )

    def _check_schedules(self) -> None:
        for spec in self.bidders:
            for field_name in ("horizon_hours", "start_hour"):
                own = getattr(spec.campaign, field_name)
                if own is not None and own != getattr(self, field_name):
                    raise ValueError(
                        f"bidder {spec.name!r}: campaign {field_name}={own} differs "
                        f"from SimConfig {field_name}={getattr(self, field_name)}; "
                        "set the schedule on SimConfig"
                    )
