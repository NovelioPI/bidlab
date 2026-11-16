"""Frozen, validated settings for one simulation."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from bidlab import rewards
from bidlab.auctions import Auction, FirstPrice, SecondPrice
from bidlab.bidders.base import Bidder
from bidlab.markets import LogisticContext, Market
from bidlab.rewards import EpisodeReward, StepReward

BidderFactory = Callable[[dict[str, float], np.random.Generator], Bidder]


@dataclass(frozen=True)
class BidderSpec:
    """One advertiser in a simulation: its bidding method, budget and goals.

    Parameters
    ----------
    name : str
        Label used in results; unique within a `SimConfig`.
    bidder : type[Bidder] or callable
        A `Bidder` subclass, or a factory ``f(params, rng) -> Bidder``. The
        simulator calls it once per run, so the object lives for all episodes.
    budget : float
        Budget per episode. Must be > 0.
    value_per_conversion : float
        What one conversion is worth to this advertiser. Must be > 0.
    cpa_target : float, optional
        Highest acceptable cost per conversion. Reward functions get it as
        ``params["cpa_target"]``. Must be > 0 when set.
    params : dict[str, float], optional
        Passed to the bidder, for tuning values.
    reward : StepReward, optional
        Called every step, ``f(observation, feedback, params) -> float``.
        Defaults to `bidlab.rewards.surplus`.
    episode_reward : EpisodeReward, optional
        Called at the end of each episode, ``g(stats, params) -> float``.
    reward_params : dict[str, float], optional
        Passed to both reward functions.
    segments : tuple[str, ...], optional
        Market segments this bidder bids in. None means every segment.

    Raises
    ------
    TypeError
        If ``bidder`` or ``segments`` is a string, or a field that must be
        callable is not.
    ValueError
        If a number is <= 0 or ``segments`` is empty.

    Examples
    --------
    >>> import bidlab as bl
    >>> spec = bl.BidderSpec(
    ...     "pid", bl.PidPacing, budget=15, value_per_conversion=5,
    ...     cpa_target=3, episode_reward=bl.rewards.cpa_score,
    ... )
    >>> spec.budget
    15.0
    """

    name: str
    bidder: BidderFactory
    budget: float
    value_per_conversion: float
    cpa_target: float | None = None
    params: dict[str, float] = field(default_factory=dict[str, float])
    reward: StepReward = rewards.surplus
    episode_reward: EpisodeReward | None = None
    reward_params: dict[str, float] = field(default_factory=dict[str, float])
    segments: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self._check_segments()
        if isinstance(self.bidder, str):
            raise TypeError(
                f"bidder {self.name!r}: got the string {self.bidder!r}; "
                "pass a Bidder subclass or a factory f(params, rng) -> Bidder"
            )

        for field_name in ("bidder", "reward"):
            if not callable(getattr(self, field_name)):
                raise TypeError(f"bidder {self.name!r}: {field_name} must be callable")

        if self.episode_reward is not None and not callable(self.episode_reward):
            raise TypeError(
                f"bidder {self.name!r}: episode_reward must be callable or None"
            )

        # An int budget would make integer numpy arrays that reject float prices.
        for field_name in ("budget", "value_per_conversion", "cpa_target"):
            number = getattr(self, field_name)
            if number is None:
                continue

            if number <= 0:
                raise ValueError(
                    f"bidder {self.name!r}: {field_name} must be > 0, got {number}"
                )

            object.__setattr__(self, field_name, float(number))

    def _check_segments(self) -> None:
        if self.segments is None:
            return

        if isinstance(self.segments, str):
            raise TypeError(
                f"bidder {self.name!r}: segments must be a tuple of names, "
                f"got the string {self.segments!r}"
            )

        object.__setattr__(self, "segments", tuple(self.segments))
        if not self.segments:
            raise ValueError(
                f"bidder {self.name!r}: segments must list at least one segment; "
                "use None to bid everywhere"
            )

    def merged_reward_params(self) -> dict[str, float]:
        """Return the ``params`` the reward functions get.

        Returns
        -------
        dict[str, float]
            ``reward_params``, plus ``cpa_target`` when set. A
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
    bidders : tuple[BidderSpec, ...]
        At least one bidder; names must be unique.
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

    Raises
    ------
    TypeError
        If ``auction`` or ``market`` has the wrong type.
    ValueError
        If ``bidders`` is empty, names repeat, ``episodes`` or ``steps`` is
        below 1, or a bidder names a segment the market does not have.

    Examples
    --------
    >>> import bidlab as bl
    >>> config = bl.SimConfig(
    ...     bidders=[
    ...         bl.BidderSpec("truthful", bl.Truthful, budget=50, value_per_conversion=5),
    ...         bl.BidderSpec("pid", bl.PidPacing, budget=15, value_per_conversion=5),
    ...     ],
    ...     auction=bl.FirstPrice(reserve=0.01),
    ...     episodes=2,
    ...     steps=100,
    ... )
    >>> bl.run(config).bids.shape
    (2, 100, 2)
    """

    bidders: tuple[BidderSpec, ...]
    episodes: int = 10
    steps: int = 1000
    seed: int = 0
    auction: Auction = field(default_factory=SecondPrice)
    market: Market = field(default_factory=LogisticContext)

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
            unknown = sorted(set(spec.segments or ()) - known)
            if unknown:
                raise ValueError(
                    f"bidder {spec.name!r}: segments {unknown} are not in the "
                    f"market's segment_names {self.market.segment_names}"
                )
