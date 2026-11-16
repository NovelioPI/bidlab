"""Per-episode metrics computed from a `RunLog`.

The definitions match ``docs/metrics.md`` in the bidlab repository.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bidlab.auctions import FirstPrice
from bidlab.rewards import cpa_score_of
from bidlab.simulator import RunLog

REGRET_MULTIPLIERS = np.round(np.arange(0.0, 3.0001, 0.05), 2)


@dataclass(frozen=True)
class BidderEpisodeMetrics:
    """One bidder's metrics for one episode.

    Attributes
    ----------
    episode : int
        Episode index.
    bidder : str
        The spec's name.
    spend : float
        Sum of prices paid.
    wins : int
        Impressions won.
    conversions : int
        Realized conversions on won impressions.
    value : float
        ``conversions`` × value per conversion.
    surplus : float
        ``value`` − ``spend``.
    budget_used : float
        ``spend`` / budget; never above 1.
    cpa : float
        ``spend`` / ``conversions``; NaN with no conversion.
    cpa_violation : float
        max(0, ``cpa`` / target − 1); 0 with no spend, +inf with spend and
        no conversion, NaN with no target.
    cpa_score : float
        See `bidlab.rewards.cpa_score_of`; NaN with no target.
    regret : float
        See `regret`.
    reward : float
        Sum of the bidder's step rewards.
    episode_reward : float
        The spec's episode reward; NaN when none is set.
    """

    episode: int
    bidder: str
    spend: float
    wins: int
    conversions: int
    value: float
    surplus: float
    budget_used: float
    cpa: float
    cpa_violation: float
    cpa_score: float
    regret: float
    reward: float
    episode_reward: float


@dataclass(frozen=True)
class BidderSegmentMetrics:
    """One bidder's metrics for one episode, over one market segment's steps.

    Attributes
    ----------
    episode : int
        Episode index.
    bidder : str
        The spec's name.
    segment : str
        Name from the market's ``segment_names``.
    spend : float
        Sum of prices paid in the segment.
    wins : int
        Impressions won in the segment.
    conversions : int
        Realized conversions in the segment.
    surplus : float
        ``conversions`` × value per conversion − ``spend``.
    """

    episode: int
    bidder: str
    segment: str
    spend: float
    wins: int
    conversions: int
    surplus: float


@dataclass(frozen=True)
class PlatformEpisodeMetrics:
    """The seller's metrics for one episode.

    Attributes
    ----------
    episode : int
        Episode index.
    revenue : float
        Sum of all prices paid.
    welfare : float
        Sum of the winner's expected value, over steps with a sale.
    """

    episode: int
    revenue: float
    welfare: float


def platform_metrics(log: RunLog) -> list[PlatformEpisodeMetrics]:
    """One row per episode with the seller's revenue and welfare.

    Examples
    --------
    >>> import bidlab as bl
    >>> spec = bl.BidderSpec("t", bl.Truthful, budget=10, value_per_conversion=5)
    >>> log = bl.run(bl.SimConfig(bidders=[spec], episodes=2, steps=50))
    >>> [row.revenue for row in bl.platform_metrics(log)]
    [0.0, 0.0]
    """
    rows = []
    for episode in range(log.winner.shape[0]):
        winner = log.winner[episode]
        sold = winner >= 0
        winner_values = log.values[episode, sold, winner[sold]]
        rows.append(
            PlatformEpisodeMetrics(
                episode=episode,
                revenue=float(log.price[episode].sum()),
                welfare=float(winner_values.sum()),
            )
        )

    return rows


def bidder_metrics(log: RunLog) -> list[BidderEpisodeMetrics]:
    """One row per episode and bidder; definitions are in docs/metrics.md.

    Examples
    --------
    >>> import bidlab as bl
    >>> config = bl.SimConfig(
    ...     bidders=[
    ...         bl.BidderSpec("a", bl.Truthful, budget=10, value_per_conversion=5),
    ...         bl.BidderSpec("b", bl.PidPacing, budget=10, value_per_conversion=5),
    ...     ],
    ...     episodes=2,
    ...     steps=50,
    ... )
    >>> [(row.episode, row.bidder) for row in bl.bidder_metrics(bl.run(config))]
    [(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b')]
    """
    rows = []
    for episode in range(log.winner.shape[0]):
        for i, spec in enumerate(log.config.bidders):
            won = log.winner[episode] == i
            spend = float(log.price[episode][won].sum())
            conversions = int((log.converted[episode] & won).sum())
            value = conversions * spec.value_per_conversion
            target = spec.cpa_target
            rows.append(
                BidderEpisodeMetrics(
                    episode=episode,
                    bidder=spec.name,
                    spend=spend,
                    wins=int(won.sum()),
                    conversions=conversions,
                    value=value,
                    surplus=value - spend,
                    budget_used=spend / spec.budget,
                    cpa=spend / conversions if conversions else np.nan,
                    cpa_violation=_cpa_violation(spend, conversions, target),
                    cpa_score=(
                        np.nan
                        if target is None
                        else cpa_score_of(value, spend, conversions, target)
                    ),
                    regret=regret(log, episode, i),
                    reward=float(log.rewards[episode, :, i].sum()),
                    episode_reward=float(log.episode_rewards[episode, i]),
                )
            )

    return rows


def segment_metrics(log: RunLog) -> list[BidderSegmentMetrics]:
    """One row per episode, bidder and market segment.

    For one episode and bidder, each metric summed over segments equals the
    value in `bidder_metrics`.

    Examples
    --------
    >>> import bidlab as bl
    >>> from bidlab.markets import Segment, Segmented
    >>> market = Segmented((Segment("day", 0.6), Segment("night", 0.4)))
    >>> spec = bl.BidderSpec("t", bl.Truthful, budget=10, value_per_conversion=5)
    >>> log = bl.run(bl.SimConfig(bidders=[spec], market=market, episodes=1, steps=50))
    >>> [row.segment for row in bl.segment_metrics(log)]
    ['day', 'night']
    """
    rows = []
    names = log.config.market.segment_names
    for episode in range(log.winner.shape[0]):
        for i, spec in enumerate(log.config.bidders):
            won = log.winner[episode] == i
            for index, name in enumerate(names):
                won_here = won & (log.segment[episode] == index)
                spend = float(log.price[episode][won_here].sum())
                conversions = int((log.converted[episode] & won_here).sum())
                rows.append(
                    BidderSegmentMetrics(
                        episode=episode,
                        bidder=spec.name,
                        segment=name,
                        spend=spend,
                        wins=int(won_here.sum()),
                        conversions=conversions,
                        surplus=conversions * spec.value_per_conversion - spend,
                    )
                )

    return rows


def _cpa_violation(spend: float, conversions: int, target: float | None) -> float:
    if target is None:
        return np.nan

    if spend == 0:
        return 0.0

    if conversions == 0:
        return np.inf

    return max(0.0, (spend / conversions) / target - 1.0)


def regret(
    log: RunLog,
    episode: int,
    bidder: int,
    multipliers: npt.NDArray[np.float64] = REGRET_MULTIPLIERS,
) -> float:
    """Best fixed-multiplier expected surplus minus the bidder's own expected surplus.

    Other bidders' logged bids stay fixed while the episode is replayed with
    this bidder bidding m × value for each multiplier m. See the Regret
    section of ``docs/metrics.md`` for every rule and its limits.

    Parameters
    ----------
    log : RunLog
        The run to replay.
    episode : int
        Episode index.
    bidder : int
        Bidder index in ``log.config.bidders``.
    multipliers : numpy.ndarray, optional
        Multipliers to try; 0, 0.05, ..., 3 by default.

    Returns
    -------
    float
        Regret; it can be negative when an adaptive bidder beats every fixed
        multiplier.

    Examples
    --------
    A truthful bidder alone in a second-price auction pays 0, so no
    multiplier beats it:

    >>> import bidlab as bl
    >>> spec = bl.BidderSpec("t", bl.Truthful, budget=10, value_per_conversion=5)
    >>> log = bl.run(bl.SimConfig(bidders=[spec], episodes=1, steps=50))
    >>> round(bl.regret(log, episode=0, bidder=0), 9)
    0.0
    """
    spec = log.config.bidders[bidder]
    auction = log.config.auction
    values = log.values[episode, :, bidder]
    others = np.delete(log.bids[episode], bidder, axis=1)
    highest_other = others.max(axis=1) if others.shape[1] else np.zeros(len(values))
    # A targeted bidder cannot win outside its segments, so neither can its replay.
    names = log.config.market.segment_names
    allowed = [i for i, name in enumerate(names) if name in (spec.segments or names)]
    takes_part = np.isin(log.segment[episode], allowed)

    budget_left = np.full(multipliers.size, spec.budget)
    replay_surplus = np.zeros(multipliers.size)
    for step, value in enumerate(values):
        bids = np.minimum(multipliers * value, budget_left)
        wins = (
            takes_part[step] & (bids >= auction.reserve) & (bids > highest_other[step])
        )
        if isinstance(auction, FirstPrice):
            prices = bids
        else:
            prices = np.full(
                multipliers.size, max(highest_other[step], auction.reserve)
            )

        replay_surplus += np.where(wins, value - prices, 0.0)
        budget_left -= np.where(wins, prices, 0.0)

    won = log.winner[episode] == bidder
    own_surplus = float((values[won] - log.price[episode][won]).sum())
    return float(replay_surplus.max() - own_surplus)
