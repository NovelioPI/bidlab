from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bidlab.auctions import FirstPrice
from bidlab.rewards import cpa_score_of
from bidlab.simulator import RunLog

REGRET_MULTIPLIERS = np.round(np.arange(0.0, 3.0001, 0.05), 2)


@dataclass(frozen=True)
class BidderEpisodeMetrics:
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
class PlatformEpisodeMetrics:
    episode: int
    revenue: float
    welfare: float


def platform_metrics(log: RunLog) -> list[PlatformEpisodeMetrics]:
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
    """One row per episode and bidder; definitions are in docs/metrics.md."""
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
    """Best fixed-multiplier expected surplus minus the bidder's own expected surplus."""
    spec = log.config.bidders[bidder]
    auction = log.config.auction
    values = log.values[episode, :, bidder]
    others = np.delete(log.bids[episode], bidder, axis=1)
    highest_other = others.max(axis=1) if others.shape[1] else np.zeros(len(values))

    budget_left = np.full(multipliers.size, spec.budget)
    replay_surplus = np.zeros(multipliers.size)
    for step, value in enumerate(values):
        bids = np.minimum(multipliers * value, budget_left)
        wins = (bids >= auction.reserve) & (bids > highest_other[step])
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
