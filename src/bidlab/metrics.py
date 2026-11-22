"""Per-episode metrics computed from a `RunLog`.

The definitions match ``docs/metrics.md`` in the bidlab repository.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bidlab.auctions import FirstPrice
from bidlab.simulator import RunLog, event_happened

REGRET_MULTIPLIERS = np.round(np.arange(0.0, 3.0001, 0.05), 2)
# The AuctionNet score cuts value by (target / cpa) to this power above target.
CPA_PENALTY_EXPONENT = 3


@dataclass(frozen=True)
class CampaignEpisodeMetrics:
    """One bidder's metrics for one episode.

    Attributes
    ----------
    episode : int
        Episode index.
    campaign : str
        The campaign's name.
    spend : float
        Sum of prices paid.
    wins : int
        Impressions won.
    clicks : int
        Won impressions that were clicked; equals ``wins`` when the market
        has no click stage.
    conversions : int
        Realized conversions on won impressions.
    realized_value : float
        The count of the campaign's event × its base value: wins, ``clicks``
        or ``conversions``, by `Campaign.event`.
    surplus : float
        ``realized_value`` − ``spend``.
    budget_used : float
        ``spend`` / budget; never above 1.
    ctr : float
        ``clicks`` / ``wins``; NaN with no win.
    cpc : float
        ``spend`` / ``clicks``; NaN with no click. Billing stays per
        impression, so this is impression spend per click.
    cpa : float
        ``spend`` / ``conversions``; NaN with no conversion, for every event.
    cpa_violation : float
        max(0, ``cpa`` / target − 1); 0 with no spend, +inf with spend and
        no conversion, NaN with no target.
    cpa_score : float
        realized_value × min((target / cpa)³, 1), the AuctionNet score; NaN
        with no target.
    regret : float
        See `regret`.
    """

    episode: int
    campaign: str
    spend: float
    wins: int
    clicks: int
    conversions: int
    realized_value: float
    surplus: float
    budget_used: float
    ctr: float
    cpc: float
    cpa: float
    cpa_violation: float
    cpa_score: float
    regret: float


@dataclass(frozen=True)
class CampaignSegmentMetrics:
    """One bidder's metrics for one episode, over one market segment's steps.

    Attributes
    ----------
    episode : int
        Episode index.
    campaign : str
        The campaign's name.
    segment : str
        Name from the market's ``segment_names``.
    spend : float
        Sum of prices paid in the segment.
    wins : int
        Impressions won in the segment.
    clicks : int
        Won impressions in the segment that were clicked.
    conversions : int
        Realized conversions in the segment.
    surplus : float
        The count of the campaign's event in the segment × its base value −
        ``spend``.
    """

    episode: int
    campaign: str
    segment: str
    spend: float
    wins: int
    clicks: int
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
        Sum of all prices campaigns paid. An auction sold outside, to the
        market price, adds nothing: its price is unknown.
    welfare : float
        Sum of the winner's expected value, over auctions a campaign won.
    """

    episode: int
    revenue: float
    welfare: float


def platform_metrics(log: RunLog) -> list[PlatformEpisodeMetrics]:
    """One row per episode with the seller's revenue and welfare.

    Examples
    --------
    >>> import bidlab as bl
    >>> campaign = bl.Campaign("t", bl.Linear(), budget=10, base_value=5)
    >>> log = bl.run(bl.SimConfig(campaigns=[campaign], episodes=2, steps=50))
    >>> [row.revenue for row in bl.platform_metrics(log)]
    [0.0, 0.0]
    """
    rows = []
    for episode in range(log.winner.shape[0]):
        winner = log.winner[episode]
        sold = winner >= 0
        winner_values = log.true_values[episode, sold, winner[sold]]
        rows.append(
            PlatformEpisodeMetrics(
                episode=episode,
                revenue=float(log.price[episode].sum()),
                welfare=float(winner_values.sum()),
            )
        )

    return rows


def campaign_metrics(log: RunLog) -> list[CampaignEpisodeMetrics]:
    """One row per episode and bidder; definitions are in docs/metrics.md.

    Examples
    --------
    >>> import bidlab as bl
    >>> config = bl.SimConfig(
    ...     campaigns=[
    ...         bl.Campaign("a", bl.Linear(), budget=10, base_value=5),
    ...         bl.Campaign("b", bl.PidPacing(), budget=10, base_value=5),
    ...     ],
    ...     episodes=2,
    ...     steps=50,
    ... )
    >>> [(row.episode, row.campaign) for row in bl.campaign_metrics(bl.run(config))]
    [(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b')]
    """
    rows = []
    for episode in range(log.winner.shape[0]):
        for i, campaign in enumerate(log.config.campaigns):
            won = log.winner[episode] == i
            spend = float(log.price[episode][won].sum())
            wins = int(won.sum())
            clicks = int((log.clicked[episode] & won).sum())
            conversions = int((log.converted[episode] & won).sum())
            events = event_happened(
                campaign.event, won, log.clicked[episode], log.converted[episode]
            )
            value = int(events.sum()) * campaign.base_value
            target = campaign.cpa_target
            rows.append(
                CampaignEpisodeMetrics(
                    episode=episode,
                    campaign=campaign.name,
                    spend=spend,
                    wins=wins,
                    clicks=clicks,
                    conversions=conversions,
                    realized_value=value,
                    surplus=value - spend,
                    budget_used=spend / campaign.budget,
                    ctr=clicks / wins if wins else np.nan,
                    cpc=spend / clicks if clicks else np.nan,
                    cpa=spend / conversions if conversions else np.nan,
                    cpa_violation=_cpa_violation(spend, conversions, target),
                    cpa_score=(
                        np.nan
                        if target is None
                        else _cpa_score(value, spend, conversions, target)
                    ),
                    regret=regret(log, episode, i),
                )
            )

    return rows


def segment_metrics(log: RunLog) -> list[CampaignSegmentMetrics]:
    """One row per episode, bidder and market segment.

    For one episode and bidder, each metric summed over segments equals the
    value in `campaign_metrics`.

    Examples
    --------
    >>> import bidlab as bl
    >>> from bidlab.markets import Segment, Segmented
    >>> market = Segmented((Segment("day", 0.6), Segment("night", 0.4)))
    >>> campaign = bl.Campaign("t", bl.Linear(), budget=10, base_value=5)
    >>> log = bl.run(bl.SimConfig(campaigns=[campaign], market=market, episodes=1, steps=50))
    >>> [row.segment for row in bl.segment_metrics(log)]
    ['day', 'night']
    """
    rows = []
    names = log.config.market.segment_names
    for episode in range(log.winner.shape[0]):
        for i, campaign in enumerate(log.config.campaigns):
            won = log.winner[episode] == i
            for index, name in enumerate(names):
                won_here = won & (log.segment[episode] == index)
                spend = float(log.price[episode][won_here].sum())
                conversions = int((log.converted[episode] & won_here).sum())
                events = event_happened(
                    campaign.event,
                    won_here,
                    log.clicked[episode],
                    log.converted[episode],
                )
                rows.append(
                    CampaignSegmentMetrics(
                        episode=episode,
                        campaign=campaign.name,
                        segment=name,
                        spend=spend,
                        wins=int(won_here.sum()),
                        clicks=int((log.clicked[episode] & won_here).sum()),
                        conversions=conversions,
                        surplus=int(events.sum()) * campaign.base_value - spend,
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
    campaign_index: int,
    multipliers: npt.NDArray[np.float64] = REGRET_MULTIPLIERS,
) -> float:
    """Best fixed-multiplier expected surplus minus the bidder's own expected surplus.

    Other bidders' logged bids, the market price and the floor stay fixed
    while the episode is replayed with this bidder bidding m × value for each
    multiplier m. See the Regret section of ``docs/metrics.md`` for every rule
    and its limits.

    Parameters
    ----------
    log : RunLog
        The run to replay.
    episode : int
        Episode index.
    campaign_index : int
        Campaign index in ``log.config.campaigns``.
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
    >>> campaign = bl.Campaign("t", bl.Linear(), budget=10, base_value=5)
    >>> log = bl.run(bl.SimConfig(campaigns=[campaign], episodes=1, steps=50))
    >>> round(bl.regret(log, episode=0, campaign_index=0), 9)
    0.0
    """
    campaign = log.config.campaigns[campaign_index]
    auction = log.config.auction
    values = log.true_values[episode, :, campaign_index]
    others = np.delete(log.bids[episode], campaign_index, axis=1)
    highest_other = others.max(axis=1) if others.shape[1] else np.zeros(len(values))
    # A targeted bidder cannot win outside its segments, so neither can its replay.
    names = log.config.market.segment_names
    allowed = [
        i for i, name in enumerate(names) if name in (campaign.segments or names)
    ]
    takes_part = np.isin(log.segment[episode], allowed)
    floors = np.maximum(auction.reserve, log.floor[episode])
    # A step without a market price has no outside bid to beat.
    outside = np.nan_to_num(log.market_price[episode], nan=-np.inf)

    budget_left = np.full(multipliers.size, campaign.budget)
    replay_surplus = np.zeros(multipliers.size)
    for step, value in enumerate(values):
        bids = np.minimum(multipliers * value, budget_left)
        wins = (
            takes_part[step]
            & (bids >= floors[step])
            & (bids > highest_other[step])
            & (bids > outside[step])
        )
        if isinstance(auction, FirstPrice):
            prices = bids
        else:
            prices = np.full(
                multipliers.size, max(highest_other[step], floors[step], outside[step])
            )

        replay_surplus += np.where(wins, value - prices, 0.0)
        budget_left -= np.where(wins, prices, 0.0)

    won = log.winner[episode] == campaign_index
    own_surplus = float((values[won] - log.price[episode][won]).sum())
    return float(replay_surplus.max() - own_surplus)


def _cpa_score(
    realized_value: float, spend: float, conversions: int, cpa_target: float
) -> float:
    if spend == 0:
        return realized_value

    if conversions == 0:
        return 0.0

    cpa = spend / conversions
    return realized_value * min((cpa_target / cpa) ** CPA_PENALTY_EXPONENT, 1.0)
