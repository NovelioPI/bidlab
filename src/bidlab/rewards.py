"""Built-in reward functions, and the signatures your own must follow.

A step reward is ``f(request, result, params) -> float``. An episode
reward is ``g(report, params) -> float``. ``params`` is the campaign's
``reward_params``, plus ``cpa_target`` when the campaign sets one.

Examples
--------
>>> import bidlab as bl
>>> def wins(request, result, params):
...     return float(result.won)
>>> campaign = bl.Campaign("t", bl.Linear(), budget=10, event_value=5, step_reward=wins)
"""

from typing import Protocol

from bidlab.types import BidRequest, BidResult, EpisodeReport

CPA_PENALTY_EXPONENT = 3


class StepReward(Protocol):
    """A reward computed after every auction the bidder took part in."""

    def __call__(
        self, request: BidRequest, result: BidResult, params: dict[str, float]
    ) -> float:
        """Return the reward for one step."""
        ...


class EpisodeReward(Protocol):
    """A reward computed once at the end of each episode."""

    def __call__(self, report: EpisodeReport, params: dict[str, float]) -> float:
        """Return the reward for one episode."""
        ...


def surplus(request: BidRequest, result: BidResult, params: dict[str, float]) -> float:
    """Realized value minus price, from `BidResult`: the default step reward.

    Examples
    --------
    >>> import bidlab as bl
    >>> request = bl.BidRequest(time=0.0, value=0.5)
    >>> bl.rewards.surplus(request, bl.BidResult(0.0, 0.6, True, 0.5, True, realized_value=5.0), {})
    4.5
    >>> bl.rewards.surplus(request, bl.BidResult(0.0, 0.6, True, 0.5, False), {})
    -0.5
    """
    return result.realized_value - result.price


def conversions(
    request: BidRequest, result: BidResult, params: dict[str, float]
) -> float:
    """1 when the bidder won and converted, else 0.

    Examples
    --------
    >>> import bidlab as bl
    >>> request = bl.BidRequest(time=0.0, value=0.5)
    >>> bl.rewards.conversions(request, bl.BidResult(0.0, 0.6, True, 0.5, True, realized_value=5.0), {})
    1.0
    """
    return float(result.converted)


def realized_value(
    request: BidRequest, result: BidResult, params: dict[str, float]
) -> float:
    """Realized value, ignoring the price.

    Examples
    --------
    >>> import bidlab as bl
    >>> request = bl.BidRequest(time=0.0, value=0.5)
    >>> bl.rewards.realized_value(request, bl.BidResult(0.0, 0.6, True, 0.5, True, realized_value=5.0), {})
    5.0
    """
    return result.realized_value


def cpa_score_of(
    realized_value: float, spend: float, conversions: int, cpa_target: float
) -> float:
    """AuctionNet score: realized_value × min((target / cpa)^3, 1).

    Parameters
    ----------
    realized_value : float
        Realized value of the conversions.
    spend : float
        Total price paid.
    conversions : int
        Number of conversions.
    cpa_target : float
        Highest acceptable cost per conversion.

    Returns
    -------
    float
        ``realized_value`` when nothing was spent; 0 when money was spent with no
        conversion.

    Examples
    --------
    >>> from bidlab.rewards import cpa_score_of
    >>> cpa_score_of(realized_value=10.0, spend=4.0, conversions=2, cpa_target=2.0)
    10.0
    >>> cpa_score_of(realized_value=10.0, spend=8.0, conversions=2, cpa_target=2.0)
    1.25
    """
    if spend == 0:
        return realized_value

    if conversions == 0:
        return 0.0

    cpa = spend / conversions
    return realized_value * min((cpa_target / cpa) ** CPA_PENALTY_EXPONENT, 1.0)


def cpa_score(report: EpisodeReport, params: dict[str, float]) -> float:
    """Episode reward: `cpa_score_of` with the campaign's ``cpa_target``.

    Raises
    ------
    ValueError
        If the campaign sets no ``cpa_target``.

    Examples
    --------
    >>> import bidlab as bl
    >>> report = bl.EpisodeReport(budget=20.0, spend=8.0, wins=4, conversions=2, realized_value=10.0)
    >>> bl.rewards.cpa_score(report, {"cpa_target": 2.0})
    1.25
    """
    if "cpa_target" not in params:
        raise ValueError("episode reward 'cpa_score' needs cpa_target on the bidder")

    return cpa_score_of(
        report.realized_value, report.spend, report.conversions, params["cpa_target"]
    )


def total_surplus(report: EpisodeReport, params: dict[str, float]) -> float:
    """Episode reward: realized value minus spend.

    Examples
    --------
    >>> import bidlab as bl
    >>> report = bl.EpisodeReport(budget=20.0, spend=6.0, wins=4, conversions=2, realized_value=10.0)
    >>> bl.rewards.total_surplus(report, {})
    4.0
    """
    return report.realized_value - report.spend
