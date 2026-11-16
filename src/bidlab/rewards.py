"""Built-in reward functions, and the signatures your own must follow.

A step reward is ``f(observation, feedback, params) -> float``. An episode
reward is ``g(stats, params) -> float``. ``params`` is the spec's
``reward_params``, plus ``cpa_target`` when the spec sets one.

Examples
--------
>>> import bidlab as bl
>>> def wins(observation, feedback, params):
...     return float(feedback.won)
>>> spec = bl.BidderSpec("t", bl.Truthful, budget=10, value_per_conversion=5, reward=wins)
"""

from typing import Protocol

from bidlab.types import EpisodeStats, Feedback, Observation

CPA_PENALTY_EXPONENT = 3


class StepReward(Protocol):
    """A reward computed after every auction the bidder took part in."""

    def __call__(
        self, observation: Observation, feedback: Feedback, params: dict[str, float]
    ) -> float:
        """Return the reward for one step."""
        ...


class EpisodeReward(Protocol):
    """A reward computed once at the end of each episode."""

    def __call__(self, stats: EpisodeStats, params: dict[str, float]) -> float:
        """Return the reward for one episode."""
        ...


def surplus(
    observation: Observation, feedback: Feedback, params: dict[str, float]
) -> float:
    """Realized value minus price: the default step reward.

    Examples
    --------
    >>> import bidlab as bl
    >>> observation = bl.Observation(0, 100, 0.5, 5.0, 20.0)
    >>> bl.rewards.surplus(observation, bl.Feedback(0.6, True, 0.5, True), {})
    4.5
    >>> bl.rewards.surplus(observation, bl.Feedback(0.6, True, 0.5, False), {})
    -0.5
    """
    return feedback.converted * observation.value_per_conversion - feedback.price


def conversions(
    observation: Observation, feedback: Feedback, params: dict[str, float]
) -> float:
    """1 when the bidder won and converted, else 0.

    Examples
    --------
    >>> import bidlab as bl
    >>> observation = bl.Observation(0, 100, 0.5, 5.0, 20.0)
    >>> bl.rewards.conversions(observation, bl.Feedback(0.6, True, 0.5, True), {})
    1.0
    """
    return float(feedback.converted)


def value(
    observation: Observation, feedback: Feedback, params: dict[str, float]
) -> float:
    """Realized value, ignoring the price.

    Examples
    --------
    >>> import bidlab as bl
    >>> observation = bl.Observation(0, 100, 0.5, 5.0, 20.0)
    >>> bl.rewards.value(observation, bl.Feedback(0.6, True, 0.5, True), {})
    5.0
    """
    return feedback.converted * observation.value_per_conversion


def cpa_score_of(
    value: float, spend: float, conversions: int, cpa_target: float
) -> float:
    """AuctionNet score: value × min((target / cpa)^3, 1).

    Parameters
    ----------
    value : float
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
        ``value`` when nothing was spent; 0 when money was spent with no
        conversion.

    Examples
    --------
    >>> from bidlab.rewards import cpa_score_of
    >>> cpa_score_of(value=10.0, spend=4.0, conversions=2, cpa_target=2.0)
    10.0
    >>> cpa_score_of(value=10.0, spend=8.0, conversions=2, cpa_target=2.0)
    1.25
    """
    if spend == 0:
        return value

    if conversions == 0:
        return 0.0

    cpa = spend / conversions
    return value * min((cpa_target / cpa) ** CPA_PENALTY_EXPONENT, 1.0)


def cpa_score(stats: EpisodeStats, params: dict[str, float]) -> float:
    """Episode reward: `cpa_score_of` with the spec's ``cpa_target``.

    Raises
    ------
    ValueError
        If the spec sets no ``cpa_target``.

    Examples
    --------
    >>> import bidlab as bl
    >>> stats = bl.EpisodeStats(budget=20.0, spend=8.0, wins=4, conversions=2, value=10.0)
    >>> bl.rewards.cpa_score(stats, {"cpa_target": 2.0})
    1.25
    """
    if "cpa_target" not in params:
        raise ValueError("episode reward 'cpa_score' needs cpa_target on the bidder")

    return cpa_score_of(
        stats.value, stats.spend, stats.conversions, params["cpa_target"]
    )


def total_surplus(stats: EpisodeStats, params: dict[str, float]) -> float:
    """Episode reward: realized value minus spend.

    Examples
    --------
    >>> import bidlab as bl
    >>> stats = bl.EpisodeStats(budget=20.0, spend=6.0, wins=4, conversions=2, value=10.0)
    >>> bl.rewards.total_surplus(stats, {})
    4.0
    """
    return stats.value - stats.spend
