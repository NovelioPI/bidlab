from typing import Protocol

from bidlab.types import EpisodeStats, Feedback, Observation

CPA_PENALTY_EXPONENT = 3


class StepReward(Protocol):
    def __call__(
        self, observation: Observation, feedback: Feedback, params: dict[str, float]
    ) -> float: ...


class EpisodeReward(Protocol):
    def __call__(self, stats: EpisodeStats, params: dict[str, float]) -> float: ...


def surplus(
    observation: Observation, feedback: Feedback, params: dict[str, float]
) -> float:
    return feedback.converted * observation.value_per_conversion - feedback.price


def conversions(
    observation: Observation, feedback: Feedback, params: dict[str, float]
) -> float:
    return float(feedback.converted)


def value(
    observation: Observation, feedback: Feedback, params: dict[str, float]
) -> float:
    return feedback.converted * observation.value_per_conversion


def cpa_score_of(
    value: float, spend: float, conversions: int, cpa_target: float
) -> float:
    """AuctionNet score: value × min((target / cpa)^3, 1)."""
    if spend == 0:
        return value
    if conversions == 0:
        return 0.0
    cpa = spend / conversions
    return value * min((cpa_target / cpa) ** CPA_PENALTY_EXPONENT, 1.0)


def cpa_score(stats: EpisodeStats, params: dict[str, float]) -> float:
    if "cpa_target" not in params:
        raise ValueError("episode reward 'cpa_score' needs cpa_target on the bidder")
    return cpa_score_of(
        stats.value, stats.spend, stats.conversions, params["cpa_target"]
    )


def total_surplus(stats: EpisodeStats, params: dict[str, float]) -> float:
    return stats.value - stats.spend
