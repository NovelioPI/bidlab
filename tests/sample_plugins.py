from auction_sim.bidders import Bidder
from auction_sim.types import EpisodeStats, Feedback, Observation


class Doubler(Bidder):
    def bid(self, obs: Observation) -> float:
        return 2 * obs.value


class NotABidder:
    pass


def double_price(
    obs: Observation, feedback: Feedback, params: dict[str, float]
) -> float:
    return 2 * feedback.price


def total_spend(stats: EpisodeStats, params: dict[str, float]) -> float:
    return stats.spend
