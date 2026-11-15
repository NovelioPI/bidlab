from auction_sim.bidders.base import Bidder
from auction_sim.types import Observation


class Truthful(Bidder):
    """Bids `multiplier` × value; multiplier 1 is truthful."""

    def bid(self, obs: Observation) -> float:
        return self.params.get("multiplier", 1.0) * obs.value
