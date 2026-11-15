from bidlab.bidders.base import Bidder
from bidlab.types import Observation


class Truthful(Bidder):
    """Bids `multiplier` × value; multiplier 1 is truthful."""

    def bid(self, observation: Observation) -> float:
        return self.params.get("multiplier", 1.0) * observation.value
