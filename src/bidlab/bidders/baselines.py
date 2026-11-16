"""Fixed bidders that do not learn, for use as baselines."""

from typing import override

from bidlab.bidders.base import Bidder
from bidlab.types import Observation


class Truthful(Bidder):
    """Bids `multiplier` × value; multiplier 1 is truthful.

    Truthful bidding is optimal in a second-price auction without a budget
    limit. Set ``multiplier`` below 1 for a fixed bid shade.

    Parameters
    ----------
    params : dict[str, float]
        ``multiplier`` (default 1.0): the bid as a fraction of the value.
    rng : numpy.random.Generator
        Not used.

    Examples
    --------
    >>> import bidlab as bl
    >>> shaded = bl.BidderSpec(
    ...     "shaded", bl.Truthful, budget=10, value_per_conversion=5,
    ...     params={"multiplier": 0.8},
    ... )
    >>> log = bl.run(bl.SimConfig(bidders=[shaded], episodes=1, steps=20))
    >>> bool((log.bids == 0.8 * log.values).all())
    True
    """

    @override
    def bid(self, observation: Observation) -> float:
        return self.params.get("multiplier", 1.0) * observation.value
