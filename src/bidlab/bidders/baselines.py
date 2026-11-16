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
    multiplier : float, optional
        The output as a fraction of the incoming value. Must be >= 0.

    Raises
    ------
    ValueError
        If ``multiplier`` is negative.

    Examples
    --------
    >>> import bidlab as bl
    >>> shaded = bl.BidderSpec(
    ...     "shaded", bl.Truthful(multiplier=0.8), bl.Campaign(budget=10, value=5)
    ... )
    >>> log = bl.run(bl.SimConfig(bidders=[shaded], episodes=1, steps=20))
    >>> bool((log.bids == 0.8 * log.values).all())
    True
    """

    def __init__(self, multiplier: float = 1.0):
        if multiplier < 0:
            raise ValueError(f"multiplier must be >= 0, got {multiplier}")

        self.multiplier = multiplier

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return self.multiplier * value
