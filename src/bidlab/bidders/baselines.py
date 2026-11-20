"""Fixed bidders that do not learn, for use as baselines."""

from typing import override

from bidlab.bidders.base import Bidder


class Linear(Bidder):
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
    >>> shaded = bl.Campaign(
    ...     "shaded", bl.Linear(multiplier=0.8), budget=10, event_value=5
    ... )
    >>> log = bl.run(bl.SimConfig(campaigns=[shaded], episodes=1, steps=20))
    >>> bool((log.bids == 0.8 * log.true_values).all())
    True
    """

    def __init__(self, multiplier: float = 1.0):
        if multiplier < 0:
            raise ValueError(f"multiplier must be >= 0, got {multiplier}")

        self.multiplier = multiplier

    @override
    def forward(self, value: float) -> float:
        return self.multiplier * value
