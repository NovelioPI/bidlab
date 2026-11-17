"""`Stack`: a bidder that chains bidders, from a value to a bid."""

from typing import override

from bidlab.bidders.base import Bidder
from bidlab.types import Observation


class Stack(Bidder):
    """A bidder that runs its bidders in order and outputs the last one's output.

    The first bidder gets the incoming value, `Observation.value` at the top;
    each later bidder gets the previous one's output. The bidders are the
    stack's children, at paths ``"bidders.0"``, ``"bidders.1"`` and so on,
    so every hook reaches each of them, in order. Order matters: pacing then
    shading differs from shading then pacing. Stacks can nest.

    Parameters
    ----------
    *bidders : Bidder
        At least one bidder instance, in the order they run.

    Raises
    ------
    ValueError
        If no bidder is given.
    TypeError
        If an argument is not a `Bidder` instance.

    Examples
    --------
    Pace the platform's value with a PID loop, after a fixed shade:

    >>> import bidlab as bl
    >>> bidder = bl.Stack(bl.Truthful(multiplier=0.9), bl.PidPacing())
    >>> spec = bl.BidderSpec("paced", bidder, bl.Campaign(budget=10, value=5))
    >>> bl.run(bl.SimConfig(bidders=[spec], episodes=1, steps=50)).bids.shape
    (1, 50, 1)
    """

    def __init__(self, *bidders: Bidder):
        if not bidders:
            raise ValueError("Stack needs at least one bidder")

        for bidder in bidders:
            if isinstance(bidder, type):
                raise TypeError(
                    f"Stack takes Bidder instances, got the class {bidder.__name__}; "
                    f"pass {bidder.__name__}()"
                )

            if not isinstance(bidder, Bidder):
                raise TypeError(
                    f"Stack takes Bidder instances, got {type(bidder).__name__}"
                )

        self.bidders = bidders

    @override
    def forward(self, value: float, observation: Observation) -> float:
        for bidder in self.bidders:
            value = bidder.forward(value, observation)

        return value
