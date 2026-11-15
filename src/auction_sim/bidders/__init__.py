from auction_sim.bidders.bandit import UcbMultiplier
from auction_sim.bidders.base import Bidder
from auction_sim.bidders.pid import PidPacing
from auction_sim.bidders.truthful import Truthful
from auction_sim.plugins import resolve

BIDDERS: dict[str, type[Bidder]] = {
    "truthful": Truthful,
    "pid_pacing": PidPacing,
    "ucb_multiplier": UcbMultiplier,
}


def resolve_bidder(name: str) -> type[Bidder]:
    """Return the bidder class for a built-in name or `module:Class`."""
    bidder_class = resolve(name, BIDDERS, "bidder")
    if not (isinstance(bidder_class, type) and issubclass(bidder_class, Bidder)):
        raise TypeError(
            f"bidder {name!r} is not a subclass of auction_sim.bidders.Bidder"
        )
    return bidder_class


__all__ = [
    "BIDDERS",
    "Bidder",
    "PidPacing",
    "Truthful",
    "UcbMultiplier",
    "resolve_bidder",
]
