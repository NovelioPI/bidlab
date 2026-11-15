from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

AuctionKind = Literal["first", "second"]
AUCTION_KINDS: tuple[AuctionKind, ...] = get_args(AuctionKind)


def run_auction(
    bids: npt.NDArray[np.float64],
    kind: AuctionKind,
    reserve: float,
    rng: np.random.Generator,
) -> tuple[int, float]:
    """Return (winner, price); winner is -1 when no bid meets the reserve."""
    if kind not in AUCTION_KINDS:
        raise ValueError(f"unknown auction kind {kind!r}; use one of {AUCTION_KINDS}")
    valid = np.flatnonzero(bids >= reserve)
    if valid.size == 0:
        return -1, 0.0
    valid_bids = bids[valid]
    top = valid_bids.max()
    tied = valid[valid_bids == top]
    winner = int(rng.choice(tied)) if tied.size > 1 else int(tied[0])
    if kind == "first":
        return winner, float(top)
    second = np.sort(valid_bids)[-2] if valid_bids.size > 1 else reserve
    return winner, float(max(second, reserve))
