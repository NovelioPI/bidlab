"""Single-slot auctions that pick a winner and a price from the bids."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

NO_WINNER = -1


def _pick_winner(
    bids: npt.NDArray[np.float64],
    floor: float,
    market_price: float | None,
    rng: np.random.Generator,
) -> tuple[int, npt.NDArray[np.float64]]:
    eligible = bids >= floor
    # Without a market price, runs skip the extra compare and keep their old path.
    if market_price is not None:
        eligible &= bids > market_price
    valid = np.flatnonzero(eligible)
    if valid.size == 0:
        return NO_WINNER, bids[valid]

    valid_bids = bids[valid]
    tied = valid[valid_bids == valid_bids.max()]
    winner = int(rng.choice(tied)) if tied.size > 1 else int(tied[0])
    return winner, valid_bids


@dataclass(frozen=True)
class FirstPrice:
    """The winner pays its own bid.

    The highest bid wins when it is at or above the floor and above the
    market price. The floor is the higher of ``reserve`` and the market's
    floor. Ties are broken at random.

    Parameters
    ----------
    reserve : float, optional
        Lowest bid that can win. Must be >= 0.

    Raises
    ------
    ValueError
        If ``reserve`` is negative.

    See Also
    --------
    SecondPrice : The winner pays the second-highest bid.

    Examples
    --------
    >>> import numpy as np
    >>> import bidlab as bl
    >>> bl.FirstPrice(reserve=0.1).run(np.array([0.5, 0.3]), np.random.default_rng(0))
    (0, 0.5)
    """

    reserve: float = 0.0

    def __post_init__(self) -> None:
        if self.reserve < 0:
            raise ValueError(f"reserve must be >= 0, got {self.reserve}")

    def run(
        self,
        bids: npt.NDArray[np.float64],
        rng: np.random.Generator,
        *,
        floor: float = 0.0,
        market_price: float | None = None,
    ) -> tuple[int, float]:
        """Pick the winner and the price it pays.

        Parameters
        ----------
        bids : numpy.ndarray
            One bid per bidder. A bidder that sits out bids ``-inf``.
        rng : numpy.random.Generator
            Breaks ties between equal highest bids.
        floor : float, optional
            The market's floor for this auction; the higher of it and
            ``reserve`` is the lowest bid that can win.
        market_price : float, optional
            The highest bid from outside the bidders; a bid must be above it
            to win. None means no outside bid.

        Returns
        -------
        winner : int
            Index of the winning bid, or -1 when no bid can win.
        price : float
            Price the winner pays; 0 when there is no winner.
        """
        lowest = max(self.reserve, floor)
        winner, valid_bids = _pick_winner(bids, lowest, market_price, rng)
        if winner == NO_WINNER:
            return NO_WINNER, 0.0

        return winner, float(valid_bids.max())


@dataclass(frozen=True)
class SecondPrice:
    """The winner pays the highest of the next bid, the floor and the market price.

    The highest bid wins when it is at or above the floor and above the
    market price. The floor is the higher of ``reserve`` and the market's
    floor. Ties are broken at random. Only bids that can win count as the
    next bid.

    Parameters
    ----------
    reserve : float, optional
        Lowest bid that can win, and the lowest price. Must be >= 0.

    Raises
    ------
    ValueError
        If ``reserve`` is negative.

    See Also
    --------
    FirstPrice : The winner pays its own bid.

    Examples
    --------
    >>> import numpy as np
    >>> import bidlab as bl
    >>> auction = bl.SecondPrice(reserve=0.1)
    >>> auction.run(np.array([0.5, 0.3]), np.random.default_rng(0))
    (0, 0.3)
    >>> auction.run(np.array([0.5, 0.05]), np.random.default_rng(0))
    (0, 0.1)
    """

    reserve: float = 0.0

    def __post_init__(self) -> None:
        if self.reserve < 0:
            raise ValueError(f"reserve must be >= 0, got {self.reserve}")

    def run(
        self,
        bids: npt.NDArray[np.float64],
        rng: np.random.Generator,
        *,
        floor: float = 0.0,
        market_price: float | None = None,
    ) -> tuple[int, float]:
        """Pick the winner and the price it pays.

        Parameters
        ----------
        bids : numpy.ndarray
            One bid per bidder. A bidder that sits out bids ``-inf``.
        rng : numpy.random.Generator
            Breaks ties between equal highest bids.
        floor : float, optional
            The market's floor for this auction; the higher of it and
            ``reserve`` is the lowest bid that can win.
        market_price : float, optional
            The highest bid from outside the bidders; a bid must be above it
            to win. None means no outside bid.

        Returns
        -------
        winner : int
            Index of the winning bid, or -1 when no bid can win.
        price : float
            Price the winner pays; 0 when there is no winner.
        """
        lowest = max(self.reserve, floor)
        winner, valid_bids = _pick_winner(bids, lowest, market_price, rng)
        if winner == NO_WINNER:
            return NO_WINNER, 0.0

        second = np.sort(valid_bids)[-2] if valid_bids.size > 1 else lowest
        # The floor is >= 0, so 0 stands in for no market price.
        outside = 0.0 if market_price is None else market_price
        return winner, float(max(second, lowest, outside))


Auction = FirstPrice | SecondPrice
