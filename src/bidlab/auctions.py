"""Single-slot auctions that pick a winner and a price from the bids."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

NO_WINNER = -1


def _pick_winner(
    bids: npt.NDArray[np.float64], reserve: float, rng: np.random.Generator
) -> tuple[int, npt.NDArray[np.float64]]:
    valid = np.flatnonzero(bids >= reserve)
    if valid.size == 0:
        return NO_WINNER, bids[valid]

    valid_bids = bids[valid]
    tied = valid[valid_bids == valid_bids.max()]
    winner = int(rng.choice(tied)) if tied.size > 1 else int(tied[0])
    return winner, valid_bids


@dataclass(frozen=True)
class FirstPrice:
    """The winner pays its own bid.

    The highest bid at or above the reserve wins. Ties are broken at random.

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
        self, bids: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> tuple[int, float]:
        """Pick the winner and the price it pays.

        Parameters
        ----------
        bids : numpy.ndarray
            One bid per bidder. A bidder that sits out bids ``-inf``.
        rng : numpy.random.Generator
            Breaks ties between equal highest bids.

        Returns
        -------
        winner : int
            Index of the winning bid, or -1 when no bid meets the reserve.
        price : float
            Price the winner pays; 0 when there is no winner.
        """
        winner, valid_bids = _pick_winner(bids, self.reserve, rng)
        if winner == NO_WINNER:
            return NO_WINNER, 0.0

        return winner, float(valid_bids.max())


@dataclass(frozen=True)
class SecondPrice:
    """The winner pays the higher of the second-highest bid and the reserve.

    The highest bid at or above the reserve wins. Ties are broken at random.
    Only bids at or above the reserve count as the second-highest bid.

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
        self, bids: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> tuple[int, float]:
        """Pick the winner and the price it pays.

        Parameters
        ----------
        bids : numpy.ndarray
            One bid per bidder. A bidder that sits out bids ``-inf``.
        rng : numpy.random.Generator
            Breaks ties between equal highest bids.

        Returns
        -------
        winner : int
            Index of the winning bid, or -1 when no bid meets the reserve.
        price : float
            Price the winner pays; 0 when there is no winner.
        """
        winner, valid_bids = _pick_winner(bids, self.reserve, rng)
        if winner == NO_WINNER:
            return NO_WINNER, 0.0

        second = np.sort(valid_bids)[-2] if valid_bids.size > 1 else self.reserve
        return winner, float(max(second, self.reserve))


Auction = FirstPrice | SecondPrice
