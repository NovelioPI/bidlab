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
    """The winner pays its own bid."""

    reserve: float = 0.0

    def __post_init__(self) -> None:
        if self.reserve < 0:
            raise ValueError(f"reserve must be >= 0, got {self.reserve}")

    def run(
        self, bids: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> tuple[int, float]:
        """Return (winner, price); winner is -1 when no bid meets the reserve."""
        winner, valid_bids = _pick_winner(bids, self.reserve, rng)
        if winner == NO_WINNER:
            return NO_WINNER, 0.0

        return winner, float(valid_bids.max())


@dataclass(frozen=True)
class SecondPrice:
    """The winner pays the higher of the second-highest bid and the reserve."""

    reserve: float = 0.0

    def __post_init__(self) -> None:
        if self.reserve < 0:
            raise ValueError(f"reserve must be >= 0, got {self.reserve}")

    def run(
        self, bids: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> tuple[int, float]:
        """Return (winner, price); winner is -1 when no bid meets the reserve."""
        winner, valid_bids = _pick_winner(bids, self.reserve, rng)
        if winner == NO_WINNER:
            return NO_WINNER, 0.0

        second = np.sort(valid_bids)[-2] if valid_bids.size > 1 else self.reserve
        return winner, float(max(second, self.reserve))


Auction = FirstPrice | SecondPrice
