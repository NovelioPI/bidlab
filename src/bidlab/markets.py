from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

ALL_SEGMENTS = "all"


@dataclass(frozen=True)
class MarketDraw:
    features: Mapping[str, str | float]
    probabilities: npt.NDArray[np.float64]
    segment: str = ALL_SEGMENTS

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        object.__setattr__(
            self, "probabilities", np.asarray(self.probabilities, dtype=float)
        )


class Market(ABC):
    """Draws each step's public features and every bidder's conversion probability."""

    @property
    def segment_names(self) -> tuple[str, ...]:
        return (ALL_SEGMENTS,)

    @abstractmethod
    def draws(self, bidders: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        """Yield one draw per step, for the whole run."""


@dataclass(frozen=True)
class LogisticContext(Market):
    """Conversion probability σ(ad vector · random context + bias) per bidder."""

    context_dim: int = 8
    bias_mean: float = -3.0
    bias_std: float = 0.5

    def draws(self, bidders: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        for logits in _logits(self, bidders, rng):
            yield MarketDraw({}, _sigmoid(logits))


def _logits(
    market: LogisticContext, bidders: int, rng: np.random.Generator
) -> Iterator[npt.NDArray[np.float64]]:
    dimensions = market.context_dim
    # Separate streams keep bidder i's draws equal when bidders are added.
    vector_rng, bias_rng, context_rng = rng.spawn(3)
    ad_vectors = vector_rng.normal(
        0.0, 1.0 / np.sqrt(dimensions), size=(bidders, dimensions)
    )
    biases = bias_rng.normal(market.bias_mean, market.bias_std, size=bidders)
    while True:
        context = context_rng.normal(size=dimensions)
        yield ad_vectors @ context + biases


def _sigmoid(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-logits))
