from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

ALL_SEGMENTS = "all"
SHARE_TOLERANCE = 1e-9


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


@dataclass(frozen=True)
class Segment:
    name: str
    share: float
    lift: float = 0.0


@dataclass(frozen=True)
class Segmented(Market):
    """Each step comes from one segment; its `lift` shifts every bidder's logit."""

    segments: tuple[Segment, ...]
    base: LogisticContext = LogisticContext()

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        if not self.segments:
            raise ValueError("Segmented needs at least one segment")

        names = [segment.name for segment in self.segments]
        if len(set(names)) != len(names):
            raise ValueError(f"segment names must be unique, got {names}")

        for segment in self.segments:
            if segment.share <= 0:
                raise ValueError(
                    f"segment {segment.name!r}: share must be > 0, got {segment.share}"
                )

        total = sum(segment.share for segment in self.segments)
        # Silent normalizing would let a typo such as 0.7 + 0.2 run unnoticed.
        if abs(total - 1.0) > SHARE_TOLERANCE:
            raise ValueError(f"segment shares must sum to 1, got {total}")

    @property
    def segment_names(self) -> tuple[str, ...]:
        return tuple(segment.name for segment in self.segments)

    def draws(self, bidders: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        base_rng, pick_rng = rng.spawn(2)
        shares = [segment.share for segment in self.segments]
        for logits in _logits(self.base, bidders, base_rng):
            segment = self.segments[pick_rng.choice(len(self.segments), p=shares)]
            yield MarketDraw(
                {"segment": segment.name},
                _sigmoid(logits + segment.lift),
                segment.name,
            )


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
        # A matrix product rounds differently as rows are added; a per-row sum does not.
        yield (ad_vectors * context).sum(axis=1) + biases


def _sigmoid(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-logits))
