"""Markets: where each step's features and conversion probabilities come from."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np
import numpy.typing as npt

ALL_SEGMENTS = "all"
SHARE_TOLERANCE = 1e-9
SAMPLE_BLOCK_SIZE = 1024


class Distribution(Protocol):
    """A scipy frozen distribution, such as `scipy.stats.uniform(0, 1)`."""

    def rvs(
        self, size: int, random_state: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """Return ``size`` independent draws from ``random_state``."""
        ...


@dataclass(frozen=True)
class MarketDraw:
    """One step's draw: public features, private probabilities and the segment.

    Parameters
    ----------
    features : Mapping[str, str | float]
        Public features every bidder sees in `Observation.features`.
    probabilities : numpy.ndarray
        Each bidder's conversion probability, shape ``(bidders,)``, in [0, 1].
        A bidder sees only its own, as part of its value.
    segment : str, optional
        Name of this step's segment; must be in the market's `segment_names`.
    """

    features: Mapping[str, str | float]
    probabilities: npt.NDArray[np.float64]
    segment: str = ALL_SEGMENTS

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        object.__setattr__(
            self, "probabilities", np.asarray(self.probabilities, dtype=float)
        )


class Market(ABC):
    """Draws each step's public features and every bidder's conversion probability.

    Subclass it and implement `draws`. Override `segment_names` when the
    market has segments.

    Examples
    --------
    A market where every bidder has the same fixed probability:

    >>> import numpy as np
    >>> import bidlab as bl
    >>> class Flat(bl.markets.Market):
    ...     def draws(self, bidders, rng):
    ...         while True:
    ...             yield bl.markets.MarketDraw({}, np.full(bidders, 0.1))
    >>> spec = bl.BidderSpec("t", bl.Truthful(), bl.Campaign(budget=10, value=5))
    >>> log = bl.run(bl.SimConfig(bidders=[spec], market=Flat(), episodes=1, steps=10))
    >>> log.values[0, :3, 0].tolist()
    [0.5, 0.5, 0.5]
    """

    @property
    def segment_names(self) -> tuple[str, ...]:
        """Every segment name a draw can have; ``("all",)`` by default."""
        return (ALL_SEGMENTS,)

    @abstractmethod
    def draws(self, bidders: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        """Yield one draw per step, for the whole run.

        Parameters
        ----------
        bidders : int
            Number of bidders; each draw holds one probability per bidder.
        rng : numpy.random.Generator
            The market's own random stream. Use only this stream, so runs
            stay reproducible.

        Yields
        ------
        MarketDraw
            The next step's draw. The generator must not stop before the run
            ends.
        """


@dataclass(frozen=True)
class LogisticContext(Market):
    """Conversion probability σ(ad vector · random context + bias) per bidder.

    Each bidder gets a fixed ad vector and bias for the whole run. Each step
    draws a new standard normal context, so bidders' probabilities move
    together on some steps and apart on others.

    Parameters
    ----------
    context_dim : int, optional
        Length of the context and ad vectors.
    bias_mean : float, optional
        Mean of the per-bidder bias; -3 gives probabilities near 5%.
    bias_std : float, optional
        Spread of the per-bidder bias.

    Examples
    --------
    >>> import numpy as np
    >>> import bidlab as bl
    >>> draws = bl.markets.LogisticContext().draws(3, np.random.default_rng(0))
    >>> next(draws).probabilities.shape
    (3,)
    """

    context_dim: int = 8
    bias_mean: float = -3.0
    bias_std: float = 0.5

    def draws(self, bidders: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        """Yield one draw per step; see `Market.draws`."""
        for logits in _logits(self, bidders, rng):
            yield MarketDraw({}, _sigmoid(logits))


@dataclass(frozen=True)
class Segment:
    """One segment of a `Segmented` market.

    Parameters
    ----------
    name : str
        Segment name; bidders list it in `BidderSpec.segments`.
    share : float
        Fraction of steps from this segment. Shares must sum to 1.
    lift : float, optional
        Added to every bidder's logit in this segment.
    """

    name: str
    share: float
    lift: float = 0.0


@dataclass(frozen=True)
class Segmented(Market):
    """Each step comes from one segment; its `lift` shifts every bidder's logit.

    The segment name goes in `Observation.features` under ``"segment"``.

    Parameters
    ----------
    segments : Sequence[Segment]
        At least one segment, stored as a tuple, with unique names and shares
        that sum to 1.
    base : LogisticContext, optional
        Draws the logits before the lift is added.

    Raises
    ------
    ValueError
        If there is no segment, a name repeats, a share is <= 0, or the
        shares do not sum to 1.

    Examples
    --------
    >>> import bidlab as bl
    >>> from bidlab.markets import Segment, Segmented
    >>> market = Segmented((Segment("mobile", 0.7, lift=0.5), Segment("desktop", 0.3)))
    >>> market.segment_names
    ('mobile', 'desktop')
    """

    segments: Sequence[Segment]
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
        """The segment names, in the order given."""
        return tuple(segment.name for segment in self.segments)

    def draws(self, bidders: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        """Yield one draw per step; see `Market.draws`."""
        base_rng, pick_rng = rng.spawn(2)
        shares = [segment.share for segment in self.segments]
        for logits in _logits(self.base, bidders, base_rng):
            segment = self.segments[pick_rng.choice(len(self.segments), p=shares)]
            yield MarketDraw(
                {"segment": segment.name},
                _sigmoid(logits + segment.lift),
                segment.name,
            )


@dataclass(frozen=True)
class IndependentValues(Market):
    """Each bidder's probability is an independent, clipped draw from `distribution`.

    Draws are clipped to [0, 1]. Each bidder has its own stream, so adding a
    bidder leaves the others' probabilities unchanged.

    Parameters
    ----------
    distribution : Distribution
        A scipy frozen distribution, such as ``scipy.stats.beta(2, 30)``.

    Raises
    ------
    TypeError
        If ``distribution`` has no ``rvs`` method.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import bidlab as bl
    >>> market = bl.markets.IndependentValues(stats.uniform(0, 0.1))
    >>> draw = next(market.draws(2, np.random.default_rng(0)))
    >>> bool(((draw.probabilities >= 0) & (draw.probabilities <= 0.1)).all())
    True
    """

    distribution: Distribution

    def __post_init__(self) -> None:
        if not callable(getattr(self.distribution, "rvs", None)):
            raise TypeError(
                "distribution must have an rvs method, such as scipy.stats.uniform(0, 1); "
                f"got {type(self.distribution).__name__}"
            )

    def draws(self, bidders: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        """Yield one draw per step; see `Market.draws`."""
        # Child stream i depends only on i, so adding a bidder keeps the others' values.
        streams = rng.spawn(bidders)
        while True:
            block = np.column_stack(
                [
                    self.distribution.rvs(size=SAMPLE_BLOCK_SIZE, random_state=stream)
                    for stream in streams
                ]
            )
            for probabilities in np.clip(block, 0.0, 1.0):
                yield MarketDraw({}, probabilities)


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
