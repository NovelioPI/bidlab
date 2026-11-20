"""Markets: where each step's features, click and conversion probabilities come from."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType
from typing import NamedTuple, Protocol

import numpy as np
import numpy.typing as npt

ALL_SEGMENTS = "all"
SHARE_TOLERANCE = 1e-9
SAMPLE_BLOCK_SIZE = 1024

type FeatureValue = str | float | npt.NDArray[np.float64]


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

    Every argument after ``features`` is keyword-only, so ``pctr`` and
    ``pcvr`` cannot be swapped by position.

    Parameters
    ----------
    features : Mapping[str, str | float | numpy.ndarray]
        Public features every bidder sees, in the ``features`` input of
        `Bidder.forward` and each as an input of its own name, which must
        not be a built-in input. A writable array is stored as a read-only
        copy; a read-only array is kept as is.
    pctr : numpy.ndarray, optional
        Each campaign's click probability, shape ``(campaigns,)``, in [0, 1].
        None means the market has no click stage: every won impression is
        clicked.
    pcvr : numpy.ndarray
        Each campaign's conversion probability after a click, shape
        ``(campaigns,)``, in [0, 1].
    segment : str, optional
        Name of this step's segment; must be in the market's `segment_names`.

    Notes
    -----
    A campaign's bidder sees its own ``pctr`` and ``pcvr`` in the ``market``
    input, and the platform's estimate of their product in ``value``.
    """

    features: Mapping[str, FeatureValue]
    _: KW_ONLY
    pctr: npt.NDArray[np.float64] | None = None
    pcvr: npt.NDArray[np.float64]
    segment: str = ALL_SEGMENTS

    def __post_init__(self) -> None:
        features = {name: _read_only(value) for name, value in self.features.items()}
        object.__setattr__(self, "features", MappingProxyType(features))
        if self.pctr is not None:
            object.__setattr__(self, "pctr", np.asarray(self.pctr, dtype=float))
        object.__setattr__(self, "pcvr", np.asarray(self.pcvr, dtype=float))


class Market(ABC):
    """Draws each step's public features and every campaign's probabilities.

    Subclass it and implement `draws`. Override `segment_names` when the
    market has segments.

    Examples
    --------
    A market where every bidder has the same fixed probability:

    >>> import numpy as np
    >>> import bidlab as bl
    >>> class Flat(bl.markets.Market):
    ...     def draws(self, count, rng):
    ...         while True:
    ...             yield bl.markets.MarketDraw({}, pcvr=np.full(count, 0.1))
    >>> campaign = bl.Campaign("t", bl.Linear(), budget=10, event_value=5)
    >>> log = bl.run(bl.SimConfig(campaigns=[campaign], market=Flat(), episodes=1, steps=10))
    >>> log.true_values[0, :3, 0].tolist()
    [0.5, 0.5, 0.5]
    """

    @property
    def segment_names(self) -> tuple[str, ...]:
        """Every segment name a draw can have; ``("all",)`` by default."""
        return (ALL_SEGMENTS,)

    @abstractmethod
    def draws(self, count: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        """Yield one draw per step, for the whole run.

        Parameters
        ----------
        count : int
            Number of campaigns; each draw holds one probability per campaign
            and stage.
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
class LogisticMarket(Market):
    """pCVR = σ(ad vector · feature vector + bias) per campaign; pCTR the same way.

    Each campaign gets fixed ad vectors and biases for the whole run. Each
    step draws a new standard normal feature vector, so campaigns'
    probabilities move together on some steps and apart on others. The first
    ``public_features`` numbers of the vector reach bidders as
    ``features["vector"]``; the rest still drive the probabilities unseen.

    Parameters
    ----------
    feature_count : int, optional
        Length of the feature vector and the ad vectors.
    ctr_bias_mean : float, optional
        Mean of the per-campaign click bias. None, the default, means no
        click stage: every won impression is clicked.
    ctr_bias_sd : float, optional
        Spread of the per-campaign click bias.
    cvr_bias_mean : float, optional
        Mean of the per-campaign conversion bias; -3 gives pCVR near 5%.
    cvr_bias_sd : float, optional
        Spread of the per-campaign conversion bias.
    public_features : int, optional
        How many features bidders see, from 0 to ``feature_count``. None
        means all.

    Raises
    ------
    TypeError
        If ``public_features`` is not an int.
    ValueError
        If ``public_features`` is outside [0, ``feature_count``].

    Examples
    --------
    >>> import numpy as np
    >>> import bidlab as bl
    >>> market = bl.markets.LogisticMarket(ctr_bias_mean=-4.0, public_features=3)
    >>> draw = next(market.draws(2, np.random.default_rng(0)))
    >>> draw.pcvr.shape, draw.pctr.shape, draw.features["vector"].shape
    ((2,), (2,), (3,))
    """

    feature_count: int = 8
    ctr_bias_mean: float | None = None
    ctr_bias_sd: float = 0.5
    cvr_bias_mean: float = -3.0
    cvr_bias_sd: float = 0.5
    public_features: int | None = None

    def __post_init__(self) -> None:
        public = self.public_features
        if public is None:
            return

        if isinstance(public, bool) or not isinstance(public, int):
            raise TypeError(
                f"public_features must be an int or None, got {type(public).__name__}"
            )

        if not 0 <= public <= self.feature_count:
            raise ValueError(
                f"public_features must be in [0, {self.feature_count}], got {public}"
            )

    def draws(self, count: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        """Yield one draw per step; see `Market.draws`."""
        for stages in _logistic_stages(self, count, rng):
            yield MarketDraw(
                {"vector": stages.public_vector},
                pctr=stages.pctr,
                pcvr=_sigmoid(stages.cvr_logits),
            )


@dataclass(frozen=True)
class Segment:
    """One segment of a `Segmented` market.

    Parameters
    ----------
    name : str
        Segment name; bidders list it in `Campaign.segments`.
    share : float
        Fraction of steps from this segment. Shares must sum to 1.
    lift : float, optional
        Added to every campaign's CVR logit in this segment.
    """

    name: str
    share: float
    lift: float = 0.0


@dataclass(frozen=True)
class Segmented(Market):
    """Each step comes from one segment; its `lift` shifts every campaign's CVR logit.

    Bidders get the segment name as the ``segment`` input of `Bidder.forward`,
    and the base market's ``"vector"`` feature and click stage unchanged.

    Parameters
    ----------
    segments : Sequence[Segment]
        At least one segment, stored as a tuple, with unique names and shares
        that sum to 1.
    base : LogisticMarket, optional
        Draws the features and logits before the lift is added.

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
    base: LogisticMarket = LogisticMarket()

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

    def draws(self, count: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        """Yield one draw per step; see `Market.draws`."""
        base_rng, pick_rng = rng.spawn(2)
        shares = [segment.share for segment in self.segments]
        for stages in _logistic_stages(self.base, count, base_rng):
            segment = self.segments[pick_rng.choice(len(self.segments), p=shares)]
            yield MarketDraw(
                {"vector": stages.public_vector},
                pctr=stages.pctr,
                pcvr=_sigmoid(stages.cvr_logits + segment.lift),
                segment=segment.name,
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
    >>> bool(((draw.pcvr >= 0) & (draw.pcvr <= 0.1)).all())
    True
    """

    distribution: Distribution

    def __post_init__(self) -> None:
        if not callable(getattr(self.distribution, "rvs", None)):
            raise TypeError(
                "distribution must have an rvs method, such as scipy.stats.uniform(0, 1); "
                f"got {type(self.distribution).__name__}"
            )

    def draws(self, count: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        """Yield one draw per step; see `Market.draws`."""
        # Child stream i depends only on i, so adding a bidder keeps the others' values.
        streams = rng.spawn(count)
        while True:
            block = np.column_stack(
                [
                    self.distribution.rvs(size=SAMPLE_BLOCK_SIZE, random_state=stream)
                    for stream in streams
                ]
            )
            for pcvr in np.clip(block, 0.0, 1.0):
                yield MarketDraw({}, pcvr=pcvr)


class _Stages(NamedTuple):
    public_vector: npt.NDArray[np.float64]
    pctr: npt.NDArray[np.float64] | None
    cvr_logits: npt.NDArray[np.float64]


def _logistic_stages(
    market: LogisticMarket, count: int, rng: np.random.Generator
) -> Iterator[_Stages]:
    dimensions = market.feature_count
    public = dimensions if market.public_features is None else market.public_features
    # Separate streams keep campaign i's draws equal when campaigns are added.
    # Risk: if the click streams were spawned first, then every CVR draw would move.
    cvr_vector_rng, cvr_bias_rng, feature_rng, ctr_vector_rng, ctr_bias_rng = rng.spawn(
        5
    )
    scale = 1.0 / np.sqrt(dimensions)
    click_weights = None
    if market.ctr_bias_mean is not None:
        click_weights = (
            ctr_vector_rng.normal(0.0, scale, size=(count, dimensions)),
            ctr_bias_rng.normal(market.ctr_bias_mean, market.ctr_bias_sd, size=count),
        )

    cvr_vectors = cvr_vector_rng.normal(0.0, scale, size=(count, dimensions))
    cvr_biases = cvr_bias_rng.normal(
        market.cvr_bias_mean, market.cvr_bias_sd, size=count
    )
    while True:
        features = feature_rng.normal(size=dimensions)
        pctr = None
        # A matrix product rounds differently as rows are added; a per-row sum does not.
        if click_weights is not None:
            ctr_vectors, ctr_biases = click_weights
            pctr = _sigmoid((ctr_vectors * features).sum(axis=1) + ctr_biases)

        cvr_logits = (cvr_vectors * features).sum(axis=1) + cvr_biases
        public_vector = features[:public]
        # Nothing else holds this step's features, so a read-only view needs no copy.
        public_vector.flags.writeable = False
        yield _Stages(public_vector, pctr, cvr_logits)


def _read_only(value: FeatureValue) -> FeatureValue:
    # A market that already froze its array owns it; copying it every step would slow runs.
    if not isinstance(value, np.ndarray) or not value.flags.writeable:
        return value

    # A copy, so neither the market nor a bidder can change what others see.
    copy = value.copy()
    copy.flags.writeable = False
    return copy


def _sigmoid(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-logits))
