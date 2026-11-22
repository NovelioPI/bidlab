"""Markets: where each step's features, click and conversion probabilities come from."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType
from typing import NamedTuple, Protocol

import numpy as np
import numpy.typing as npt
from scipy.special import expit, logit

ALL_SEGMENTS = "all"
SHARE_TOLERANCE = 1e-9
SAMPLE_BLOCK_SIZE = 1024
DEFAULT_CVR_BIAS_MEAN = -3.0
QUADRATURE_NODES, QUADRATURE_WEIGHTS = np.polynomial.hermite.hermgauss(200)
BISECTION_STEPS = 60

type FeatureValue = str | float | npt.NDArray[np.float64]


class Distribution(Protocol):
    """A scipy frozen distribution, such as `scipy.stats.uniform(0, 1)`."""

    def rvs(
        self, size: int, random_state: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """Return ``size`` independent draws from ``random_state``."""
        ...


def _checked_floor_share(
    floor_share: Sequence[float] | None, has_price: bool, price_name: str
) -> tuple[float, float] | None:
    if floor_share is None:
        return None

    if not has_price:
        raise ValueError(f"floor_share needs {price_name}: a floor is a share of it")

    shape = tuple(float(parameter) for parameter in floor_share)
    if len(shape) != 2 or not all(parameter > 0 for parameter in shape):
        raise ValueError(
            f"floor_share must be two Beta parameters (a, b), both > 0, got {floor_share}"
        )

    return shape[0], shape[1]


def _check_rate(name: str, rate: float) -> None:
    # A negated check also catches NaN, which fails every comparison.
    if not 0 < rate < 1:
        raise ValueError(f"{name} must be in (0, 1), got {rate}")


def _check_distribution(name: str, distribution: object) -> None:
    if not callable(getattr(distribution, "rvs", None)):
        raise TypeError(
            f"{name} must have an rvs method, such as scipy.stats.uniform(0, 1); "
            f"got {type(distribution).__name__}"
        )


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
    market_price : float, optional
        The highest bid from outside the config, >= 0. A campaign wins only
        with a bid above it. None, the default, means no outside competition.
    floor : float, optional
        The publisher's floor for this auction, >= 0. The lowest bid that can
        win is the higher of this and the auction's ``reserve``.

    Notes
    -----
    A campaign's bidder sees its own ``pctr`` and ``pcvr`` in the ``market``
    input. No bidder sees ``market_price``; each sees the effective floor as
    ``floor``.
    """

    features: Mapping[str, FeatureValue]
    _: KW_ONLY
    pctr: npt.NDArray[np.float64] | None = None
    pcvr: npt.NDArray[np.float64]
    segment: str = ALL_SEGMENTS
    market_price: float | None = None
    floor: float = 0.0

    def __post_init__(self) -> None:
        features = {name: _read_only(value) for name, value in self.features.items()}
        object.__setattr__(self, "features", MappingProxyType(features))
        if self.pctr is not None:
            object.__setattr__(self, "pctr", np.asarray(self.pctr, dtype=float))
        object.__setattr__(self, "pcvr", np.asarray(self.pcvr, dtype=float))
        if self.market_price is not None:
            object.__setattr__(self, "market_price", float(self.market_price))
        object.__setattr__(self, "floor", float(self.floor))


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
    >>> campaign = bl.Campaign(
    ...     "t", bl.Stack(bl.OracleValue(), bl.Linear()), budget=10, base_value=5
    ... )
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

    Set each stage by its rate, or by its bias for the logit model itself.

    Parameters
    ----------
    feature_count : int, optional
        Length of the feature vector and the ad vectors.
    ctr : float, optional
        Click rate, in (0, 1): campaign i's mean pCTR over a run is
        σ(logit(ctr) + δᵢ), with δᵢ ~ N(0, ``ctr_spread``). Turns on the
        click stage.
    cvr : float, optional
        Conversion rate per click, in (0, 1), set the same way with
        ``cvr_spread``.
    ctr_bias_mean : float, optional
        Mean of the per-campaign click bias, in place of ``ctr``. Without
        either, the market has no click stage: every won impression is
        clicked.
    ctr_spread : float, optional
        Spread of the per-campaign click bias, or of δᵢ with ``ctr``, on the
        logit scale.
    cvr_bias_mean : float, optional
        Mean of the per-campaign conversion bias, in place of ``cvr``. None,
        the default, means -3 without ``cvr``, which gives pCVR near 5%.
    cvr_spread : float, optional
        Spread of the per-campaign conversion bias, or of δᵢ with ``cvr``.
    public_features : int, optional
        How many features bidders see, from 0 to ``feature_count``. None
        means all.
    price_median : float, optional
        Median market price, in money, > 0. None, the default, means no
        market price and no floor.
    price_spread : float, optional
        Standard deviation of log market price around the median and the
        quality term, >= 0.
    price_quality : float, optional
        How strongly log market price follows the impression's quality: the
        feature vector · a hidden market vector, of variance near 1.
    rival_similarity : float, optional
        How much the campaigns' ad vectors share the market vector, in
        [0, 1]. At 0, the default, market price follows the features but no
        campaign's value; above 0, it rises with each campaign's value.
    floor_share : tuple of float, optional
        Parameters (a, b) of a Beta distribution, both > 0. Each auction's
        floor is a share drawn from it times the market price. None means no
        floor. Needs ``price_median``.

    Raises
    ------
    TypeError
        If ``public_features`` is not an int.
    ValueError
        If ``public_features`` is outside [0, ``feature_count``], a rate or a
        price setting is out of range, or a rate and its bias are both set.

    Notes
    -----
    With quality q and a standard normal ε drawn each step::

        log m = log(price_median) + price_quality × q + price_spread × ε

    Each ad vector is ρ ψ + √(1 − ρ²) η, where ψ is the market vector, η the
    campaign's own vector, and ρ is ``rival_similarity``.

    Examples
    --------
    A 2% click rate and a 5% conversion rate, with 3 public features:

    >>> import numpy as np
    >>> import bidlab as bl
    >>> market = bl.markets.LogisticMarket(ctr=0.02, cvr=0.05, public_features=3)
    >>> draw = next(market.draws(2, np.random.default_rng(0)))
    >>> draw.pcvr.shape, draw.pctr.shape, draw.features["vector"].shape
    ((2,), (2,), (3,))

    A market price that follows each campaign's value, with a floor:

    >>> market = bl.markets.LogisticMarket(
    ...     price_median=0.2, rival_similarity=0.8, floor_share=(2, 5)
    ... )
    >>> draw = next(market.draws(2, np.random.default_rng(0)))
    >>> bool(0 <= draw.floor <= draw.market_price)
    True
    """

    feature_count: int = 8
    ctr: float | None = None
    cvr: float | None = None
    ctr_bias_mean: float | None = None
    ctr_spread: float = 0.5
    cvr_bias_mean: float | None = None
    cvr_spread: float = 0.5
    public_features: int | None = None
    price_median: float | None = None
    price_spread: float = 0.5
    price_quality: float = 1.0
    rival_similarity: float = 0.0
    floor_share: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self._check_rates()
        self._check_price()
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

    def _check_rates(self) -> None:
        stages = (
            ("ctr", self.ctr, self.ctr_bias_mean),
            ("cvr", self.cvr, self.cvr_bias_mean),
        )
        for name, rate, bias_mean in stages:
            if rate is None:
                continue

            _check_rate(name, rate)
            if bias_mean is not None:
                raise ValueError(f"set {name} or {name}_bias_mean, not both")

    def _check_price(self) -> None:
        # Negated checks also catch NaN, which fails every comparison.
        if self.price_median is not None and not self.price_median > 0:
            raise ValueError(f"price_median must be > 0, got {self.price_median}")

        if not self.price_spread >= 0:
            raise ValueError(f"price_spread must be >= 0, got {self.price_spread}")

        if not 0 <= self.rival_similarity <= 1:
            raise ValueError(
                f"rival_similarity must be in [0, 1], got {self.rival_similarity}"
            )

        floor_share = _checked_floor_share(
            self.floor_share, self.price_median is not None, "price_median"
        )
        object.__setattr__(self, "floor_share", floor_share)

    def draws(self, count: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        """Yield one draw per step; see `Market.draws`."""
        for step in _LogisticStages(self, count, rng).steps():
            pctr = None if step.ctr_logits is None else _sigmoid(step.ctr_logits)
            yield MarketDraw(
                {"vector": step.public_vector},
                pctr=pctr,
                pcvr=_sigmoid(step.cvr_logits),
                market_price=step.market_price,
                floor=step.floor,
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
    ctr : float, optional
        Click rate in this segment, in (0, 1), in place of the market's.
        Campaign i's mean pCTR here is σ(logit(ctr) + δᵢ), with the same δᵢ
        as in the rest of the market. Needs a click stage in the base market.
    cvr : float, optional
        Conversion rate per click in this segment, set the same way.
    ctr_lift : float, optional
        Added to every campaign's click logit in this segment, in place of
        ``ctr``. Needs a click stage in the base market.
    cvr_lift : float, optional
        Added to every campaign's CVR logit in this segment, in place of
        ``cvr``.

    Raises
    ------
    ValueError
        If a rate is outside (0, 1), or a stage's rate and lift are both set.

    Examples
    --------
    >>> from bidlab.markets import Segment
    >>> mobile = Segment("mobile", 0.7, ctr=0.03, cvr=0.04)
    >>> sports = Segment("sports", 0.3, cvr_lift=1.0)
    """

    name: str
    share: float
    _: KW_ONLY
    ctr: float | None = None
    cvr: float | None = None
    ctr_lift: float = 0.0
    cvr_lift: float = 0.0

    def __post_init__(self) -> None:
        stages = (("ctr", self.ctr, self.ctr_lift), ("cvr", self.cvr, self.cvr_lift))
        for name, rate, lift in stages:
            if rate is None:
                continue

            _check_rate(name, rate)
            if lift != 0:
                raise ValueError(
                    f"segment {self.name!r}: set {name} or {name}_lift, not both"
                )


@dataclass(frozen=True)
class Segmented(Market):
    """Each step comes from one segment, which can set its own rates or lifts.

    Bidders get the segment name as the ``segment`` input of `Bidder.forward`,
    and the base market's ``"vector"`` feature. A segment without a rate or a
    lift keeps the base market's rates. The market price and floor are the
    base market's too.

    Parameters
    ----------
    segments : Sequence[Segment]
        At least one segment, stored as a tuple, with unique names and shares
        that sum to 1.
    base : LogisticMarket, optional
        Draws the features and logits before a segment changes them.

    Raises
    ------
    ValueError
        If there is no segment, a name repeats, a share is <= 0, the shares
        do not sum to 1, or a segment sets a click rate or lift when the base
        has no click stage.

    Examples
    --------
    >>> import bidlab as bl
    >>> from bidlab.markets import Segment, Segmented
    >>> market = Segmented((Segment("mobile", 0.7, cvr_lift=0.5), Segment("desktop", 0.3)))
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

        if self.base.ctr is not None or self.base.ctr_bias_mean is not None:
            return

        for segment in self.segments:
            if segment.ctr is not None or segment.ctr_lift != 0:
                raise ValueError(
                    f"segment {segment.name!r} sets a click rate or lift, but the "
                    "base market has no click stage; set ctr on the base"
                )

    @property
    def segment_names(self) -> tuple[str, ...]:
        """The segment names, in the order given."""
        return tuple(segment.name for segment in self.segments)

    def draws(self, count: int, rng: np.random.Generator) -> Iterator[MarketDraw]:
        """Yield one draw per step; see `Market.draws`."""
        base_rng, pick_rng = rng.spawn(2)
        shares = [segment.share for segment in self.segments]
        stages = _LogisticStages(self.base, count, base_rng)
        click, conversion = stages.click, stages.conversion
        # A lift stays a float, so a segment without a rate keeps every old bit.
        ctr_shifts = [
            segment.ctr_lift
            if segment.ctr is None or click is None
            else click.shift(segment.ctr)
            for segment in self.segments
        ]
        cvr_shifts = [
            segment.cvr_lift if segment.cvr is None else conversion.shift(segment.cvr)
            for segment in self.segments
        ]
        for step in stages.steps():
            index = pick_rng.choice(len(self.segments), p=shares)
            segment = self.segments[index]
            pctr = None
            if step.ctr_logits is not None:
                pctr = _sigmoid(step.ctr_logits + ctr_shifts[index])
            yield MarketDraw(
                {"vector": step.public_vector},
                pctr=pctr,
                pcvr=_sigmoid(step.cvr_logits + cvr_shifts[index]),
                segment=segment.name,
                market_price=step.market_price,
                floor=step.floor,
            )


@dataclass(frozen=True)
class IndependentValues(Market):
    """Each bidder's probability is an independent, clipped draw from `distribution`.

    Draws are clipped to [0, 1]. Each bidder has its own stream, so adding a
    bidder leaves the others' probabilities unchanged. The market price and
    floor come from the market's own stream, apart from every bidder's.

    Parameters
    ----------
    distribution : Distribution
        A scipy frozen distribution, such as ``scipy.stats.beta(2, 30)``.
    market_price : Distribution, optional
        Each auction's market price is a draw from it, clipped at 0. It does
        not follow any probability. None, the default, means no market price.
    floor_share : tuple of float, optional
        Parameters (a, b) of a Beta distribution, both > 0. Each auction's
        floor is a share drawn from it times the market price. None means no
        floor. Needs ``market_price``.

    Raises
    ------
    TypeError
        If ``distribution`` or ``market_price`` has no ``rvs`` method.
    ValueError
        If ``floor_share`` is set without ``market_price``, or a parameter is
        not > 0.

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
    market_price: Distribution | None = None
    floor_share: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _check_distribution("distribution", self.distribution)
        if self.market_price is not None:
            _check_distribution("market_price", self.market_price)
        floor_share = _checked_floor_share(
            self.floor_share, self.market_price is not None, "market_price"
        )
        object.__setattr__(self, "floor_share", floor_share)

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
            prices: list[float | None] = [None] * SAMPLE_BLOCK_SIZE
            floors = np.zeros(SAMPLE_BLOCK_SIZE)
            # Spawning leaves the parent stream unused, so the price draws move no pCVR.
            if self.market_price is not None:
                price_block = np.clip(
                    self.market_price.rvs(size=SAMPLE_BLOCK_SIZE, random_state=rng),
                    0.0,
                    None,
                )
                prices = price_block.tolist()
                if self.floor_share is not None:
                    shares = rng.beta(*self.floor_share, size=SAMPLE_BLOCK_SIZE)
                    floors = price_block * shares

            rows = zip(np.clip(block, 0.0, 1.0), prices, floors, strict=True)
            for pcvr, price, floor in rows:
                yield MarketDraw({}, pcvr=pcvr, market_price=price, floor=floor)


class _Step(NamedTuple):
    public_vector: npt.NDArray[np.float64]
    ctr_logits: npt.NDArray[np.float64] | None
    cvr_logits: npt.NDArray[np.float64]
    market_price: float | None
    floor: float


@dataclass(frozen=True, eq=False)
class _Stage:
    vectors: npt.NDArray[np.float64]
    biases: npt.NDArray[np.float64]
    offsets: npt.NDArray[np.float64]

    def logits(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # A matrix product rounds differently as rows are added; a per-row sum does not.
        return (self.vectors * features).sum(axis=1) + self.biases

    def shift(self, rate: float) -> npt.NDArray[np.float64]:
        return _rate_biases(self.vectors, self.offsets, rate) - self.biases


def _bias_stage(
    vectors: npt.NDArray[np.float64],
    bias_rng: np.random.Generator,
    mean: float,
    spread: float,
) -> _Stage:
    biases = bias_rng.normal(mean, spread, size=len(vectors))
    return _Stage(vectors, biases, biases - mean)


def _rate_stage(
    vectors: npt.NDArray[np.float64],
    bias_rng: np.random.Generator,
    rate: float,
    spread: float,
) -> _Stage:
    # One draw per campaign, as a bias stage takes, so no other draw moves.
    offsets = bias_rng.normal(0.0, spread, size=len(vectors))
    return _Stage(vectors, _rate_biases(vectors, offsets, rate), offsets)


def _rate_biases(
    vectors: npt.NDArray[np.float64], offsets: npt.NDArray[np.float64], rate: float
) -> npt.NDArray[np.float64]:
    scales = np.sqrt((vectors**2).sum(axis=1))
    return _solve_biases(expit(logit(rate) + offsets), scales)


def _solve_biases(
    targets: npt.NDArray[np.float64], scales: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    # The logit is N(b, s²), so the mean rate E[σ(b + sZ)] is a Gauss–Hermite sum.
    points = np.sqrt(2.0) * scales[:, np.newaxis] * QUADRATURE_NODES
    weights = QUADRATURE_WEIGHTS / np.sqrt(np.pi)
    # Spread moves the mean rate by less than 1 + s² on the logit scale.
    width = 2.0 * (1.0 + scales**2)
    low = logit(targets) - width
    high = logit(targets) + width
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2
        above = (expit(middle[:, np.newaxis] + points) * weights).sum(axis=1) > targets
        high = np.where(above, middle, high)
        low = np.where(above, low, middle)

    return (low + high) / 2


class _LogisticStages:
    def __init__(
        self, market: LogisticMarket, count: int, rng: np.random.Generator
    ) -> None:
        dimensions = market.feature_count
        # Separate streams keep campaign i's draws equal when campaigns are added.
        # Risk: if new streams came before the first five, then every older draw would move.
        (
            cvr_vector_rng,
            cvr_bias_rng,
            self._feature_rng,
            ctr_vector_rng,
            ctr_bias_rng,
            market_vector_rng,
            self._price_rng,
            self._floor_rng,
        ) = rng.spawn(8)
        self._market = market
        scale = 1.0 / np.sqrt(dimensions)
        self._market_vector = np.zeros(dimensions)
        if market.price_median is not None or market.rival_similarity > 0:
            self._market_vector = market_vector_rng.normal(0.0, scale, size=dimensions)

        def ad_vectors(vector_rng: np.random.Generator) -> npt.NDArray[np.float64]:
            return _share_market_vector(
                vector_rng.normal(0.0, scale, size=(count, dimensions)),
                self._market_vector,
                market.rival_similarity,
            )

        self.click: _Stage | None = None
        if market.ctr is not None:
            self.click = _rate_stage(
                ad_vectors(ctr_vector_rng), ctr_bias_rng, market.ctr, market.ctr_spread
            )
        elif market.ctr_bias_mean is not None:
            self.click = _bias_stage(
                ad_vectors(ctr_vector_rng),
                ctr_bias_rng,
                market.ctr_bias_mean,
                market.ctr_spread,
            )

        if market.cvr is not None:
            self.conversion = _rate_stage(
                ad_vectors(cvr_vector_rng), cvr_bias_rng, market.cvr, market.cvr_spread
            )
        else:
            mean = market.cvr_bias_mean
            self.conversion = _bias_stage(
                ad_vectors(cvr_vector_rng),
                cvr_bias_rng,
                DEFAULT_CVR_BIAS_MEAN if mean is None else mean,
                market.cvr_spread,
            )

    def steps(self) -> Iterator[_Step]:
        market = self._market
        dimensions = market.feature_count
        public = (
            dimensions if market.public_features is None else market.public_features
        )
        while True:
            features = self._feature_rng.normal(size=dimensions)
            ctr_logits = None if self.click is None else self.click.logits(features)
            cvr_logits = self.conversion.logits(features)
            market_price = None
            floor = 0.0
            if market.price_median is not None:
                quality = (self._market_vector * features).sum()
                noise = self._price_rng.standard_normal()
                market_price = float(
                    market.price_median
                    * np.exp(
                        market.price_quality * quality + market.price_spread * noise
                    )
                )
                if market.floor_share is not None:
                    floor = market_price * float(
                        self._floor_rng.beta(*market.floor_share)
                    )

            public_vector = features[:public]
            # Nothing else holds this step's features, so a read-only view needs no copy.
            public_vector.flags.writeable = False
            yield _Step(public_vector, ctr_logits, cvr_logits, market_price, floor)


def _share_market_vector(
    vectors: npt.NDArray[np.float64],
    market_vector: npt.NDArray[np.float64],
    similarity: float,
) -> npt.NDArray[np.float64]:
    # At 0 the vectors pass through untouched, so existing runs keep every bit.
    if similarity == 0:
        return vectors

    return similarity * market_vector + np.sqrt(1 - similarity**2) * vectors


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
