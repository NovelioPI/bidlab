"""The `Bidder` base class that every bidding method builds on."""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol, overload

import numpy as np

from bidlab.types import BidResult, EpisodeReport

if TYPE_CHECKING:
    from bidlab.config import Campaign

_LOGGER = "_bidlab_logger"
_FORWARD_PLAN = "_bidlab_forward_plan"

LogValue = float | np.floating[Any] | np.integer[Any]
"""A number `Bidder.log` accepts: a Python number or a numpy scalar."""
_SEQUENCES = (list, tuple)
_HIDDEN = (list, tuple, dict, set, frozenset)


class Logger(Protocol):
    """Where a bidder's logged values go during a run; the simulator makes one."""

    def write(self, name: str, value: float, log_prefix: bool) -> None:
        """Store one value."""
        ...


class Bidder(ABC):
    """Base class for a bidding method: implement `forward`.

    A bidder turns a value into another value. Alone, it gets the platform's
    value as the ``value`` input, and its output is the bid. In a `Stack`,
    each bidder gets the previous one's output as ``value``. A value model,
    such as a pCVR model, returns its own value and may ignore the one it
    gets. A bid-control method, such as pacing or shading, turns a value into
    a bid. `forward` declares the inputs it needs as parameters, like a
    PyTorch module's ``forward``.

    Take settings as constructor arguments, and keep learned state on
    ``self``. The simulator deep-copies the bidder once per run, so every
    run starts from the bidder as built, and the bidder keeps what it learns
    from one episode to the next. The simulator calls, in order:
    `on_episode_start`, then `forward` and `update_batch` once per auction,
    then `on_episode_end`. Outside the campaign's ``segments``, `forward` and
    `update_batch` are not called.

    A bidder may hold other bidders, its children, in attributes, lists or
    tuples. Call a child's `forward` yourself, with any arguments it takes.
    The simulator calls every other hook on every child, parent first, and
    gives each child its own ``rng``. Do not call a child's
    `on_episode_start`, `update_batch` or `on_episode_end` yourself, or it
    runs twice. Create children in ``__init__``: a child added during a run
    raises `ValueError`. Outside a run, `Campaign.build` binds the whole
    tree, and `bind` binds only this bidder.

    Save any number with `log`, such as a learned multiplier. The run keeps
    it in `RunLog.logged`, and `run_many` in `Results.logged`.

    Attributes
    ----------
    rng : numpy.random.Generator
        This bidder's own random stream, set by `bind` before the run. Use it
        for every random choice, so runs stay reproducible.

    See Also
    --------
    Stack : Chains bidders.
    named_bidders : Lists this bidder's children.
    Campaign : Puts a bidder in a simulation.

    Examples
    --------
    >>> import bidlab as bl
    >>> class Floor(bl.Bidder):
    ...     def __init__(self, minimum):
    ...         self.minimum = minimum
    ...     def forward(self, value):
    ...         return max(value, self.minimum)
    >>> bidder = bl.Stack(bl.Linear(multiplier=0.5), Floor(0.3))
    >>> campaign = bl.Campaign("floored", bidder, budget=10, event_value=5)
    >>> log = bl.run(bl.SimConfig(campaigns=[campaign], episodes=1, steps=20))
    >>> bool((log.bids >= 0.3).all())
    True
    """

    rng: np.random.Generator

    def bind(self, rng: np.random.Generator) -> None:
        """Give this bidder its random stream; the simulator calls this per run.

        Parameters
        ----------
        rng : numpy.random.Generator
            The bidder's random stream for the run.
        """
        self.rng = rng

    def on_episode_start(self, campaign: "Campaign") -> None:
        """Reset per-episode state before the first auction of an episode.

        Parameters
        ----------
        campaign : Campaign
            The campaign this bidder runs for: its budget, value, CPA target
            and segments. Its ``bidder`` is the running copy, the top bidder
            of this tree, not the object the user built.
        """

    # Each bidder declares its own inputs, so the base accepts any signature.
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> float:
        """Turn the incoming value and inputs into this bidder's output.

        Declare the inputs you need as parameters, such as
        ``forward(self, value, time_left, country)``, or take every input
        with ``**inputs``. The simulator passes the top bidder only the
        inputs it names; a `Stack` does the same for each of its bidders. A
        parent calls a child's `forward` with any arguments it chooses.

        Parameters
        ----------
        value : float
            The platform's value of the impression: `Campaign.event_value`
            times its estimate of the conversion probability, exact unless
            `SimConfig.estimate_noise` is set. In a `Stack`, the previous
            bidder's output.
        time : float
            Seconds since the episode start. Auctions are spaced evenly over
            the horizon, so the first is at 0.
        time_left : float
            Seconds left in the episode: ``horizon - time``.
        horizon : float
            Episode length in seconds.
        hour_of_day : float
            Hour of day at ``time``, in [0, 24).
        budget_left : float
            The campaign's budget left before this auction.
        segment : str
            The auction's market segment; ``"all"`` in a market without
            segments.
        **features : str or float
            Each public feature of the impression from the market, under its
            own name, the same for every bidder.

        Returns
        -------
        float
            The bid, or the value passed to the next bidder in a `Stack`. A
            final bid of NaN or infinity becomes 0, and a bid above the
            budget left is capped at it.
        """

    def update_batch(self, bid_results: Sequence[BidResult]) -> None:
        """Learn from the outcomes of auctions this bidder took part in.

        Today each batch holds the result of one auction. The default calls
        `update` once per item, in time order.

        Parameters
        ----------
        results : Sequence[BidResult]
            This bidder's own outcomes. Each holds the price only when the
            bidder won. In a `Stack`, `BidResult.bid` is the final bid.
        """
        for result in bid_results:
            self.update(result)

    def update(self, result: BidResult) -> None:
        """Learn from the outcome of one auction this bidder took part in.

        Parameters
        ----------
        result : BidResult
            This bidder's own outcome.
        """

    def on_episode_end(self, report: EpisodeReport) -> None:
        """Learn from the episode's totals after its last auction.

        Parameters
        ----------
        report : EpisodeReport
            This bidder's totals for the episode.
        """

    @overload
    def log(
        self, name: str, value: LogValue, /, *, log_prefix: bool = True
    ) -> None: ...

    @overload
    def log(
        self, values: Mapping[str, LogValue], /, *, log_prefix: bool = True
    ) -> None: ...

    def log(
        self,
        name_or_values: str | Mapping[str, LogValue],
        value: LogValue | None = None,
        /,
        *,
        log_prefix: bool = True,
    ) -> None:
        """Save a number, or a dict of numbers, in the run's `RunLog.logged`.

        A value logged in `forward`, `update_batch` or `update` is saved at
        the current auction step; one logged in `on_episode_start` or
        `on_episode_end` is saved per episode. Logging one name twice in one
        step keeps the last value. Outside a run, such as in a unit test or in
        `bind`, nothing is saved.

        Parameters
        ----------
        name_or_values : str or Mapping[str, LogValue]
            A name, or a dict of names to values.
        value : LogValue, optional
            The value, when ``name_or_values`` is a name.
        log_prefix : bool, optional
            Save a child bidder's names under its path, such as
            ``"bidders.1.lam"``. False saves the plain name.

        Raises
        ------
        TypeError
            If a name is not a non-empty str, or a value is not a real number.

        Examples
        --------
        >>> import bidlab as bl
        >>> class Half(bl.Bidder):
        ...     def forward(self, value):
        ...         self.log("factor", 0.5)
        ...         return value / 2
        >>> campaign = bl.Campaign("half", Half(), budget=10, event_value=5)
        >>> log = bl.run(bl.SimConfig(campaigns=[campaign], episodes=2, steps=50))
        >>> log.logged["factor"]["half"].shape, float(log.logged["factor"]["half"][1, 49])
        ((2, 50), 0.5)
        """
        values = _checked_values(name_or_values, value)
        logger: Logger | None = vars(self).get(_LOGGER)
        if logger is None:
            return

        for name, number in values.items():
            logger.write(name, number, log_prefix)

    def named_bidders(self) -> Iterator[tuple[str, "Bidder"]]:
        """Yield every bidder in this bidder's tree, with its path.

        A child is a `Bidder` held in an attribute, or as an item of a list or
        tuple attribute. The order is parent first, then each child's tree in
        attribute order and item order. A bidder reached twice is listed once,
        at its first path. Nothing is bound or changed.

        Yields
        ------
        path : str
            ``""`` for this bidder; attribute names and list indexes joined by
            dots for the rest, such as ``"shade"`` or ``"bidders.0"``.
        bidder : Bidder
            The bidder at that path.

        Raises
        ------
        TypeError
            If a `dict`, `set` or `frozenset` attribute holds a bidder, or a
            list or tuple item is a container that holds one.

        Examples
        --------
        >>> import bidlab as bl
        >>> bidder = bl.Stack(bl.Linear(), bl.Stack(bl.PidPacing()))
        >>> [path for path, _ in bidder.named_bidders()]
        ['', 'bidders.0', 'bidders.1', 'bidders.1.bidders.0']
        """
        for path, bidder, _ in _walk(self, "", {id(self)}):
            yield path, bidder


def bind_tree(bidder: Bidder, rng: np.random.Generator) -> None:
    """Bind every bidder in ``bidder``'s tree to its own stream; the simulator calls this.

    ``bidder`` gets ``rng``. Each bidder's stream spawns one stream per new
    child, in tree order, before the bidder's `Bidder.bind` runs. So a
    `bind` override that spawns its own streams cannot shift the children's.

    Parameters
    ----------
    bidder : Bidder
        The top bidder of the tree.
    rng : numpy.random.Generator
        The top bidder's random stream for the run.
    """
    streams = {id(bidder): rng}
    for _, node, children in _walk(bidder, "", {id(bidder)}):
        stream = streams[id(node)]
        for child, child_stream in zip(
            children, stream.spawn(len(children)), strict=True
        ):
            streams[id(child)] = child_stream

        node.bind(stream)


@dataclass(frozen=True)
class _ForwardPlan:
    takes_all: bool
    names: tuple[str, ...]
    required: tuple[str, ...]


def call_forward(bidder: Bidder, inputs: Mapping[str, object]) -> float:
    """Call ``bidder.forward`` with the inputs it names; the simulator and `Stack` call this.

    A ``forward`` with ``**kwargs`` gets every input. Otherwise it gets the
    inputs its parameters name, and a parameter with no input gets its
    default.

    Parameters
    ----------
    bidder : Bidder
        The bidder to call.
    inputs : Mapping[str, object]
        Every input of this auction, by name.

    Returns
    -------
    float
        What ``bidder.forward`` returns.

    Raises
    ------
    TypeError
        If ``forward`` has a positional-only parameter, or a parameter with
        no default that is not in ``inputs``.
    """
    plan: _ForwardPlan | None = vars(bidder).get(_FORWARD_PLAN)
    # Reading the signature on every auction would slow every run.
    if plan is None:
        plan = _forward_plan(bidder)
        vars(bidder)[_FORWARD_PLAN] = plan

    for name in plan.required:
        if name not in inputs:
            raise TypeError(
                f"{type(bidder).__name__}.forward needs {name!r}, which is not an "
                f"input here; give it a default or use one of {sorted(inputs)}"
            )

    if plan.takes_all:
        return bidder.forward(**inputs)

    return bidder.forward(
        **{name: inputs[name] for name in plan.names if name in inputs}
    )


def _forward_plan(bidder: Bidder) -> _ForwardPlan:
    takes_all = False
    names: list[str] = []
    required: list[str] = []
    for parameter in inspect.signature(bidder.forward).parameters.values():
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{type(bidder).__name__}.forward: {parameter.name!r} is "
                "positional-only, but inputs are passed by name"
            )

        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            takes_all = True
        elif parameter.kind is not inspect.Parameter.VAR_POSITIONAL:
            names.append(parameter.name)
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter.name)

    return _ForwardPlan(takes_all, tuple(names), tuple(required))


def attach_logger(bidder: Bidder, logger: Logger) -> None:
    """Send ``bidder``'s logged values to ``logger``; the simulator calls this.

    Parameters
    ----------
    bidder : Bidder
        One bidder of a tree built for a run.
    logger : Logger
        Where its values go.
    """
    vars(bidder)[_LOGGER] = logger


def _checked_values(
    name_or_values: str | Mapping[str, LogValue], value: LogValue | None
) -> dict[str, float]:
    if isinstance(name_or_values, Mapping):
        if value is not None:
            raise TypeError("pass a dict alone, or a name and a value")

        pairs = list(name_or_values.items())
    else:
        pairs = [(name_or_values, value)]

    checked: dict[str, float] = {}
    for name, number in pairs:
        if not isinstance(name, str) or not name:
            raise TypeError(f"a logged name must be a non-empty str, got {name!r}")
        # Real covers bool, int, float and numpy numbers, and rejects arrays.

        if not isinstance(number, Real):
            raise TypeError(
                f"logged value {name!r} must be a real number, "
                f"got {type(number).__name__}"
            )

        checked[name] = float(number)

    return checked


def _walk(
    node: Bidder, path: str, claimed: set[int]
) -> Iterator[tuple[str, Bidder, list[Bidder]]]:
    # A node claims all its new children before the walk enters any of them.
    children: list[tuple[str, Bidder]] = []
    for child_path, child in _direct_children(node, path):
        if id(child) not in claimed:
            claimed.add(id(child))
            children.append((child_path, child))

    yield path, node, [child for _, child in children]
    for child_path, child in children:
        yield from _walk(child, child_path, claimed)


def _direct_children(node: Bidder, path: str) -> Iterator[tuple[str, Bidder]]:
    for name, value in vars(node).items():
        child_path = f"{path}.{name}" if path else name
        if isinstance(value, Bidder):
            yield child_path, value
        elif isinstance(value, _SEQUENCES):
            for index, item in enumerate(value):
                item_path = f"{child_path}.{index}"
                if isinstance(item, Bidder):
                    yield item_path, item
                else:
                    _reject_hidden(item, item_path)

        else:
            _reject_hidden(value, child_path)


def _reject_hidden(value: object, path: str) -> None:
    if not isinstance(value, _HIDDEN):
        return

    items = value.values() if isinstance(value, dict) else value
    if any(isinstance(item, Bidder) for item in items):
        raise TypeError(
            f"the {type(value).__name__} at {path!r} holds a bidder the simulator "
            "cannot find; hold child bidders in an attribute, list or tuple"
        )
