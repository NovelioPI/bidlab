"""The `Bidder` base class that every bidding method builds on."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

import numpy as np

from bidlab.types import Campaign, EpisodeStats, Feedback, Observation

_SEQUENCES = (list, tuple)
_HIDDEN = (list, tuple, dict, set, frozenset)


class Bidder(ABC):
    """Base class for a bidding method: implement `forward`.

    A bidder turns a value into another value. Alone, it gets the platform's
    value, `Observation.value`, and its output is the bid. In a `Stack`, each
    bidder gets the previous one's output. A value model, such as a pCVR
    model, returns its own value and may ignore the one it gets. A
    bid-control method, such as pacing or shading, turns a value into a bid.

    Take settings as constructor arguments, and keep learned state on
    ``self``. The simulator deep-copies the bidder once per run, so every
    run starts from the bidder as built, and the bidder keeps what it learns
    from one episode to the next. The simulator calls, in order:
    `on_episode_start`, then `forward` and `update_batch` once per auction,
    then `on_episode_end`. Outside the campaign's ``segments``, `forward` and
    `update_batch` are not called.

    Attributes
    ----------
    rng : numpy.random.Generator
        This bidder's own random stream, set by `bind` before the run. Use it
        for every random choice, so runs stay reproducible.

    See Also
    --------
    Stack : Chains bidders.
    BidderSpec : Puts a bidder in a simulation.

    Examples
    --------
    >>> import bidlab as bl
    >>> class Floor(bl.Bidder):
    ...     def __init__(self, minimum):
    ...         self.minimum = minimum
    ...     def forward(self, value, observation):
    ...         return max(value, self.minimum)
    >>> bidder = bl.Stack(bl.Truthful(multiplier=0.5), Floor(0.3))
    >>> spec = bl.BidderSpec("floored", bidder, bl.Campaign(budget=10, value=5))
    >>> log = bl.run(bl.SimConfig(bidders=[spec], episodes=1, steps=20))
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

    def on_episode_start(self, campaign: Campaign) -> None:
        """Reset per-episode state before the first auction of an episode.

        Parameters
        ----------
        campaign : Campaign
            The bidder's fixed facts, with the schedule filled in.
        """

    @abstractmethod
    def forward(self, value: float, observation: Observation) -> float:
        """Turn the incoming value into this bidder's output.

        Parameters
        ----------
        value : float
            `Observation.value`, or the previous bidder's output in a
            `Stack`.
        observation : Observation
            What this bidder can see about the impression.

        Returns
        -------
        float
            The bid, or the value passed to the next bidder in a `Stack`. A
            final bid of NaN or infinity becomes 0, and a bid above the
            budget left is capped at it.
        """

    def update_batch(self, feedbacks: Sequence[Feedback]) -> None:
        """Learn from the outcomes of auctions this bidder took part in.

        Today each batch holds the feedback of one auction. The default calls
        `update` once per item, in time order.

        Parameters
        ----------
        feedbacks : Sequence[Feedback]
            This bidder's own outcomes. Each holds the price only when the
            bidder won. In a `Stack`, `Feedback.bid` is the final bid.
        """
        for feedback in feedbacks:
            self.update(feedback)

    def update(self, feedback: Feedback) -> None:
        """Learn from the outcome of one auction this bidder took part in.

        Parameters
        ----------
        feedback : Feedback
            This bidder's own outcome.
        """

    def on_episode_end(self, stats: EpisodeStats, episode_reward: float | None) -> None:
        """Learn from the episode's totals after its last auction.

        Parameters
        ----------
        stats : EpisodeStats
            This bidder's totals for the episode.
        episode_reward : float or None
            The spec's ``episode_reward`` for this episode, or None when the
            spec sets none.
        """

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
        >>> bidder = bl.Stack(bl.Truthful(), bl.Stack(bl.PidPacing()))
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
            f"the bidder at {path!r} is inside a {type(value).__name__}, so the "
            "simulator cannot find it; hold child bidders in an attribute, "
            "list or tuple"
        )
