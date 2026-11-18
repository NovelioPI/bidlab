"""`Callback`: read-only hooks that watch a run, or a set of seeds, as it goes."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from bidlab.config import SimConfig
from bidlab.results import Results
from bidlab.types import EpisodeStats

if TYPE_CHECKING:
    from bidlab.simulator import RunLog


class Callback:
    """Base class for code that watches a run: override the hooks you need.

    Hooks run in the main process, in the order the callbacks are given, and
    cannot change the run. `run` calls `on_run_start`, `on_episode_end` and
    `on_run_end`; `run_many` calls only `on_seed_end`, also with
    ``workers > 1``. An error raised in a hook reaches the caller.

    Examples
    --------
    >>> import bidlab as bl
    >>> class Progress(bl.Callback):
    ...     def on_episode_end(self, episode, stats):
    ...         print(episode, stats["t"].wins > 0)
    >>> spec = bl.BidderSpec("t", bl.Truthful(), bl.Campaign(budget=10, value=5))
    >>> config = bl.SimConfig(bidders=[spec], episodes=2, steps=50)
    >>> log = bl.run(config, callbacks=[Progress()])
    0 True
    1 True
    """

    def on_run_start(self, config: SimConfig) -> None:
        """Run once, after the bidders are built and before the first episode.

        Parameters
        ----------
        config : SimConfig
            The run's config.
        """

    def on_episode_end(self, episode: int, stats: Mapping[str, EpisodeStats]) -> None:
        """Run after every bidder's `Bidder.on_episode_end` for ``episode``.

        Parameters
        ----------
        episode : int
            The episode index, from 0.
        stats : Mapping[str, EpisodeStats]
            Each spec name's totals for the episode.
        """

    def on_run_end(self, log: "RunLog") -> None:
        """Run once with the finished, read-only log that `run` returns.

        Parameters
        ----------
        log : RunLog
            The run's log.
        """

    def on_seed_end(self, seed: int, results: Results) -> None:
        """Run in `run_many` after each seed, in the order the seeds were given.

        Parameters
        ----------
        seed : int
            The seed that finished.
        results : Results
            That seed's rows and logs, as `run_many` with that seed alone.
        """


def checked_callbacks(callbacks: Sequence[object]) -> tuple[Callback, ...]:
    """Return ``callbacks`` as a tuple after checking each is a `Callback` instance.

    Raises
    ------
    TypeError
        If an item is a class, or not a `Callback`.
    """
    checked: list[Callback] = []
    for callback in callbacks:
        if isinstance(callback, type):
            raise TypeError(
                f"callbacks take Callback instances, got the class "
                f"{callback.__name__}; pass an instance, such as {callback.__name__}()"
            )

        if not isinstance(callback, Callback):
            raise TypeError(
                f"callbacks take Callback instances, got {type(callback).__name__}"
            )

        checked.append(callback)

    return tuple(checked)
