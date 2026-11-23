"""`Callback`: read-only hooks that watch a run, or a set of seeds, as it goes."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, NoReturn

from tqdm.auto import tqdm

from bidlab.config import SimConfig
from bidlab.results import Results
from bidlab.types import EpisodeReport

if TYPE_CHECKING:
    from bidlab.simulator import RunLog


class Callback:
    """Base class for code that watches a run: override the hooks you need.

    Hooks run in the main process, in the order the callbacks are given, and
    cannot change the run. `run` calls `on_run_start`, `on_episode_end` and
    `on_run_end`; `run_many` calls only `on_run_many_start`, `on_seed_end`
    and `on_run_many_end`, also with ``workers > 1``. An error raised in a
    hook reaches the caller.

    Examples
    --------
    >>> import bidlab as bl
    >>> class Progress(bl.Callback):
    ...     def on_episode_end(self, episode, reports):
    ...         print(episode, reports["t"].wins > 0)
    >>> campaign = bl.Campaign(
    ...     "t", bl.Stack(bl.OracleValue(), bl.Linear()), budget=10, base_value=5
    ... )
    >>> config = bl.SimConfig(campaigns=[campaign], episodes=2, steps=50)
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

    def on_episode_end(
        self, episode: int, reports: Mapping[str, EpisodeReport]
    ) -> None:
        """Run after every bidder's `Bidder.on_episode_end` for ``episode``.

        Parameters
        ----------
        episode : int
            The episode index, from 0.
        reports : Mapping[str, EpisodeReport]
            Each campaign name's totals for the episode.
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

    def on_run_many_start(self, config: SimConfig, seeds: Sequence[int]) -> None:
        """Run once in `run_many`, before the first seed.

        Parameters
        ----------
        config : SimConfig
            The config every seed runs.
        seeds : Sequence[int]
            Every seed, in the order they will run.
        """

    def on_run_many_end(self, results: Results) -> None:
        """Run once with the `Results` that `run_many` returns.

        Parameters
        ----------
        results : Results
            Every seed's rows and logs.
        """


class ProgressBar(Callback):
    """Show a tqdm progress bar on stderr: episodes in `run`, seeds in `run_many`.

    In a notebook it shows tqdm's notebook widget. One instance can watch any
    number of runs; each gets its own bar.

    Examples
    --------
    >>> import bidlab as bl
    >>> campaign = bl.Campaign(
    ...     "t", bl.Stack(bl.OracleValue(), bl.Linear()), budget=10, base_value=5
    ... )
    >>> config = bl.SimConfig(campaigns=[campaign], episodes=2, steps=50)
    >>> log = bl.run(config, callbacks=[bl.ProgressBar()])
    >>> results = bl.run_many(config, seeds=range(3), callbacks=[bl.ProgressBar()])
    """

    def __init__(self) -> None:
        self._bar: tqdm[NoReturn] | None = None

    def on_run_start(self, config: SimConfig) -> None:
        """Open a bar with one tick per episode."""
        self._bar = tqdm(total=config.episodes, desc="run", unit="episode")

    def on_episode_end(
        self, episode: int, reports: Mapping[str, EpisodeReport]
    ) -> None:
        """Tick the episode bar."""
        if self._bar is not None:
            self._bar.update()

    def on_run_end(self, log: "RunLog") -> None:
        """Close the episode bar."""
        self._close()

    def on_run_many_start(self, config: SimConfig, seeds: Sequence[int]) -> None:
        """Open a bar with one tick per seed."""
        self._bar = tqdm(total=len(seeds), desc="run_many", unit="seed")

    def on_seed_end(self, seed: int, results: Results) -> None:
        """Tick the seed bar."""
        if self._bar is not None:
            self._bar.update()

    def on_run_many_end(self, results: Results) -> None:
        """Close the seed bar."""
        self._close()

    def _close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


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
