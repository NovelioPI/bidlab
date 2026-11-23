"""`Callback`: read-only hooks that watch a run, or a set of seeds, as it goes."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from tqdm.auto import tqdm

from bidlab.config import SimConfig
from bidlab.results import Results
from bidlab.types import EpisodeReport

if TYPE_CHECKING:
    from bidlab.simulator import RunLog

SEEDS_FORMAT = (
    "{desc}   {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]"
)
# n and total stay unscaled; unit_scale shortens only the rate, as in 1.60k step/s.
STEPS_FORMAT = (
    "{desc} {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"
)


class Callback:
    """Base class for code that watches a run: override the hooks you need.

    Hooks run in the main process, in the order the callbacks are given, and
    cannot change the run. `run` calls `on_run_start`, `on_step_end`,
    `on_episode_end` and `on_run_end` for an int seed; for a seed list it
    calls only `on_seeds_start`, `on_seed_progress`, `on_seed_end` and
    `on_seeds_end`, also with ``workers > 1``. An error raised in a hook
    reaches the caller.

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

    def on_run_start(self, config: SimConfig, seed: int) -> None:
        """Run once, after the bidders are built and before the first episode.

        Parameters
        ----------
        config : SimConfig
            The run's config.
        seed : int
            The run's seed, as `RunLog.seed` will record it.
        """

    def on_step_end(self, episode: int, step: int) -> None:
        """Run after each step's auction and any feedback delivered at that step.

        It runs once per auction, so keep it fast.

        Parameters
        ----------
        episode : int
            The episode index, from 0.
        step : int
            The step index within the episode, from 0.
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

    def on_seed_progress(self, seed: int, episode: int, step: int) -> None:
        """Run for a seed list as a seed's run reaches ``step`` of ``episode``.

        It runs about every ``max(1, steps // 20)`` steps and at each
        episode's last step, in the main process. Each seed's calls come in
        step order and before its `on_seed_end`; with ``workers > 1``, calls
        for seeds that run at the same time interleave as they arrive.

        Parameters
        ----------
        seed : int
            The seed whose run advanced.
        episode : int
            The episode index, from 0.
        step : int
            The step index within the episode, from 0, that just ended.
        """

    def on_seed_end(self, seed: int, results: Results) -> None:
        """Run after each seed of a seed list, in the order the seeds were given.

        Parameters
        ----------
        seed : int
            The seed that finished.
        results : Results
            That seed's rows and logs, as `run` with ``seed=[seed]``.
        """

    def on_seeds_start(self, config: SimConfig, seeds: Sequence[int]) -> None:
        """Run once for a seed list, before the first seed.

        Parameters
        ----------
        config : SimConfig
            The config every seed runs.
        seeds : Sequence[int]
            Every seed, in the order they will run.
        """

    def on_seeds_end(self, results: Results) -> None:
        """Run once with the `Results` that `run` returns for a seed list.

        Parameters
        ----------
        results : Results
            Every seed's rows and logs.
        """


class ProgressBar(Callback):
    """Show tqdm progress bars on stderr: steps for one seed, seeds for a list.

    `run` adds one by default; pass ``progress_bar=False`` to hide it.
    Passing an instance in ``callbacks`` adds no second bar. For an int
    seed, one line labeled ``seed 42 episode 2/5`` counts the current
    episode's steps and starts again at each episode. For a seed list, a
    ``seeds`` line counts finished seeds, and each running seed has a line
    of that kind below it; a finished seed's line is cleared and reused by
    the next seed, so there are at most ``workers`` such lines. In a
    notebook it shows tqdm's notebook widgets. One instance can watch any
    number of runs; each gets its own bars.

    Examples
    --------
    >>> import bidlab as bl
    >>> campaign = bl.Campaign(
    ...     "t", bl.Stack(bl.OracleValue(), bl.Linear()), budget=10, base_value=5
    ... )
    >>> config = bl.SimConfig(campaigns=[campaign], episodes=2, steps=50)
    >>> log = bl.run(config, callbacks=[bl.ProgressBar()])
    >>> results = bl.run(config, seed=range(3), callbacks=[bl.ProgressBar()])
    """

    def __init__(self) -> None:
        self._bar: tqdm[NoReturn] | None = None
        self._episodes = 0
        self._seed = 0
        self._steps = 0
        self._seed_bars: dict[int, _SeedBar] = {}

    def on_run_start(self, config: SimConfig, seed: int) -> None:
        """Open a bar with one tick per step of the first episode."""
        self._episodes = config.episodes
        self._steps = config.steps
        self._seed = seed
        self._bar = self._steps_bar(seed, 0, line=None)

    def on_step_end(self, episode: int, step: int) -> None:
        """Tick the step bar."""
        if self._bar is not None:
            self._bar.update()

    def on_episode_end(
        self, episode: int, reports: Mapping[str, EpisodeReport]
    ) -> None:
        """Start the bar again for the next episode, on the same line."""
        if self._bar is not None and episode + 1 < self._episodes:
            self._bar.set_description_str(
                self._label(self._seed, episode + 1), refresh=False
            )
            self._bar.reset()

    def on_run_end(self, log: "RunLog") -> None:
        """Close the step bar."""
        self._close()

    def on_seeds_start(self, config: SimConfig, seeds: Sequence[int]) -> None:
        """Open the ``seeds`` bar with one tick per seed."""
        self._episodes = config.episodes
        self._steps = config.steps
        self._bar = tqdm(
            total=len(seeds), desc="seeds", position=0, bar_format=SEEDS_FORMAT
        )

    def on_seed_progress(self, seed: int, episode: int, step: int) -> None:
        """Move the seed's own bar to ``step`` of ``episode``; clear it at the end."""
        if seed not in self._seed_bars:
            self._seed_bars[seed] = self._open_seed_bar(seed, episode)

        shown = self._seed_bars[seed]
        if episode != shown.episode:
            shown.bar.set_description_str(self._label(seed, episode), refresh=False)
            shown.bar.reset()
            shown.episode = episode

        shown.bar.update(step + 1 - shown.bar.n)
        # Seeds finish out of order with workers, but on_seed_end comes in seed order.
        if episode == self._episodes - 1 and step == self._steps - 1:
            self._seed_bars.pop(seed).bar.close()

    def on_seed_end(self, seed: int, results: Results) -> None:
        """Tick the ``seeds`` bar."""
        if self._bar is not None:
            self._bar.update()

    def on_seeds_end(self, results: Results) -> None:
        """Close every bar."""
        for shown in self._seed_bars.values():
            shown.bar.close()

        self._seed_bars.clear()
        self._close()

    def _open_seed_bar(self, seed: int, episode: int) -> "_SeedBar":
        # A finished seed's line is free again, so the lowest free one is reused.
        taken = {shown.line for shown in self._seed_bars.values()}
        line = next(line for line in range(1, len(taken) + 2) if line not in taken)
        bar = self._steps_bar(seed, episode, line)
        # Progress comes only about 20 times an episode, so each call gets a frame.
        bar.mininterval = 0
        return _SeedBar(bar=bar, episode=episode, line=line)

    def _steps_bar(self, seed: int, episode: int, line: int | None) -> "tqdm[NoReturn]":
        # A line under the seeds bar is cleared at the end; a run's own bar stays.
        return tqdm(
            total=self._steps,
            desc=self._label(seed, episode),
            position=line,
            leave=line is None,
            unit=" step",
            unit_scale=True,
            bar_format=STEPS_FORMAT,
        )

    def _label(self, seed: int, episode: int) -> str:
        return f"seed {seed} episode {episode + 1}/{self._episodes}"

    def _close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


@dataclass
class _SeedBar:
    bar: "tqdm[NoReturn]"
    episode: int
    # The terminal line below the seeds bar, from 1.
    line: int


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


def with_progress_bar(callbacks: tuple[Callback, ...]) -> tuple[Callback, ...]:
    """Return ``callbacks`` with a `ProgressBar` added unless one is there."""
    if any(isinstance(callback, ProgressBar) for callback in callbacks):
        return callbacks

    return (*callbacks, ProgressBar())
