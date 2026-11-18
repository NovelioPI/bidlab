"""Storage for the values bidders log with `Bidder.log` during a run."""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

Logs = Mapping[str, Mapping[str, npt.NDArray[np.float64]]]
"""Stored name, then spec name, to the logged values."""

_Store = dict[str, dict[str, npt.NDArray[np.float64]]]


class LogBook:
    """Collects one run's logged values on a grid of episodes and steps.

    The simulator moves the clock with `at_episode` and `at_step`. A value
    logged in an episode hook lands in an (episodes,) array; any other value
    in an (episodes, steps) array. Cells nobody logged stay NaN.

    Parameters
    ----------
    episodes, steps : int
        The run's grid size.
    """

    def __init__(self, episodes: int, steps: int):
        self.episodes = episodes
        self.steps = steps
        self.episode = 0
        self.step: int | None = None
        self.active = False
        self.step_values: _Store = {}
        self.episode_values: _Store = {}
        self.owners: dict[tuple[str, str], str] = {}

    def logger(self, spec: str, path: str) -> "NodeLogger":
        """Return the logger for the bidder at ``path`` in spec ``spec``."""
        return NodeLogger(self, spec, path)

    def at_episode(self, episode: int) -> None:
        """Point the clock at an episode hook of ``episode``."""
        self.active = True
        self.episode = episode
        self.step = None

    def at_step(self, episode: int, step: int) -> None:
        """Point the clock at auction ``step`` of ``episode``."""
        self.active = True
        self.episode = episode
        self.step = step

    def close(self) -> None:
        """Ignore every later write, so a bidder kept after the run logs nothing."""
        self.active = False

    def write(self, spec: str, path: str, name: str, value: float) -> None:
        """Store ``value`` under ``name`` for spec ``spec`` at the clock.

        Raises
        ------
        ValueError
            If another bidder in the tree already stored ``name``, or
            ``name`` was stored in the other shape.
        """
        if not self.active:
            return

        owner = self.owners.setdefault((spec, name), path)
        if owner != path:
            raise ValueError(
                f"bidder {spec!r}: {name!r} is logged by the bidders at {owner!r} "
                f"and {path!r}; give one a different name or keep its path prefix"
            )

        if self.step is None:
            mine, other, shape = self.episode_values, self.step_values, "episode"
            cell: tuple[int, ...] = (self.episode,)
            size: tuple[int, ...] = (self.episodes,)
        else:
            mine, other, shape = self.step_values, self.episode_values, "step"
            cell = (self.episode, self.step)
            size = (self.episodes, self.steps)

        if spec in other.get(name, {}):
            raise ValueError(
                f"bidder {spec!r}: {name!r} is logged both per step and per "
                f"episode; this {shape} log needs a different name"
            )

        arrays = mine.setdefault(name, {})
        if spec not in arrays:
            arrays[spec] = np.full(size, np.nan)

        arrays[spec][cell] = value

    def result(self) -> Logs:
        """Return every logged array, read-only."""
        merged: _Store = {}
        for store in (self.step_values, self.episode_values):
            for name, arrays in store.items():
                merged.setdefault(name, {}).update(arrays)

        for arrays in merged.values():
            for array in arrays.values():
                array.flags.writeable = False

        logs = {name: MappingProxyType(arrays) for name, arrays in merged.items()}
        return MappingProxyType(logs)


class NodeLogger:
    """Writes one bidder's logged values to its run's `LogBook`."""

    def __init__(self, book: LogBook, spec: str, path: str):
        self.book = book
        self.spec = spec
        self.path = path

    def write(self, name: str, value: float, log_prefix: bool) -> None:
        """Store ``value`` under ``name``, prefixed with the path unless told not to."""
        stored = f"{self.path}.{name}" if self.path and log_prefix else name
        self.book.write(self.spec, self.path, stored, value)
