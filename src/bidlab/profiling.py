"""Time each bidder hook and sample memory during a run: `Profile`."""

import csv
import functools
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import psutil

from bidlab.bidders import Bidder

HOOKS = (
    "forward",
    "update_batch",
    "update",
    "update_conversions",
    "on_episode_start",
    "on_episode_end",
)
HEADER = ("campaign", "node", "hook", "calls", "total_s", "self_s", "mean_us", "share")
TOP_NODE = "(top)"
# Campaign, node and hook align left; the numbers after them align right.
LABEL_COLUMNS = 3
MICROSECONDS = 1e6
MEBIBYTE = 1024 * 1024
CSV_COLUMNS = (
    "seed",
    "campaign",
    "path",
    "hook",
    "calls",
    "total_seconds",
    "self_seconds",
    "wall_seconds",
    "steps",
    "memory_start",
    "memory_peak",
)


@dataclass(frozen=True)
class HookTime:
    """Time spent in one hook of one bidder node, summed over a run.

    Attributes
    ----------
    campaign : str
        The campaign's name.
    path : str
        The node's path in the campaign's bidder tree, as in
        `Bidder.named_bidders`: ``""`` for the top bidder.
    hook : str
        The hook's name, such as ``"forward"``.
    calls : int
        How many times the hook was called.
    total : float
        Seconds from each call's start to its end, summed.
    self_time : float
        `total` minus the time of the hooks this hook called, such as a
        `Stack`'s children, in seconds.
    """

    campaign: str
    path: str
    hook: str
    calls: int
    total: float
    self_time: float


@dataclass(frozen=True)
class Profile:
    """Where a profiled run spent its time, and how much memory it used.

    Attributes
    ----------
    rows : tuple[HookTime, ...]
        One per campaign, node and hook called at least once.
    wall_time : float
        Seconds from the first episode start to the run end.
    steps : int
        Episodes × steps.
    bidder_time : float
        The sum of every row's self time, in seconds: each second inside a
        bidder counts once.
    simulator_time : float
        `wall_time` minus `bidder_time`: the market, the auction, feedback
        and logging, in seconds.
    memory_start : int
        Resident memory of the process at the run start, in bytes.
    memory_peak : int
        The highest resident memory sampled at the run start, each episode
        end and the run end, in bytes.
    seed : int
        The run's seed, as in `RunLog.seed`.

    Examples
    --------
    >>> import bidlab as bl
    >>> bidder = bl.Stack(bl.OracleValue(), bl.PidPacing())
    >>> config = bl.SimConfig(
    ...     campaigns=[bl.Campaign("pid", bidder, budget=10, base_value=5)],
    ...     episodes=2,
    ...     steps=50,
    ... )
    >>> profile = bl.run(config, profiler=True).profile
    >>> [(row.path, row.hook, row.calls) for row in profile.rows[:3]]
    [('', 'forward', 100), ('', 'on_episode_start', 2), ('', 'on_episode_end', 2)]
    >>> profile.steps, profile.bidder_time <= profile.wall_time
    (100, True)
    >>> profile.summary().splitlines()[0].split()
    ['campaign', 'node', 'hook', 'calls', 'total_s', 'self_s', 'mean_us', 'share']
    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = pathlib.Path(folder) / "profile.csv"
    ...     profile.save_csv(path)
    ...     path.read_text().splitlines()[0].split(",")[:5]
    ['seed', 'campaign', 'path', 'hook', 'calls']
    """

    rows: tuple[HookTime, ...]
    wall_time: float
    steps: int
    bidder_time: float
    simulator_time: float
    memory_start: int
    memory_peak: int
    seed: int

    def summary(self) -> str:
        """Return a table of the rows, slowest first, and a line of run totals.

        Each line shows a row's calls, its total and self time in seconds,
        its mean time per call in microseconds, and its total's share of
        `wall_time`. A `Stack`'s total includes its children's, so shares can
        add up to more than 100%. The last line shows the wall time, steps
        per second, the simulator's time and share, and the highest memory
        sample with its growth over the run start, in MiB.
        """
        lines = [
            (
                row.campaign,
                row.path or TOP_NODE,
                row.hook,
                str(row.calls),
                f"{row.total:.6f}",
                f"{row.self_time:.6f}",
                f"{row.total / row.calls * MICROSECONDS:.1f}",
                f"{row.total / self.wall_time:.1%}",
            )
            for row in self._slowest_first()
        ]
        widths = [max(map(len, column)) for column in zip(HEADER, *lines, strict=True)]
        table = [
            "  ".join(
                cell.ljust(width) if index < LABEL_COLUMNS else cell.rjust(width)
                for index, (cell, width) in enumerate(zip(cells, widths, strict=True))
            )
            for cells in (HEADER, *lines)
        ]
        peak = self.memory_peak / MEBIBYTE
        growth = (self.memory_peak - self.memory_start) / MEBIBYTE
        table.append(
            f"wall {self.wall_time:.3f} s, "
            f"{self.steps / self.wall_time:,.0f} steps/s, "
            f"simulator {self.simulator_time:.3f} s "
            f"({self.simulator_time / self.wall_time:.1%}), "
            f"memory peak {peak:.1f} MiB (+{growth:.1f} MiB over start)"
        )
        return "\n".join(table)

    def save_csv(self, path: str | Path) -> None:
        """Write one CSV line per row, in `summary` order, with the run totals.

        Every line repeats the seed, wall time, steps and memory, so the
        lines of many runs can share one file, as in
        `Results.save_profile_csv`. Floats keep every digit, so reading the
        file back gives each row's exact times.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to write, with the columns ``seed``, ``campaign``,
            ``path``, ``hook``, ``calls``, ``total_seconds``,
            ``self_seconds``, ``wall_seconds``, ``steps``, ``memory_start``
            and ``memory_peak``.
        """
        write_profile_csv(path, [self])

    def _slowest_first(self) -> list[HookTime]:
        return sorted(self.rows, key=lambda row: row.total, reverse=True)

    def _csv_lines(self) -> list[tuple[str | int | float, ...]]:
        return [
            (
                self.seed,
                row.campaign,
                row.path,
                row.hook,
                row.calls,
                row.total,
                row.self_time,
                self.wall_time,
                self.steps,
                self.memory_start,
                self.memory_peak,
            )
            for row in self._slowest_first()
        ]


def write_profile_csv(path: str | Path, profiles: Iterable[Profile]) -> None:
    """Write every profile's lines to one CSV file; `Profile.save_csv` calls this.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to write.
    profiles : Iterable[Profile]
        The profiles, in the order their lines go in the file.
    """
    with Path(path).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        for profile in profiles:
            # csv writes a float with str, which keeps the digits that read it back exactly.
            writer.writerows(profile._csv_lines())


def check_profiler(profiler: object) -> None:
    """Reject a ``profiler`` argument that is not a bool; `run` and `run_many` call this.

    Parameters
    ----------
    profiler : object
        The argument as given.

    Raises
    ------
    TypeError
        If ``profiler`` is not a bool.
    """
    if not isinstance(profiler, bool):
        raise TypeError(f"profiler must be a bool, got {type(profiler).__name__}")


@dataclass(slots=True)
class _Tally:
    calls: int = 0
    total: float = 0.0
    self_time: float = 0.0


class Recorder:
    """Times the bidder hooks of one run; the simulator makes one per profiled run."""

    def __init__(self) -> None:
        self._tallies: dict[tuple[str, str, str], _Tally] = {}
        # The time of each running hook's finished children; the innermost hook is last.
        self._children: list[float] = []
        self._start = 0.0
        self._memory_start = 0
        self._memory_peak = 0

    def wrap(self, campaign: str, nodes: Iterable[tuple[str, Bidder]]) -> None:
        """Time every hook of each node, setting a wrapper on the instance.

        Parameters
        ----------
        campaign : str
            The campaign's name.
        nodes : Iterable[tuple[str, Bidder]]
            Each node of the running copy, with its path.
        """
        for path, node in nodes:
            for hook in HOOKS:
                tally = self._tallies[(campaign, path, hook)] = _Tally()
                vars(node)[hook] = self._timed(getattr(node, hook), tally)

    def start(self) -> None:
        """Sample memory and start the run's clock, before the first episode."""
        self._memory_start = self._memory_peak = _resident_bytes()
        self._start = perf_counter()

    def sample_memory(self) -> None:
        """Keep the highest resident memory seen; the simulator calls this per episode."""
        self._memory_peak = max(self._memory_peak, _resident_bytes())

    def profile(self, seed: int, steps: int) -> Profile:
        """Stop the clock, sample memory once more, and return the run's profile.

        Parameters
        ----------
        seed : int
            The run's seed.
        steps : int
            Episodes × steps of the run.

        Returns
        -------
        Profile
            The rows of every hook called at least once, and the run totals.
        """
        wall_time = perf_counter() - self._start
        self.sample_memory()
        rows = tuple(
            HookTime(campaign, path, hook, tally.calls, tally.total, tally.self_time)
            for (campaign, path, hook), tally in self._tallies.items()
            if tally.calls
        )
        bidder_time = sum(row.self_time for row in rows)
        return Profile(
            rows=rows,
            wall_time=wall_time,
            steps=steps,
            bidder_time=bidder_time,
            simulator_time=wall_time - bidder_time,
            memory_start=self._memory_start,
            memory_peak=self._memory_peak,
            seed=seed,
        )

    def _timed(
        self, method: Callable[..., object], tally: _Tally
    ) -> Callable[..., object]:
        children = self._children

        # wraps sets __wrapped__, so call_forward reads the hook's own signature.
        @functools.wraps(method)
        def timed(*args: object, **kwargs: object) -> object:
            children.append(0.0)
            start = perf_counter()
            try:
                return method(*args, **kwargs)
            finally:
                elapsed = perf_counter() - start
                tally.calls += 1
                tally.total += elapsed
                tally.self_time += elapsed - children.pop()
                if children:
                    children[-1] += elapsed

        return timed


def _resident_bytes() -> int:
    # psutil has no peak on Linux or macOS, so the recorder samples the current value.
    return psutil.Process().memory_info().rss
