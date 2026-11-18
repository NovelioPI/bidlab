"""Run one simulation per seed and collect the metrics in a table."""

import dataclasses
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import get_context

import cloudpickle
import numpy as np
import numpy.typing as npt

from bidlab.config import SimConfig
from bidlab.logs import Logs
from bidlab.metrics import (
    BidderEpisodeMetrics,
    bidder_metrics,
    platform_metrics,
    segment_metrics,
)
from bidlab.results import ResultRow, Results
from bidlab.simulator import run

BIDDER_COLUMNS = tuple(
    f.name
    for f in dataclasses.fields(BidderEpisodeMetrics)
    if f.name not in ("episode", "bidder")
)
COLUMNS = ("seed", "bidder", *BIDDER_COLUMNS, "revenue", "welfare")
SEGMENT_METRIC_COLUMNS = ("spend", "wins", "conversions", "surplus")
SEGMENT_COLUMNS = ("seed", "bidder", "segment", *SEGMENT_METRIC_COLUMNS)

RowBuilder = dict[str, str | int | float]


@dataclasses.dataclass(frozen=True)
class _SeedRows:
    main: list[ResultRow]
    segments: list[ResultRow]
    # Plain dicts, since a worker cannot pickle the run log's read-only mappings.
    logs: dict[str, dict[str, npt.NDArray[np.float64]]]


def run_many(config: SimConfig, seeds: Iterable[int], workers: int = 1) -> Results:
    """Run each seed; rows are ordered by seed, then bidder.

    Each row holds one bidder's metrics for one seed, averaged over the
    episodes. ``workers > 1`` runs seeds in parallel processes and gives the
    same results as ``workers=1``.

    Parameters
    ----------
    config : SimConfig
        The simulation; its ``seed`` is replaced by each seed in turn.
    seeds : Iterable[int]
        One run per seed.
    workers : int, optional
        Number of processes.

    Returns
    -------
    Results
        The bidder table, with the per-segment table in ``.segments``.

    Raises
    ------
    ValueError
        If ``workers`` is below 1.
    TypeError
        If ``workers > 1`` and a bidder cannot be pickled with cloudpickle.

    Examples
    --------
    >>> import bidlab as bl
    >>> config = bl.SimConfig(
    ...     bidders=[
    ...         bl.BidderSpec("a", bl.Truthful(), bl.Campaign(budget=10, value=5)),
    ...         bl.BidderSpec("b", bl.PidPacing(), bl.Campaign(budget=10, value=5)),
    ...     ],
    ...     episodes=2,
    ...     steps=50,
    ... )
    >>> results = bl.run_many(config, seeds=[0, 1])
    >>> [(row["seed"], row["bidder"]) for row in results.rows]
    [(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b')]
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    seed_list = list(seeds)
    if workers == 1:
        per_seed = [_seed_rows(config, seed) for seed in seed_list]
    else:
        payload = _pickle_config(config)
        # spawn behaves the same on every OS and avoids forking PyTorch threads.
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            per_seed = list(pool.map(_run_seed, repeat(payload), seed_list))

    return _results(per_seed)


def _results(per_seed: list[_SeedRows]) -> Results:
    rows = tuple(row for seed_rows in per_seed for row in seed_rows.main)
    segment_rows = tuple(row for seed_rows in per_seed for row in seed_rows.segments)
    return Results(
        columns=COLUMNS,
        rows=rows,
        segments=Results(columns=SEGMENT_COLUMNS, rows=segment_rows),
        logs=_stacked_logs([seed_rows.logs for seed_rows in per_seed]),
    )


def _stacked_logs(
    per_seed: list[dict[str, dict[str, npt.NDArray[np.float64]]]],
) -> Logs:
    shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    for logs in per_seed:
        for name, arrays in logs.items():
            for spec, array in arrays.items():
                shapes.setdefault(name, {})[spec] = array.shape

    return {
        name: {
            spec: np.stack(
                [
                    logs.get(name, {}).get(spec, np.full(shape, np.nan))
                    for logs in per_seed
                ]
            )
            for spec, shape in specs.items()
        }
        for name, specs in shapes.items()
    }


def _pickle_config(config: SimConfig) -> bytes:
    for spec in config.bidders:
        try:
            cloudpickle.dumps(spec)
        except Exception as error:
            raise TypeError(
                f"bidder {spec.name!r} cannot be sent to worker processes "
                f"({error}); use workers=1"
            ) from error

    return cloudpickle.dumps(config)


def _run_seed(payload: bytes, seed: int) -> _SeedRows:
    return _seed_rows(cloudpickle.loads(payload), seed)


def _seed_rows(config: SimConfig, seed: int) -> _SeedRows:
    log = run(dataclasses.replace(config, seed=seed))
    platform = platform_metrics(log)
    revenue = float(np.mean([p.revenue for p in platform]))
    welfare = float(np.mean([p.welfare for p in platform]))
    episodes = bidder_metrics(log)
    rows: list[ResultRow] = []
    for spec in config.bidders:
        own = [m for m in episodes if m.bidder == spec.name]
        row: RowBuilder = {"seed": seed, "bidder": spec.name}
        for column in BIDDER_COLUMNS:
            row[column] = float(np.mean([getattr(m, column) for m in own]))

        row["revenue"] = revenue
        row["welfare"] = welfare
        rows.append(row)

    segments = segment_metrics(log)
    segment_rows: list[ResultRow] = []
    for spec in config.bidders:
        for name in config.market.segment_names:
            own = [m for m in segments if m.bidder == spec.name and m.segment == name]
            segment_row: RowBuilder = {
                "seed": seed,
                "bidder": spec.name,
                "segment": name,
            }
            for column in SEGMENT_METRIC_COLUMNS:
                segment_row[column] = float(np.mean([getattr(m, column) for m in own]))
            segment_rows.append(segment_row)

    logs = {name: dict(arrays) for name, arrays in log.logs.items()}
    return _SeedRows(main=rows, segments=segment_rows, logs=logs)
