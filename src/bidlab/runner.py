import dataclasses
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import get_context

import cloudpickle
import numpy as np

from bidlab.config import SimConfig
from bidlab.metrics import BidderEpisodeMetrics, bidder_metrics, platform_metrics
from bidlab.results import ResultRow, Results
from bidlab.simulator import run

BIDDER_COLUMNS = tuple(
    f.name
    for f in dataclasses.fields(BidderEpisodeMetrics)
    if f.name not in ("episode", "bidder")
)
COLUMNS = ("seed", "bidder", *BIDDER_COLUMNS, "revenue", "welfare")

RowBuilder = dict[str, str | int | float]


def run_many(config: SimConfig, seeds: Iterable[int], workers: int = 1) -> Results:
    """Run each seed; rows are ordered by seed, then bidder."""
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

    rows = tuple(row for seed_rows in per_seed for row in seed_rows)
    return Results(columns=COLUMNS, rows=rows)


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


def _run_seed(payload: bytes, seed: int) -> list[ResultRow]:
    return _seed_rows(cloudpickle.loads(payload), seed)


def _seed_rows(config: SimConfig, seed: int) -> list[ResultRow]:
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

    return rows
