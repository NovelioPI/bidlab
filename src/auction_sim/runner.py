import dataclasses
from collections.abc import Iterable

import numpy as np

from auction_sim.config import SimConfig
from auction_sim.metrics import BidderEpisodeMetrics, bidder_metrics, platform_metrics
from auction_sim.simulator import run

ResultRow = dict[str, str | int | float]

BIDDER_COLUMNS = tuple(
    f.name
    for f in dataclasses.fields(BidderEpisodeMetrics)
    if f.name not in ("episode", "bidder")
)
COLUMNS = ("seed", "bidder", *BIDDER_COLUMNS, "revenue", "welfare")


def run_many(config: SimConfig, seeds: Iterable[int]) -> list[ResultRow]:
    """One row per seed and bidder, each metric averaged over episodes."""
    rows: list[ResultRow] = []
    for seed in seeds:
        log = run(dataclasses.replace(config, seed=seed))
        platform = platform_metrics(log)
        revenue = float(np.mean([p.revenue for p in platform]))
        welfare = float(np.mean([p.welfare for p in platform]))
        episodes = bidder_metrics(log)
        for spec in config.bidders:
            own = [m for m in episodes if m.bidder == spec.name]
            row: ResultRow = {"seed": seed, "bidder": spec.name}
            for column in BIDDER_COLUMNS:
                row[column] = float(np.mean([getattr(m, column) for m in own]))
            row["revenue"] = revenue
            row["welfare"] = welfare
            rows.append(row)
    return rows
