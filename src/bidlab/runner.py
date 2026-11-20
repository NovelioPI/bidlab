"""Run one simulation per seed and collect the metrics in a table."""

import dataclasses
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import get_context

import cloudpickle
import numpy as np
import numpy.typing as npt

from bidlab.callbacks import Callback, checked_callbacks
from bidlab.config import SimConfig
from bidlab.logs import Logs
from bidlab.metrics import (
    CampaignEpisodeMetrics,
    campaign_metrics,
    platform_metrics,
    segment_metrics,
)
from bidlab.results import ResultRow, Results
from bidlab.simulator import run

BIDDER_COLUMNS = tuple(
    f.name
    for f in dataclasses.fields(CampaignEpisodeMetrics)
    if f.name not in ("episode", "campaign")
)
COLUMNS = ("seed", "campaign", *BIDDER_COLUMNS, "revenue", "welfare")
SEGMENT_METRIC_COLUMNS = ("spend", "wins", "clicks", "conversions", "surplus")
SEGMENT_COLUMNS = ("seed", "campaign", "segment", *SEGMENT_METRIC_COLUMNS)

RowBuilder = dict[str, str | int | float]


@dataclasses.dataclass(frozen=True)
class _SeedRows:
    rows: list[ResultRow]
    segment_rows: list[ResultRow]
    # Plain dicts, since a worker cannot pickle the run log's read-only mappings.
    logged: dict[str, dict[str, npt.NDArray[np.float64]]]


def run_many(
    config: SimConfig,
    seeds: Iterable[int],
    workers: int = 1,
    callbacks: Sequence[Callback] = (),
) -> Results:
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
    callbacks : Sequence[Callback], optional
        Read-only hooks; each gets `Callback.on_seed_end` in the main process
        after each seed, in seed order.

    Returns
    -------
    Results
        The bidder table, with the per-segment table in ``.segment_rows``.

    Raises
    ------
    ValueError
        If ``workers`` is below 1.
    TypeError
        If ``workers > 1`` and a bidder cannot be pickled with cloudpickle, or
        a callback is not a `Callback` instance.

    Examples
    --------
    >>> import bidlab as bl
    >>> config = bl.SimConfig(
    ...     campaigns=[
    ...         bl.Campaign("a", bl.Linear(), budget=10, event_value=5),
    ...         bl.Campaign("b", bl.PidPacing(), budget=10, event_value=5),
    ...     ],
    ...     episodes=2,
    ...     steps=50,
    ... )
    >>> results = bl.run_many(config, seeds=[0, 1])
    >>> [(row["seed"], row["campaign"]) for row in results.rows]
    [(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b')]
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    checked = checked_callbacks(callbacks)
    seed_list = list(seeds)
    if workers == 1:
        computed = (_seed_rows(config, seed) for seed in seed_list)
        per_seed = _collect(seed_list, computed, checked)
    else:
        payload = _pickle_config(config)
        # spawn behaves the same on every OS and avoids forking PyTorch threads.
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            computed = pool.map(_run_seed, repeat(payload), seed_list)
            per_seed = _collect(seed_list, computed, checked)

    return _results(per_seed)


def _collect(
    seeds: list[int], computed: Iterator[_SeedRows], callbacks: tuple[Callback, ...]
) -> list[_SeedRows]:
    per_seed: list[_SeedRows] = []
    for seed, seed_rows in zip(seeds, computed, strict=True):
        per_seed.append(seed_rows)
        for callback in callbacks:
            callback.on_seed_end(seed, _results([seed_rows]))
    return per_seed


def _results(per_seed: list[_SeedRows]) -> Results:
    rows = tuple(row for seed_rows in per_seed for row in seed_rows.rows)
    segment_rows = tuple(
        row for seed_rows in per_seed for row in seed_rows.segment_rows
    )
    return Results(
        columns=COLUMNS,
        rows=rows,
        segment_rows=Results(columns=SEGMENT_COLUMNS, rows=segment_rows),
        logged=_stacked_logs([seed_rows.logged for seed_rows in per_seed]),
    )


def _stacked_logs(
    per_seed: list[dict[str, dict[str, npt.NDArray[np.float64]]]],
) -> Logs:
    shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    for logs in per_seed:
        for name, arrays in logs.items():
            for campaign, array in arrays.items():
                shapes.setdefault(name, {})[campaign] = array.shape

    return {
        name: {
            campaign: np.stack(
                [
                    logs.get(name, {}).get(campaign, np.full(shape, np.nan))
                    for logs in per_seed
                ]
            )
            for campaign, shape in campaigns.items()
        }
        for name, campaigns in shapes.items()
    }


def _pickle_config(config: SimConfig) -> bytes:
    for campaign in config.campaigns:
        try:
            cloudpickle.dumps(campaign)
        except Exception as error:
            raise TypeError(
                f"campaign {campaign.name!r} cannot be sent to worker processes "
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
    episodes = campaign_metrics(log)
    rows: list[ResultRow] = []
    for campaign in config.campaigns:
        own = [m for m in episodes if m.campaign == campaign.name]
        row: RowBuilder = {"seed": seed, "campaign": campaign.name}
        for column in BIDDER_COLUMNS:
            row[column] = float(np.mean([getattr(m, column) for m in own]))

        row["revenue"] = revenue
        row["welfare"] = welfare
        rows.append(row)

    segments = segment_metrics(log)
    segment_rows: list[ResultRow] = []
    for campaign in config.campaigns:
        for name in config.market.segment_names:
            own = [
                m for m in segments if m.campaign == campaign.name and m.segment == name
            ]
            segment_row: RowBuilder = {
                "seed": seed,
                "campaign": campaign.name,
                "segment": name,
            }
            for column in SEGMENT_METRIC_COLUMNS:
                segment_row[column] = float(np.mean([getattr(m, column) for m in own]))
            segment_rows.append(segment_row)

    logs = {name: dict(arrays) for name, arrays in log.logged.items()}
    return _SeedRows(rows=rows, segment_rows=segment_rows, logged=logs)
