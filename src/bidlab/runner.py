"""Run one simulation per seed of a seed list and collect the metrics in a table."""

import dataclasses
import math
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import get_context

import cloudpickle
import numpy as np
import numpy.typing as npt

from bidlab.callbacks import Callback, checked_callbacks, with_progress_bar
from bidlab.config import SimConfig
from bidlab.logs import Logs
from bidlab.metrics import (
    CampaignEpisodeMetrics,
    CampaignSegmentMetrics,
    campaign_metrics,
    platform_metrics,
    segment_metrics,
)
from bidlab.profiling import Profile, check_profiler
from bidlab.results import ResultRow, Results
from bidlab.simulator import run

BIDDER_COLUMNS = tuple(
    f.name
    for f in dataclasses.fields(CampaignEpisodeMetrics)
    if f.name not in ("episode", "campaign")
)
COLUMNS = ("seed", "campaign", *BIDDER_COLUMNS, "revenue", "welfare")
SEGMENT_METRIC_COLUMNS = tuple(
    f.name
    for f in dataclasses.fields(CampaignSegmentMetrics)
    if f.name not in ("episode", "campaign", "segment")
)
SEGMENT_COLUMNS = ("seed", "campaign", "segment", *SEGMENT_METRIC_COLUMNS)

RowBuilder = dict[str, str | int | float]


@dataclasses.dataclass(frozen=True)
class _SeedRows:
    rows: list[ResultRow]
    segment_rows: list[ResultRow]
    # Plain dicts, since a worker cannot pickle the run log's read-only mappings.
    logged: dict[str, dict[str, npt.NDArray[np.float64]]]
    profile: Profile | None


def run_seeds(
    config: SimConfig,
    seeds: object,
    workers: int,
    callbacks: Sequence[Callback],
    progress_bar: bool,
    profiler: bool,
) -> Results:
    """Run ``config`` once per seed for `run`, which documents every argument.

    Raises
    ------
    TypeError
        If ``seeds`` is not an iterable of ints, or as in `run`.
    ValueError
        If ``seeds`` is empty.
    """
    seed_list = _seed_list(seeds)
    check_profiler(profiler)
    checked = checked_callbacks(callbacks)
    if progress_bar:
        checked = with_progress_bar(checked)

    for callback in checked:
        callback.on_run_many_start(config, tuple(seed_list))

    if workers == 1:
        computed = (_seed_rows(config, seed, profiler) for seed in seed_list)
        per_seed = _collect(seed_list, computed, checked)
    else:
        payload = _pickle_config(config)
        # spawn behaves the same on every OS and avoids forking PyTorch threads.
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            computed = pool.map(_run_seed, repeat(payload), seed_list, repeat(profiler))
            per_seed = _collect(seed_list, computed, checked)

    results = _results(per_seed)
    for callback in checked:
        callback.on_run_many_end(results)

    return results


def _seed_list(seeds: object) -> list[int]:
    # A str is iterable, but its characters are no seeds.
    if isinstance(seeds, bool | str | bytes) or not isinstance(seeds, Iterable):
        raise TypeError(
            f"seed must be an int or an iterable of ints, got {type(seeds).__name__}"
        )

    seed_list: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
            raise TypeError(
                f"a seed list must hold only ints, got {type(seed).__name__}"
            )

        seed_list.append(int(seed))

    if not seed_list:
        raise ValueError("the seed list is empty; pass at least one seed")

    return seed_list


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
        profile=_profiles(per_seed),
    )


def _profiles(per_seed: list[_SeedRows]) -> tuple[Profile, ...] | None:
    # Every seed runs with the same profiler setting, so all or none have one.
    profiles = tuple(
        seed_rows.profile for seed_rows in per_seed if seed_rows.profile is not None
    )
    return profiles or None


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


def _run_seed(payload: bytes, seed: int, profiler: bool) -> _SeedRows:
    return _seed_rows(cloudpickle.loads(payload), seed, profiler)


def _seed_rows(config: SimConfig, seed: int, profiler: bool) -> _SeedRows:
    log = run(config, seed, progress_bar=False, profiler=profiler)
    platform = platform_metrics(log)
    revenue = float(np.mean([p.revenue for p in platform]))
    welfare = float(np.mean([p.welfare for p in platform]))
    episodes = campaign_metrics(log)
    rows: list[ResultRow] = []
    for index, campaign in enumerate(config.campaigns):
        own = [m for m in episodes if m.campaign == campaign.name]
        row: RowBuilder = {"seed": seed, "campaign": campaign.name}
        for column in BIDDER_COLUMNS:
            row[column] = float(np.mean([getattr(m, column) for m in own]))

        # The budget is the same every episode, and a mean of equal floats can move it.
        row["budget"] = float(log.budgets[index])
        # An episode whose budget never ran out would hide the others in a mean.
        ran_out = [m.budget_out for m in own if not math.isnan(m.budget_out)]
        row["budget_out"] = float(np.mean(ran_out)) if ran_out else math.nan

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
    return _SeedRows(
        rows=rows, segment_rows=segment_rows, logged=logs, profile=log.profile
    )
