"""The `Results` table that `run_many` returns."""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import stats

if TYPE_CHECKING:
    import pandas

ResultRow = Mapping[str, str | int | float]

SUMMARY_METRICS = ("spend", "conversions", "surplus", "budget_used", "regret", "reward")
SEGMENT_SUMMARY_METRICS = ("spend", "wins", "conversions", "surplus")
SUMMARY_COLUMN_WIDTH = 22
CONFIDENCE = 0.95


@dataclass(frozen=True)
class Results:
    """Rows from `run_many`: one per seed and bidder, averaged over episodes.

    Compare two tables with `equals`, not ``==``: some metrics are NaN by
    design.

    Attributes
    ----------
    columns : tuple[str, ...]
        Column names, in order.
    rows : tuple[Mapping[str, str | int | float], ...]
        Read-only rows, ordered by seed, then bidder.
    segments : Results or None
        One row per seed, bidder and market segment; None inside that table.

    Examples
    --------
    >>> import bidlab as bl
    >>> config = bl.SimConfig(
    ...     bidders=[bl.BidderSpec("t", bl.Truthful, budget=10, value_per_conversion=5)],
    ...     episodes=2,
    ...     steps=50,
    ... )
    >>> results = bl.run_many(config, seeds=range(3))
    >>> len(results), results.columns[:3]
    (3, ('seed', 'bidder', 'spend'))
    >>> results.equals(bl.run_many(config, seeds=range(3)))
    True
    """

    columns: tuple[str, ...]
    rows: tuple[ResultRow, ...]
    segments: "Results | None" = None

    def __post_init__(self) -> None:
        read_only = tuple(MappingProxyType(dict(row)) for row in self.rows)
        object.__setattr__(self, "rows", read_only)

    def __len__(self) -> int:
        return len(self.rows)

    def equals(self, other: "Results") -> bool:
        """Exact equality of this table's columns and rows; NaN equals NaN."""
        if self.columns != other.columns or len(self.rows) != len(other.rows):
            return False

        return all(
            _same_value(mine[column], theirs[column])
            for mine, theirs in zip(self.rows, other.rows, strict=True)
            for column in self.columns
        )

    def summary(self) -> str:
        """Mean ± 95% Student-t half-width across seeds, per bidder (and segment)."""
        keys, metrics = ("bidder",), SUMMARY_METRICS
        if "segment" in self.columns:
            keys, metrics = ("bidder", "segment"), SEGMENT_SUMMARY_METRICS
        groups = list(
            dict.fromkeys(tuple(str(row[k]) for k in keys) for row in self.rows)
        )
        widths = [
            max(len(key), *(len(group[k]) for group in groups))
            for k, key in enumerate(keys)
        ]
        lines = [_summary_line(keys, widths, metrics)]
        for group in groups:
            own = [
                row for row in self.rows if tuple(str(row[k]) for k in keys) == group
            ]
            cells = [
                _mean_and_interval(np.array([float(row[metric]) for row in own]))
                for metric in metrics
            ]
            lines.append(_summary_line(group, widths, cells))

        return "\n".join(lines)

    def save_csv(self, path: str | Path) -> None:
        """Write the rows to a CSV file with a header line."""
        with Path(path).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)

    def to_pandas(self) -> "pandas.DataFrame":
        """Return the rows as a DataFrame; needs ``pip install bidlab[pandas]``.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas
        except ImportError:
            raise ImportError(
                "to_pandas() needs pandas: pip install bidlab[pandas]"
            ) from None

        records = [dict(row) for row in self.rows]
        return pandas.DataFrame(records, columns=list(self.columns))


def _summary_line(
    labels: Sequence[str], widths: list[int], cells: Sequence[str]
) -> str:
    label = " ".join(
        f"{text:<{width}}" for text, width in zip(labels, widths, strict=True)
    )
    return label + "".join(f"{cell:>{SUMMARY_COLUMN_WIDTH}}" for cell in cells)


def _mean_and_interval(values: npt.NDArray[np.float64]) -> str:
    mean = values.mean()
    if values.size < 2:
        return f"{mean:.3f} ± n/a"

    standard_error = values.std(ddof=1) / np.sqrt(values.size)
    quantile = stats.t.ppf((1 + CONFIDENCE) / 2, values.size - 1)
    return f"{mean:.3f} ± {quantile * standard_error:.3f}"


def _same_value(mine: str | float, theirs: str | float) -> bool:
    both_nan = (
        isinstance(mine, float)
        and isinstance(theirs, float)
        and np.isnan(mine)
        and np.isnan(theirs)
    )
    return both_nan or mine == theirs
