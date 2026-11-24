"""On a replayed first-price log, does PID pacing win more conversions than a fixed bid?

The log is generated here in place of a real one: two days of bid requests,
busier in the evening, from a first-price exchange. Such a log never shows the
market price. A win only says the price was between the floor and the own bid,
and a loss only says it was above the own bid. ReplayMarket fills each price
from a distribution cut to those bounds.

polars is not a bidlab dependency: install it with `pip install polars`.
"""

from datetime import UTC, datetime

import numpy as np
import polars as pl
from scipy import stats

import bidlab as bl

STEPS = 4_000
SEEDS = range(5)
# A lognormal the user would fit to their own prices; bidlab fits nothing.
PRICES = stats.lognorm(s=0.6, scale=0.15)


def production_log(rng):
    """Bid requests of two days, as a DSP would export them."""
    days = []
    for day, count in ((1, 5_200), (2, 4_700)):
        # More traffic from 18:00 to 23:00, as on many sites.
        evening = rng.random(count) < 0.4
        seconds = np.where(
            evening,
            rng.uniform(18 * 3600, 23 * 3600, count),
            rng.uniform(0, 86_400, count),
        )
        start = datetime(2026, 11, day, tzinfo=UTC)
        days.append(
            pl.DataFrame(
                {
                    "ts": [start.timestamp() + s for s in seconds],
                    "site": rng.choice(["news", "games"], count, p=[0.7, 0.3]),
                    "pcvr": rng.beta(2, 60, count),
                    "floor": rng.uniform(0.02, 0.08, count),
                    "bid": rng.uniform(0.05, 0.3, count),
                    "true_price": PRICES.rvs(count, random_state=rng),
                }
            )
        )

    log = pl.concat(days).with_columns(
        ts=pl.from_epoch(pl.col("ts"), time_unit="s"),
        won=pl.col("bid") > pl.max_horizontal("true_price", "floor"),
    )
    # The exchange never reports the price itself.
    return log.drop("true_price")


def replay_table(log):
    """Sample each day to STEPS rows, then add the time and price bounds."""
    by_day = log.with_columns(day=pl.col("ts").dt.date()).group_by("day")
    sampled = pl.concat(rows.sample(STEPS, seed=0) for _, rows in by_day)
    return sampled.sort("ts").with_columns(
        time=(pl.col("ts") - pl.col("ts").dt.truncate("1d")).dt.total_seconds(),
        low=pl.when("won").then("floor").otherwise("bid"),
        high=pl.when("won").then("bid").otherwise(float("inf")),
    )


def config(table):
    market = bl.markets.ReplayMarket(
        table,
        pcvr="pcvr",
        floor="floor",
        segment="site",
        time="time",
        price_low="low",
        price_high="high",
        price_distribution=PRICES,
    )
    campaigns = [
        bl.Campaign(
            name,
            bl.Stack(bl.OracleValue(), bidder),
            base_value=8.0,
            budget_fraction=0.3,
        )
        for name, bidder in (("fixed", bl.Linear()), ("pid", bl.PidPacing()))
    ]
    return bl.SimConfig(
        campaigns=campaigns,
        market=market,
        auction=bl.FirstPrice(),
        episodes=2,
        steps=STEPS,
        competing=False,
    )


def mean(results, campaign, column):
    values = [row[column] for row in results.rows if row["campaign"] == campaign]
    return sum(values) / len(values)


if __name__ == "__main__":
    table = replay_table(production_log(np.random.default_rng(7)))
    results = bl.run(config(table), seed=SEEDS)
    print(results.summary())
    fixed, pid = (mean(results, name, "conversions") for name in ("fixed", "pid"))
    answer = "yes" if pid > fixed else "no"
    print(f"Answer: {answer}, {pid:.1f} conversions per day against {fixed:.1f}")
