"""Does a reserve price of 0.1 raise the platform's revenue?

With two truthful bidders, the second-highest bid is often low. A reserve makes
the winner pay at least the reserve, and drops auctions where every bid is
below it.
"""

import dataclasses
import tempfile
from pathlib import Path

import bidlab as bl

RESERVE = 0.1

config = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            name, bl.Stack(bl.OracleValue(), bl.Linear()), budget=1e9, base_value=5
        )
        for name in ("first", "second")
    ],
    auction=bl.SecondPrice(reserve=0.0),
    episodes=3,
    steps=500,
)


def platform_mean(results, column):
    """Mean over seeds; every bidder row of a seed holds the same platform value."""
    values = [float(row[column]) for row in results.rows if row["campaign"] == "first"]
    return sum(values) / len(values)


if __name__ == "__main__":
    without = bl.run(config, seed=range(5))
    reserved = dataclasses.replace(config, auction=bl.SecondPrice(reserve=RESERVE))
    with_reserve = bl.run(reserved, seed=range(5))
    print(with_reserve.summary())
    print(f"columns: {with_reserve.columns}")
    print(
        f"same seeds give the same results: {without.equals(bl.run(config, seed=range(5)))}"
    )

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "reserve.csv"
        with_reserve.save_csv(path)
        print(f"CSV lines: {len(path.read_text().splitlines())}")

    try:
        frame = with_reserve.to_pandas()
    except ImportError:
        print("to_pandas skipped: pandas is not installed")
    else:
        print(frame.groupby("campaign")[["revenue", "welfare"]].mean())

    raised = platform_mean(with_reserve, "revenue")
    base = platform_mean(without, "revenue")
    answer = "yes" if raised > base else "no"
    print(
        f"Answer: {answer}, revenue {raised:.1f} with reserve {RESERVE} against {base:.1f}"
    )
