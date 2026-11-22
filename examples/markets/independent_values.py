"""In a first-price auction with uniform values, does shading to (n-1)/n beat truthful bidding?

A truthful bidder in first price pays its full value when it wins, so it earns
about zero surplus. With n bidders and values uniform on [0, 1], the textbook
equilibrium bid is (n-1)/n of the value.
"""

from scipy import stats

import bidlab as bl

BIDDERS = 3
BUDGET = 1e9

config = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            name,
            bl.Linear(multiplier=multiplier),
            budget=BUDGET,
            base_value=1,
        )
        for name, multiplier in (
            ("truthful", 1.0),
            ("equilibrium", (BIDDERS - 1) / BIDDERS),
            ("low", 0.4),
        )
    ],
    market=bl.markets.IndependentValues(stats.uniform(0, 1)),
    auction=bl.FirstPrice(),
    episodes=5,
    steps=2000,
)


def mean(results, bidder, column):
    values = [row[column] for row in results.rows if row["campaign"] == bidder]
    return sum(values) / len(values)


if __name__ == "__main__":
    results = bl.run_many(config, seeds=range(5))
    print(results.summary())
    shaded = mean(results, "equilibrium", "surplus")
    truthful = mean(results, "truthful", "surplus")
    answer = "yes" if shaded > truthful else "no"
    print(f"Answer: {answer}, surplus {shaded:.1f} when shaded against {truthful:.1f}")
