"""First price with independent uniform values: truthful bidding earns about zero surplus."""

from scipy import stats

import bidlab as bl

BIDDERS = 3
BUDGET = 1e9

config = bl.SimConfig(
    bidders=[
        bl.BidderSpec(
            name,
            bl.Truthful(multiplier=multiplier),
            bl.Campaign(budget=BUDGET, value=1),
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

if __name__ == "__main__":
    results = bl.run_many(config, seeds=range(5))
    print(results.summary())
