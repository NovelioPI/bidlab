"""On one shared market, does PID pacing earn more surplus than truthful bidding?

With `competing=False`, each campaign bids alone against the market price, on
one shared draw: the same impressions, rates, prices and floors. The two
campaigns differ only in their bid policy, so the gap in surplus is the
policy's. Each seed gives both policies the same traffic, so the comparison is
paired.
"""

import numpy as np

import bidlab as bl

SEEDS = range(10)


def campaign(name, policy):
    return bl.Campaign(
        name, bl.Stack(bl.OracleValue(), policy), budget=3, base_value=20
    )


config = bl.SimConfig(
    campaigns=[campaign("truthful", bl.Linear()), campaign("pid", bl.PidPacing())],
    market=bl.markets.LogisticMarket(price_median=0.2),
    competing=False,
    episodes=5,
    steps=500,
)

if __name__ == "__main__":
    results = bl.run_many(config, seeds=SEEDS)
    print(results.summary())
    surplus = {
        name: np.array(
            [row["surplus"] for row in results.rows if row["campaign"] == name]
        )
        for name in ("truthful", "pid")
    }
    gain = surplus["pid"] - surplus["truthful"]
    print(f"pid minus truthful surplus, per seed: {gain.round(1)}")
    better = int((gain > 0).sum())
    answer = "yes" if better == len(gain) else "no"
    print(f"Answer: {answer}, PID earned more in {better} of {len(gain)} seeds")
