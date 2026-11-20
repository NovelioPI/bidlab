"""When the budget is tight, does PID pacing earn more surplus than truthful bidding?

A truthful bidder spends its small budget in the first hours of the day and
then sits out. PID pacing spreads the same budget over the whole day.
"""

import bidlab as bl

config = bl.SimConfig(
    campaigns=[
        bl.Campaign("truthful", bl.Linear(), budget=15, base_value=5),
        bl.Campaign("pid", bl.PidPacing(), budget=15, base_value=5),
        bl.Campaign("big", bl.Linear(), budget=50, base_value=5),
    ],
    auction=bl.SecondPrice(reserve=0.01),
    market=bl.markets.LogisticMarket(),
    episodes=10,
    steps=500,
)


def mean(results, bidder, column):
    values = [row[column] for row in results.rows if row["campaign"] == bidder]
    return sum(values) / len(values)


if __name__ == "__main__":
    results = bl.run_many(config, seeds=range(5), workers=4)
    print(results.summary())
    pid = mean(results, "pid", "surplus")
    truthful = mean(results, "truthful", "surplus")
    answer = "yes" if pid > truthful else "no"
    print(f"Answer: {answer}, surplus {pid:.1f} with PID against {truthful:.1f}")
