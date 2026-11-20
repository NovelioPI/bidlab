"""When the budget is tight, does PID pacing earn more surplus than truthful bidding?

A truthful bidder spends its small budget in the first hours of the day and
then sits out. PID pacing spreads the same budget over the whole day.
"""

import bidlab as bl

config = bl.SimConfig(
    bidders=[
        bl.BidderSpec("truthful", bl.Truthful(), bl.Campaign(budget=15, value=5)),
        bl.BidderSpec("pid", bl.PidPacing(), bl.Campaign(budget=15, value=5)),
        bl.BidderSpec("big", bl.Truthful(), bl.Campaign(budget=50, value=5)),
    ],
    auction=bl.SecondPrice(reserve=0.01),
    market=bl.markets.LogisticContext(),
    episodes=10,
    steps=500,
)


def mean(results, bidder, column):
    values = [row[column] for row in results.rows if row["bidder"] == bidder]
    return sum(values) / len(values)


if __name__ == "__main__":
    results = bl.run_many(config, seeds=range(5), workers=4)
    print(results.summary())
    pid = mean(results, "pid", "surplus")
    truthful = mean(results, "truthful", "surplus")
    answer = "yes" if pid > truthful else "no"
    print(f"Answer: {answer}, surplus {pid:.1f} with PID against {truthful:.1f}")
