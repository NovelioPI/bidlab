"""Is UcbMultiplier's update slower per call than PidPacing's?

`profiler=True` times every hook of every bidder in each campaign's `Stack`.
Each row of `Profile.rows` names the campaign, the node's path and the hook,
so the two `update` rows compare directly. Timings change from run to run, so
the script compares the mean per call over three seeds.
"""

import bidlab as bl

config = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "pid", bl.Stack(bl.OracleValue(), bl.PidPacing()), budget=15, base_value=5
        ),
        bl.Campaign(
            "ucb",
            bl.Stack(bl.OracleValue(), bl.UcbMultiplier()),
            budget=15,
            base_value=5,
        ),
    ],
    market=bl.markets.LogisticMarket(price_median=0.2),
    episodes=10,
    steps=1000,
)


def mean_update(profile, campaign):
    """Microseconds per call of the campaign's bid-control update."""
    for row in profile.rows:
        if (row.campaign, row.path, row.hook) == (campaign, "bidders.1", "update"):
            return row.total / row.calls * 1e6

    raise ValueError(f"{campaign} has no update row")


if __name__ == "__main__":
    results = bl.run_many(config, seeds=range(3), profiler=True)
    # profiler=True always fills it; the assert tells type checkers so.
    assert results.profile is not None
    print(results.profile[0].summary())
    slower = []
    for seed, profile in enumerate(results.profile):
        pid, ucb = mean_update(profile, "pid"), mean_update(profile, "ucb")
        print(f"seed {seed}: update {pid:.2f} µs per call for pid, {ucb:.2f} for ucb")
        slower.append(ucb > pid)

    answer = "yes" if all(slower) else "no"
    print(f"Answer: {answer}, slower in {sum(slower)} of {len(slower)} seeds")
