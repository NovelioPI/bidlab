"""Does bidding target/value of the value meet a CPA target that truthful bidding misses?

In a second-price auction a bidder never pays more than its bid. A bid of
target/value times the value costs at most the target per expected conversion.
The `cpa_score` metric scores an episode by its value, cut sharply when the
CPA is above target.
"""

import bidlab as bl

CPA_TARGET = 1.5
VALUE = 5
SAFETY = 0.8


def campaign(name, bidder):
    return bl.Campaign(
        name,
        bl.Stack(bl.OracleValue(), bidder),
        budget=50,
        base_value=VALUE,
        cpa_target=CPA_TARGET,
    )


config = bl.SimConfig(
    campaigns=[
        campaign("truthful", bl.Linear()),
        campaign("at_target", bl.Linear(multiplier=CPA_TARGET / VALUE)),
        campaign("strict", bl.Linear(multiplier=CPA_TARGET * SAFETY / VALUE)),
        bl.Campaign(
            "big", bl.Stack(bl.OracleValue(), bl.Linear()), budget=50, base_value=VALUE
        ),
    ],
    auction=bl.SecondPrice(reserve=0.01),
    episodes=10,
    steps=500,
)


def mean(results, bidder, column):
    values = [float(row[column]) for row in results.rows if row["campaign"] == bidder]
    return sum(values) / len(values)


if __name__ == "__main__":
    results = bl.run(config, seed=range(5))
    print(results.summary())
    at_target = mean(results, "at_target", "cpa")
    truthful = mean(results, "truthful", "cpa")
    answer = "yes" if at_target <= CPA_TARGET < truthful else "no"
    print(
        f"Answer: {answer}, CPA {at_target:.2f} at target/value against"
        f" {truthful:.2f} truthful, target {CPA_TARGET}"
    )
