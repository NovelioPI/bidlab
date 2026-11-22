"""Does noise in the platform's value cost a truthful bidder surplus?

The `value` input is the platform's estimate of the value. With
`estimate_noise` above 0 the estimate is wrong, so a truthful bidder overbids on
some impressions and underbids on others. Metrics always use the true value.
"""

import dataclasses

import bidlab as bl

NOISE = 1.0

config = bl.SimConfig(
    campaigns=[
        bl.Campaign("truthful", bl.Linear(), budget=1e9, base_value=5),
        bl.Campaign("other", bl.Linear(), budget=1e9, base_value=5),
    ],
    auction=bl.SecondPrice(),
    episodes=5,
    steps=500,
)


def mean(results, bidder, column):
    values = [float(row[column]) for row in results.rows if row["campaign"] == bidder]
    return sum(values) / len(values)


if __name__ == "__main__":
    exact = bl.run_many(config, seeds=range(5))
    noisy = bl.run_many(
        dataclasses.replace(config, estimate_noise=NOISE), seeds=range(5)
    )
    print(noisy.summary())
    with_noise = mean(noisy, "truthful", "surplus")
    without = mean(exact, "truthful", "surplus")
    answer = "yes" if with_noise < without else "no"
    print(
        f"Answer: {answer}, surplus {with_noise:.1f} with noise {NOISE}"
        f" against {without:.1f} without"
    )
