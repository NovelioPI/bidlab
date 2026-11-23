"""Does noise in the value estimate cost a truthful bidder surplus?

`OracleValue(noise=...)` adds normal noise to the logit of each true rate, so
the value it passes on is wrong, and a truthful bidder after it overbids on
some impressions and underbids on others. Metrics always use the true value.
"""

import bidlab as bl

NOISE = 1.0


def config(noise):
    def truthful():
        return bl.Stack(bl.OracleValue(noise=noise), bl.Linear())

    return bl.SimConfig(
        campaigns=[
            bl.Campaign("truthful", truthful(), budget=1e9, base_value=5),
            bl.Campaign("other", truthful(), budget=1e9, base_value=5),
        ],
        auction=bl.SecondPrice(),
        episodes=5,
        steps=500,
    )


def mean(results, bidder, column):
    values = [float(row[column]) for row in results.rows if row["campaign"] == bidder]
    return sum(values) / len(values)


if __name__ == "__main__":
    exact = bl.run(config(0.0), seed=range(5))
    noisy = bl.run(config(NOISE), seed=range(5))
    print(noisy.summary())
    with_noise = mean(noisy, "truthful", "surplus")
    without = mean(exact, "truthful", "surplus")
    answer = "yes" if with_noise < without else "no"
    print(
        f"Answer: {answer}, surplus {with_noise:.1f} with noise {NOISE}"
        f" against {without:.1f} without"
    )
