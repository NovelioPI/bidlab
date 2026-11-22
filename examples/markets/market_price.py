"""Does a lone campaign pay more per conversion when the market price follows its value?

The market price is the highest bid from outside the config. With
rival_similarity 0, it follows the features but not the campaign's value, so
the impressions the campaign wants most are no dearer than others. With 0.8,
outside advertisers want the same impressions, so those cost more.
"""

import bidlab as bl

SEEDS = range(5)


def config(similarity):
    return bl.SimConfig(
        campaigns=[
            bl.Campaign(
                "pid",
                bl.Stack(bl.OracleValue(), bl.PidPacing()),
                budget=30,
                base_value=5,
            )
        ],
        market=bl.markets.LogisticMarket(
            price_median=0.2, rival_similarity=similarity, floor_share=(2, 5)
        ),
        episodes=3,
        steps=500,
    )


def mean(results, column):
    values = [row[column] for row in results.rows]
    return sum(values) / len(values)


if __name__ == "__main__":
    independent = bl.run_many(config(0.0), seeds=SEEDS)
    rival = bl.run_many(config(0.8), seeds=SEEDS)
    print(rival.summary())
    for name, results in (("similarity 0.0", independent), ("similarity 0.8", rival)):
        print(
            f"{name}: cpa {mean(results, 'cpa'):.2f}, "
            f"conversions {mean(results, 'conversions'):.1f}, "
            f"surplus {mean(results, 'surplus'):.1f}"
        )
    dear, cheap = mean(rival, "cpa"), mean(independent, "cpa")
    answer = "yes" if dear > cheap else "no"
    print(f"Answer: {answer}, cost per conversion {dear:.2f} against {cheap:.2f}")
