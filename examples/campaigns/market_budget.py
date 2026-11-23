"""With a budget at half the market's cost, does truthful bidding run out early while PID pacing lasts the day?

`budget=None` sizes each budget from the market: the mean cost per episode of
every request worth its cost, which is what a truthful bidder spends with no
budget limit. `budget_fraction=0.5` keeps half of it, so the budget is equally
tight in any market. A bidder without pacing spends it as fast as it wins; a
pacer spreads it over the day.
"""

import numpy as np

import bidlab as bl

HALF = 0.5


def campaign(name, policy):
    return bl.Campaign(
        name,
        bl.Stack(bl.OracleValue(), policy),
        base_value=20,
        budget_fraction=HALF,
    )


config = bl.SimConfig(
    campaigns=[campaign("truthful", bl.Linear()), campaign("pid", bl.PidPacing())],
    market=bl.markets.LogisticMarket(price_median=0.2, floor_share=(2, 5)),
    competing=False,
    episodes=3,
    steps=500,
)


def day_share_before_spent(log, index):
    """Mean share of each episode's steps before 95% of the budget is spent."""
    spent = np.cumsum(log.price[:, :, index], axis=1)
    exhausted = spent >= 0.95 * log.budgets[index]
    steps = log.config.steps
    first = np.where(exhausted.any(axis=1), exhausted.argmax(axis=1), steps)
    return float(np.mean(first / steps))


if __name__ == "__main__":
    log = bl.run(config)
    shares = {}
    for index, c in enumerate(config.campaigns):
        shares[c.name] = day_share_before_spent(log, index)
        print(
            f"{c.name}: budget {log.budgets[index]:.2f}, "
            f"spent {log.price[:, :, index].sum(axis=1).mean():.2f} per episode, "
            f"budget lasts {shares[c.name]:.0%} of the day"
        )

    early = shares["truthful"] < 0.5 < shares["pid"]
    answer = "yes" if early else "no"
    print(
        f"Answer: {answer}, truthful lasts {shares['truthful']:.0%} of the day "
        f"and PID {shares['pid']:.0%}"
    )
