"""Does a bandit rewarded for conversions pick a higher multiplier than one rewarded for surplus?

`UcbMultiplier` picks one multiplier per episode and learns from the reward it
sums in `update`: the surplus, by default. A subclass changes the reward by
overriding `update`. A conversion reward ignores the price, so a higher
multiplier always looks better.
"""

import dataclasses

import bidlab as bl

LAST_EPISODES = 10


class ConversionBandit(bl.UcbMultiplier):
    def update(self, result):
        self.arm_reward += float(result.converted)


class ValueBandit(bl.UcbMultiplier):
    def update(self, result):
        self.arm_reward += result.realized_value


class CheapConversionBandit(bl.UcbMultiplier):
    """A conversion counts 1; each unit of price costs `cost_weight`."""

    def __init__(self, cost_weight, **settings):
        super().__init__(**settings)
        self.cost_weight = cost_weight

    def update(self, result):
        self.arm_reward += float(result.converted) - self.cost_weight * result.price


def bandit(bidder):
    return bl.Campaign(
        "bandit", bl.Stack(bl.OracleValue(), bidder), budget=1e9, base_value=5
    )


BANDITS = {
    "surplus": bandit(bl.UcbMultiplier(multiplier_step=0.6)),
    "conversions": bandit(ConversionBandit(multiplier_step=0.6)),
    "realized_value": bandit(ValueBandit(multiplier_step=0.6)),
    "cheap_conversions": bandit(
        CheapConversionBandit(cost_weight=0.5, multiplier_step=0.6)
    ),
}

base = bl.SimConfig(
    campaigns=[
        BANDITS["surplus"],
        bl.Campaign(
            "other", bl.Stack(bl.OracleValue(), bl.Linear()), budget=1e9, base_value=5
        ),
    ],
    episodes=30,
    steps=300,
)


if __name__ == "__main__":
    arms = {}
    for name, campaign in BANDITS.items():
        config = dataclasses.replace(base, campaigns=[campaign, base.campaigns[1]])
        results = bl.run(config, seed=range(5))
        arms[name] = results.logged["bidders.1.multiplier"]["bandit"][
            :, -LAST_EPISODES:
        ].mean(axis=1)
        print(f"{name:18s} mean multiplier per seed: {arms[name].round(2)}")

    higher = arms["conversions"] > arms["surplus"]
    answer = "yes" if higher.all() else "no"
    print(f"Answer: {answer}, higher in {higher.sum()} of {len(higher)} seeds")
