"""Does a bandit rewarded for conversions pick a higher multiplier than one rewarded for surplus?

`UcbMultiplier` picks one multiplier per episode and learns from the episode's
reward. With no episode reward, that is the sum of its step rewards. A
conversion reward ignores the price, so a higher multiplier always looks
better.
"""

import dataclasses

import bidlab as bl

LAST_EPISODES = 10


def cheap_conversions(observation, feedback, params):
    """A conversion counts 1; each unit of price costs `cost_weight`."""
    return float(feedback.converted) - params["cost_weight"] * feedback.price


REWARDS = {
    "surplus": {"reward": bl.rewards.surplus},
    "conversions": {"reward": bl.rewards.conversions},
    "value": {"reward": bl.rewards.value},
    "cheap_conversions": {
        "reward": cheap_conversions,
        "reward_params": {"cost_weight": 0.5},
    },
    "total_surplus": {"episode_reward": bl.rewards.total_surplus},
}

base = bl.SimConfig(
    bidders=[
        bl.BidderSpec(
            "bandit",
            bl.UcbMultiplier(multiplier_step=0.6),
            bl.Campaign(budget=1e9, value=5),
        ),
        bl.BidderSpec("other", bl.Truthful(), bl.Campaign(budget=1e9, value=5)),
    ],
    episodes=30,
    steps=300,
)


def config_for(rewards):
    bandit = dataclasses.replace(base.bidders[0], **rewards)
    return dataclasses.replace(base, bidders=[bandit, base.bidders[1]])


if __name__ == "__main__":
    arms = {}
    for name, rewards in REWARDS.items():
        results = bl.run_many(config_for(rewards), seeds=range(5))
        arms[name] = results.logs["multiplier"]["bandit"][:, -LAST_EPISODES:].mean(
            axis=1
        )
        print(f"{name:18s} mean multiplier per seed: {arms[name].round(2)}")

    higher = arms["conversions"] > arms["surplus"]
    answer = "yes" if higher.all() else "no"
    print(f"Answer: {answer}, higher in {higher.sum()} of {len(higher)} seeds")
