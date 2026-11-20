"""Does a bandit rewarded for conversions pick a higher multiplier than one rewarded for surplus?

`UcbMultiplier` picks one multiplier per episode and learns from the episode's
reward. With no episode reward, that is the sum of its step rewards. A
conversion reward ignores the price, so a higher multiplier always looks
better.
"""

import dataclasses

import bidlab as bl

LAST_EPISODES = 10


def cheap_conversions(
    request: bl.BidRequest, result: bl.BidResult, params: dict[str, float]
) -> float:
    """A conversion counts 1; each unit of price costs `cost_weight`."""
    return float(result.converted) - params["cost_weight"] * result.price


def bandit(**rewards):
    return bl.Campaign(
        "bandit",
        bl.UcbMultiplier(multiplier_step=0.6),
        budget=1e9,
        event_value=5,
        **rewards,
    )


BANDITS = {
    "surplus": bandit(step_reward=bl.rewards.surplus),
    "conversions": bandit(step_reward=bl.rewards.conversions),
    "realized_value": bandit(step_reward=bl.rewards.realized_value),
    "cheap_conversions": bandit(
        step_reward=cheap_conversions, reward_params={"cost_weight": 0.5}
    ),
    "total_surplus": bandit(episode_reward=bl.rewards.total_surplus),
}

base = bl.SimConfig(
    campaigns=[
        BANDITS["surplus"],
        bl.Campaign("other", bl.Linear(), budget=1e9, event_value=5),
    ],
    episodes=30,
    steps=300,
)


if __name__ == "__main__":
    arms = {}
    for name, campaign in BANDITS.items():
        config = dataclasses.replace(base, campaigns=[campaign, base.campaigns[1]])
        results = bl.run_many(config, seeds=range(5))
        arms[name] = results.logged["multiplier"]["bandit"][:, -LAST_EPISODES:].mean(
            axis=1
        )
        print(f"{name:18s} mean multiplier per seed: {arms[name].round(2)}")

    higher = arms["conversions"] > arms["surplus"]
    answer = "yes" if higher.all() else "no"
    print(f"Answer: {answer}, higher in {higher.sum()} of {len(higher)} seeds")
