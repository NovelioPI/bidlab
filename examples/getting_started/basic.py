"""Four bidders in a second-price auction, run over five seeds in parallel."""

import bidlab as bl

config = bl.SimConfig(
    bidders=[
        bl.BidderSpec("truthful", bl.Truthful, budget=50, value_per_conversion=5),
        bl.BidderSpec(
            "shaded",
            bl.Truthful,
            budget=50,
            value_per_conversion=5,
            params={"multiplier": 0.8},
        ),
        bl.BidderSpec(
            "pid",
            bl.PidPacing,
            budget=15,
            value_per_conversion=5,
            cpa_target=3,
            episode_reward=bl.rewards.cpa_score,
        ),
        bl.BidderSpec("bandit", bl.UcbMultiplier, budget=50, value_per_conversion=5),
    ],
    auction=bl.SecondPrice(reserve=0.01),
    episodes=20,
    steps=1000,
)

if __name__ == "__main__":
    results = bl.run_many(config, seeds=range(5), workers=4)
    print(results.summary())
