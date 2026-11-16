"""Four bidders in a second-price auction, run over five seeds in parallel."""

import bidlab as bl

config = bl.SimConfig(
    bidders=[
        bl.BidderSpec("truthful", bl.Truthful(), bl.Campaign(budget=50, value=5)),
        bl.BidderSpec(
            "shaded", bl.Truthful(multiplier=0.8), bl.Campaign(budget=50, value=5)
        ),
        bl.BidderSpec(
            "pid",
            bl.PidPacing(),
            bl.Campaign(budget=15, value=5, cpa_target=3),
            episode_reward=bl.rewards.cpa_score,
        ),
        bl.BidderSpec("bandit", bl.UcbMultiplier(), bl.Campaign(budget=50, value=5)),
    ],
    auction=bl.SecondPrice(reserve=0.01),
    episodes=20,
    steps=1000,
)

if __name__ == "__main__":
    results = bl.run_many(config, seeds=range(5), workers=4)
    print(results.summary())
