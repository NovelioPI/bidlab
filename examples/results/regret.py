"""With no budget limit in a second-price auction, is truthful bidding's regret zero?

Regret replays an episode with the bidder bidding m times its value for each
fixed multiplier m, with the other bids held fixed. In second price, bidding the
value is the best reply to any fixed bids, so no multiplier beats it.
"""

import dataclasses

import bidlab as bl

TOLERANCE = 1e-9

config = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "truthful",
            bl.Stack(bl.OracleValue(), bl.Linear()),
            budget=1e9,
            base_value=5,
        ),
        bl.Campaign(
            "shaded",
            bl.Stack(bl.OracleValue(), bl.Linear(multiplier=0.5)),
            budget=1e9,
            base_value=5,
        ),
        bl.Campaign(
            "other", bl.Stack(bl.OracleValue(), bl.Linear()), budget=1e9, base_value=5
        ),
    ],
    auction=bl.SecondPrice(),
    episodes=3,
    steps=500,
)

if __name__ == "__main__":
    log = bl.run(config)
    print(
        f"config: {len(log.config.campaigns)} campaigns, {log.config.episodes} episodes"
    )
    print(
        f"bids and true values per campaign: {log.bids.shape}, {log.true_values.shape}"
    )
    print(f"won, price, converted per step and campaign: {log.won.shape}")
    print(f"first segments: {log.segment[0, :3]}, logged: {dict(log.logged)}")
    print(f"auctions with no winner: {(~log.won.any(axis=2)).sum()}")
    print(f"price paid: {log.price[log.converted].mean():.3f} on converted steps")
    row = bl.campaign_metrics(log)[0]
    print(
        f"{row.campaign}, episode 0: surplus {row.surplus:.2f}, regret {row.regret:.2g}"
    )
    segment = bl.segment_metrics(log)[0]
    print(f"{segment.campaign} in segment {segment.segment!r}: {segment.wins} wins")
    platform = bl.platform_metrics(log)[0]
    print(f"platform: revenue {platform.revenue:.1f}, welfare {platform.welfare:.1f}")

    truthful = []
    shaded = []
    for seed in range(5):
        seed_log = bl.run(dataclasses.replace(config, seed=seed))
        for episode in range(config.episodes):
            truthful.append(bl.regret(seed_log, episode, 0))
            shaded.append(bl.regret(seed_log, episode, 1))

    answer = "yes" if max(abs(r) for r in truthful) < TOLERANCE else "no"
    print(
        f"Answer: {answer}, largest truthful regret {max(truthful):.2g},"
        f" smallest shaded regret {min(shaded):.2f}"
    )
