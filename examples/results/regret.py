"""With no budget limit in a second-price auction, is truthful bidding's regret zero?

Regret replays an episode with the bidder bidding m times its value for each
fixed multiplier m, with the other bids held fixed. In second price, bidding the
value is the best reply to any fixed bids, so no multiplier beats it.
"""

import dataclasses

import bidlab as bl

TOLERANCE = 1e-9

config = bl.SimConfig(
    bidders=[
        bl.BidderSpec("truthful", bl.Truthful(), bl.Campaign(budget=1e9, value=5)),
        bl.BidderSpec(
            "shaded", bl.Truthful(multiplier=0.5), bl.Campaign(budget=1e9, value=5)
        ),
        bl.BidderSpec("other", bl.Truthful(), bl.Campaign(budget=1e9, value=5)),
    ],
    auction=bl.SecondPrice(),
    episodes=3,
    steps=500,
)

if __name__ == "__main__":
    log = bl.run(config)
    print(f"config: {len(log.config.bidders)} bidders, {log.config.episodes} episodes")
    print(f"bids and values per bidder: {log.bids.shape}, {log.values.shape}")
    print(f"winner, price, converted per step: {log.winner.shape}")
    print(f"rewards: {log.rewards.shape}, episode rewards: {log.episode_rewards.shape}")
    print(f"first segments: {log.segment[0, :3]}, logs: {dict(log.logs)}")
    print(f"auctions with no winner: {(log.winner == -1).sum()}")
    print(f"price paid: {log.price[log.converted].mean():.3f} on converted steps")
    bidder = bl.bidder_metrics(log)[0]
    print(
        f"{bidder.bidder}, episode 0: surplus {bidder.surplus:.2f}, regret {bidder.regret:.2g}"
    )
    segment = bl.segment_metrics(log)[0]
    print(f"{segment.bidder} in segment {segment.segment!r}: {segment.wins} wins")
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
