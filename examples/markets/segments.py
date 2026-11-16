"""A segmented market: one bidder bids everywhere, one only in sports."""

import bidlab as bl

market = bl.markets.Segmented(
    (
        bl.markets.Segment("news", share=0.5),
        bl.markets.Segment("sports", share=0.3, lift=1.0),
        bl.markets.Segment("games", share=0.2, lift=-0.5),
    )
)

config = bl.SimConfig(
    bidders=[
        bl.BidderSpec("everywhere", bl.Truthful, budget=50, value_per_conversion=5),
        bl.BidderSpec(
            "sports_only",
            bl.Truthful,
            budget=50,
            value_per_conversion=5,
            segments=("sports",),
        ),
        bl.BidderSpec("pid", bl.PidPacing, budget=15, value_per_conversion=5),
    ],
    market=market,
    auction=bl.SecondPrice(reserve=0.01),
    episodes=10,
    steps=1000,
)

if __name__ == "__main__":
    results = bl.run_many(config, seeds=range(5))
    print(results.segments.summary())
