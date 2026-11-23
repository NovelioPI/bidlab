"""Does a bidder that counts conversions per segment learn that sports converts better than news?

`SegmentRate` is an estimator: it learns each segment's conversion rate from
its own wins, and `PidPacing` after it in the `Stack` turns that value into a
paced bid.
"""

import bidlab as bl


class SegmentRate(bl.Bidder):
    def __init__(self, prior=0.05):
        self.prior = prior
        self.wins = {}
        self.conversions = {}
        self.pending = {}

    def on_episode_start(self, campaign):
        self.base_value = campaign.base_value

    def rate(self, segment):
        conversions = self.conversions.get(segment, 0) + self.prior
        return conversions / (self.wins.get(segment, 0) + 1)

    def forward(self, segment, time):
        # BidResult carries the auction time but not the segment.
        self.pending[time] = segment
        return self.rate(segment) * self.base_value

    def update_batch(self, bid_results):
        for result in bid_results:
            segment = self.pending.pop(result.time)
            if result.won:
                self.wins[segment] = self.wins.get(segment, 0) + 1
                self.conversions[segment] = (
                    self.conversions.get(segment, 0) + result.converted
                )

    def on_episode_end(self, report):
        self.log({f"rate_{segment}": self.rate(segment) for segment in self.wins})


market = bl.markets.Segmented(
    (
        bl.markets.Segment("news", share=0.5),
        bl.markets.Segment("sports", share=0.5, cvr_lift=1.0),
    )
)

config = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "learned",
            bl.Stack(SegmentRate(), bl.PidPacing()),
            budget=15,
            base_value=5,
        ),
        bl.Campaign(
            "platform",
            bl.Stack(bl.OracleValue(), bl.PidPacing()),
            budget=15,
            base_value=5,
        ),
    ],
    market=market,
    episodes=5,
    steps=500,
)

if __name__ == "__main__":
    results = bl.run(config, seed=range(5))
    print(results.summary())
    # The stack's first child, at path "bidders.0", logged the rates.
    news = results.logged["bidders.0.rate_news"]["learned"][:, -1]
    sports = results.logged["bidders.0.rate_sports"]["learned"][:, -1]
    answer = "yes" if (sports > news).all() else "no"
    print(
        f"Answer: {answer}, last-episode rate {sports.mean():.3f} in sports"
        f" against {news.mean():.3f} in news, over {len(news)} seeds"
    )
