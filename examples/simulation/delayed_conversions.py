"""When conversions arrive 6 hours late, does a learner that counts each unconverted click as a miss bid lower?

`ConversionRate` estimates its conversion rate within each day: conversions
over clicks, from a prior of 1 in 10. It counts a click when the `BidResult`
arrives, and a conversion when the `Conversion` arrives in
`update_conversions`. With `bl.Feedback(conversion_delay=6 * 3600)`, every
click of the last 6 hours still looks unconverted, so the estimate drops.
It reads the true pCTR from ``market``, so only the conversion rate is learned.
"""

import dataclasses

import numpy as np

import bidlab as bl

PRIOR_CLICKS = 10.0
PRIOR_RATE = 0.1
DELAY = 6 * 3600


class ConversionRate(bl.Bidder):
    def on_episode_start(self, campaign):
        self.base_value = campaign.base_value
        self.clicks = PRIOR_CLICKS
        self.conversions = PRIOR_CLICKS * PRIOR_RATE

    def forward(self, market):
        rate = self.conversions / self.clicks
        self.log("rate", rate)
        return self.base_value * market.pctr * rate

    def update(self, result):
        self.clicks += result.clicked
        self.conversions += result.converted

    def update_conversions(self, conversions):
        self.conversions += len(conversions)


config = bl.SimConfig(
    campaigns=[
        bl.Campaign("learner", ConversionRate(), budget=50, base_value=5),
        bl.Campaign(
            "rival", bl.Stack(bl.OracleValue(), bl.Linear()), budget=50, base_value=5
        ),
    ],
    market=bl.markets.LogisticMarket(ctr=0.05, cvr=0.1),
    episodes=5,
)
late = dataclasses.replace(config, feedback=bl.Feedback(conversion_delay=DELAY))


if __name__ == "__main__":
    lower = []
    for seed in range(5):
        on_time = bl.run(config, seed=seed)
        delayed = bl.run(late, seed=seed)
        bids = [log.bids[:, :, 0].mean() for log in (on_time, delayed)]
        rates = [
            np.nanmean(log.logged["rate"]["learner"]) for log in (on_time, delayed)
        ]
        print(
            f"seed {seed}: mean bid {bids[0]:.4f} on time, {bids[1]:.4f} late; "
            f"mean estimated rate {rates[0]:.3f} on time, {rates[1]:.3f} late"
        )
        lower.append(bids[1] < bids[0])

    answer = "yes" if all(lower) else "no"
    print(f"Answer: {answer}, lower in {sum(lower)} of {len(lower)} seeds")
