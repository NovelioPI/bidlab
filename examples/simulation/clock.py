"""With a 12-hour day from 9:00, does an even-schedule bidder still have budget at 20:00, when truthful bidding has run out?

`EvenSpend` tracks its own spend in `update`, one feedback at a time. It bids
the value only while its spend is behind an even plan over the day, which it
reads from `campaign.time_left`.
"""

import dataclasses

import numpy as np

import bidlab as bl

CHECK_HOUR = 20
OUT_OF_BUDGET = 0.99


class EvenSpend(bl.Bidder):
    def on_episode_start(self, campaign):
        self.campaign = campaign
        self.spent = 0.0
        self.day = campaign.time_left(0.0)

    def forward(self, value, observation):
        self.log("hour", self.campaign.hour_of_day(observation.time))
        elapsed = 1 - self.campaign.time_left(observation.time) / self.day
        return value if self.spent <= elapsed * self.campaign.budget else 0.0

    def update(self, feedback):
        if feedback.won:
            self.spent += feedback.price


config = bl.SimConfig(
    bidders=[
        bl.BidderSpec("even", EvenSpend(), bl.Campaign(budget=15, value=5)),
        bl.BidderSpec("truthful", bl.Truthful(), bl.Campaign(budget=15, value=5)),
        bl.BidderSpec("big", bl.Truthful(), bl.Campaign(budget=50, value=5)),
    ],
    episodes=5,
    steps=480,
    horizon_hours=12,
    start_hour=9,
)


def hour_out_of_budget(log, bidder, hours):
    """Hour each episode's spend first reaches 99% of the budget; NaN if never."""
    budget = log.config.bidders[bidder].campaign.budget
    spend = np.where(log.winner == bidder, log.price, 0.0).cumsum(axis=1)
    out = spend >= OUT_OF_BUDGET * budget
    first = out.argmax(axis=1)
    return np.where(out.any(axis=1), hours[np.arange(len(first)), first], np.nan)


if __name__ == "__main__":
    still_bidding = []
    for seed in range(5):
        log = bl.run(dataclasses.replace(config, seed=seed))
        hours = log.logs["hour"]["even"]
        even = hour_out_of_budget(log, 0, hours)
        truthful = hour_out_of_budget(log, 1, hours)
        print(
            f"seed {seed}: out of budget at {truthful.round(1)} truthful, {even.round(1)} even"
        )
        # NaN compares False, so a bidder that never ran out passes the first test.
        still_bidding.append(
            not (even < CHECK_HOUR).any() and (truthful < CHECK_HOUR).all()
        )

    answer = "yes" if all(still_bidding) else "no"
    print(
        f"Answer: {answer}, true in {sum(still_bidding)} of {len(still_bidding)} seeds"
    )
