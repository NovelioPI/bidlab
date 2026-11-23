"""With a 12-hour day from 9:00, does an even-schedule bidder still have budget at 20:00, when truthful bidding has run out?

`EvenSpend` sits after `OracleValue` in a `Stack`. It tracks its own spend and
surplus in `update`, one result at a time. It bids the value only while its spend is behind an even plan over the
day, which it reads from its `time` and `horizon` inputs.
"""

import dataclasses

import numpy as np

import bidlab as bl

CHECK_HOUR = 20
OUT_OF_BUDGET = 0.99


class EvenSpend(bl.Bidder):
    def on_episode_start(self, campaign):
        self.budget = campaign.budget
        self.spent = 0.0

    def forward(self, value, time, horizon, hour_of_day):
        self.log("hour", hour_of_day)
        return value if self.spent <= time / horizon * self.budget else 0.0

    def update(self, result):
        surplus = result.realized_value - result.price
        self.log({"bid": result.bid, "reward": surplus})
        if result.won:
            self.spent += result.price


config = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "even", bl.Stack(bl.OracleValue(), EvenSpend()), budget=15, base_value=5
        ),
        bl.Campaign(
            "truthful", bl.Stack(bl.OracleValue(), bl.Linear()), budget=15, base_value=5
        ),
        bl.Campaign(
            "big", bl.Stack(bl.OracleValue(), bl.Linear()), budget=50, base_value=5
        ),
    ],
    episodes=5,
    steps=480,
    horizon_hours=12,
    start_hour=9,
)


def hour_out_of_budget(log, bidder, hours):
    """Hour each episode's spend first reaches 99% of the budget; NaN if never."""
    budget = log.config.campaigns[bidder].budget
    spend = log.price[:, :, bidder].cumsum(axis=1)
    out = spend >= OUT_OF_BUDGET * budget
    first = out.argmax(axis=1)
    return np.where(out.any(axis=1), hours[np.arange(len(first)), first], np.nan)


if __name__ == "__main__":
    still_bidding = []
    for seed in range(5):
        log = bl.run(dataclasses.replace(config, seed=seed))
        hours = log.logged["bidders.1.hour"]["even"]
        paused = (log.logged["bidders.1.bid"]["even"] == 0).mean()
        reward = log.logged["bidders.1.reward"]["even"].sum(axis=1).mean()
        print(
            f"seed {seed}: even paused in {paused:.0%} of auctions, reward {reward:.1f} per day"
        )
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
