"""When a parent shades harder at night, does its child PID raise its multiplier at night?

`NightShade` holds two child bidders and calls their `forward` itself. The
simulator gives each child its own random stream and every other hook, so the
PID child keeps pacing while its parent changes the value it receives.
"""

import dataclasses

import numpy as np

import bidlab as bl

NIGHT_END_HOUR = 6


class Shade(bl.Bidder):
    def forward(self, value, observation, factor=1.0):
        return value * factor


class NightShade(bl.Bidder):
    def __init__(self, night_factor=0.6, day_factor=0.9):
        self.night_factor = night_factor
        self.day_factor = day_factor
        self.shade = Shade()
        self.pacing = bl.PidPacing()

    def on_episode_start(self, campaign):
        self.campaign = campaign

    def forward(self, value, observation):
        night = self.campaign.hour_of_day(observation.time) < NIGHT_END_HOUR
        factor = self.night_factor if night else self.day_factor
        shaded = self.shade.forward(value, observation, factor=factor)
        return self.pacing.forward(shaded, observation)


config = bl.SimConfig(
    bidders=[
        bl.BidderSpec("night", NightShade(), bl.Campaign(budget=15, value=5)),
        bl.BidderSpec("pid", bl.PidPacing(), bl.Campaign(budget=15, value=5)),
        bl.BidderSpec("big", bl.Truthful(), bl.Campaign(budget=50, value=5)),
    ],
    episodes=5,
    steps=480,
)

if __name__ == "__main__":
    print([path for path, _ in config.bidders[0].bidder.named_bidders()])
    hours = np.arange(config.steps) * config.horizon_hours / config.steps
    at_night = hours < NIGHT_END_HOUR
    higher = []
    for seed in range(5):
        log = bl.run(dataclasses.replace(config, seed=seed))
        multiplier = log.logs["pacing.multiplier"]["night"]
        night = multiplier[:, at_night].mean()
        day = multiplier[:, ~at_night].mean()
        print(f"seed {seed}: child multiplier {night:.2f} at night, {day:.2f} by day")
        higher.append(night > day)

    answer = "yes" if all(higher) else "no"
    print(f"Answer: {answer}, higher at night in {sum(higher)} of {len(higher)} seeds")
