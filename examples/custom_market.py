"""A custom market with its own segments: busy days and quiet nights."""

import bidlab as bl

STEPS_PER_DAY = 100
DAY_STEPS = 60


class DayAndNight(bl.markets.Market):
    """Conversion rates are higher by day; each bidder's rate varies per step."""

    @property
    def segment_names(self):
        return ("day", "night")

    def draws(self, bidders, rng):
        step = 0
        while True:
            name = "day" if step % STEPS_PER_DAY < DAY_STEPS else "night"
            rate = 0.08 if name == "day" else 0.03
            probabilities = rate * rng.uniform(0.5, 1.5, size=bidders)
            yield bl.markets.MarketDraw({"segment": name}, probabilities, name)
            step += 1


config = bl.SimConfig(
    bidders=[
        bl.BidderSpec("truthful", bl.Truthful, budget=40, value_per_conversion=5),
        bl.BidderSpec(
            "night_only",
            bl.Truthful,
            budget=40,
            value_per_conversion=5,
            segments=("night",),
        ),
        bl.BidderSpec("bandit", bl.UcbMultiplier, budget=40, value_per_conversion=5),
    ],
    market=DayAndNight(),
    episodes=10,
    steps=1000,
)

if __name__ == "__main__":
    results = bl.run_many(config, seeds=range(5))
    print(results.segments.summary())
