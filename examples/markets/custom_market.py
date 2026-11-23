"""Does the all-day bidder pay more per win at night, when the night-only bidder joins?

Conversion rates are higher by day, so day impressions are worth more. At night
a third bidder competes, and in a second-price auction more competition raises
the price.
"""

import bidlab as bl

STEPS_PER_DAY = 100
DAY_STEPS = 60


class DayAndNight(bl.Market):
    """Conversion rates are higher by day; each bidder's rate varies per step."""

    @property
    def segment_names(self):
        return ("day", "night")

    def draws(self, count, rng):
        step = 0
        while True:
            name = "day" if step % STEPS_PER_DAY < DAY_STEPS else "night"
            rate = 0.08 if name == "day" else 0.03
            pcvr = rate * rng.uniform(0.5, 1.5, size=count)
            yield bl.markets.MarketDraw({}, pcvr=pcvr, segment=name)
            step += 1


config = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "truthful", bl.Stack(bl.OracleValue(), bl.Linear()), budget=40, base_value=5
        ),
        bl.Campaign(
            "night_only",
            bl.Stack(bl.OracleValue(), bl.Linear()),
            budget=40,
            base_value=5,
            segments=("night",),
        ),
        bl.Campaign(
            "bandit",
            bl.Stack(bl.OracleValue(), bl.UcbMultiplier()),
            budget=40,
            base_value=5,
        ),
    ],
    market=DayAndNight(),
    episodes=10,
    steps=1000,
)

if __name__ == "__main__":
    results = bl.run(config, seed=range(5))
    assert results.segment_rows is not None
    print(results.segment_rows.summary())
    spend = {"day": 0.0, "night": 0.0}
    wins = {"day": 0.0, "night": 0.0}
    for row in results.segment_rows.rows:
        if row["campaign"] == "truthful":
            segment = str(row["segment"])
            spend[segment] += float(row["spend"])
            wins[segment] += float(row["wins"])

    night = spend["night"] / wins["night"]
    day = spend["day"] / wins["day"]
    answer = "yes" if night > day else "no"
    print(f"Answer: {answer}, {night:.3f} per win at night against {day:.3f} by day")
