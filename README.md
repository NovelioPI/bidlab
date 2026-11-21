# bidlab

A Python library for testing ad bidding methods, such as pacing, bid shading, and learning bidders, in simulated auctions.

You write a bidder class and a short Python script. bidlab runs the auctions, hides what a real bidder could not see, and reports the same metrics for every method.

## Install

bidlab is not on PyPI yet. From a clone of this repo:

```bash
uv sync                      # core: numpy, scipy, cloudpickle
uv sync --extra pandas       # adds Results.to_pandas()
```

## Quickstart

```python
import bidlab as bl

config = bl.SimConfig(
    campaigns=[
        bl.Campaign("truthful", bl.Linear(), budget=50, base_value=5),
        bl.Campaign("pid", bl.PidPacing(), budget=15, base_value=5),
        bl.Campaign("bandit", bl.UcbMultiplier(), budget=50, base_value=5),
    ],
    auction=bl.SecondPrice(reserve=0.01),
    episodes=5,
    steps=500,
)

results = bl.run_many(config, seeds=range(3))
print(results.summary())
```

`summary()` shows, for each bidder, the mean across seeds and a 95% Student-t interval.

## How one step works

1. The market draws the impression's public features and each campaign's conversion probability pᵢ, and optionally a click probability qᵢ. The default market, `LogisticMarket`, uses pᵢ = σ(x · φᵢ + βᵢ) for a random feature vector x, with no click stage (qᵢ = 1).
2. Each bidder's `forward` gets the inputs it names, such as `value`, `time`, `budget_left`, `floor` and each public feature. Its `value` is the platform's value of the impression: the campaign's base value w times the chance of its event, which is 1 for an impression, qᵢ for a click, and qᵢ × pᵢ for a conversion, the default.
3. Each bid is capped at the bidder's remaining budget. The bidder does not see that amount; it tracks its own spend.
4. The auction picks a winner: `FirstPrice` or `SecondPrice`, with an optional reserve price. A market can add a floor and a hidden market price, the highest bid from outside; a campaign must beat it to win.
5. The winner is clicked with probability q, then converts with probability p of the winner.
6. Each bidder gets its own `BidResult`: the auction's `time`, its bid, and whether it won. It also holds the price it paid, whether it was clicked and converted, and its `realized_value`. A bidder computes its own reward from it. A loser never sees the winning price.

The `value` input is exact by default. [Simulation](https://github.com/NovelioPI/bidlab/blob/main/docs/guides/simulation.md) covers the clock, noise in that value, and reproducibility.

## Write a bidder

Subclass `bl.Bidder` and implement `forward`. Like a PyTorch module, `forward` names the inputs it needs, such as `forward(self, value, time_left)`. Alone, a bidder gets the platform's `value`, and its output is the bid.

```python
class FlatBid(bl.Bidder):
    def __init__(self, amount):
        self.amount = amount

    def forward(self, value):
        return self.amount


flat = bl.SimConfig(
    campaigns=[
        bl.Campaign("flat", FlatBid(0.2), budget=50, base_value=5),
        bl.Campaign("truthful", bl.Linear(), budget=50, base_value=5),
    ],
    episodes=2,
    steps=200,
)
print(bl.run_many(flat, seeds=range(2)).summary())
```

[Bidders](https://github.com/NovelioPI/bidlab/blob/main/docs/guides/bidders.md) covers every input, the other hooks, learning from results, and stacking bidders with `bl.Stack`.

## Guides

| Guide | Covers |
|---|---|
| [Simulation](https://github.com/NovelioPI/bidlab/blob/main/docs/guides/simulation.md) | Each step in detail, the clock, estimate noise, parallel runs, reproducibility |
| [Campaigns](https://github.com/NovelioPI/bidlab/blob/main/docs/guides/campaigns.md) | Budget, the event a campaign pays for, base value, CPA target, segments |
| [Bidders](https://github.com/NovelioPI/bidlab/blob/main/docs/guides/bidders.md) | Inputs, hooks, value models and bid control, `Stack`, built-in bidders |
| [Composite bidders](https://github.com/NovelioPI/bidlab/blob/main/docs/guides/composite-bidders.md) | Bidders that hold and call other bidders |
| [Markets](https://github.com/NovelioPI/bidlab/blob/main/docs/guides/markets.md) | Built-in markets, segments and targeting, custom markets |
| [Rewards](https://github.com/NovelioPI/bidlab/blob/main/docs/guides/rewards.md) | A reward in the bidder's own code, logging it, comparing bidders |
| [Logging and callbacks](https://github.com/NovelioPI/bidlab/blob/main/docs/guides/logging.md) | `self.log` in a bidder, and watching a run |
| [Results](https://github.com/NovelioPI/bidlab/blob/main/docs/guides/results.md) | `Results`, `RunLog`, metric functions, regret |

[Metrics](https://github.com/NovelioPI/bidlab/blob/main/docs/metrics.md) defines every metric and its limits.

## Examples

Each script runs on its own, prints its results, and ends with the answer to its question.

| Script | Question |
|---|---|
| [getting_started/basic.py](https://github.com/NovelioPI/bidlab/blob/main/examples/getting_started/basic.py) | When the budget is tight, does PID pacing earn more surplus than truthful bidding? |
| [campaigns/events.py](https://github.com/NovelioPI/bidlab/blob/main/examples/campaigns/events.py) | With the same budget, does a click campaign pay less per click than a conversion campaign? |
| [bidders/segment_rate.py](https://github.com/NovelioPI/bidlab/blob/main/examples/bidders/segment_rate.py) | Does a bidder that counts conversions per segment learn that sports converts better than news? |
| [bidders/value_estimators.py](https://github.com/NovelioPI/bidlab/blob/main/examples/bidders/value_estimators.py) | Does a two-stage model on the public features estimate conversions better than counting? |
| [composite/night_shade.py](https://github.com/NovelioPI/bidlab/blob/main/examples/composite/night_shade.py) | When a parent shades harder at night, does its child PID raise its multiplier at night? |
| [markets/segments.py](https://github.com/NovelioPI/bidlab/blob/main/examples/markets/segments.py) | Does a sports-only bidder win any auction outside sports? |
| [markets/custom_market.py](https://github.com/NovelioPI/bidlab/blob/main/examples/markets/custom_market.py) | Does the all-day bidder pay more per win at night, when the night-only bidder joins? |
| [markets/independent_values.py](https://github.com/NovelioPI/bidlab/blob/main/examples/markets/independent_values.py) | In a first-price auction with uniform values, does shading to (n-1)/n beat truthful bidding? |
| [rewards/cpa_target.py](https://github.com/NovelioPI/bidlab/blob/main/examples/rewards/cpa_target.py) | Does bidding target/value of the value meet a CPA target that truthful bidding misses? |
| [rewards/custom_reward.py](https://github.com/NovelioPI/bidlab/blob/main/examples/rewards/custom_reward.py) | Does a bandit rewarded for conversions pick a higher multiplier than one rewarded for surplus? |
| [logging/pid_trace.py](https://github.com/NovelioPI/bidlab/blob/main/examples/logging/pid_trace.py) | Does PID pacing spend its whole budget in each episode? |
| [simulation/estimate_noise.py](https://github.com/NovelioPI/bidlab/blob/main/examples/simulation/estimate_noise.py) | Does noise in the platform's value cost a truthful bidder surplus? |
| [simulation/clock.py](https://github.com/NovelioPI/bidlab/blob/main/examples/simulation/clock.py) | With a 12-hour day from 9:00, does an even-schedule bidder still have budget at 20:00, when truthful bidding has run out? |
| [results/regret.py](https://github.com/NovelioPI/bidlab/blob/main/examples/results/regret.py) | With no budget limit in a second-price auction, is truthful bidding's regret zero? |
| [results/reserve.py](https://github.com/NovelioPI/bidlab/blob/main/examples/results/reserve.py) | Does a reserve price of 0.1 raise the platform's revenue? |
