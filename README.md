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
        bl.Campaign("truthful", bl.Linear(), budget=50, event_value=5),
        bl.Campaign("pid", bl.PidPacing(), budget=15, event_value=5),
        bl.Campaign("bandit", bl.UcbMultiplier(), budget=50, event_value=5),
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

1. The market draws the impression's public features and each bidder's conversion probability pᵢ. The default market, `LogisticContext`, uses pᵢ = σ(x · φᵢ + βᵢ) for a random context x.
2. Each bidder sees its own `BidRequest`. It holds the `time` in seconds since the episode start and the public `features`. Its `value` is the platform's value of the impression, w × pᵢ.
3. Each bid is capped at the bidder's remaining budget. The bidder does not see that amount; it tracks its own spend.
4. The auction picks a winner: `FirstPrice` or `SecondPrice`, with an optional reserve price.
5. The winner converts with probability p of the winner.
6. Each bidder gets its own `BidResult`: the auction's `time`, its bid, and whether it won. It also holds the price it paid, whether it converted, its `realized_value`, and its reward. A loser never sees the winning price.

`request.value` is exact by default. [Simulation](docs/guides/simulation.md) covers the clock, noise in that value, and reproducibility.

## Write a bidder

Subclass `bl.Bidder` and implement `forward(value, request)`. Alone, a bidder gets `request.value`, and its output is the bid.

```python
class FlatBid(bl.Bidder):
    def __init__(self, amount):
        self.amount = amount

    def forward(self, value, request):
        return self.amount


flat = bl.SimConfig(
    campaigns=[
        bl.Campaign("flat", FlatBid(0.2), budget=50, event_value=5),
        bl.Campaign("truthful", bl.Linear(), budget=50, event_value=5),
    ],
    episodes=2,
    steps=200,
)
print(bl.run_many(flat, seeds=range(2)).summary())
```

[Bidders](docs/guides/bidders.md) covers the other hooks, learning from results, and stacking bidders with `bl.Stack`.

## Guides

| Guide | Covers |
|---|---|
| [Simulation](docs/guides/simulation.md) | Each step in detail, the clock, estimate noise, parallel runs, reproducibility |
| [Bidders](docs/guides/bidders.md) | Hooks, value models and bid control, `Stack`, built-in bidders |
| [Composite bidders](docs/guides/composite-bidders.md) | Bidders that hold and call other bidders |
| [Markets](docs/guides/markets.md) | Built-in markets, segments and targeting, custom markets |
| [Rewards](docs/guides/rewards.md) | Step and episode rewards, CPA targets, your own reward |
| [Logging and callbacks](docs/guides/logging.md) | `self.log` in a bidder, and watching a run |
| [Results](docs/guides/results.md) | `Results`, `RunLog`, metric functions, regret |

[Metrics](docs/metrics.md) defines every metric and its limits.

## Examples

Each script runs on its own, prints its results, and ends with the answer to its question.

| Script | Question |
|---|---|
| [getting_started/basic.py](examples/getting_started/basic.py) | When the budget is tight, does PID pacing earn more surplus than truthful bidding? |
| [bidders/segment_rate.py](examples/bidders/segment_rate.py) | Does a bidder that counts conversions per segment learn that sports converts better than news? |
| [composite/night_shade.py](examples/composite/night_shade.py) | When a parent shades harder at night, does its child PID raise its multiplier at night? |
| [markets/segments.py](examples/markets/segments.py) | Does a sports-only bidder win any auction outside sports? |
| [markets/custom_market.py](examples/markets/custom_market.py) | Does the all-day bidder pay more per win at night, when the night-only bidder joins? |
| [markets/independent_values.py](examples/markets/independent_values.py) | In a first-price auction with uniform values, does shading to (n-1)/n beat truthful bidding? |
| [rewards/cpa_target.py](examples/rewards/cpa_target.py) | Does bidding target/value of the value meet a CPA target that truthful bidding misses? |
| [rewards/custom_reward.py](examples/rewards/custom_reward.py) | Does a bandit rewarded for conversions pick a higher multiplier than one rewarded for surplus? |
| [logging/pid_trace.py](examples/logging/pid_trace.py) | Does PID pacing spend its whole budget in each episode? |
| [simulation/estimate_noise.py](examples/simulation/estimate_noise.py) | Does noise in the platform's value cost a truthful bidder surplus? |
| [simulation/clock.py](examples/simulation/clock.py) | With a 12-hour day from 9:00, does an even-schedule bidder still have budget at 20:00, when truthful bidding has run out? |
| [results/regret.py](examples/results/regret.py) | With no budget limit in a second-price auction, is truthful bidding's regret zero? |
| [results/reserve.py](examples/results/reserve.py) | Does a reserve price of 0.1 raise the platform's revenue? |
