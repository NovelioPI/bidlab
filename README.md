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
    bidders=[
        bl.BidderSpec("truthful", bl.Truthful, budget=50, value_per_conversion=5),
        bl.BidderSpec("pid", bl.PidPacing, budget=15, value_per_conversion=5),
        bl.BidderSpec("bandit", bl.UcbMultiplier, budget=50, value_per_conversion=5),
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
2. Each bidder sees its own `Observation`: step, steps left, value vᵢ = wᵢ × pᵢ, value per conversion wᵢ, budget left, and the public features.
3. Each bid is capped at the bidder's remaining budget.
4. The auction picks a winner: `FirstPrice` or `SecondPrice`, with an optional reserve price.
5. The winner converts with probability p of the winner.
6. Each bidder gets its own `Feedback`: its bid, whether it won, the price it paid, whether it converted, and its reward. A loser never sees the winning price.

Budgets reset at the start of each episode. Bidder objects persist for the whole run, so a learning bidder keeps what it learned.

## Write a bidder

Subclass `bl.Bidder` and override `bid`. The optional hooks are `on_episode_start(budget, steps)`, `update(feedback)`, and `on_episode_end(stats, episode_reward)`.

```python
class HalfValue(bl.Bidder):
    def bid(self, observation):
        return self.params.get("fraction", 0.5) * observation.value


mine = bl.SimConfig(
    bidders=[
        bl.BidderSpec("half", HalfValue, budget=50, value_per_conversion=5),
        bl.BidderSpec("pid", bl.PidPacing, budget=50, value_per_conversion=5),
    ],
    episodes=5,
    steps=500,
)
print(bl.run_many(mine, seeds=range(3)).summary())
```

To pass an object the bidder needs, such as a trained model, give a factory in place of the class. The simulator calls it once per run with the spec's `params` and the bidder's own random generator.

```python
def make_half(params, rng):
    return HalfValue({"fraction": 0.4}, rng)


factory_spec = bl.BidderSpec("half", make_half, budget=50, value_per_conversion=5)
```

## Built-in bidders

| Class | Family | Behavior | Params |
|---|---|---|---|
| `Truthful` | `bl.bidders.baselines` | Bids `multiplier` × value | `multiplier` (1.0) |
| `PidPacing` | `bl.bidders.pacing` | A PID loop moves a bid multiplier so spend tracks the budget evenly | `initial_multiplier` (1.0), `kp` (20), `ki` (2), `kd` (0), `min_multiplier` (0), `max_multiplier` (10) |
| `UcbMultiplier` | `bl.bidders.bandits` | UCB1 over a grid of multipliers; one arm per episode | `min_multiplier` (0.2), `max_multiplier` (2.0), `multiplier_step` (0.2), `exploration` (1.0) |

## Rewards

Each bidder has a `reward`, called every step, and an optional `episode_reward`, called once per episode. Both are plain functions.

| Kind | Built-in functions | Your own |
|---|---|---|
| Step | `bl.rewards.surplus` (default), `conversions`, `value` | `f(observation, feedback, params) -> float` |
| Episode | `bl.rewards.cpa_score`, `total_surplus` | `g(stats, params) -> float` |

The `params` dict holds the bidder's `reward_params`, plus `cpa_target` when the bidder has one.

```python
cpa_spec = bl.BidderSpec(
    "pid",
    bl.PidPacing,
    budget=15,
    value_per_conversion=5,
    cpa_target=3,
    episode_reward=bl.rewards.cpa_score,
)
```

## Markets

A market draws each step's public features and each bidder's private conversion probability. Pass one as `SimConfig(market=...)`.

| Market | Draws |
|---|---|
| `bl.markets.LogisticContext()` | The default: pᵢ = σ(x · φᵢ + βᵢ) for a random context x |
| `bl.markets.Segmented(segments)` | A segment per step by its `share`; its `lift` is added to every bidder's logit |
| `bl.markets.IndependentValues(distribution)` | Each pᵢ drawn on its own from a scipy frozen distribution, clipped to [0, 1] |

A segmented market puts the segment name in `observation.features["segment"]`. A bidder with `segments` bids only in those segments. Elsewhere its `bid` and `update` methods are not called, and its logged bid is 0.

```python
market = bl.markets.Segmented(
    (
        bl.markets.Segment("news", share=0.7),
        bl.markets.Segment("sports", share=0.3, lift=1.0),
    )
)
segmented = bl.SimConfig(
    bidders=[
        bl.BidderSpec("everywhere", bl.Truthful, budget=50, value_per_conversion=5),
        bl.BidderSpec(
            "sports_only",
            bl.Truthful,
            budget=50,
            value_per_conversion=5,
            segments=("sports",),
        ),
    ],
    market=market,
    episodes=5,
    steps=500,
)
segment_results = bl.run_many(segmented, seeds=range(3))
print(segment_results.segments.summary())
```

`results.segments` is a second `Results` table with spend, wins, conversions, and surplus per seed, bidder, and segment. A market without segments reports one segment, `"all"`. The main table does not change.

A targeted bidder still sees `steps_left` count every step, including steps in other segments. A pacing bidder may under-spend unless it adjusts for that.

To write your own market, subclass `bl.markets.Market`. Its `draws` generator yields one `MarketDraw` per step for the whole run. Use only the `rng` it receives, so runs repeat with the seed.

```python
import numpy as np


class Daytime(bl.markets.Market):
    def draws(self, bidders, rng):
        step = 0
        while True:
            hour = step % 24
            rate = 0.08 if 9 <= hour < 18 else 0.02
            yield bl.markets.MarketDraw(
                {"hour": hour}, rng.uniform(0.5, 1.5, size=bidders) * rate
            )
            step += 1


daytime = bl.SimConfig(
    bidders=[bl.BidderSpec("pid", bl.PidPacing, budget=15, value_per_conversion=5)],
    market=Daytime(),
    episodes=2,
    steps=240,
)
print(bl.run_many(daytime, seeds=range(2)).summary())
```

`IndependentValues(scipy.stats.uniform(0, 1))` with `value_per_conversion=1` is the textbook independent private values setting.

## Results

`run_many` returns a `Results` object with one row per seed and bidder. Each row holds every metric, averaged over episodes.

| Method | Gives |
|---|---|
| `results.summary()` | A text table of means and 95% intervals |
| `results.save_csv(path)` | A CSV file with one row per seed and bidder |
| `results.to_pandas()` | A DataFrame; needs `pip install bidlab[pandas]` |
| `results.rows`, `results.columns` | The raw rows as dicts |
| `results.equals(other)` | `True` when two runs gave the same rows. Use it, not `==`: some metrics are NaN by definition, and NaN never equals NaN |
| `results.segments` | The per-segment table, also a `Results`; `equals` on the main table does not compare it |

## Parallel runs

`run_many(config, seeds, workers=4)` runs seeds in 4 processes and gives the same rows as `workers=1`. Lambdas and closures work. Workers start with `spawn`, which re-imports your script, so put the call under a main guard:

```python
if __name__ == "__main__":
    parallel = bl.run_many(mine, seeds=range(4), workers=2)
    print(parallel.summary())
```

A factory that holds a large object, such as a model, is copied to each worker.

## Metrics

`docs/metrics.md` defines every metric. Regret compares a bidder with the best fixed bid multiplier in hindsight, with other bids held fixed. Read its limits there before you compare methods.

## Reproducibility

Each source of randomness has its own seeded stream: the market, tie-breaking, conversions, and each bidder. The same seed gives the same log. Adding a bidder does not change the impressions the others see.
