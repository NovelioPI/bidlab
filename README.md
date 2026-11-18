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
        bl.BidderSpec("truthful", bl.Truthful(), bl.Campaign(budget=50, value=5)),
        bl.BidderSpec("pid", bl.PidPacing(), bl.Campaign(budget=15, value=5)),
        bl.BidderSpec("bandit", bl.UcbMultiplier(), bl.Campaign(budget=50, value=5)),
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
2. Each bidder sees its own `Observation`: the `time` in seconds since the episode start, the platform's `value` of the impression, w × pᵢ, and the public `features`.
3. Each bid is capped at the bidder's remaining budget. The bidder does not see that amount; it tracks its own spend.
4. The auction picks a winner: `FirstPrice` or `SecondPrice`, with an optional reserve price.
5. The winner converts with probability p of the winner.
6. Each bidder gets its own `Feedback`: the auction's `time`, its bid, whether it won, the price it paid, whether it converted, the realized `value`, and its reward. A loser never sees the winning price.

Budgets reset at the start of each episode, and each episode starts with `on_episode_start(campaign)`. Auctions are spaced evenly over `SimConfig.horizon_hours`, 24 by default. Bidder objects persist for the whole run, so a learning bidder keeps what it learned.

`observation.value` is the platform's value: w times its estimate of p. The estimate is exact by default. Set `SimConfig(estimate_noise=0.5)` to add normal noise with that standard deviation to logit(p), and test how a bidder copes with a wrong value. A bidder with its own model ignores it and predicts from `features`. Metrics always use the true value.

## Write a bidder

Subclass `bl.Bidder` and implement `forward(value, observation)`. A bidder turns a value into another value. Alone, it gets the platform's value, `observation.value`, and its output is the bid. Combine bidders with `bl.Stack`: each gets the previous one's output, and the last one's output is the bid. A value model, such as a conversion-rate model, returns its own value. A bid-control method, such as pacing, turns a value into a bid. Bidders in a stack run in order, so pacing then shading differs from shading then pacing.

```python
class SegmentRate(bl.Bidder):
    """Learns each segment's conversion rate by counting, and ignores the platform value."""

    def __init__(self, prior=0.05):
        self.prior = prior
        self.wins = {}
        self.conversions = {}
        self.pending = {}

    def on_episode_start(self, campaign):
        self.campaign = campaign

    def forward(self, value, observation):
        segment = observation.features["segment"]
        self.pending[observation.time] = segment
        wins = self.wins.get(segment, 0)
        rate = (self.conversions.get(segment, 0) + self.prior) / (wins + 1)
        return rate * self.campaign.value

    def update_batch(self, feedbacks):
        for feedback in feedbacks:
            segment = self.pending.pop(feedback.time)
            if feedback.won:
                self.wins[segment] = self.wins.get(segment, 0) + 1
                self.conversions[segment] = (
                    self.conversions.get(segment, 0) + feedback.converted
                )


two_segments = bl.markets.Segmented(
    (bl.markets.Segment("news", 0.5), bl.markets.Segment("sports", 0.5, lift=1.0))
)
learning = bl.SimConfig(
    bidders=[
        bl.BidderSpec(
            "learned", bl.Stack(SegmentRate(), bl.PidPacing()), bl.Campaign(15, 5)
        ),
        bl.BidderSpec("platform", bl.PidPacing(), bl.Campaign(budget=15, value=5)),
    ],
    market=two_segments,
    episodes=5,
    steps=500,
)
print(bl.run_many(learning, seeds=range(3)).summary())
```

A bidder keeps what it learns on `self`. Take settings as constructor arguments. The hooks are `on_episode_start(campaign)`, `forward(value, observation)`, `update_batch(feedbacks)` (or `update(feedback)` for one at a time), and `on_episode_end(stats, episode_reward)`. The simulator deep-copies the bidder for each run, so every seed starts from the bidder as you built it, and a pretrained model passes in as is. Each bidder, and each child bidder inside it, gets its own random stream as `self.rng`.

A bidder that ignores the incoming value works alone, too:

```python
class FlatBid(bl.Bidder):
    def __init__(self, amount):
        self.amount = amount

    def forward(self, value, observation):
        return self.amount


flat_spec = bl.BidderSpec("flat", FlatBid(0.2), bl.Campaign(budget=50, value=5))
```

A bidder can hold other bidders, its children, in attributes, lists or tuples. Call each child's `forward` yourself, with any extra keyword arguments it takes. `bl.Stack` is one such bidder.

Analogy: a PyTorch module that holds submodules.

The simulator does the rest. Each child gets its own `self.rng`, and `on_episode_start`, `update_batch` and `on_episode_end`, parent first. Do not call those on a child yourself, or it learns twice. Create children in `__init__`: a child added during a run raises an error.

```python
class Shade(bl.Bidder):
    def forward(self, value, observation, factor=1.0):
        return value * factor


class NightShade(bl.Bidder):
    """Shades harder at night, then paces. Its children still get every hook."""

    def __init__(self):
        self.shade = Shade()
        self.pacing = bl.PidPacing()

    def on_episode_start(self, campaign):
        self.campaign = campaign

    def forward(self, value, observation):
        factor = 0.6 if self.campaign.hour_of_day(observation.time) < 6 else 0.9
        shaded = self.shade.forward(value, observation, factor=factor)
        return self.pacing.forward(shaded, observation)


night_spec = bl.BidderSpec("night", NightShade(), bl.Campaign(budget=15, value=5))
print(
    [path for path, _ in night_spec.bidder.named_bidders()]
)  # ['', 'shade', 'pacing']
```

## Built-in bidders

| Class | Family | Behavior | Arguments |
|---|---|---|---|
| `Truthful` | `bl.bidders.baselines` | Outputs `multiplier` × value | `multiplier` (1.0) |
| `PidPacing` | `bl.bidders.pacing` | A PID loop moves a multiplier so spend tracks the budget evenly over the horizon; logs `"multiplier"` per auction | `initial_multiplier` (1.0), `kp` (20), `ki` (2), `kd` (0), `multiplier_range` ((0, 10)), `log_state` (True) |
| `UcbMultiplier` | `bl.bidders.bandits` | UCB1 over a grid of multipliers; one arm per episode; logs its arm as `"multiplier"` per episode | `min_multiplier` (0.2), `max_multiplier` (2.0), `multiplier_step` (0.2), `exploration` (1.0), `log_state` (True) |

## Rewards

Each bidder has a `reward`, called every step, and an optional `episode_reward`, called once per episode. Both are plain functions.

| Kind | Built-in functions | Your own |
|---|---|---|
| Step | `bl.rewards.surplus` (default), `conversions`, `value` | `f(observation, feedback, params) -> float` |
| Episode | `bl.rewards.cpa_score`, `total_surplus` | `g(stats, params) -> float` |

A step reward reads the realized value from `feedback.value`. The `params` dict holds the bidder's `reward_params`, plus the campaign's `cpa_target` when it has one.

```python
cpa_spec = bl.BidderSpec(
    "pid",
    bl.PidPacing(),
    bl.Campaign(budget=15, value=5, cpa_target=3),
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

A segmented market puts the segment name in `observation.features["segment"]`. A campaign with `segments` bids only in those segments. Elsewhere its bidder is not called, and its logged bid is 0.

```python
market = bl.markets.Segmented(
    (
        bl.markets.Segment("news", share=0.7),
        bl.markets.Segment("sports", share=0.3, lift=1.0),
    )
)
segmented = bl.SimConfig(
    bidders=[
        bl.BidderSpec("everywhere", bl.Truthful(), bl.Campaign(budget=50, value=5)),
        bl.BidderSpec(
            "sports_only",
            bl.Truthful(),
            bl.Campaign(budget=50, value=5, segments=("sports",)),
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

A targeted bidder sees only its segments' auctions. `PidPacing` plans spend by time, so it still spreads the budget over the whole horizon.

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
    bidders=[bl.BidderSpec("pid", bl.PidPacing(), bl.Campaign(budget=15, value=5))],
    market=Daytime(),
    episodes=2,
    steps=240,
)
print(bl.run_many(daytime, seeds=range(2)).summary())
```

A custom market can define its own segments. Override the `segment_names` property, and pass `segment=` to each `MarketDraw`. Targeting and `results.segments` then work as they do for `Segmented`.

`IndependentValues(scipy.stats.uniform(0, 1))` with `Campaign(value=1)` is the textbook independent private values setting.

## Examples

Each script in `examples/` runs on its own and prints a summary table. Scripts are grouped by topic.

| Script | Shows |
|---|---|
| `getting_started/basic.py` | Four built-in bidders over five seeds in parallel |
| `markets/segments.py` | A segmented market and a bidder that targets one segment |
| `markets/custom_market.py` | A custom market with its own day and night segments |
| `markets/independent_values.py` | First price with independent uniform values, truthful against shaded bids |

## Results

`run_many` returns a `Results` object with one row per seed and bidder. Each row holds every metric, averaged over episodes.

| Method | Gives |
|---|---|
| `results.summary()` | A text table of means and 95% intervals |
| `results.save_csv(path)` | A CSV file with one row per seed and bidder |
| `results.to_pandas()` | A DataFrame; needs `pip install bidlab[pandas]` |
| `results.rows`, `results.columns` | The raw rows as dicts |
| `results.equals(other)` | `True` when two runs gave the same rows and logs. Use it, not `==`: some metrics are NaN by definition, and NaN never equals NaN |
| `results.logs` | Values bidders logged, per seed; see below |
| `results.segments` | The per-segment table, also a `Results`; `equals` on the main table does not compare it |

## Log and watch a run

Call `self.log(name, value)`, or `self.log({name: value, ...})`, in any bidder hook. The run saves it automatically.

| Logged in | Saved in `log.logs[name][spec]` as | In `results.logs[name][spec]` as |
|---|---|---|
| `forward`, `update_batch`, `update` | (episodes, steps), one cell per auction | (seeds, episodes, steps) |
| `on_episode_start`, `on_episode_end` | (episodes,) | (seeds, episodes) |

Cells nobody logged are NaN. A child bidder's names carry its path, such as `"bidders.1.multiplier"`; pass `log_prefix=False` to keep the plain name. Outside a run, `self.log` does nothing. The built-in `PidPacing` and `UcbMultiplier` log their multiplier unless built with `log_state=False`.

A `bl.Callback` watches a run as it goes, without changing it. `run` calls `on_run_start`, `on_episode_end` and `on_run_end`. `run_many` calls `on_seed_end` after each seed, in seed order. Callbacks always run in the main process, also with workers.

```python
class Shaded(bl.Bidder):
    def __init__(self, factor=0.8):
        self.factor = factor

    def forward(self, value, observation):
        self.log({"factor": self.factor, "value_seen": value})
        return self.factor * value


class Progress(bl.Callback):
    def on_episode_end(self, episode, stats):
        print(f"episode {episode}: paced spent {stats['paced'].spend:.2f}")

    def on_seed_end(self, seed, results):
        print(f"seed {seed} done")


watched = bl.SimConfig(
    bidders=[
        bl.BidderSpec("paced", bl.PidPacing(), bl.Campaign(budget=10, value=5)),
        bl.BidderSpec(
            "shaded",
            bl.Stack(Shaded(), bl.PidPacing(log_state=False)),
            bl.Campaign(budget=10, value=5),
        ),
    ],
    episodes=3,
    steps=200,
)
log = bl.run(watched, callbacks=[Progress()])
print(log.logs["multiplier"]["paced"].shape)  # (3, 200)
print(log.logs["bidders.0.factor"]["shaded"][0, :3])  # [0.8 0.8 0.8]
results = bl.run_many(watched, seeds=range(4), callbacks=[Progress()])
print(results.logs["multiplier"]["paced"].shape)  # (4, 3, 200)
```

## Parallel runs

`run_many(config, seeds, workers=4)` runs seeds in 4 processes and gives the same rows as `workers=1`. Lambdas and closures work. Workers start with `spawn`, which re-imports your script, so put the call under a main guard:

```python
if __name__ == "__main__":
    parallel = bl.run_many(learning, seeds=range(4), workers=2)
    print(parallel.summary())
```

A bidder that holds a large object, such as a model, is copied to each worker and once per run.

## Metrics

`docs/metrics.md` defines every metric. Regret compares a bidder with the best fixed bid multiplier in hindsight, with other bids held fixed. Read its limits there before you compare methods.

## Reproducibility

Each source of randomness has its own seeded stream: the market, tie-breaking, conversions, each bidder's estimate error, and each bidder. A `Stack` splits its bidder's stream so each inner bidder gets its own. On one OS and CPU type, the same seed gives a bit-identical log. Other platforms round some floats differently, so their logs can differ in the last bits. Adding a bidder does not change the impressions the others see.
