# Bidders

This guide shows how to write a bidder, stack bidders, estimate values, and use the built-in ones.

## The idea

A bidder turns its inputs into a number: the bid, or a value for the next bidder. The simulator passes no value; each bidder estimates its own.

Combine bidders with `bl.Stack`: the first gets the inputs, each later one also gets the previous one's output as `value`, and the last one's output is the bid. Two kinds of bidder fit together this way:

| Kind | Example | Turns |
|---|---|---|
| Value estimator | `bl.OracleValue`, a conversion-rate model | Inputs into a value |
| Bid control | Pacing, shading | A value into a bid |

`bl.Stack(bl.OracleValue(), bl.PidPacing())` paces the true value. Swap `OracleValue` for your own estimator to test how its errors cost the campaign.

Bidders in a stack run in order, so pacing then shading differs from shading then pacing.

## Hooks

Subclass `bl.Bidder` and implement `forward`. Every other hook is optional.

| Hook | Called |
|---|---|
| `on_episode_start(campaign)` | At the start of each episode, with the `Campaign`: budget, event, base value, CPA target, segments |
| `forward(...)` | At each auction the bidder takes part in, with the inputs it names; returns the bid or value |
| `update_batch(results)` | After each step, with that step's `BidResult` list |
| `update(result)` | Once per result, if you do not override `update_batch` |
| `on_episode_end(report)` | At the end of each episode, with its `EpisodeReport` |

A bidder keeps what it learns on `self`. Take settings as constructor arguments. Each bidder gets its own random stream as `self.rng`.

The simulator deep-copies the bidder for each run. Every seed starts from the bidder as you built it, and a pretrained model passes in as is. The `campaign.bidder` passed to `on_episode_start` is that running copy, not your object.

A bidder defines its own reward. Compute it in `update` or `on_episode_end` and record it with `self.log`; see [Rewards](rewards.md).

## Inputs

Like a PyTorch module, `forward` declares the inputs it needs as parameters. The simulator passes the top bidder only those, by name.

| Input | Meaning |
|---|---|
| `value` | In a `Stack`, the previous bidder's output; the simulator does not pass it |
| `time` | Seconds since the episode start |
| `time_left` | Seconds left in the episode |
| `horizon` | Episode length in seconds |
| `hour_of_day` | Hour of day at `time`, in [0, 24) |
| `budget_left` | The campaign's budget left before this auction |
| `segment` | The auction's market segment; `"all"` without segments |
| `floor` | The lowest bid that can win: the higher of the auction's `reserve` and the market's floor |
| `features` | Every public feature from the market, as a read-only mapping |
| `market` | A `bl.MarketTruth` with this campaign's true `pctr` and `pcvr`; for checks and oracles, not for a fair benchmark |
| a feature name | Each public feature from the market, such as `country`, or the array `vector` from `LogisticMarket` |

```python
import bidlab as bl


class EvenPacer(bl.Bidder):
    def on_episode_start(self, campaign):
        self.budget = campaign.budget

    def forward(self, value, time_left, horizon, budget_left):
        # Bid only while spend is behind an even plan over the episode.
        behind = budget_left / self.budget >= time_left / horizon
        return value if behind else 0.0


paced = bl.Stack(bl.OracleValue(), EvenPacer())
```

- A `forward` with `**inputs` gets every input.
- A parameter with a default gets its default when the input is missing.
- A parameter without a default and without an input raises `TypeError` at that auction, listing the inputs offered.
- A market feature with the name of a built-in input, or `value`, raises `ValueError`.
- A `Stack` passes each of its bidders the inputs it names, with `value` set to the previous output. Its first bidder gets no `value`, so a bid control such as `bl.Linear()` cannot come first; it raises `TypeError` that names `bl.OracleValue`.

## A value model before pacing

`BidResult` carries the auction's `time` but not its inputs. To learn from an input, keep it by time in `forward`, then read it back in `update_batch`.

```python
class SegmentRate(bl.Bidder):
    def __init__(self, prior=0.05):
        self.prior = prior
        self.wins = {}
        self.conversions = {}
        self.pending = {}

    def on_episode_start(self, campaign):
        self.base_value = campaign.base_value

    def forward(self, segment, time):
        self.pending[time] = segment
        wins = self.wins.get(segment, 0)
        rate = (self.conversions.get(segment, 0) + self.prior) / (wins + 1)
        return rate * self.base_value

    def update_batch(self, bid_results):
        for result in bid_results:
            segment = self.pending.pop(result.time)
            if result.won:
                self.wins[segment] = self.wins.get(segment, 0) + 1
                self.conversions[segment] = (
                    self.conversions.get(segment, 0) + result.converted
                )


market = bl.markets.Segmented(
    (bl.markets.Segment("news", 0.5), bl.markets.Segment("sports", 0.5, cvr_lift=1.0))
)
learning = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "learned",
            bl.Stack(SegmentRate(), bl.PidPacing()),
            budget=15,
            base_value=5,
        ),
        bl.Campaign(
            "oracle",
            bl.Stack(bl.OracleValue(), bl.PidPacing()),
            budget=15,
            base_value=5,
        ),
    ],
    market=market,
    episodes=3,
    steps=200,
)
print(len(bl.run_many(learning, seeds=range(2))))  # 4 rows
```

## Value estimators

A value estimator is a bidder: the first stage of a `Stack`, before a bid control. It outputs `base_value` times its own estimate of the chance of the campaign's event. Four common kinds:

| Estimator | Estimate |
|---|---|
| Oracle | The true chance, from the `market` input: `bl.OracleValue` |
| Fixed | One rate for every impression |
| Counting | Conversions per win so far |
| Model | A model on the public features, such as one logistic model per stage |

```python
import numpy as np


class FixedRate(bl.Bidder):
    def __init__(self, rate, base_value):
        self.rate = rate
        self.base_value = base_value

    def forward(self):
        return self.rate * self.base_value


class CountedRate(bl.Bidder):
    def __init__(self, base_value, prior=0.05):
        self.base_value = base_value
        self.prior = prior
        self.wins = 0
        self.conversions = 0

    def forward(self):
        return (self.conversions + self.prior) / (self.wins + 1) * self.base_value

    def update(self, result):
        self.wins += result.won
        self.conversions += result.converted


class ConversionModel(bl.Bidder):
    """Online logistic regression of a conversion on the public feature vector."""

    def __init__(self, base_value, feature_count, learning_rate=0.05):
        self.base_value = base_value
        self.weights = np.zeros(feature_count)
        self.learning_rate = learning_rate
        self.pending = {}

    def predict(self, vector):
        return 1.0 / (1.0 + np.exp(-(self.weights @ vector) + 3.0))

    def forward(self, features, time):
        vector = features["vector"]
        self.pending[time] = vector
        return self.predict(vector) * self.base_value

    def update(self, result):
        vector = self.pending.pop(result.time)
        if result.won:
            error = result.converted - self.predict(vector)
            self.weights += self.learning_rate * error * vector


estimated = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            name,
            bl.Stack(estimator, bl.PidPacing()),
            budget=15,
            base_value=5,
        )
        for name, estimator in (
            ("fixed", FixedRate(0.05, base_value=5)),
            ("counted", CountedRate(base_value=5)),
            ("model", ConversionModel(base_value=5, feature_count=8)),
        )
    ],
    episodes=3,
    steps=200,
)
print(len(bl.run_many(estimated, seeds=range(2))))  # 6 rows
```

`bl.OracleValue(noise=0.5)` adds normal noise with that standard deviation to logit(pCTR) and logit(pCVR), from its own random stream. Only the stages the campaign's event uses enter its value, and a pCTR of 1, from a market without a click stage, stays exact. Use it to test how a bid control copes with a known error. Metrics always use the true value, in `RunLog.true_values`.

With a click stage, a model can learn each stage apart: clicks from every won impression, conversions from clicked ones, using `result.clicked`. The example [bidders/value_estimators.py](../../examples/bidders/value_estimators.py) does this and compares the three estimators with the market's truth.

## A bidder that uses randomness

Draw only from `self.rng`, so runs repeat with the seed.

```python
class Jitter(bl.Bidder):
    def __init__(self, spread=0.1):
        self.spread = spread

    def forward(self, value):
        return value * self.rng.uniform(1 - self.spread, 1 + self.spread)


jitter = bl.Campaign(
    "jitter",
    bl.Stack(bl.OracleValue(), Jitter(), bl.PidPacing()),
    budget=15,
    base_value=5,
)
```

## Built-in bidders

| Class | Behavior | Arguments |
|---|---|---|
| `OracleValue` | Outputs the true value, `base_value` × the chance of the campaign's event, from the `market` input; with `noise`, a noisy estimate | `noise` (0.0) |
| `Linear` | Outputs `multiplier` × value | `multiplier` (1.0) |
| `PidPacing` | A PID loop moves a multiplier so spend tracks the budget evenly over the horizon; logs `"multiplier"` per auction | `initial_multiplier` (1.0), `kp` (20), `ki` (2), `kd` (0), `multiplier_range` ((0, 10)), `log_state` (True) |
| `UcbMultiplier` | UCB1 over a grid of multipliers; one arm per episode, rewarded by the episode's surplus; override `update` for another reward; logs its arm as `"multiplier"` per episode | `min_multiplier` (0.2), `max_multiplier` (2.0), `multiplier_step` (0.2), `exploration` (1.0), `log_state` (True) |

They live in `bl.bidders.baselines`, `bl.bidders.pacing` and `bl.bidders.bandits`, and `bl` exports each one. Every one but `OracleValue` needs a `value`, so it goes after an estimator in a `Stack`. There, its logged names carry its path, such as `"bidders.1.multiplier"`.

A bidder can also hold other bidders and call them itself. See [Composite bidders](composite-bidders.md).

## Examples

| Script | Question |
|---|---|
| [bidders/segment_rate.py](../../examples/bidders/segment_rate.py) | Does a bidder that counts conversions per segment learn that sports converts better than news? |
| [bidders/value_estimators.py](../../examples/bidders/value_estimators.py) | Does a two-stage model on the public features estimate conversions better than counting? |
| [bidders/oracle_noise.py](../../examples/bidders/oracle_noise.py) | Does noise in the value estimate cost a truthful bidder surplus? |
