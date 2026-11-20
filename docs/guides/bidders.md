# Bidders

This guide shows how to write a bidder, stack bidders, and use the built-in ones.

## The idea

A bidder turns a value into another value. Alone, it gets the platform's value, `request.value`, and its output is the bid.

Combine bidders with `bl.Stack`: each gets the previous one's output, and the last one's output is the bid. Two kinds of bidder fit together this way:

| Kind | Example | Turns |
|---|---|---|
| Value model | A conversion-rate model | Features into its own value; it may ignore the incoming one |
| Bid control | Pacing, shading | A value into a bid |

Bidders in a stack run in order, so pacing then shading differs from shading then pacing.

## Hooks

Subclass `bl.Bidder` and implement `forward`. Every other hook is optional.

| Hook | Called |
|---|---|
| `on_episode_start(brief)` | At the start of each episode, with the campaign's `Brief` and its clock |
| `forward(value, request)` | At each auction the bidder takes part in; returns the bid or value |
| `update_batch(results)` | After each step, with that step's `BidResult` list |
| `update(result)` | Once per result, if you do not override `update_batch` |
| `on_episode_end(report, episode_reward)` | At the end of each episode |

A bidder keeps what it learns on `self`. Take settings as constructor arguments. Each bidder gets its own random stream as `self.rng`.

The simulator deep-copies the bidder for each run. Every seed starts from the bidder as you built it, and a pretrained model passes in as is.

## A value model before pacing

`BidResult` carries the auction's `time` but not its features. To learn from features, keep them by time in `forward`, then read them back in `update_batch`.

```python
import bidlab as bl


class SegmentRate(bl.Bidder):
    def __init__(self, prior=0.05):
        self.prior = prior
        self.wins = {}
        self.conversions = {}
        self.pending = {}

    def on_episode_start(self, brief):
        self.brief = brief

    def forward(self, value, request):
        segment = request.features["segment"]
        self.pending[request.time] = segment
        wins = self.wins.get(segment, 0)
        rate = (self.conversions.get(segment, 0) + self.prior) / (wins + 1)
        return rate * self.brief.event_value

    def update_batch(self, bid_results):
        for result in bid_results:
            segment = self.pending.pop(result.time)
            if result.won:
                self.wins[segment] = self.wins.get(segment, 0) + 1
                self.conversions[segment] = (
                    self.conversions.get(segment, 0) + result.converted
                )


market = bl.markets.Segmented(
    (bl.markets.Segment("news", 0.5), bl.markets.Segment("sports", 0.5, lift=1.0))
)
learning = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "learned",
            bl.Stack(SegmentRate(), bl.PidPacing()),
            budget=15,
            event_value=5,
        ),
        bl.Campaign("platform", bl.PidPacing(), budget=15, event_value=5),
    ],
    market=market,
    episodes=3,
    steps=200,
)
print(len(bl.run_many(learning, seeds=range(2))))  # 4 rows
```

## A bidder that uses randomness

Draw only from `self.rng`, so runs repeat with the seed.

```python
class Jitter(bl.Bidder):
    def __init__(self, spread=0.1):
        self.spread = spread

    def forward(self, value, request):
        return value * self.rng.uniform(1 - self.spread, 1 + self.spread)


jitter = bl.Campaign(
    "jitter", bl.Stack(Jitter(), bl.PidPacing()), budget=15, event_value=5
)
```

## Built-in bidders

| Class | Behavior | Arguments |
|---|---|---|
| `Linear` | Outputs `multiplier` × value | `multiplier` (1.0) |
| `PidPacing` | A PID loop moves a multiplier so spend tracks the budget evenly over the horizon; logs `"multiplier"` per auction | `initial_multiplier` (1.0), `kp` (20), `ki` (2), `kd` (0), `multiplier_range` ((0, 10)), `log_state` (True) |
| `UcbMultiplier` | UCB1 over a grid of multipliers; one arm per episode; logs its arm as `"multiplier"` per episode | `min_multiplier` (0.2), `max_multiplier` (2.0), `multiplier_step` (0.2), `exploration` (1.0), `log_state` (True) |

They live in `bl.bidders.baselines`, `bl.bidders.pacing` and `bl.bidders.bandits`, and `bl` exports each one.

A bidder can also hold other bidders and call them itself. See [Composite bidders](composite-bidders.md).

## Examples

| Script | Question |
|---|---|
| [bidders/segment_rate.py](../../examples/bidders/segment_rate.py) | Does a bidder that counts conversions per segment learn that sports converts better than news? |
