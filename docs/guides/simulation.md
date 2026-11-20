# Simulation

This guide explains what happens inside a run: each step, the clock, estimate noise, parallel runs and reproducibility.

## One step in detail

The README lists the six parts of a step. These rules hold on top of them.

- A bid that is NaN or infinite counts as 0.
- A bid above the bidder's remaining budget is cut to that budget. The bidder does not see this; it tracks its own spend.
- One conversion draw happens at every step, even when nobody wins. This keeps runs repeatable when bidders change.
- A market with a click stage also gets one click draw per step, from a separate stream. A market without one draws no click, so its runs keep every number.
- Only a clicked impression can convert.
- Each bidder gets only its own `BidResult`. A loser never sees the winning price.

Budgets reset at the start of each episode, and each episode starts with `on_episode_start(campaign)`. Bidder objects persist for the whole run, so a learning bidder keeps what it learned.

## The clock

Auctions are spaced evenly over `SimConfig.horizon_hours`, 24 by default, from `SimConfig.start_hour`, 0 by default.

Set the clock on `SimConfig`. A bidder reads it through the `forward` inputs it names:

| Input | Meaning |
|---|---|
| `time` | Seconds since the episode start |
| `time_left` | Seconds left in the episode |
| `horizon` | Episode length in seconds |
| `hour_of_day` | The hour in [0, 24) at `time` |

```python
import bidlab as bl


class OfficeHours(bl.Bidder):
    def forward(self, value, hour_of_day):
        return value if 9 <= hour_of_day < 17 else 0.0


office = bl.SimConfig(
    campaigns=[bl.Campaign("office", OfficeHours(), budget=10, event_value=5)],
    episodes=2,
    steps=96,
    horizon_hours=12,
    start_hour=6,
)
log = bl.run(office)
print(log.bids[0, :12, 0])  # 0 until 9:00, which is step 24
```

## Estimate noise

The `value` input is the platform's value: w times its estimate of pCTR × pCVR. The estimate is exact by default.

`SimConfig(estimate_noise=0.5)` adds normal noise with that standard deviation to logit(pCVR), and, with a click stage, an independent error of the same size to logit(pCTR). Use it to test how a bidder copes with a wrong value. Metrics always use the true value, in `RunLog.true_values`, and the `market` input always holds the true rates.

```python
import dataclasses

noisy = dataclasses.replace(office, estimate_noise=0.5)
print(bl.run(noisy).true_values.shape)  # (2, 96, 1)
```

## Parallel runs

`run_many(config, seeds, workers=4)` runs seeds in 4 processes and gives the same rows as `workers=1`. Lambdas and closures work.

Workers start with `spawn`, which imports your script again. Put the call under a `__main__` guard:

```python
if __name__ == "__main__":
    parallel = bl.run_many(office, seeds=range(4), workers=2)
    print(len(parallel))  # 4 rows: one per seed and campaign
```

A bidder that holds a large object, such as a model, is copied to each worker and once per run.

## Reproducibility

Each source of randomness has its own seeded stream: the market, tie-breaking, conversions, each bidder's estimate error, and each bidder. A `Stack` splits its bidder's stream, so each inner bidder gets its own.

On one OS and CPU type, the same seed gives a bit-identical log. Other platforms round some floats differently, so the last bits can differ. Adding a bidder does not change the impressions the others see.

## Examples

| Script | Question |
|---|---|
| [getting_started/basic.py](../../examples/getting_started/basic.py) | When the budget is tight, does PID pacing earn more surplus than truthful bidding? |
| [simulation/estimate_noise.py](../../examples/simulation/estimate_noise.py) | Does noise in the platform's value cost a truthful bidder surplus? |
| [simulation/clock.py](../../examples/simulation/clock.py) | With a 12-hour day from 9:00, does an even-schedule bidder still have budget at 20:00, when truthful bidding has run out? |
