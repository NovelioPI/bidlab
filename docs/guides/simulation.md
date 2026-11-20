# Simulation

This guide explains what happens inside a run: each step, the clock, estimate noise, parallel runs and reproducibility.

## One step in detail

The README lists the six parts of a step. These rules hold on top of them.

- A bid that is NaN or infinite counts as 0.
- A bid above the bidder's remaining budget is cut to that budget. The bidder does not see this; it tracks its own spend.
- One conversion draw happens at every step, even when nobody wins. This keeps runs repeatable when bidders change.
- Each bidder gets only its own `Feedback`. A loser never sees the winning price.

Budgets reset at the start of each episode, and each episode starts with `on_episode_start(campaign)`. Bidder objects persist for the whole run, so a learning bidder keeps what it learned.

## The clock

Auctions are spaced evenly over `SimConfig.horizon_hours`, 24 by default, from `SimConfig.start_hour`, 0 by default. `observation.time` is the number of seconds since the episode start.

Set the clock on `SimConfig`. The simulator copies it into each campaign before `on_episode_start`, so a bidder can ask its campaign:

| Method | Returns |
|---|---|
| `campaign.hour_of_day(time)` | The hour in [0, 24) at `time` |
| `campaign.time_left(time)` | Seconds left in the episode at `time` |

```python
import bidlab as bl


class OfficeHours(bl.Bidder):
    def on_episode_start(self, campaign):
        self.campaign = campaign

    def forward(self, value, observation):
        hour = self.campaign.hour_of_day(observation.time)
        return value if 9 <= hour < 17 else 0.0


office = bl.SimConfig(
    bidders=[bl.BidderSpec("office", OfficeHours(), bl.Campaign(budget=10, value=5))],
    episodes=2,
    steps=96,
    horizon_hours=12,
    start_hour=6,
)
log = bl.run(office)
print(log.bids[0, :12, 0])  # 0 until 9:00, which is step 24
```

## Estimate noise

`observation.value` is the platform's value: w times its estimate of p. The estimate is exact by default.

`SimConfig(estimate_noise=0.5)` adds normal noise with that standard deviation to logit(p). Use it to test how a bidder copes with a wrong value. Metrics always use the true value, in `RunLog.values`.

```python
import dataclasses

noisy = dataclasses.replace(office, estimate_noise=0.5)
print(bl.run(noisy).values.shape)  # (2, 96, 1)
```

## Parallel runs

`run_many(config, seeds, workers=4)` runs seeds in 4 processes and gives the same rows as `workers=1`. Lambdas and closures work.

Workers start with `spawn`, which imports your script again. Put the call under a `__main__` guard:

```python
if __name__ == "__main__":
    parallel = bl.run_many(office, seeds=range(4), workers=2)
    print(len(parallel))  # 4 rows: one per seed and bidder
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
