# Simulation

This guide explains what happens inside a run: each step, the clock, feedback delivery, parallel runs and reproducibility.

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

Auctions are spaced evenly over `SimConfig.horizon_hours`, 24 by default, from `SimConfig.start_hour`, 0 by default. A market can set each step's time instead, as `ReplayMarket` does from a log's time column.

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
    campaigns=[
        bl.Campaign(
            "office", bl.Stack(bl.OracleValue(), OfficeHours()), budget=10, base_value=5
        )
    ],
    episodes=2,
    steps=96,
    horizon_hours=12,
    start_hour=6,
)
log = bl.run(office)
print(log.bids[0, :12, 0])  # 0 until 9:00, which is step 24
```

## Feedback delivery

By default, each bidder gets its `BidResult` right after its auction, with the click and the conversion in it. Real bidders learn later: conversions arrive hours after the click, and reports arrive in batches. Set this with `SimConfig(feedback=bl.Feedback(...))`. Every setting is in seconds and defaults to 0.

| Setting | Meaning |
|---|---|
| `interval` | Seconds between deliveries; 0 delivers after every auction |
| `report_delay` | Seconds from an auction until its `BidResult` is due |
| `conversion_delay` | Seconds from an auction until its conversion is due, never before its `BidResult` |

```python
late = bl.SimConfig(
    campaigns=office.campaigns,
    feedback=bl.Feedback(interval=900, report_delay=120, conversion_delay=6 * 3600),
)
```

The delivery rule:

1. An item is due at its auction's `time` plus its delay.
2. A delivery step is every step when `interval` is 0. Otherwise it is the first step at or after each multiple of `interval`.
3. After the auction of a delivery step, each bidder gets every item now due. `update_batch` gets the results first, then `update_conversions` gets the conversions.
4. After an episode's last auction, everything still pending arrives, before `on_episode_end`. So `EpisodeReport` and the delivered items agree.

Under a `conversion_delay`, `BidResult.converted` is always False, and a `conversion` campaign's `BidResult.realized_value` is 0. The conversion arrives later as a `bl.Conversion(time, realized_value)`, whose `time` names the auction. ⚠️ A learner that reads only `result.converted` sees no conversion at all. Override `update_conversions` to count them.

Under a `report_delay`, the `budget_left` input counts only the spend in delivered results, so a bidder that trusts it can bid as if it had more money. The simulator still caps each bid at the true budget left, so spend never passes the budget.

Delays draw no random numbers: a run whose bidders ignore feedback gives the same log with or without them.

## Parallel runs

`run(config, seed=range(8), workers=4)` runs the seeds in 4 processes and gives the same rows as `workers=1`. `workers` needs a seed list: with an int seed it raises `ValueError`. Lambdas and closures work.

Workers start with `spawn`, which imports your script again. Put the call under a `__main__` guard:

```python
if __name__ == "__main__":
    parallel = bl.run(office, seed=range(4), workers=2)
    print(len(parallel))  # 4 rows: one per seed and campaign
```

A bidder that holds a large object, such as a model, is copied to each worker and once per run.

To see where a run spends its time, pass `profiler=True` to `run`. [Profiling](profiling.md) explains the table it gives.

## Reproducibility

Each source of randomness has its own seeded stream: the market, tie-breaking, conversions, and each bidder. A `Stack` splits its bidder's stream, so each inner bidder gets its own.

`run(config, seed=7)` seeds every stream from 7; the seed defaults to 0 and is not part of `SimConfig`, so one config runs under any seed. `RunLog.seed` records it. On one OS and CPU type, the same seed gives a bit-identical log. Other platforms round some floats differently, so the last bits can differ. Adding a bidder does not change the impressions the others see.

## Examples

| Script | Question |
|---|---|
| [getting_started/basic.py](../../examples/getting_started/basic.py) | When the budget is tight, does PID pacing earn more surplus than truthful bidding? |
| [simulation/clock.py](../../examples/simulation/clock.py) | With a 12-hour day from 9:00, does an even-schedule bidder still have budget at 20:00, when truthful bidding has run out? |
| [simulation/delayed_conversions.py](../../examples/simulation/delayed_conversions.py) | When conversions arrive 6 hours late, does a learner that counts each unconverted click as a miss bid lower? |
