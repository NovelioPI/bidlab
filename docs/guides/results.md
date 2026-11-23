# Results

This guide shows what `run` and `run_many` return, and how to read and compare the metrics.

## Results from `run_many`

`run_many` returns a `Results` object with one row per seed and campaign. Each row holds every metric, averaged over episodes. [Metrics](../metrics.md) defines each one.

| Method | Gives |
|---|---|
| `results.summary()` | A text table of means and 95% Student-t intervals |
| `print(results)`, or `results` in a notebook | The row count and seed count, then the summary table |
| `results.save_csv(path)` | A CSV file with one row per seed and campaign |
| `results.to_pandas()` | A DataFrame; needs `uv sync --extra pandas` |
| `results.rows`, `results.columns` | The raw rows as dicts |
| `results.equals(other)` | `True` when two runs gave the same rows and logs |
| `results.logged` | Values bidders logged, per seed; see [Logging](logging.md) |
| `results.segment_rows` | The per-segment table, also a `Results` |

Compare with `equals`, not `==`. Some metrics are NaN by definition, and NaN never equals NaN. `equals` on the main table does not compare `results.segment_rows`.

```python
import bidlab as bl

config = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "truthful", bl.Stack(bl.OracleValue(), bl.Linear()), budget=15, base_value=5
        ),
        bl.Campaign(
            "pid", bl.Stack(bl.OracleValue(), bl.PidPacing()), budget=15, base_value=5
        ),
    ],
    episodes=2,
    steps=200,
)
results = bl.run_many(config, seeds=range(3))
print(results.equals(bl.run_many(config, seeds=range(3))))  # True
print(results.columns[:4])  # ('seed', 'campaign', 'spend', 'wins')
```

## The log from `run`

`run` returns a `RunLog` for one seed. Its arrays are read-only.

| Field | Shape | Holds |
|---|---|---|
| `bids`, `true_values` | (episodes, steps, campaigns) | Bids after capping, true values |
| `won`, `price`, `clicked`, `converted` | (episodes, steps, campaigns) | Whether each campaign won, the price it paid (0 if it lost), its click and its conversion |
| `segment`, `market_price`, `floor` | (episodes, steps) | The segment index, the market price (NaN if none), the market floor |
| `logged` | per name and campaign | Values bidders logged, such as their own rewards |
| `config` | | The `SimConfig` that made it |

Four functions turn a `RunLog` into metrics, one row per episode:

| Function | Rows |
|---|---|
| `bl.campaign_metrics(log)` | Per episode and bidder |
| `bl.segment_metrics(log)` | Per episode, bidder and segment |
| `bl.platform_metrics(log)` | Per episode: the seller's revenue and welfare |
| `bl.regret(log, episode, bidder)` | One number: best fixed multiplier's surplus minus the bidder's |

```python
log = bl.run(config)
print(log.bids.shape)  # (2, 200, 2)
print(len(bl.campaign_metrics(log)))  # 4 rows: episodes × bidders
```

Regret compares a bidder with the best fixed bid multiplier in hindsight, with other bids held fixed. Read its limits in [Metrics](../metrics.md) before you compare methods.

## Examples

| Script | Question |
|---|---|
| [getting_started/basic.py](../../examples/getting_started/basic.py) | When the budget is tight, does PID pacing earn more surplus than truthful bidding? |
| [results/regret.py](../../examples/results/regret.py) | With no budget limit in a second-price auction, is truthful bidding's regret zero? |
| [results/reserve.py](../../examples/results/reserve.py) | Does a reserve price of 0.1 raise the platform's revenue? |
