# Profiling

This guide shows how much time each bidder hook and the simulator take, and how much memory a run uses.

## Profile a run

Pass `profiler=True` to `bl.run`. The log then holds a `bl.Profile` in `log.profile`:

```python
import bidlab as bl

config = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "pid", bl.Stack(bl.OracleValue(), bl.PidPacing()), budget=10, base_value=5
        )
    ],
    episodes=2,
    steps=500,
)
log = bl.run(config, profiler=True)
print(log.profile.summary())
```

The table has one line per campaign, bidder node and hook that was called, slowest first:

```text
campaign  node       hook              calls   total_s    self_s  mean_us  share
pid       (top)      forward            1000  0.003265  0.001689      3.3  11.0%
pid       bidders.1  update_batch       1000  0.002699  0.000353      2.7   9.1%
pid       bidders.1  update             1000  0.002345  0.002345      2.3   7.9%
pid       bidders.1  forward            1000  0.001372  0.001372      1.4   4.6%
pid       bidders.0  forward            1000  0.000204  0.000204      0.2   0.7%
...
wall 0.030 s, 33,615 steps/s, simulator 0.024 s (79.9%), memory peak 91.7 MiB (+0.0 MiB over start)
```

| Column | Meaning |
|---|---|
| `node` | The node's path in the bidder tree, as in `Bidder.named_bidders`; `(top)` is the campaign's bidder |
| `hook` | `forward`, `update_batch`, `update`, `update_conversions`, `on_episode_start` or `on_episode_end` |
| `calls` | How many times the hook ran |
| `total_s` | Seconds from each call's start to its end, summed |
| `self_s` | `total_s` minus the time of the hooks this call ran, in seconds |
| `mean_us` | `total_s` per call, in microseconds |
| `share` | `total_s` as a share of the run's wall time |

The last line gives the run's wall time, the steps per second, the simulator's time and share, and the memory.

Each row is also in `log.profile.rows` as a `HookTime(campaign, path, hook, calls, total, self_time)`, to compare or plot.

## Self time

A hook's total time includes every hook it calls. A `Stack`'s `forward` calls each child's `forward`, and the default `update_batch` calls `update` once per result. So `total_s` counts a child's time in its parent too, and shares can add up to more than 100%.

Self time leaves the children out. A hook's self time is its total minus the totals of the hooks it called, on any node. Each second inside a bidder counts once, in the hook that was running at that moment. A `Stack`'s self time is the cost of passing inputs from one child to the next.

`profile.bidder_time` is the sum of every row's self time. `profile.simulator_time` is the rest of `profile.wall_time`: the market, the auction, feedback and logging.

The profiler's own timers add about 0.1 µs to each call it times, and the table includes that time. A profiled run is about 15% slower. Without the profiler, nothing is timed, so a run keeps its speed.

## Memory

The profiler reads the process's resident memory with `psutil` at the run start, after each episode and at the run end. `profile.memory_start` is the first sample and `profile.memory_peak` the highest, both in bytes. The growth `memory_peak - memory_start` is what the run added.

Memory is sampled, not traced, so an object made and freed within one episode can be missed. Memory per bidder is not measured.

## Many seeds

`bl.run_many(config, seeds, profiler=True)` profiles each seed's run. `results.profile` holds one `Profile` per seed, in seed order:

```python
results = bl.run_many(config, seeds=range(3), profiler=True)
for profile in results.profile:
    print(f"{profile.steps / profile.wall_time:,.0f} steps/s")
```

Each seed's memory comes from the process that ran it. With `workers=1`, every seed runs in one process. So a later seed's `memory_start` includes what earlier seeds left, but the growth stays per seed.

## Save a profile

`profile.save_csv(path)` writes one line per row, in table order. `results.save_profile_csv(path)` writes every seed's lines to one file, in seed order:

```python
log.profile.save_csv("profile.csv")
results.save_profile_csv("profiles.csv")
```

| Column | Meaning |
|---|---|
| `seed` | The run's `config.seed`, also in `profile.seed` |
| `campaign`, `path`, `hook`, `calls` | As in `HookTime`; `path` is empty for the top bidder |
| `total_seconds`, `self_seconds` | The row's total and self time |
| `wall_seconds`, `steps`, `memory_start`, `memory_peak` | The run totals, repeated on every line of the run |

Times keep every digit, so reading the file back gives the exact numbers. `save_profile_csv` raises `ValueError` when `run_many` ran without `profiler=True`.

## Profiles stay out of the results

Timings change from run to run, so a profile never changes anything else. Every array of a profiled log equals the unprofiled one bit for bit. `RunLog`'s `==` and `Results.equals` ignore profiles, so a profiled run equals an unprofiled one. Without `profiler=True`, `log.profile` and `results.profile` are None.

## Examples

| Script | Question |
|---|---|
| [simulation/profiling.py](../../examples/simulation/profiling.py) | Is UcbMultiplier's update slower per call than PidPacing's? |
