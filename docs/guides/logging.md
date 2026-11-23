# Logging and callbacks

This guide shows how a bidder logs its own values, and how a callback watches a run.

## Log from a bidder

Call `self.log(name, value)`, or `self.log({name: value, ...})`, in any bidder hook. The run saves it automatically.

| Logged in | Saved in `log.logged[name][campaign]` as | In `results.logged[name][campaign]` as |
|---|---|---|
| `forward`, `update_batch`, `update`, `update_conversions` | (episodes, steps), one cell per auction; delayed feedback logs at the step it arrives | (seeds, episodes, steps) |
| `on_episode_start`, `on_episode_end` | (episodes,) | (seeds, episodes) |

Cells nobody logged are NaN. Outside a run, `self.log` does nothing.

A child bidder's names carry its path, such as `"bidders.1.multiplier"`. Pass `log_prefix=False` to keep the plain name.

The built-in `PidPacing` and `UcbMultiplier` log their multiplier unless built with `log_state=False`.

```python
import bidlab as bl


class Shaded(bl.Bidder):
    def __init__(self, factor=0.8):
        self.factor = factor

    def forward(self, value):
        self.log({"factor": self.factor, "value_seen": value})
        return self.factor * value


watched = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "paced", bl.Stack(bl.OracleValue(), bl.PidPacing()), budget=10, base_value=5
        ),
        bl.Campaign(
            "shaded",
            bl.Stack(bl.OracleValue(), Shaded(), bl.PidPacing(log_state=False)),
            budget=10,
            base_value=5,
        ),
    ],
    episodes=3,
    steps=200,
)
log = bl.run(watched)
print(log.logged["bidders.1.multiplier"]["paced"].shape)  # (3, 200)
print(log.logged["bidders.1.factor"]["shaded"][0, :3])  # [0.8 0.8 0.8]
```

## Watch a run

A `bl.Callback` watches a run without changing it. Callbacks always run in the main process, also with workers.

| Hook | Called by | When |
|---|---|---|
| `on_run_start(config, seed)` | `run` with an int seed | Before the first episode |
| `on_step_end(episode, step)` | `run` with an int seed | After each step's auction; it runs once per auction, so keep it fast |
| `on_episode_end(episode, reports)` | `run` with an int seed | After each episode; `reports` maps each campaign name to its `EpisodeReport` |
| `on_run_end(log)` | `run` with an int seed | With the finished `RunLog` |
| `on_seeds_start(config, seeds)` | `run` with a seed list | Before the first seed |
| `on_seed_progress(seed, episode, step)` | `run` with a seed list | About every `steps // 20` steps of a seed's episode, and at its last step; before that seed's `on_seed_end` |
| `on_seed_end(seed, results)` | `run` with a seed list | After each seed, in seed order |
| `on_seeds_end(results)` | `run` with a seed list | With the `Results` it returns |

A seed list calls only the seed hooks, not the run hooks, with any number of workers.

```python
class Progress(bl.Callback):
    def on_episode_end(self, episode, reports):
        print(f"episode {episode}: paced spent {reports['paced'].spend:.2f}")

    def on_seed_end(self, seed, results):
        print(f"seed {seed} done")


bl.run(watched, callbacks=[Progress()])
results = bl.run(watched, seed=range(2), callbacks=[Progress()])
print(results.logged["bidders.1.multiplier"]["paced"].shape)  # (2, 3, 200)
```

`bl.ProgressBar()` is a built-in callback that shows tqdm bars on stderr. `run` adds one by default, after your callbacks; a `ProgressBar` you pass yourself is not doubled. With an int seed, one line counts the current episode's steps and starts again at each episode. With a seed list, a `seeds` line counts finished seeds, and each running seed has its own line below it, so there are at most `workers` such lines:

```text
seeds   2/5  40%|████      | [00:12<00:18]
seed 2 episode 1/3  45%|████▌     | 900/2000 [00:01<00:01, 1.6k step/s]
```

In a notebook it shows tqdm's widgets. Pass `progress_bar=False` to hide it:

```python
results = bl.run(watched, seed=range(2), progress_bar=False)
```

## Examples

| Script | Question |
|---|---|
| [logging/pid_trace.py](../../examples/logging/pid_trace.py) | Does PID pacing spend its whole budget in each episode? |
