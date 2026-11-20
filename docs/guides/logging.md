# Logging and callbacks

This guide shows how a bidder logs its own values during a run, and how a callback watches a run as it goes.

## Log from a bidder

Call `self.log(name, value)`, or `self.log({name: value, ...})`, in any bidder hook. The run saves it automatically.

| Logged in | Saved in `log.logs[name][spec]` as | In `results.logs[name][spec]` as |
|---|---|---|
| `forward`, `update_batch`, `update` | (episodes, steps), one cell per auction | (seeds, episodes, steps) |
| `on_episode_start`, `on_episode_end` | (episodes,) | (seeds, episodes) |

Cells nobody logged are NaN. Outside a run, `self.log` does nothing.

A child bidder's names carry its path, such as `"bidders.1.multiplier"`. Pass `log_prefix=False` to keep the plain name.

The built-in `PidPacing` and `UcbMultiplier` log their multiplier unless built with `log_state=False`.

```python
import bidlab as bl


class Shaded(bl.Bidder):
    def __init__(self, factor=0.8):
        self.factor = factor

    def forward(self, value, observation):
        self.log({"factor": self.factor, "value_seen": value})
        return self.factor * value


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
log = bl.run(watched)
print(log.logs["multiplier"]["paced"].shape)  # (3, 200)
print(log.logs["bidders.0.factor"]["shaded"][0, :3])  # [0.8 0.8 0.8]
```

## Watch a run

A `bl.Callback` watches a run without changing it. Callbacks always run in the main process, also with workers.

| Hook | Called by | When |
|---|---|---|
| `on_run_start(config)` | `run` | Before the first episode |
| `on_episode_end(episode, stats)` | `run` | After each episode; `stats` maps each spec name to its `EpisodeStats` |
| `on_run_end(log)` | `run` | With the finished `RunLog` |
| `on_seed_end(seed, results)` | `run_many` | After each seed, in seed order |

```python
class Progress(bl.Callback):
    def on_episode_end(self, episode, stats):
        print(f"episode {episode}: paced spent {stats['paced'].spend:.2f}")

    def on_seed_end(self, seed, results):
        print(f"seed {seed} done")


bl.run(watched, callbacks=[Progress()])
results = bl.run_many(watched, seeds=range(2), callbacks=[Progress()])
print(results.logs["multiplier"]["paced"].shape)  # (2, 3, 200)
```

## Examples

| Script | Question |
|---|---|
| [logging/pid_trace.py](../../examples/logging/pid_trace.py) | Does PID pacing spend its whole budget in each episode? |
