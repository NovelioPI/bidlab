"""Does PID pacing spend its whole budget in each episode?

A callback watches each episode end and records how much of its budget PID
spent. PID logs its multiplier at every auction, so the run also shows how the
multiplier moves over the day.
"""

import numpy as np

import bidlab as bl

FULL_SPEND = 0.95
BLOCK_HOURS = 4


class SpendShare(bl.Callback):
    def __init__(self):
        self.shares = []

    def on_run_start(self, config):
        print(f"run: {config.episodes} episodes of {config.steps} auctions")

    def on_episode_end(self, episode, stats):
        pid = stats["pid"]
        self.shares.append(pid.spend / pid.budget)
        print(
            f"episode {episode}: spent {pid.spend:.2f} of {pid.budget:.0f},"
            f" {pid.wins} wins, {pid.conversions} conversions worth {pid.value:.1f}"
        )

    def on_run_end(self, log):
        print(f"stored names: {sorted(log.logs)}")

    def on_seed_end(self, seed, results):
        print(f"seed {seed} done")


config = bl.SimConfig(
    bidders=[
        bl.BidderSpec("pid", bl.PidPacing(), bl.Campaign(budget=15, value=5)),
        bl.BidderSpec(
            "quiet", bl.PidPacing(log_state=False), bl.Campaign(budget=15, value=5)
        ),
        bl.BidderSpec("big", bl.Truthful(), bl.Campaign(budget=50, value=5)),
    ],
    episodes=5,
    steps=480,
)

if __name__ == "__main__":
    watcher = SpendShare()
    log = bl.run(config, callbacks=[watcher])
    multiplier = log.logs["multiplier"]["pid"]
    per_block = int(config.steps * BLOCK_HOURS / config.horizon_hours)
    blocks = multiplier[-1].reshape(-1, per_block).mean(axis=1)
    print(f"last-episode multiplier per {BLOCK_HOURS} hours: {np.round(blocks, 2)}")

    results = bl.run_many(config, seeds=range(3), callbacks=[watcher])
    print(f"multiplier over seeds: {results.logs['multiplier']['pid'].shape}")
    print(results.summary())

    shares = watcher.shares
    answer = "yes" if min(shares) >= FULL_SPEND else "no"
    print(f"Answer: {answer}, the lowest episode spent {min(shares):.0%} of its budget")
