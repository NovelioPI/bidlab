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
        names = [campaign.name for campaign in config.campaigns]
        print(f"run: {names}, {config.episodes} episodes of {config.steps} auctions")

    def on_episode_end(self, episode, reports):
        pid = reports["pid"]
        self.shares.append(pid.spend / pid.budget)
        print(
            f"episode {episode}: spent {pid.spend:.2f} of {pid.budget:.0f},"
            f" {pid.wins} wins, {pid.conversions} conversions worth {pid.realized_value:.1f}"
        )

    def on_run_end(self, log: bl.RunLog):
        print(f"stored names: {sorted(log.logged)}")

    def on_seed_end(self, seed: int, results: bl.Results):
        print(f"seed {seed} done")


config = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "pid", bl.Stack(bl.OracleValue(), bl.PidPacing()), budget=15, base_value=5
        ),
        bl.Campaign(
            "quiet",
            bl.Stack(bl.OracleValue(), bl.PidPacing(log_state=False)),
            budget=15,
            base_value=5,
        ),
        bl.Campaign(
            "big", bl.Stack(bl.OracleValue(), bl.Linear()), budget=50, base_value=5
        ),
    ],
    episodes=5,
    steps=480,
)

if __name__ == "__main__":
    watcher = SpendShare()
    log = bl.run(config, callbacks=[watcher])
    multiplier = log.logged["bidders.1.multiplier"]["pid"]
    per_block = int(config.steps * BLOCK_HOURS / config.horizon_hours)
    blocks = multiplier[-1].reshape(-1, per_block).mean(axis=1)
    print(f"last-episode multiplier per {BLOCK_HOURS} hours: {np.round(blocks, 2)}")

    results = bl.run_many(config, seeds=range(3), callbacks=[watcher])
    print(
        f"multiplier over seeds: {results.logged['bidders.1.multiplier']['pid'].shape}"
    )
    print(results.summary())

    shares = watcher.shares
    answer = "yes" if min(shares) >= FULL_SPEND else "no"
    print(f"Answer: {answer}, the lowest episode spent {min(shares):.0%} of its budget")
