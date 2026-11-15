# auction-sim

A small simulator of online ad auctions, for testing bidding methods such as pacing, bid shading, and learning bidders.

You write one bidder class and one YAML file. The simulator runs the auctions, hides what a real bidder could not see, and reports the same metrics for every method.

## Quick start

```bash
uv sync
uv run auction-sim run examples/basic.yaml --seeds 0-4 --out results.csv
```

The command prints a mean per bidder and writes one CSV row per seed and bidder.

## How one step works

1. Each bidder i gets a conversion probability pᵢ = σ(x · φᵢ + βᵢ) for a random context x.
2. Each bidder sees its own `Observation`: step, steps left, value vᵢ = wᵢ × pᵢ, value per conversion wᵢ, and budget left.
3. Each bid is capped at the bidder's remaining budget.
4. The auction picks a winner: first price or second price, with an optional reserve.
5. The winner converts with probability p of the winner.
6. Each bidder gets its own `Feedback`: its bid, whether it won, the price it paid, whether it converted, and its reward. A loser never sees the winning price.

Budgets reset at the start of each episode. Bidder objects persist for the whole run, so a learning bidder keeps what it learned.

## Write a bidder

```python
# my_bidders.py
from auction_sim.bidders import Bidder


class HalfValue(Bidder):
    def bid(self, obs):
        return self.params.get("fraction", 0.5) * obs.value
```

Then name it in the config as `type: "my_bidders:HalfValue"`. The module must be importable, so run from its folder with `PYTHONPATH=. uv run auction-sim run config.yaml`. Optional hooks are `on_episode_start(budget, steps)`, `update(feedback)`, and `on_episode_end(stats, episode_reward)`.

## Built-in bidders

| Type | Behavior | Params |
|---|---|---|
| `truthful` | Bids `multiplier` × value | `multiplier` (1.0) |
| `pid_pacing` | A PID loop moves a bid multiplier so spend tracks the budget evenly | `initial_multiplier` (1.0), `kp` (20), `ki` (2), `kd` (0), `min_multiplier` (0), `max_multiplier` (10) |
| `ucb_multiplier` | UCB1 over a grid of multipliers; one arm per episode | `min_multiplier` (0.2), `max_multiplier` (2.0), `multiplier_step` (0.2), `exploration` (1.0) |

## Rewards

Each bidder has a `reward`, called every step, and an optional `episode_reward`, called once per episode.

| Kind | Built-in names | Custom function |
|---|---|---|
| Step | `surplus` (default), `conversions`, `value` | `f(obs, feedback, params) -> float` |
| Episode | `cpa_score`, `total_surplus` | `g(stats, params) -> float` |

Name a custom function as `"module:function"`. The `params` dict holds the bidder's `reward_params`, plus `cpa_target` when the bidder has one.

## Config

See `examples/basic.yaml`. An unknown key is an error, so a typo does not pass silently.

| Key | Meaning |
|---|---|
| `episodes`, `steps`, `seed` | Run length and the default seed |
| `auction.kind`, `auction.reserve` | `first` or `second`, and the reserve price |
| `world.context_dim`, `world.bias_mean`, `world.bias_std` | Context size and the spread of conversion rates |
| `bidders[].name`, `type`, `budget`, `value_per_conversion` | Required for each bidder |
| `bidders[].cpa_target`, `params`, `reward`, `episode_reward`, `reward_params` | Optional |

## Metrics

`docs/metrics.md` defines every metric. Regret compares a bidder with the best fixed bid multiplier in hindsight, with other bids held fixed. Read its limits there before you compare methods.

## Reproducibility

Each source of randomness has its own seeded stream: the world, tie-breaking, conversions, and each bidder. The same seed gives the same log. Adding a bidder does not change the impressions the others see.
