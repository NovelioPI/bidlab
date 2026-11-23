# Changelog

This file lists the changes in each bidlab release, newest first.

bidlab follows [semantic versioning](https://semver.org/). Before 1.0, a minor version may break the API; each break is listed under **Breaking**.

## 0.5.0 (2026-11-23)

### Breaking

- `run_many` is removed: `run` takes the seed. `run(config, seed=[0, 1, 2])`, or any other iterable of ints, returns the `Results` that `run_many(config, seeds=[0, 1, 2])` did, with the same `workers`, `callbacks`, `progress_bar` and `profiler` arguments. `seed=[42]` also returns `Results`.
- `SimConfig.seed` is removed: pass `run(config, seed=7)`. The seed defaults to 0, and `RunLog.seed` records it. A `RunLog` built by hand needs `seed`.
- `run` takes `callbacks`, `progress_bar` and `profiler` as keywords only.
- Callback hooks are renamed: `on_run_many_start(config, seeds)` is `on_seeds_start(config, seeds)` and `on_run_many_end(results)` is `on_seeds_end(results)`. `on_run_start(config)` is `on_run_start(config, seed)`, so a callback sees the seed during a run.
- Every `Campaign` setting after `bidder` is keyword-only: `Campaign(name, bidder, budget=..., base_value=...)`. A positional budget would now swap with the value, so it raises `TypeError`.
- `Campaign.budget` defaults to None, which sizes the budget from the market. Set `budget` to keep a fixed budget.
- A `RunLog` built by hand needs `budgets`.
- Seed-list results and their CSV files have a new `budget` column.
- Seed-list results have nine new columns after `regret`, so `revenue` and `welfare` move right in a CSV. Code that reads columns by name is not affected.

### Runs

- `run(config, seed)` raises `TypeError` for a seed that is neither an int nor an iterable of ints (a bool counts as neither), and `ValueError` for an empty seed list or for `workers > 1` with an int seed.
- `Callback.on_seed_progress(seed, episode, step)` reports each seed's progress in a seed list, about 20 times per episode and at its last step, in the main process, also with `workers > 1`.
- `ProgressBar` labels a run's bar `seed 42 episode 1/3`. For a seed list it shows a `seeds` line and one line per running seed, at most `workers` of them.

### Campaigns

- `Campaign(budget=None)` sizes each budget from the market: the mean cost per episode of every request worth its cost to the campaign, which is what a truthful bidder spends with no limit. The run replays its own market draws once to find it, so no random draw changes.
- `Campaign(budget_fraction=f)` scales the budget, set or sized, for budget sweeps.
- `RunLog.budgets`, `EpisodeReport.budget` and the `budget` metric hold the budget each campaign got; `budget_used` and `regret` use it.

### Metrics

- New campaign metrics in `campaign_metrics` and seed-list results: `win_rate`, `expected_surplus`, `budget_out`, `cvr` and `cost_per_expected_event`.
- `lost_budget`, `lost_floor`, `lost_market` and `lost_rival` count why each request was lost. `docs/metrics.md` gives the order of the rules.
- `budget_out` is the share of the episode before the budget first blocked a win. Seed-list results average it over the episodes that ran out.
- `segment_metrics` and `results.segment_rows` have `expected_surplus`.

### Profiling

- `run(config, profiler=True)` times every hook of every bidder node and samples memory into `RunLog.profile`, a `bl.Profile`. `Profile.summary()` prints a table of total and self time per hook, slowest first, then wall time, steps per second, simulator time and memory.
- `run(config, seed=[...], profiler=True)` gives one `Profile` per seed in `Results.profile`, in seed order.
- `Profile.save_csv(path)` and `Results.save_profile_csv(path)` write the rows with their seed and run totals to a CSV file, with exact times.
- Profiles change no array and take no part in `RunLog` `==` or `Results.equals`. Without `profiler=True`, runs keep their speed and every bit.
- `psutil` is a new runtime dependency, used to read memory on every OS.

## 0.4.0 (2026-11-23)

### Breaking

- `RunLog.winner` is removed. `RunLog.won`, `price`, `clicked` and `converted` now have shape (episodes, steps, campaigns); `price` is 0 where a campaign lost. Replace `log.winner == i` with `log.won[..., i]`.

### Simulation

- `SimConfig(competing=False)` runs each campaign alone against the market price, on one shared market draw, to compare bidders on the same traffic. Every draw then needs a market price; platform `revenue` and `welfare` are NaN, and `regret` replays without other campaigns' bids.

## 0.3.1 (2026-11-23)

### Callbacks

- `run` and `run_many` show a `bl.ProgressBar` by default; pass `progress_bar=False` to hide it. `run_many` shows only its seed bar.

## 0.3.0 (2026-11-23)

### Markets

- `bl.Market` is the market base class; `bl.markets.Market` still works.

### Callbacks

- `bl.ProgressBar()` shows a tqdm progress bar: the current episode's steps in `run`, seeds in `run_many`. bidlab now depends on `tqdm`.
- `Callback.on_step_end(episode, step)` runs in `run` after each step's auction.
- `Callback.on_run_many_start(config, seeds)` and `Callback.on_run_many_end(results)` run before the first seed and after the last in `run_many`.

### Results

- `print(results)`, or `results` in a notebook, shows the summary table instead of the raw rows.

### Bidders

- A bidder prints as its constructor call, such as `Stack(OracleValue(), PidPacing(kp=5.0))`, also inside `SimConfig` and `RunLog`. It reads each argument from the attribute of the same name; without one, it prints as before.
- `UcbMultiplier` keeps `min_multiplier`, `max_multiplier` and `multiplier_step` as attributes.

## 0.2.0 (2026-11-23)

### Simulation

- `SimConfig(feedback=bl.Feedback(interval=, report_delay=, conversion_delay=))` delays what a bidder learns, in seconds. The defaults are 0, so runs without it keep every number.
- Under a `report_delay`, the `budget_left` input counts only the spend in delivered results; the bid cap still uses the true budget.
- Feedback still pending after an episode's last auction is delivered before `on_episode_end`.

### Bidders

- `Bidder.update_conversions(conversions)` receives late conversions as `bl.Conversion(time, realized_value)` under a `conversion_delay`; `BidResult.converted` is then False.
- `UcbMultiplier` adds late conversion value to its arm reward.

## 0.1.0 (2026-11-22)

The first release on PyPI.

### Simulation

- `bl.run(config)` runs one simulation and returns a read-only `RunLog`.
- `bl.run_many(config, seeds, workers=...)` runs one simulation per seed, in parallel if asked, and returns `Results` tables per campaign and per segment.
- Runs repeat bit for bit for a seed. Adding a campaign leaves the other campaigns' draws unchanged.
- `bl.Callback` hooks read a run as it goes, without changing it.

### Bidders

- Subclass `bl.Bidder`. Its `forward` names the inputs it needs, such as `budget_left`, `time_left`, `floor` or `features`.
- `bl.Stack` passes each bidder's output to the next as `value`, such as `bl.Stack(bl.OracleValue(), bl.PidPacing())`.
- A bidder can hold child bidders; each gets its own random stream and hooks.
- `self.log(name, value)` saves any number per step or episode into `RunLog.logged` and `Results.logged`.
- Built-in bidders: `OracleValue`, `Linear`, `PidPacing` and `UcbMultiplier`.

### Campaigns and auctions

- `bl.Campaign` holds a budget, a base value, the event it pays for (`impression`, `click` or `conversion`), an optional CPA target and target segments.
- `bl.FirstPrice` and `bl.SecondPrice` take a reserve price.

### Markets

- `LogisticMarket(ctr=, cvr=)` sets click and conversion rates directly, or through logit biases.
- Public features: bidders see the first `public_features` numbers of each step's feature vector.
- Outside competition: a market price and a floor per auction, which can follow each campaign's value.
- `Segmented` splits traffic into segments with their own rates or lifts.
- `IndependentValues` draws each probability from any scipy distribution.
- Subclass `bl.markets.Market` to write your own.

### Metrics

- Per campaign: spend, wins, clicks, conversions, surplus, budget used, CTR, CPC, CPA, CPA violation, CPA score and regret.
- Per platform: revenue and welfare. Per segment: spend, wins, clicks, conversions and surplus.
