# Changelog

This file lists the changes in each bidlab release, newest first.

bidlab follows [semantic versioning](https://semver.org/). Before 1.0, a minor version may break the API; each break is listed under **Breaking**.

## Unreleased

### Markets

- `bl.Market` is the market base class; `bl.markets.Market` still works.

### Callbacks

- `bl.ProgressBar()` shows a tqdm progress bar: the current episode's steps in `run`, seeds in `run_many`. bidlab now depends on `tqdm`.
- `Callback.on_step_end(episode, step)` runs in `run` after each step's auction.
- `Callback.on_run_many_start(config, seeds)` and `Callback.on_run_many_end(results)` run before the first seed and after the last in `run_many`.

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
