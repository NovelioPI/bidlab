# Metrics

This file defines every metric the simulator reports, so results from different methods mean the same thing.

The code is in `src/bidlab/metrics.py`. Each metric is computed per episode, then averaged over episodes per seed. `budget_out` is the one exception: see [Lost requests](#lost-requests).

## Symbols

| Symbol | Meaning | Unit |
|---|---|---|
| t | Step in the episode | step |
| vᵢₜ | True expected value of campaign i at step t: wᵢ × eᵢₜ, as in `RunLog.true_values` | money |
| wᵢ | Value of one event to campaign i: `Campaign.base_value` | money |
| eᵢₜ | True chance of campaign i's event at step t, by `Campaign.event`: 1 for an impression, cᵢₜ for a click, cᵢₜ × pᵢₜ for a conversion | probability |
| nᵢ | Count of campaign i's event: wins, clicks or conversions, by `Campaign.event` | count |
| cᵢₜ | True click probability (pCTR) of campaign i at step t; 1 when the market has no click stage | probability |
| pᵢₜ | True conversion probability after a click (pCVR) of campaign i at step t, not the bidder's estimate | probability |
| Wᵢ | Steps that campaign i won | set of steps |
| Rᵢ | Requests of campaign i: steps in its `segments`, or every step without them | set of steps |
| bᵢₜ | Logged bid of campaign i at step t, after the budget cap | money |
| priceₜ | Price paid at step t; 0 when no campaign won | money |
| mₜ | Market price at step t: the highest bid from outside the config; −∞ without one | money |
| rₜ | Effective floor at step t: max(reserve, market floor) | money |
| Bᵢ | Budget of campaign i per episode | money |
| targetᵢ | CPA target of campaign i, if set | money per conversion |

## Campaign metrics

| Metric | Definition | Notes |
|---|---|---|
| spend | Σ priceₜ over Wᵢ | |
| wins | \|Wᵢ\| | |
| clicks | Realized clicks over Wᵢ | Equals wins when the market has no click stage |
| conversions | Realized conversions over Wᵢ | Only a click converts |
| realized_value | nᵢ × wᵢ | Realized, so it includes luck |
| surplus | realized_value − spend | |
| budget | Bᵢ, from `RunLog.budgets` | `budget_fraction` × the set budget, or × the market cost when `budget=None` |
| budget_used | spend / Bᵢ | Never above 1 |
| ctr | clicks / wins | NaN when wins = 0 |
| cpc | spend / clicks | NaN when clicks = 0. Billing is per impression, so this is impression spend per click |
| cpa | spend / conversions | NaN when conversions = 0. Counts conversions for every event |
| cpa_violation | max(0, cpa / targetᵢ − 1) | 0 when spend = 0. +inf when spend > 0 and conversions = 0. NaN with no target |
| cpa_score | realized_value × min((targetᵢ / cpa)³, 1) | AuctionNet score. Equals realized_value when spend = 0. 0 when spend > 0 and conversions = 0. NaN with no target |
| regret | See below | Uses expected value, not realized |
| win_rate | wins / \|Rᵢ\| | NaN when \|Rᵢ\| = 0. A bid of 0 still counts as a request |
| expected_surplus | Σ vᵢₜ over Wᵢ − spend | Surplus without the luck of the event draws |
| budget_out | See [Lost requests](#lost-requests) | NaN when no request was lost to the budget |
| cvr | conversions / clicks | NaN when clicks = 0 |
| cost_per_expected_event | spend / (Σ vᵢₜ over Wᵢ / wᵢ) | A CPA without draw luck, for any event. NaN when wins = 0 |
| lost_budget | Requests lost to the budget | See [Lost requests](#lost-requests) |
| lost_floor | Requests lost to the floor | See [Lost requests](#lost-requests) |
| lost_market | Requests lost to the market price | See [Lost requests](#lost-requests) |
| lost_rival | Requests lost to another campaign | See [Lost requests](#lost-requests). Always 0 with `competing=False` |

## Lost requests

A lost request is a step in Rᵢ that campaign i did not win. The metrics read only the `RunLog`.

| Term | Plain meaning |
|---|---|
| Budget left | Bᵢ minus the prices campaign i paid at earlier steps of the episode |
| Price to win | max(rₜ, mₜ, highest other bid). mₜ counts as 0 without a market price. The highest other bid counts only with `competing=True`, else 0 |

Each lost request counts in exactly one kind. The first rule that applies wins:

1. `lost_budget` when the budget left is below the price to win.
2. `lost_floor` when bᵢₜ < rₜ.
3. `lost_market` when the step has a market price and bᵢₜ ≤ mₜ.
4. `lost_rival` otherwise.

So lost_budget + lost_floor + lost_market + lost_rival = \|Rᵢ\| − wins. A budget left equal to the price to win is affordable. A bid that ties another campaign's top bid can lose the random tie break; that loss counts as `lost_rival`.

`budget_out` = k / S, where k is the step of the first request lost to the budget and S is the episode's steps. It is NaN when no request was lost to the budget. So "never ran out" stays apart from "ran out at the last step".

In `run_many`, `budget_out` is the mean over the episodes where it is not NaN. It is NaN when the budget never ran out in any episode. A plain mean would let one easy episode hide the others.

## Platform metrics

| Metric | Definition |
|---|---|
| revenue | Σ priceₜ over all steps: only campaigns' payments, since an auction sold outside has price 0 |
| welfare | Σ vᵢₜ of the winner, over steps a campaign won |

Both are NaN with `SimConfig(competing=False)`. Each campaign then buys the impression in its own auction, so no single sale exists.

## Segment metrics

`results.segment_rows` holds these columns, one row per seed, campaign, and market segment. Each value is a per-episode sum over the segment's steps, averaged over episodes.

| Metric | Definition | Notes |
|---|---|---|
| segment | The segment name from `market.segment_names` | `"all"` for a market without segments |
| spend | Σ priceₜ over Wᵢ in the segment | |
| wins | \|Wᵢ\| in the segment | |
| clicks | Realized clicks over Wᵢ in the segment | |
| conversions | Realized conversions over Wᵢ in the segment | |
| surplus | nᵢ × wᵢ − spend, with nᵢ counted in the segment | |
| expected_surplus | Σ vᵢₜ over Wᵢ in the segment − spend | |

For one seed and campaign, each metric summed over segments equals the main table's value.

## Regret

Regret compares the bidder with the best fixed multiplier in hindsight.

1. Hold every other bidder's logged bids, the market price and the floor fixed. With `competing=False`, hold only the market price and the floor: other campaigns are not in bidder i's auction, so the highest other bid is 0.
2. For each multiplier m in 0, 0.05, …, 3, replay the episode with bidder i bidding m × vᵢₜ.
3. Cap each replayed bid at the replayed remaining budget.
4. A replayed bid wins when it is at or above rₜ, and strictly above both the highest other bid and mₜ. A bidder with `segments` cannot win outside them in the replay either.
5. The replayed price follows the auction rule: the bid in first price, max(highest other bid, rₜ, mₜ) in second price.
6. Replay surplus(m) = Σ (vᵢₜ − priceₜ) over the replayed wins.
7. Own surplus = Σ (vᵢₜ − priceₜ) over the logged wins.
8. regret = max over m of replay surplus(m), minus own surplus.

## Limits of regret

- Regret can be negative. An adaptive bidder can beat every fixed multiplier.
- The replay bids on the true value. With `bl.OracleValue(noise=...)`, or a bidder's own model, the bidder saw a different value, so regret includes the cost of that wrong estimate.
- Regret ignores CPA targets. A bidder that underbids to keep its CPA shows high regret. Read it together with `cpa_violation`.
- Other bidders do not react in the replay. Learning competitors would have bid differently against the replayed bidder.
- Ties lose in the replay, but the simulator breaks them at random. With continuous values, ties almost never happen.
