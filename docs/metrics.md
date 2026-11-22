# Metrics

This file defines every metric the simulator reports, so results from different methods mean the same thing.

The code is in `src/bidlab/metrics.py`. Each metric is computed per episode, then averaged over episodes per seed.

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
| budget_used | spend / Bᵢ | Never above 1 |
| ctr | clicks / wins | NaN when wins = 0 |
| cpc | spend / clicks | NaN when clicks = 0. Billing is per impression, so this is impression spend per click |
| cpa | spend / conversions | NaN when conversions = 0. Counts conversions for every event |
| cpa_violation | max(0, cpa / targetᵢ − 1) | 0 when spend = 0. +inf when spend > 0 and conversions = 0. NaN with no target |
| cpa_score | realized_value × min((targetᵢ / cpa)³, 1) | AuctionNet score. Equals realized_value when spend = 0. 0 when spend > 0 and conversions = 0. NaN with no target |
| regret | See below | Uses expected value, not realized |

## Platform metrics

| Metric | Definition |
|---|---|
| revenue | Σ priceₜ over all steps: only campaigns' payments, since an auction sold outside has price 0 |
| welfare | Σ vᵢₜ of the winner, over steps a campaign won |

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

For one seed and campaign, each metric summed over segments equals the main table's value.

## Regret

Regret compares the bidder with the best fixed multiplier in hindsight.

1. Hold every other bidder's logged bids, the market price and the floor fixed.
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
