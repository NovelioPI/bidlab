# Metrics

This file defines every metric the simulator reports, so results from different methods mean the same thing.

The code is in `src/bidlab/metrics.py`. Each metric is computed per episode, then averaged over episodes per seed.

## Symbols

| Symbol | Meaning | Unit |
|---|---|---|
| t | Step in the episode | step |
| vᵢₜ | Expected value of bidder i at step t: wᵢ × pᵢₜ | money |
| wᵢ | Value per conversion of bidder i | money |
| pᵢₜ | Conversion probability of bidder i at step t | probability |
| Wᵢ | Steps that bidder i won | set of steps |
| priceₜ | Price paid at step t | money |
| Bᵢ | Budget of bidder i per episode | money |
| targetᵢ | CPA target of bidder i, if set | money per conversion |

## Bidder metrics

| Metric | Definition | Notes |
|---|---|---|
| spend | Σ priceₜ over Wᵢ | |
| wins | \|Wᵢ\| | |
| conversions | Realized conversions over Wᵢ | |
| value | conversions × wᵢ | Realized, so it includes luck |
| surplus | value − spend | |
| budget_used | spend / Bᵢ | Never above 1 |
| cpa | spend / conversions | NaN when conversions = 0 |
| cpa_violation | max(0, cpa / targetᵢ − 1) | 0 when spend = 0. +inf when spend > 0 and conversions = 0. NaN with no target |
| cpa_score | value × min((targetᵢ / cpa)³, 1) | AuctionNet score. Equals value when spend = 0. 0 when spend > 0 and conversions = 0. NaN with no target |
| regret | See below | Uses expected value, not realized |
| reward | Σ of the bidder's step rewards | Uses the bidder's configured `reward` |
| episode_reward | The configured `episode_reward` | NaN when none is set |

## Platform metrics

| Metric | Definition |
|---|---|
| revenue | Σ priceₜ over all steps |
| welfare | Σ vᵢₜ of the winner, over steps with a sale |

## Segment metrics

`results.segments` holds these columns, one row per seed, bidder, and market segment. Each value is a per-episode sum over the segment's steps, averaged over episodes.

| Metric | Definition | Notes |
|---|---|---|
| segment | The segment name from `market.segment_names` | `"all"` for a market without segments |
| spend | Σ priceₜ over Wᵢ in the segment | |
| wins | \|Wᵢ\| in the segment | |
| conversions | Realized conversions over Wᵢ in the segment | |
| surplus | conversions × wᵢ − spend | |

For one seed and bidder, each metric summed over segments equals the main table's value.

## Regret

Regret compares the bidder with the best fixed multiplier in hindsight.

1. Hold every other bidder's logged bids fixed.
2. For each multiplier m in 0, 0.05, …, 3, replay the episode with bidder i bidding m × vᵢₜ.
3. Cap each replayed bid at the replayed remaining budget.
4. A replayed bid wins when it meets the reserve and is strictly above the highest other bid. A bidder with `segments` cannot win outside them in the replay either.
5. The replayed price follows the auction rule: the bid in first price, max(highest other bid, reserve) in second price.
6. Replay surplus(m) = Σ (vᵢₜ − priceₜ) over the replayed wins.
7. Own surplus = Σ (vᵢₜ − priceₜ) over the logged wins.
8. regret = max over m of replay surplus(m), minus own surplus.

## Limits of regret

- Regret can be negative. An adaptive bidder can beat every fixed multiplier.
- Regret ignores CPA targets. A bidder that underbids to keep its CPA shows high regret. Read it together with `cpa_violation`.
- Other bidders do not react in the replay. Learning competitors would have bid differently against the replayed bidder.
- Ties lose in the replay, but the simulator breaks them at random. With continuous values, ties almost never happen.
