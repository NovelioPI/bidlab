# Campaigns

This guide shows a campaign's settings: its budget, the event it pays for, its base value, a CPA target and segments.

## Settings

| Setting | Meaning |
|---|---|
| `name` | Label in results; unique within a `SimConfig` |
| `bidder` | A `Bidder` instance; the simulator runs a fresh copy per run |
| `budget` | Budget per episode |
| `base_value` | What one event is worth to the advertiser; a bidder may use it as is or change it |
| `event` | What the campaign pays for: `"impression"`, `"click"` or `"conversion"` (the default), or a `bl.Event` member |
| `cpa_target` | Highest acceptable cost per conversion, for every event; the `cpa_score` metric uses it |
| `segments` | Market segments the campaign bids in; None means every segment |

A bidder gets the whole `Campaign` in `on_episode_start(campaign)`, so it can read `campaign.event` and `campaign.base_value`.

## Events

The event sets the campaign's expected value and what it earns:

| Event | Expected value | Earns `base_value` per |
|---|---|---|
| `impression` | `base_value` | Won impression |
| `click` | `base_value` × pCTR | Clicked won impression |
| `conversion` | `base_value` × pCTR × pCVR | Converted won impression |

The expected value fills `RunLog.true_values`, and `regret` and `welfare` use it. `bl.OracleValue` outputs it exactly; with `noise`, only the stages the event uses get noise. `BidResult.realized_value`, `EpisodeReport.realized_value` and the `realized_value` and `surplus` columns count the event.

```python
import bidlab as bl


def paced():
    return bl.Stack(bl.OracleValue(), bl.PidPacing())


goals = bl.SimConfig(
    campaigns=[
        bl.Campaign("brand", paced(), budget=15, base_value=0.02, event="impression"),
        bl.Campaign("traffic", paced(), budget=15, base_value=0.4, event="click"),
        bl.Campaign("shop", paced(), budget=15, base_value=5),
    ],
    market=bl.markets.LogisticMarket(ctr=0.05),
    episodes=2,
    steps=200,
)
for row in bl.run_many(goals, seeds=[0]).rows:
    print(row["campaign"], row["ctr"], row["cpc"])
```

The `ctr` column is clicks per win, and `cpc` is spend per click. Both are NaN when there is nothing to divide by.

Keep these limits in mind:

- Billing stays per impression for every event. A click campaign pays for each won impression, so `cpc` is impression spend per click.
- `cpa_target` limits cost per conversion, even for a click or impression campaign.
- A market without a click stage counts every won impression as clicked. There, a click campaign acts like an impression campaign.

## Examples

| Script | Question |
|---|---|
| [campaigns/events.py](../../examples/campaigns/events.py) | With the same budget, does a click campaign pay less per click than a conversion campaign? |
| [rewards/cpa_target.py](../../examples/rewards/cpa_target.py) | Does bidding target/value of the value meet a CPA target that truthful bidding misses? |
