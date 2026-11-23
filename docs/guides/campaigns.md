# Campaigns

This guide shows a campaign's settings: its budget, the event it pays for, its base value, a CPA target and segments. It also shows how to size a budget from the market and how to compare campaigns that do not compete.

## Settings

| Setting | Meaning |
|---|---|
| `name` | Label in results; unique within a `SimConfig` |
| `bidder` | A `Bidder` instance; the simulator runs a fresh copy per run |
| `budget` | Budget per episode; None, the default, sizes it from the market |
| `budget_fraction` | Scales the budget, set or sized; 1 by default |
| `base_value` | What one event is worth to the advertiser; a bidder may use it as is or change it |
| `event` | What the campaign pays for: `"impression"`, `"click"` or `"conversion"` (the default), or a `bl.Event` member |
| `cpa_target` | Highest acceptable cost per conversion, for every event; the `cpa_score` metric uses it |
| `segments` | Market segments the campaign bids in; None means every segment |

Every setting after `bidder` is keyword-only. A bidder gets the whole `Campaign` in `on_episode_start(campaign)`, so it can read `campaign.event` and `campaign.base_value`.

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
for row in bl.run(goals, seed=[0]).rows:
    print(row["campaign"], row["ctr"], row["cpc"])
```

The `ctr` column is clicks per win, and `cpc` is spend per click. Both are NaN when there is nothing to divide by.

Keep these limits in mind:

- Billing stays per impression for every event. A click campaign pays for each won impression, so `cpc` is impression spend per click.
- `cpa_target` limits cost per conversion, even for a click or impression campaign.
- A market without a click stage counts every won impression as clicked. There, a click campaign acts like an impression campaign.

## Size the budget from the market

A budget of 3 can bind hard in one market and never bind in another. With `budget=None`, the run sizes each campaign's budget from the market, so a budget is equally tight in any market:

```python
import bidlab as bl

sized = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "half",
            bl.Stack(bl.OracleValue(), bl.PidPacing()),
            base_value=20,
            budget_fraction=0.5,
        )
    ],
    market=bl.markets.LogisticMarket(price_median=0.2, floor_share=(2, 5)),
    episodes=2,
    steps=200,
)
log = bl.run(sized)
print(log.budgets)
```

The budget per episode is `budget_fraction` times the market cost:

| Term | Meaning |
|---|---|
| Request cost | The highest of the market price, the auction's `reserve` and the floor: the lowest price that wins the request alone |
| Worth its cost | The campaign's true value for the request, its `RunLog.true_values` entry, is at least the request cost |
| Market cost | The sum of the request costs of every request in the campaign's segments that is worth its cost, divided by `episodes` |

So `budget_fraction=1` is what a truthful bidder spends with no budget limit, and a fraction below 1 binds. Campaigns with the same base value, event and segments get the same budget.

The run finds the market cost by replaying its own market draws once, before the first episode. The budget is the same in every episode, and no random draw of the run changes: the run equals one with the number typed in, bit for bit. A `budget=None` run takes longer, since it draws the market twice.

| Case | Behavior |
|---|---|
| Set `budget` | The budget is `budget_fraction` × `budget`, with no replay |
| What bidders see | `campaign.budget` in `on_episode_start` is the scaled budget, and `campaign.budget_fraction` is 1 |
| Where to read it | `RunLog.budgets`, `EpisodeReport.budget`, and the `budget` column of a seed list's `Results`. `config.campaigns[i].budget` stays None |
| Market cost 0 | `ValueError`: a market with no market price and reserve 0, segments the run never draws, or a value below every cost |
| `FirstPrice` | A bid equal to the market price loses, so the cost is a lower limit there |
| `competing=True` | The cost counts the market only, not other campaigns' bids, since those bids depend on the budgets |
| A market whose draws use more than their `rng` | `RuntimeError` after the run, since the replay sized the budget from other draws |

A budget below the market cost binds for any bidder. A bidder without pacing, such as `bl.Linear`, still spends it as fast as it wins; pacing spreads it over the day.

## Compare campaigns without competition

By default, all campaigns bid in one auction per step, so each campaign's result depends on the others' bids. To benchmark bid policies, set `competing=False`:

```python
import bidlab as bl


def campaign(name, policy):
    return bl.Campaign(
        name, bl.Stack(bl.OracleValue(), policy), budget=3, base_value=20
    )


benchmark = bl.SimConfig(
    campaigns=[campaign("truthful", bl.Linear()), campaign("pid", bl.PidPacing())],
    market=bl.markets.LogisticMarket(price_median=0.2),
    competing=False,
    episodes=2,
    steps=200,
)
print(bl.run(benchmark, seed=range(3)).summary())
```

With `competing=False`, the run changes in these ways:

| Part | Behavior |
|---|---|
| Market | Drawn once per step for one campaign. Every campaign sees the same features, pCTR, pCVR, market price and floor |
| Auction | Each campaign bids alone against the market price and floor, so several campaigns can win one impression |
| Clicks and conversions | One click draw and one conversion draw per step decide every winner, so equal campaigns get equal outcomes |
| Market price | Required on every draw. Without one, a campaign alone at reserve 0 wins every impression for free, so the run raises `ValueError` |
| Platform `revenue` and `welfare` | NaN: no single sale exists |
| `regret` | Replays without other campaigns' bids |

Each campaign's results equal a run of that campaign alone, with the same market and seed, bit for bit. This holds while its bidder draws no random number, since a bidder's stream depends on its place in the config. Give the campaigns the same budget, base value, event and segments, and only the bidder differs.

## Examples

| Script | Question |
|---|---|
| [campaigns/events.py](../../examples/campaigns/events.py) | With the same budget, does a click campaign pay less per click than a conversion campaign? |
| [campaigns/benchmark.py](../../examples/campaigns/benchmark.py) | On one shared market, does PID pacing earn more surplus than truthful bidding? |
| [campaigns/market_budget.py](../../examples/campaigns/market_budget.py) | With a budget at half the market's cost, does truthful bidding run out early while PID pacing lasts the day? |
| [rewards/cpa_target.py](../../examples/rewards/cpa_target.py) | Does bidding target/value of the value meet a CPA target that truthful bidding misses? |
