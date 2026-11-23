# Markets

This guide shows the built-in markets, clicks and conversions, rates, public features, segments and targeting, the market price and floor, and how to write your own market.

## Built-in markets

A market draws each step's public features and each campaign's private probabilities. Each feature reaches `forward` as an input of the same name, which must not be a built-in input such as `time`, and all of them together as the `features` input. Pass one as `SimConfig(market=...)`.

| Market | Draws |
|---|---|
| `bl.markets.LogisticMarket()` | The default: pCVRᵢ = σ(x · φᵢ + βᵢ) for a random feature vector x; optionally a click stage |
| `bl.markets.Segmented(segments)` | A segment per step by its `share`; a segment can set its own rates or lifts |
| `bl.markets.IndependentValues(distribution)` | Each pᵢ drawn on its own from a scipy frozen distribution, clipped to [0, 1] |

`IndependentValues(scipy.stats.uniform(0, 1))` with a campaign `base_value` of 1 is the textbook independent private values setting.

## Clicks and conversions

A won impression is clicked with probability pCTR, and a click converts with probability pCVR. A market without a click stage counts every won impression as clicked, so pCVR is the conversion chance of a win.

The true rates set `RunLog.true_values`, through each campaign's event; see [Campaigns](campaigns.md). `RunLog.clicked`, `BidResult.clicked` and the `clicks` column record the clicks.

A bidder that declares the `market` input gets a `bl.MarketTruth` with its own true `pctr` and `pcvr`; `pctr` is 1.0 without a click stage. Use it to check an estimator or to build an oracle. A fair benchmark bidder does not read it.

## Rates

Set each stage's rate, and `LogisticMarket` finds the logit bias that gives it.

```python
import bidlab as bl

rated = bl.markets.LogisticMarket(ctr=0.02, cvr=0.05)
```

| Setting | Effect |
|---|---|
| `ctr` (None) | Click rate in (0, 1); turns on the click stage |
| `cvr` (None) | Conversion rate per click, in (0, 1) |
| `ctr_spread`, `cvr_spread` (0.5) | How much campaigns differ, on the logit scale |

Campaign i's mean pCTR over a run is σ(logit(`ctr`) + δᵢ). Its offset δᵢ ~ N(0, `ctr_spread`) is drawn once per run. The median campaign has the set rate, and a spread of 0 gives every campaign exactly that rate. Each campaign is solved on its own, so adding a campaign changes no other campaign's draws.

The rate is a mean over all auctions. The `ctr` column in results counts clicks per win. Bidders win the better impressions, so that column sits above the set rate.

### Bias settings

Behind the rates is a logit model on the same feature vector x: pCVRᵢ = σ(x · φᵢ + βᵢ) and pCTRᵢ = σ(x · ψᵢ + γᵢ). Set a bias mean in place of a rate to choose the biases directly. A rate and its bias mean together raise a `ValueError`.

| Setting | Effect |
|---|---|
| `ctr_bias_mean` (None) | γᵢ ~ N(`ctr_bias_mean`, `ctr_spread`); turns on the click stage |
| `cvr_bias_mean` (None) | βᵢ ~ N(`cvr_bias_mean`, `cvr_spread`); None without `cvr` means -3, so pCVR is near 5% |

Without `ctr` or `ctr_bias_mean`, the market has no click stage and gives the same runs as before clicks existed.

## Public features and predictability

`LogisticMarket` sends the first `public_features` numbers of x as the read-only array `features["vector"]`, also the `vector` input. The rest still drive both probabilities, so a model on the public features cannot learn them exactly.

| Setting | Effect |
|---|---|
| `feature_count` (8) | Length of x |
| `public_features` (None) | How many numbers of x bidders see; None means all |

With every feature public, logit(pCVR) is an exact linear function of the vector. With `public_features=3`, the hidden five add noise no model can explain.

## Segments and targeting

A bidder gets the segment name as its `segment` input. A campaign with `segments` bids only in those segments. Elsewhere its bidder is not called, and its logged bid is 0.

```python
import bidlab as bl

market = bl.markets.Segmented(
    (
        bl.markets.Segment("news", share=0.7),
        bl.markets.Segment("sports", share=0.3, cvr_lift=1.0),
    )
)
segmented = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "everywhere",
            bl.Stack(bl.OracleValue(), bl.Linear()),
            budget=50,
            base_value=5,
        ),
        bl.Campaign(
            "sports_only",
            bl.Stack(bl.OracleValue(), bl.Linear()),
            budget=50,
            base_value=5,
            segments=("sports",),
        ),
    ],
    market=market,
    episodes=2,
    steps=200,
)
segment_results = bl.run_many(segmented, seeds=range(2))
print(len(segment_results.segment_rows))  # 8 rows: seeds × bidders × segments
```

`results.segment_rows` is a second table with spend, wins, clicks, conversions and surplus per seed, bidder and segment. A market without segments reports one segment, `"all"`.

A targeted bidder sees only its segments' auctions. `PidPacing` plans spend by time, so it still spreads the budget over the whole horizon.

### Segment rates and lifts

| Setting | Effect |
|---|---|
| `ctr`, `cvr` | The segment's own rate, in place of the market's; each campaign keeps its offset δᵢ |
| `ctr_lift`, `cvr_lift` | Added to every campaign's logit in the segment |

A segment sets a stage's rate or its lift, not both. Click settings need a click stage in the base market. The market's own rate then holds only in segments without a rate of their own.

```python
by_device = bl.markets.Segmented(
    (
        bl.markets.Segment("mobile", share=0.7, ctr=0.03),
        bl.markets.Segment("desktop", share=0.3, cvr_lift=0.5),
    ),
    base=bl.markets.LogisticMarket(ctr=0.02, cvr=0.05),
)
```

## Market price and floor

By default, campaigns compete only with each other. A campaign alone then pays the reserve, so its costs mean little. A market can add outside competition and a floor per auction.

| Term | Plain meaning |
|---|---|
| Market price, m | The highest bid from outside the config at one auction |
| Market floor | The publisher's floor at one auction |
| Effective floor, r | The higher of the auction's `reserve` and the market floor |

A campaign wins when its bid is above m, at or above r, and the highest campaign bid. A campaign bid equal to m loses. When no campaign wins, the auction is sold outside: `log.winner` is -1, `log.price` is 0, and nothing is clicked or converted.

| Auction | The winner pays |
|---|---|
| `SecondPrice` | The highest of m, r and the next campaign bid at or above r |
| `FirstPrice` | Its bid |

No bidder sees m. A winner sees only the price it paid, and a loser sees nothing. A bidder that declares the `floor` input gets r, as OpenRTB sends `bidfloor`. `RunLog.market_price` holds m per step, NaN without one, and `RunLog.floor` holds the market floor. `regret` replays against both, and `revenue` and `welfare` count only the auctions campaigns won.

### LogisticMarket

The market draws a hidden market vector ψ once per run: what outside advertisers value. Each step, with feature vector x and a standard normal ε:

```text
q = x · ψ
log m = log(price_median) + price_quality × q + price_spread × ε
floor = s × m, with s ~ Beta(a, b)
```

| Setting | Effect |
|---|---|
| `price_median` (None) | Median of m, in money; None means no market price and no floor |
| `price_spread` (0.5) | Standard deviation of log m around the quality term |
| `price_quality` (1.0) | How strongly log m follows the quality q, whose variance is near 1 |
| `rival_similarity` (0.0) | ρ in [0, 1]: each ad vector becomes ρ ψ + √(1 − ρ²) η, for its own vector η |
| `floor_share` (None) | (a, b) of the Beta share; None means no floor; needs `price_median` |

At ρ = 0, m follows the features but not any campaign's value. An impression a campaign values highly is then no dearer than any other. Set ρ above 0 so m rises with each campaign's value; at ρ = 0.8 and `price_spread` 0.5, the correlation of log m with a campaign's CVR logit is near 0.7. With ρ > 0, every pCTR and pCVR differ from the same market at ρ = 0, and each campaign's pCTR and pCVR move together more.

```python
import bidlab as bl

priced = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "pid", bl.Stack(bl.OracleValue(), bl.PidPacing()), budget=20, base_value=5
        )
    ],
    market=bl.markets.LogisticMarket(
        price_median=0.2, rival_similarity=0.8, floor_share=(2, 5)
    ),
    episodes=2,
    steps=200,
)
priced_log = bl.run(priced)
print((priced_log.winner == -1).mean())  # share of auctions sold outside
```

### IndependentValues

`IndependentValues(distribution, market_price=price_distribution, floor_share=(a, b))` draws m from any scipy frozen distribution, clipped at 0, and the floor as a Beta share of m. Here m does not follow any probability.

## Write a market

Subclass `bl.Market`. Its `draws` generator yields one `MarketDraw(features, pctr=..., pcvr=..., segment=...)` per step for the whole run; every argument after `features` is keyword-only. Leave `pctr` out for a market without a click stage. Pass `market_price=` and `floor=` for outside competition and a floor. Use only the `rng` it receives, so runs repeat with the seed.

```python
class Daytime(bl.Market):
    def draws(self, count, rng):
        step = 0
        while True:
            hour = step % 24
            rate = 0.08 if 9 <= hour < 18 else 0.02
            yield bl.markets.MarketDraw(
                {"hour": hour}, pcvr=rng.uniform(0.5, 1.5, size=count) * rate
            )
            step += 1


daytime = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "pid", bl.Stack(bl.OracleValue(), bl.PidPacing()), budget=15, base_value=5
        )
    ],
    market=Daytime(),
    episodes=2,
    steps=96,
)
print(bl.run(daytime).true_values.shape)  # (2, 96, 1)
```

A custom market can define its own segments. Override the `segment_names` property, and pass `segment=` to each `MarketDraw`. Targeting and `results.segment_rows` then work as they do for `Segmented`.

## Examples

| Script | Question |
|---|---|
| [markets/segments.py](../../examples/markets/segments.py) | Does a sports-only bidder win any auction outside sports? |
| [markets/custom_market.py](../../examples/markets/custom_market.py) | Does the all-day bidder pay more per win at night, when the night-only bidder joins? |
| [markets/independent_values.py](../../examples/markets/independent_values.py) | In a first-price auction with uniform values, does shading to (n-1)/n beat truthful bidding? |
| [markets/market_price.py](../../examples/markets/market_price.py) | Does a lone campaign pay more per conversion when the market price follows its value? |
