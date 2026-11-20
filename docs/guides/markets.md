# Markets

This guide shows the built-in markets, clicks and conversions, public features, segments and targeting, and how to write your own market.

## Built-in markets

A market draws each step's public features and each campaign's private probabilities. Each feature reaches `forward` as an input of the same name, which must not be a built-in input such as `time`, and all of them together as the `features` input. Pass one as `SimConfig(market=...)`.

| Market | Draws |
|---|---|
| `bl.markets.LogisticMarket()` | The default: pCVRᵢ = σ(x · φᵢ + βᵢ) for a random feature vector x; optionally a click stage |
| `bl.markets.Segmented(segments)` | A segment per step by its `share`; its `lift` is added to every campaign's CVR logit |
| `bl.markets.IndependentValues(distribution)` | Each pᵢ drawn on its own from a scipy frozen distribution, clipped to [0, 1] |

`IndependentValues(scipy.stats.uniform(0, 1))` with a campaign `event_value` of 1 is the textbook independent private values setting.

## Clicks and conversions

A won impression is clicked with probability pCTR, and a click converts with probability pCVR. A market without a click stage counts every won impression as clicked, so pCVR is the conversion chance of a win.

`LogisticMarket(ctr_bias_mean=-4.0)` adds a click stage: pCTRᵢ = σ(x · ψᵢ + γᵢ), with its own ad vectors ψᵢ and biases γᵢ ~ N(`ctr_bias_mean`, `ctr_bias_sd`) on the same feature vector x. Without `ctr_bias_mean`, the market has no click stage and gives the same runs as before clicks existed.

The true pCTR × pCVR sets `RunLog.true_values`. `RunLog.clicked`, `BidResult.clicked` and the `clicks` column record the clicks.

A bidder that declares the `market` input gets a `bl.MarketTruth` with its own true `pctr` and `pcvr`; `pctr` is 1.0 without a click stage. Use it to check an estimator or to build an oracle. A fair benchmark bidder does not read it.

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
        bl.markets.Segment("sports", share=0.3, lift=1.0),
    )
)
segmented = bl.SimConfig(
    campaigns=[
        bl.Campaign("everywhere", bl.Linear(), budget=50, event_value=5),
        bl.Campaign(
            "sports_only",
            bl.Linear(),
            budget=50,
            event_value=5,
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

## Write a market

Subclass `bl.markets.Market`. Its `draws` generator yields one `MarketDraw(features, pcvr, segment, pctr)` per step for the whole run. Leave `pctr` out for a market without a click stage. Use only the `rng` it receives, so runs repeat with the seed.

```python
class Daytime(bl.markets.Market):
    def draws(self, count, rng):
        step = 0
        while True:
            hour = step % 24
            rate = 0.08 if 9 <= hour < 18 else 0.02
            yield bl.markets.MarketDraw(
                {"hour": hour}, rng.uniform(0.5, 1.5, size=count) * rate
            )
            step += 1


daytime = bl.SimConfig(
    campaigns=[bl.Campaign("pid", bl.PidPacing(), budget=15, event_value=5)],
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
