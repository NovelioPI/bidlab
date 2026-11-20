# Markets

This guide shows the built-in markets, segments and targeting, and how to write your own market.

## Built-in markets

A market draws each step's public features and each bidder's private conversion probability. Pass one as `SimConfig(market=...)`.

| Market | Draws |
|---|---|
| `bl.markets.LogisticContext()` | The default: pᵢ = σ(x · φᵢ + βᵢ) for a random context x |
| `bl.markets.Segmented(segments)` | A segment per step by its `share`; its `lift` is added to every bidder's logit |
| `bl.markets.IndependentValues(distribution)` | Each pᵢ drawn on its own from a scipy frozen distribution, clipped to [0, 1] |

`IndependentValues(scipy.stats.uniform(0, 1))` with `Campaign(value=1)` is the textbook independent private values setting.

## Segments and targeting

A segmented market puts the segment name in `observation.features["segment"]`. A campaign with `segments` bids only in those segments. Elsewhere its bidder is not called, and its logged bid is 0.

```python
import bidlab as bl

market = bl.markets.Segmented(
    (
        bl.markets.Segment("news", share=0.7),
        bl.markets.Segment("sports", share=0.3, lift=1.0),
    )
)
segmented = bl.SimConfig(
    bidders=[
        bl.BidderSpec("everywhere", bl.Truthful(), bl.Campaign(budget=50, value=5)),
        bl.BidderSpec(
            "sports_only",
            bl.Truthful(),
            bl.Campaign(budget=50, value=5, segments=("sports",)),
        ),
    ],
    market=market,
    episodes=2,
    steps=200,
)
segment_results = bl.run_many(segmented, seeds=range(2))
print(len(segment_results.segments))  # 8 rows: seeds × bidders × segments
```

`results.segments` is a second table with spend, wins, conversions and surplus per seed, bidder and segment. A market without segments reports one segment, `"all"`.

A targeted bidder sees only its segments' auctions. `PidPacing` plans spend by time, so it still spreads the budget over the whole horizon.

## Write a market

Subclass `bl.markets.Market`. Its `draws` generator yields one `MarketDraw` per step for the whole run. Use only the `rng` it receives, so runs repeat with the seed.

```python
class Daytime(bl.markets.Market):
    def draws(self, bidders, rng):
        step = 0
        while True:
            hour = step % 24
            rate = 0.08 if 9 <= hour < 18 else 0.02
            yield bl.markets.MarketDraw(
                {"hour": hour}, rng.uniform(0.5, 1.5, size=bidders) * rate
            )
            step += 1


daytime = bl.SimConfig(
    bidders=[bl.BidderSpec("pid", bl.PidPacing(), bl.Campaign(budget=15, value=5))],
    market=Daytime(),
    episodes=2,
    steps=96,
)
print(bl.run(daytime).values.shape)  # (2, 96, 1)
```

A custom market can define its own segments. Override the `segment_names` property, and pass `segment=` to each `MarketDraw`. Targeting and `results.segments` then work as they do for `Segmented`.

## Examples

| Script | Question |
|---|---|
| [markets/segments.py](../../examples/markets/segments.py) | Does a sports-only bidder win any auction outside sports? |
| [markets/custom_market.py](../../examples/markets/custom_market.py) | Does the all-day bidder pay more per win at night, when the night-only bidder joins? |
| [markets/independent_values.py](../../examples/markets/independent_values.py) | In a first-price auction with uniform values, does shading to (n-1)/n beat truthful bidding? |
