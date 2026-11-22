"""Does a sports-only bidder win any auction outside sports?

A campaign with `segments` bids only there. Elsewhere its bidder is not called
and bids -inf, so it cannot win even at a reserve of 0.
"""

import bidlab as bl

market = bl.markets.Segmented(
    (
        bl.markets.Segment("news", share=0.5),
        bl.markets.Segment("sports", share=0.3, lift=1.0),
        bl.markets.Segment("games", share=0.2, lift=-0.5),
    )
)

config = bl.SimConfig(
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
        bl.Campaign(
            "pid", bl.Stack(bl.OracleValue(), bl.PidPacing()), budget=15, base_value=5
        ),
    ],
    market=market,
    auction=bl.SecondPrice(reserve=0.01),
    episodes=10,
    steps=1000,
)

if __name__ == "__main__":
    results = bl.run_many(config, seeds=range(5))
    assert results.segment_rows is not None
    print(results.segment_rows.summary())
    wins = {"sports": 0.0, "other": 0.0}
    for row in results.segment_rows.rows:
        if row["campaign"] == "sports_only":
            where = "sports" if row["segment"] == "sports" else "other"
            wins[where] += float(row["wins"])

    answer = "yes" if wins["other"] > 0 else "no"
    print(
        f"Answer: {answer}, {wins['other']:.0f} wins outside sports"
        f" and {wins['sports']:.0f} in sports"
    )
