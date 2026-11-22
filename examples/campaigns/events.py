"""With the same budget, does a click campaign pay less per click than a conversion campaign?

A campaign values one event: an impression, a click or a conversion. A click
campaign's value follows pCTR alone, so it bids most where clicks are likely.
A conversion campaign also weighs pCVR, so it pays for some impressions that
are rarely clicked. Both pay per won impression.
"""

import bidlab as bl

BUDGET = 15

config = bl.SimConfig(
    campaigns=[
        bl.Campaign(
            "click",
            bl.Stack(bl.OracleValue(), bl.PidPacing()),
            budget=BUDGET,
            base_value=0.4,
            event="click",
        ),
        bl.Campaign(
            "conversion",
            bl.Stack(bl.OracleValue(), bl.PidPacing()),
            budget=BUDGET,
            base_value=5,
            event=bl.Event.CONVERSION,
        ),
    ],
    market=bl.markets.LogisticMarket(ctr=0.05),
    episodes=5,
    steps=500,
)


def mean(results, campaign, column):
    values = [float(row[column]) for row in results.rows if row["campaign"] == campaign]
    return sum(values) / len(values)


if __name__ == "__main__":
    results = bl.run_many(config, seeds=range(5))
    for campaign in ("click", "conversion"):
        clicks = mean(results, campaign, "clicks")
        cpc = mean(results, campaign, "cpc")
        print(f"{campaign:>10}: {clicks:.1f} clicks per episode, {cpc:.3f} per click")
    click_cpc = mean(results, "click", "cpc")
    conversion_cpc = mean(results, "conversion", "cpc")
    answer = "yes" if click_cpc < conversion_cpc else "no"
    print(
        f"Answer: {answer}, the click campaign pays {click_cpc:.3f} per click"
        f" against {conversion_cpc:.3f}"
    )
