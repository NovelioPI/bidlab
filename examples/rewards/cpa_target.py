"""Does bidding target/value of the value meet a CPA target that truthful bidding misses?

In a second-price auction a bidder never pays more than its bid. A bid of
target/value times the value costs at most the target per expected conversion.
`cpa_score` rewards an episode by its value, cut sharply when the CPA is above
target.
"""

import bidlab as bl

CPA_TARGET = 1.5
VALUE = 5
SAFETY = 0.8


def strict_cpa_score(report: bl.EpisodeReport, params: dict[str, float]) -> float:
    """`cpa_score` against a target lowered by the campaign's safety factor."""
    target = params["cpa_target"] * params["safety"]
    return bl.rewards.cpa_score_of(
        report.realized_value, report.spend, report.conversions, target
    )


def campaign(name, bidder):
    return bl.Campaign(
        name,
        bidder,
        budget=50,
        event_value=VALUE,
        cpa_target=CPA_TARGET,
        episode_reward=bl.rewards.cpa_score,
    )


config = bl.SimConfig(
    campaigns=[
        campaign("truthful", bl.Linear()),
        campaign("at_target", bl.Linear(multiplier=CPA_TARGET / VALUE)),
        bl.Campaign(
            "strict",
            bl.Linear(multiplier=CPA_TARGET * SAFETY / VALUE),
            budget=50,
            event_value=VALUE,
            cpa_target=CPA_TARGET,
            episode_reward=strict_cpa_score,
            reward_params={"safety": SAFETY},
        ),
        bl.Campaign("big", bl.Linear(), budget=50, event_value=VALUE),
    ],
    auction=bl.SecondPrice(reserve=0.01),
    episodes=10,
    steps=500,
)


def mean(results, bidder, column):
    values = [float(row[column]) for row in results.rows if row["campaign"] == bidder]
    return sum(values) / len(values)


if __name__ == "__main__":
    results = bl.run_many(config, seeds=range(5))
    print(results.summary())
    at_target = mean(results, "at_target", "cpa")
    truthful = mean(results, "truthful", "cpa")
    answer = "yes" if at_target <= CPA_TARGET < truthful else "no"
    print(
        f"Answer: {answer}, CPA {at_target:.2f} at target/value against"
        f" {truthful:.2f} truthful, target {CPA_TARGET}"
    )
