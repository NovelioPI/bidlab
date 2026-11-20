# Rewards

This guide shows the reward each bidder learns from, the built-in reward functions, and how to write your own.

## Step and episode rewards

Each `Campaign` has a `step_reward`, called after every auction, and an optional `episode_reward`, called once per episode. Both are plain functions.

| Kind | Built-in functions | Your own |
|---|---|---|
| Step | `bl.rewards.surplus` (default), `conversions`, `realized_value` | `f(request, result, params) -> float` |
| Episode | `bl.rewards.cpa_score`, `total_surplus` | `g(report, params) -> float` |

A step reward reads the realized value from `result.realized_value`. The bidder sees it as `result.reward`. An episode reward reaches the bidder as `on_episode_end(report, episode_reward)`.

## Parameters

The `params` dict holds the campaign's `reward_params`, plus its `cpa_target` when it has one.

```python
import bidlab as bl


def cheap_conversions(request, result, params):
    return float(result.converted) - params["cost_weight"] * result.price


campaign = bl.Campaign(
    "bandit",
    bl.UcbMultiplier(),
    budget=50,
    event_value=5,
    step_reward=cheap_conversions,
    reward_params={"cost_weight": 0.5},
)
```

## CPA targets

`cpa_score` is the AuctionNet score: the episode's value, cut by (target / CPA)³ when the CPA is above target. `bl.rewards.cpa_score_of` computes it from plain numbers, for your own episode rewards.

```python
cpa_campaign = bl.Campaign(
    "pid",
    bl.PidPacing(),
    budget=15,
    event_value=5,
    cpa_target=3,
    episode_reward=bl.rewards.cpa_score,
)
config = bl.SimConfig(campaigns=[campaign, cpa_campaign], episodes=2, steps=200)
print(bl.run(config).episode_rewards.shape)  # (2, 2)
```

## Examples

| Script | Question |
|---|---|
| [rewards/cpa_target.py](../../examples/rewards/cpa_target.py) | Does bidding target/value of the value meet a CPA target that truthful bidding misses? |
| [rewards/custom_reward.py](../../examples/rewards/custom_reward.py) | Does a bandit rewarded for conversions pick a higher multiplier than one rewarded for surplus? |
