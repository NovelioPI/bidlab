# Rewards

This guide shows the reward each bidder learns from, the built-in reward functions, and how to write your own.

## Step and episode rewards

Each `BidderSpec` has a `reward`, called after every auction, and an optional `episode_reward`, called once per episode. Both are plain functions.

| Kind | Built-in functions | Your own |
|---|---|---|
| Step | `bl.rewards.surplus` (default), `conversions`, `value` | `f(observation, feedback, params) -> float` |
| Episode | `bl.rewards.cpa_score`, `total_surplus` | `g(stats, params) -> float` |

A step reward reads the realized value from `feedback.value`. The bidder sees it as `feedback.reward`. An episode reward reaches the bidder as `on_episode_end(stats, episode_reward)`.

## Parameters

The `params` dict holds the spec's `reward_params`, plus the campaign's `cpa_target` when it has one.

```python
import bidlab as bl


def cheap_conversions(observation, feedback, params):
    return float(feedback.converted) - params["cost_weight"] * feedback.price


spec = bl.BidderSpec(
    "bandit",
    bl.UcbMultiplier(),
    bl.Campaign(budget=50, value=5),
    reward=cheap_conversions,
    reward_params={"cost_weight": 0.5},
)
```

## CPA targets

`cpa_score` is the AuctionNet score: the episode's value, cut by (target / CPA)³ when the CPA is above target. `bl.rewards.cpa_score_of` computes it from plain numbers, for your own episode rewards.

```python
cpa_spec = bl.BidderSpec(
    "pid",
    bl.PidPacing(),
    bl.Campaign(budget=15, value=5, cpa_target=3),
    episode_reward=bl.rewards.cpa_score,
)
config = bl.SimConfig(bidders=[spec, cpa_spec], episodes=2, steps=200)
print(bl.run(config).episode_rewards.shape)  # (2, 2)
```

## Examples

| Script | Question |
|---|---|
| [rewards/cpa_target.py](../../examples/rewards/cpa_target.py) | Does bidding target/value of the value meet a CPA target that truthful bidding misses? |
| [rewards/custom_reward.py](../../examples/rewards/custom_reward.py) | Does a bandit rewarded for conversions pick a higher multiplier than one rewarded for surplus? |
