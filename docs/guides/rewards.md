# Rewards

This guide shows how a bidder defines the reward it learns from, and how to record it.

## A reward is the bidder's own code

bidlab has no reward setting. A bidder computes its reward from what it learns, in `update` or `on_episode_end`, and learns from it there. This keeps the learning signal with the algorithm, as a PyTorch module keeps its loss with its training step.

| Hook | Reads | Typical reward |
|---|---|---|
| `update(result)` | One `BidResult` | `result.realized_value - result.price`, `float(result.converted)` |
| `on_episode_end(report)` | The episode's `EpisodeReport` | `report.realized_value - report.spend` |
| `on_episode_start(campaign)` | The `Campaign` | Settings such as `campaign.cpa_target` |

To see the reward in results, record it with `self.log`. A value logged in `update` lands per auction; one logged in `on_episode_end` lands per episode. See [Logging](logging.md).

```python
import bidlab as bl


class CheapConversions(bl.Bidder):
    """Learns nothing; logs a conversion reward that charges for price."""

    def __init__(self, cost_weight=0.5):
        self.cost_weight = cost_weight

    def forward(self, value):
        return value

    def update(self, result):
        reward = float(result.converted) - self.cost_weight * result.price
        self.log("reward", reward)


campaign = bl.Campaign("cheap", CheapConversions(), budget=50, base_value=5)
config = bl.SimConfig(campaigns=[campaign], episodes=2, steps=200)
print(bl.run(config).logged["reward"]["cheap"].shape)  # (2, 200)
```

## A learning bidder with another reward

`UcbMultiplier` sums the surplus, `realized_value - price`, in `update` into `arm_reward`, and learns from it at the episode end. Override `update` to reward something else.

```python
class ConversionBandit(bl.UcbMultiplier):
    def update(self, result):
        self.arm_reward += float(result.converted)
```

## Comparing bidders

Each bidder may define its reward differently, so compare bidders on metrics instead: `surplus`, `cpa_score` and `regret` in [Results](results.md). `cpa_score` is the AuctionNet score: the episode's value, cut by (target / CPA)³ when the CPA is above the campaign's `cpa_target`.

## Examples

| Script | Question |
|---|---|
| [rewards/cpa_target.py](../../examples/rewards/cpa_target.py) | Does bidding target/value of the value meet a CPA target that truthful bidding misses? |
| [rewards/custom_reward.py](../../examples/rewards/custom_reward.py) | Does a bandit rewarded for conversions pick a higher multiplier than one rewarded for surplus? |
