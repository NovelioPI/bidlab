import numpy as np
import pytest

from auction_sim.bidders import Truthful, resolve_bidder
from auction_sim.rewards import resolve_episode_reward, resolve_step_reward
from auction_sim.types import EpisodeStats, Feedback, Observation


def test_builtin_bidder_resolves():
    assert resolve_bidder("truthful") is Truthful


def test_custom_bidder_by_module_path():
    bidder_class = resolve_bidder("sample_plugins:Doubler")
    bidder = bidder_class({}, np.random.default_rng(0))
    assert (
        bidder.bid(
            Observation(
                step=0,
                steps_left=1,
                value=0.3,
                value_per_conversion=3.0,
                budget_left=1.0,
            )
        )
        == 0.6
    )


def test_unknown_bidder_lists_names():
    with pytest.raises(ValueError, match="truthful"):
        resolve_bidder("nonexistent")


def test_module_path_to_non_bidder_raises():
    with pytest.raises(TypeError, match="not a subclass"):
        resolve_bidder("sample_plugins:NotABidder")
    with pytest.raises(ValueError, match="has no"):
        resolve_bidder("sample_plugins:Missing")


OBS = Observation(
    step=0, steps_left=1, value=0.3, value_per_conversion=4.0, budget_left=1.0
)


def test_builtin_step_rewards():
    converted = Feedback(bid=0.5, won=True, price=0.25, converted=True)
    lost = Feedback(bid=0.1, won=False, price=0.0, converted=False)
    assert resolve_step_reward("surplus")(OBS, converted, {}) == 3.75
    assert resolve_step_reward("conversions")(OBS, converted, {}) == 1.0
    assert resolve_step_reward("value")(OBS, converted, {}) == 4.0
    assert resolve_step_reward("surplus")(OBS, lost, {}) == 0.0


def test_reward_by_module_path():
    feedback = Feedback(bid=0.5, won=True, price=0.25, converted=False)
    assert resolve_step_reward("sample_plugins:double_price")(OBS, feedback, {}) == 0.5
    stats = EpisodeStats(budget=10, spend=3, wins=5, conversions=1, value=4)
    assert resolve_episode_reward("sample_plugins:total_spend")(stats, {}) == 3


@pytest.mark.parametrize(
    ("spend", "conversions", "expected"),
    [
        (0.0, 0, 8.0),  # nothing spent: no penalty
        (2.0, 4, 8.0),  # cpa 0.5 <= target 1
        (4.0, 2, 8.0 * 0.5**3),  # cpa 2 = twice the target
        (3.0, 0, 0.0),  # spent with no conversion
    ],
)
def test_cpa_score_matches_auctionnet(spend, conversions, expected):
    stats = EpisodeStats(
        budget=10, spend=spend, wins=4, conversions=conversions, value=8.0
    )
    assert resolve_episode_reward("cpa_score")(stats, {"cpa_target": 1.0}) == expected


def test_cpa_score_without_target_raises():
    stats = EpisodeStats(budget=10, spend=1, wins=1, conversions=1, value=1)
    with pytest.raises(ValueError, match="cpa_target"):
        resolve_episode_reward("cpa_score")(stats, {})
