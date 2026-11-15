import numpy as np

from bidlab import rewards
from bidlab.bidders import PidPacing, Truthful, UcbMultiplier
from bidlab.config import BidderSpec, SimConfig
from bidlab.simulator import run

STEPS = 2000


def competitors():
    return (
        BidderSpec("x", Truthful, 1e6, 4.0),
        BidderSpec("y", Truthful, 1e6, 6.0, params={"multiplier": 0.8}),
        BidderSpec("z", Truthful, 1e6, 5.0, params={"multiplier": 0.9}),
    )


def spend_curve(bidder_class, budget, seed=11):
    me = BidderSpec("me", bidder_class, budget, 5.0)
    log = run(
        SimConfig(bidders=(me, *competitors()), episodes=1, steps=STEPS, seed=seed)
    )
    paid = np.where(log.winner[0] == 0, log.price[0], 0.0)
    return np.cumsum(paid)


def test_pid_spends_near_budget():
    unconstrained = spend_curve(Truthful, 1e6)[-1]
    budget = 0.3 * unconstrained
    truthful = spend_curve(Truthful, budget)
    pid = spend_curve(PidPacing, budget)
    # Reason: the cap lets a near-empty bidder spend crumbs, so "out" means 99%.
    assert truthful[STEPS // 2] >= 0.99 * budget
    # PID ends within 10% of the budget and still has budget left at 80% of the episode.
    assert 0.9 * budget <= pid[-1] <= budget
    assert pid[int(0.8 * STEPS)] < budget


def bandit_log(reward, episodes=80, seed=5):
    me = BidderSpec("me", UcbMultiplier, 1e6, 5.0, reward=reward)
    return run(
        SimConfig(bidders=(me, *competitors()), episodes=episodes, steps=300, seed=seed)
    )


def multiplier_per_episode(log):
    return np.median(log.bids[:, :, 0] / log.values[:, :, 0], axis=1)


def test_bandit_improves():
    log = bandit_log(rewards.surplus)
    surplus = log.rewards[:, :, 0].sum(axis=1)
    fifth = len(surplus) // 5
    assert surplus[-fifth:].mean() > surplus[:fifth].mean()


def most_pulled_multiplier(log):
    values, counts = np.unique(
        np.round(multiplier_per_episode(log), 2), return_counts=True
    )
    return values[np.argmax(counts)]


def test_bandit_uses_configured_reward():
    by_surplus = most_pulled_multiplier(bandit_log(rewards.surplus, episodes=300))
    by_conversions = most_pulled_multiplier(
        bandit_log(rewards.conversions, episodes=300)
    )
    assert by_conversions > by_surplus
