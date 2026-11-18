import numpy as np
import pytest

from bidlab import bidders, rewards
from bidlab.auctions import FirstPrice
from bidlab.bidders import PidPacing, Truthful, UcbMultiplier
from bidlab.config import BidderSpec, SimConfig
from bidlab.simulator import run
from bidlab.types import Campaign

STEPS = 2000


def competitors():
    return (
        BidderSpec("x", Truthful(), Campaign(1e6, 4.0)),
        BidderSpec("y", Truthful(multiplier=0.8), Campaign(1e6, 6.0)),
        BidderSpec("z", Truthful(multiplier=0.9), Campaign(1e6, 5.0)),
    )


def spend_curve(bidder, budget, seed=11):
    me = BidderSpec("me", bidder, Campaign(budget, 5.0))
    log = run(
        SimConfig(bidders=(me, *competitors()), episodes=1, steps=STEPS, seed=seed)
    )
    paid = np.where(log.winner[0] == 0, log.price[0], 0.0)
    return np.cumsum(paid)


def test_pid_spends_near_budget():
    unconstrained = spend_curve(Truthful(), 1e6)[-1]
    budget = 0.3 * unconstrained
    truthful = spend_curve(Truthful(), budget)
    pid = spend_curve(PidPacing(), budget)
    # Reason: the cap lets a near-empty bidder spend crumbs, so "out" means 99%.
    assert truthful[STEPS // 2] >= 0.99 * budget
    # PID ends within 10% of the budget and still has budget left at 80% of the episode.
    assert 0.9 * budget <= pid[-1] <= budget
    assert pid[int(0.8 * STEPS)] < budget


def bandit_log(reward, episodes=80, seed=5):
    me = BidderSpec("me", UcbMultiplier(), Campaign(1e6, 5.0), reward=reward)
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


def test_pid_spends_evenly_alone():
    pid = BidderSpec("pid", PidPacing(), Campaign(budget=10, value=5))
    log = run(SimConfig(bidders=(pid,), auction=FirstPrice(), episodes=1, steps=200))
    first, second = log.price[0, :100].sum(), log.price[0, 100:].sum()
    assert 4 <= first <= 6
    assert 4 <= second <= 6
    assert first + second <= 10 + 1e-12


@pytest.mark.parametrize(
    ("build", "argument"),
    [
        (lambda: PidPacing(kp=-1), "kp"),
        (lambda: PidPacing(ki=-1), "ki"),
        (lambda: PidPacing(kd=-1), "kd"),
        (lambda: PidPacing(initial_multiplier=0), "initial_multiplier"),
        (lambda: PidPacing(multiplier_range=(2, 1)), "multiplier_range"),
        (lambda: PidPacing(multiplier_range=(-1, 1)), "multiplier_range"),
        (lambda: UcbMultiplier(multiplier_step=0), "multiplier_step"),
        (lambda: UcbMultiplier(min_multiplier=-1), "min_multiplier"),
        (lambda: UcbMultiplier(exploration=-1), "exploration"),
        (lambda: Truthful(multiplier=-1), "multiplier"),
    ],
)
def test_stage_arguments_validated(build, argument):
    with pytest.raises(ValueError, match=argument):
        build()


def test_pid_gains_as_arguments():
    pid = PidPacing(kp=30, ki=1, kd=0.5)
    assert (pid.kp, pid.ki, pid.kd) == (30.0, 1.0, 0.5)
    spec = BidderSpec("pid", pid, Campaign(budget=10.0, value=5.0))
    log = run(SimConfig(bidders=[spec], episodes=1, steps=50))
    assert log.bids.shape == (1, 50, 1)


def test_pid_gains_removed():
    assert not hasattr(bidders, "PidGains")
    with pytest.raises(TypeError, match="gains"):
        PidPacing(gains=None)  # pyright: ignore[reportCallIssue]  Reason: the test checks the removed argument fails at run time.
