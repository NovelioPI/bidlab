import numpy as np
import pytest

from bidlab import bidders
from bidlab.auctions import FirstPrice
from bidlab.bidders import Linear, OracleValue, PidPacing, Stack, UcbMultiplier
from bidlab.config import Campaign, SimConfig
from bidlab.simulator import run
from bidlab.types import BidResult, EpisodeReport

STEPS = 2000


def competitors():
    return (
        Campaign("x", Stack(OracleValue(), Linear()), budget=1e6, base_value=4.0),
        Campaign(
            "y",
            Stack(OracleValue(), Linear(multiplier=0.8)),
            budget=1e6,
            base_value=6.0,
        ),
        Campaign(
            "z",
            Stack(OracleValue(), Linear(multiplier=0.9)),
            budget=1e6,
            base_value=5.0,
        ),
    )


def spend_curve(bidder, budget, seed=11):
    me = Campaign("me", Stack(OracleValue(), bidder), budget=budget, base_value=5.0)
    log = run(
        SimConfig(campaigns=(me, *competitors()), episodes=1, steps=STEPS), seed=seed
    )
    paid = log.price[0, :, 0]
    return np.cumsum(paid)


def test_pid_spends_near_budget():
    unconstrained = spend_curve(Linear(), 1e6)[-1]
    budget = 0.3 * unconstrained
    truthful = spend_curve(Linear(), budget)
    pid = spend_curve(PidPacing(), budget)
    # Reason: the cap lets a near-empty bidder spend crumbs, so "out" means 99%.
    assert truthful[STEPS // 2] >= 0.99 * budget
    # PID ends within 10% of the budget and still has budget left at 80% of the episode.
    assert 0.9 * budget <= pid[-1] <= budget
    assert pid[int(0.8 * STEPS)] < budget


class ConversionBandit(UcbMultiplier):
    """Rewards each conversion and ignores the price."""

    def update(self, result):
        self.arm_reward += float(result.converted)


def bandit_log(bandit, episodes=80, seed=5):
    me = Campaign("me", Stack(OracleValue(), bandit), budget=1e6, base_value=5.0)
    return run(
        SimConfig(campaigns=(me, *competitors()), episodes=episodes, steps=300),
        seed=seed,
    )


def multiplier_per_episode(log):
    return np.median(log.bids[:, :, 0] / log.true_values[:, :, 0], axis=1)


def test_bandit_improves():
    log = bandit_log(UcbMultiplier())
    won = log.won[:, :, 0]
    paid = np.where(won, log.price[:, :, 0], 0.0).sum(axis=1)
    surplus = np.where(won & log.converted[:, :, 0], 5.0, 0.0).sum(axis=1) - paid
    fifth = len(surplus) // 5
    assert surplus[-fifth:].mean() > surplus[:fifth].mean()


def most_pulled_multiplier(log):
    values, counts = np.unique(
        np.round(multiplier_per_episode(log), 2), return_counts=True
    )
    return values[np.argmax(counts)]


def test_bandit_update_override_changes_reward():
    by_surplus = most_pulled_multiplier(bandit_log(UcbMultiplier(), episodes=300))
    by_conversions = most_pulled_multiplier(
        bandit_log(ConversionBandit(), episodes=300)
    )
    assert by_conversions > by_surplus


def test_ucb_reward_is_episode_surplus():
    bandit = UcbMultiplier()
    bandit.bind(np.random.default_rng(0))
    bandit.on_episode_start(Campaign("me", bandit, budget=10, base_value=5))
    results = [
        BidResult(
            time=0.0,
            bid=1.0,
            won=True,
            price=0.4,
            clicked=True,
            converted=True,
            realized_value=5.0,
        ),
        BidResult(
            time=1.0, bid=1.0, won=True, price=0.7, clicked=True, converted=False
        ),
        BidResult(
            time=2.0, bid=1.0, won=False, price=0.0, clicked=False, converted=False
        ),
    ]
    bandit.update_batch(results)
    bandit.on_episode_end(
        EpisodeReport(
            budget=10, spend=1.1, wins=2, clicks=2, conversions=1, realized_value=5.0
        )
    )
    assert bandit.reward_sums[bandit.arm] == (5.0 - 0.4) + (0.0 - 0.7) + 0.0
    assert bandit.pulls[bandit.arm] == 1


def test_pid_spends_evenly_alone():
    pid = Campaign("pid", Stack(OracleValue(), PidPacing()), budget=10, base_value=5)
    log = run(SimConfig(campaigns=(pid,), auction=FirstPrice(), episodes=1, steps=200))
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
        (lambda: Linear(multiplier=-1), "multiplier"),
    ],
)
def test_stage_arguments_validated(build, argument):
    with pytest.raises(ValueError, match=argument):
        build()


def test_pid_gains_as_arguments():
    pid = PidPacing(kp=30, ki=1, kd=0.5)
    assert (pid.kp, pid.ki, pid.kd) == (30.0, 1.0, 0.5)
    campaign = Campaign("pid", Stack(OracleValue(), pid), budget=10.0, base_value=5.0)
    log = run(SimConfig(campaigns=[campaign], episodes=1, steps=50))
    assert log.bids.shape == (1, 50, 1)


def test_pid_gains_removed():
    assert not hasattr(bidders, "PidGains")
    with pytest.raises(TypeError, match="gains"):
        PidPacing(gains=None)  # pyright: ignore[reportCallIssue]  Reason: the test checks the removed argument fails at run time.
