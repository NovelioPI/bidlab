import numpy as np
import pytest

from auction_sim.config import AuctionConfig, BidderSpec, SimConfig
from auction_sim.metrics import bidder_metrics, platform_metrics, regret
from auction_sim.simulator import RunLog, run


def hand_log(kind="second", reserve=0.0, budget=10.0, cpa_target=None):
    """One episode, 3 steps, 2 bidders, with every number chosen by hand."""
    bidders = (
        BidderSpec(
            "a", "truthful", budget, value_per_conversion=4.0, cpa_target=cpa_target
        ),
        BidderSpec("b", "truthful", 10.0, value_per_conversion=2.0),
    )
    config = SimConfig(
        bidders=bidders, episodes=1, steps=3, auction=AuctionConfig(kind, reserve)
    )
    values = np.array([[[1.0, 0.5], [0.2, 0.6], [0.8, 0.3]]])
    bids = values.copy()
    return RunLog(
        config=config,
        bids=bids,
        values=values,
        winner=np.array([[0, 1, 0]]),
        price=np.array([[0.5, 0.2, 0.3]]),
        converted=np.array([[True, False, False]]),
        rewards=np.array([[[1.0, 0.0], [0.0, 2.0], [0.5, 0.0]]]),
        episode_rewards=np.array([[np.nan, np.nan]]),
    )


def test_metrics_on_hand_built_log():
    a, b = bidder_metrics(hand_log())
    assert (a.spend, a.wins, a.conversions, a.value) == (0.8, 2, 1, 4.0)
    assert a.surplus == pytest.approx(3.2)
    assert a.budget_util == pytest.approx(0.08)
    assert a.cpa == pytest.approx(0.8)
    assert np.isnan(a.cpa_violation) and np.isnan(a.cpa_score)
    assert a.reward == 1.5 and np.isnan(a.episode_reward)
    assert (b.spend, b.wins, b.conversions, b.value) == (0.2, 1, 0, 0.0)
    assert np.isnan(b.cpa)
    (platform,) = platform_metrics(hand_log())
    assert platform.revenue == pytest.approx(1.0)
    assert platform.welfare == pytest.approx(1.0 + 0.6 + 0.8)


@pytest.mark.parametrize(
    ("target", "violation", "score"),
    [
        (1.6, 0.0, 4.0),  # cpa 0.8 within target
        (0.4, 1.0, 4.0 * 0.5**3),  # cpa twice the target
    ],
)
def test_cpa_violation_cases(target, violation, score):
    a, _ = bidder_metrics(hand_log(cpa_target=target))
    assert a.cpa_violation == pytest.approx(violation)
    assert a.cpa_score == pytest.approx(score)


def test_cpa_violation_infinite_when_spend_without_conversions():
    log = hand_log(cpa_target=1.0)
    no_conversions = RunLog(
        **{**log.__dict__, "converted": np.zeros((1, 3), dtype=bool)}
    )
    a, _ = bidder_metrics(no_conversions)
    assert a.cpa_violation == np.inf
    assert a.cpa_score == 0.0


def test_regret_on_hand_built_log():
    # Second price, others' bids (0.5, 0.6, 0.3). Own surplus: (1-0.5)+(0.8-0.3) = 1.0.
    # Multiplier 3 wins all three: (1-0.5)+(0.2-0.6)+(0.8-0.3) = 0.6. Best is 1.0.
    assert regret(hand_log(), 0, 0) == pytest.approx(0.0)
    # First price, multiplier m wins step 1 if m > 0.5 and step 3 if 0.8m > 0.3.
    # Best: m = 0.55 pays 0.55 and 0.44, surplus 0.45 + 0.36 = 0.81. Own surplus 0.5+0.5.
    assert regret(hand_log(kind="first"), 0, 0) == pytest.approx(0.81 - 1.0)


def test_regret_respects_budget():
    # Budget 0.5 at multiplier 1: step 1 bid capped to 0.5 ties 0.5 and loses.
    log = hand_log(budget=0.5)
    best = regret(log, 0, 0, multipliers=np.array([1.0]))
    assert best == pytest.approx((0.8 - 0.3) - 1.0)


def test_regret_nonnegative_for_fixed_multipliers():
    for multiplier in (0.5, 1.0, 1.5):
        bidders = (
            BidderSpec("me", "truthful", 50.0, 5.0, params={"multiplier": multiplier}),
            BidderSpec("x", "truthful", 50.0, 4.0),
            BidderSpec("y", "truthful", 50.0, 6.0, params={"multiplier": 0.7}),
        )
        log = run(SimConfig(bidders=bidders, episodes=2, steps=300, seed=3))
        for episode in range(2):
            assert regret(log, episode, 0) >= -1e-9
