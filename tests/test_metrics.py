import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest

from bidlab.auctions import NO_WINNER, FirstPrice, SecondPrice
from bidlab.bidders import Truthful
from bidlab.config import BidderSpec, SimConfig
from bidlab.markets import Segment, Segmented
from bidlab.metrics import bidder_metrics, platform_metrics, regret
from bidlab.simulator import RunLog, run


def hand_log(kind="second", reserve=0.0, budget=10.0, cpa_target=None):
    """One episode, 3 steps, 2 bidders, with every number chosen by hand."""
    bidders = (
        BidderSpec(
            "a", Truthful, budget, value_per_conversion=4.0, cpa_target=cpa_target
        ),
        BidderSpec("b", Truthful, 10.0, value_per_conversion=2.0),
    )
    config = SimConfig(
        bidders=bidders,
        episodes=1,
        steps=3,
        auction=FirstPrice(reserve) if kind == "first" else SecondPrice(reserve),
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
        segment=np.zeros((1, 3), dtype=np.int64),
    )


def test_metrics_on_hand_built_log():
    a, b = bidder_metrics(hand_log())
    assert (a.spend, a.wins, a.conversions, a.value) == (0.8, 2, 1, 4.0)
    assert a.surplus == pytest.approx(3.2)
    assert a.budget_used == pytest.approx(0.08)
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
            BidderSpec("me", Truthful, 50.0, 5.0, params={"multiplier": multiplier}),
            BidderSpec("x", Truthful, 50.0, 4.0),
            BidderSpec("y", Truthful, 50.0, 6.0, params={"multiplier": 0.7}),
        )
        log = run(SimConfig(bidders=bidders, episodes=2, steps=300, seed=3))
        for episode in range(2):
            assert regret(log, episode, 0) >= -1e-9


def test_regret_accepts_integer_budget():
    def config(budget):
        bidders = (
            BidderSpec("me", Truthful, budget, 5.0),
            BidderSpec("x", Truthful, 50.0, 4.0),
        )
        return SimConfig(bidders=bidders, episodes=1, steps=200, seed=4)

    integer_log = run(config(3))
    float_log = run(config(3.0))
    assert regret(integer_log, 0, 0) == regret(float_log, 0, 0)


@dataclass(frozen=True)
class Episode:
    """One episode's logged arrays, written per step."""

    values: list[list[float]]
    bids: list[list[float]]
    winner: list[int]
    price: list[float]


def one_episode_log(episode, bidders, auction):
    steps, count = len(episode.values), len(bidders)
    return RunLog(
        config=SimConfig(bidders=bidders, episodes=1, steps=steps, auction=auction),
        bids=np.array([episode.bids], dtype=float),
        values=np.array([episode.values], dtype=float),
        winner=np.array([episode.winner]),
        price=np.array([episode.price], dtype=float),
        converted=np.zeros((1, steps), dtype=bool),
        rewards=np.zeros((1, steps, count)),
        episode_rewards=np.full((1, count), np.nan),
        segment=np.zeros((1, steps), dtype=np.int64),
    )


def test_cpa_metrics_divide_spend_by_conversions():
    # a wins steps 0 and 2 for 0.5 + 0.3 and converts twice: cpa 0.8 / 2 = 0.4.
    log = hand_log(cpa_target=0.2)
    two_conversions = RunLog(
        **{**log.__dict__, "converted": np.array([[True, False, True]])}
    )
    a, _ = bidder_metrics(two_conversions)
    assert a.cpa == pytest.approx(0.4)
    assert a.cpa_violation == pytest.approx(0.4 / 0.2 - 1)


def test_no_spend_means_no_cpa_violation():
    # a never wins, so it spends nothing and converts nothing.
    log = hand_log(cpa_target=1.0)
    never_wins = RunLog(**{**log.__dict__, "winner": np.array([[1, 1, 1]])})
    a, _ = bidder_metrics(never_wins)
    assert a.spend == 0.0
    assert a.cpa_violation == 0.0
    assert a.cpa_score == 0.0


def test_metric_rows_carry_their_episode():
    bidders = (
        BidderSpec("a", Truthful, 50.0, 5.0),
        BidderSpec("b", Truthful, 50.0, 4.0),
    )
    log = run(SimConfig(bidders=bidders, episodes=2, steps=50, seed=1))
    assert [row.episode for row in platform_metrics(log)] == [0, 1]
    assert [row.episode for row in bidder_metrics(log)] == [0, 0, 1, 1]


def test_regret_with_a_single_bidder():
    # Alone in second price with reserve 0.1, every bid above the reserve wins at 0.1.
    # Own surplus = 0.9 + 0.1 + 0.7 = 1.7, the best any multiplier can reach.
    log = one_episode_log(
        Episode(
            values=[[1.0], [0.2], [0.8]],
            bids=[[1.0], [0.2], [0.8]],
            winner=[0, 0, 0],
            price=[0.1, 0.1, 0.1],
        ),
        bidders=(BidderSpec("solo", Truthful, 10.0, 5.0),),
        auction=SecondPrice(0.1),
    )
    assert regret(log, 0, 0) == pytest.approx(0.0)


def test_regret_replay_bid_equal_to_reserve_wins():
    # Multiplier 0.5 on value 1.0 bids exactly the reserve 0.5 and wins at 0.5.
    log = one_episode_log(
        Episode(
            values=[[1.0]],
            bids=[[0.5]],
            winner=[0],
            price=[0.5],
        ),
        bidders=(BidderSpec("solo", Truthful, 10.0, 5.0),),
        auction=SecondPrice(0.5),
    )
    assert regret(log, 0, 0, multipliers=np.array([0.5])) == pytest.approx(0.0)


def test_regret_replay_spends_its_budget():
    # Budget 0.6. Step 0: bid 0.6 beats 0.5, pays 0.5, leaving 0.1.
    # Step 1: bid capped at 0.1 loses to 0.4. Replay surplus = 1.0 - 0.5 = 0.5.
    log = one_episode_log(
        Episode(
            values=[[1.0, 0.0], [0.9, 0.0]],
            bids=[[0.6, 0.5], [0.1, 0.4]],
            winner=[0, 1],
            price=[0.5, 0.0],
        ),
        bidders=(
            BidderSpec("me", Truthful, 0.6, 5.0),
            BidderSpec("other", Truthful, 10.0, 5.0),
        ),
        auction=SecondPrice(),
    )
    assert regret(log, 0, 0, multipliers=np.array([1.0])) == pytest.approx(0.0)


def test_regret_masks_other_segments():
    # "me" bids only in segment x. Own surplus = (1.0-0.1) + (0.6-0.1) = 1.4.
    # Without the mask, multiplier 3 also wins step 1 in y: 1.4 + 0.7 = 2.1.
    log = one_episode_log(
        Episode(
            values=[[1.0], [0.8], [0.6]],
            bids=[[1.0], [0.0], [0.6]],
            winner=[0, NO_WINNER, 0],
            price=[0.1, 0.0, 0.1],
        ),
        bidders=(BidderSpec("me", Truthful, 10.0, 5.0),),
        auction=SecondPrice(0.1),
    )
    market = Segmented((Segment("x", 0.5), Segment("y", 0.5)))
    targeted = dataclasses.replace(
        log,
        config=dataclasses.replace(
            log.config,
            market=market,
            bidders=(BidderSpec("me", Truthful, 10.0, 5.0, segments=("x",)),),
        ),
        segment=np.array([[0, 1, 0]]),
    )
    assert regret(targeted, 0, 0) == pytest.approx(0.0)
    untargeted = dataclasses.replace(
        targeted, config=dataclasses.replace(log.config, market=market)
    )
    assert regret(untargeted, 0, 0) == pytest.approx(0.7)
