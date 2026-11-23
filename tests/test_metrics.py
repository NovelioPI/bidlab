import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest

from bidlab.auctions import NO_WINNER, FirstPrice, SecondPrice
from bidlab.bidders import Linear, OracleValue, Stack
from bidlab.config import Campaign, SimConfig
from bidlab.markets import Segment, Segmented
from bidlab.metrics import campaign_metrics, platform_metrics, regret, segment_metrics
from bidlab.simulator import RunLog, run


def hand_log(kind="second", reserve=0.0, budget=10.0, cpa_target=None):
    """One episode, 3 steps, 2 campaigns, with every number chosen by hand."""
    campaigns = (
        Campaign(
            "a",
            Stack(OracleValue(), Linear()),
            budget=budget,
            base_value=4.0,
            cpa_target=cpa_target,
        ),
        Campaign("b", Stack(OracleValue(), Linear()), budget=10.0, base_value=2.0),
    )
    config = SimConfig(
        campaigns=campaigns,
        episodes=1,
        steps=3,
        auction=FirstPrice(reserve) if kind == "first" else SecondPrice(reserve),
    )
    values = np.array([[[1.0, 0.5], [0.2, 0.6], [0.8, 0.3]]])
    bids = values.copy()
    return RunLog(
        config=config,
        bids=bids,
        true_values=values,
        won=np.array([[[True, False], [False, True], [True, False]]]),
        price=np.array([[[0.5, 0.0], [0.0, 0.2], [0.3, 0.0]]]),
        clicked=np.array([[[True, False], [False, True], [True, False]]]),
        converted=np.array([[[True, False], [False, False], [False, False]]]),
        segment=np.zeros((1, 3), dtype=np.int64),
        market_price=np.full((1, 3), np.nan),
        floor=np.zeros((1, 3)),
        budgets=np.array([budget, 10.0]),
    )


def test_metrics_on_hand_built_log():
    a, b = campaign_metrics(hand_log())
    assert (a.spend, a.wins, a.conversions, a.realized_value) == (0.8, 2, 1, 4.0)
    assert a.surplus == pytest.approx(3.2)
    assert a.budget_used == pytest.approx(0.08)
    assert a.cpa == pytest.approx(0.8)
    assert np.isnan(a.cpa_violation) and np.isnan(a.cpa_score)
    assert (b.spend, b.wins, b.conversions, b.realized_value) == (0.2, 1, 0, 0.0)
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
    a, _ = campaign_metrics(hand_log(cpa_target=target))
    assert a.cpa_violation == pytest.approx(violation)
    assert a.cpa_score == pytest.approx(score)


def test_cpa_violation_infinite_when_spend_without_conversions():
    log = hand_log(cpa_target=1.0)
    no_conversions = RunLog(
        **{**log.__dict__, "converted": np.zeros((1, 3, 2), dtype=bool)}
    )
    a, _ = campaign_metrics(no_conversions)
    assert a.cpa_violation == np.inf
    assert a.cpa_score == 0.0


def test_cpa_score_is_value_when_conversions_cost_nothing():
    log = hand_log(cpa_target=1.0)
    free = RunLog(**{**log.__dict__, "price": np.zeros((1, 3, 2))})
    a, _ = campaign_metrics(free)
    assert (a.spend, a.conversions) == (0.0, 1)
    assert a.cpa_score == 4.0


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
        campaigns = (
            Campaign(
                "me",
                Stack(OracleValue(), Linear(multiplier=multiplier)),
                budget=50.0,
                base_value=5.0,
            ),
            Campaign("x", Stack(OracleValue(), Linear()), budget=50.0, base_value=4.0),
            Campaign(
                "y",
                Stack(OracleValue(), Linear(multiplier=0.7)),
                budget=50.0,
                base_value=6.0,
            ),
        )
        log = run(SimConfig(campaigns=campaigns, episodes=2, steps=300, seed=3))
        for episode in range(2):
            assert regret(log, episode, 0) >= -1e-9


def test_regret_accepts_integer_budget():
    def config(budget):
        campaigns = (
            Campaign(
                "me", Stack(OracleValue(), Linear()), budget=budget, base_value=5.0
            ),
            Campaign("x", Stack(OracleValue(), Linear()), budget=50.0, base_value=4.0),
        )
        return SimConfig(campaigns=campaigns, episodes=1, steps=200, seed=4)

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


def one_episode_log(episode, campaigns, auction):
    steps = len(episode.values)
    won = np.array(episode.winner)[:, None] == np.arange(len(campaigns))
    return RunLog(
        config=SimConfig(campaigns=campaigns, episodes=1, steps=steps, auction=auction),
        bids=np.array([episode.bids], dtype=float),
        true_values=np.array([episode.values], dtype=float),
        won=won[None],
        price=np.where(won, np.array(episode.price)[:, None], 0.0)[None],
        clicked=won[None],
        converted=np.zeros((1, steps), dtype=bool),
        segment=np.zeros((1, steps), dtype=np.int64),
        market_price=np.full((1, steps), np.nan),
        floor=np.zeros((1, steps)),
        budgets=np.array([campaign.budget for campaign in campaigns], dtype=float),
    )


def test_cpa_metrics_divide_spend_by_conversions():
    # a wins steps 0 and 2 for 0.5 + 0.3 and converts twice: cpa 0.8 / 2 = 0.4.
    log = hand_log(cpa_target=0.2)
    two_conversions = RunLog(
        **{
            **log.__dict__,
            "converted": np.array([[[True, False], [False, False], [True, False]]]),
        }
    )
    a, _ = campaign_metrics(two_conversions)
    assert a.cpa == pytest.approx(0.4)
    assert a.cpa_violation == pytest.approx(0.4 / 0.2 - 1)


def test_no_spend_means_no_cpa_violation():
    # a never wins, so it spends nothing and converts nothing.
    log = hand_log(cpa_target=1.0)
    b_wins = np.array([[[False, True]] * 3])
    never_wins = RunLog(
        **{
            **log.__dict__,
            "won": b_wins,
            "price": np.where(b_wins, log.price.sum(axis=2, keepdims=True), 0.0),
            "clicked": b_wins,
            "converted": np.zeros_like(b_wins),
        }
    )
    a, _ = campaign_metrics(never_wins)
    assert a.spend == 0.0
    assert a.cpa_violation == 0.0
    assert a.cpa_score == 0.0


def test_metric_rows_carry_their_episode():
    campaigns = (
        Campaign("a", Stack(OracleValue(), Linear()), budget=50.0, base_value=5.0),
        Campaign("b", Stack(OracleValue(), Linear()), budget=50.0, base_value=4.0),
    )
    log = run(SimConfig(campaigns=campaigns, episodes=2, steps=50, seed=1))
    assert [row.episode for row in platform_metrics(log)] == [0, 1]
    assert [row.episode for row in campaign_metrics(log)] == [0, 0, 1, 1]


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
        campaigns=(
            Campaign(
                "solo", Stack(OracleValue(), Linear()), budget=10.0, base_value=5.0
            ),
        ),
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
        campaigns=(
            Campaign(
                "solo", Stack(OracleValue(), Linear()), budget=10.0, base_value=5.0
            ),
        ),
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
        campaigns=(
            Campaign("me", Stack(OracleValue(), Linear()), budget=0.6, base_value=5.0),
            Campaign(
                "other", Stack(OracleValue(), Linear()), budget=10.0, base_value=5.0
            ),
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
        campaigns=(
            Campaign("me", Stack(OracleValue(), Linear()), budget=10.0, base_value=5.0),
        ),
        auction=SecondPrice(0.1),
    )
    market = Segmented((Segment("x", 0.5), Segment("y", 0.5)))
    targeted = dataclasses.replace(
        log,
        config=dataclasses.replace(
            log.config,
            market=market,
            campaigns=(
                Campaign(
                    "me",
                    Stack(OracleValue(), Linear()),
                    budget=10.0,
                    base_value=5.0,
                    segments=("x",),
                ),
            ),
        ),
        segment=np.array([[0, 1, 0]]),
    )
    assert regret(targeted, 0, 0) == pytest.approx(0.0)
    untargeted = dataclasses.replace(
        targeted, config=dataclasses.replace(log.config, market=market)
    )
    assert regret(untargeted, 0, 0) == pytest.approx(0.7)


def test_segment_rows_carry_their_episode():
    log = run(
        SimConfig(
            campaigns=(
                Campaign(
                    "a", Stack(OracleValue(), Linear()), budget=5.0, base_value=4.0
                ),
            ),
            episodes=3,
            steps=5,
        )
    )
    assert [row.episode for row in segment_metrics(log)] == [0, 1, 2]


def targeted_hand_log(segment):
    """`hand_log` with campaign a bidding only in segment x of a two-segment market."""
    log = hand_log()
    a, b = log.config.campaigns
    return dataclasses.replace(
        log,
        config=dataclasses.replace(
            log.config,
            market=Segmented((Segment("x", 0.5), Segment("y", 0.5))),
            campaigns=(dataclasses.replace(a, segments=("x",)), b),
        ),
        segment=np.array([segment]),
    )


def test_win_rate_counts_requests_in_segments():
    # a wins steps 0 and 2 of 3 requests; b wins step 1.
    a, b = campaign_metrics(hand_log())
    assert a.win_rate == pytest.approx(2 / 3)
    assert b.win_rate == pytest.approx(1 / 3)
    # Step 1 is in segment y, outside a's segments, so a has 2 requests.
    a, b = campaign_metrics(targeted_hand_log([0, 1, 0]))
    assert a.win_rate == 1.0
    assert b.win_rate == pytest.approx(1 / 3)


def test_win_rate_is_nan_without_requests():
    log = targeted_hand_log([1, 1, 1])
    a_never_wins = np.array([[[False, False], [False, True], [False, False]]])
    log = dataclasses.replace(
        log,
        won=a_never_wins,
        price=np.where(a_never_wins, log.price, 0.0),
        clicked=a_never_wins,
        converted=np.zeros_like(a_never_wins),
    )
    a, _ = campaign_metrics(log)
    assert np.isnan(a.win_rate)


def test_cvr_divides_conversions_by_clicks():
    # a: 1 conversion of 2 clicks; b: 0 of 1.
    a, b = campaign_metrics(hand_log())
    assert a.cvr == 0.5
    assert b.cvr == 0.0
    log = hand_log()
    no_clicks = dataclasses.replace(
        log,
        clicked=np.zeros_like(log.clicked),
        converted=np.zeros_like(log.converted),
    )
    a, b = campaign_metrics(no_clicks)
    assert np.isnan(a.cvr) and np.isnan(b.cvr)


def test_expected_surplus_on_hand_built_log():
    # a wins values 1.0 and 0.8 for 0.5 + 0.3; b wins 0.6 for 0.2.
    a, b = campaign_metrics(hand_log())
    assert a.expected_surplus == pytest.approx(1.8 - 0.8)
    assert b.expected_surplus == pytest.approx(0.6 - 0.2)


def test_cost_per_expected_event_on_hand_built_log():
    # a: 1.8 expected value / base value 4 = 0.45 expected events, for spend 0.8.
    # b: 0.6 / 2 = 0.3 expected events, for spend 0.2.
    a, b = campaign_metrics(hand_log())
    assert a.cost_per_expected_event == pytest.approx(0.8 / 0.45)
    assert b.cost_per_expected_event == pytest.approx(0.2 / 0.3)
    log = hand_log()
    nobody_wins = dataclasses.replace(
        log,
        won=np.zeros_like(log.won),
        price=np.zeros_like(log.price),
        clicked=np.zeros_like(log.clicked),
        converted=np.zeros_like(log.converted),
    )
    a, b = campaign_metrics(nobody_wins)
    assert np.isnan(a.cost_per_expected_event) and np.isnan(b.cost_per_expected_event)
