import dataclasses
from typing import Any, ClassVar

import numpy as np
import pytest

from bidlab.auctions import FirstPrice
from bidlab.bidders import Bidder
from bidlab.config import Campaign, SimConfig
from bidlab.markets import LogisticMarket, Market, MarketDraw, Segment, Segmented
from bidlab.simulator import BUILT_IN_INPUTS, run
from bidlab.types import BidResult, EpisodeReport, MarketTruth

SEED = 7


class Spy(Bidder):
    """Bids `multiplier` × its true value and records every call it receives."""

    instances: ClassVar[list["Spy"]] = []

    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier
        self.results: list[BidResult] = []
        self.requests: list[dict[str, Any]] = []
        self.starts: list[Campaign] = []
        self.ends: list[EpisodeReport] = []

    def bind(self, rng):
        # Each run binds a fresh copy, so this lists the copies the run used.
        super().bind(rng)
        Spy.instances.append(self)

    def on_episode_start(self, campaign):
        self.starts.append(campaign)
        self.base_value = campaign.base_value

    def forward(self, market, **inputs):
        self.requests.append({"market": market, **inputs})
        # The same product, in the same order, as RunLog.true_values for a conversion.
        return self.multiplier * (self.base_value * market.pctr * market.pcvr)

    def update(self, result):
        self.results.append(result)

    def on_episode_end(self, report):
        self.ends.append(report)


class BatchSpy(Spy):
    """A Spy that overrides update_batch and records each batch."""

    def __init__(self, multiplier=1.0):
        super().__init__(multiplier)
        self.batches: list[list[BidResult]] = []

    def update_batch(self, bid_results):
        self.batches.append(list(bid_results))


def campaign(name, multiplier=1.0, budget=100.0, bidder=Spy, base_value=5.0, **kwargs):
    return Campaign(
        name,
        bidder(multiplier),
        budget=budget,
        base_value=base_value,
        **kwargs,
    )


def config(*bidders, **kwargs):
    defaults: dict[str, Any] = {"episodes": 3, "steps": 200}
    return SimConfig(campaigns=tuple(bidders), **(defaults | kwargs))


def run_spies(cfg):
    Spy.instances.clear()
    log = run(cfg, seed=SEED)
    return log, list(Spy.instances)


def test_outcome_arrays_have_a_campaign_axis():
    clicks = LogisticMarket(ctr_bias_mean=-1.0)
    log = run(
        config(campaign("a"), campaign("b"), episodes=2, steps=50, market=clicks),
        seed=SEED,
    )
    assert log.bids.shape == log.true_values.shape == (2, 50, 2)
    for array in (log.won, log.price, log.clicked, log.converted):
        assert array.shape == (2, 50, 2)
    assert not hasattr(log, "winner")
    lost = ~log.won
    assert lost.any() and log.won.any()
    assert (log.price[lost] == 0.0).all()
    assert not log.clicked[lost].any() and not log.converted[lost].any()


def test_competing_run_has_one_winner_per_step():
    log = run(
        config(campaign("a"), campaign("b"), campaign("c", multiplier=0.9)), seed=SEED
    )
    assert log.won.sum(axis=2).max() == 1
    assert (log.won.sum(axis=2) == 1).any()


def test_loser_result_has_no_price():
    _, spies = run_spies(config(campaign("a"), campaign("b", multiplier=0.9)))
    losses = [fb for spy in spies for fb in spy.results if not fb.won]
    assert losses
    assert all(fb.price == 0.0 and not fb.converted for fb in losses)


def test_result_fields_are_only_own_outcome():
    fields = [f.name for f in dataclasses.fields(BidResult)]
    assert fields == [
        "time",
        "bid",
        "won",
        "price",
        "clicked",
        "converted",
        "realized_value",
    ]


def test_episode_start_gets_campaign_with_running_bidder():
    a = campaign("a", cpa_target=2.0, segments=("all",))
    _, spies = run_spies(config(a, campaign("b")))
    assert spies[0].starts == [dataclasses.replace(a, bidder=spies[0])] * 3
    assert all(start.bidder is spies[0] for start in spies[0].starts)
    assert a.bidder is not spies[0]


def test_time_input_spaced_over_horizon():
    _, spies = run_spies(
        config(campaign("a"), episodes=1, steps=1000, horizon_hours=24)
    )
    times = np.array([o["time"] for o in spies[0].requests])
    assert times[0] == 0.0
    np.testing.assert_allclose(times, np.arange(1000) * 86.4, rtol=0, atol=1e-9)


def test_result_time_matches_time_input():
    _, spies = run_spies(config(campaign("a"), campaign("b", multiplier=0.9)))
    for spy in spies:
        assert [f.time for f in spy.results] == [o["time"] for o in spy.requests]


def test_result_value_is_realized():
    log, spies = run_spies(
        config(campaign("a", base_value=3.0), campaign("b", multiplier=0.9))
    )
    for i, (spy, value) in enumerate(zip(spies, (3.0, 5.0), strict=True)):
        realized = np.where(log.converted[:, :, i], value, 0.0).ravel()
        assert [f.realized_value for f in spy.results] == list(realized)
        assert any(f.realized_value == value for f in spy.results)


def test_update_batch_gets_one_result_per_auction():
    _, spies = run_spies(config(campaign("a", bidder=BatchSpy), campaign("b")))
    batching = spies[0]
    assert isinstance(batching, BatchSpy)
    assert len(batching.batches) == 3 * 200
    assert all(len(batch) == 1 for batch in batching.batches)
    times = [batch[0].time for batch in batching.batches]
    per_episode = np.array(times).reshape(3, 200)
    assert np.all(np.diff(per_episode, axis=1) > 0)
    assert batching.results == []


def test_winner_result_matches_log():
    log, spies = run_spies(
        config(campaign("a"), campaign("b", multiplier=0.9), episodes=1)
    )
    for i, spy in enumerate(spies):
        for step, fb in enumerate(spy.results):
            assert fb.won == log.won[0, step, i]
            if fb.won:
                assert fb.price == log.price[0, step, i]
                assert fb.converted == log.converted[0, step, i]


def test_spend_never_exceeds_budget():
    greedy = campaign("greedy", multiplier=1e9, budget=2.0)
    log = run(config(greedy, campaign("b")), seed=SEED)
    for episode in range(log.won.shape[0]):
        spend = log.price[episode, :, 0][log.won[episode, :, 0]].sum()
        assert spend <= 2.0 + 1e-12
    assert np.all(log.bids[:, :, 0] <= 2.0)


def test_budget_resets_each_episode():
    log, spies = run_spies(
        config(campaign("greedy", multiplier=1e9, budget=2.0), campaign("b"))
    )
    greedy = spies[0]
    assert [campaign.budget for campaign in greedy.starts] == [2.0] * 3
    # A huge bid is capped at the budget left, so each first bid shows a full budget.
    np.testing.assert_array_equal(log.bids[:, 0, 0], [2.0] * 3)


def test_same_bidder_object_and_hook_per_episode():
    _, spies = run_spies(config(campaign("a"), campaign("b")))
    assert len(spies) == 2
    assert all(len(spy.starts) == 3 and len(spy.ends) == 3 for spy in spies)
    assert all(len(spy.results) == 3 * 200 for spy in spies)


def test_same_seed_same_log():
    first = run(config(campaign("a"), campaign("b", multiplier=0.8)), seed=SEED)
    second = run(config(campaign("a"), campaign("b", multiplier=0.8)), seed=SEED)
    np.testing.assert_array_equal(first.bids, second.bids)
    np.testing.assert_array_equal(first.won, second.won)
    np.testing.assert_array_equal(first.converted, second.converted)
    other = run(config(campaign("a"), campaign("b", multiplier=0.8)), seed=8)
    assert not np.array_equal(first.true_values, other.true_values)


def test_adding_bidder_keeps_world_draws():
    two = run(config(campaign("a"), campaign("b")), seed=SEED)
    three = run(config(campaign("a"), campaign("b"), campaign("c")), seed=SEED)
    np.testing.assert_array_equal(two.true_values, three.true_values[:, :, :2])


def test_episode_end_gets_report_only():
    _, spies = run_spies(config(campaign("a"), campaign("b")))
    assert len(spies[0].ends) == 3
    assert all(isinstance(report, EpisodeReport) for report in spies[0].ends)


def test_reserve_above_all_bids_gives_no_sales():
    log = run(config(campaign("a"), auction=FirstPrice(reserve=1e6)), seed=SEED)
    assert not log.won.any()
    assert np.all(log.price == 0)


class FixedMarket(Market):
    """Yields the given pCVR, pCTR and features at every step."""

    def __init__(self, pcvr, features=None, pctr=None):
        self.pcvr = pcvr
        self.pctr = pctr
        self.features = features or {}

    def draws(self, count, rng):
        pctr = None if self.pctr is None else np.array(self.pctr)
        while True:
            yield MarketDraw(self.features, pctr=pctr, pcvr=np.array(self.pcvr))


class CountingMarket(Market):
    """Step k gives bidder i probability (k mod 10 + i) / 20."""

    def draws(self, count, rng):
        step = 0
        while True:
            yield MarketDraw({}, pcvr=(step % 10 + np.arange(count)) / 20)
            step += 1


def test_custom_market_values_are_weight_times_probability():
    cfg = config(
        campaign("a", base_value=2.0),
        campaign("b", base_value=4.0),
        market=CountingMarket(),
        steps=30,
    )
    log = run(cfg, seed=SEED)
    step = np.arange(cfg.episodes * cfg.steps).reshape(cfg.episodes, cfg.steps)
    expected = (step[..., None] % 10 + np.arange(2)) / 20 * np.array([2.0, 4.0])
    np.testing.assert_array_equal(log.true_values, expected)


@pytest.mark.parametrize(
    "pcvr",
    [[0.5], [0.5, 1.2], [0.5, -0.1], [0.5, np.nan]],
    ids=["wrong-length", "above-one", "below-zero", "nan"],
)
def test_bad_market_draw_rejected(pcvr):
    cfg = config(campaign("a"), campaign("b"), market=FixedMarket(pcvr))
    with pytest.raises(ValueError, match="FixedMarket at episode 0, step 0"):
        run(cfg, seed=SEED)


def test_market_that_stops_names_the_market():
    class Short(Market):
        def draws(self, count, rng):
            yield MarketDraw({}, pcvr=np.full(count, 0.5))

    with pytest.raises(RuntimeError, match="Short at episode 0, step 1"):
        run(config(campaign("a"), market=Short()), seed=SEED)


def test_features_reach_every_bidder():
    market = FixedMarket([0.5, 0.5], features={"hour": 3})
    _, spies = run_spies(config(campaign("a"), campaign("b"), market=market, steps=5))
    for spy in spies:
        assert [request["hour"] for request in spy.requests] == [3] * 15


def test_no_features_gives_only_built_in_inputs():
    _, spies = run_spies(config(campaign("a"), market=FixedMarket([0.5]), steps=5))
    for request in spies[0].requests:
        assert request.keys() == BUILT_IN_INPUTS


NEWS_AND_SPORTS = Segmented((Segment("news", 0.5), Segment("sports", 0.5)))


def test_targeted_bidder_skips_other_segments():
    cfg = config(
        campaign("news_only", segments=("news",)),
        campaign("everywhere"),
        campaign("news_batches", segments=("news",), bidder=BatchSpy),
        market=NEWS_AND_SPORTS,
    )
    log, (news_only, _, news_batches) = run_spies(cfg)
    sports = log.segment == NEWS_AND_SPORTS.segment_names.index("sports")
    assert sports.any() and (~sports).any()
    assert np.all(log.bids[sports][:, 0] == 0.0)
    assert not np.any(log.won[sports][:, 0])
    assert len(news_only.requests) == np.sum(~sports)
    assert len(news_only.results) == np.sum(~sports)
    assert isinstance(news_batches, BatchSpy)
    assert len(news_batches.batches) == np.sum(~sports)
    assert {o["segment"] for o in news_only.requests} == {"news"}


def test_lone_targeted_bidder_never_wins_outside_its_segments():
    cfg = config(campaign("news_only", segments=("news",)), market=NEWS_AND_SPORTS)
    log = run(cfg, seed=SEED)
    sports = log.segment == NEWS_AND_SPORTS.segment_names.index("sports")
    assert not log.won[sports].any()
    assert np.all(log.price[sports] == 0.0)


def test_untargeted_bidder_takes_every_step():
    cfg = config(
        campaign("a"), campaign("b", segments=("sports",)), market=NEWS_AND_SPORTS
    )
    _, (everywhere, _) = run_spies(cfg)
    assert len(everywhere.requests) == cfg.episodes * cfg.steps
    assert len(everywhere.results) == cfg.episodes * cfg.steps


class DayAndNight(Market):
    """First 6 of every 10 steps are "day", the rest "night"."""

    @property
    def segment_names(self):
        return ("day", "night")

    def draws(self, count, rng):
        step = 0
        while True:
            name = "day" if step % 10 < 6 else "night"
            yield MarketDraw({}, pcvr=np.full(count, 0.1), segment=name)
            step += 1


def test_custom_market_segments_drive_targeting_and_log():
    cfg = config(
        campaign("night_only", segments=("night",)), campaign("a"), market=DayAndNight()
    )
    log, (night_only, _) = run_spies(cfg)
    step = np.arange(cfg.episodes * cfg.steps).reshape(cfg.episodes, cfg.steps)
    np.testing.assert_array_equal(log.segment, (step % 10 >= 6).astype(int))
    assert len(night_only.requests) == np.sum(step % 10 >= 6)
    assert not np.any(log.won[log.segment == 0][:, 0])


def click_config(pctr, pcvr, **kwargs):
    return config(
        *(campaign(name) for name in "ab"[: len(pcvr)]),
        market=FixedMarket(pcvr, pctr=pctr),
        **kwargs,
    )


def test_click_and_conversion_shares():
    cfg = click_config([0.2], [0.5], episodes=1, steps=20_000)
    log = run(
        dataclasses.replace(cfg, campaigns=(campaign("a", budget=1e9),)), seed=SEED
    )
    wins = log.won[:, :, 0]
    assert wins.all()
    assert abs(log.clicked[:, :, 0][wins].mean() - 0.2) < 0.02
    assert abs(log.converted[log.clicked].mean() - 0.5) < 0.03


@pytest.mark.parametrize("pctr", [None, [0.3, 0.6]])
def test_no_conversion_without_click(pctr):
    log = run(click_config(pctr, [0.9, 0.8]), seed=SEED)
    assert log.converted.any()
    assert not (log.converted & ~log.clicked).any()


def test_no_click_stage_clicks_every_win():
    log, spies = run_spies(click_config(None, [0.1, 0.1]))
    np.testing.assert_array_equal(log.clicked, log.won)
    for spy in spies:
        assert all(result.clicked == result.won for result in spy.results)


def test_clicked_array_shape_and_read_only():
    cfg = click_config([0.5, 0.5], [0.5, 0.5], auction=FirstPrice(reserve=0.3))
    log = run(cfg, seed=SEED)
    assert log.clicked.shape == (cfg.episodes, cfg.steps, 2)
    assert log.clicked.dtype == np.bool_
    assert not log.clicked.flags.writeable
    unsold = ~log.won.any(axis=2)
    assert unsold.any()
    assert not log.clicked[unsold].any()


def test_bid_result_clicked_only_for_clicked_winner():
    log, spies = run_spies(click_config([0.4, 0.4], [0.5, 0.5]))
    for i, spy in enumerate(spies):
        clicked = [result.clicked for result in spy.results]
        expected = log.clicked[:, :, i].ravel().tolist()
        assert clicked == expected
        assert any(clicked)
        assert all(result.won for result in spy.results if result.clicked)


def test_true_values_use_both_stages():
    clicks = run(click_config([0.2, 0.4], [0.5, 0.25]), seed=SEED)
    np.testing.assert_array_equal(
        clicks.true_values[0, 0], [5.0 * 0.2 * 0.5, 5.0 * 0.4 * 0.25]
    )
    no_clicks = run(click_config(None, [0.5, 0.25]), seed=SEED)
    np.testing.assert_array_equal(no_clicks.true_values[0, 0], [5.0 * 0.5, 5.0 * 0.25])


@pytest.mark.parametrize(
    "pctr",
    [[0.5], [0.5, 1.2], [0.5, -0.1], [0.5, np.nan]],
    ids=["wrong-length", "above-one", "below-zero", "nan"],
)
def test_bad_pctr_rejected(pctr):
    with pytest.raises(ValueError, match="FixedMarket at episode 0, step 0: .*pctr"):
        run(click_config(pctr, [0.5, 0.5]), seed=SEED)


@pytest.mark.parametrize("pctr", [None, [0.3, 0.6]])
def test_market_input_holds_truth(pctr):
    _, spies = run_spies(click_config(pctr, [0.1, 0.2]))
    for i, spy in enumerate(spies):
        expected = MarketTruth(
            pctr=1.0 if pctr is None else pctr[i], pcvr=[0.1, 0.2][i]
        )
        assert {request["market"] for request in spy.requests} == {expected}


def test_logistic_market_inputs():
    _, (spy,) = run_spies(config(campaign("a"), market=LogisticMarket(), steps=3))
    for request in spy.requests:
        assert request.keys() == BUILT_IN_INPUTS | {"vector"}
        assert isinstance(request["market"], MarketTruth)
        assert request["features"]["vector"] is request["vector"]


def test_features_input_is_read_only_mapping():
    market = FixedMarket([0.5], features={"country": "id", "device": 1.0})
    _, (spy,) = run_spies(config(campaign("a"), market=market, steps=3))
    for request in spy.requests:
        assert dict(request["features"]) == {"country": "id", "device": 1.0}
        assert request["features"]["country"] == request["country"]
        with pytest.raises(TypeError):
            request["features"]["country"] = "us"
    _, (bare,) = run_spies(config(campaign("a"), market=FixedMarket([0.5]), steps=3))
    assert all(dict(request["features"]) == {} for request in bare.requests)


class Oracle(Bidder):
    """Bids the true value from the market input alone."""

    def forward(self, market):
        return 5.0 * market.pctr * market.pcvr


def test_market_input_by_name():
    oracle = Campaign("o", Oracle(), budget=100.0, base_value=5.0)
    other = Campaign("p", Oracle(), budget=100.0, base_value=5.0)
    log = run(
        config(oracle, other, market=FixedMarket([0.1, 0.2], pctr=[0.5, 0.25])),
        seed=SEED,
    )
    np.testing.assert_array_equal(log.bids[..., 0], 5.0 * 0.5 * 0.1)
    np.testing.assert_array_equal(log.bids[..., 1], 5.0 * 0.25 * 0.2)


class FixedBid(Bidder):
    def __init__(self, bid):
        self.bid = bid

    def forward(self):
        return self.bid


@pytest.mark.parametrize("bid", [np.nan, np.inf, -np.inf, -1.0])
def test_bad_bid_counts_as_zero(bid):
    bad = Campaign("bad", FixedBid(bid), budget=100.0, base_value=5.0)
    log = run(config(bad, campaign("b"), episodes=1, steps=50), seed=SEED)
    assert (log.bids[:, :, 0] == 0.0).all()
    assert not log.won[:, :, 0].any()


def test_run_log_records_its_seed():
    cfg = config(campaign("a"), episodes=1, steps=20)
    assert run(cfg).seed == 0
    seeded = run(cfg, seed=SEED)
    assert seeded.seed == SEED
    assert not np.array_equal(seeded.true_values, run(cfg).true_values)


def test_numpy_integer_seed_runs_like_its_int():
    cfg = config(campaign("a"), campaign("b"), episodes=1, steps=50)
    plain = run(cfg, seed=SEED)
    numpy_seeded = run(cfg, seed=np.int64(SEED))
    assert type(numpy_seeded.seed) is int
    for name in ("bids", "true_values", "won", "price", "converted"):
        assert getattr(numpy_seeded, name).tobytes() == getattr(plain, name).tobytes()


@pytest.mark.parametrize("seed", [True, 1.0, "1", None])
def test_seed_must_be_an_int(seed):
    with pytest.raises(TypeError, match="seed"):
        run(config(campaign("a"), episodes=1, steps=5), seed=seed)  # pyright: ignore[reportArgumentType]  Reason: the test checks the runtime error for a wrong type.


def test_config_has_no_seed():
    with pytest.raises(TypeError, match="seed"):
        SimConfig(campaigns=[campaign("a")], seed=1)  # pyright: ignore[reportCallIssue]  Reason: the test checks that the removed field is rejected.
