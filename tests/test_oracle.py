import math

import numpy as np
import pytest
from scipy.special import logit

import bidlab as bl
from bidlab.markets import LogisticMarket, Market, MarketDraw
from bidlab.types import EventName

UNSPENDABLE = 1e9
NOISE = 0.5
CALLS = 20_000


class FixedMarket(Market):
    """Yields the given pCTR and pCVR at every step."""

    def __init__(self, pctr, pcvr):
        self.pctr = pctr
        self.pcvr = pcvr

    def draws(self, count, rng):
        pctr = None if self.pctr is None else np.full(count, self.pctr)
        while True:
            yield MarketDraw({}, pctr=pctr, pcvr=np.full(count, self.pcvr))


def oracle_campaign(name, event: EventName = "conversion", noise=0.0, base_value=5.0):
    return bl.Campaign(
        name,
        bl.Stack(bl.OracleValue(noise=noise), bl.Linear()),
        budget=UNSPENDABLE,
        base_value=base_value,
        event=event,
    )


def started_oracle(event: EventName, noise=NOISE, base_value=2.0):
    oracle = bl.OracleValue(noise=noise)
    oracle.bind(np.random.default_rng(0))
    oracle.on_episode_start(
        bl.Campaign("a", oracle, budget=1.0, base_value=base_value, event=event)
    )
    return oracle


@pytest.mark.parametrize("clicks", [True, False])
@pytest.mark.parametrize("event", list(bl.Event))
def test_oracle_stage_bids_true_value(event, clicks):
    market = LogisticMarket(ctr_bias_mean=-2.0 if clicks else None)
    log = bl.run(
        bl.SimConfig(
            campaigns=[oracle_campaign("a", event), oracle_campaign("b")],
            episodes=2,
            steps=200,
            market=market,
        ),
        seed=3,
    )
    assert log.bids[:, :, 0].tobytes() == log.true_values[:, :, 0].tobytes()


def test_oracle_alone_bids_true_value():
    campaign = bl.Campaign(
        "a", bl.OracleValue(), budget=UNSPENDABLE, base_value=5.0, event="click"
    )
    log = bl.run(
        bl.SimConfig(
            campaigns=[campaign],
            episodes=1,
            steps=200,
            market=LogisticMarket(ctr_bias_mean=-2.0),
        )
    )
    assert log.bids.tobytes() == log.true_values.tobytes()


@pytest.mark.parametrize("noise", [-0.1, math.nan])
def test_negative_noise_rejected(noise):
    with pytest.raises(ValueError, match="noise"):
        bl.OracleValue(noise=noise)


def logit_errors(event: EventName, pctr, pcvr, true_rate):
    oracle = started_oracle(event)
    truth = bl.MarketTruth(pctr=pctr, pcvr=pcvr)
    values = np.array([oracle.forward(truth) for _ in range(CALLS)])
    return logit(values / 2.0) - logit(true_rate)


def test_noise_spread_on_pcvr():
    errors = logit_errors("conversion", pctr=1.0, pcvr=0.2, true_rate=0.2)
    assert abs(errors.mean()) < 0.02
    assert abs(errors.std() - NOISE) < 0.02


def test_noise_spread_on_pctr():
    errors = logit_errors("click", pctr=0.3, pcvr=0.2, true_rate=0.3)
    assert abs(errors.mean()) < 0.02
    assert abs(errors.std() - NOISE) < 0.02


@pytest.mark.parametrize("event", ["impression", "click"])
def test_noise_keeps_missing_click_stage(event):
    log = bl.run(
        bl.SimConfig(
            campaigns=[oracle_campaign("a", event, noise=NOISE)],
            episodes=1,
            steps=200,
            market=FixedMarket(None, 0.2),
        )
    )
    assert (log.bids == 5.0).all()


def first_campaign_log(*campaigns):
    config = bl.SimConfig(campaigns=campaigns, episodes=2, steps=200)
    log = bl.run(config, seed=5)
    return log.bids[:, :, 0], log.true_values[:, :, 0]


def test_noise_reproducible():
    first, true_values = first_campaign_log(oracle_campaign("a", noise=NOISE))
    second, _ = first_campaign_log(oracle_campaign("a", noise=NOISE))
    assert first.tobytes() == second.tobytes()
    assert (first != true_values).any()


def test_noise_keeps_true_values():
    _, noisy = first_campaign_log(oracle_campaign("a", noise=NOISE))
    _, exact = first_campaign_log(oracle_campaign("a"))
    assert noisy.tobytes() == exact.tobytes()


def test_noise_unchanged_by_new_campaign():
    alone, _ = first_campaign_log(oracle_campaign("a", noise=NOISE))
    joined, _ = first_campaign_log(
        oracle_campaign("a", noise=NOISE), oracle_campaign("b", noise=NOISE)
    )
    assert alone.tobytes() == joined.tobytes()


def test_noise_independent_per_stage():
    # Independent errors add their variances in log value; one shared error would add 4×.
    def log_values(event):
        oracle = started_oracle(event)
        truth = bl.MarketTruth(pctr=0.3, pcvr=0.3)
        return np.log([oracle.forward(truth) for _ in range(CALLS)])

    ratio = log_values("conversion").var() / log_values("click").var()
    assert 1.7 < ratio < 2.3
