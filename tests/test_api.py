import hashlib
import importlib
import importlib.util
import tomllib
from pathlib import Path

import pytest

from bidlab import rewards
from bidlab.auctions import FirstPrice, SecondPrice
from bidlab.bidders import PidPacing, Truthful, UcbMultiplier
from bidlab.config import BidderSpec, SimConfig
from bidlab.simulator import run

ROOT = Path(__file__).parent.parent

# Hashes recorded from commit 2a8797b plus the budget fix, before the Python API change.
RUN_LOG_HASHES = {
    "first": "05d813761620ab4fc954210c26f5674779050fe0aaee8b94574fedf901dd32e2",
    "second": "9b59dcbebee36f3fc26ed276765c06283704e79f221722ec679f35390f378eb6",
}


def fingerprint_config(kind):
    bidders = (
        BidderSpec("t", Truthful, 40.0, 5.0),
        BidderSpec("p", PidPacing, 10.0, 5.0),
        BidderSpec("b", UcbMultiplier, 40.0, 4.0, reward=rewards.conversions),
        BidderSpec(
            "s",
            Truthful,
            40.0,
            6.0,
            params={"multiplier": 0.8},
            cpa_target=3.0,
            episode_reward=rewards.cpa_score,
        ),
    )
    return SimConfig(
        bidders=bidders,
        episodes=3,
        steps=300,
        seed=12,
        auction=FirstPrice(0.01) if kind == "first" else SecondPrice(0.01),
    )


@pytest.mark.parametrize("kind", ["first", "second"])
def test_run_log_fingerprint(kind):
    log = run(fingerprint_config(kind))
    digest = hashlib.sha256()
    for array in (
        log.bids,
        log.values,
        log.winner,
        log.price,
        log.converted,
        log.rewards,
        log.episode_rewards,
    ):
        digest.update(array.tobytes())

    assert digest.hexdigest() == RUN_LOG_HASHES[kind]


PUBLIC_NAMES = {
    "SimConfig",
    "BidderSpec",
    "WorldConfig",
    "FirstPrice",
    "SecondPrice",
    "Bidder",
    "Truthful",
    "PidPacing",
    "UcbMultiplier",
    "Observation",
    "Feedback",
    "EpisodeStats",
    "RunLog",
    "run",
    "run_many",
    "Results",
    "bidder_metrics",
    "platform_metrics",
    "regret",
    "rewards",
    "bidders",
}


def test_import_bidlab_not_auction_sim():
    importlib.import_module("bidlab")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("auction_sim")


def test_public_names():
    bl = importlib.import_module("bidlab")
    assert set(bl.__all__) == PUBLIC_NAMES
    assert all(getattr(bl, name) is not None for name in bl.__all__)


def test_bidder_families_are_same_objects():
    import bidlab as bl

    assert bl.bidders.baselines.Truthful is bl.Truthful
    assert bl.bidders.pacing.PidPacing is bl.PidPacing
    assert bl.bidders.bandits.UcbMultiplier is bl.UcbMultiplier


def test_builtin_rewards_are_callable():
    import bidlab as bl

    observation = bl.Observation(
        step=0, steps_left=1, value=0.2, value_per_conversion=4.0, budget_left=1.0
    )
    feedback = bl.Feedback(bid=0.3, won=True, price=0.25, converted=True)
    stats = bl.EpisodeStats(budget=10, spend=2, wins=3, conversions=1, value=4)
    assert bl.rewards.surplus(observation, feedback, {}) == 3.75
    assert bl.rewards.conversions(observation, feedback, {}) == 1.0
    assert bl.rewards.value(observation, feedback, {}) == 4.0
    assert bl.rewards.cpa_score(stats, {"cpa_target": 2.0}) == 4.0
    assert bl.rewards.total_surplus(stats, {}) == 2.0


def test_removed_interfaces():
    for module in ("bidlab.cli", "bidlab.plugins"):
        assert importlib.util.find_spec(module) is None
    import bidlab.config

    assert not hasattr(bidlab.config, "load_config")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = " ".join(pyproject["project"]["dependencies"]).lower()
    assert "yaml" not in dependencies
    assert "scripts" not in pyproject["project"]
    assert pyproject["project"]["name"] == "bidlab"
