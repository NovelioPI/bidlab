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
from bidlab.runner import run_many
from bidlab.simulator import run
from bidlab.types import Campaign

ROOT = Path(__file__).parent.parent

# Recorded with numpy 2.5.4 after PidPacing began planning spend by time.
# test_run_log_without_pid_matches_main shows every other float is unchanged.
RUN_LOG_HASHES = {
    "first": "2bae08b40187f061e33dc66695216400970b159d05ceddca884b61567def6196",
    "second": "d4da1e3bb9b8788845c833e0d13f4294fcbd0e43207c94bef87073305a1c28f9",
}
# Main results table CSV for seeds 0 and 1, recorded with the run-log hashes above.
RESULTS_HASH = "4b6300a39837e77c144d1af327aab7ed8d5064f4e412b973838ba6a2c9bee51e"


def fingerprint_config(kind):
    bidders = (
        BidderSpec("t", Truthful(), Campaign(40.0, 5.0)),
        BidderSpec("p", PidPacing(), Campaign(10.0, 5.0)),
        BidderSpec(
            "b", UcbMultiplier(), Campaign(40.0, 4.0), reward=rewards.conversions
        ),
        BidderSpec(
            "s",
            Truthful(multiplier=0.8),
            Campaign(40.0, 6.0, cpa_target=3.0),
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


def test_results_fingerprint(tmp_path):
    path = tmp_path / "results.csv"
    run_many(fingerprint_config("second"), seeds=[0, 1]).save_csv(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == RESULTS_HASH


# Recorded on main (151995c) with numpy 2.5.4, before the bidder interface changed.
MAIN_HASHES_WITHOUT_PID = {
    "first": "353a32401bb86c966629d5388d61fa94770ee111ed9b2aba644d72d798fb2f01",
    "second": "2b269e595d37ee73bdf6f30e4f9979c0711b5c208b583369c36091db60359a92",
}


@pytest.mark.parametrize("kind", ["first", "second"])
def test_run_log_without_pid_matches_main(kind):
    bidders = (
        BidderSpec("t", Truthful(), Campaign(40.0, 5.0)),
        BidderSpec(
            "b", UcbMultiplier(), Campaign(40.0, 4.0), reward=rewards.conversions
        ),
        BidderSpec(
            "s",
            Truthful(multiplier=0.8),
            Campaign(40.0, 6.0, cpa_target=3.0),
            episode_reward=rewards.cpa_score,
        ),
    )
    auction = FirstPrice(0.01) if kind == "first" else SecondPrice(0.01)
    log = run(
        SimConfig(bidders=bidders, episodes=3, steps=300, seed=12, auction=auction)
    )
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

    assert digest.hexdigest() == MAIN_HASHES_WITHOUT_PID[kind]


PUBLIC_NAMES = {
    "Callback",
    "SimConfig",
    "Stack",
    "BidderSpec",
    "Campaign",
    "markets",
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
    "segment_metrics",
}


def test_import_bidlab_not_auction_sim():
    importlib.import_module("bidlab")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("auction_sim")


def test_public_names():
    bl = importlib.import_module("bidlab")
    assert set(bl.__all__) == PUBLIC_NAMES
    assert all(getattr(bl, name) is not None for name in bl.__all__)


def test_markets_module_exports():
    import bidlab as bl

    for name in (
        "Market",
        "MarketDraw",
        "LogisticContext",
        "Segment",
        "Segmented",
        "IndependentValues",
    ):
        assert getattr(bl.markets, name) is not None


def test_metrics_doc_defines_segment_columns():
    from bidlab.runner import SEGMENT_COLUMNS

    doc = (ROOT / "docs" / "metrics.md").read_text()
    section = doc.split("## Segment metrics", 1)[1].split("\n## ", 1)[0]
    for column in SEGMENT_COLUMNS[2:]:
        assert f"| {column} |" in section


def test_bidder_families_are_same_objects():
    import bidlab as bl

    assert bl.bidders.baselines.Truthful is bl.Truthful
    assert bl.bidders.pacing.PidPacing is bl.PidPacing
    assert bl.bidders.bandits.UcbMultiplier is bl.UcbMultiplier


def test_builtin_rewards_are_callable():
    import bidlab as bl

    observation = bl.Observation(time=0.0, value=0.2)
    feedback = bl.Feedback(
        time=0.0, bid=0.3, won=True, price=0.25, converted=True, value=4.0
    )
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
    assert importlib.util.find_spec("bidlab.world") is None
    with pytest.raises(AttributeError):
        _ = importlib.import_module("bidlab").WorldConfig
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = " ".join(pyproject["project"]["dependencies"]).lower()
    assert "yaml" not in dependencies
    assert "scripts" not in pyproject["project"]
    assert pyproject["project"]["name"] == "bidlab"
