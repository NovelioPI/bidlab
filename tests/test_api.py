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

ROOT = Path(__file__).parent.parent

# Recorded with numpy 2.5.4 after logits became a per-row sum; winners and conversions match c4e510f.
RUN_LOG_HASHES = {
    "first": "d7d3c1922d2a874932b7807b55812a371c3f2d7ae0af375498a79a24e57f25f6",
    "second": "eea773aa17fca4d9a72fc669f20c0ec0fc8a96ddceb6f6ad268305117a25189d",
}
# Main results table CSV for seeds 0 and 1, recorded with the run-log hashes above.
RESULTS_HASH = "b1b2fbb506bad681c00d35dcf2bd5b851bc0b515ece996bc434559ccd4294d85"


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


def test_results_fingerprint(tmp_path):
    path = tmp_path / "results.csv"
    run_many(fingerprint_config("second"), seeds=[0, 1]).save_csv(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == RESULTS_HASH


PUBLIC_NAMES = {
    "SimConfig",
    "BidderSpec",
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
    assert importlib.util.find_spec("bidlab.world") is None
    with pytest.raises(AttributeError):
        _ = importlib.import_module("bidlab").WorldConfig
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = " ".join(pyproject["project"]["dependencies"]).lower()
    assert "yaml" not in dependencies
    assert "scripts" not in pyproject["project"]
    assert pyproject["project"]["name"] == "bidlab"
