import dataclasses
import hashlib
import importlib
import importlib.util
import re
import tomllib
from pathlib import Path
from typing import override

import pytest

from bidlab.auctions import FirstPrice, SecondPrice
from bidlab.bidders import Linear, PidPacing, UcbMultiplier
from bidlab.config import Campaign, SimConfig
from bidlab.runner import run_many
from bidlab.simulator import run
from bidlab.types import BidResult

ROOT = Path(__file__).parent.parent

# Recorded with numpy 2.5.4 when the reward arrays left RunLog (feature 009).
# The five arrays hashed equal those of the commit before, bffd15a, bit for bit.
RUN_LOG_HASHES = {
    "first": "3f42bf23edfa6f8f3ee3462aebb5d24b38aed3088ddd8cc90aa4123b838cebc5",
    "second": "cd529f75ad49b9df40659b8d715a21b5ace97c4d004f52b4e2cf096522db7edc",
}
# Main results table CSV for seeds 0 and 1, recorded with the run-log hashes above.
# Re-recorded without the reward columns; every other cell is unchanged.
RESULTS_HASH = "87ce5e0b95761cef26038273b7080f9b2816492bd4ba28f2a9cc599d4ca37fb7"


class ConversionBandit(UcbMultiplier):
    """Rewards each conversion, as ``step_reward=rewards.conversions`` did."""

    @override
    def update(self, result: BidResult) -> None:
        self.episode_reward += float(result.converted)


def fingerprint_config(kind):
    campaigns = (
        Campaign("t", Linear(), budget=40.0, event_value=5.0),
        Campaign("p", PidPacing(), budget=10.0, event_value=5.0),
        Campaign(
            "b",
            ConversionBandit(),
            budget=40.0,
            event_value=4.0,
        ),
        Campaign(
            "s",
            Linear(multiplier=0.8),
            budget=40.0,
            event_value=6.0,
            cpa_target=3.0,
        ),
    )
    return SimConfig(
        campaigns=campaigns,
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
        log.true_values,
        log.winner,
        log.price,
        log.converted,
    ):
        digest.update(array.tobytes())

    assert digest.hexdigest() == RUN_LOG_HASHES[kind]


def test_results_fingerprint(tmp_path):
    path = tmp_path / "results.csv"
    run_many(fingerprint_config("second"), seeds=[0, 1]).save_csv(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == RESULTS_HASH


# Main's floats from 151995c, re-hashed without the reward arrays (feature 009).
MAIN_HASHES_WITHOUT_PID = {
    "first": "c61ecaf187fb74d57d3d487d2c21cde59a1cfd1c0702753b7e5eaa6a010a80fc",
    "second": "398cb477c2848bbec49260123ce3fe30f85c77dfea60e1cd20df9da016a94270",
}


@pytest.mark.parametrize("kind", ["first", "second"])
def test_run_log_without_pid_matches_main(kind):
    campaigns = (
        Campaign("t", Linear(), budget=40.0, event_value=5.0),
        Campaign(
            "b",
            ConversionBandit(),
            budget=40.0,
            event_value=4.0,
        ),
        Campaign(
            "s",
            Linear(multiplier=0.8),
            budget=40.0,
            event_value=6.0,
            cpa_target=3.0,
        ),
    )
    auction = FirstPrice(0.01) if kind == "first" else SecondPrice(0.01)
    log = run(
        SimConfig(campaigns=campaigns, episodes=3, steps=300, seed=12, auction=auction)
    )
    digest = hashlib.sha256()
    for array in (
        log.bids,
        log.true_values,
        log.winner,
        log.price,
        log.converted,
    ):
        digest.update(array.tobytes())

    assert digest.hexdigest() == MAIN_HASHES_WITHOUT_PID[kind]


PUBLIC_NAMES = {
    "Callback",
    "SimConfig",
    "Stack",
    "Campaign",
    "markets",
    "FirstPrice",
    "SecondPrice",
    "Bidder",
    "Linear",
    "PidPacing",
    "UcbMultiplier",
    "BidResult",
    "EpisodeReport",
    "RunLog",
    "run",
    "run_many",
    "Results",
    "campaign_metrics",
    "platform_metrics",
    "regret",
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

    assert bl.bidders.baselines.Linear is bl.Linear
    assert bl.bidders.pacing.PidPacing is bl.PidPacing
    assert bl.bidders.bandits.UcbMultiplier is bl.UcbMultiplier


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


@pytest.mark.parametrize("name", ["Brief", "BidRequest"])
def test_removed_names(name):
    assert not hasattr(importlib.import_module("bidlab"), name)
    assert not hasattr(importlib.import_module("bidlab.types"), name)


def test_rewards_module_removed():
    assert not hasattr(importlib.import_module("bidlab"), "rewards")
    assert importlib.util.find_spec("bidlab.rewards") is None


def test_run_log_and_result_have_no_reward():
    log = run(fingerprint_config("second"))
    assert not hasattr(log, "rewards") and not hasattr(log, "episode_rewards")
    assert "reward" not in {field.name for field in dataclasses.fields(BidResult)}


def relative_links(markdown: str) -> list[str]:
    targets = re.findall(r"\]\(([^)]+)\)", markdown)
    return [target for target in targets if not target.startswith(("https://", "#"))]


def test_readme_links_work_on_pypi():
    # PyPI shows the README without the repo, so a relative link breaks there.
    assert relative_links((ROOT / "README.md").read_text()) == []
