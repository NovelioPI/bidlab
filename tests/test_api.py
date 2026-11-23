import dataclasses
import hashlib
import importlib
import importlib.util
import re
import tomllib
from pathlib import Path
from typing import override

import pytest
from scipy import stats

from bidlab.auctions import FirstPrice, SecondPrice
from bidlab.bidders import Linear, OracleValue, PidPacing, Stack, UcbMultiplier
from bidlab.config import Campaign, SimConfig
from bidlab.markets import IndependentValues, LogisticMarket, Segment, Segmented
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
# Re-recorded with the ctr and cpc columns (feature 010); without them the CSV equals main's.
RESULTS_HASH = "f328e6d91cc5b5089072a703d58221a8499097c9d3c4fa84dcd4d3b28250dd45"


class ConversionBandit(UcbMultiplier):
    """Rewards each conversion, as ``step_reward=rewards.conversions`` did."""

    @override
    def update(self, result: BidResult) -> None:
        self.arm_reward += float(result.converted)


def fingerprint_config(kind):
    campaigns = (
        Campaign("t", Stack(OracleValue(), Linear()), budget=40.0, base_value=5.0),
        Campaign("p", Stack(OracleValue(), PidPacing()), budget=10.0, base_value=5.0),
        Campaign(
            "b",
            Stack(OracleValue(), ConversionBandit()),
            budget=40.0,
            base_value=4.0,
        ),
        Campaign(
            "s",
            Stack(OracleValue(), Linear(multiplier=0.8)),
            budget=40.0,
            base_value=6.0,
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
        Campaign("t", Stack(OracleValue(), Linear()), budget=40.0, base_value=5.0),
        Campaign(
            "b",
            Stack(OracleValue(), ConversionBandit()),
            budget=40.0,
            base_value=4.0,
        ),
        Campaign(
            "s",
            Stack(OracleValue(), Linear(multiplier=0.8)),
            budget=40.0,
            base_value=6.0,
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


# Recorded on main (a6f29f4) with numpy 2.5.4, before markets could draw a click stage.
MAIN_HASHES_WITHOUT_CLICKS = {
    "logistic": "87f5c19f555a9d5ec19f8b0c96cca68d477bf36bb3533c62f282d0af4f5638e2",
    "segmented": "a4fef502d9ac7fa081de7c1e4dfd6345c3893095356032a121692e4f4774fe89",
    "independent": "b77c2546f13de2579c756195b9c08d31618d6f758dd0d657729777da94f3ede8",
}


def market_without_clicks(kind):
    if kind == "logistic":
        return LogisticMarket()
    if kind == "segmented":
        return Segmented(
            (Segment("mobile", 0.6, cvr_lift=0.5), Segment("desktop", 0.4))
        )
    return IndependentValues(stats.uniform(0, 0.1))


@pytest.mark.parametrize("kind", list(MAIN_HASHES_WITHOUT_CLICKS))
def test_no_click_stage_matches_main(kind):
    campaigns = (
        Campaign("t", Stack(OracleValue(), Linear()), budget=20.0, base_value=5.0),
        Campaign(
            "b", Stack(OracleValue(), UcbMultiplier()), budget=20.0, base_value=4.0
        ),
        Campaign(
            "s",
            Stack(OracleValue(), Linear(multiplier=0.8)),
            budget=20.0,
            base_value=6.0,
        ),
    )
    config = SimConfig(
        campaigns=campaigns,
        episodes=2,
        steps=300,
        seed=7,
        market=market_without_clicks(kind),
    )
    log = run(config)
    digest = hashlib.sha256()
    for array in (log.bids, log.true_values, log.winner, log.price, log.converted):
        digest.update(array.tobytes())

    assert digest.hexdigest() == MAIN_HASHES_WITHOUT_CLICKS[kind]


# Recorded on main (5627127) with numpy 2.5.4, before campaigns had an event.
MAIN_HASHES_WITH_CLICKS = {
    "logistic": "2201c7f2d63518fbbe63cb96e5b4dbe35b726337dbe5940d39fb94fa4fb9f87f",
    "segmented": "bbe84aa1f13b9247735f301b654c8dd3013d75c8364b5562fb51f16aaa7bc483",
}


def market_with_clicks(kind):
    base = LogisticMarket(ctr_bias_mean=-3.0)
    if kind == "logistic":
        return base
    return Segmented(
        (Segment("mobile", 0.6, cvr_lift=0.5), Segment("desktop", 0.4)), base=base
    )


@pytest.mark.parametrize("kind", list(MAIN_HASHES_WITH_CLICKS))
def test_click_stage_matches_main(kind):
    campaigns = (
        Campaign("t", Stack(OracleValue(), Linear()), budget=20.0, base_value=5.0),
        Campaign(
            "b", Stack(OracleValue(), UcbMultiplier()), budget=20.0, base_value=4.0
        ),
        Campaign(
            "s",
            Stack(OracleValue(), Linear(multiplier=0.8)),
            budget=20.0,
            base_value=6.0,
        ),
    )
    config = SimConfig(
        campaigns=campaigns,
        episodes=2,
        steps=300,
        seed=7,
        market=market_with_clicks(kind),
    )
    log = run(config)
    digest = hashlib.sha256()
    for array in (
        log.bids,
        log.true_values,
        log.winner,
        log.price,
        log.clicked,
        log.converted,
    ):
        digest.update(array.tobytes())

    assert digest.hexdigest() == MAIN_HASHES_WITH_CLICKS[kind]


PUBLIC_NAMES = {
    "Callback",
    "SimConfig",
    "Feedback",
    "Conversion",
    "Stack",
    "Campaign",
    "markets",
    "FirstPrice",
    "SecondPrice",
    "Bidder",
    "Linear",
    "MarketTruth",
    "OracleValue",
    "PidPacing",
    "UcbMultiplier",
    "BidResult",
    "EpisodeReport",
    "Event",
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
        "LogisticMarket",
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
