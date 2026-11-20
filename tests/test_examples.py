import ast
import dataclasses
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

import bidlab as bl

ROOT = Path(__file__).parent.parent


# Script: (expected answer, guides that must link it).
EXAMPLES = {
    "getting_started/basic.py": ("yes", ("results.md", "simulation.md")),
    "markets/segments.py": ("no", ("markets.md",)),
    "markets/custom_market.py": ("yes", ("markets.md",)),
    "markets/independent_values.py": ("yes", ("markets.md",)),
    "bidders/segment_rate.py": ("yes", ("bidders.md",)),
    "composite/night_shade.py": ("yes", ("composite-bidders.md",)),
    "logging/pid_trace.py": ("yes", ("logging.md",)),
    "rewards/cpa_target.py": ("yes", ("rewards.md",)),
    "rewards/custom_reward.py": ("yes", ("rewards.md",)),
    "simulation/estimate_noise.py": ("yes", ("simulation.md",)),
    "simulation/clock.py": ("yes", ("simulation.md",)),
    "results/regret.py": ("yes", ("results.md",)),
    "results/reserve.py": ("yes", ("results.md",)),
}


def test_every_example_is_listed():
    examples = ROOT / "examples"
    scripts = {path.relative_to(examples).as_posix() for path in examples.rglob("*.py")}
    assert scripts == set(EXAMPLES)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_states_question(name):
    tree = ast.parse((ROOT / "examples" / name).read_text())
    docstring = ast.get_docstring(tree)
    assert docstring is not None
    assert docstring.splitlines()[0].endswith("?")


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_answers(name):
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    last = result.stdout.strip().splitlines()[-1]
    expected, _ = EXAMPLES[name]
    assert last.startswith(f"Answer: {expected},"), f"{name} printed: {last}"


def test_reserve_runs_without_pandas():
    script = ROOT / "examples" / "results" / "reserve.py"
    hide_pandas = (
        "import runpy, sys; sys.modules['pandas'] = None; "
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", hide_pandas],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert "to_pandas skipped: pandas is not installed" in lines
    assert lines[-1].startswith("Answer: yes,")


MEMBER_CLASSES = (
    bl.Bidder,
    bl.markets.Market,
    bl.Callback,
    bl.Results,
    bl.Campaign,
    bl.RunLog,
    bl.Observation,
    bl.Feedback,
    bl.EpisodeStats,
    bl.SimConfig,
    bl.BidderSpec,
    bl.FirstPrice,
    bl.SecondPrice,
    bl.markets.Segment,
)

# Public names a user never calls or writes, so no example needs them.
COVERAGE_EXCLUSIONS = {
    "Bidder.bind": "the simulator calls it",
    "BidderSpec.build": "the simulator calls it",
    "BidderSpec.merged_reward_params": "the simulator calls it",
    "bl.markets.Distribution": "a Protocol type for type hints",
    "bl.rewards.StepReward": "a Protocol type for type hints",
    "bl.rewards.EpisodeReward": "a Protocol type for type hints",
}


def module_names(module):
    prefix = module.__name__.replace("bidlab", "bl")
    return {
        f"{prefix}.{name}"
        for name, value in vars(module).items()
        if (inspect.isclass(value) or inspect.isfunction(value))
        and value.__module__ == module.__name__
        and not name.startswith("_")
    }


def member_names(cls):
    prefix = cls.__name__
    names = {
        name
        for name, value in inspect.getmembers(cls)
        if not name.startswith("_")
        and (inspect.isfunction(value) or isinstance(value, property))
    }
    if dataclasses.is_dataclass(cls):
        names |= {field.name for field in dataclasses.fields(cls)}

    return {f"{prefix}.{name}" for name in names}


def public_api():
    names = {
        f"bl.{name}" for name in bl.__all__ if not inspect.ismodule(getattr(bl, name))
    }
    names |= module_names(bl.markets) | module_names(bl.rewards)
    for cls in MEMBER_CLASSES:
        names |= member_names(cls)

    return names


def dotted_from_bl(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value

    if isinstance(node, ast.Name) and node.id == "bl":
        return ".".join(["bl", *reversed(parts)])

    return None


def names_used_in_examples():
    used = set()
    for path in (ROOT / "examples").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute):
                used.add(node.attr)
                used.add(dotted_from_bl(node))
            elif isinstance(node, ast.keyword):
                used.add(node.arg)
            elif isinstance(node, ast.FunctionDef):
                used.add(node.name)

    return used


def test_examples_cover_public_api():
    used = names_used_in_examples()
    wanted = public_api() - COVERAGE_EXCLUSIONS.keys()
    # A module name must appear in full; a member such as Feedback.price counts as `.price`.
    missing = sorted(
        name
        for name in wanted
        if (name if name.startswith("bl.") else name.split(".")[-1]) not in used
    )
    assert not missing, f"no example uses: {missing}"


def test_coverage_exclusions_exist():
    stale = sorted(COVERAGE_EXCLUSIONS.keys() - public_api())
    assert not stale, f"excluded names no longer in bidlab: {stale}"


def test_readme_code_runs(tmp_path):
    readme = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    script = tmp_path / "readme_script.py"
    script.write_text("\n".join(blocks))
    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("bidder") >= 3


def test_readme_shows_segmented_market_with_targeted_bidder():
    readme = (ROOT / "README.md").read_text()
    blocks = "\n".join(re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL))
    assert "bl.markets.Segmented(" in blocks
    assert "segments=(" in blocks
    assert "segment_names" in readme


def test_readme_shows_value_stage_before_pacing():
    readme = (ROOT / "README.md").read_text()
    blocks = "\n".join(re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL))
    assert "class SegmentRate(bl.Bidder):" in blocks
    assert "self.pending[observation.time]" in blocks
    assert "def update_batch(self, feedbacks):" in blocks
    assert "bl.Stack(SegmentRate(), bl.PidPacing())" in blocks


def test_readme_shows_logging():
    readme = (ROOT / "README.md").read_text()
    blocks = "\n".join(re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL))
    assert 'self.log({"factor": self.factor' in blocks
    assert "class Progress(bl.Callback):" in blocks
    assert 'log.logs["multiplier"]["paced"]' in blocks
    assert 'results.logs["multiplier"]["paced"]' in blocks
    assert "callbacks=[Progress()]" in blocks


def test_readme_bidder_table_arguments():
    readme = (ROOT / "README.md").read_text()
    rows = {
        line.split("|")[1].strip(): line
        for line in readme.splitlines()
        if line.startswith("| `")
    }
    for gain in ("`kp` (20)", "`ki` (2)", "`kd` (0)", "`log_state` (True)"):
        assert gain in rows["`PidPacing`"]

    assert "`log_state` (True)" in rows["`UcbMultiplier`"]


def test_readme_shows_composite_bidder():
    readme = (ROOT / "README.md").read_text()
    blocks = "\n".join(re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL))
    assert "self.shade = Shade()" in blocks
    assert "self.pacing = bl.PidPacing()" in blocks
    assert "factor=factor" in blocks
    assert "named_bidders()" in blocks
