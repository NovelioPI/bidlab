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
GUIDES = sorted((ROOT / "docs" / "guides").glob("*.md"))
DOCS = [ROOT / "README.md", *GUIDES]


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


def python_blocks(path):
    return re.findall(r"```python\n(.*?)```", path.read_text(), flags=re.DOTALL)


def local_links(path):
    targets = re.findall(r"\]\(([^)#\s]+)", path.read_text())
    return [target for target in targets if not target.startswith("http")]


@pytest.mark.parametrize("path", DOCS, ids=lambda path: path.name)
def test_guide_script_runs(path, tmp_path):
    script = tmp_path / "script.py"
    script.write_text("\n".join(python_blocks(path)))
    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_local_links_resolve():
    broken = [
        f"{path.relative_to(ROOT)}: {target}"
        for path in DOCS
        for target in local_links(path)
        if not (path.parent / target).exists()
    ]
    assert not broken, f"links to missing files: {broken}"


def test_examples_linked_from_guides():
    missing = []
    for name, (_, guides) in EXAMPLES.items():
        if not guides:
            missing.append(f"no guide is listed for {name}")

        for guide in guides:
            targets = local_links(ROOT / "docs" / "guides" / guide)
            if f"../../examples/{name}" not in targets:
                missing.append(f"{guide} does not link {name}")

    assert not missing, missing


def test_readme_links_everything():
    linked = {(ROOT / target).resolve() for target in local_links(ROOT / "README.md")}
    examples = [ROOT / "examples" / name for name in EXAMPLES]
    missing = [
        str(path.relative_to(ROOT))
        for path in [*GUIDES, *examples]
        if path.resolve() not in linked
    ]
    assert not missing, f"README does not link: {missing}"
