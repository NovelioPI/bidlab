import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

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
