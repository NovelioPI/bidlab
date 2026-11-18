import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


EXAMPLE_LABELS = {
    "getting_started/basic.py": [("truthful",), ("shaded",), ("pid",), ("bandit",)],
    "markets/segments.py": [
        (bidder, segment)
        for bidder in ("everywhere", "sports_only", "pid")
        for segment in ("news", "sports", "games")
    ],
    "markets/custom_market.py": [
        (bidder, segment)
        for bidder in ("truthful", "night_only", "bandit")
        for segment in ("day", "night")
    ],
    "markets/independent_values.py": [("truthful",), ("equilibrium",), ("low",)],
}


def test_every_example_is_listed():
    examples = ROOT / "examples"
    scripts = {path.relative_to(examples).as_posix() for path in examples.rglob("*.py")}
    assert scripts == set(EXAMPLE_LABELS)


@pytest.mark.parametrize("name", sorted(EXAMPLE_LABELS))
def test_example_runs(name):
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0].split()[0] == "bidder"
    width = len(EXAMPLE_LABELS[name][0])
    assert [tuple(line.split()[:width]) for line in lines[1:]] == EXAMPLE_LABELS[name]


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
