import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_basic_example_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "basic.py")],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0].split()[0] == "bidder"
    assert [line.split()[0] for line in lines[1:]] == [
        "truthful",
        "shaded",
        "pid",
        "bandit",
    ]


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
