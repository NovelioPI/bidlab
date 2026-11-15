import csv
import subprocess
from pathlib import Path

import pytest

from auction_sim.cli import main, parse_seeds
from auction_sim.config import load_config
from auction_sim.runner import COLUMNS, run_many

EXAMPLE = Path(__file__).parent.parent / "examples" / "basic.yaml"

SMALL = """
episodes: 2
steps: 100
bidders:
  - {name: a, type: truthful, budget: 10, value_per_conversion: 5}
  - {name: b, type: "sample_plugins:Doubler", budget: 10, value_per_conversion: 4,
     reward: "sample_plugins:double_price", reward_params: {weight: 2}}
"""


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_builtin_and_module_path_resolve(tmp_path):
    config = load_config(write(tmp_path, SMALL))
    assert [b.type for b in config.bidders] == ["truthful", "sample_plugins:Doubler"]
    assert config.bidders[1].reward_params == {"weight": 2.0}
    assert config.auction.kind == "second"
    rows = run_many(config, [0])
    assert [row["bidder"] for row in rows] == ["a", "b"]


def test_unknown_bidder_lists_names(tmp_path):
    config = load_config(
        write(tmp_path, SMALL.replace("type: truthful", "type: truthfull"))
    )
    with pytest.raises(ValueError, match="pid_pacing"):
        run_many(config, [0])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("episodes: 2", "episodes: 2\nepisdoes: 3"), "unknown keys"),
        (("budget: 10, value", "budget: 10, budgt: 1, value"), "unknown keys"),
        (("budget: 10, value", "budget: 0, value"), "budget must be positive"),
        (("name: b", "name: a"), "unique"),
        (("episodes: 2", "episodes: 2\nauction: {kind: vcg}"), "auction kind"),
    ],
)
def test_bad_config_is_rejected(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, SMALL.replace(*change, 1)))


def test_parse_seeds():
    assert parse_seeds("0-3") == [0, 1, 2, 3]
    assert parse_seeds("5,2") == [5, 2]


def test_run_many_rows(tmp_path):
    rows = run_many(load_config(write(tmp_path, SMALL)), [0, 1, 2])
    assert len(rows) == 3 * 2
    assert all(set(row) == set(COLUMNS) for row in rows)
    assert [row["seed"] for row in rows] == [0, 0, 1, 1, 2, 2]
    assert rows[0] != rows[2]


def test_cli_writes_csv(tmp_path, capsys):
    out = tmp_path / "results.csv"
    main(["run", str(write(tmp_path, SMALL)), "--seeds", "0-1", "--out", str(out)])
    with out.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert tuple(rows[0]) == COLUMNS
    assert "Mean over 2 seed(s)" in capsys.readouterr().out


def test_example_config_runs_from_command_line(tmp_path):
    out = tmp_path / "results.csv"
    result = subprocess.run(
        [
            "auction-sim",
            "run",
            str(EXAMPLE),
            "--seeds",
            "0",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "bandit" in result.stdout
    assert len(out.read_text().splitlines()) == 1 + 4
