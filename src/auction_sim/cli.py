import argparse
import csv
from pathlib import Path

import numpy as np

from auction_sim.config import load_config
from auction_sim.runner import COLUMNS, ResultRow, run_many

SUMMARY_COLUMNS = ("spend", "conversions", "surplus", "budget_util", "regret", "reward")


def parse_seeds(text: str) -> list[int]:
    """Parse '0-9' or '1,4,7' into a list of seeds."""
    if "-" in text:
        start, end = (int(part) for part in text.split("-", 1))
        return list(range(start, end + 1))
    return [int(part) for part in text.split(",")]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="auction-sim")
    commands = parser.add_subparsers(dest="command", required=True)
    run_command = commands.add_parser("run", help="run an experiment config")
    run_command.add_argument("config", type=Path)
    run_command.add_argument("--seeds", type=parse_seeds, help="'0-9' or '1,4,7'")
    run_command.add_argument(
        "--out", type=Path, help="write one CSV row per seed and bidder"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    rows = run_many(config, args.seeds or [config.seed])
    if args.out:
        with args.out.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    _print_summary(rows)


def _print_summary(rows: list[ResultRow]) -> None:
    bidders = list(dict.fromkeys(str(row["bidder"]) for row in rows))
    seeds = len({row["seed"] for row in rows})
    print(f"Mean over {seeds} seed(s):")
    print(f"{'bidder':<16}" + "".join(f"{c:>13}" for c in SUMMARY_COLUMNS))
    for bidder in bidders:
        own = [row for row in rows if row["bidder"] == bidder]
        means = [np.mean([float(row[c]) for row in own]) for c in SUMMARY_COLUMNS]
        print(f"{bidder:<16}" + "".join(f"{m:>13.3f}" for m in means))
