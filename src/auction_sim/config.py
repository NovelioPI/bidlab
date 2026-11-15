from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from auction_sim.auctions import AUCTION_KINDS, AuctionKind
from auction_sim.world import WorldConfig


@dataclass(frozen=True)
class AuctionConfig:
    kind: AuctionKind = "second"
    reserve: float = 0.0


@dataclass(frozen=True)
class BidderSpec:
    name: str
    type: str
    budget: float
    value_per_conversion: float
    cpa_target: float | None = None
    params: dict[str, float] = field(default_factory=dict)
    reward: str = "surplus"
    episode_reward: str | None = None
    reward_params: dict[str, float] = field(default_factory=dict)

    def merged_reward_params(self) -> dict[str, float]:
        if self.cpa_target is None:
            return dict(self.reward_params)
        return {"cpa_target": self.cpa_target, **self.reward_params}


@dataclass(frozen=True)
class SimConfig:
    bidders: tuple[BidderSpec, ...]
    episodes: int = 10
    steps: int = 1000
    seed: int = 0
    auction: AuctionConfig = field(default_factory=AuctionConfig)
    world: WorldConfig = field(default_factory=WorldConfig)


def load_config(path: Path) -> SimConfig:
    with path.open() as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: top level must be a mapping")
    _reject_unknown(raw, SimConfig, "config")
    if not raw.get("bidders"):
        raise ValueError(f"{path}: 'bidders' must list at least one bidder")
    bidders = tuple(_bidder_spec(entry) for entry in raw["bidders"])
    names = [bidder.name for bidder in bidders]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: bidder names must be unique, got {names}")
    auction = raw.get("auction", {})
    world = raw.get("world", {})
    _reject_unknown(auction, AuctionConfig, "auction")
    _reject_unknown(world, WorldConfig, "world")
    if auction.get("kind", "second") not in AUCTION_KINDS:
        raise ValueError(f"auction kind must be one of {AUCTION_KINDS}")
    return SimConfig(
        bidders=bidders,
        episodes=int(raw.get("episodes", 10)),
        steps=int(raw.get("steps", 1000)),
        seed=int(raw.get("seed", 0)),
        auction=AuctionConfig(**auction),
        world=WorldConfig(**world),
    )


def _bidder_spec(entry: dict) -> BidderSpec:
    _reject_unknown(entry, BidderSpec, f"bidder {entry.get('name', '?')!r}")
    for key in ("name", "type", "budget", "value_per_conversion"):
        if key not in entry:
            raise ValueError(f"bidder entry {entry} is missing {key!r}")
    if entry["budget"] <= 0:
        raise ValueError(f"bidder {entry['name']!r}: budget must be positive")
    return BidderSpec(
        name=str(entry["name"]),
        type=str(entry["type"]),
        budget=float(entry["budget"]),
        value_per_conversion=float(entry["value_per_conversion"]),
        cpa_target=None
        if entry.get("cpa_target") is None
        else float(entry["cpa_target"]),
        params={k: float(v) for k, v in entry.get("params", {}).items()},
        reward=str(entry.get("reward", "surplus")),
        episode_reward=entry.get("episode_reward"),
        reward_params={k: float(v) for k, v in entry.get("reward_params", {}).items()},
    )


def _reject_unknown(raw: dict, schema: type, where: str) -> None:
    unknown = set(raw) - {f.name for f in fields(schema)}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
