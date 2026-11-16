"""bidlab: simulate ad auctions to test and compare bidding methods.

Examples
--------
>>> import bidlab as bl
>>> config = bl.SimConfig(
...     bidders=[
...         bl.BidderSpec("truthful", bl.Truthful(), bl.Campaign(budget=50, value=5)),
...         bl.BidderSpec("pid", bl.PidPacing(), bl.Campaign(budget=15, value=5)),
...     ],
...     auction=bl.SecondPrice(reserve=0.01),
...     episodes=2,
...     steps=100,
... )
>>> results = bl.run_many(config, seeds=range(2))
>>> len(results)
4
"""

from bidlab import bidders, markets, rewards
from bidlab.auctions import FirstPrice, SecondPrice
from bidlab.bidders import Bidder, PidPacing, Stack, Truthful, UcbMultiplier
from bidlab.config import BidderSpec, SimConfig
from bidlab.metrics import bidder_metrics, platform_metrics, regret, segment_metrics
from bidlab.results import Results
from bidlab.runner import run_many
from bidlab.simulator import RunLog, run
from bidlab.types import Campaign, EpisodeStats, Feedback, Observation

__all__ = [
    "Bidder",
    "BidderSpec",
    "Campaign",
    "EpisodeStats",
    "Feedback",
    "FirstPrice",
    "Observation",
    "PidPacing",
    "Results",
    "RunLog",
    "SecondPrice",
    "SimConfig",
    "Stack",
    "Truthful",
    "UcbMultiplier",
    "bidder_metrics",
    "bidders",
    "markets",
    "platform_metrics",
    "regret",
    "rewards",
    "run",
    "run_many",
    "segment_metrics",
]
