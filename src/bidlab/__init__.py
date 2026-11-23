"""bidlab: simulate ad auctions to test and compare bidding methods.

Examples
--------
>>> import bidlab as bl
>>> config = bl.SimConfig(
...     campaigns=[
...         bl.Campaign(
...             "truthful",
...             bl.Stack(bl.OracleValue(), bl.Linear()),
...             budget=50,
...             base_value=5,
...         ),
...         bl.Campaign(
...             "pid",
...             bl.Stack(bl.OracleValue(), bl.PidPacing()),
...             budget=15,
...             base_value=5,
...         ),
...     ],
...     auction=bl.SecondPrice(reserve=0.01),
...     episodes=2,
...     steps=100,
... )
>>> results = bl.run_many(config, seeds=range(2))
>>> len(results)
4
"""

from bidlab import bidders, markets
from bidlab.auctions import FirstPrice, SecondPrice
from bidlab.bidders import (
    Bidder,
    Linear,
    OracleValue,
    PidPacing,
    Stack,
    UcbMultiplier,
)
from bidlab.callbacks import Callback
from bidlab.config import Campaign, Feedback, SimConfig
from bidlab.metrics import campaign_metrics, platform_metrics, regret, segment_metrics
from bidlab.results import Results
from bidlab.runner import run_many
from bidlab.simulator import RunLog, run
from bidlab.types import BidResult, Conversion, EpisodeReport, Event, MarketTruth

__all__ = [
    "BidResult",
    "Bidder",
    "Callback",
    "Campaign",
    "Conversion",
    "EpisodeReport",
    "Event",
    "Feedback",
    "FirstPrice",
    "Linear",
    "MarketTruth",
    "OracleValue",
    "PidPacing",
    "Results",
    "RunLog",
    "SecondPrice",
    "SimConfig",
    "Stack",
    "UcbMultiplier",
    "bidders",
    "campaign_metrics",
    "markets",
    "platform_metrics",
    "regret",
    "run",
    "run_many",
    "segment_metrics",
]
