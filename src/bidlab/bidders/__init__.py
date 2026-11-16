"""Built-in bidders, grouped by family, and the `Bidder` and `Stack` classes.

| Family | Bidders |
|---|---|
| `baselines` | `Truthful` |
| `pacing` | `PidPacing` |
| `bandits` | `UcbMultiplier` |
"""

from bidlab.bidders import bandits, baselines, pacing
from bidlab.bidders.bandits import UcbMultiplier
from bidlab.bidders.base import Bidder
from bidlab.bidders.baselines import Truthful
from bidlab.bidders.pacing import PidGains, PidPacing
from bidlab.bidders.stack import Stack

__all__ = [
    "Bidder",
    "PidGains",
    "PidPacing",
    "Stack",
    "Truthful",
    "UcbMultiplier",
    "bandits",
    "baselines",
    "pacing",
]
