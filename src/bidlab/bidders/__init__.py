"""Built-in bidders, grouped by family, and the `Bidder` base class.

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
from bidlab.bidders.pacing import PidPacing

__all__ = [
    "Bidder",
    "PidPacing",
    "Truthful",
    "UcbMultiplier",
    "bandits",
    "baselines",
    "pacing",
]
