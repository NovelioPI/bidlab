"""Built-in bidders, grouped by family, and the `Bidder` and `Stack` classes.

| Family | Bidders |
|---|---|
| `baselines` | `Linear`, `OracleValue` |
| `pacing` | `PidPacing` |
| `bandits` | `UcbMultiplier` |
"""

from bidlab.bidders import bandits, baselines, pacing
from bidlab.bidders.bandits import UcbMultiplier
from bidlab.bidders.base import Bidder
from bidlab.bidders.baselines import Linear, OracleValue
from bidlab.bidders.pacing import PidPacing
from bidlab.bidders.stack import Stack

__all__ = [
    "Bidder",
    "Linear",
    "OracleValue",
    "PidPacing",
    "Stack",
    "UcbMultiplier",
    "bandits",
    "baselines",
    "pacing",
]
