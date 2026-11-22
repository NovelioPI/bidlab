"""Fixed bidders that do not learn, for use as baselines."""

from typing import TYPE_CHECKING, override

from scipy.special import expit, logit

from bidlab.bidders.base import Bidder
from bidlab.types import Event, MarketTruth

if TYPE_CHECKING:
    from bidlab.config import Campaign


class Linear(Bidder):
    """Bids `multiplier` × value; multiplier 1 is truthful.

    The value is the ``value`` input from the bidder before it in a `Stack`,
    such as `OracleValue`. Truthful bidding is optimal in a second-price
    auction without a budget limit. Set ``multiplier`` below 1 for a fixed
    bid shade.

    Parameters
    ----------
    multiplier : float, optional
        The output as a fraction of the incoming value. Must be >= 0.

    Raises
    ------
    ValueError
        If ``multiplier`` is negative.

    Examples
    --------
    >>> import bidlab as bl
    >>> bidder = bl.Stack(bl.OracleValue(), bl.Linear(multiplier=0.8))
    >>> shaded = bl.Campaign("shaded", bidder, budget=10, base_value=5)
    >>> log = bl.run(bl.SimConfig(campaigns=[shaded], episodes=1, steps=20))
    >>> bool((log.bids == 0.8 * log.true_values).all())
    True
    """

    def __init__(self, multiplier: float = 1.0):
        if multiplier < 0:
            raise ValueError(f"multiplier must be >= 0, got {multiplier}")

        self.multiplier = multiplier

    @override
    def forward(self, value: float) -> float:
        return self.multiplier * value


class OracleValue(Bidder):
    """Outputs the campaign's true value, read from the market, with optional noise.

    The true value is ``base_value`` times the chance of the campaign's
    event: 1 for ``impression``, pCTR for ``click``, pCTR × pCVR for
    ``conversion``. It reads the ``market`` input, so it is an oracle: put
    it first in a `Stack` to give a bid policy the value it would have with
    a perfect estimator. With ``noise`` above 0, each auction adds normal
    noise to logit(pCTR) and logit(pCVR), so the value is an estimate
    with a known error. A pCTR of 1, from a market without a click stage,
    stays exact.

    Parameters
    ----------
    noise : float, optional
        Standard deviation of the noise added to each rate's logit.
        Must be >= 0. With noise above 0, each auction draws two numbers
        from the bidder's stream; with 0, none.

    Raises
    ------
    ValueError
        If ``noise`` is negative or NaN.

    See Also
    --------
    Stack : Chains bidders, from a value to a bid.

    Examples
    --------
    >>> import bidlab as bl
    >>> bidder = bl.Stack(bl.OracleValue(), bl.Linear())
    >>> truthful = bl.Campaign("truthful", bidder, budget=1e6, base_value=5)
    >>> log = bl.run(bl.SimConfig(campaigns=[truthful], episodes=1, steps=20))
    >>> bool((log.bids == log.true_values).all())
    True
    """

    def __init__(self, noise: float = 0.0):
        # The negated check also rejects NaN.
        if not noise >= 0:
            raise ValueError(f"noise must be >= 0, got {noise}")

        self.noise = float(noise)

    @override
    def on_episode_start(self, campaign: "Campaign") -> None:
        self.base_value = campaign.base_value
        self.event = campaign.event

    @override
    def forward(self, market: MarketTruth) -> float:
        pctr, pcvr = market.pctr, market.pcvr
        if self.noise:
            ctr_error, cvr_error = self.rng.standard_normal(2)
            pctr = float(expit(logit(pctr) + self.noise * ctr_error))
            pcvr = float(expit(logit(pcvr) + self.noise * cvr_error))

        if self.event is Event.IMPRESSION:
            return self.base_value

        if self.event is Event.CLICK:
            return self.base_value * pctr

        # Risk: if grouped as base × (pCTR × pCVR), then bits differ from true_values.
        return self.base_value * pctr * pcvr
