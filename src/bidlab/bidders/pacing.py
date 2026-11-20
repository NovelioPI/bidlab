"""Budget pacing bidders that steer spend toward a plan."""

from typing import TYPE_CHECKING, override

import numpy as np

from bidlab.bidders.base import Bidder
from bidlab.types import BidResult

if TYPE_CHECKING:
    from bidlab.config import Campaign


class PidPacing(Bidder):
    """Bids λ × value; a PID loop moves λ so spend tracks the budget evenly.

    The plan spends the budget evenly over the episode's horizon. After each
    auction, the error is (planned spend − actual spend) / budget, where the
    planned spend is budget × the auction's time / horizon. Then
    λ = ``initial_multiplier`` × exp(kp·error + ki·mean error + kd·Δerror),
    clipped to ``multiplier_range``. λ resets at the start of each episode.

    Parameters
    ----------
    initial_multiplier : float, optional
        λ at the start of each episode. Must be > 0.
    kp, ki, kd : float, optional
        Proportional, integral and derivative gains. Must be >= 0.
    multiplier_range : tuple[float, float], optional
        Lowest and highest λ, with 0 <= lowest <= highest.
    log_state : bool, optional
        Log λ as ``"multiplier"`` at every auction, the value the bid used.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    Alone in a first-price auction, the bidder spends about half its budget
    in each half of the episode:

    >>> import bidlab as bl
    >>> pid = bl.Campaign("pid", bl.PidPacing(), budget=10, event_value=5)
    >>> config = bl.SimConfig(
    ...     campaigns=[pid], auction=bl.FirstPrice(), episodes=1, steps=200
    ... )
    >>> log = bl.run(config)
    >>> round(float(log.price[0, :100].sum()), 1), round(float(log.price[0, 100:].sum()), 1)
    (5.4, 4.6)
    """

    def __init__(
        self,
        initial_multiplier: float = 1.0,
        kp: float = 20.0,
        ki: float = 2.0,
        kd: float = 0.0,
        multiplier_range: tuple[float, float] = (0.0, 10.0),
        log_state: bool = True,
    ):
        if initial_multiplier <= 0:
            raise ValueError(
                f"initial_multiplier must be > 0, got {initial_multiplier}"
            )

        for name, gain in (("kp", kp), ("ki", ki), ("kd", kd)):
            if gain < 0:
                raise ValueError(f"{name} must be >= 0, got {gain}")

        lowest, highest = multiplier_range
        if not 0 <= lowest <= highest:
            raise ValueError(
                f"multiplier_range needs 0 <= lowest <= highest, got {multiplier_range}"
            )

        self.initial_multiplier = initial_multiplier
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.multiplier_range = multiplier_range
        self.log_state = log_state

    @override
    def on_episode_start(self, campaign: "Campaign") -> None:
        self.budget = campaign.budget
        self.updates = 0
        self.spent = 0.0
        self.integral = 0.0
        self.previous_error = 0.0
        self.multiplier = self.initial_multiplier

    @override
    def forward(self, value: float, horizon: float) -> float:
        # update needs the horizon, and only forward gets the inputs.
        self.horizon = horizon
        if self.log_state:
            self.log("multiplier", self.multiplier)

        return self.multiplier * value

    @override
    def update(self, result: BidResult) -> None:
        self.updates += 1
        self.spent += result.price
        planned = self.budget * result.time / self.horizon
        # Dividing by budget makes the gains independent of budget size.
        error = (planned - self.spent) / self.budget
        # A per-update sum winds up over long episodes; the mean does not.
        self.integral += (error - self.integral) / self.updates
        derivative = error - self.previous_error
        self.previous_error = error
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        lowest, highest = self.multiplier_range
        self.multiplier = float(
            np.clip(self.initial_multiplier * np.exp(control), lowest, highest)
        )
