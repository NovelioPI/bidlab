"""Budget pacing bidders that steer spend toward a plan."""

from typing import override

import numpy as np

from bidlab.bidders.base import Bidder
from bidlab.types import Campaign, Feedback, Observation


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

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    Alone in a first-price auction, the bidder spends about half its budget
    in each half of the episode:

    >>> import bidlab as bl
    >>> pid = bl.BidderSpec("pid", bl.PidPacing(), bl.Campaign(budget=10, value=5))
    >>> config = bl.SimConfig(
    ...     bidders=[pid], auction=bl.FirstPrice(), episodes=1, steps=200
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

    @override
    def on_episode_start(self, campaign: Campaign) -> None:
        self.budget = campaign.budget
        self.horizon = campaign.time_left(0.0)
        self.updates = 0
        self.spent = 0.0
        self.integral = 0.0
        self.previous_error = 0.0
        self.multiplier = self.initial_multiplier

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return self.multiplier * value

    @override
    def update(self, feedback: Feedback) -> None:
        self.updates += 1
        self.spent += feedback.price
        planned = self.budget * feedback.time / self.horizon
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
