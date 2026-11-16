"""Budget pacing bidders that steer spend toward a plan."""

from typing import override

import numpy as np

from bidlab.bidders.base import Bidder
from bidlab.types import Feedback, Observation


class PidPacing(Bidder):
    """Bids λ × value; a PID loop moves λ so spend tracks the budget evenly.

    The plan spends the budget evenly over the episode's steps. After each
    step, the error is (planned spend − actual spend) / budget, and
    λ = ``initial_multiplier`` × exp(kp·error + ki·mean error + kd·Δerror),
    clipped to [``min_multiplier``, ``max_multiplier``]. λ resets at the start
    of each episode.

    Parameters
    ----------
    params : dict[str, float]
        ``initial_multiplier`` (1.0), ``kp`` (20), ``ki`` (2), ``kd`` (0),
        ``min_multiplier`` (0), ``max_multiplier`` (10).
    rng : numpy.random.Generator
        Not used.

    Examples
    --------
    Alone in a first-price auction, the bidder spends about half its budget
    in each half of the episode:

    >>> import bidlab as bl
    >>> pid = bl.BidderSpec("pid", bl.PidPacing, budget=10, value_per_conversion=5)
    >>> config = bl.SimConfig(
    ...     bidders=[pid], auction=bl.FirstPrice(), episodes=1, steps=200
    ... )
    >>> log = bl.run(config)
    >>> round(float(log.price[0, :100].sum()), 1), round(float(log.price[0, 100:].sum()), 1)
    (5.5, 4.5)
    """

    def __init__(self, params: dict[str, float], rng: np.random.Generator):
        super().__init__(params, rng)
        self.initial_multiplier = params.get("initial_multiplier", 1.0)
        self.kp = params.get("kp", 20.0)
        self.ki = params.get("ki", 2.0)
        self.kd = params.get("kd", 0.0)
        self.min_multiplier = params.get("min_multiplier", 0.0)
        self.max_multiplier = params.get("max_multiplier", 10.0)

    @override
    def on_episode_start(self, budget: float, steps: int) -> None:
        self.budget = budget
        self.steps = steps
        self.steps_done = 0
        self.spent = 0.0
        self.integral = 0.0
        self.previous_error = 0.0
        self.multiplier = self.initial_multiplier

    @override
    def bid(self, observation: Observation) -> float:
        return self.multiplier * observation.value

    @override
    def update(self, feedback: Feedback) -> None:
        self.steps_done += 1
        self.spent += feedback.price
        planned = self.budget * self.steps_done / self.steps
        # Dividing by budget makes the gains independent of budget size.
        error = (planned - self.spent) / self.budget
        # A per-step sum winds up over long episodes; the mean does not.
        self.integral += (error - self.integral) / self.steps_done
        derivative = error - self.previous_error
        self.previous_error = error
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.multiplier = float(
            np.clip(
                self.initial_multiplier * np.exp(control),
                self.min_multiplier,
                self.max_multiplier,
            )
        )
