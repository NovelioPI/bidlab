from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class WorldConfig:
    context_dim: int = 8
    bias_mean: float = -3.0
    bias_std: float = 0.5


class World:
    """Draws each step's conversion probabilities and values for all bidders."""

    def __init__(
        self,
        config: WorldConfig,
        value_per_conversion: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ):
        bidders = value_per_conversion.size
        dim = config.context_dim
        # Reason: separate streams keep bidder i's draws equal when bidders are added.
        vector_rng, bias_rng, self._rng = rng.spawn(3)
        self._value_per_conversion = value_per_conversion
        self._ad_vectors = vector_rng.normal(
            0.0, 1.0 / np.sqrt(dim), size=(bidders, dim)
        )
        self._biases = bias_rng.normal(config.bias_mean, config.bias_std, size=bidders)

    def step(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (conversion probabilities, values), one entry per bidder."""
        context = self._rng.normal(size=self._ad_vectors.shape[1])
        logits = self._ad_vectors @ context + self._biases
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return probabilities, self._value_per_conversion * probabilities
