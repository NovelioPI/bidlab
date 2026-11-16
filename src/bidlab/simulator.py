from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from bidlab.auctions import NO_WINNER
from bidlab.bidders import Bidder
from bidlab.config import BidderSpec, SimConfig
from bidlab.markets import MarketDraw
from bidlab.rewards import EpisodeReward, StepReward
from bidlab.types import EpisodeStats, Feedback, Observation


@dataclass(frozen=True)
class RunLog:
    """Arrays are (episodes, steps[, bidders]); episode_rewards is (episodes, bidders)."""

    config: SimConfig
    bids: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    winner: npt.NDArray[np.int64]
    price: npt.NDArray[np.float64]
    converted: npt.NDArray[np.bool_]
    rewards: npt.NDArray[np.float64]
    episode_rewards: npt.NDArray[np.float64]


@dataclass(frozen=True)
class _Participant:
    bidder: Bidder
    budget: float
    value_per_conversion: float
    step_reward: StepReward
    episode_reward: EpisodeReward | None
    reward_params: dict[str, float]


def run(config: SimConfig) -> RunLog:
    simulation = _Simulation(config)
    simulation.run_all()
    return simulation.log


def _build_bidder(spec: BidderSpec, rng: np.random.Generator) -> Bidder:
    bidder = spec.bidder(dict(spec.params), rng)
    if not isinstance(bidder, Bidder):
        raise TypeError(
            f"bidder {spec.name!r}: the factory returned "
            f"{type(bidder).__name__}, not a Bidder"
        )
    return bidder


class _Simulation:
    def __init__(self, config: SimConfig):
        count = len(config.bidders)
        # One stream per role, so adding a bidder keeps the market's draws equal.
        world_seed, tie_seed, conversion_seed, *bidder_seeds = np.random.SeedSequence(
            config.seed
        ).spawn(3 + count)
        self.config = config
        self.weights = np.array([s.value_per_conversion for s in config.bidders])
        self.market_draws = config.market.draws(
            count, np.random.default_rng(world_seed)
        )
        self.tie_rng = np.random.default_rng(tie_seed)
        self.conversion_rng = np.random.default_rng(conversion_seed)
        self.participants = [
            _Participant(
                bidder=_build_bidder(spec, np.random.default_rng(seed)),
                budget=spec.budget,
                value_per_conversion=spec.value_per_conversion,
                step_reward=spec.reward,
                episode_reward=spec.episode_reward,
                reward_params=spec.merged_reward_params(),
            )
            for spec, seed in zip(config.bidders, bidder_seeds, strict=True)
        ]
        shape = (config.episodes, config.steps)
        self.log = RunLog(
            config=config,
            bids=np.zeros((*shape, count)),
            values=np.zeros((*shape, count)),
            winner=np.full(shape, NO_WINNER),
            price=np.zeros(shape),
            converted=np.zeros(shape, dtype=bool),
            rewards=np.zeros((*shape, count)),
            episode_rewards=np.full((config.episodes, count), np.nan),
        )

    def run_all(self) -> None:
        for episode in range(self.config.episodes):
            self._run_episode(episode)
        # Callers must not change the log, or later metrics silently disagree.
        for array in (
            self.log.bids,
            self.log.values,
            self.log.winner,
            self.log.price,
            self.log.converted,
            self.log.rewards,
            self.log.episode_rewards,
        ):
            array.flags.writeable = False

    def _run_episode(self, episode: int) -> None:
        steps = self.config.steps
        budget_left = np.array([p.budget for p in self.participants])
        for participant in self.participants:
            participant.bidder.on_episode_start(participant.budget, steps)
        for step in range(steps):
            self._run_step(episode, step, budget_left)
        self._end_episode(episode)

    def _run_step(
        self, episode: int, step: int, budget_left: npt.NDArray[np.float64]
    ) -> None:
        draw = self._next_draw(episode, step)
        probabilities = draw.probabilities
        values = self.weights * probabilities
        observations = [
            Observation(
                step=step,
                steps_left=self.config.steps - step,
                value=float(values[i]),
                value_per_conversion=p.value_per_conversion,
                budget_left=float(budget_left[i]),
                features=draw.features,
            )
            for i, p in enumerate(self.participants)
        ]
        bids = np.array(
            [
                p.bidder.bid(observation)
                for p, observation in zip(self.participants, observations, strict=True)
            ],
            dtype=float,
        )
        # Risk: if bids are not capped at the remaining budget, then spend can exceed it.
        bids = np.clip(
            np.nan_to_num(bids, nan=0.0, posinf=0.0, neginf=0.0), 0.0, budget_left
        )
        winner, price = self.config.auction.run(bids, self.tie_rng)
        # One draw every step keeps the conversion stream aligned across runs.
        conversion_draw = self.conversion_rng.random()
        converted = winner != NO_WINNER and conversion_draw < probabilities[winner]
        if winner != NO_WINNER:
            budget_left[winner] -= price

        for i, (participant, observation) in enumerate(
            zip(self.participants, observations, strict=True)
        ):
            won = i == winner
            feedback = Feedback(
                bid=float(bids[i]),
                won=won,
                price=price if won else 0.0,
                converted=converted and won,
            )
            reward = participant.step_reward(
                observation, feedback, participant.reward_params
            )
            participant.bidder.update(replace(feedback, reward=reward))
            self.log.rewards[episode, step, i] = reward

        self.log.bids[episode, step] = bids
        self.log.values[episode, step] = values
        self.log.winner[episode, step] = winner
        self.log.price[episode, step] = price
        self.log.converted[episode, step] = converted

    def _next_draw(self, episode: int, step: int) -> MarketDraw:
        market = type(self.config.market).__name__
        where = f"{market} at episode {episode}, step {step}"
        try:
            draw = next(self.market_draws)
        except StopIteration:
            raise RuntimeError(f"{where}: the market stopped yielding draws") from None

        probabilities = draw.probabilities
        if probabilities.shape != (len(self.participants),):
            raise ValueError(
                f"{where}: got probabilities of shape {probabilities.shape}, "
                f"expected ({len(self.participants)},)"
            )
        # The negated check also catches NaN, which fails every comparison.
        if not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):
            raise ValueError(f"{where}: probabilities must lie in [0, 1]")
        return draw

    def _end_episode(self, episode: int) -> None:
        winner = self.log.winner[episode]
        for i, participant in enumerate(self.participants):
            won = winner == i
            conversions = int(np.sum(self.log.converted[episode] & won))
            stats = EpisodeStats(
                budget=participant.budget,
                spend=float(np.sum(self.log.price[episode][won])),
                wins=int(np.sum(won)),
                conversions=conversions,
                value=conversions * participant.value_per_conversion,
            )
            episode_reward = None
            if participant.episode_reward is not None:
                episode_reward = participant.episode_reward(
                    stats, participant.reward_params
                )
                self.log.episode_rewards[episode, i] = episode_reward
            participant.bidder.on_episode_end(stats, episode_reward)
