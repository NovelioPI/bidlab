from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from auction_sim.auctions import run_auction
from auction_sim.bidders import Bidder, resolve_bidder
from auction_sim.config import SimConfig
from auction_sim.rewards import (
    EpisodeReward,
    StepReward,
    resolve_episode_reward,
    resolve_step_reward,
)
from auction_sim.types import EpisodeStats, Feedback, Observation
from auction_sim.world import World

NO_WINNER = -1


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
    return _Simulation(config).run()


class _Simulation:
    def __init__(self, config: SimConfig):
        count = len(config.bidders)
        # Reason: one stream per role, so adding a bidder keeps world draws equal.
        world_seed, tie_seed, conversion_seed, *bidder_seeds = np.random.SeedSequence(
            config.seed
        ).spawn(3 + count)
        self.config = config
        self.weights = np.array([s.value_per_conversion for s in config.bidders])
        self.world = World(
            config.world, self.weights, np.random.default_rng(world_seed)
        )
        self.tie_rng = np.random.default_rng(tie_seed)
        self.conversion_rng = np.random.default_rng(conversion_seed)
        self.participants = [
            _Participant(
                bidder=resolve_bidder(spec.type)(
                    dict(spec.params), np.random.default_rng(seed)
                ),
                budget=spec.budget,
                value_per_conversion=spec.value_per_conversion,
                step_reward=resolve_step_reward(spec.reward),
                episode_reward=(
                    resolve_episode_reward(spec.episode_reward)
                    if spec.episode_reward
                    else None
                ),
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

    def run(self) -> RunLog:
        for episode in range(self.config.episodes):
            self._run_episode(episode)
        return self.log

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
        probabilities, values = self.world.step()
        observations = [
            Observation(
                step=step,
                steps_left=self.config.steps - step,
                value=float(values[i]),
                value_per_conversion=p.value_per_conversion,
                budget_left=float(budget_left[i]),
            )
            for i, p in enumerate(self.participants)
        ]
        bids = np.array(
            [
                p.bidder.bid(obs)
                for p, obs in zip(self.participants, observations, strict=True)
            ],
            dtype=float,
        )
        # Risk: if bids are not capped at the remaining budget, then spend can exceed it.
        bids = np.clip(
            np.nan_to_num(bids, nan=0.0, posinf=0.0, neginf=0.0), 0.0, budget_left
        )
        auction = self.config.auction
        winner, price = run_auction(bids, auction.kind, auction.reserve, self.tie_rng)
        # Reason: one draw every step keeps the conversion stream aligned across runs.
        draw = self.conversion_rng.random()
        converted = winner != NO_WINNER and draw < probabilities[winner]
        if winner != NO_WINNER:
            budget_left[winner] -= price

        for i, (participant, obs) in enumerate(
            zip(self.participants, observations, strict=True)
        ):
            won = i == winner
            feedback = Feedback(
                bid=float(bids[i]),
                won=won,
                price=price if won else 0.0,
                converted=converted and won,
            )
            reward = participant.step_reward(obs, feedback, participant.reward_params)
            participant.bidder.update(replace(feedback, reward=reward))
            self.log.rewards[episode, step, i] = reward

        self.log.bids[episode, step] = bids
        self.log.values[episode, step] = values
        self.log.winner[episode, step] = winner
        self.log.price[episode, step] = price
        self.log.converted[episode, step] = converted

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
