"""The core loop: `run` simulates every auction and records a `RunLog`."""

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from scipy.special import expit, logit

from bidlab.auctions import NO_WINNER
from bidlab.bidders import Bidder
from bidlab.config import BidderSpec, SimConfig
from bidlab.markets import MarketDraw
from bidlab.rewards import EpisodeReward, StepReward
from bidlab.types import SECONDS_PER_HOUR, Campaign, EpisodeStats, Feedback, Observation


@dataclass(frozen=True)
class RunLog:
    """Arrays are (episodes, steps[, bidders]); episode_rewards is (episodes, bidders).

    Every array is read-only, so metrics computed from it always agree.

    Attributes
    ----------
    config : SimConfig
        The config that made this log.
    bids : numpy.ndarray
        Bids after cleaning and capping; 0 for a bidder outside its segments.
    values : numpy.ndarray
        Each bidder's true expected value per step: `Campaign.value` × the
        true conversion probability.
    winner : numpy.ndarray
        Index of the winning bidder per step; -1 when nobody won.
    price : numpy.ndarray
        Price the winner paid per step; 0 when nobody won.
    converted : numpy.ndarray
        Whether the winner converted per step.
    rewards : numpy.ndarray
        Each bidder's step reward; 0 outside its segments.
    episode_rewards : numpy.ndarray
        Each bidder's episode reward; NaN when the spec sets none.
    segment : numpy.ndarray
        Index into ``config.market.segment_names`` per step.

    Examples
    --------
    >>> import bidlab as bl
    >>> spec = bl.BidderSpec("t", bl.Truthful(), bl.Campaign(budget=10, value=5))
    >>> log = bl.run(bl.SimConfig(bidders=[spec], episodes=2, steps=50))
    >>> won = log.winner[0] == 0
    >>> bool(log.price[0][won].sum() <= spec.campaign.budget)
    True
    >>> log.bids.flags.writeable
    False
    """

    config: SimConfig
    bids: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    winner: npt.NDArray[np.int64]
    price: npt.NDArray[np.float64]
    converted: npt.NDArray[np.bool_]
    rewards: npt.NDArray[np.float64]
    episode_rewards: npt.NDArray[np.float64]
    segment: npt.NDArray[np.int64]


@dataclass(frozen=True)
class _Participant:
    bidder: Bidder
    campaign: Campaign
    step_reward: StepReward
    episode_reward: EpisodeReward | None
    reward_params: dict[str, float]

    def takes_part_in(self, segment: str) -> bool:
        segments = self.campaign.segments
        return segments is None or segment in segments


def run(config: SimConfig) -> RunLog:
    """Simulate every episode of ``config`` and return the full log.

    On one OS and CPU type, the same config and seed give a bit-identical
    log. Each bidder sees only its own observation and feedback.

    Raises
    ------
    TypeError
        If a bidder cannot be copied for a fresh run.
    ValueError
        If the market yields a draw with the wrong shape, a probability
        outside [0, 1], or an unknown segment.
    RuntimeError
        If the market stops yielding draws before the run ends.

    Examples
    --------
    >>> import bidlab as bl
    >>> config = bl.SimConfig(
    ...     bidders=[bl.BidderSpec("t", bl.Truthful(), bl.Campaign(budget=10, value=5))],
    ...     episodes=2,
    ...     steps=50,
    ... )
    >>> log = bl.run(config)
    >>> log.bids.shape, log.winner.shape
    ((2, 50, 1), (2, 50))
    >>> bool((bl.run(config).bids == log.bids).all())
    True
    """
    simulation = _Simulation(config)
    simulation.run_all()
    return simulation.log


class _Simulation:
    def __init__(self, config: SimConfig):
        count = len(config.bidders)
        # One stream per role, so adding a bidder keeps the market's draws equal.
        market_seed, tie_seed, conversion_seed, *bidder_seeds = np.random.SeedSequence(
            config.seed
        ).spawn(3 + count)
        self.config = config
        self.weights = np.array([spec.campaign.value for spec in config.bidders])
        self.seconds_per_step = config.horizon_hours * SECONDS_PER_HOUR / config.steps
        self.market_draws = config.market.draws(
            count, np.random.default_rng(market_seed)
        )
        self.tie_rng = np.random.default_rng(tie_seed)
        self.conversion_rng = np.random.default_rng(conversion_seed)
        self.estimate_seeds: list[np.random.SeedSequence] = []
        self.participants = [
            self._participant(spec, seed)
            for spec, seed in zip(config.bidders, bidder_seeds, strict=True)
        ]
        self.estimate_rngs = [
            np.random.default_rng(seed) for seed in self.estimate_seeds
        ]
        self.segment_index = {
            name: index for index, name in enumerate(config.market.segment_names)
        }
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
            segment=np.zeros(shape, dtype=np.int64),
        )

    def _participant(
        self, spec: BidderSpec, seed: np.random.SeedSequence
    ) -> _Participant:
        # Risk: if this child is spawned after the bidder is built, then a stack's
        # stages take it and the platform-estimate stream shifts.
        self.estimate_seeds.append(seed.spawn(1)[0])
        campaign = replace(
            spec.campaign,
            horizon_hours=self.config.horizon_hours,
            start_hour=self.config.start_hour,
        )
        return _Participant(
            bidder=spec.build(np.random.default_rng(seed)),
            campaign=campaign,
            step_reward=spec.reward,
            episode_reward=spec.episode_reward,
            reward_params=spec.merged_reward_params(),
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
            self.log.segment,
        ):
            array.flags.writeable = False

    def _run_episode(self, episode: int) -> None:
        budget_left = np.array([p.campaign.budget for p in self.participants])
        for participant in self.participants:
            participant.bidder.on_episode_start(participant.campaign)

        for step in range(self.config.steps):
            self._run_step(episode, step, budget_left)

        self._end_episode(episode)

    def _run_step(
        self, episode: int, step: int, budget_left: npt.NDArray[np.float64]
    ) -> None:
        draw = self._next_draw(episode, step)
        probabilities = draw.probabilities
        values = self.weights * probabilities
        observed = self._observed_values(values, probabilities)
        time = step * self.seconds_per_step
        # Bidders outside their segments are not asked to bid, so they cannot learn from it.
        observations = {
            i: Observation(time=time, value=float(observed[i]), features=draw.features)
            for i, p in enumerate(self.participants)
            if p.takes_part_in(draw.segment)
        }
        bids = np.zeros(len(self.participants))
        for i, observation in observations.items():
            bidder = self.participants[i].bidder
            bids[i] = bidder.forward(observation.value, observation)
        # Risk: if bids are not capped at the remaining budget, then spend can exceed it.

        bids = np.clip(
            np.nan_to_num(bids, nan=0.0, posinf=0.0, neginf=0.0), 0.0, budget_left
        )
        # A logged bid of 0 would win at reserve 0, so absent bidders bid -inf.
        auction_bids = np.full(len(self.participants), -np.inf)
        active = list(observations)
        auction_bids[active] = bids[active]
        winner, price = self.config.auction.run(auction_bids, self.tie_rng)
        # One draw every step keeps the conversion stream aligned across runs.
        conversion_draw = self.conversion_rng.random()
        converted = winner != NO_WINNER and conversion_draw < probabilities[winner]
        if winner != NO_WINNER:
            budget_left[winner] -= price

        for i, observation in observations.items():
            participant = self.participants[i]
            won = i == winner
            feedback = Feedback(
                time=time,
                bid=float(bids[i]),
                won=won,
                price=price if won else 0.0,
                converted=converted and won,
                value=participant.campaign.value if converted and won else 0.0,
            )
            reward = participant.step_reward(
                observation, feedback, participant.reward_params
            )
            participant.bidder.update_batch([replace(feedback, reward=reward)])
            self.log.rewards[episode, step, i] = reward

        self.log.bids[episode, step] = bids
        self.log.values[episode, step] = values
        self.log.winner[episode, step] = winner
        self.log.price[episode, step] = price
        self.log.converted[episode, step] = converted
        self.log.segment[episode, step] = self.segment_index[draw.segment]

    def _observed_values(
        self,
        values: npt.NDArray[np.float64],
        probabilities: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        noise = self.config.estimate_noise
        # Without noise, the true values pass through untouched, bit for bit.
        if noise == 0:
            return values

        # Every bidder draws every step, taking part or not, so no stream shifts.
        errors = np.array([rng.standard_normal() for rng in self.estimate_rngs])
        # logit(0) and logit(1) are infinite, so p = 0 and p = 1 stay exact.
        estimates = expit(logit(probabilities) + noise * errors)
        return self.weights * estimates

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

        if draw.segment not in self.segment_index:
            raise ValueError(
                f"{where}: segment {draw.segment!r} is not in "
                f"segment_names {tuple(self.segment_index)}"
            )

        return draw

    def _end_episode(self, episode: int) -> None:
        winner = self.log.winner[episode]
        for i, participant in enumerate(self.participants):
            won = winner == i
            conversions = int(np.sum(self.log.converted[episode] & won))
            stats = EpisodeStats(
                budget=participant.campaign.budget,
                spend=float(np.sum(self.log.price[episode][won])),
                wins=int(np.sum(won)),
                conversions=conversions,
                value=conversions * participant.campaign.value,
            )
            episode_reward = None
            if participant.episode_reward is not None:
                episode_reward = participant.episode_reward(
                    stats, participant.reward_params
                )
                self.log.episode_rewards[episode, i] = episode_reward

            participant.bidder.on_episode_end(stats, episode_reward)
