"""The core loop: `run` simulates every auction and records a `RunLog`."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
from scipy.special import expit, logit

from bidlab.auctions import NO_WINNER
from bidlab.bidders import Bidder
from bidlab.bidders.base import attach_logger, call_forward
from bidlab.callbacks import Callback, checked_callbacks
from bidlab.config import Campaign, SimConfig
from bidlab.logs import LogBook, Logs
from bidlab.markets import MarketDraw
from bidlab.rewards import EpisodeReward, StepReward
from bidlab.types import (
    HOURS_PER_DAY,
    SECONDS_PER_HOUR,
    BidRequest,
    BidResult,
    Brief,
    EpisodeReport,
)

BUILT_IN_INPUTS = frozenset(
    {"value", "time", "time_left", "horizon", "hour_of_day", "budget_left", "segment"}
)


@dataclass(frozen=True)
class RunLog:
    """Arrays are (episodes, steps[, campaigns]); episode_rewards is (episodes, campaigns).

    Every array is read-only, so metrics computed from it always agree.

    Attributes
    ----------
    config : SimConfig
        The config that made this log.
    bids : numpy.ndarray
        Bids after cleaning and capping; 0 for a bidder outside its segments.
    true_values : numpy.ndarray
        Each bidder's true expected value per step: `Campaign.event_value` × the
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
        Each bidder's episode reward; NaN when the campaign sets none.
    segment : numpy.ndarray
        Index into ``config.market.segment_names`` per step.
    logged : Mapping[str, Mapping[str, numpy.ndarray]]
        Values bidders saved with `Bidder.log`: stored name, then campaign name,
        to an (episodes, steps) array, or (episodes,) for values logged in an
        episode hook. NaN where nothing was logged.

    Examples
    --------
    >>> import bidlab as bl
    >>> campaign = bl.Campaign("t", bl.Linear(), budget=10, event_value=5)
    >>> log = bl.run(bl.SimConfig(campaigns=[campaign], episodes=2, steps=50))
    >>> won = log.winner[0] == 0
    >>> bool(log.price[0][won].sum() <= campaign.budget)
    True
    >>> log.bids.flags.writeable
    False
    """

    config: SimConfig
    bids: npt.NDArray[np.float64]
    true_values: npt.NDArray[np.float64]
    winner: npt.NDArray[np.int64]
    price: npt.NDArray[np.float64]
    converted: npt.NDArray[np.bool_]
    rewards: npt.NDArray[np.float64]
    episode_rewards: npt.NDArray[np.float64]
    segment: npt.NDArray[np.int64]
    logged: Logs = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class _Participant:
    name: str
    bidder: Bidder
    nodes: tuple[Bidder, ...]
    brief: Brief
    step_reward: StepReward
    episode_reward: EpisodeReward | None
    reward_params: dict[str, float]

    def takes_part_in(self, segment: str) -> bool:
        segments = self.brief.segments
        return segments is None or segment in segments

    def check_tree(self) -> None:
        # Risk: if a late child were bound here, then its stream would depend on timing.
        known = {id(node) for node in self.nodes}
        for path, node in self.bidder.named_bidders():
            if id(node) not in known:
                raise ValueError(
                    f"campaign {self.name!r}: child bidder at {path!r} appeared after "
                    "the run started; create child bidders in __init__ and reset "
                    "their state in on_episode_start"
                )


def run(config: SimConfig, callbacks: Sequence[Callback] = ()) -> RunLog:
    """Simulate every episode of ``config`` and return the full log.

    On one OS and CPU type, the same config and seed give a bit-identical
    log. Each bidder sees only its own request and result.

    Parameters
    ----------
    config : SimConfig
        The simulation.
    callbacks : Sequence[Callback], optional
        Read-only hooks called at run start, at each episode end and at run
        end, in this order.

    Raises
    ------
    TypeError
        If a bidder cannot be copied for a fresh run, or a callback is not a
        `Callback` instance.
    ValueError
        If the market yields a draw with the wrong shape, a probability
        outside [0, 1], or an unknown segment.
    RuntimeError
        If the market stops yielding draws before the run ends.

    Examples
    --------
    >>> import bidlab as bl
    >>> config = bl.SimConfig(
    ...     campaigns=[bl.Campaign("t", bl.Linear(), budget=10, event_value=5)],
    ...     episodes=2,
    ...     steps=50,
    ... )
    >>> log = bl.run(config)
    >>> log.bids.shape, log.winner.shape
    ((2, 50, 1), (2, 50))
    >>> bool((bl.run(config).bids == log.bids).all())
    True
    """
    checked = checked_callbacks(callbacks)
    simulation = _Simulation(config, checked)
    for callback in checked:
        callback.on_run_start(config)
    simulation.run_all()
    for callback in checked:
        callback.on_run_end(simulation.log)
    return simulation.log


class _Simulation:
    def __init__(self, config: SimConfig, callbacks: tuple[Callback, ...]):
        count = len(config.campaigns)
        # One stream per role, so adding a bidder keeps the market's draws equal.
        market_seed, tie_seed, conversion_seed, *bidder_seeds = np.random.SeedSequence(
            config.seed
        ).spawn(3 + count)
        self.config = config
        self.callbacks = callbacks
        self.event_values = np.array(
            [campaign.event_value for campaign in config.campaigns]
        )
        self.horizon = config.horizon_hours * SECONDS_PER_HOUR
        self.seconds_per_step = self.horizon / config.steps
        self.market_draws = config.market.draws(
            count, np.random.default_rng(market_seed)
        )
        self.tie_rng = np.random.default_rng(tie_seed)
        self.conversion_rng = np.random.default_rng(conversion_seed)
        self.estimate_seeds: list[np.random.SeedSequence] = []
        self.book = LogBook(config.episodes, config.steps)
        self.participants = [
            self._participant(campaign, seed)
            for campaign, seed in zip(config.campaigns, bidder_seeds, strict=True)
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
            true_values=np.zeros((*shape, count)),
            winner=np.full(shape, NO_WINNER),
            price=np.zeros(shape),
            converted=np.zeros(shape, dtype=bool),
            rewards=np.zeros((*shape, count)),
            episode_rewards=np.full((config.episodes, count), np.nan),
            segment=np.zeros(shape, dtype=np.int64),
        )

    def _participant(
        self, campaign: Campaign, seed: np.random.SeedSequence
    ) -> _Participant:
        # Risk: if this child is spawned after the bidder is bound, then a stack's
        # inner bidders take it and the platform-estimate stream shifts.
        self.estimate_seeds.append(seed.spawn(1)[0])
        brief = campaign.brief(self.config.horizon_hours, self.config.start_hour)
        bidder = campaign.build(np.random.default_rng(seed))
        nodes = tuple(bidder.named_bidders())
        for path, node in nodes:
            attach_logger(node, self.book.logger(campaign.name, path))
        return _Participant(
            name=campaign.name,
            bidder=bidder,
            nodes=tuple(node for _, node in nodes),
            brief=brief,
            step_reward=campaign.step_reward,
            episode_reward=campaign.episode_reward,
            reward_params=campaign.merged_reward_params(),
        )

    def run_all(self) -> None:
        for episode in range(self.config.episodes):
            self._run_episode(episode)
        self.book.close()
        self.log = replace(self.log, logged=self.book.result())
        # Callers must not change the log, or later metrics silently disagree.

        for array in (
            self.log.bids,
            self.log.true_values,
            self.log.winner,
            self.log.price,
            self.log.converted,
            self.log.rewards,
            self.log.episode_rewards,
            self.log.segment,
        ):
            array.flags.writeable = False

    def _run_episode(self, episode: int) -> None:
        budget_left = np.array([p.brief.budget for p in self.participants])
        self.book.at_episode(episode)
        for participant in self.participants:
            for node in participant.nodes:
                node.on_episode_start(participant.brief)
            participant.check_tree()

        for step in range(self.config.steps):
            self._run_step(episode, step, budget_left)

        self._end_episode(episode)

    def _run_step(
        self, episode: int, step: int, budget_left: npt.NDArray[np.float64]
    ) -> None:
        draw = self._next_draw(episode, step)
        self.book.at_step(episode, step)
        probabilities = draw.probabilities
        values = self.event_values * probabilities
        estimated = self._estimated_values(values, probabilities)
        time = step * self.seconds_per_step
        # Bidders outside their segments are not asked to bid, so they cannot learn from it.
        requests = {
            i: BidRequest(time=time, value=float(estimated[i]), features=draw.features)
            for i, p in enumerate(self.participants)
            if p.takes_part_in(draw.segment)
        }
        shared = {
            **draw.features,
            "time": time,
            "time_left": self.horizon - time,
            "horizon": self.horizon,
            "hour_of_day": (self.config.start_hour + time / SECONDS_PER_HOUR)
            % HOURS_PER_DAY,
            "segment": draw.segment,
        }
        bids = np.zeros(len(self.participants))
        for i, request in requests.items():
            inputs = {
                **shared,
                "value": request.value,
                "budget_left": float(budget_left[i]),
            }
            bids[i] = call_forward(self.participants[i].bidder, inputs)
        # Risk: if bids are not capped at the remaining budget, then spend can exceed it.

        bids = np.clip(
            np.nan_to_num(bids, nan=0.0, posinf=0.0, neginf=0.0), 0.0, budget_left
        )
        # A logged bid of 0 would win at reserve 0, so absent bidders bid -inf.
        auction_bids = np.full(len(self.participants), -np.inf)
        active = list(requests)
        auction_bids[active] = bids[active]
        winner, price = self.config.auction.run(auction_bids, self.tie_rng)
        # One draw every step keeps the conversion stream aligned across runs.
        conversion_draw = self.conversion_rng.random()
        converted = winner != NO_WINNER and conversion_draw < probabilities[winner]
        if winner != NO_WINNER:
            budget_left[winner] -= price

        for i, request in requests.items():
            participant = self.participants[i]
            won = i == winner
            result = BidResult(
                time=time,
                bid=float(bids[i]),
                won=won,
                price=price if won else 0.0,
                converted=converted and won,
                realized_value=participant.brief.event_value
                if converted and won
                else 0.0,
            )
            reward = participant.step_reward(request, result, participant.reward_params)
            batch = [replace(result, reward=reward)]
            for node in participant.nodes:
                node.update_batch(batch)
            self.log.rewards[episode, step, i] = reward

        self.log.bids[episode, step] = bids
        self.log.true_values[episode, step] = values
        self.log.winner[episode, step] = winner
        self.log.price[episode, step] = price
        self.log.converted[episode, step] = converted
        self.log.segment[episode, step] = self.segment_index[draw.segment]

    def _estimated_values(
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
        return self.event_values * estimates

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

        clashes = draw.features.keys() & BUILT_IN_INPUTS
        if clashes:
            raise ValueError(
                f"{where}: features {sorted(clashes)} have the names of built-in "
                "inputs; rename them"
            )

        if draw.segment not in self.segment_index:
            raise ValueError(
                f"{where}: segment {draw.segment!r} is not in "
                f"segment_names {tuple(self.segment_index)}"
            )

        return draw

    def _end_episode(self, episode: int) -> None:
        self.book.at_episode(episode)
        winner = self.log.winner[episode]
        reports: dict[str, EpisodeReport] = {}
        for i, participant in enumerate(self.participants):
            won = winner == i
            conversions = int(np.sum(self.log.converted[episode] & won))
            report = EpisodeReport(
                budget=participant.brief.budget,
                spend=float(np.sum(self.log.price[episode][won])),
                wins=int(np.sum(won)),
                conversions=conversions,
                realized_value=conversions * participant.brief.event_value,
            )
            episode_reward = None
            if participant.episode_reward is not None:
                episode_reward = participant.episode_reward(
                    report, participant.reward_params
                )
                self.log.episode_rewards[episode, i] = episode_reward

            for node in participant.nodes:
                node.on_episode_end(report, episode_reward)
            participant.check_tree()
            reports[participant.name] = report

        read_only = MappingProxyType(reports)
        for callback in self.callbacks:
            callback.on_episode_end(episode, read_only)
