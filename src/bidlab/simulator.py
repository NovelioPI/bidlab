"""The core loop: `run` simulates every auction and records a `RunLog`."""

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from bidlab.auctions import NO_WINNER
from bidlab.bidders import Bidder, Stack
from bidlab.bidders.base import attach_logger, call_forward, takes_input
from bidlab.callbacks import Callback, checked_callbacks, with_progress_bar
from bidlab.config import Campaign, SimConfig
from bidlab.logs import LogBook, Logs
from bidlab.markets import MarketDraw
from bidlab.types import (
    HOURS_PER_DAY,
    SECONDS_PER_HOUR,
    BidResult,
    Conversion,
    EpisodeReport,
    Event,
    EventName,
    MarketTruth,
)

BUILT_IN_INPUTS = frozenset(
    {
        "time",
        "time_left",
        "horizon",
        "hour_of_day",
        "budget_left",
        "segment",
        "market",
        "features",
        "floor",
    }
)
# A Stack passes each bidder's output on as "value", so no feature may take it.
RESERVED_NAMES = BUILT_IN_INPUTS | {"value"}


@dataclass(frozen=True)
class RunLog:
    """Arrays are (episodes, steps[, campaigns]).

    Every array is read-only, so metrics computed from it always agree.

    Attributes
    ----------
    config : SimConfig
        The config that made this log.
    bids : numpy.ndarray
        Bids after cleaning and capping; 0 for a bidder outside its segments.
    true_values : numpy.ndarray
        Each bidder's true expected value per step: `Campaign.base_value` ×
        the true rate of its event: 1 for an impression, pCTR for a click,
        pCTR × pCVR for a conversion, with pCTR 1 when the market has no click
        stage.
    winner : numpy.ndarray
        Index of the winning bidder per step; -1 when nobody won.
    price : numpy.ndarray
        Price the winner paid per step; 0 when nobody won.
    clicked : numpy.ndarray
        Whether the winner was clicked per step; every win is a click when the
        market has no click stage. False when nobody won.
    converted : numpy.ndarray
        Whether the winner converted per step; only a click can convert.
    segment : numpy.ndarray
        Index into ``config.market.segment_names`` per step.
    market_price : numpy.ndarray
        The market's price per step, the highest bid from outside the config;
        NaN where the market drew none.
    floor : numpy.ndarray
        The market's floor per step; 0 where it drew none. The auction's
        ``reserve`` is not included.
    logged : Mapping[str, Mapping[str, numpy.ndarray]]
        Values bidders saved with `Bidder.log`: stored name, then campaign name,
        to an (episodes, steps) array, or (episodes,) for values logged in an
        episode hook. NaN where nothing was logged.

    Examples
    --------
    >>> import bidlab as bl
    >>> campaign = bl.Campaign(
    ...     "t", bl.Stack(bl.OracleValue(), bl.Linear()), budget=10, base_value=5
    ... )
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
    clicked: npt.NDArray[np.bool_]
    converted: npt.NDArray[np.bool_]
    segment: npt.NDArray[np.int64]
    market_price: npt.NDArray[np.float64]
    floor: npt.NDArray[np.float64]
    logged: Logs = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class _Participant:
    # Its bidder is the running copy, so a bidder that reaches it finds itself.
    campaign: Campaign
    nodes: tuple[Bidder, ...]
    # Calling the empty update hooks of every node cost about 5% of run time.
    learners: tuple[Bidder, ...]

    def takes_part_in(self, segment: str) -> bool:
        segments = self.campaign.segments
        return segments is None or segment in segments

    def check_tree(self) -> None:
        # Risk: if a late child were bound here, then its stream would depend on timing.
        known = {id(node) for node in self.nodes}
        for path, node in self.campaign.bidder.named_bidders():
            if id(node) not in known:
                raise ValueError(
                    f"campaign {self.campaign.name!r}: child bidder at {path!r} appeared after "
                    "the run started; create child bidders in __init__ and reset "
                    "their state in on_episode_start"
                )


def run(
    config: SimConfig, callbacks: Sequence[Callback] = (), progress_bar: bool = True
) -> RunLog:
    """Simulate every episode of ``config`` and return the full log.

    On one OS and CPU type, the same config and seed give a bit-identical
    log. Each bidder sees only its own inputs and results.

    Parameters
    ----------
    config : SimConfig
        The simulation.
    callbacks : Sequence[Callback], optional
        Read-only hooks called at run start, at each step end, at each
        episode end and at run end, in this order.
    progress_bar : bool, optional
        Show a `ProgressBar` on stderr, after the other callbacks.

    Raises
    ------
    TypeError
        If a bidder cannot be copied for a fresh run, or a callback is not a
        `Callback` instance.
    ValueError
        If the market yields a draw with the wrong shape, a probability
        outside [0, 1], a market price or floor that is negative, NaN or
        infinite, a feature named like a built-in input, or an unknown
        segment.
    RuntimeError
        If the market stops yielding draws before the run ends.

    Examples
    --------
    >>> import bidlab as bl
    >>> bidder = bl.Stack(bl.OracleValue(), bl.Linear())
    >>> config = bl.SimConfig(
    ...     campaigns=[bl.Campaign("t", bidder, budget=10, base_value=5)],
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
    if progress_bar:
        checked = with_progress_bar(checked)
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
        self.base_values = np.array(
            [campaign.base_value for campaign in config.campaigns]
        )
        events = [campaign.event for campaign in config.campaigns]
        self.values_click = np.array([event != Event.IMPRESSION for event in events])
        self.values_conversion = np.array(
            [event == Event.CONVERSION for event in events]
        )
        self.only_conversions = bool(self.values_conversion.all())
        self.horizon = config.horizon_hours * SECONDS_PER_HOUR
        self.seconds_per_step = self.horizon / config.steps
        self.market_draws = config.market.draws(
            count, np.random.default_rng(market_seed)
        )
        self.tie_rng = np.random.default_rng(tie_seed)
        self.conversion_rng = np.random.default_rng(conversion_seed)
        # A child stream, so markets without a click stage keep every draw.
        self.click_rng = np.random.default_rng(conversion_seed.spawn(1)[0])
        # Without a click stage, every won impression counts as clicked.
        self.every_click = np.ones(count)
        # Filled at each campaign's first auction, where a bad forward signature raises.
        self.reads_market: list[bool | None] = [None] * count
        self.book = LogBook(config.episodes, config.steps)
        self.participants = [
            self._participant(campaign, seed)
            for campaign, seed in zip(config.campaigns, bidder_seeds, strict=True)
        ]
        self.inboxes = [_Inbox() for _ in config.campaigns]
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
            clicked=np.zeros(shape, dtype=bool),
            converted=np.zeros(shape, dtype=bool),
            segment=np.zeros(shape, dtype=np.int64),
            market_price=np.full(shape, np.nan),
            floor=np.zeros(shape),
        )

    def _participant(
        self, campaign: Campaign, seed: np.random.SeedSequence
    ) -> _Participant:
        # Child 0 fed the removed estimate noise; spawning it keeps child streams.
        seed.spawn(1)
        bidder = campaign.build(np.random.default_rng(seed))
        nodes = tuple(bidder.named_bidders())
        for path, node in nodes:
            attach_logger(node, self.book.logger(campaign.name, path))
        return _Participant(
            campaign=replace(campaign, bidder=bidder),
            nodes=tuple(node for _, node in nodes),
            learners=tuple(node for _, node in nodes if _learns(node)),
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
            self.log.clicked,
            self.log.converted,
            self.log.segment,
            self.log.market_price,
            self.log.floor,
        ):
            array.flags.writeable = False

    def _run_episode(self, episode: int) -> None:
        budget_left = np.array([p.campaign.budget for p in self.participants])
        reported_left = budget_left.copy()
        self.book.at_episode(episode)
        for participant in self.participants:
            for node in participant.nodes:
                node.on_episode_start(participant.campaign)
            participant.check_tree()

        for step in range(self.config.steps):
            self._run_step(episode, step, budget_left, reported_left)
            for callback in self.callbacks:
                callback.on_step_end(episode, step)

        # Pending feedback arrives before the episode report, so the two agree.
        self._deliver_due(math.inf, reported_left)

        self._end_episode(episode)

    def _run_step(
        self,
        episode: int,
        step: int,
        budget_left: npt.NDArray[np.float64],
        reported_left: npt.NDArray[np.float64],
    ) -> None:
        draw = self._next_draw(episode, step)
        self.book.at_step(episode, step)
        pctr = self.every_click if draw.pctr is None else draw.pctr
        pcvr = draw.pcvr
        values = self._event_values(pctr, pcvr)
        time = step * self.seconds_per_step
        # Bidders outside their segments are not asked to bid, so they cannot learn from it.
        active = [
            i
            for i, participant in enumerate(self.participants)
            if participant.takes_part_in(draw.segment)
        ]
        shared = {
            **draw.features,
            "features": draw.features,
            "time": time,
            "time_left": self.horizon - time,
            "horizon": self.horizon,
            "hour_of_day": (self.config.start_hour + time / SECONDS_PER_HOUR)
            % HOURS_PER_DAY,
            "segment": draw.segment,
            "floor": max(self.config.auction.reserve, draw.floor),
        }
        bids = np.zeros(len(self.participants))
        for i in active:
            bidder = self.participants[i].campaign.bidder
            inputs = {
                **shared,
                "budget_left": float(reported_left[i]),
            }
            # Building the truth for bidders that never read it would slow every run.
            if self._reads_market(i):
                inputs["market"] = MarketTruth(float(pctr[i]), float(pcvr[i]))
            bids[i] = call_forward(bidder, inputs)
        # Risk: if bids are not capped at the remaining budget, then spend can exceed it.

        # Same bits as clip(nan_to_num(...)), which took about 15% of run time.
        bids[~np.isfinite(bids)] = 0.0
        bids = np.minimum(np.maximum(bids, 0.0), budget_left)
        # A logged bid of 0 would win at reserve 0, so absent bidders bid -inf.
        auction_bids = np.full(len(self.participants), -np.inf)
        auction_bids[active] = bids[active]
        winner, price = self.config.auction.run(
            auction_bids,
            self.tie_rng,
            floor=draw.floor,
            market_price=draw.market_price,
        )
        clicked, converted = self._click_and_conversion(draw, winner)
        if winner != NO_WINNER:
            budget_left[winner] -= price

        for i in active:
            self._post_feedback(
                i, time, float(bids[i]), winner, price, clicked, converted
            )

        if self._delivers_at(step):
            self._deliver_due(time, reported_left)

        self.log.bids[episode, step] = bids
        self.log.true_values[episode, step] = values
        self.log.winner[episode, step] = winner
        self.log.price[episode, step] = price
        self.log.clicked[episode, step] = clicked
        self.log.converted[episode, step] = converted
        self.log.segment[episode, step] = self.segment_index[draw.segment]
        self._log_market(episode, step, draw)

    def _post_feedback(
        self,
        index: int,
        time: float,
        bid: float,
        winner: int,
        price: float,
        clicked: bool,
        converted: bool,
    ) -> None:
        feedback = self.config.feedback
        campaign = self.participants[index].campaign
        inbox = self.inboxes[index]
        won = index == winner
        late_conversion = converted and won and feedback.conversion_delay > 0
        reported_conversion = converted and not late_conversion
        result = BidResult(
            time=time,
            bid=bid,
            won=won,
            price=price if won else 0.0,
            clicked=clicked and won,
            converted=reported_conversion and won,
            realized_value=campaign.base_value
            if won and event_happened(campaign.event, won, clicked, reported_conversion)
            else 0.0,
        )
        inbox.reports.append((time + feedback.report_delay, result))
        if late_conversion:
            value = campaign.base_value if campaign.event == Event.CONVERSION else 0.0
            # A conversion never arrives before its auction's report.
            delay = max(feedback.conversion_delay, feedback.report_delay)
            inbox.conversions.append((time + delay, Conversion(time, value)))

    def _delivers_at(self, step: int) -> bool:
        interval = self.config.feedback.interval
        if interval == 0:
            return True

        if step == 0:
            return False

        # The first step at or after each multiple of the interval delivers.
        time = step * self.seconds_per_step
        before = (step - 1) * self.seconds_per_step
        return math.floor(time / interval) > math.floor(before / interval)

    def _deliver_due(self, time: float, reported_left: npt.NDArray[np.float64]) -> None:
        # A campaign outside this step's segment can still have feedback due.
        for i, inbox in enumerate(self.inboxes):
            self._deliver(i, *inbox.take_due(time), reported_left)

    def _deliver(
        self,
        index: int,
        reports: list[BidResult],
        conversions: list[Conversion],
        reported_left: npt.NDArray[np.float64],
    ) -> None:
        for report in reports:
            if report.won:
                reported_left[index] -= report.price

        learners = self.participants[index].learners
        if reports:
            for node in learners:
                node.update_batch(reports)

        if conversions:
            for node in learners:
                node.update_conversions(conversions)

    def _log_market(self, episode: int, step: int, draw: MarketDraw) -> None:
        # The arrays start as "none", so most runs skip both writes.
        if draw.market_price is not None:
            self.log.market_price[episode, step] = draw.market_price
        if draw.floor:
            self.log.floor[episode, step] = draw.floor

    def _reads_market(self, index: int) -> bool:
        reads_market = self.reads_market[index]
        if reads_market is None:
            # A plain Stack takes every input only to pass it on.
            reads_market = self.reads_market[index] = any(
                takes_input(node, "market")
                for node in self.participants[index].nodes
                if type(node) is not Stack
            )

        return reads_market

    def _click_and_conversion(self, draw: MarketDraw, winner: int) -> tuple[bool, bool]:
        clicked = winner != NO_WINNER
        # Only a click stage draws a click, every step, so other runs keep their streams.
        if draw.pctr is not None:
            click_draw = self.click_rng.random()
            clicked = clicked and click_draw < draw.pctr[winner]

        # One draw every step keeps the conversion stream aligned across runs.
        conversion_draw = self.conversion_rng.random()
        return clicked, clicked and conversion_draw < draw.pcvr[winner]

    def _event_values(
        self, pctr: npt.NDArray[np.float64], pcvr: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # Risk: if regrouped as base × (pCTR × pCVR), then floats round differently from main.
        if self.only_conversions:
            # Most runs value only conversions; skipping the masks keeps them as fast as before.
            return self.base_values * pctr * pcvr

        return (
            self.base_values
            * np.where(self.values_click, pctr, 1.0)
            * np.where(self.values_conversion, pcvr, 1.0)
        )

    def _next_draw(self, episode: int, step: int) -> MarketDraw:
        market = type(self.config.market).__name__
        where = f"{market} at episode {episode}, step {step}"
        try:
            draw = next(self.market_draws)
        except StopIteration:
            raise RuntimeError(f"{where}: the market stopped yielding draws") from None

        for name, rates in (("pctr", draw.pctr), ("pcvr", draw.pcvr)):
            if rates is None:
                continue

            if rates.shape != (len(self.participants),):
                raise ValueError(
                    f"{where}: got {name} of shape {rates.shape}, "
                    f"expected ({len(self.participants)},)"
                )

            # The negated check also catches NaN, which fails every comparison.
            if not np.all((rates >= 0.0) & (rates <= 1.0)):
                raise ValueError(f"{where}: {name} must lie in [0, 1]")

        # Most draws have neither, so the checks are skipped there to keep runs fast.
        if draw.market_price is not None:
            _check_money(where, "market_price", draw.market_price)
        if draw.floor:
            _check_money(where, "floor", draw.floor)

        clashes = draw.features.keys() & RESERVED_NAMES
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
            clicked = self.log.clicked[episode]
            converted = self.log.converted[episode]
            events = event_happened(participant.campaign.event, won, clicked, converted)
            report = EpisodeReport(
                budget=participant.campaign.budget,
                spend=float(np.sum(self.log.price[episode][won])),
                wins=int(np.sum(won)),
                clicks=int(np.sum(clicked & won)),
                conversions=int(np.sum(converted & won)),
                realized_value=int(np.sum(events)) * participant.campaign.base_value,
            )
            for node in participant.nodes:
                node.on_episode_end(report)
            participant.check_tree()
            reports[participant.campaign.name] = report

        read_only = MappingProxyType(reports)
        for callback in self.callbacks:
            callback.on_episode_end(episode, read_only)


class _Inbox:
    def __init__(self) -> None:
        # Each item adds a fixed delay to its auction time, so due times only rise.
        self.reports: deque[tuple[float, BidResult]] = deque()
        self.conversions: deque[tuple[float, Conversion]] = deque()

    def take_due(self, now: float) -> tuple[list[BidResult], list[Conversion]]:
        return _pop_due(self.reports, now), _pop_due(self.conversions, now)


def _pop_due[Item](queue: deque[tuple[float, Item]], now: float) -> list[Item]:
    items: list[Item] = []
    while queue and queue[0][0] <= now:
        items.append(queue.popleft()[1])

    return items


def event_happened[Outcome: (bool, npt.NDArray[np.bool_])](
    event: EventName, won: Outcome, clicked: Outcome, converted: Outcome
) -> Outcome:
    """Whether a campaign's event happened, per auction or per step of an array.

    ``clicked`` and ``converted`` hold the winner's outcome, so each is
    combined with ``won``.
    """
    if event == Event.IMPRESSION:
        return won

    if event == Event.CLICK:
        return clicked & won

    return converted & won


def _check_money(where: str, name: str, amount: float) -> None:
    if not (math.isfinite(amount) and amount >= 0):
        raise ValueError(f"{where}: {name} must be finite and >= 0, got {amount}")


def _learns(bidder: Bidder) -> bool:
    hooks = type(bidder)
    return (
        hooks.update_batch is not Bidder.update_batch
        or hooks.update is not Bidder.update
        or hooks.update_conversions is not Bidder.update_conversions
    )
