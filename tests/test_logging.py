import math
from collections.abc import Callable, Sequence
from typing import Any, override

import numpy as np
import pytest

from bidlab.auctions import FirstPrice
from bidlab.bidders import Bidder, PidPacing, Stack, Truthful, UcbMultiplier
from bidlab.config import BidderSpec, SimConfig
from bidlab.simulator import RunLog, run
from bidlab.types import Campaign, EpisodeStats, Feedback, Observation

CAMPAIGN = Campaign(budget=1e6, value=5.0)

Action = Callable[[Bidder, int], None]


class Logs(Bidder):
    """Runs a given action in a given hook, with a per-episode auction count."""

    def __init__(self, hook: str, action: Action):
        self.hook = hook
        self.action = action
        self.auction = 0

    def act(self, hook: str) -> None:
        if hook == self.hook:
            self.action(self, self.auction)

    @override
    def on_episode_start(self, campaign: Campaign) -> None:
        self.auction = 0
        self.act("on_episode_start")

    @override
    def forward(self, value: float, observation: Observation) -> float:
        self.act("forward")
        return value

    @override
    def update_batch(self, feedbacks: Sequence[Feedback]) -> None:
        self.act("update_batch")
        super().update_batch(feedbacks)
        self.auction += 1

    @override
    def update(self, feedback: Feedback) -> None:
        self.act("update")

    @override
    def on_episode_end(self, stats: EpisodeStats, episode_reward: float | None) -> None:
        self.act("on_episode_end")


class Holder(Bidder):
    """Holds a child and passes its output on."""

    def __init__(self, child: Bidder):
        self.child = child

    @override
    def forward(self, value: float, observation: Observation) -> float:
        return self.child.forward(value, observation)


def config(*specs: BidderSpec, **kwargs: Any) -> SimConfig:
    defaults: dict[str, Any] = {"episodes": 2, "steps": 50, "seed": 3}
    return SimConfig(bidders=specs, **(defaults | kwargs))


def logged(hook: str, action: Action, **kwargs: Any) -> RunLog:
    return run(config(BidderSpec("b", Logs(hook, action), CAMPAIGN), **kwargs))


def test_step_values_fill_grid():
    log = logged("forward", lambda bidder, auction: bidder.log("lam", auction / 10))
    expected = np.tile(np.arange(50) / 10, (2, 1))
    np.testing.assert_array_equal(log.logs["lam"]["b"], expected)


def test_dict_equals_single_calls():
    both = logged("forward", lambda bidder, _: bidder.log({"a": 1.0, "b": 2.0}))

    def one_by_one(bidder: Bidder, _: int) -> None:
        bidder.log("a", 1.0)
        bidder.log("b", 2.0)

    single = logged("forward", one_by_one)
    for name, value in (("a", 1.0), ("b", 2.0)):
        assert (both.logs[name]["b"] == value).all()
        np.testing.assert_array_equal(both.logs[name]["b"], single.logs[name]["b"])


def test_missing_cells_are_nan():
    def every_third(bidder: Bidder, auction: int) -> None:
        if auction % 3 == 0:
            bidder.log("x", 1.0)

    values = logged("forward", every_third).logs["x"]["b"]
    logged_cells = np.zeros((2, 50), dtype=bool)
    logged_cells[:, ::3] = True
    assert (values[logged_cells] == 1.0).all()
    assert np.isnan(values[~logged_cells]).all()


@pytest.mark.parametrize("hook", ["update", "update_batch"])
def test_update_logs_land_on_their_step(hook: str):
    log = logged(hook, lambda bidder, auction: bidder.log("seen", auction))
    np.testing.assert_array_equal(log.logs["seen"]["b"], np.tile(np.arange(50), (2, 1)))


@pytest.mark.parametrize("hook", ["on_episode_start", "on_episode_end"])
def test_episode_values_shape(hook: str):
    episodes = iter(range(100))
    log = logged(
        hook, lambda bidder, _: bidder.log("episode", next(episodes)), episodes=3
    )
    np.testing.assert_array_equal(log.logs["episode"]["b"], [0.0, 1.0, 2.0])


def test_last_value_wins():
    def twice(bidder: Bidder, auction: int) -> None:
        bidder.log("x", -1.0)
        bidder.log("x", auction)

    assert logged("forward", twice).logs["x"]["b"][1, 7] == 7.0

    def twice_per_episode(bidder: Bidder, _: int) -> None:
        bidder.log("y", -1.0)
        bidder.log("y", 4.0)

    assert (logged("on_episode_end", twice_per_episode).logs["y"]["b"] == 4.0).all()


class Both(Bidder):
    """Logs one name per step and per episode."""

    @override
    def forward(self, value: float, observation: Observation) -> float:
        self.log("x", 1.0)
        return value

    @override
    def on_episode_end(self, stats: EpisodeStats, episode_reward: float | None) -> None:
        self.log("x", 2.0)


def test_step_and_episode_name_raises():
    with pytest.raises(ValueError, match=r"'both'.*'x'.*per step and per episode"):
        run(config(BidderSpec("both", Both(), CAMPAIGN)))


@pytest.mark.parametrize(
    "arguments",
    [
        ("x", "1.0"),
        ("x", None),
        ("x", 1j),
        ("x", [1.0]),
        ("x", np.array([1.0, 2.0])),
        ("", 1.0),
        (3, 1.0),
        ({"x": "1"},),
        ({7: 1.0},),
    ],
    ids=[
        "str",
        "none",
        "complex",
        "list",
        "array",
        "empty",
        "int-name",
        "dict-str",
        "dict-key",
    ],
)
def test_bad_values_raise(arguments: tuple[Any, ...]):
    def bad(bidder: Bidder, _: int) -> None:
        bidder.log(*arguments)

    with pytest.raises(TypeError, match="logged"):
        logged("forward", bad)

    with pytest.raises(TypeError, match="logged"):
        Truthful().log(*arguments)


def test_number_kinds_stored():
    def kinds(bidder: Bidder, _: int) -> None:
        bidder.log(
            {
                "bool": True,
                "int": 3,
                "numpy": np.float32(0.5),
                "nan": math.nan,
                "inf": math.inf,
            }
        )

    logs = logged("forward", kinds).logs
    cells = {name: logs[name]["b"][0, 0] for name in logs}
    assert cells["bool"] == 1.0 and cells["int"] == 3.0 and cells["numpy"] == 0.5
    assert np.isnan(cells["nan"]) and cells["inf"] == math.inf


class LogsInBind(Truthful):
    """Logs while it is bound, before the run starts."""

    @override
    def bind(self, rng: np.random.Generator) -> None:
        super().bind(rng)
        self.log("bound", 1.0)


def test_log_outside_run_does_nothing():
    bidder = Truthful()
    bidder.log("x", 1.0)
    bidder.log({"y": 2.0})
    assert bidder.forward(2.0, Observation(time=0.0, value=2.0)) == 2.0
    log = run(config(BidderSpec("bind", LogsInBind(), CAMPAIGN)))
    assert "bound" not in log.logs


def test_specs_keep_own_arrays():
    first = BidderSpec("one", Logs("forward", lambda b, _: b.log("x", 1.0)), CAMPAIGN)
    second = BidderSpec("two", Logs("forward", lambda b, _: b.log("x", 2.0)), CAMPAIGN)
    logs = run(config(first, second)).logs["x"]
    assert set(logs) == {"one", "two"}
    assert (logs["one"] == 1.0).all() and (logs["two"] == 2.0).all()


def test_logs_read_only():
    log = logged("forward", lambda bidder, _: bidder.log("x", 1.0))
    with pytest.raises(TypeError):
        log.logs["y"] = {}  # pyright: ignore[reportIndexIssue]  Reason: the test checks the mapping is read-only.

    with pytest.raises(TypeError):
        log.logs["x"]["c"] = np.zeros(1)  # pyright: ignore[reportIndexIssue]  Reason: the test checks the inner mapping is read-only.

    assert not log.logs["x"]["b"].flags.writeable


def test_no_logs_empty():
    log = run(config(BidderSpec("t", Truthful(), CAMPAIGN)))
    assert dict(log.logs) == {}


def test_child_name_prefixed():
    child = Logs("forward", lambda bidder, _: bidder.log("lam", 1.0))
    log = run(config(BidderSpec("s", Stack(Truthful(), child), CAMPAIGN)))
    assert set(log.logs) == {"bidders.1.lam"}


@pytest.mark.parametrize("as_dict", [False, True])
def test_plain_name(as_dict: bool):
    def plain(bidder: Bidder, _: int) -> None:
        if as_dict:
            bidder.log({"lam": 1.0}, log_prefix=False)
        else:
            bidder.log("lam", 1.0, log_prefix=False)

    log = run(
        config(BidderSpec("s", Stack(Truthful(), Logs("forward", plain)), CAMPAIGN))
    )
    assert set(log.logs) == {"lam"}


class TopLogs(Holder):
    """Logs ``lam`` itself and holds a child that logs it plainly."""

    def __init__(self):
        super().__init__(
            Logs("forward", lambda b, _: b.log("lam", 2.0, log_prefix=False))
        )

    @override
    def forward(self, value: float, observation: Observation) -> float:
        self.log("lam", 1.0)
        return super().forward(value, observation)


def test_name_clash_raises():
    with pytest.raises(ValueError, match=r"'clash'.*'lam'.*'' and 'child'"):
        run(config(BidderSpec("clash", TopLogs(), CAMPAIGN)))


def test_pid_logs_bid_multiplier():
    spec = BidderSpec("pid", PidPacing(), Campaign(budget=3.0, value=5.0))
    log = run(config(spec, auction=FirstPrice()))
    multiplier = log.logs["multiplier"]["pid"]
    assert multiplier.shape == (2, 50)
    assert len(np.unique(multiplier)) > 1
    budget_left = 3.0 - np.cumsum(np.where(log.winner == 0, log.price, 0.0), axis=1)
    before = np.concatenate([np.full((2, 1), 3.0), budget_left[:, :-1]], axis=1)
    # cumsum rounds differently from the simulator's running subtraction.
    uncapped = log.bids[:, :, 0] < before - 1e-9
    assert uncapped.sum() > 50
    np.testing.assert_allclose(
        log.bids[:, :, 0][uncapped], (multiplier * log.values[:, :, 0])[uncapped]
    )


def test_ucb_logs_arm():
    bandit = UcbMultiplier()
    log = run(config(BidderSpec("ucb", bandit, CAMPAIGN), episodes=4))
    arms = log.logs["multiplier"]["ucb"]
    assert arms.shape == (4,)
    assert all(np.isclose(bandit.arms, arm).any() for arm in arms)
    np.testing.assert_allclose(log.bids[:, :, 0], arms[:, None] * log.values[:, :, 0])


def test_log_state_off():
    specs = (
        BidderSpec("pid", PidPacing(log_state=False), CAMPAIGN),
        BidderSpec("ucb", UcbMultiplier(log_state=False), CAMPAIGN),
    )
    assert dict(run(config(*specs)).logs) == {}


def test_stacked_pid_prefixed():
    log = run(config(BidderSpec("s", Stack(Truthful(), PidPacing()), CAMPAIGN)))
    assert set(log.logs) == {"bidders.1.multiplier"}


def test_logging_leaves_run_unchanged():
    def specs(log_state: bool) -> tuple[BidderSpec, ...]:
        return (
            BidderSpec("pid", PidPacing(log_state=log_state), Campaign(3.0, 5.0)),
            BidderSpec("ucb", UcbMultiplier(log_state=log_state), Campaign(5.0, 4.0)),
            BidderSpec("logs", Logs("forward", lambda b, a: b.log("a", a)), CAMPAIGN),
        )

    on, off = (run(config(*specs(state), episodes=4)) for state in (True, False))
    for name in ("bids", "price", "winner", "rewards"):
        np.testing.assert_array_equal(getattr(on, name), getattr(off, name))
