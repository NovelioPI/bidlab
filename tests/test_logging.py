import math
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, override

import numpy as np
import pytest

from bidlab.auctions import FirstPrice
from bidlab.bidders import Bidder, Linear, PidPacing, Stack, UcbMultiplier
from bidlab.config import Campaign, SimConfig
from bidlab.results import Results
from bidlab.runner import run_many
from bidlab.simulator import RunLog, run
from bidlab.types import BidResult, EpisodeReport

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
    def forward(self, value: float) -> float:
        self.act("forward")
        return value

    @override
    def update_batch(self, bid_results: Sequence[BidResult]) -> None:
        self.act("update_batch")
        super().update_batch(bid_results)
        self.auction += 1

    @override
    def update(self, result: BidResult) -> None:
        self.act("update")

    @override
    def on_episode_end(self, report: EpisodeReport) -> None:
        self.act("on_episode_end")


class Holder(Bidder):
    """Holds a child and passes its output on."""

    def __init__(self, child: Bidder):
        self.child = child

    @override
    def forward(self, value: float) -> float:
        return self.child.forward(value)


def config(*campaigns: Campaign, **kwargs: Any) -> SimConfig:
    defaults: dict[str, Any] = {"episodes": 2, "steps": 50, "seed": 3}
    return SimConfig(campaigns=campaigns, **(defaults | kwargs))


def logged(hook: str, action: Action, **kwargs: Any) -> RunLog:
    return run(
        config(
            Campaign("b", Logs(hook, action), budget=1e6, base_value=5.0),
            **kwargs,
        )
    )


def test_step_values_fill_grid():
    log = logged("forward", lambda bidder, auction: bidder.log("lam", auction / 10))
    expected = np.tile(np.arange(50) / 10, (2, 1))
    np.testing.assert_array_equal(log.logged["lam"]["b"], expected)


def test_dict_equals_single_calls():
    both = logged("forward", lambda bidder, _: bidder.log({"a": 1.0, "b": 2.0}))

    def one_by_one(bidder: Bidder, _: int) -> None:
        bidder.log("a", 1.0)
        bidder.log("b", 2.0)

    single = logged("forward", one_by_one)
    for name, value in (("a", 1.0), ("b", 2.0)):
        assert (both.logged[name]["b"] == value).all()
        np.testing.assert_array_equal(both.logged[name]["b"], single.logged[name]["b"])


def test_missing_cells_are_nan():
    def every_third(bidder: Bidder, auction: int) -> None:
        if auction % 3 == 0:
            bidder.log("x", 1.0)

    values = logged("forward", every_third).logged["x"]["b"]
    logged_cells = np.zeros((2, 50), dtype=bool)
    logged_cells[:, ::3] = True
    assert (values[logged_cells] == 1.0).all()
    assert np.isnan(values[~logged_cells]).all()


@pytest.mark.parametrize("hook", ["update", "update_batch"])
def test_update_logs_land_on_their_step(hook: str):
    log = logged(hook, lambda bidder, auction: bidder.log("seen", auction))
    np.testing.assert_array_equal(
        log.logged["seen"]["b"], np.tile(np.arange(50), (2, 1))
    )


@pytest.mark.parametrize("hook", ["on_episode_start", "on_episode_end"])
def test_episode_values_shape(hook: str):
    episodes = iter(range(100))
    log = logged(
        hook, lambda bidder, _: bidder.log("episode", next(episodes)), episodes=3
    )
    np.testing.assert_array_equal(log.logged["episode"]["b"], [0.0, 1.0, 2.0])


def test_last_value_wins():
    def twice(bidder: Bidder, auction: int) -> None:
        bidder.log("x", -1.0)
        bidder.log("x", auction)

    assert logged("forward", twice).logged["x"]["b"][1, 7] == 7.0

    def twice_per_episode(bidder: Bidder, _: int) -> None:
        bidder.log("y", -1.0)
        bidder.log("y", 4.0)

    assert (logged("on_episode_end", twice_per_episode).logged["y"]["b"] == 4.0).all()


class Both(Bidder):
    """Logs one name per step and per episode."""

    @override
    def forward(self, value: float) -> float:
        self.log("x", 1.0)
        return value

    @override
    def on_episode_end(self, report: EpisodeReport) -> None:
        self.log("x", 2.0)


class EpisodeFirst(Bidder):
    """Logs one name per episode, then per step."""

    @override
    def on_episode_start(self, campaign: Campaign) -> None:
        self.log("x", 2.0)

    @override
    def forward(self, value: float) -> float:
        self.log("x", 1.0)
        return value


@pytest.mark.parametrize(
    ("bidder", "second"), [(Both(), "episode"), (EpisodeFirst(), "step")]
)
def test_step_and_episode_name_raises(bidder: Bidder, second: str):
    pattern = rf"'both'.*'x'.*per step and per episode; this {second} log"
    with pytest.raises(ValueError, match=pattern):
        run(config(Campaign("both", bidder, budget=1e6, base_value=5.0)))


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
        Linear().log(*arguments)


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

    logs = logged("forward", kinds).logged
    cells = {name: logs[name]["b"][0, 0] for name in logs}
    assert cells["bool"] == 1.0 and cells["int"] == 3.0 and cells["numpy"] == 0.5
    assert np.isnan(cells["nan"]) and cells["inf"] == math.inf


class LogsInBind(Linear):
    """Logs while it is bound, before the run starts."""

    @override
    def bind(self, rng: np.random.Generator) -> None:
        super().bind(rng)
        self.log("bound", 1.0)


def test_log_outside_run_does_nothing():
    bidder = Linear()
    bidder.log("x", 1.0)
    bidder.log({"y": 2.0})
    assert bidder.forward(2.0) == 2.0
    log = run(config(Campaign("bind", LogsInBind(), budget=1e6, base_value=5.0)))
    assert "bound" not in log.logged


class Keeps(Bidder):
    """Keeps a reference to its run's copy, as a user's bidder might."""

    kept: ClassVar[list[Bidder]] = []

    @override
    def forward(self, value: float) -> float:
        Keeps.kept.append(self)
        self.log("x", 1.0)
        return value


def test_log_after_run_does_nothing():
    Keeps.kept.clear()
    log = run(config(Campaign("k", Keeps(), budget=1e6, base_value=5.0)))
    copy = Keeps.kept[-1]
    copy.log("x", 99.0)
    copy.log("fresh", 5.0)
    assert set(log.logged) == {"x"}
    assert (log.logged["x"]["k"] == 1.0).all()


def test_specs_keep_own_arrays():
    first = Campaign(
        "one",
        Logs("forward", lambda b, _: b.log("x", 1.0)),
        budget=1e6,
        base_value=5.0,
    )
    second = Campaign(
        "two",
        Logs("forward", lambda b, _: b.log("x", 2.0)),
        budget=1e6,
        base_value=5.0,
    )
    logs = run(config(first, second)).logged["x"]
    assert set(logs) == {"one", "two"}
    assert (logs["one"] == 1.0).all() and (logs["two"] == 2.0).all()


def test_logs_read_only():
    log = logged("forward", lambda bidder, _: bidder.log("x", 1.0))
    with pytest.raises(TypeError):
        log.logged["y"] = {}  # pyright: ignore[reportIndexIssue]  Reason: the test checks the mapping is read-only.

    with pytest.raises(TypeError):
        log.logged["x"]["c"] = np.zeros(1)  # pyright: ignore[reportIndexIssue]  Reason: the test checks the inner mapping is read-only.

    assert not log.logged["x"]["b"].flags.writeable


def test_no_logs_empty():
    log = run(config(Campaign("t", Linear(), budget=1e6, base_value=5.0)))
    assert dict(log.logged) == {}


def test_child_name_prefixed():
    child = Logs("forward", lambda bidder, _: bidder.log("lam", 1.0))
    log = run(config(Campaign("s", Stack(Linear(), child), budget=1e6, base_value=5.0)))
    assert set(log.logged) == {"bidders.1.lam"}


@pytest.mark.parametrize("as_dict", [False, True])
def test_plain_name(as_dict: bool):
    def plain(bidder: Bidder, _: int) -> None:
        if as_dict:
            bidder.log({"lam": 1.0}, log_prefix=False)
        else:
            bidder.log("lam", 1.0, log_prefix=False)

    log = run(
        config(
            Campaign(
                "s",
                Stack(Linear(), Logs("forward", plain)),
                budget=1e6,
                base_value=5.0,
            )
        )
    )
    assert set(log.logged) == {"lam"}


class TopLogs(Holder):
    """Logs ``lam`` itself and holds a child that logs it plainly."""

    def __init__(self):
        super().__init__(
            Logs("forward", lambda b, _: b.log("lam", 2.0, log_prefix=False))
        )

    @override
    def forward(self, value: float) -> float:
        self.log("lam", 1.0)
        return super().forward(value)


class TopAndChild(Holder):
    """Logs ``a`` itself and holds a child that logs ``b``."""

    def __init__(self):
        super().__init__(Logs("forward", lambda b, _: b.log("b", 2.0)))

    @override
    def forward(self, value: float) -> float:
        self.log("a", 1.0)
        return super().forward(value)


def test_different_names_in_tree_allowed():
    log = run(config(Campaign("t", TopAndChild(), budget=1e6, base_value=5.0)))
    assert set(log.logged) == {"a", "child.b"}


def test_name_clash_raises():
    with pytest.raises(ValueError, match=r"'clash'.*'lam'.*'' and 'child'"):
        run(config(Campaign("clash", TopLogs(), budget=1e6, base_value=5.0)))


def test_pid_logs_bid_multiplier():
    campaign = Campaign("pid", PidPacing(), budget=3.0, base_value=5.0)
    log = run(config(campaign, auction=FirstPrice()))
    multiplier = log.logged["multiplier"]["pid"]
    assert multiplier.shape == (2, 50)
    assert len(np.unique(multiplier)) > 1
    budget_left = 3.0 - np.cumsum(np.where(log.winner == 0, log.price, 0.0), axis=1)
    before = np.concatenate([np.full((2, 1), 3.0), budget_left[:, :-1]], axis=1)
    # cumsum rounds differently from the simulator's running subtraction.
    uncapped = log.bids[:, :, 0] < before - 1e-9
    assert uncapped.sum() > 50
    np.testing.assert_allclose(
        log.bids[:, :, 0][uncapped], (multiplier * log.true_values[:, :, 0])[uncapped]
    )


def test_ucb_logs_arm():
    bandit = UcbMultiplier()
    log = run(config(Campaign("ucb", bandit, budget=1e6, base_value=5.0), episodes=4))
    arms = log.logged["multiplier"]["ucb"]
    assert arms.shape == (4,)
    assert all(np.isclose(bandit.arms, arm).any() for arm in arms)
    np.testing.assert_allclose(
        log.bids[:, :, 0], arms[:, None] * log.true_values[:, :, 0]
    )


def test_log_state_off():
    campaigns = (
        Campaign("pid", PidPacing(log_state=False), budget=1e6, base_value=5.0),
        Campaign("ucb", UcbMultiplier(log_state=False), budget=1e6, base_value=5.0),
    )
    assert dict(run(config(*campaigns)).logged) == {}


def test_stacked_pid_prefixed():
    log = run(
        config(Campaign("s", Stack(Linear(), PidPacing()), budget=1e6, base_value=5.0))
    )
    assert set(log.logged) == {"bidders.1.multiplier"}


def test_logging_leaves_run_unchanged():
    def campaigns(log_state: bool) -> tuple[Campaign, ...]:
        return (
            Campaign("pid", PidPacing(log_state=log_state), budget=3.0, base_value=5.0),
            Campaign(
                "ucb",
                UcbMultiplier(log_state=log_state),
                budget=5.0,
                base_value=4.0,
            ),
            Campaign(
                "logs",
                Logs("forward", lambda b, a: b.log("a", a)),
                budget=1e6,
                base_value=5.0,
            ),
        )

    on, off = (run(config(*campaigns(state), episodes=4)) for state in (True, False))
    for name in ("bids", "price", "winner", "converted"):
        np.testing.assert_array_equal(getattr(on, name), getattr(off, name))


class Sometimes(Bidder):
    """Logs only in runs whose first draw from its stream is below one half."""

    @override
    def on_episode_start(self, campaign: Campaign) -> None:
        self.logging = self.rng.random() < 0.5

    @override
    def forward(self, value: float) -> float:
        if self.logging:
            self.log("x", value)
        return value


def per_auction() -> Campaign:
    return Campaign(
        "a",
        Logs("forward", lambda b, a: b.log("lam", a * 2)),
        budget=1e6,
        base_value=5.0,
    )


def test_run_many_stacks_seeds():
    results = run_many(config(per_auction(), episodes=2), seeds=range(3))
    stacked = results.logged["lam"]["a"]
    assert stacked.shape == (3, 2, 50)
    for seed in range(3):
        single = run(config(per_auction(), episodes=2, seed=seed))
        np.testing.assert_array_equal(stacked[seed], single.logged["lam"]["a"])


def test_run_many_episode_values():
    campaign = Campaign("ucb", UcbMultiplier(), budget=1e6, base_value=5.0)
    results = run_many(config(campaign, episodes=4), seeds=range(3))
    assert results.logged["multiplier"]["ucb"].shape == (3, 4)


def test_run_many_missing_seed_nan():
    campaign = Campaign("s", Sometimes(), budget=1e6, base_value=5.0)
    stacked = run_many(config(campaign, episodes=1), seeds=range(8)).logged["x"]["s"]
    logged = [
        "x" in run(config(campaign, episodes=1, seed=seed)).logged for seed in range(8)
    ]
    assert any(logged) and not all(logged)
    for seed, did_log in enumerate(logged):
        assert np.isnan(stacked[seed]).all() != did_log


def test_run_many_workers_logs_equal():
    pid = Campaign("pid", PidPacing(), budget=3.0, base_value=5.0)
    serial = run_many(config(pid, per_auction()), seeds=range(3))
    parallel = run_many(config(pid, per_auction()), seeds=range(3), workers=2)
    assert parallel.equals(serial)
    assert set(parallel.logged) == {"multiplier", "lam"}
    for name in parallel.logged:
        for campaign in parallel.logged[name]:
            np.testing.assert_array_equal(
                parallel.logged[name][campaign], serial.logged[name][campaign]
            )


def test_equals_compares_logs():
    results = run_many(config(per_auction()), seeds=range(2))
    changed = {"lam": {"a": results.logged["lam"]["a"] + 1.0}}
    other = Results(columns=results.columns, rows=results.rows, logged=changed)
    same = Results(columns=results.columns, rows=results.rows, logged=results.logged)
    assert same.equals(results)
    assert not other.equals(results)
    assert not Results(columns=results.columns, rows=results.rows).equals(results)


class SurplusLogger(Bidder):
    """Logs its own surplus reward in update."""

    @override
    def forward(self, value: float) -> float:
        return value

    @override
    def update(self, result: BidResult) -> None:
        self.log("reward", result.realized_value - result.price)


def test_reward_logged_in_update():
    campaign = Campaign("s", SurplusLogger(), budget=1e6, base_value=5.0)
    config = SimConfig(campaigns=[campaign], episodes=2, steps=20, auction=FirstPrice())
    log = run(config)
    won = log.winner == 0
    expected = np.where(won & log.converted, 5.0, 0.0) - np.where(won, log.price, 0.0)
    np.testing.assert_array_equal(log.logged["reward"]["s"], expected)
    results = run_many(config, seeds=range(2))
    assert results.logged["reward"]["s"].shape == (2, 2, 20)
