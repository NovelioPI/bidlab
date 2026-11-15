import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from auction_sim.auctions import run_auction


def rng():
    return np.random.default_rng(0)


def test_first_price_pays_bid():
    assert run_auction(np.array([0.3, 0.9, 0.5]), "first", 0.0, rng()) == (1, 0.9)


def test_second_price_pays_max_of_second_and_reserve():
    bids = np.array([0.3, 0.9, 0.5])
    assert run_auction(bids, "second", 0.0, rng()) == (1, 0.5)
    assert run_auction(bids, "second", 0.7, rng()) == (1, 0.7)
    assert run_auction(np.array([0.9]), "second", 0.2, rng()) == (0, 0.2)


def test_bids_below_reserve_lose():
    assert run_auction(np.array([0.1, 0.2]), "second", 0.5, rng()) == (-1, 0.0)
    assert run_auction(np.array([0.1, 0.2]), "first", 0.5, rng()) == (-1, 0.0)


def test_ties_split_and_repeat_with_seed():
    bids = np.array([0.8, 0.8, 0.1])
    g = np.random.default_rng(42)
    results = [run_auction(bids, "second", 0.0, g) for _ in range(200)]
    winners = {w for w, _ in results}
    assert winners == {0, 1}
    assert all(p == 0.8 for _, p in results)
    g2 = np.random.default_rng(42)
    assert results == [run_auction(bids, "second", 0.0, g2) for _ in range(200)]


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        run_auction(np.array([1.0]), "vcg", 0.0, rng())


@given(
    bids=st.lists(st.floats(0, 10), min_size=1, max_size=8),
    reserve=st.floats(0, 5),
    kind=st.sampled_from(["first", "second"]),
)
def test_price_between_reserve_and_winning_bid(bids, reserve, kind):
    array = np.array(bids)
    winner, price = run_auction(array, kind, reserve, rng())
    if winner == -1:
        assert array.max() < reserve
        return
    assert array[winner] == array.max()
    assert reserve <= price <= array[winner]
