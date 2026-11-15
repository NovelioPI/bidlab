import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from bidlab.auctions import FirstPrice, SecondPrice


def rng():
    return np.random.default_rng(0)


def test_first_price_pays_bid():
    assert FirstPrice().run(np.array([0.3, 0.9, 0.5]), rng()) == (1, 0.9)


def test_second_price_pays_max_of_second_and_reserve():
    bids = np.array([0.3, 0.9, 0.5])
    assert SecondPrice().run(bids, rng()) == (1, 0.5)
    assert SecondPrice(0.7).run(bids, rng()) == (1, 0.7)
    assert SecondPrice(0.2).run(np.array([0.9]), rng()) == (0, 0.2)


def test_bids_below_reserve_lose():
    assert SecondPrice(0.5).run(np.array([0.1, 0.2]), rng()) == (-1, 0.0)
    assert FirstPrice(0.5).run(np.array([0.1, 0.2]), rng()) == (-1, 0.0)


def test_ties_split_and_repeat_with_seed():
    bids = np.array([0.8, 0.8, 0.1])
    g = np.random.default_rng(42)
    results = [SecondPrice().run(bids, g) for _ in range(200)]
    winners = {w for w, _ in results}
    assert winners == {0, 1}
    assert all(p == 0.8 for _, p in results)
    g2 = np.random.default_rng(42)
    assert results == [SecondPrice().run(bids, g2) for _ in range(200)]


@pytest.mark.parametrize("auction_class", [FirstPrice, SecondPrice])
def test_negative_reserve_rejected(auction_class):
    with pytest.raises(ValueError, match="reserve"):
        auction_class(reserve=-1)


@given(
    bids=st.lists(st.floats(0, 10), min_size=1, max_size=8),
    reserve=st.floats(0, 5),
    auction_class=st.sampled_from([FirstPrice, SecondPrice]),
)
def test_price_between_reserve_and_winning_bid(bids, reserve, auction_class):
    array = np.array(bids)
    winner, price = auction_class(reserve).run(array, rng())
    if winner == -1:
        assert array.max() < reserve
        return

    assert array[winner] == array.max()
    assert reserve <= price <= array[winner]
