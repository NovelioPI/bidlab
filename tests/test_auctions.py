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


AUCTIONS = [FirstPrice, SecondPrice]


@pytest.mark.parametrize("auction_class", AUCTIONS)
def test_win_needs_above_market_price_and_floor(auction_class):
    auction = auction_class(0.1)
    bids = np.array([0.5, 0.3])
    assert auction.run(bids, rng(), market_price=0.6) == (-1, 0.0)
    assert auction.run(bids, rng(), market_price=0.4)[0] == 0
    assert auction.run(bids, rng(), floor=0.55) == (-1, 0.0)
    # Beating the market price is not enough below the floor.
    assert auction.run(bids, rng(), floor=0.55, market_price=0.4) == (-1, 0.0)
    # A bid at the floor wins, from the market floor or from the reserve.
    assert auction.run(bids, rng(), floor=0.5)[0] == 0
    assert auction_class(0.5).run(bids, rng(), floor=0.2)[0] == 0
    assert auction_class(0.6).run(bids, rng(), floor=0.2) == (-1, 0.0)


@pytest.mark.parametrize("auction_class", AUCTIONS)
def test_tie_with_market_price_sells_outside(auction_class):
    auction = auction_class()
    assert auction.run(np.array([0.6, 0.6]), rng(), market_price=0.6) == (-1, 0.0)
    assert auction.run(np.array([0.6]), rng(), market_price=0.6) == (-1, 0.0)


def test_campaign_ties_break_as_before_with_market_price():
    bids = np.array([0.8, 0.8, 0.1])
    plain = [SecondPrice().run(bids, rng()) for _ in range(5)]
    priced = [SecondPrice().run(bids, rng(), market_price=0.5) for _ in range(5)]
    assert [w for w, _ in priced] == [w for w, _ in plain]


@pytest.mark.parametrize(
    ("reserve", "floor", "market_price", "bids", "price"),
    [
        (0.0, 0.0, 0.2, [0.9, 0.5], 0.5),  # next bid highest
        (0.0, 0.0, 0.6, [0.9, 0.5], 0.6),  # market price highest
        (0.0, 0.7, 0.6, [0.9, 0.5], 0.7),  # market floor highest
        (0.8, 0.7, 0.6, [0.9, 0.5], 0.8),  # reserve highest
        (0.0, 0.0, 0.3, [0.9], 0.3),  # no other campaign
        (0.0, 0.0, None, [0.9], 0.0),  # no market price, no floor: as before
        (0.0, 0.4, None, [0.9, 0.3], 0.4),  # next bid below the floor does not count
    ],
)
def test_second_price_pays_max_of_price_floor_and_next_bid(
    reserve, floor, market_price, bids, price
):
    auction = SecondPrice(reserve)
    result = auction.run(np.array(bids), rng(), floor=floor, market_price=market_price)
    assert result == (0, price)


def test_first_price_pays_bid_against_market_price():
    bids = np.array([0.9, 0.5])
    assert FirstPrice(0.1).run(bids, rng(), floor=0.7, market_price=0.8) == (0, 0.9)


@given(
    bids=st.lists(st.floats(0, 10), min_size=1, max_size=8),
    reserve=st.floats(0, 5),
    floor=st.floats(0, 5),
    market_price=st.none() | st.floats(0, 10),
    auction_class=st.sampled_from(AUCTIONS),
)
def test_winner_beats_market_price_and_floor(
    bids, reserve, floor, market_price, auction_class
):
    array = np.array(bids)
    lowest = max(reserve, floor)
    outside = -np.inf if market_price is None else market_price
    winner, price = auction_class(reserve).run(
        array, rng(), floor=floor, market_price=market_price
    )
    if winner == -1:
        assert price == 0.0
        assert array.max() < lowest or array.max() <= outside
        return

    assert array[winner] == array.max()
    assert array[winner] >= lowest and array[winner] > outside
    assert max(lowest, outside) <= price <= array[winner]
