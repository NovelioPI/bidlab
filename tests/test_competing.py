import pytest

import bidlab as bl


def campaign(name: str = "a") -> bl.Campaign:
    return bl.Campaign(
        name, bl.Stack(bl.OracleValue(), bl.Linear()), budget=10.0, base_value=5.0
    )


def test_competing_defaults_to_true():
    assert bl.SimConfig(campaigns=[campaign()]).competing is True


@pytest.mark.parametrize("value", [0, 1, "no", None])
def test_competing_must_be_bool(value):
    with pytest.raises(TypeError, match="competing"):
        bl.SimConfig(campaigns=[campaign()], competing=value)  # pyright: ignore[reportArgumentType]  Reason: the test passes a wrong type on purpose.
