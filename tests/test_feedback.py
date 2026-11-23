import math

import pytest

import bidlab as bl

SETTINGS = ("interval", "report_delay", "conversion_delay")


def lone_campaign():
    return bl.Campaign(
        "c", bl.Stack(bl.OracleValue(), bl.Linear()), budget=10, base_value=5
    )


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_bad_feedback_setting_rejected(setting, bad):
    with pytest.raises(ValueError, match=setting):
        bl.Feedback(**{setting: bad})


def test_feedback_settings_are_floats():
    feedback = bl.Feedback(interval=900, report_delay=60, conversion_delay=3600)
    assert all(type(getattr(feedback, name)) is float for name in SETTINGS)
    assert bl.SimConfig(campaigns=[lone_campaign()]).feedback == bl.Feedback(0, 0, 0)


def test_feedback_must_be_feedback():
    with pytest.raises(TypeError, match="feedback"):
        bl.SimConfig(
            campaigns=[lone_campaign()],
            feedback={"interval": 900},  # pyright: ignore[reportArgumentType]  Reason: checks the run-time type check.
        )
