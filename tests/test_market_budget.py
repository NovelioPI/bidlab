import math

import pytest

from bidlab.bidders import Linear, OracleValue, PidPacing, Stack
from bidlab.config import Campaign


def truthful():
    return Stack(OracleValue(), Linear())


def test_budget_fraction_defaults_to_one():
    campaign = Campaign("a", truthful(), budget=3, base_value=5)
    assert campaign.budget_fraction == 1.0
    assert type(campaign.budget_fraction) is float


@pytest.mark.parametrize("fraction", [0, -0.5, math.nan, math.inf])
def test_bad_fraction_raises(fraction):
    with pytest.raises(ValueError, match="budget_fraction"):
        Campaign("a", truthful(), budget=3, base_value=5, budget_fraction=fraction)


@pytest.mark.parametrize("budget", [0, -1])
def test_bad_budget_raises_and_none_builds(budget):
    with pytest.raises(ValueError, match="budget"):
        Campaign("a", truthful(), budget=budget, base_value=5)

    campaign = Campaign("a", truthful(), budget=None, base_value=5, budget_fraction=2)
    assert campaign.budget is None
    assert campaign.budget_fraction == 2.0


def test_pid_pacing_rejects_an_unresolved_budget():
    campaign = Campaign("a", truthful(), budget=None, base_value=5)
    with pytest.raises(ValueError, match="PidPacing needs a number budget"):
        PidPacing().on_episode_start(campaign)
