"""Does a two-stage model on the public features estimate conversions better than counting?

Each estimator is the first stage of a `Stack`: it ignores the platform's value
and outputs its own, `event_value` × its estimated chance that the impression
converts. `Linear` after it bids that value. The market has a click stage, so
an impression converts only after a click.

- `FixedRate` guesses one rate for every impression.
- `CountedRate` counts conversions per win.
- `TwoStageModel` fits one logistic model for clicks and one for conversions
  after a click, on `features["vector"]`.

Each logs its estimate, and the script compares it with the market's truth in
`RunLog.true_values`.
"""

import numpy as np

import bidlab as bl

EVENT_VALUE = 5.0


class FixedRate(bl.Bidder):
    def __init__(self, rate=0.05):
        self.rate = rate

    def forward(self, value):
        self.log("estimate", self.rate)
        return self.rate * EVENT_VALUE


class CountedRate(bl.Bidder):
    def __init__(self, prior=0.05):
        self.prior = prior
        self.wins = 0
        self.conversions = 0

    def forward(self, value):
        rate = (self.conversions + self.prior) / (self.wins + 1)
        self.log("estimate", rate)
        return rate * EVENT_VALUE

    def update(self, result):
        self.wins += result.won
        self.conversions += result.converted


class Logistic:
    """Online logistic regression, one gradient step per example."""

    def __init__(self, size, learning_rate):
        self.weights = np.zeros(size + 1)
        self.learning_rate = learning_rate

    def predict(self, vector):
        return 1.0 / (1.0 + np.exp(-(self.weights[:-1] @ vector + self.weights[-1])))

    def learn(self, vector, label):
        error = label - self.predict(vector)
        self.weights[:-1] += self.learning_rate * error * vector
        self.weights[-1] += self.learning_rate * error


class TwoStageModel(bl.Bidder):
    def __init__(self, feature_count, learning_rate=0.05):
        self.clicks = Logistic(feature_count, learning_rate)
        self.conversions = Logistic(feature_count, learning_rate)
        self.pending = {}

    def forward(self, value, features, time):
        vector = features["vector"]
        # BidResult carries the auction time but not the features.
        self.pending[time] = vector
        rate = self.clicks.predict(vector) * self.conversions.predict(vector)
        self.log("estimate", rate)
        return rate * EVENT_VALUE

    def update(self, result):
        vector = self.pending.pop(result.time)
        if not result.won:
            return

        self.clicks.learn(vector, result.clicked)
        # Only a click can convert, so only clicks teach the conversion stage.
        if result.clicked:
            self.conversions.learn(vector, result.converted)


FEATURES = 4
market = bl.markets.LogisticMarket(
    feature_count=FEATURES, cvr_bias_mean=-1.0, ctr_bias_mean=-1.0
)
estimators = {
    "fixed": FixedRate(),
    "counted": CountedRate(),
    "model": TwoStageModel(FEATURES),
}


def last_episode_error(estimator):
    # One campaign alone wins every auction, so each estimator learns from every impression.
    campaign = bl.Campaign(
        "c", bl.Stack(estimator, bl.Linear()), budget=1e9, event_value=EVENT_VALUE
    )
    config = bl.SimConfig(campaigns=[campaign], market=market, episodes=5, steps=2000)
    log = bl.run(config)
    estimate = log.logged["bidders.0.estimate"]["c"][-1]
    truth = log.true_values[-1, :, 0] / EVENT_VALUE
    return float(np.abs(estimate - truth).mean())


if __name__ == "__main__":
    errors = {name: last_episode_error(bidder) for name, bidder in estimators.items()}
    for name, error in errors.items():
        print(f"{name:>8}: mean absolute error {error:.4f}")
    others = [errors["fixed"], errors["counted"]]
    answer = "yes" if errors["model"] < min(others) else "no"
    print(
        f"Answer: {answer}, the two-stage model's last-episode error is "
        f"{errors['model']:.4f}, against {errors['counted']:.4f} for counting"
    )
