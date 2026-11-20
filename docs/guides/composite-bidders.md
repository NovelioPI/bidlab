# Composite bidders

This guide shows how one bidder can hold other bidders, its children, and call them with its own logic.

Analogy: a PyTorch module that holds submodules.

## Children

A bidder can hold children in attributes, lists or tuples. Call each child's `forward` yourself, with any extra keyword arguments it takes. `bl.Stack` is one such bidder.

The simulator does the rest:

- Each child gets its own `self.rng`.
- Each child gets `on_episode_start`, `update_batch` and `on_episode_end`, parent first.

Do not call those hooks on a child yourself, or it learns twice. Create children in `__init__`: a child added during a run raises an error.

```python
import bidlab as bl


class Shade(bl.Bidder):
    def forward(self, value, request, factor=1.0):
        return value * factor


class NightShade(bl.Bidder):
    """Shades harder at night, then paces."""

    def __init__(self):
        self.shade = Shade()
        self.pacing = bl.PidPacing()

    def on_episode_start(self, brief):
        self.brief = brief

    def forward(self, value, request):
        factor = 0.6 if self.brief.hour_of_day(request.time) < 6 else 0.9
        shaded = self.shade.forward(value, request, factor=factor)
        return self.pacing.forward(shaded, request)
```

## Paths

Each bidder in a tree has a path. `named_bidders()` lists them, parent first.

```python
night = NightShade()
print([path for path, _ in night.named_bidders()])  # ['', 'shade', 'pacing']
print([path for path, _ in bl.Stack(bl.Linear(), bl.PidPacing()).named_bidders()])
# ['', 'bidders.0', 'bidders.1']
```

A child's logged names carry its path, such as `"pacing.multiplier"`. See [Logging](logging.md).

```python
config = bl.SimConfig(
    campaigns=[bl.Campaign("night", night, budget=15, event_value=5)],
    episodes=2,
    steps=200,
)
print(bl.run(config).logged["pacing.multiplier"]["night"].shape)  # (2, 200)
```

## Examples

| Script | Question |
|---|---|
| [composite/night_shade.py](../../examples/composite/night_shade.py) | When a parent shades harder at night, does its child PID raise its multiplier at night? |
