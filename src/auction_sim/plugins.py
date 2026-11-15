import importlib


def resolve[T](name: str, builtins: dict[str, T], kind: str) -> T:
    """Look up a built-in by name, or import `module:attribute`."""
    if name in builtins:
        return builtins[name]
    module_name, separator, attribute = name.partition(":")
    if not separator:
        raise ValueError(
            f"unknown {kind} {name!r}; use one of {sorted(builtins)} or 'module:attribute'"
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError:
        raise ValueError(
            f"{kind} {name!r}: module {module_name!r} has no {attribute!r}"
        ) from None
