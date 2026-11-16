import doctest
import importlib
import inspect
import pkgutil

import pytest

import bidlab as bl

MODULES = sorted(
    info.name for info in pkgutil.walk_packages(bl.__path__, prefix="bidlab.")
)
PUBLIC_OBJECTS = [
    name for name in bl.__all__ if not inspect.ismodule(getattr(bl, name))
]


@pytest.mark.parametrize("name", ["bidlab", *MODULES])
def test_docstring_examples_run(name):
    module = importlib.import_module(name)
    result = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0


@pytest.mark.parametrize("name", PUBLIC_OBJECTS)
def test_public_name_has_example(name):
    docstring = inspect.getdoc(getattr(bl, name)) or ""
    assert "\nExamples\n--------\n" in docstring
