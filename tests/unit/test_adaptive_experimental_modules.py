import importlib

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "mod_name",
    [
        "recovar.adaptive_homogeneous",
        "recovar.adaptive_kernel_two",
    ],
)
def test_adaptive_experimental_modules_are_explicitly_unavailable(mod_name):
    mod = importlib.import_module(mod_name)

    assert isinstance(mod.__doc__, str)
    assert "Deprecated experimental" in mod.__doc__
    assert mod.__all__ == ()

    with pytest.raises(AttributeError, match="is unavailable"):
        getattr(mod, "adaptive_disc")
