import importlib

import pytest


@pytest.mark.unit
def test_can_import_lightweight_package_entrypoint():
    mod = importlib.import_module("recovar")
    assert hasattr(mod, "__version__")


@pytest.mark.unit
def test_can_import_fourier_transform_utils_module():
    pytest.importorskip("jax")
    mod = importlib.import_module("recovar.fourier_transform_utils")
    assert hasattr(mod, "get_dft3")
    assert hasattr(mod, "get_idft3")

