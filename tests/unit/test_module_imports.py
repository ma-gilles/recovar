import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


SKIP_IMPORT_MODULES = {
    # CLI entrypoint packages are tested elsewhere.
    "__init__",
    "_version",
}

# Optional dependency packages used by some modules.
OPTIONAL_IMPORT_ERROR_TOKENS = (
    "matplotlib_scalebar",
    "dataframe_image",
)


def _module_names():
    root = Path(__file__).resolve().parents[2] / "recovar"
    return sorted(p.stem for p in root.glob("*.py") if p.stem not in SKIP_IMPORT_MODULES)


@pytest.mark.parametrize("mod_name", _module_names())
def test_all_top_level_modules_import(mod_name):
    full_name = f"recovar.{mod_name}"
    try:
        importlib.import_module(full_name)
    except ImportError as e:
        msg = str(e).lower()
        if any(tok in msg for tok in OPTIONAL_IMPORT_ERROR_TOKENS):
            pytest.skip(f"optional dependency missing while importing {full_name}: {e}")
        raise
