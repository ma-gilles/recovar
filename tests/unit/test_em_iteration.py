import pytest

from recovar import em

pytestmark = pytest.mark.unit


def test_em_module_exports_expected_symbols():
    for name in ["EMState", "SGDState", "HeterogeneousEMState", "E_M_batches_2", "split_E_M_v2"]:
        assert hasattr(em, name)
