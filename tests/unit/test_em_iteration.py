import pytest

from recovar import EM_iteration as em_iteration
from recovar import em

pytestmark = pytest.mark.unit


def test_em_iteration_shim_reexports_em_symbols():
    # Backward-compatibility contract for legacy imports.
    for name in ["EMState", "SGDState", "HeterogeneousEMState", "E_M_batches_2", "split_E_M_v2"]:
        assert hasattr(em_iteration, name)
        assert getattr(em_iteration, name) is getattr(em, name)
