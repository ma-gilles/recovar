import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.core import linalg
from recovar.output import output
from recovar.ppca import ppca as ppca_module
from recovar.reconstruction import regularization

pytestmark = pytest.mark.unit


def test_ppca_modules_import_and_expose_expected_symbols():
    assert hasattr(ppca_module, "EM")
    assert hasattr(ppca_module, "EM_step_half")
    assert callable(output.mkdir_safe)
    assert callable(linalg.batch_linear_solver)


def test_ppca_regularization_helper_matches_current_module_surface():
    values = np.ones((2, 4 * 4 * 4), dtype=np.float32)
    averaged = regularization.batch_average_over_shells(values, (4, 4, 4), 0)
    assert averaged.shape == (2, 1)
