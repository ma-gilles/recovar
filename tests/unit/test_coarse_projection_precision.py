"""Consumer precision must precede projection, not merely narrow scores."""
import numpy as np
import pytest
from recovar.em.scoring.significance import _prepare_coarse_relion_projector
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("classes", [1, 4])
@pytest.mark.parametrize("projection_double", [None, False, True])
@pytest.mark.parametrize("score_double", [False, True])
def test_coarse_projector_precision_preserves_source(classes, projection_double, score_double):
    source = np.arange(classes * 7 * 7 * 4).reshape(classes, 7, 7, 4).astype(np.complex128)
    source += 2**-27 + 1j * 2**-26
    saved = source.copy()
    result = _prepare_coarse_relion_projector(
        source, use_float64_scoring=score_double,
        use_float64_projections=projection_double,
    )
    expected_dtype = np.complex128 if projection_double is None or projection_double or score_double else np.complex64
    assert result.dtype == expected_dtype
    np.testing.assert_array_equal(np.asarray(result), source.astype(expected_dtype))
    np.testing.assert_array_equal(source, saved)


def test_unspecified_projection_precision_preserves_legacy_float32_input():
    source = np.zeros((1, 7, 7, 4), dtype=np.complex64)
    result = _prepare_coarse_relion_projector(source, use_float64_scoring=True, use_float64_projections=None)
    assert result.dtype == np.complex64
