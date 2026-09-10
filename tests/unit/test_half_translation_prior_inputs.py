"""Per-half RELION translation prior inputs have one owner in ``orientation_priors``.

The regular iterations and the final all-data pass previously repeated the
``pdf_offset`` / ``wsum_sigma2_offset`` center construction, the cold-start
engine center and the prior-grid selection inline.
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import orientation_priors as op

pytestmark = pytest.mark.unit

VOXEL = 1.7
BASE = np.asarray([[-1.0, -1.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)


def _build(previous, *, current=BASE, base=BASE, dtype=np.float32):
    return op.relion_half_translation_prior_inputs(
        previous, voxel_size=VOXEL, base_translations=base, current_translations=current, dtype=dtype
    )


def _same(a, b):
    assert type(a) is type(b) and a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cold_start_uses_flat_score_prior_and_zero_engine_center(dtype):
    inputs = _build(None, dtype=dtype)
    assert inputs.prior_center is None and inputs.local_prior_center is None and inputs.sigma_center is None
    _same(inputs.engine_prior_center, np.zeros(2, dtype=dtype))
    _same(inputs.prior_translations, np.asarray(BASE, dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_centers_match_the_separate_relion_formulas(dtype):
    previous = np.asarray([[0.6, -1.4], [2.5, 0.49], [-0.5, 0.5]], dtype=np.float64)
    inputs = _build(previous, dtype=dtype)
    _same(inputs.prior_center, op.relion_translation_prior_center(previous, VOXEL, dtype=dtype))
    _same(inputs.local_prior_center, inputs.prior_center)
    assert inputs.local_prior_center is not inputs.prior_center
    assert not np.shares_memory(inputs.local_prior_center, inputs.prior_center)
    _same(inputs.sigma_center, op.relion_sigma_offset_prior_center(previous, dtype=dtype))
    assert inputs.engine_prior_center is inputs.sigma_center
    assert previous.tolist() == [[0.6, -1.4], [2.5, 0.49], [-0.5, 0.5]]


def test_prior_translations_use_base_grid_when_sizes_match():
    current = BASE + 0.25
    inputs = _build(None, current=current)
    _same(inputs.prior_translations, np.asarray(BASE, dtype=np.float32))


def test_single_current_translation_selects_central_base_translation():
    inputs = _build(None, current=np.zeros((1, 2)))
    _same(inputs.prior_translations, np.asarray(BASE[2:3], dtype=np.float32))


def test_other_size_mismatch_uses_current_grid():
    current = np.asarray([[0.5, 0.5], [1.5, -0.5], [2.0, 2.0]], dtype=np.float64)
    inputs = _build(None, current=current)
    _same(inputs.prior_translations, np.asarray(current, dtype=np.float32))


def test_single_current_and_single_base_translation_use_the_base_grid():
    base = np.asarray([[0.0, 0.0]], dtype=np.float64)
    inputs = _build(None, current=np.zeros((1, 2)), base=base)
    _same(inputs.prior_translations, np.asarray(base, dtype=np.float32))
