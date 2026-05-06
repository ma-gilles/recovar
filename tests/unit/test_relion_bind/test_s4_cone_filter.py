"""Phase 5 (S4): exact local-search cone parity against RELION formulas.

The active implementation of ``get_local_rotation_grid_fast`` now mirrors the
non-helical C1 SPA path in
``HealpixSampling::selectOrientationsWithNonZeroPriorProbability``:

- direction cone on ``(rot, tilt)``
- psi cone on ``psi``
- Cartesian product of the two selections
- normalized ``log(direction_prior) + log(psi_prior)``

These tests compare recovar directly against the analytical RELION formulas on
the same HEALPix x psi grid.  The RELION binding does not expose the selector
itself, but it does expose the NEST-ordered direction and psi grids needed to
reproduce the formula exactly.
"""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from recovar.relion_bind._relion_bind_core import (
    get_angular_sampling,
    get_healpix_directions,
)

from recovar.em.sampling import (
    get_local_rotation_grid_fast,
    rotation_grid_size,
    rotation_indices_to_relion_eulers,
)
from recovar.utils.helpers import R_from_relion


def _relion_direction_vector(rot_deg, tilt_deg):
    rot = np.deg2rad(rot_deg)
    tilt = np.deg2rad(tilt_deg)
    return np.array(
        [
            np.sin(tilt) * np.cos(rot),
            np.sin(tilt) * np.sin(rot),
            np.cos(tilt),
        ],
        dtype=np.float64,
    )


def _wrapped_abs_diff_deg(values_deg, ref_deg):
    diff = np.abs(np.asarray(values_deg, dtype=np.float64) - float(ref_deg))
    return np.where(diff > 180.0, np.abs(diff - 360.0), diff)


def _normalized_log_weights(diff_deg, sigma_deg):
    if sigma_deg <= 0.0:
        return np.full(diff_deg.shape, -np.log(max(diff_deg.size, 1)), dtype=np.float32)
    weights = np.exp(-0.5 * (np.asarray(diff_deg, dtype=np.float64) / float(sigma_deg)) ** 2)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        weights.fill(1.0 / max(weights.size, 1))
    else:
        weights /= total
    return np.log(np.clip(weights, np.finfo(np.float32).tiny, None)).astype(np.float32)


def _relion_expected_single_prior(
    prior_rot_deg,
    prior_tilt_deg,
    prior_psi_deg,
    healpix_order,
    sigma_rot_deg,
    sigma_psi_deg,
    sigma_cutoff,
):
    """Analytical C1 RELION local-search selection in recovar's RELION grid order."""
    relion_dirs = get_healpix_directions(healpix_order)
    prior_dir = _relion_direction_vector(prior_rot_deg, prior_tilt_deg)
    grid_dirs = np.array([_relion_direction_vector(rot, tilt) for rot, tilt in relion_dirs], dtype=np.float64)
    diffang = np.rad2deg(np.arccos(np.clip(grid_dirs @ prior_dir, -1.0, 1.0)))
    biggest_sigma_deg = float(max(sigma_rot_deg, sigma_rot_deg))
    dir_mask = diffang < float(sigma_cutoff) * biggest_sigma_deg
    selected_idirs = np.flatnonzero(dir_mask).astype(np.int64)
    if selected_idirs.size == 0:
        selected_idirs = np.array([int(np.argmin(diffang))], dtype=np.int64)
        dir_log_prior = np.zeros(1, dtype=np.float32)
    else:
        dir_log_prior = _normalized_log_weights(diffang[selected_idirs], biggest_sigma_deg)

    angular_step = get_angular_sampling(healpix_order)
    n_psi = int(np.ceil(360.0 / angular_step))
    psi_deg = (360.0 / n_psi) * np.arange(n_psi, dtype=np.float64)
    diffpsi = _wrapped_abs_diff_deg(psi_deg, float(np.mod(prior_psi_deg, 360.0)))
    psi_mask = diffpsi < float(sigma_cutoff) * sigma_psi_deg
    selected_ipsi = np.flatnonzero(psi_mask).astype(np.int64)
    if selected_ipsi.size == 0:
        selected_ipsi = np.array([int(np.argmin(diffpsi))], dtype=np.int64)
        psi_log_prior = np.zeros(1, dtype=np.float32)
    else:
        psi_log_prior = _normalized_log_weights(diffpsi[selected_ipsi], sigma_psi_deg)

    n_pixels = healpy.nside2npix(2**healpix_order)

    index_to_log_prior = {}
    for psi_idx, psi_lp in zip(selected_ipsi.tolist(), psi_log_prior.tolist()):
        for nest_pix, dir_lp in zip(selected_idirs.tolist(), dir_log_prior.tolist()):
            index_to_log_prior[int(psi_idx * n_pixels + nest_pix)] = float(psi_lp + dir_lp)

    selected_indices = np.array(sorted(index_to_log_prior), dtype=np.int64)
    log_prior = np.array([index_to_log_prior[int(idx)] for idx in selected_indices], dtype=np.float32)
    return selected_indices, log_prior


def _prior_to_matrix(prior_rot_deg, prior_tilt_deg, prior_psi_deg):
    return R_from_relion(
        np.array([[prior_rot_deg, prior_tilt_deg, prior_psi_deg]], dtype=np.float64)
    )[0]


_PRIORS = [
    (45.0, 30.0, 60.0),
    (0.0, 90.0, 0.0),
    (-120.0, 15.0, 170.0),
    (180.0, 60.0, -90.0),
]
_SIGMA_CUTOFF = 3.0
_SIGMA_ROT_DEG = 10.0
_SIGMA_PSI_DEG = 10.0


class TestExactSinglePriorParity:
    @pytest.mark.parametrize("order", [2, 3])
    @pytest.mark.parametrize("prior_rot,prior_tilt,prior_psi", _PRIORS)
    def test_selected_indices_match_relion_formula(self, order, prior_rot, prior_tilt, prior_psi):
        actual_indices, actual_log_prior = get_local_rotation_grid_fast(
            np.array([[prior_rot, prior_tilt, prior_psi]], dtype=np.float32),
            np.deg2rad(_SIGMA_ROT_DEG),
            np.deg2rad(_SIGMA_PSI_DEG),
            order,
            sigma_cutoff=_SIGMA_CUTOFF,
            per_image=True,
        )
        expected_indices, expected_log_prior = _relion_expected_single_prior(
            prior_rot,
            prior_tilt,
            prior_psi,
            order,
            _SIGMA_ROT_DEG,
            _SIGMA_PSI_DEG,
            _SIGMA_CUTOFF,
        )

        np.testing.assert_array_equal(actual_indices, expected_indices)
        np.testing.assert_allclose(actual_log_prior[0], expected_log_prior, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("order", [2, 3])
    @pytest.mark.parametrize("prior_rot,prior_tilt,prior_psi", _PRIORS)
    def test_rotation_index_input_matches_grid_euler_input(self, order, prior_rot, prior_tilt, prior_psi):
        expected_indices, expected_log_prior = _relion_expected_single_prior(
            prior_rot,
            prior_tilt,
            prior_psi,
            order,
            _SIGMA_ROT_DEG,
            _SIGMA_PSI_DEG,
            _SIGMA_CUTOFF,
        )
        # The highest-weight RELION sample is a valid exact prior index.
        prior_index = int(expected_indices[np.argmax(expected_log_prior)])
        prior_eulers = rotation_indices_to_relion_eulers(np.array([prior_index], dtype=np.int64), order)

        from_indices = get_local_rotation_grid_fast(
            np.array([prior_index], dtype=np.int64),
            np.deg2rad(_SIGMA_ROT_DEG),
            np.deg2rad(_SIGMA_PSI_DEG),
            order,
            sigma_cutoff=_SIGMA_CUTOFF,
            per_image=True,
        )
        from_eulers = get_local_rotation_grid_fast(
            prior_eulers,
            np.deg2rad(_SIGMA_ROT_DEG),
            np.deg2rad(_SIGMA_PSI_DEG),
            order,
            sigma_cutoff=_SIGMA_CUTOFF,
            per_image=True,
        )

        np.testing.assert_array_equal(from_indices[0], from_eulers[0])
        np.testing.assert_allclose(from_indices[1], from_eulers[1], rtol=1e-6, atol=1e-6)


class TestPerImageUnionParity:
    @pytest.mark.parametrize("order", [2, 3])
    def test_union_matches_individual_relion_priors(self, order):
        actual_indices, actual_log_prior = get_local_rotation_grid_fast(
            np.array(_PRIORS[:3], dtype=np.float32),
            np.deg2rad(_SIGMA_ROT_DEG),
            np.deg2rad(_SIGMA_PSI_DEG),
            order,
            sigma_cutoff=_SIGMA_CUTOFF,
            per_image=True,
        )

        expected_parts = [
            _relion_expected_single_prior(
                prior_rot,
                prior_tilt,
                prior_psi,
                order,
                _SIGMA_ROT_DEG,
                _SIGMA_PSI_DEG,
                _SIGMA_CUTOFF,
            )
            for prior_rot, prior_tilt, prior_psi in _PRIORS[:3]
        ]
        expected_union = np.array(
            sorted({int(idx) for indices, _ in expected_parts for idx in indices.tolist()}),
            dtype=np.int64,
        )
        expected_log_prior = np.full((len(expected_parts), expected_union.shape[0]), -1e30, dtype=np.float32)
        union_pos = {int(idx): pos for pos, idx in enumerate(expected_union.tolist())}
        for row, (indices, log_prior) in enumerate(expected_parts):
            positions = np.array([union_pos[int(idx)] for idx in indices.tolist()], dtype=np.int64)
            expected_log_prior[row, positions] = log_prior

        np.testing.assert_array_equal(actual_indices, expected_union)
        np.testing.assert_allclose(actual_log_prior, expected_log_prior, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("order", [2, 3])
    def test_union_indices_are_valid_subset_of_full_grid(self, order):
        actual_indices, _ = get_local_rotation_grid_fast(
            np.array(_PRIORS, dtype=np.float32),
            np.deg2rad(_SIGMA_ROT_DEG),
            np.deg2rad(_SIGMA_PSI_DEG),
            order,
            sigma_cutoff=_SIGMA_CUTOFF,
            per_image=False,
        )
        assert np.all(actual_indices >= 0)
        assert np.all(actual_indices < rotation_grid_size(order))


class TestBindingGap:
    def test_no_select_orientations_binding(self):
        from recovar.relion_bind import _relion_bind_core as core

        assert not hasattr(core, "select_orientations_with_prior")
        assert not hasattr(core, "selectOrientationsWithNonZeroPriorProbability")
