"""Tests for the device-compacted coarse significance path (T13, CPU only).

Ticket: `em_parity_tickets_20260918/T13_device_significance_compaction.md`.

The host path is the oracle in every test here: the compacted CSR must equal
`compact_significant_sample_indices_from_mask` per image bitwise, and the
candidate tables built from that CSR must equal, field by field, the tables
built through `_prepare_per_image_pass2_inputs` for the same support.
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.scoring.significant_samples import (
    compact_significant_sample_indices_from_mask,
)
from recovar.em.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs
from recovar.em.sparse_pass2.resident_candidates import build_resident_candidate_tables
from recovar.em.sparse_pass2.resident_significance import (
    CoarseSignificanceCSR,
    DeviceCompactedSignificantSamples,
    build_coarse_significance_csr,
    build_resident_candidate_tables_from_csr,
    compact_batch_significance,
    csr_capacity_for_total,
    host_support_rows,
)

pytestmark = pytest.mark.unit

# Override fixture: a fine rotation grid given explicitly, as the adaptive
# K-class route supplies it (k_class.py passes fine_rotations_override and
# fine_rotation_parent_override into pass 2).
N_COARSE_ROT = 16
CHILDREN = 4
N_COARSE_TRANS = 5
N_FINE_TRANS = 15
FINE_TRANS_PARENT = np.repeat(np.arange(N_COARSE_TRANS, dtype=np.int32), 3)

# Standard fixture: no override, so both paths call the sampling generator.
# healpix level 0 has 12 pixels and 6 in-plane angles.
STD_NSIDE_LEVEL = 0
STD_N_COARSE_ROT = 72
STD_OVERSAMPLING = 1


def _z_rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _fine_rotation_override():
    """A fine grid whose parents are deliberately not in ascending order."""

    parent = np.concatenate(
        [
            np.repeat(np.arange(N_COARSE_ROT // 2, N_COARSE_ROT, dtype=np.int64), CHILDREN),
            np.repeat(np.arange(0, N_COARSE_ROT // 2, dtype=np.int64), CHILDREN),
        ],
    )
    rotations = np.stack([_z_rotation(0.01 * k) for k in range(parent.size)]).astype(np.float32)
    return rotations, parent


def _supports(n_images: int, n_samples: int, seed: int = 13) -> list[np.ndarray]:
    """Per-image significant coarse cell ids: varied sizes, one empty image."""

    rng = np.random.default_rng(seed)
    supports = []
    for image in range(n_images):
        if image == 1:
            supports.append(np.zeros(0, dtype=np.int32))
            continue
        count = int(rng.integers(1, max(2, n_samples // 4)))
        ids = rng.choice(n_samples, size=count, replace=False)
        supports.append(np.sort(ids).astype(np.int32))
    return supports


def _mask_from_supports(supports, n_samples: int) -> np.ndarray:
    mask = np.zeros((len(supports), n_samples), dtype=bool)
    for row, ids in enumerate(supports):
        mask[row, np.asarray(ids, dtype=np.int64)] = True
    return mask


def _csr_from_supports(supports, *, n_coarse_rot, n_coarse_trans) -> CoarseSignificanceCSR:
    counts = np.asarray([np.asarray(s).size for s in supports], dtype=np.int32)
    ids = (
        np.concatenate([np.asarray(s, dtype=np.int32) for s in supports])
        if supports
        else np.zeros(0, dtype=np.int32)
    )
    return build_coarse_significance_csr(
        n_images=len(supports),
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        counts_per_batch=[counts],
        ids_per_batch=[ids],
    )


# --- The device compaction reproduces the host encoding --------------------


def test_compaction_matches_flatnonzero_per_image_bitwise():
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(9, n_samples)
    mask = _mask_from_supports(supports, n_samples)
    counts_host = mask.sum(axis=1).astype(np.int32)

    counts, ids, rot_any = compact_batch_significance(
        mask,
        actual_batch_size=mask.shape[0],
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        batch_n_sig=counts_host,
    )
    np.testing.assert_array_equal(counts, counts_host)
    csr = build_coarse_significance_csr(
        n_images=mask.shape[0],
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        counts_per_batch=[counts],
        ids_per_batch=[ids],
    )
    for image in range(mask.shape[0]):
        np.testing.assert_array_equal(
            csr.image_ids(image),
            np.flatnonzero(mask[image]).astype(np.int32),
        )
    np.testing.assert_array_equal(
        rot_any,
        mask.reshape(mask.shape[0], N_COARSE_ROT, N_COARSE_TRANS).any(axis=(0, 2)),
    )


def test_compaction_ignores_padded_image_rows():
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(6, n_samples)
    mask = _mask_from_supports(supports, n_samples)
    actual = 4
    counts_host = mask[:actual].sum(axis=1).astype(np.int32)

    counts, ids, _rot_any = compact_batch_significance(
        mask,
        actual_batch_size=actual,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        batch_n_sig=mask.sum(axis=1).astype(np.int32),
    )
    np.testing.assert_array_equal(counts, counts_host)
    assert ids.size == int(counts_host.sum())
    csr = build_coarse_significance_csr(
        n_images=actual,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        counts_per_batch=[counts],
        ids_per_batch=[ids],
    )
    for image in range(actual):
        np.testing.assert_array_equal(
            csr.image_ids(image),
            np.flatnonzero(mask[image]).astype(np.int32),
        )


def test_host_support_rows_match_the_host_encoder():
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(7, n_samples)
    mask = _mask_from_supports(supports, n_samples)
    csr = _csr_from_supports(
        supports,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
    )
    rows = host_support_rows(csr)
    for image in range(mask.shape[0]):
        expected = compact_significant_sample_indices_from_mask(mask[image])
        np.testing.assert_array_equal(np.asarray(rows[image]), np.asarray(expected))
        assert np.asarray(rows[image]).dtype == np.int32


def test_compaction_refuses_the_complement_regime():
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    mask = np.zeros((2, n_samples), dtype=bool)
    mask[0, : n_samples // 2 + 1] = True
    with pytest.raises(NotImplementedError, match="complement"):
        compact_batch_significance(
            mask,
            actual_batch_size=2,
            n_coarse_rot=N_COARSE_ROT,
            n_coarse_trans=N_COARSE_TRANS,
            batch_n_sig=mask.sum(axis=1).astype(np.int32),
        )


def test_capacity_ladder_is_power_of_two_and_covers_the_total():
    for total in (0, 1, 4095, 4096, 4097, 100000):
        capacity = csr_capacity_for_total(total)
        assert capacity >= max(total, 4096)
        assert capacity & (capacity - 1) == 0


def test_support_list_carries_its_csr_and_stays_a_list():
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(5, n_samples)
    csr = _csr_from_supports(
        supports,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
    )
    rows = DeviceCompactedSignificantSamples(host_support_rows(csr), csr=csr)
    assert isinstance(rows, list)
    assert len(rows) == len(supports)
    for image, ids in enumerate(rows):
        np.testing.assert_array_equal(ids, supports[image])
    assert rows.csr is csr
    with pytest.raises(ValueError, match="CSR covers"):
        DeviceCompactedSignificantSamples(rows[:-1], csr=csr)


# --- The candidate tables equal the host path's, field by field ------------


def _assert_tables_equal(got, expected):
    assert got.n_images == expected.n_images
    assert got.n_rows == expected.n_rows
    assert got.n_fine_trans == expected.n_fine_trans
    assert got.n_coarse_trans == expected.n_coarse_trans
    for name in (
        "row_offsets",
        "row_image",
        "row_fine_rot",
        "row_parent_local",
        "mask_mode",
        "parent_offsets",
        "parent_trans_bits",
    ):
        got_value = getattr(got, name)
        expected_value = getattr(expected, name)
        assert got_value.dtype == expected_value.dtype, name
        np.testing.assert_array_equal(got_value, expected_value, err_msg=name)
    np.testing.assert_array_equal(got.row_log_prior, expected.row_log_prior)


@pytest.mark.parametrize("execution_order", [False, True])
def test_tables_from_csr_match_the_host_path_with_a_fine_grid_override(execution_order):
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(11, n_samples)
    fine_rotations, fine_parent = _fine_rotation_override()
    rotation_log_prior = np.linspace(-2.0, 2.0, N_COARSE_ROT, dtype=np.float32)

    per_image_inputs = _prepare_per_image_pass2_inputs(
        supports,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=0,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=execution_order,
        dtype=np.float32,
    )
    expected = build_resident_candidate_tables(
        per_image_inputs,
        n_coarse_trans=N_COARSE_TRANS,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
    )
    csr = _csr_from_supports(
        supports,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
    )
    got = build_resident_candidate_tables_from_csr(
        csr,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=0,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=0.0,
        fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=execution_order,
        dtype=np.float32,
    )
    _assert_tables_equal(got, expected)


@pytest.mark.parametrize("execution_order", [False, True])
def test_tables_from_csr_match_the_host_path_on_the_generated_grid(execution_order):
    n_samples = STD_N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(9, n_samples, seed=7)
    rotation_log_prior = np.linspace(-1.0, 1.0, STD_N_COARSE_ROT, dtype=np.float32)

    per_image_inputs = _prepare_per_image_pass2_inputs(
        supports,
        n_coarse_rot=STD_N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=STD_OVERSAMPLING,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=0.0,
        relion_parent_execution_order=execution_order,
        dtype=np.float32,
    )
    expected = build_resident_candidate_tables(
        per_image_inputs,
        n_coarse_trans=N_COARSE_TRANS,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
    )
    csr = _csr_from_supports(
        supports,
        n_coarse_rot=STD_N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
    )
    got = build_resident_candidate_tables_from_csr(
        csr,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=STD_OVERSAMPLING,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=0.0,
        relion_parent_execution_order=execution_order,
        dtype=np.float32,
    )
    _assert_tables_equal(got, expected)


def test_tables_from_csr_match_the_host_path_without_a_rotation_prior():
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(6, n_samples, seed=3)
    fine_rotations, fine_parent = _fine_rotation_override()

    per_image_inputs = _prepare_per_image_pass2_inputs(
        supports,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=0,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=True,
        dtype=np.float32,
    )
    expected = build_resident_candidate_tables(
        per_image_inputs,
        n_coarse_trans=N_COARSE_TRANS,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
    )
    csr = _csr_from_supports(
        supports,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
    )
    got = build_resident_candidate_tables_from_csr(
        csr,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=0,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=True,
        dtype=np.float32,
    )
    _assert_tables_equal(got, expected)


def test_tables_from_csr_refuse_the_complement_regime():
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = [np.arange(n_samples // 2 + 1, dtype=np.int32)]
    csr = _csr_from_supports(
        supports,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
    )
    _rotations, fine_parent = _fine_rotation_override()
    with pytest.raises(NotImplementedError, match="complement"):
        build_resident_candidate_tables_from_csr(
            csr,
            nside_level=STD_NSIDE_LEVEL,
            oversampling_order=0,
            n_fine_trans=N_FINE_TRANS,
            fine_translation_parent=FINE_TRANS_PARENT,
            rotation_log_prior=None,
            random_perturbation=0.0,
            fine_rotation_parent_override=fine_parent,
            relion_parent_execution_order=True,
            dtype=np.float32,
        )


def test_fixture_exercises_both_mask_modes():
    """Guard the fixture: it must cover an empty image and a bitset image."""

    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(11, n_samples)
    _rotations, fine_parent = _fine_rotation_override()
    per_image_inputs = _prepare_per_image_pass2_inputs(
        supports,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=0,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=_fine_rotation_override()[0],
        fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=True,
        dtype=np.float32,
    )
    modes = {mask.mode for mask in per_image_inputs["candidate_mask"]}
    assert modes == {"coarse", "empty"}, (
        f"fixture built modes {modes}; update the fixture, not the assertion"
    )


def test_compaction_fills_an_exactly_full_capacity():
    """A total support equal to the id capacity must not be corrupted.

    Cells outside the support scatter with an out-of-bounds index and are
    dropped.  When the total exactly fills the buffer there is no slack left,
    so this is the case that would expose a dropped index being wrapped to the
    last slot instead.
    """

    n_coarse_rot, n_coarse_trans = 2048, 8
    n_samples = n_coarse_rot * n_coarse_trans
    capacity = csr_capacity_for_total(0)
    rng = np.random.default_rng(5)
    ids = np.sort(rng.choice(n_samples, size=capacity, replace=False)).astype(np.int32)
    mask = np.zeros((2, n_samples), dtype=bool)
    mask[0, ids.astype(np.int64)] = True

    counts, compacted, _rot_any = compact_batch_significance(
        mask,
        actual_batch_size=2,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        batch_n_sig=mask.sum(axis=1).astype(np.int32),
    )
    assert int(counts.sum()) == capacity
    np.testing.assert_array_equal(compacted, ids)
