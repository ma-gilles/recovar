"""The vectorized per-image hypothesis preparation must equal the per-image loop exactly."""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import sparse_bucket_arrays as sba
from recovar.em.dense_single_volume.helpers.significant_samples import ComplementSignificantSampleIndices

pytestmark = [pytest.mark.unit]


def _fine_grid(rng, n_coarse_rot, children_per_parent, dtype):
    n_fine = n_coarse_rot * children_per_parent
    parent = np.repeat(np.arange(n_coarse_rot), children_per_parent).astype(np.int64)
    rots = rng.standard_normal((n_fine, 3, 3)).astype(dtype)
    mstep = rng.standard_normal((n_fine, 3, 3)).astype(dtype)
    eulers = rng.standard_normal((n_fine, 3)).astype(np.float64)
    return rots, mstep, parent, eulers


def _samples(rng, n_images, n_coarse_rot, n_coarse_trans, *, with_specials):
    total = n_coarse_rot * n_coarse_trans
    out = []
    for i in range(n_images):
        if with_specials and i % 7 == 3:
            out.append(None)
        elif with_specials and i % 7 == 5:
            out.append(np.zeros(0, dtype=np.int32))
        elif with_specials and i % 11 == 4:
            excluded = rng.choice(total, size=rng.integers(1, 6), replace=False).astype(np.int32)
            out.append(ComplementSignificantSampleIndices(excluded_indices=excluded, total_size=total))
        else:
            k = int(rng.integers(1, 40))
            out.append(np.sort(rng.choice(total, size=k, replace=False)).astype(np.int32))
    return out


def _run(monkeypatch, enabled, samples, grid, *, n_coarse_rot, n_coarse_trans, n_fine_trans, ftp, prior, mstep, eulers, dtype):
    rots, mstep_rots, parent, source_eulers = grid
    monkeypatch.setenv(sba.VECTORIZED_HYPOTHESIS_PREP_ENV, "1" if enabled else "0")
    return sba._prepare_per_image_pass2_inputs(
        samples,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        nside_level=1,
        oversampling_order=1,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=ftp,
        rotation_log_prior=prior,
        random_perturbation=0.0,
        fine_source_eulers_override=source_eulers if eulers else None,
        fine_rotations_override=rots,
        fine_mstep_rotations_override=mstep_rots if mstep else None,
        fine_rotation_parent_override=parent,
        dtype=dtype,
    )


_PER_IMAGE_KEYS = ("source_eulers", "oversampled_rots", "oversampled_mstep_rots", "parent_map", "oversampled_rot_indices", "unique_rot", "log_prior", "candidate_mask")


def _assert_same(loop, fast):
    assert loop.keys() == fast.keys()
    n = len(loop["oversampled_rots"])
    for key in _PER_IMAGE_KEYS:
        assert len(fast[key]) == n, key
    for key in ("rotation_table", "mstep_rotation_table"):
        if loop[key] is None:
            assert fast[key] is None
        else:
            np.testing.assert_array_equal(loop[key], fast[key])
            assert loop[key].dtype == fast[key].dtype
    for i in range(n):
        for key in ("oversampled_rots", "oversampled_mstep_rots", "parent_map", "oversampled_rot_indices", "unique_rot", "log_prior"):
            a, b = loop[key][i], fast[key][i]
            assert a.dtype == b.dtype, (key, i, a.dtype, b.dtype)
            np.testing.assert_array_equal(a, b, err_msg=f"{key}[{i}]")
        if loop["source_eulers"][i] is None:
            assert fast["source_eulers"][i] is None
        else:
            np.testing.assert_array_equal(loop["source_eulers"][i], fast["source_eulers"][i])
        # A shared M-step grid is signalled by object identity downstream.
        assert (loop["oversampled_mstep_rots"][i] is loop["oversampled_rots"][i]) == (
            fast["oversampled_mstep_rots"][i] is fast["oversampled_rots"][i]
        )
        ma, mb = loop["candidate_mask"][i], fast["candidate_mask"][i]
        assert ma.mode == mb.mode and ma.n_rows == mb.n_rows and ma.n_fine_trans == mb.n_fine_trans and ma.count == mb.count
        for field in ("parent_map", "coarse_valid", "coarse_excluded", "fine_translation_parent"):
            va, vb = getattr(ma, field), getattr(mb, field)
            if va is None:
                assert vb is None, field
            else:
                assert va.dtype == vb.dtype, field
                np.testing.assert_array_equal(va, vb, err_msg=f"mask.{field}[{i}]")
        np.testing.assert_array_equal(np.asarray(ma), np.asarray(mb))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("mstep", [False, True])
@pytest.mark.parametrize("with_specials", [False, True])
def test_vectorized_hypothesis_prep_matches_loop(monkeypatch, dtype, prior, mstep, with_specials):
    rng = np.random.default_rng(20260913 + int(prior) + 2 * int(mstep) + 4 * int(with_specials))
    n_coarse_rot, n_coarse_trans, children = 48, 5, 8
    n_fine_trans = 21
    ftp = rng.integers(0, n_coarse_trans, size=n_fine_trans).astype(np.int32)
    grid = _fine_grid(rng, n_coarse_rot, children, dtype)
    samples = _samples(rng, 60, n_coarse_rot, n_coarse_trans, with_specials=with_specials)
    prior_np = rng.standard_normal(n_coarse_rot).astype(dtype) if prior else None
    kwargs = dict(n_coarse_rot=n_coarse_rot, n_coarse_trans=n_coarse_trans, n_fine_trans=n_fine_trans, ftp=ftp, prior=prior_np, mstep=mstep, eulers=True, dtype=dtype)
    loop = _run(monkeypatch, False, samples, grid, **kwargs)
    fast = _run(monkeypatch, True, samples, grid, **kwargs)
    _assert_same(loop, fast)


def test_vectorized_hypothesis_prep_matches_loop_with_duplicate_significant_indices(monkeypatch):
    """Duplicate significant indices count once, as in the loop's boolean coarse table."""

    rng = np.random.default_rng(9)
    n_coarse_rot, n_coarse_trans, children = 2, 2, 1
    grid = _fine_grid(rng, n_coarse_rot, children, np.float32)
    ftp = np.array([0, 1], dtype=np.int32)
    samples = [np.array(s, dtype=np.int32) for s in ([0], [0, 0], [0, 1, 0], [3, 0, 3], [1, 1, 1, 2])]
    kwargs = dict(n_coarse_rot=n_coarse_rot, n_coarse_trans=n_coarse_trans, n_fine_trans=2, ftp=ftp, prior=None, mstep=False, eulers=False, dtype=np.float32)
    loop = _run(monkeypatch, False, samples, grid, **kwargs)
    fast = _run(monkeypatch, True, samples, grid, **kwargs)
    assert [m.count for m in loop["candidate_mask"]] == [1, 1, 2, 2, 2]  # one fine row per parent, one fine translation per coarse translation
    _assert_same(loop, fast)


def test_vectorized_hypothesis_prep_falls_back_when_grid_is_not_parent_major(monkeypatch):
    rng = np.random.default_rng(5)
    n_coarse_rot, n_coarse_trans, children = 12, 3, 4
    rots, mstep_rots, parent, eulers = _fine_grid(rng, n_coarse_rot, children, np.float32)
    perm = rng.permutation(parent.shape[0])
    grid = (rots[perm], mstep_rots[perm], parent[perm], eulers[perm])
    ftp = rng.integers(0, n_coarse_trans, size=9).astype(np.int32)
    samples = _samples(rng, 20, n_coarse_rot, n_coarse_trans, with_specials=False)
    kwargs = dict(n_coarse_rot=n_coarse_rot, n_coarse_trans=n_coarse_trans, n_fine_trans=9, ftp=ftp, prior=None, mstep=False, eulers=False, dtype=np.float32)
    assert sba._fine_children_ranges(grid[2], n_coarse_rot) is None
    loop = _run(monkeypatch, False, samples, grid, **kwargs)
    fast = _run(monkeypatch, True, samples, grid, **kwargs)
    _assert_same(loop, fast)


def test_vectorized_hypothesis_prep_flag_is_strict(monkeypatch):
    monkeypatch.setenv(sba.VECTORIZED_HYPOTHESIS_PREP_ENV, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        sba.vectorized_hypothesis_prep_enabled()
    monkeypatch.delenv(sba.VECTORIZED_HYPOTHESIS_PREP_ENV)
    assert sba.vectorized_hypothesis_prep_enabled() is False


def test_resident_hypothesis_tables_match_host_device_index_build(monkeypatch):
    """The device index builder fed from the resident flat tables equals the host-built
    (images, cR, cT)/(images, rows) path bitwise, including capacity rows and the
    fall-back for chunks that contain a non-vectorized image."""

    jax = pytest.importorskip("jax")
    rng = np.random.default_rng(31)
    n_coarse_rot, n_coarse_trans, children = 24, 4, 3
    n_fine_trans = 10
    ftp = rng.integers(0, n_coarse_trans, size=n_fine_trans).astype(np.int32)
    grid = _fine_grid(rng, n_coarse_rot, children, np.float32)
    samples = _samples(rng, 30, n_coarse_rot, n_coarse_trans, with_specials=True)
    kwargs = dict(n_coarse_rot=n_coarse_rot, n_coarse_trans=n_coarse_trans, n_fine_trans=n_fine_trans, ftp=ftp, prior=None, mstep=False, eulers=False, dtype=np.float32)
    fast = _run(monkeypatch, True, samples, grid, **kwargs)
    assert fast["resident_hypothesis"] is not None
    positions = fast["resident_hypothesis"]["positions"]
    vectorized_images = np.flatnonzero(positions >= 0)
    special_images = np.flatnonzero(positions < 0)
    assert vectorized_images.size >= 8 and special_images.size >= 1
    monkeypatch.setenv(sba.COMPACT_PAIR_DEVICE_INDEX_ENV, "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_LAZY_TABLES", "1")
    chunk = np.sort(vectorized_images[:6])
    pair_bucket_size = 1 << int(max(m.count for m in fast["candidate_mask"]) - 1).bit_length()
    with jax.default_device(jax.devices("cpu")[0]):
        for image_indices in (chunk, np.concatenate([chunk[:3], special_images[:1]])):
            bucket = {"pair_bucket_size": pair_bucket_size, "image_indices": image_indices}
            monkeypatch.setenv(sba.RESIDENT_HYPOTHESIS_TABLES_ENV, "0")
            host = sba._build_compact_pair_bucket_arrays_from_per_image_inputs(bucket, fast, capacity_rows=8)
            monkeypatch.setenv(sba.RESIDENT_HYPOTHESIS_TABLES_ENV, "1")
            resident = sba._build_compact_pair_bucket_arrays_from_per_image_inputs(bucket, fast, capacity_rows=8)
            for key in ("pair_counts", "local_rotation_row", "translation_idx", "pair_mask", "image_indices"):
                a, b = np.asarray(host[key]), np.asarray(resident[key])
                assert a.dtype == b.dtype, key
                np.testing.assert_array_equal(a, b, err_msg=key)
            assert np.asarray(resident["local_rotation_row"]).shape == (8, pair_bucket_size)


def test_resident_row_indices_gather_matches_index_array_build(monkeypatch):
    """Gathering the padded rotations from the resident per-row index table must equal the
    build that uploads an (images, rows) index array, bitwise, including identity padding,
    capacity rows and the shared-M-step identity; a chunk containing a loop-prepared image
    falls back to the index-array build."""

    jax = pytest.importorskip("jax")
    rng = np.random.default_rng(41)
    n_coarse_rot, n_coarse_trans, children = 24, 4, 3
    n_fine_trans = 10
    ftp = rng.integers(0, n_coarse_trans, size=n_fine_trans).astype(np.int32)
    grid = _fine_grid(rng, n_coarse_rot, children, np.float32)
    samples = _samples(rng, 26, n_coarse_rot, n_coarse_trans, with_specials=True)
    kwargs = dict(n_coarse_rot=n_coarse_rot, n_coarse_trans=n_coarse_trans, n_fine_trans=n_fine_trans,
                  ftp=ftp, prior=None, mstep=True, eulers=False, dtype=np.float32)
    fast = _run(monkeypatch, True, samples, grid, **kwargs)
    resident = fast["resident_hypothesis"]
    assert resident is not None and resident.get("rot_indices_flat") is not None
    positions = resident["positions"]
    vectorized = np.flatnonzero(positions >= 0)
    special = np.flatnonzero(positions < 0)
    assert vectorized.size >= 6 and special.size >= 1

    monkeypatch.setenv(sba.ROTATIONS_BY_INDEX_ENV, "1")
    bucket_size = int(max(int(np.asarray(r).shape[0]) for r in fast["oversampled_rots"]))
    with jax.default_device(jax.devices("cpu")[0]):
        for image_indices, expect_resident in ((np.sort(vectorized[:5]), True),
                                               (np.concatenate([vectorized[:3], special[:1]]), False)):
            bucket = {"bucket_size": bucket_size, "image_indices": image_indices}
            monkeypatch.setenv(sba.RESIDENT_HYPOTHESIS_TABLES_ENV, "0")
            by_index = sba._build_bucket_arrays(bucket, fast, n_fine_trans, capacity_rows=8, device_rotations=True)
            monkeypatch.setenv(sba.RESIDENT_HYPOTHESIS_TABLES_ENV, "1")
            resident_build = sba._build_bucket_arrays(bucket, fast, n_fine_trans, capacity_rows=8, device_rotations=True)
            for key in ("rotations", "mstep_rotations"):
                a, b = np.asarray(by_index[key]), np.asarray(resident_build[key])
                assert a.dtype == b.dtype, key
                np.testing.assert_array_equal(b, a, err_msg=f"{key} (resident expected={expect_resident})")
            assert (resident_build["mstep_rotations"] is resident_build["rotations"]) == (
                by_index["mstep_rotations"] is by_index["rotations"])
            np.testing.assert_array_equal(np.asarray(resident_build["rotations"])[len(image_indices):],
                                          np.broadcast_to(np.eye(3, dtype=np.float32), (8 - len(image_indices), bucket_size, 3, 3)))
