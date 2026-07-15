from __future__ import annotations

import logging
from pathlib import Path
import inspect
import re

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
import recovar.core.slicing as slicing
import recovar.cuda_backproject as cuda_backproject
from recovar.em.dense_single_volume.helpers import half_volume_mstep
from recovar.reconstruction import regularization

pytestmark = pytest.mark.unit


def _random_complex(shape, seed=0, dtype=np.complex64):
    rng = np.random.default_rng(seed)
    real = rng.normal(size=shape).astype(np.float32)
    imag = rng.normal(size=shape).astype(np.float32)
    return (real + 1j * imag).astype(dtype)


def _valid_relion_x_half_grid(volume_shape, seed=0):
    rng = np.random.default_rng(seed)
    real_volume = rng.normal(size=volume_shape).astype(np.float32)
    full = ftu.get_dft3(jnp.asarray(real_volume))
    return np.asarray(ftu.full_volume_to_half_volume(full, volume_shape), dtype=np.complex64)


def _expected_relion_x_half_to_recovar_full(half_grid, volume_shape):
    relion_full = np.asarray(
        ftu.half_volume_to_full_volume(jnp.asarray(half_grid), volume_shape)
    ).reshape(volume_shape)
    return relion_full.transpose(2, 1, 0).reshape(-1)


def test_relion_x_half_native_half_matches_full_expansion():
    volume_shape = (8, 8, 8)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_grid = _valid_relion_x_half_grid(volume_shape, seed=123)

    expected = _expected_relion_x_half_to_recovar_full(half_grid, volume_shape)

    native_half = half_volume_mstep.relion_x_half_volume_to_native_half(
        jnp.asarray(half_grid).reshape(-1),
        volume_shape,
    )
    actual = np.asarray(
        ftu.half_volume_to_full_volume(jnp.asarray(native_half).reshape(half_shape), volume_shape)
    ).reshape(-1)

    assert isinstance(native_half, np.ndarray)
    assert native_half.shape == (int(np.prod(half_shape)),)
    assert native_half.dtype == half_grid.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)


def test_relion_x_half_native_half_matches_odd_bpref_full_expansion():
    volume_shape = (9, 9, 9)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_grid = _valid_relion_x_half_grid(volume_shape, seed=321)

    expected = _expected_relion_x_half_to_recovar_full(half_grid, volume_shape)

    native_half = half_volume_mstep.relion_x_half_volume_to_native_half(
        jnp.asarray(half_grid).reshape(-1),
        volume_shape,
    )
    actual = np.asarray(
        ftu.half_volume_to_full_volume(jnp.asarray(native_half).reshape(half_shape), volume_shape)
    ).reshape(-1)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)


def test_relion_x_half_public_layout_guard_avoids_full_expand_when_forced(monkeypatch):
    volume_shape = (10, 10, 10)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_grid = _valid_relion_x_half_grid(volume_shape, seed=456)
    expected = _expected_relion_x_half_to_recovar_full(half_grid, volume_shape)
    original_half_volume_to_full_volume = ftu.half_volume_to_full_volume

    def fail_half_volume_to_full_volume(*args, **kwargs):
        raise AssertionError("JAX half-volume expansion should not be used")

    monkeypatch.setenv("RECOVAR_RELION_X_HALF_TO_NATIVE_HALF", "1")
    monkeypatch.setattr(
        half_volume_mstep.fourier_transform_utils,
        "half_volume_to_full_volume",
        fail_half_volume_to_full_volume,
    )

    native_half = half_volume_mstep.relion_x_half_volume_to_public_layout(
        jnp.asarray(half_grid).reshape(-1),
        volume_shape,
    )
    actual = np.asarray(original_half_volume_to_full_volume(jnp.asarray(native_half).reshape(half_shape), volume_shape))

    assert native_half.shape == (int(np.prod(half_shape)),)
    np.testing.assert_allclose(actual.reshape(-1), expected, rtol=1e-6, atol=1e-5)


def test_relion_x_half_native_half_threshold_keeps_default_256_padded_grid_off(monkeypatch):
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_TO_NATIVE_HALF", raising=False)
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_TO_NATIVE_HALF_MIN_VOXELS", raising=False)

    assert not half_volume_mstep._large_relion_x_half_to_native_half_enabled(512**3)
    assert half_volume_mstep._large_relion_x_half_to_native_half_enabled(768**3)


def test_relion_x_half_full_host_threshold_enables_default_256_padded_grid(monkeypatch):
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_FULL_HOST", raising=False)
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_FULL_HOST_MIN_VOXELS", raising=False)

    assert not half_volume_mstep._large_relion_x_half_full_host_enabled(259**3)
    assert half_volume_mstep._large_relion_x_half_full_host_enabled(512**3)
    assert half_volume_mstep._large_relion_x_half_full_host_enabled(768**3)


def test_relion_x_half_host_x0_threshold_is_memory_safety_default(monkeypatch):
    monkeypatch.delenv(half_volume_mstep._RELION_X_HALF_HOST_X0_ENV, raising=False)
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_TO_NATIVE_HALF_MIN_VOXELS", raising=False)
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_HOST_X0_MIN_VOXELS", raising=False)
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_TO_NATIVE_HALF", "0")

    assert not half_volume_mstep._large_relion_x_half_host_x0_enabled(512**3)
    assert half_volume_mstep._large_relion_x_half_host_x0_enabled(768**3)

    monkeypatch.setenv(half_volume_mstep._RELION_X_HALF_HOST_X0_ENV, "0")
    assert not half_volume_mstep._large_relion_x_half_host_x0_enabled(768**3)


def test_relion_x_half_mstep_accumulator_dtypes_default_to_relion_acc_float(monkeypatch):
    monkeypatch.delenv(half_volume_mstep._RELION_X_HALF_MSTEP_DOUBLE_ENV, raising=False)

    y_dtype, ctf_dtype = half_volume_mstep.relion_x_half_mstep_accumulator_dtypes(
        np.complex64,
        use_relion_x_half_mstep=True,
    )

    assert y_dtype == np.dtype(np.complex64)
    assert ctf_dtype == np.dtype(np.float32)

    y_dtype, ctf_dtype = half_volume_mstep.relion_x_half_mstep_accumulator_dtypes(
        np.complex128,
        use_relion_x_half_mstep=True,
    )

    assert y_dtype == np.dtype(np.complex64)
    assert ctf_dtype == np.dtype(np.float32)


def test_relion_x_half_mstep_accumulator_dtypes_can_opt_into_double(monkeypatch):
    monkeypatch.setenv(half_volume_mstep._RELION_X_HALF_MSTEP_DOUBLE_ENV, "1")

    y_dtype, ctf_dtype = half_volume_mstep.relion_x_half_mstep_accumulator_dtypes(
        np.complex64,
        use_relion_x_half_mstep=True,
    )

    assert y_dtype == np.dtype(np.complex128)
    assert ctf_dtype == np.dtype(np.float64)


def test_relion_x_half_mstep_accumulator_dtypes_can_opt_out(monkeypatch):
    monkeypatch.setenv(half_volume_mstep._RELION_X_HALF_MSTEP_DOUBLE_ENV, "0")

    y_dtype, ctf_dtype = half_volume_mstep.relion_x_half_mstep_accumulator_dtypes(
        np.complex64,
        use_relion_x_half_mstep=True,
    )

    assert y_dtype == np.dtype(np.complex64)
    assert ctf_dtype == np.dtype(np.float32)


def test_non_relion_mstep_keeps_dataset_accumulator_dtype(monkeypatch):
    monkeypatch.delenv(half_volume_mstep._RELION_X_HALF_MSTEP_DOUBLE_ENV, raising=False)

    y_dtype, ctf_dtype = half_volume_mstep.relion_x_half_mstep_accumulator_dtypes(
        np.complex64,
        use_relion_x_half_mstep=False,
    )

    assert y_dtype == np.dtype(np.complex64)
    assert ctf_dtype == np.dtype(np.complex64)


def test_exact_local_mstep_splits_when_accumulator_dtypes_differ():
    from recovar.em.dense_single_volume.local_big_jit import _exact_local_mstep_should_split_adjoints

    assert _exact_local_mstep_should_split_adjoints(
        (259, 259, 259),
        np.zeros((1, 4), dtype=np.complex128),
        np.zeros((1, 4), dtype=np.float64),
        np.zeros((8,), dtype=np.complex128),
        np.zeros((8,), dtype=np.float64),
    )
    assert not _exact_local_mstep_should_split_adjoints(
        (259, 259, 259),
        np.zeros((1, 4), dtype=np.complex64),
        np.zeros((1, 4), dtype=np.complex64),
        np.zeros((8,), dtype=np.complex64),
        np.zeros((8,), dtype=np.complex64),
    )
    assert _exact_local_mstep_should_split_adjoints(
        (515, 515, 515),
        np.zeros((1, 4), dtype=np.complex64),
        np.zeros((1, 4), dtype=np.complex64),
    )


def test_relion_backprojector_volume_shape_matches_initzeros_formula():
    assert half_volume_mstep.relion_backprojector_volume_shape((128, 128, 128), 2) == (259, 259, 259)
    assert half_volume_mstep.relion_backprojector_volume_shape((128, 128, 128), 2, current_size=60) == (
        123,
        123,
        123,
    )
    assert half_volume_mstep.relion_backprojector_volume_shape((128, 128, 128), 2, current_size=61) == (
        123,
        123,
        123,
    )
    assert half_volume_mstep.relion_backprojector_volume_shape((128, 128, 128), 2, current_size=999) == (
        259,
        259,
        259,
    )
    assert half_volume_mstep.relion_backprojector_volume_shape((16, 16, 16), 1.5, current_size=7) == (
        13,
        13,
        13,
    )


def test_relion_backprojector_volume_shape_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="cubic"):
        half_volume_mstep.relion_backprojector_volume_shape((16, 16, 18), 2)
    with pytest.raises(ValueError, match="positive"):
        half_volume_mstep.relion_backprojector_volume_shape((16, 16, 16), 0)


def test_enforce_relion_x0_hermitian_uses_centered_odd_grid_partner():
    from recovar.em.dense_single_volume.local_backprojection import (
        enforce_relion_half_volume_x0_hermitian,
        enforce_relion_half_volume_x0_hermitian_host,
    )

    volume_shape = (5, 5, 5)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    rng = np.random.default_rng(17)
    original = _random_complex(half_shape, seed=17)
    original[:, :, 1:] = rng.normal(size=original[:, :, 1:].shape).astype(np.float32)

    got = np.asarray(
        enforce_relion_half_volume_x0_hermitian(jnp.asarray(original).reshape(-1), volume_shape)
    ).reshape(half_shape)
    got_host = enforce_relion_half_volume_x0_hermitian_host(jnp.asarray(original).reshape(-1), volume_shape).reshape(
        half_shape
    )

    i0 = np.arange(volume_shape[0])
    i1 = np.arange(volume_shape[1])
    centered_p0 = (volume_shape[0] - (volume_shape[0] % 2) - i0) % volume_shape[0]
    centered_p1 = (volume_shape[1] - (volume_shape[1] % 2) - i1) % volume_shape[1]
    expected_plane = original[:, :, 0] + np.conj(original[np.ix_(centered_p0, centered_p1)][:, :, 0])
    self_partner = (centered_p0[:, None] == i0[:, None]) & (centered_p1[None, :] == i1[None, :])
    expected_plane = np.where(self_partner, original[:, :, 0], expected_plane)
    expected = original.copy()
    expected[:, :, 0] = expected_plane

    unshifted_p0 = (-i0) % volume_shape[0]
    unshifted_p1 = (-i1) % volume_shape[1]
    unshifted_plane = original[:, :, 0] + np.conj(original[np.ix_(unshifted_p0, unshifted_p1)][:, :, 0])

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_host, expected, rtol=1e-6, atol=1e-6)
    assert np.max(np.abs(unshifted_plane - expected_plane)) > 1e-3


def test_enforce_half_volume_x0_uses_host_path_for_large_grids(monkeypatch):
    volume_shape = (6, 6, 6)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    Ft_y = _random_complex(half_shape, seed=91)
    Ft_ctf = _random_complex(half_shape, seed=92)

    expected_y = np.asarray(
        half_volume_mstep.enforce_relion_half_volume_x0_hermitian(jnp.asarray(Ft_y).reshape(-1), volume_shape)
    )
    expected_ctf = np.asarray(
        half_volume_mstep.enforce_relion_half_volume_x0_hermitian(jnp.asarray(Ft_ctf).reshape(-1), volume_shape)
    )

    monkeypatch.setattr(half_volume_mstep, "_large_relion_x_half_host_x0_enabled", lambda full_voxels: True)

    def fail_device_enforcement(*args, **kwargs):
        raise AssertionError("device x0 enforcement should not run for large-grid host path")

    monkeypatch.setattr(half_volume_mstep, "enforce_relion_half_volume_x0_hermitian", fail_device_enforcement)

    got_y, got_ctf = half_volume_mstep.enforce_half_volume_x0(
        jnp.asarray(Ft_y).reshape(-1),
        jnp.asarray(Ft_ctf).reshape(-1),
        volume_shape,
        logger=logging.getLogger(__name__),
        label="unit",
    )

    assert isinstance(got_y, np.ndarray)
    assert isinstance(got_ctf, np.ndarray)
    np.testing.assert_allclose(got_y, expected_y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_ctf, expected_ctf, rtol=1e-6, atol=1e-6)


def test_relion_x_half_production_allocators_use_current_size_backprojector_shape():
    from recovar.em.dense_single_volume import k_class
    from recovar.em.dense_single_volume import local_em_engine
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed

    def assert_uses_current_size_shape(fn):
        source = inspect.getsource(fn)
        assert "relion_backprojector_volume_shape(" in source
        calls = re.findall(r"relion_backprojector_volume_shape\([^)]*\)", source, flags=re.DOTALL)
        assert calls
        assert any(
            "current_size=current_size" in call or 'current_size=common["current_size"]' in call
            for call in calls
        )
        for call in calls:
            if "reconstruction_padding_factor" in call or 'common["reconstruction_padding_factor"]' in call:
                assert "current_size=current_size" in call or 'current_size=common["current_size"]' in call

    assert_uses_current_size_shape(local_em_engine.run_local_em_exact)
    assert_uses_current_size_shape(sparse_pass2_bucketed.compute_pass2_stats_sparse_bucketed)
    assert_uses_current_size_shape(sparse_pass2_bucketed.compute_k_class_pass2_stats_sparse_fused)
    assert_uses_current_size_shape(k_class._run_sparse_k_class_adaptive_pass2)


def test_relion_x_half_public_layout_shape_switch(monkeypatch):
    volume_shape = (4, 4, 4)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_grid = _valid_relion_x_half_grid(volume_shape, seed=789)

    monkeypatch.setenv("RECOVAR_RELION_X_HALF_TO_NATIVE_HALF", "0")
    full = half_volume_mstep.relion_x_half_volume_to_public_layout(
        jnp.asarray(half_grid).reshape(-1),
        volume_shape,
    )
    assert full.shape == (int(np.prod(volume_shape)),)

    monkeypatch.setenv("RECOVAR_RELION_X_HALF_TO_NATIVE_HALF", "1")
    native_half = half_volume_mstep.relion_x_half_volume_to_public_layout(
        jnp.asarray(half_grid).reshape(-1),
        volume_shape,
    )
    assert native_half.shape == (int(np.prod(half_shape)),)


def test_relion_x_half_public_full_layout_uses_host_expand_when_forced(monkeypatch):
    volume_shape = (10, 10, 10)
    half_grid = _valid_relion_x_half_grid(volume_shape, seed=790)
    expected = _expected_relion_x_half_to_recovar_full(half_grid, volume_shape)

    def fail_half_volume_to_full_volume(*args, **kwargs):
        raise AssertionError("JAX half-volume expansion should not be used")

    monkeypatch.setenv("RECOVAR_RELION_X_HALF_TO_NATIVE_HALF", "0")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_FULL_HOST", "1")
    monkeypatch.setattr(
        half_volume_mstep.fourier_transform_utils,
        "half_volume_to_full_volume",
        fail_half_volume_to_full_volume,
    )

    full = half_volume_mstep.relion_x_half_volume_to_public_layout(
        jnp.asarray(half_grid).reshape(-1),
        volume_shape,
    )

    assert isinstance(full, np.ndarray)
    assert full.shape == (int(np.prod(volume_shape)),)
    np.testing.assert_allclose(full, expected, rtol=1e-6, atol=1e-5)


def test_relion_x_half_public_full_tau2_shell_stats_use_relion_x_axis(monkeypatch):
    """Tau2 shell stats must match RELION x-half storage after public-full expansion."""

    monkeypatch.setenv("RECOVAR_RELION_X_HALF_TO_NATIVE_HALF", "0")
    volume_shape = (8, 8, 8)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    rng = np.random.default_rng(112)
    relion_x_half_weight = rng.uniform(0.1, 2.0, size=half_shape).astype(np.float32)
    relion_x_public_full = half_volume_mstep.relion_x_half_volume_to_full(
        jnp.asarray(relion_x_half_weight).reshape(-1),
        volume_shape,
    )
    fsc = np.full(volume_shape[0] // 2 + 1, 0.5, dtype=np.float64)

    _, _, packed_details = regularization.compute_relion_tau2_from_weights(
        relion_x_half_weight.reshape(-1),
        relion_x_half_weight.reshape(-1),
        fsc,
        volume_shape,
        return_details=True,
    )
    _, _, public_full_details = regularization.compute_relion_tau2_from_weights(
        relion_x_public_full,
        relion_x_public_full,
        fsc,
        volume_shape,
        return_details=True,
        full_half_axis=0,
    )
    _, _, wrong_axis_details = regularization.compute_relion_tau2_from_weights(
        relion_x_public_full,
        relion_x_public_full,
        fsc,
        volume_shape,
        return_details=True,
    )

    np.testing.assert_allclose(
        np.asarray(public_full_details["sigma2_shells"]),
        np.asarray(packed_details["sigma2_shells"]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.max(
        np.abs(
            np.asarray(wrong_axis_details["sigma2_shells"])
            - np.asarray(packed_details["sigma2_shells"])
        )
    ) > 1e-2


def test_relion_x_half_backproject_rotation_transform_matches_relion_ainv():
    from recovar.cuda_backproject import (
        _relion_x_half_backproject_rotation_to_kernel,
        _rot_to_compact,
    )

    rng = np.random.default_rng(44)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0

    kernel_matrix = _relion_x_half_backproject_rotation_to_kernel(jnp.asarray(q[None], dtype=jnp.float64))
    rot6 = np.asarray(_rot_to_compact(kernel_matrix), dtype=np.float64)[0]

    for x, y in [(1.0, 2.0), (7.25, -3.5), (-4.0, 6.0)]:
        got = y * rot6[:3] + x * rot6[3:]
        relion_xyz = np.linalg.inv(q) @ np.asarray([x, y, 0.0], dtype=np.float64)
        expected_kernel_zyx = relion_xyz[[2, 1, 0]]
        np.testing.assert_allclose(got, expected_kernel_zyx, rtol=1e-12, atol=1e-12)


def test_relion_x_half_backproject_rotation_uses_inverse_for_nyquist_boundary():
    from recovar.cuda_backproject import (
        _relion_x_half_backproject_rotation_to_kernel,
        _rot_to_compact,
    )

    # RELION calls Matrix2D::inv(); using A.T instead includes the x=Nyquist
    # source pixel for this rotation, adding one spurious unit of BPref weight.
    rotation = np.asarray(
        [
            [-0.20223536142961612, 0.9760677738394581, -0.07995348309809173],
            [-0.7519838250487249, -0.1024667595107726, 0.6511688644740704],
            [0.627392369616113, 0.19181309670469796, 0.7547095802227721],
        ],
        dtype=np.float64,
    )
    kernel_matrix = _relion_x_half_backproject_rotation_to_kernel(jnp.asarray(rotation[None], dtype=jnp.float64))
    rot6 = np.asarray(_rot_to_compact(kernel_matrix), dtype=np.float64)[0]

    source = np.asarray([8.0, 0.0, 0.0], dtype=np.float64)
    got = source[1] * rot6[:3] + source[0] * rot6[3:]
    expected = (np.linalg.inv(rotation) @ source)[[2, 1, 0]]
    transpose_result = (rotation.T @ source)[[2, 1, 0]]

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(got) > np.linalg.norm(transpose_result)


def test_relion_x_half_float_acc_rotation_casts_cpu_orthonormal_inverse():
    from recovar.cuda_backproject import _relion_x_half_backproject_rotation_to_kernel

    rotation = jnp.asarray(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=jnp.float64,
    )
    actual = _relion_x_half_backproject_rotation_to_kernel(rotation, jnp.float32)

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(rotation, dtype=np.float32)[..., [2, 1, 0]],
    )


def test_relion_x_half_indexed_adjoint_preserves_accumulator_dtype(monkeypatch):
    observed = {}

    def fake_backproject_indexed(volume, slices, *args, **kwargs):
        observed["volume_dtype"] = volume.dtype
        observed["slice_dtype"] = slices.dtype
        return volume

    monkeypatch.setattr(slicing, "_use_cuda_backproject", lambda order: True)
    monkeypatch.setattr(cuda_backproject, "backproject_indexed", fake_backproject_indexed)

    result = slicing.adjoint_slice_volume_indexed(
        jnp.ones((1, 1), dtype=jnp.complex128),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.eye(3, dtype=jnp.float64)[None],
        (4, 4),
        (4, 4, 4),
        "linear_interp",
        volume=jnp.zeros(4 * 4 * 3, dtype=jnp.complex64),
        half_image=True,
        half_volume=True,
        relion_x_half=True,
    )

    assert result.dtype == jnp.complex64
    assert observed == {"volume_dtype": jnp.complex64, "slice_dtype": jnp.complex64}


def test_relion_x_half_batched_indexed_adjoint_preserves_accumulator_dtype(monkeypatch):
    observed = {}

    def fake_batch_backproject_indexed(volumes, slices, *args, **kwargs):
        observed["volume_dtype"] = volumes.dtype
        observed["slice_dtype"] = slices.dtype
        return volumes

    monkeypatch.setattr(slicing, "_use_cuda_backproject", lambda order: True)
    monkeypatch.setattr(cuda_backproject, "batch_backproject_indexed", fake_batch_backproject_indexed)

    result = slicing.batch_adjoint_slice_volume_indexed(
        jnp.ones((2, 1, 1), dtype=jnp.complex128),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.eye(3, dtype=jnp.float64)[None],
        (4, 4),
        (4, 4, 4),
        "linear_interp",
        volumes=jnp.zeros((2, 4 * 4 * 3), dtype=jnp.complex64),
        half_image=True,
        half_volume=True,
        relion_x_half=True,
    )

    assert result.dtype == jnp.complex64
    assert observed == {"volume_dtype": jnp.complex64, "slice_dtype": jnp.complex64}


def test_relion_x_half_cuda_skips_fftw_x0_negative_row_duplicate():
    cuda_source = Path(__file__).resolve().parents[2] / "recovar" / "cuda" / "cuda_backproject.cu"
    text = cuda_source.read_text()

    duplicate_x0_guard = "relion_fold_x && HALF_IMG && HALF_VOL && k1_idx == 0 && k0_idx >= image_w"
    assert text.count(duplicate_x0_guard) == 2
    assert "RELION iterates FFTW half-images in native row order" in text
    assert "k0_idx < image_w" in text
    assert "k1_unscaled = (T)k1_idx" in text
    assert "k1 = k1_unscaled * upsampling" in text
    assert "BackProjector::backproject2Dto3D skips" in text


def test_relion_x_half_cuda_rotates_before_applying_padding_factor():
    cuda_source = Path(__file__).resolve().parents[2] / "recovar" / "cuda" / "cuda_backproject.cu"
    text = cuda_source.read_text()

    # Strict RELION backprojection must preserve the operation ordering in
    # cuda_kernel_backproject3D. At outer-shell pixels, distributing the
    # padding multiplication into this float32 dot product can change the
    # redundant rotated-radius decision by one ulp.
    expected_rk0 = "(R[3] * k1_unscaled + R[0] * k0_unscaled) *"
    expected_rk1 = "(R[4] * k1_unscaled + R[1] * k0_unscaled) *"
    expected_rk2 = "(R[5] * k1_unscaled + R[2] * k0_unscaled) *"
    # Indexed, batch, and fused backprojectors plus the C64 and double-output
    # texture projectors all preserve RELION's matrix-x*source-x-first order.
    assert text.count(expected_rk0) == 5
    assert text.count(expected_rk1) == 5
    assert text.count(expected_rk2) == 5
    assert "(k0_unscaled * R[0] + k1_unscaled * R[3]) *" not in text
    assert "matrix-x*source-x first" in text
    assert "Reversing the addends changes CUDA's contracted FMA" in text


def test_relion_x_half_cuda_pins_physical_radius_accumulation_order():
    cuda_source = Path(__file__).resolve().parents[2] / "recovar" / "cuda" / "cuda_backproject.cu"
    text = cuda_source.read_text()

    helper = text[text.index("float relion_radius_squared(") : text.index("#define BLOCK_SIZE")]
    assert "__fmul_rn(rk1, rk1)" in helper
    assert "__fmaf_rn(rk2, rk2, y2)" in helper
    assert "__fmaf_rn(rk0, rk0, xy2)" in helper
    assert text.count("relion_radius_squared(rk0, rk1, rk2)") == 3


def test_relion_x_half_bp_block_topology_env_is_off_by_default(monkeypatch):
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", raising=False)
    assert cuda_backproject.relion_x_half_bp_block_topology_enabled() is False

    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", "1")
    assert cuda_backproject.relion_x_half_bp_block_topology_enabled() is True


def test_relion_x_half_bp_block_topology_expands_native_current_square():
    # Full 8x8 FFTW-half indices for (ky,kx)=(0,1),(+2,2),(-1,1).
    pixel_indices = jnp.asarray([1, 2 * 5 + 2, 7 * 5 + 1], dtype=jnp.int32)
    images = jnp.asarray([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]], dtype=jnp.float32)

    dense, dense_indices, current_height, current_half_width = (
        cuda_backproject._prepare_relion_x_half_block_topology_operands(
            images,
            pixel_indices,
            (8, 8),
            max_r=2,
        )
    )

    assert (current_height, current_half_width) == (4, 3)
    np.testing.assert_array_equal(np.asarray(dense_indices), np.arange(12, dtype=np.int32))
    expected = np.zeros((2, 12), dtype=np.float32)
    expected[:, [1, 8, 10]] = np.asarray(images)
    np.testing.assert_array_equal(np.asarray(dense), expected)


def test_relion_x_half_bp_block_topology_expands_batched_operands():
    pixel_indices = jnp.asarray([1, 7 * 5 + 1], dtype=jnp.int32)
    images = jnp.arange(2 * 3 * 2, dtype=jnp.float32).reshape(2, 3, 2)

    dense, _, current_height, current_half_width = (
        cuda_backproject._prepare_relion_x_half_block_topology_operands(
            images,
            pixel_indices,
            (8, 8),
            max_r=2,
        )
    )

    assert (current_height, current_half_width) == (4, 3)
    assert dense.shape == (2, 3, 12)
    np.testing.assert_array_equal(np.asarray(dense[..., [1, 10]]), np.asarray(images))
    assert np.count_nonzero(np.asarray(dense)) == np.count_nonzero(np.asarray(images))


def test_relion_x_half_bp_block_topology_actual_256_to_48_support_is_unique():
    full_height = 256
    full_half_width = full_height // 2 + 1
    max_r = 24
    signed_coordinates = [
        (ky, kx)
        for ky in range(-max_r, max_r + 1)
        for kx in range(max_r + 1)
        if ky * ky + kx * kx <= max_r * max_r and not (kx == 0 and ky < 0)
    ]
    pixel_indices = jnp.asarray(
        [((ky % full_height) * full_half_width + kx) for ky, kx in signed_coordinates],
        dtype=jnp.int32,
    )
    images = jnp.arange(1, len(signed_coordinates) + 1, dtype=jnp.float32)[None]

    dense, _, current_height, current_half_width = (
        cuda_backproject._prepare_relion_x_half_block_topology_operands(
            images,
            pixel_indices,
            (full_height, full_height),
            max_r=max_r,
        )
    )

    expected_indices = np.asarray(
        [(ky % current_height) * current_half_width + kx for ky, kx in signed_coordinates]
    )
    assert np.unique(expected_indices).size == expected_indices.size
    assert np.all(expected_indices < current_height * current_half_width)
    np.testing.assert_array_equal(np.asarray(dense)[0, expected_indices], np.asarray(images)[0])
    assert np.count_nonzero(np.asarray(dense)) == len(signed_coordinates)


def test_relion_x_half_bp_block_topology_cuda_source_covers_single_and_batch():
    cuda_source = Path(__file__).resolve().parents[2] / "recovar" / "cuda" / "cuda_backproject.cu"
    text = cuda_source.read_text()

    assert "RELION_BLOCK_TOPOLOGY ? 128 : n_pixels" in text
    assert "dim3 relion_block(128)" in text
    assert text.count("relion_block_topology") >= 10
    assert "batch_backproject_indexed_kernel<T, O, HV, HI, RD, true>" in text
    assert '.Attr<int64_t>("relion_block_topology")' in text

    single_source = inspect.getsource(cuda_backproject.backproject_indexed)
    batch_source = inspect.getsource(cuda_backproject.batch_backproject_indexed)
    gate = "relion_x_half and relion_x_half_bp_block_topology_enabled()"
    assert gate in single_source
    assert gate in batch_source


def test_relion_fused_x_half_wrapper_uses_mixed_aliases_and_native_square(monkeypatch):
    observed = {}

    def fake_ffi_call(target, result_types, **options):
        observed["target"] = target
        observed["result_types"] = result_types
        observed["options"] = options

        def call(*args, **attrs):
            observed["args"] = args
            observed["attrs"] = attrs
            return args[4], args[5]

        return call

    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda_backproject.jax.ffi, "ffi_call", fake_ffi_call)

    image_shape = (8, 8)
    volume_shape = (7, 7, 7)
    volume_size = 7 * 7 * 4
    pixel_indices = jnp.asarray([1, 2 * 5 + 2, 7 * 5 + 1], dtype=jnp.int32)
    data_rows = jnp.asarray(
        [[1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j], [7.0 + 8.0j, 9.0 + 10.0j, 11.0 + 12.0j]],
        dtype=jnp.complex64,
    )
    weight_rows = jnp.arange(1, 7, dtype=jnp.float32).reshape(2, 3)
    rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (2, 3, 3))
    data_volume = jnp.zeros(volume_size, dtype=jnp.complex64)
    weight_volume = jnp.zeros(volume_size, dtype=jnp.float32)

    result = cuda_backproject.relion_fused_x_half_backproject_indexed.__wrapped__(
        data_volume,
        weight_volume,
        data_rows,
        weight_rows,
        pixel_indices,
        rotations,
        image_shape,
        volume_shape,
        2.0,
    )

    assert observed["target"] == cuda_backproject._TARGET_RELION_FUSED_X_HALF_BP
    assert observed["options"]["input_output_aliases"] == {4: 0, 5: 1}
    assert observed["options"]["vmap_method"] == "sequential"
    assert [item.dtype for item in observed["result_types"]] == [jnp.complex64, jnp.float32]
    dense_data, dense_weight, dense_indices, rot6, data_in, weight_in = observed["args"]
    assert dense_data.shape == (2, 12)
    assert dense_weight.shape == (2, 12)
    np.testing.assert_array_equal(np.asarray(dense_indices), np.arange(12, dtype=np.int32))
    assert rot6.shape == (2, 6) and rot6.dtype == jnp.float32
    assert data_in is data_volume and weight_in is weight_volume
    assert observed["attrs"]["image_h"] == 4
    assert observed["attrs"]["image_w"] == 3
    assert observed["attrs"]["full_image_w"] == 4
    assert result[0] is data_volume and result[1] is weight_volume


@pytest.mark.parametrize(
    "data_dtype,weight_dtype,index_dtype,error_match",
    [
        (jnp.complex128, jnp.float32, jnp.int32, "data volume must be complex64"),
        (jnp.complex64, jnp.float64, jnp.int32, "weight volume must be float32"),
        (jnp.complex64, jnp.float32, jnp.int64, "pixel indices must be int32"),
    ],
)
def test_relion_fused_x_half_wrapper_rejects_non_relion_dtypes(
    monkeypatch, data_dtype, weight_dtype, index_dtype, error_match
):
    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    data_volume = jnp.zeros(7 * 7 * 4, dtype=data_dtype)
    weight_volume = jnp.zeros(7 * 7 * 4, dtype=weight_dtype)
    data_rows = jnp.ones((1, 1), dtype=jnp.complex64)
    weight_rows = jnp.ones((1, 1), dtype=jnp.float32)
    pixel_indices = jnp.asarray([0], dtype=index_dtype)
    rotations = jnp.eye(3, dtype=jnp.float32)[None]

    with pytest.raises(TypeError, match=error_match):
        cuda_backproject.relion_fused_x_half_backproject_indexed.__wrapped__(
            data_volume,
            weight_volume,
            data_rows,
            weight_rows,
            pixel_indices,
            rotations,
            (8, 8),
            (7, 7, 7),
            2.0,
        )


def test_relion_fused_x_half_cuda_source_interleaves_neighbor_atomics():
    cuda_source = Path(__file__).resolve().parents[2] / "recovar" / "cuda" / "cuda_backproject.cu"
    text = cuda_source.read_text()

    assert "relion_fused_x_half_backproject_kernel" in text
    assert "for (int pix = (int)threadIdx.x; pix < n_pixels; pix += 128)" in text
    assert "dim3 block(128)" in text
    assert "if (!(Fweight > 0.0f)) {" in text
    atomic_sequence = re.compile(
        r"atomicAdd\(&data_volume\[off\]\.x, sre\);\s*"
        r"atomicAdd\(&data_volume\[off\]\.y, sim\);\s*"
        r"atomicAdd\(&weight_volume\[off\], w \* Fweight\);"
    )
    assert atomic_sequence.search(text)
    handler = text[
        text.index("RelionFusedXHalfBackproject, RelionFusedXHalfBackprojectImpl") :
        text.index(
            "RelionFusedXHalfBackprojectSignature, "
            "RelionFusedXHalfBackprojectSignatureImpl"
        )
    ]
    assert handler.count(".Ret<ffi::AnyBuffer>()") == 2


def test_relion_fused_x_half_signature_inertness_gate_rejects_shadow_mismatch():
    data_accumulator = np.asarray([1.0 + 2.0j], dtype=np.complex64)
    weight_accumulator = np.asarray([3.0], dtype=np.float32)
    expected_operands = (
        np.asarray([[4.0 + 5.0j]], dtype=np.complex64),
        np.asarray([[6.0]], dtype=np.float32),
        np.asarray([7], dtype=np.int32),
        np.asarray([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([8], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
    )
    outputs = (
        data_accumulator,
        weight_accumulator,
        *(np.asarray([0], dtype=np.int32) for _ in range(7)),
        data_accumulator.copy(),
        weight_accumulator.copy(),
        *(operand.copy() for operand in expected_operands),
    )
    cuda_backproject._require_signature_inertness_outputs(outputs, expected_operands)

    mismatched = list(outputs)
    mismatched[12] = mismatched[12].copy()
    mismatched[12][0, 0] = np.nextafter(
        mismatched[12][0, 0], np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(RuntimeError, match="weight_rows"):
        cuda_backproject._require_signature_inertness_outputs(
            tuple(mismatched), expected_operands
        )


def test_relion_fused_x_half_signature_cuda_source_copies_before_read_only_kernel():
    cuda_source = Path(__file__).resolve().parents[2] / "recovar" / "cuda" / "cuda_backproject.cu"
    text = cuda_source.read_text()
    start = text.index("cudaError_t launch_relion_fused_x_half_backproject_with_signature(")
    launch = text[start : text.index("template <typename T>", start)]

    ordinary = launch.index("launch_relion_fused_x_half_backproject(")
    shadow = launch.index("cudaMemcpyAsync(accumulator_shadow_data")
    signature = launch.index("relion_fused_x_half_backproject_kernel<true, false>")
    assert ordinary < shadow < signature
    assert "relion_fused_x_half_backproject_kernel<true, true>" not in launch
    for name in (
        "accumulator_shadow_weight",
        "operand_shadow_data_rows",
        "operand_shadow_weight_rows",
        "operand_shadow_pixel_indices",
        "operand_shadow_rot",
        "operand_shadow_canonical_rotation_keys",
        "operand_shadow_signature_row_indices",
    ):
        assert f"cudaMemcpyAsync({name}" in launch


@pytest.mark.gpu
def test_relion_fused_x_half_cuda_matches_separate_topology(
    monkeypatch, custom_cuda_lib, gpu_device
):
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", "1")
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)

    image_shape = (8, 8)
    volume_shape = (7, 7, 7)
    volume_size = 7 * 7 * 4
    pixel_indices = jnp.asarray([1, 2 * 5 + 2, 7 * 5 + 1], dtype=jnp.int32)
    data_rows = jnp.asarray(
        [[1.0 + 2.0j, -3.0 + 1.5j, 0.25 - 2.0j], [0.0 + 0.0j, 2.0 - 1.0j, -1.0 + 0.5j]],
        dtype=jnp.complex64,
    )
    weight_rows = jnp.asarray([[1.0, 0.5, 2.0], [0.75, 1.25, 0.25]], dtype=jnp.float32)
    rotations = jnp.asarray(
        [
            np.eye(3, dtype=np.float32),
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=jnp.float32,
    )

    with cuda_backproject.jax.default_device(gpu_device):
        data_initial = (
            np.arange(volume_size, dtype=np.float32) * np.complex64(1.0 + 0.5j) / volume_size
        ).astype(np.complex64)
        weight_initial = np.arange(volume_size, dtype=np.float32) / volume_size
        expected_data_volume = cuda_backproject.jax.device_put(data_initial.copy())
        expected_weight_volume = cuda_backproject.jax.device_put(weight_initial.copy())
        actual_data_volume = cuda_backproject.jax.device_put(data_initial.copy())
        actual_weight_volume = cuda_backproject.jax.device_put(weight_initial.copy())
        data_rows = cuda_backproject.jax.device_put(data_rows)
        weight_rows = cuda_backproject.jax.device_put(weight_rows)
        pixel_indices = cuda_backproject.jax.device_put(pixel_indices)
        rotations = cuda_backproject.jax.device_put(rotations)
        expected_data = cuda_backproject.backproject_indexed(
            expected_data_volume,
            data_rows,
            pixel_indices,
            rotations,
            image_shape,
            volume_shape,
            order=1,
            half_volume=True,
            half_image=True,
            max_r=2.0,
            relion_x_half=True,
        )
        expected_weight = cuda_backproject.backproject_indexed(
            expected_weight_volume,
            weight_rows,
            pixel_indices,
            rotations,
            image_shape,
            volume_shape,
            order=1,
            half_volume=True,
            half_image=True,
            max_r=2.0,
            relion_x_half=True,
        )
        actual_data, actual_weight = cuda_backproject.relion_fused_x_half_backproject_indexed(
            actual_data_volume,
            actual_weight_volume,
            data_rows,
            weight_rows,
            pixel_indices,
            rotations,
            image_shape,
            volume_shape,
            2.0,
        )

    np.testing.assert_allclose(np.asarray(actual_data), np.asarray(expected_data), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual_weight), np.asarray(expected_weight), rtol=1e-6, atol=1e-6)
