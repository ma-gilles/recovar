"""CUDA/RELION parity for streamed BPref point-group symmetry."""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")


pytestmark = [pytest.mark.unit, pytest.mark.gpu]

_ORI_SIZE = 4
_PADDING_FACTOR = 1
_CURRENT_SIZE = 4
_R_MAX = 2
_PAD_SIZE = 7
_HALF_SHAPE = (_PAD_SIZE, _PAD_SIZE, _PAD_SIZE // 2 + 1)
_VOLUME_SHAPE = (_PAD_SIZE,) * 3


@pytest.fixture(autouse=True)
def _use_custom_cuda_lib(monkeypatch, custom_cuda_lib):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)


def _skip_if_unavailable(*, require_relion_bind: bool = True):
    from recovar.cuda_backproject import cuda_available

    if not cuda_available():
        pytest.skip("custom CUDA BPref symmetry finalizer is unavailable")
    if require_relion_bind:
        pytest.importorskip("recovar.relion_bind._relion_bind_core")


@pytest.mark.parametrize(
    "symmetry",
    ["C2", "D2", "T", "O", "I", "I1", "I3", "I4"],
)
@pytest.mark.parametrize(
    ("complex_dtype", "real_dtype", "atol", "rtol"),
    [
        (np.complex64, np.float32, 4e-4, 4e-5),
        (np.complex128, np.float64, 2e-11, 2e-11),
    ],
)
def test_streamed_cuda_matches_relion_cpu_oracle(
    symmetry,
    complex_dtype,
    real_dtype,
    atol,
    rtol,
):
    _skip_if_unavailable()
    from recovar.relion_bind._relion_bind_core import (
        apply_point_group_symmetry_to_bpref,
    )

    from recovar.em.cuda.kernels import relion_point_group_symmetrise_bpref
    from recovar.em.symmetry import rotational_operators

    rng = np.random.default_rng(20260830)
    raw_data = (
        rng.standard_normal(_HALF_SHAPE) + 1j * rng.standard_normal(_HALF_SHAPE)
    ).astype(complex_dtype)
    raw_weight = rng.uniform(0.1, 2.0, size=_HALF_SHAPE).astype(real_dtype)
    operators = rotational_operators(symmetry, dtype=real_dtype)

    cuda_data, cuda_weight = relion_point_group_symmetrise_bpref(
        jnp.asarray(raw_data.reshape(-1)),
        jnp.asarray(raw_weight.reshape(-1)),
        jnp.asarray(operators),
        _VOLUME_SHAPE,
        _R_MAX,
    )
    cuda_data.block_until_ready()
    cuda_weight.block_until_ready()
    oracle_data, oracle_weight = apply_point_group_symmetry_to_bpref(
        np.ascontiguousarray(raw_data, dtype=np.complex128),
        np.ascontiguousarray(raw_weight, dtype=np.float64),
        symmetry=symmetry,
        ori_size=_ORI_SIZE,
        padding_factor=_PADDING_FACTOR,
        current_size=_CURRENT_SIZE,
        r_max=_R_MAX,
        enforce_hermitian=True,
    )

    np.testing.assert_allclose(
        np.asarray(cuda_data).reshape(_HALF_SHAPE),
        oracle_data,
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(cuda_weight).reshape(_HALF_SHAPE),
        oracle_weight,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("symmetry", ["C1", "C2", "I1"])
def test_split_ranged_cuda_is_bitwise_equal_to_complex_input(symmetry):
    _skip_if_unavailable(require_relion_bind=False)
    from recovar.em.cuda.kernels import (
        relion_point_group_symmetrise_bpref,
        relion_point_group_symmetrise_bpref_split_host,
    )
    from recovar.em.symmetry import rotational_operators

    rng = np.random.default_rng(20260831)
    raw_data = (
        rng.standard_normal(_HALF_SHAPE) + 1j * rng.standard_normal(_HALF_SHAPE)
    ).astype(np.complex64)
    raw_weight = rng.uniform(0.1, 2.0, size=_HALF_SHAPE).astype(np.float32)
    operators = jnp.asarray(rotational_operators(symmetry, dtype=np.float32))

    complex_data, complex_weight = relion_point_group_symmetrise_bpref(
        jnp.asarray(raw_data.reshape(-1)),
        jnp.asarray(raw_weight.reshape(-1)),
        operators,
        _VOLUME_SHAPE,
        _R_MAX,
    )
    split_data, split_weight = relion_point_group_symmetrise_bpref_split_host(
        jnp.asarray(raw_data.real.reshape(-1)),
        jnp.asarray(raw_data.imag.reshape(-1)),
        jnp.asarray(raw_weight.reshape(-1)),
        operators,
        _VOLUME_SHAPE,
        _R_MAX,
        chunk_voxels=37,
    )

    np.testing.assert_array_equal(
        split_data.view(np.uint32),
        np.asarray(complex_data).view(np.uint32),
    )
    np.testing.assert_array_equal(
        split_weight.view(np.uint32),
        np.asarray(complex_weight).view(np.uint32),
    )


def test_split_ranged_c1_is_bitwise_equal_to_historical_host_x0_path():
    _skip_if_unavailable(require_relion_bind=False)
    from recovar.em.cuda.kernels import relion_point_group_symmetrise_bpref_split_host
    from recovar.em.helpers.half_volume_mstep import (
        enforce_relion_half_volume_x0_hermitian_host,
    )
    from recovar.em.symmetry import rotational_operators

    rng = np.random.default_rng(20260831)
    raw_data = (
        rng.standard_normal(_HALF_SHAPE) + 1j * rng.standard_normal(_HALF_SHAPE)
    ).astype(np.complex64)
    raw_weight = rng.uniform(0.1, 2.0, size=_HALF_SHAPE).astype(np.float32)
    flat_data = raw_data.reshape(-1)
    flat_weight = raw_weight.reshape(-1)
    partner_x0 = np.ravel_multi_index((_PAD_SIZE - 1, _PAD_SIZE - 1, 0), _HALF_SHAPE)
    self_x0 = np.ravel_multi_index((_PAD_SIZE // 2, _PAD_SIZE // 2, 0), _HALF_SHAPE)
    flat_data.real[[0, partner_x0, self_x0, -1]] = np.asarray(
        [-0.0, 0.0, -0.0, -0.0],
        dtype=np.float32,
    )
    flat_data.imag[[0, partner_x0, self_x0, -1]] = np.asarray(
        [0.0, -0.0, -0.0, 0.0],
        dtype=np.float32,
    )
    flat_weight[[0, partner_x0, self_x0, -1]] = np.asarray(
        [-0.0, 0.0, -0.0, -0.0],
        dtype=np.float32,
    )

    expected_data = enforce_relion_half_volume_x0_hermitian_host(
        flat_data,
        _VOLUME_SHAPE,
    )
    expected_weight = enforce_relion_half_volume_x0_hermitian_host(
        flat_weight,
        _VOLUME_SHAPE,
    )
    chunk_voxels = 37
    assert raw_data.size % chunk_voxels != 0
    split_data, split_weight = relion_point_group_symmetrise_bpref_split_host(
        jnp.asarray(flat_data.real),
        jnp.asarray(flat_data.imag),
        jnp.asarray(flat_weight),
        jnp.asarray(rotational_operators("C1", dtype=np.float32)),
        _VOLUME_SHAPE,
        _R_MAX,
        chunk_voxels=chunk_voxels,
    )

    np.testing.assert_array_equal(
        split_data.view(np.uint32),
        np.asarray(expected_data).view(np.uint32),
    )
    np.testing.assert_array_equal(
        split_weight.view(np.uint32),
        np.asarray(expected_weight).view(np.uint32),
    )


def test_split_range_zero_pads_only_past_the_final_voxel():
    _skip_if_unavailable(require_relion_bind=False)
    from recovar.em.cuda import kernels as em_cuda_kernels
    from recovar.em.symmetry import rotational_operators

    rng = np.random.default_rng(20260831)
    raw_data = (
        rng.standard_normal(_HALF_SHAPE) + 1j * rng.standard_normal(_HALF_SHAPE)
    ).astype(np.complex64)
    raw_weight = rng.uniform(0.1, 2.0, size=_HALF_SHAPE).astype(np.float32)
    operators = jnp.asarray(rotational_operators("C1", dtype=np.float32))
    expected_data, expected_weight = em_cuda_kernels.relion_point_group_symmetrise_bpref(
        jnp.asarray(raw_data.reshape(-1)),
        jnp.asarray(raw_weight.reshape(-1)),
        operators,
        _VOLUME_SHAPE,
        _R_MAX,
    )
    range_size = 17
    valid_count = 5
    range_start = raw_data.size - valid_count
    range_data, range_weight = (
        em_cuda_kernels._relion_point_group_symmetrise_bpref_split_range_static(
            jnp.asarray(raw_data.real.reshape(-1)),
            jnp.asarray(raw_data.imag.reshape(-1)),
            jnp.asarray(raw_weight.reshape(-1)),
            operators,
            jnp.asarray([range_start], dtype=jnp.int64),
            _VOLUME_SHAPE,
            _R_MAX,
            range_size,
        )
    )
    range_data = np.asarray(range_data)
    range_weight = np.asarray(range_weight)
    np.testing.assert_array_equal(
        range_data[:valid_count].view(np.uint32),
        np.asarray(expected_data)[range_start:].view(np.uint32),
    )
    np.testing.assert_array_equal(
        range_weight[:valid_count].view(np.uint32),
        np.asarray(expected_weight)[range_start:].view(np.uint32),
    )
    np.testing.assert_array_equal(range_data[valid_count:], np.zeros(12, dtype=np.complex64))
    np.testing.assert_array_equal(range_weight[valid_count:], np.zeros(12, dtype=np.float32))


def test_cuda_negative_x_weight_sum_and_radius_boundary():
    _skip_if_unavailable()
    from recovar.em.cuda.kernels import relion_point_group_symmetrise_bpref
    from recovar.em.symmetry import rotational_operators

    center = _PAD_SIZE // 2
    boundary = (center, center, _R_MAX)
    outside = (center, center, _R_MAX + 1)
    data = np.zeros(_HALF_SHAPE, dtype=np.complex64)
    weight = np.zeros(_HALF_SHAPE, dtype=np.float32)
    data[boundary] = 1.0 + 2.0j
    data[outside] = 5.0 + 7.0j
    weight[boundary] = 2.0
    weight[outside] = 3.0

    data_out, weight_out = relion_point_group_symmetrise_bpref(
        jnp.asarray(data.reshape(-1)),
        jnp.asarray(weight.reshape(-1)),
        jnp.asarray(rotational_operators("C2", dtype=np.float32)),
        _VOLUME_SHAPE,
        _R_MAX,
    )
    data_out = np.asarray(data_out).reshape(_HALF_SHAPE)
    weight_out = np.asarray(weight_out).reshape(_HALF_SHAPE)

    assert data_out[boundary] == 2.0 + 0.0j
    assert weight_out[boundary] == 4.0
    assert data_out[outside] == data[outside]
    assert weight_out[outside] == weight[outside]


@pytest.mark.parametrize("symmetry", ["C1", "C2", "I1"])
@pytest.mark.parametrize("via_finalizer", [False, True])
def test_complex_ranged_cuda_is_bitwise_equal_to_full_output(symmetry, via_finalizer, monkeypatch):
    _skip_if_unavailable(require_relion_bind=False)
    from recovar.em.cuda.kernels import (
        relion_point_group_symmetrise_bpref,
        relion_point_group_symmetrise_bpref_host,
    )
    from recovar.em.symmetry import rotational_operators

    rng = np.random.default_rng(20260831)
    raw_data = (
        rng.standard_normal(_HALF_SHAPE) + 1j * rng.standard_normal(_HALF_SHAPE)
    ).astype(np.complex64)
    raw_weight = rng.uniform(0.1, 2.0, size=_HALF_SHAPE).astype(np.float32)
    operators = jnp.asarray(rotational_operators(symmetry, dtype=np.float32))

    complex_data, complex_weight = relion_point_group_symmetrise_bpref(
        jnp.asarray(raw_data.reshape(-1)),
        jnp.asarray(raw_weight.reshape(-1)),
        operators,
        _VOLUME_SHAPE,
        _R_MAX,
    )
    if via_finalizer:
        import logging
        from recovar.em.helpers.half_volume_mstep import finalize_half_volume_bpref
        monkeypatch.setenv("RECOVAR_RELION_BPREF_SYMMETRY_CHUNK_VOXELS", "37")
        split_data, split_weight = finalize_half_volume_bpref(
            jnp.asarray(raw_data.reshape(-1)),
            jnp.asarray(raw_weight.reshape(-1)),
            _VOLUME_SHAPE, logger=logging.getLogger(__name__), label="bounded",
            symmetry_label=symmetry, relion_x_half=True, force_host=True,
        )
        assert isinstance(split_data, np.ndarray)
        assert isinstance(split_weight, np.ndarray)
    else:
        split_data, split_weight = relion_point_group_symmetrise_bpref_host(
            jnp.asarray(raw_data.reshape(-1)),
            jnp.asarray(raw_weight.reshape(-1)),
            operators,
            _VOLUME_SHAPE,
            _R_MAX,
            chunk_voxels=37,
        )

    np.testing.assert_array_equal(
        split_data.view(np.uint32),
        np.asarray(complex_data).view(np.uint32),
    )
    np.testing.assert_array_equal(
        split_weight.view(np.uint32),
        np.asarray(complex_weight).view(np.uint32),
    )
