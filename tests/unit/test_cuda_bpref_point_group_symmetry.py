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


def _skip_if_unavailable():
    from recovar.cuda_backproject import cuda_available

    if not cuda_available():
        pytest.skip("custom CUDA BPref symmetry finalizer is unavailable")
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

    from recovar.cuda_backproject import relion_point_group_symmetrise_bpref
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


def test_cuda_negative_x_weight_sum_and_radius_boundary():
    _skip_if_unavailable()
    from recovar.cuda_backproject import relion_point_group_symmetrise_bpref
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
