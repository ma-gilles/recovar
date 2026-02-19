import numpy as np
import pytest

pytest.importorskip("jax")
import recovar.core as core
import recovar.core_forward as core_forward

pytestmark = pytest.mark.unit


def _ones_ctf(ctf_params, image_shape, voxel_size):
    return np.ones((ctf_params.shape[0], image_shape[0] * image_shape[1]), dtype=np.float32)


def _twos_ctf(ctf_params, image_shape, voxel_size):
    return 2.0 * np.ones((ctf_params.shape[0], image_shape[0] * image_shape[1]), dtype=np.float32)


def test_core_reexports_forward_api():
    assert core.forward_model_from_map is core_forward.forward_model_from_map
    assert core.compute_A_t_Av_forward_model_from_map is core_forward.compute_A_t_Av_forward_model_from_map
    assert core.batch_translate_images is core_forward.batch_translate_images


def test_forward_model_from_map_skip_ctf():
    volume_shape = (4, 4, 4)
    image_shape = (2, 2)
    volume = np.zeros(np.prod(volume_shape), dtype=np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    out = np.asarray(
        core_forward.forward_model_from_map(
            volume,
            ctf_params,
            rotation_matrices,
            image_shape,
            volume_shape,
            1.0,
            _ones_ctf,
            "nearest",
            True,
        )
    )
    assert out.shape == (1, image_shape[0] * image_shape[1])


def test_adjoint_forward_model_from_map_shape():
    volume_shape = (4, 4, 4)
    image_shape = (2, 2)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)
    slices = np.zeros((1, image_shape[0] * image_shape[1]), dtype=np.float32)

    out = np.asarray(
        core_forward.adjoint_forward_model_from_map(
            slices,
            ctf_params,
            rotation_matrices,
            image_shape,
            volume_shape,
            1.0,
            _ones_ctf,
            "nearest",
            True,
        )
    )
    assert out.shape == (np.prod(volume_shape),)


def test_forward_model_from_map_applies_ctf_when_enabled():
    volume_shape = (4, 4, 4)
    image_shape = (2, 2)
    volume = np.ones(np.prod(volume_shape), dtype=np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    out_skip = np.asarray(
        core_forward.forward_model_from_map(
            volume,
            ctf_params,
            rotation_matrices,
            image_shape,
            volume_shape,
            1.0,
            _twos_ctf,
            "nearest",
            True,
        )
    )
    out_ctf = np.asarray(
        core_forward.forward_model_from_map(
            volume,
            ctf_params,
            rotation_matrices,
            image_shape,
            volume_shape,
            1.0,
            _twos_ctf,
            "nearest",
            False,
        )
    )
    np.testing.assert_allclose(out_ctf, 2.0 * out_skip)


def test_forward_model_from_map_and_return_adjoint_contracts():
    volume_shape = (4, 4, 4)
    image_shape = (2, 2)
    volume = np.zeros(np.prod(volume_shape), dtype=np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    slices, adj = core_forward.forward_model_from_map_and_return_adjoint(
        volume,
        ctf_params,
        rotation_matrices,
        image_shape,
        volume_shape,
        1.0,
        _ones_ctf,
        "nearest",
        True,
    )
    assert np.asarray(slices).shape == (1, image_shape[0] * image_shape[1])
    pulled = adj(np.ones_like(np.asarray(slices)))[0]
    assert np.asarray(pulled).shape == (np.prod(volume_shape),)


def test_compute_A_t_Av_forward_model_from_map_returns_singleton_tuple():
    volume_shape = (4, 4, 4)
    image_shape = (2, 2)
    volume = np.ones(np.prod(volume_shape), dtype=np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    out = core_forward.compute_A_t_Av_forward_model_from_map(
        volume,
        ctf_params,
        rotation_matrices,
        image_shape,
        volume_shape,
        1.0,
        _ones_ctf,
        "nearest",
        noise_variance=1.0,
        skip_ctf=True,
    )
    assert isinstance(out, tuple)
    assert len(out) == 1
    assert np.asarray(out[0]).shape == (np.prod(volume_shape),)
