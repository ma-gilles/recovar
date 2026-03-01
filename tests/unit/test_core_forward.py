
import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import recovar.core.forward as core_forward
from recovar.core.configs import ForwardModelConfig

pytestmark = pytest.mark.unit


def _ones_ctf(ctf_params, image_shape, voxel_size):
    return np.ones((ctf_params.shape[0], image_shape[0] * image_shape[1]), dtype=np.float32)


def _twos_ctf(ctf_params, image_shape, voxel_size):
    return 2.0 * np.ones((ctf_params.shape[0], image_shape[0] * image_shape[1]), dtype=np.float32)


def _make_config(image_shape=(2, 2), volume_shape=(4, 4, 4), disc_type="nearest", ctf_fun=None):
    if ctf_fun is None:
        ctf_fun = _ones_ctf
    return ForwardModelConfig(
        image_shape=image_shape,
        volume_shape=volume_shape,
        grid_size=volume_shape[0],
        voxel_size=1.0,
        padding=0,
        disc_type=disc_type,
        CTF_fun=ctf_fun,
        premultiplied_ctf=False,
        volume_mask_threshold=0.0,
    )


def test_batch_translate_images_reexported():
    assert hasattr(core_forward, "batch_translate_images")


def test_forward_model_skip_ctf():
    config = _make_config()
    volume = np.zeros(np.prod(config.volume_shape), dtype=np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    out = np.asarray(
        core_forward.forward_model(config, volume, ctf_params, rotation_matrices, skip_ctf=True)
    )
    assert out.shape == (1, config.image_size)


def test_adjoint_forward_model_shape():
    config = _make_config()
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)
    slices = np.zeros((1, config.image_size), dtype=np.float32)

    out = np.asarray(
        core_forward.adjoint_forward_model(config, slices, ctf_params, rotation_matrices, skip_ctf=True)
    )
    assert out.shape == (config.volume_size,)


def test_forward_model_applies_ctf_when_enabled():
    config = _make_config(ctf_fun=_twos_ctf)
    volume = np.ones(config.volume_size, dtype=np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    out_skip = np.asarray(
        core_forward.forward_model(config, volume, ctf_params, rotation_matrices, skip_ctf=True)
    )
    out_ctf = np.asarray(
        core_forward.forward_model(config, volume, ctf_params, rotation_matrices, skip_ctf=False)
    )
    np.testing.assert_allclose(out_ctf, 2.0 * out_skip)


def test_forward_model_and_adjoint_contracts():
    config = _make_config()
    volume = np.zeros(config.volume_size, dtype=np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    slices, adj = core_forward.forward_model_and_adjoint(
        config, volume, ctf_params, rotation_matrices, skip_ctf=True,
    )
    assert np.asarray(slices).shape == (1, config.image_size)
    pulled = adj(np.ones_like(np.asarray(slices)))[0]
    assert np.asarray(pulled).shape == (config.volume_size,)


def test_adjoint_forward_model_trilinear_shape():
    """adjoint_forward_model_trilinear should produce correct output shape."""
    config = _make_config(disc_type="linear_interp")
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)
    slices = np.ones((1, config.image_size), dtype=np.float32)
    out = np.asarray(
        core_forward.adjoint_forward_model_trilinear(
            config, slices, ctf_params, rotation_matrices, skip_ctf=True
        )
    )
    assert out.shape == (config.volume_size,)
    assert np.all(np.isfinite(out))


def test_adjoint_forward_model_trilinear_applies_ctf():
    """adjoint_forward_model_trilinear should scale by CTF when not skipped."""
    config_2x = _make_config(disc_type="linear_interp", ctf_fun=_twos_ctf)
    config_1x = _make_config(disc_type="linear_interp", ctf_fun=_ones_ctf)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)
    slices = np.ones((1, config_2x.image_size), dtype=np.float32)

    out_1x = np.asarray(
        core_forward.adjoint_forward_model_trilinear(
            config_1x, slices, ctf_params, rotation_matrices, skip_ctf=False
        )
    )
    out_2x = np.asarray(
        core_forward.adjoint_forward_model_trilinear(
            config_2x, slices, ctf_params, rotation_matrices, skip_ctf=False
        )
    )
    np.testing.assert_allclose(out_2x, 2.0 * out_1x)


def test_compute_AtAv_returns_singleton_tuple():
    config = _make_config()
    volume = np.ones(config.volume_size, dtype=np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    out = core_forward.compute_AtAv(
        config, volume, ctf_params, rotation_matrices,
        noise_variance=1.0, skip_ctf=True,
    )
    assert isinstance(out, tuple)
    assert len(out) == 1
    assert np.asarray(out[0]).shape == (config.volume_size,)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_forward_model_on_gpu(gpu_device):
    config = _make_config()
    rng = np.random.default_rng(42)
    volume = rng.standard_normal(config.volume_size).astype(np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    cpu_out = np.asarray(
        core_forward.forward_model(config, volume, ctf_params, rotation_matrices, skip_ctf=True)
    )
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(
            core_forward.forward_model(
                config, jax.device_put(volume), jax.device_put(ctf_params),
                jax.device_put(rotation_matrices), skip_ctf=True,
            )
        )
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_adjoint_forward_model_on_gpu(gpu_device):
    config = _make_config()
    rng = np.random.default_rng(43)
    slices = rng.standard_normal((1, config.image_size)).astype(np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    cpu_out = np.asarray(
        core_forward.adjoint_forward_model(config, slices, ctf_params, rotation_matrices, skip_ctf=True)
    )
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(
            core_forward.adjoint_forward_model(
                config, jax.device_put(slices), jax.device_put(ctf_params),
                jax.device_put(rotation_matrices), skip_ctf=True,
            )
        )
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_forward_model_applies_ctf_on_gpu(gpu_device):
    config = _make_config(ctf_fun=_twos_ctf)
    volume = np.ones(config.volume_size, dtype=np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    with jax.default_device(gpu_device):
        out_skip = np.asarray(
            core_forward.forward_model(
                config, jax.device_put(volume), jax.device_put(ctf_params),
                jax.device_put(rotation_matrices), skip_ctf=True,
            )
        )
        out_ctf = np.asarray(
            core_forward.forward_model(
                config, jax.device_put(volume), jax.device_put(ctf_params),
                jax.device_put(rotation_matrices), skip_ctf=False,
            )
        )
    np.testing.assert_allclose(out_ctf, 2.0 * out_skip, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_compute_AtAv_on_gpu(gpu_device):
    config = _make_config()
    volume = np.ones(config.volume_size, dtype=np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    cpu_out = np.asarray(
        core_forward.compute_AtAv(config, volume, ctf_params, rotation_matrices, noise_variance=1.0, skip_ctf=True)[0]
    )
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(
            core_forward.compute_AtAv(
                config, jax.device_put(volume), jax.device_put(ctf_params),
                jax.device_put(rotation_matrices), noise_variance=1.0, skip_ctf=True,
            )[0]
        )
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_new_api_forward_model_on_gpu(gpu_device):
    config = _make_config()
    rng = np.random.default_rng(44)
    volume = rng.standard_normal(config.volume_size).astype(np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    cpu_out = np.asarray(
        core_forward.forward_model(config, volume, ctf_params, rotation_matrices, skip_ctf=True)
    )
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(
            core_forward.forward_model(
                config, jax.device_put(volume), jax.device_put(ctf_params),
                jax.device_put(rotation_matrices), skip_ctf=True,
            )
        )
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_new_api_adjoint_forward_model_on_gpu(gpu_device):
    config = _make_config()
    rng = np.random.default_rng(45)
    slices = rng.standard_normal((1, config.image_size)).astype(np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    cpu_out = np.asarray(
        core_forward.adjoint_forward_model(config, slices, ctf_params, rotation_matrices, skip_ctf=True)
    )
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(
            core_forward.adjoint_forward_model(
                config, jax.device_put(slices), jax.device_put(ctf_params),
                jax.device_put(rotation_matrices), skip_ctf=True,
            )
        )
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_new_api_compute_AtAv_on_gpu(gpu_device):
    config = _make_config()
    volume = np.ones(config.volume_size, dtype=np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    cpu_out = np.asarray(
        core_forward.compute_AtAv(config, volume, ctf_params, rotation_matrices, noise_variance=1.0, skip_ctf=True)[0]
    )
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(
            core_forward.compute_AtAv(
                config, jax.device_put(volume), jax.device_put(ctf_params),
                jax.device_put(rotation_matrices), noise_variance=1.0, skip_ctf=True,
            )[0]
        )
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_forward_model_and_adjoint_roundtrip_on_gpu(gpu_device):
    config = _make_config()
    rng = np.random.default_rng(46)
    volume = rng.standard_normal(config.volume_size).astype(np.float32)
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]
    ctf_params = np.zeros((1, 9), dtype=np.float32)

    with jax.default_device(gpu_device):
        slices, f_adj = core_forward.forward_model_and_adjoint(
            config, jax.device_put(volume), jax.device_put(ctf_params),
            jax.device_put(rotation_matrices), skip_ctf=True,
        )
        assert np.asarray(slices).shape == (1, config.image_size)
        pulled = f_adj(jax.device_put(np.ones_like(np.asarray(slices))))[0]
        assert np.asarray(pulled).shape == (config.volume_size,)
        assert np.isfinite(np.asarray(pulled)).all()
