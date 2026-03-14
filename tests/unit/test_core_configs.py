"""Unit tests for recovar.core.configs."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.configs as configs
from recovar.core.ctf import CTFEvaluator, CTFMode, as_ctf_evaluator

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ForwardModelConfig
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Create a ForwardModelConfig with reasonable defaults."""
    defaults = dict(
        image_shape=(64, 64),
        volume_shape=(64, 64, 64),
        grid_size=64,
        voxel_size=1.0,
        padding=0,
        disc_type="linear_interp",
        ctf=as_ctf_evaluator(lambda params, shape, vs, **kw: jnp.ones(shape)),
        premultiplied_ctf=False,
        volume_mask_threshold=0.0,
    )
    defaults.update(overrides)
    return configs.ForwardModelConfig(**defaults)


class TestForwardModelConfig:
    def test_creation(self):
        cfg = _make_config()
        assert cfg.image_shape == (64, 64)
        assert cfg.volume_shape == (64, 64, 64)
        assert cfg.grid_size == 64

    def test_volume_size(self):
        cfg = _make_config(volume_shape=(4, 5, 6))
        assert cfg.volume_size == 120

    def test_image_size(self):
        cfg = _make_config(image_shape=(32, 32))
        assert cfg.image_size == 1024

    def test_replace(self):
        cfg = _make_config(padding=0)
        cfg2 = cfg.replace(padding=4)
        assert cfg.padding == 0
        assert cfg2.padding == 4

    def test_replace_preserves_other_fields(self):
        cfg = _make_config(disc_type="linear_interp", grid_size=64)
        cfg2 = cfg.replace(disc_type="cubic")
        assert cfg2.disc_type == "cubic"
        assert cfg2.grid_size == 64
        assert cfg2.image_shape == cfg.image_shape

    def test_compute_ctf(self):
        called_with = {}

        def mock_ctf(params, shape, vs, **kw):
            called_with["params"] = params
            called_with["shape"] = shape
            called_with["vs"] = vs
            return jnp.ones(10)

        cfg = _make_config(
            ctf=as_ctf_evaluator(mock_ctf),
            image_shape=(32, 32),
            voxel_size=1.5,
        )
        result = cfg.compute_ctf(jnp.array([1.0, 2.0]))
        assert called_with["shape"] == (32, 32)
        assert called_with["vs"] == 1.5
        assert result.shape == (10,)

    def test_compute_ctf_half(self):
        """config.compute_ctf_half should return half-spectrum CTF."""
        cfg = _make_config(ctf=CTFEvaluator(mode=CTFMode.SPA))
        import jax.numpy as jnp
        params = jnp.zeros((1, 9), dtype=jnp.float32)
        params = params.at[:, 3].set(300.0)
        params = params.at[:, 4].set(2.7)
        params = params.at[:, 5].set(0.1)
        params = params.at[:, 8].set(1.0)
        half = cfg.compute_ctf_half(params)
        assert half.shape[1] == cfg.image_shape[0] * (cfg.image_shape[1] // 2 + 1)

    def test_compute_ctf_at_shape(self):
        called_with = {}

        def mock_ctf(params, shape, vs, **kw):
            called_with["shape"] = shape
            return jnp.ones(10)

        cfg = _make_config(
            ctf=as_ctf_evaluator(mock_ctf),
            image_shape=(32, 32),
        )
        cfg.compute_ctf_at_shape(jnp.array([1.0, 2.0]), (64, 64))
        assert called_with["shape"] == (64, 64)

    def test_process_fn_default_none(self):
        cfg = _make_config()
        assert cfg.process_fn is None


# ---------------------------------------------------------------------------
# BatchData
# ---------------------------------------------------------------------------

class TestBatchData:
    def test_creation(self):
        bd = configs.BatchData(
            images=jnp.ones((5, 64)),
            rotation_matrices=jnp.eye(3)[None].repeat(5, axis=0),
            translations=jnp.zeros((5, 2)),
            ctf_params=jnp.ones((5, 9)),
        )
        assert bd.images.shape == (5, 64)
        assert bd.noise_variance is None

    def test_with_noise_variance(self):
        bd = configs.BatchData(
            images=jnp.ones((3, 16)),
            rotation_matrices=jnp.eye(3)[None].repeat(3, axis=0),
            translations=jnp.zeros((3, 2)),
            ctf_params=jnp.ones((3, 9)),
            noise_variance=jnp.ones(3),
        )
        assert bd.noise_variance.shape == (3,)


# ---------------------------------------------------------------------------
# ModelState
# ---------------------------------------------------------------------------

class TestModelState:
    def test_creation(self):
        ms = configs.ModelState(
            mean_estimate=jnp.zeros(100),
            volume_mask=jnp.ones(100),
        )
        assert ms.mean_estimate.shape == (100,)
        assert ms.basis is None
        assert ms.eigenvalues is None

    def test_with_basis(self):
        ms = configs.ModelState(
            mean_estimate=jnp.zeros(100),
            volume_mask=jnp.ones(100),
            basis=jnp.ones((100, 10)),
            eigenvalues=jnp.ones(10),
        )
        assert ms.basis.shape == (100, 10)
        assert ms.eigenvalues.shape == (10,)


# ---------------------------------------------------------------------------
# CovarianceOpts
# ---------------------------------------------------------------------------

class TestCovColumnOpts:
    def test_defaults(self):
        opts = configs.CovColumnOpts()
        assert opts.right_kernel == "triangular"
        assert opts.left_kernel == "triangular"
        assert opts.right_kernel_width == 2
        assert opts.mask_images is True
        assert opts.soften_mask == 3

    def test_custom_values(self):
        opts = configs.CovColumnOpts(
            right_kernel="square",
            left_kernel="square",
            right_kernel_width=1,
            mask_images=False,
            soften_mask=5,
        )
        assert opts.right_kernel == "square"
        assert opts.mask_images is False
        assert opts.soften_mask == 5


class TestCovarianceOpts:
    def test_defaults(self):
        opts = configs.CovarianceOpts(disc_type_u="linear_interp")
        assert opts.disc_type_u == "linear_interp"
        assert opts.do_mask_images is True
        assert opts.shared_label is False
        assert opts.soften == 5

    def test_custom_values(self):
        opts = configs.CovarianceOpts(
            disc_type_u="cubic",
            do_mask_images=False,
            shared_label=True,
            soften=3,
        )
        assert opts.disc_type_u == "cubic"
        assert opts.do_mask_images is False
        assert opts.shared_label is True
        assert opts.soften == 3


# ---------------------------------------------------------------------------
# EmbeddingOpts
# ---------------------------------------------------------------------------

class TestEmbeddingOpts:
    def test_defaults(self):
        opts = configs.EmbeddingOpts()
        assert opts.compute_covariances is False
        assert opts.compute_bias is False
        assert opts.shared_label is False
        assert opts.contrast_shared_across_tilt_series is True

    def test_custom_values(self):
        opts = configs.EmbeddingOpts(
            compute_covariances=True,
            compute_bias=True,
        )
        assert opts.compute_covariances is True
        assert opts.compute_bias is True
