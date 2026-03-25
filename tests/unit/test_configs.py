"""Tests for recovar.configs Equinox modules and the new core_forward API."""

import numpy as np
import pytest

pytest.importorskip("equinox")
pytest.importorskip("jax")

import jax.numpy as jnp

from recovar.core.configs import (
    CovarianceOpts,
    EmbeddingOpts,
    ForwardModelConfig,
    ModelState,
)
import recovar.core.forward as core_forward
from recovar.core.ctf import as_ctf_evaluator

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOLUME_SHAPE = (4, 4, 4)
IMAGE_SHAPE = (2, 2)
VOXEL_SIZE = 1.0


def _ones_ctf(ctf_params, image_shape, voxel_size, **kw):
    return np.ones((ctf_params.shape[0], image_shape[0] * image_shape[1]), dtype=np.float32)


def _twos_ctf(ctf_params, image_shape, voxel_size, **kw):
    return 2.0 * np.ones((ctf_params.shape[0], image_shape[0] * image_shape[1]), dtype=np.float32)


def _make_config(ctf_fun=_ones_ctf, disc_type="nearest"):
    return ForwardModelConfig(
        image_shape=IMAGE_SHAPE,
        volume_shape=VOLUME_SHAPE,
        grid_size=4,
        voxel_size=VOXEL_SIZE,
        padding=0,
        disc_type=disc_type,
        ctf=as_ctf_evaluator(ctf_fun),
        premultiplied_ctf=False,
        volume_mask_threshold=0.25,
    )


# ---------------------------------------------------------------------------
# ForwardModelConfig tests
# ---------------------------------------------------------------------------


class TestForwardModelConfig:
    def test_compute_ctf_returns_correct_shape(self):
        config = _make_config()
        ctf_params = np.zeros((3, 9), dtype=np.float32)
        result = config.compute_ctf(ctf_params)
        assert np.asarray(result).shape == (3, IMAGE_SHAPE[0] * IMAGE_SHAPE[1])

    def test_volume_size_property(self):
        config = _make_config()
        assert config.volume_size == 64

    def test_image_size_property(self):
        config = _make_config()
        assert config.image_size == 4

    def test_from_dataset_with_mock(self):
        """Test from_dataset with a duck-typed object."""

        class MockDataset:
            image_shape = IMAGE_SHAPE
            volume_shape = VOLUME_SHAPE
            grid_size = 4
            voxel_size = VOXEL_SIZE
            padding = 0
            premultiplied_ctf = False
            volume_mask_threshold = 0.25

            def ctf_evaluator(self, *args):
                return _ones_ctf(*args)

        config = ForwardModelConfig.from_dataset(MockDataset(), disc_type="nearest")
        assert config.grid_size == 4
        assert config.volume_shape == VOLUME_SHAPE
        assert config.disc_type == "nearest"

    def test_static_fields_trigger_recompile_on_change(self):
        """Different configs should produce different JAX traces."""
        c1 = _make_config(disc_type="nearest")
        c2 = _make_config(disc_type="linear_interp")
        # Static fields differ, so they shouldn't be equal in pytree sense
        assert c1.disc_type != c2.disc_type


# ---------------------------------------------------------------------------
# ModelState tests
# ---------------------------------------------------------------------------


class TestModelState:
    def test_optional_fields_default_to_none(self):
        ms = ModelState(
            mean_estimate=jnp.zeros(64),
            volume_mask=jnp.ones(64),
        )
        assert ms.basis is None
        assert ms.eigenvalues is None

    def test_all_fields_populated(self):
        ms = ModelState(
            mean_estimate=jnp.zeros(64),
            volume_mask=jnp.ones(64),
            basis=jnp.zeros((10, 64)),
            eigenvalues=jnp.ones(10),
        )
        assert ms.basis.shape == (10, 64)


# ---------------------------------------------------------------------------
# Option modules tests
# ---------------------------------------------------------------------------


class TestOptionModules:
    def test_covariance_opts_defaults(self):
        opts = CovarianceOpts(disc_type_u="linear_interp")
        assert opts.do_mask_images is True
        assert opts.shared_label is False
        assert opts.soften == 5

    def test_embedding_opts_defaults(self):
        opts = EmbeddingOpts()
        assert opts.compute_covariances is False
        assert opts.compute_bias is False
        assert opts.shared_label is False
        assert opts.contrast_shared_across_tilt_series is True


# ---------------------------------------------------------------------------
# New core_forward Equinox API tests
# ---------------------------------------------------------------------------


class TestNewForwardModelAPI:
    def test_forward_model_skip_ctf(self):
        config = _make_config()
        volume = np.zeros(np.prod(VOLUME_SHAPE), dtype=np.float32)
        rots = np.eye(3, dtype=np.float32)[None, ...]
        ctf_params = np.zeros((1, 9), dtype=np.float32)

        out = np.asarray(core_forward.forward_model(config, volume, ctf_params, rots, skip_ctf=True))
        assert out.shape == (1, IMAGE_SHAPE[0] * IMAGE_SHAPE[1])

    def test_forward_model_applies_ctf(self):
        config = _make_config(ctf_fun=_twos_ctf)
        volume = np.ones(np.prod(VOLUME_SHAPE), dtype=np.float32)
        rots = np.eye(3, dtype=np.float32)[None, ...]
        ctf_params = np.zeros((1, 9), dtype=np.float32)

        out_skip = np.asarray(core_forward.forward_model(config, volume, ctf_params, rots, skip_ctf=True))
        out_ctf = np.asarray(core_forward.forward_model(config, volume, ctf_params, rots, skip_ctf=False))
        np.testing.assert_allclose(out_ctf, 2.0 * out_skip)

    def test_adjoint_forward_model_shape(self):
        config = _make_config()
        rots = np.eye(3, dtype=np.float32)[None, ...]
        ctf_params = np.zeros((1, 9), dtype=np.float32)
        slices = np.zeros((1, IMAGE_SHAPE[0] * IMAGE_SHAPE[1]), dtype=np.float32)

        out = np.asarray(core_forward.adjoint_forward_model(config, slices, ctf_params, rots, skip_ctf=True))
        assert out.shape == (np.prod(VOLUME_SHAPE),)

    def test_forward_model_and_adjoint_contracts(self):
        config = _make_config()
        volume = np.zeros(np.prod(VOLUME_SHAPE), dtype=np.float32)
        rots = np.eye(3, dtype=np.float32)[None, ...]
        ctf_params = np.zeros((1, 9), dtype=np.float32)

        slices, adj = core_forward.forward_model_and_adjoint(
            config,
            volume,
            ctf_params,
            rots,
            skip_ctf=True,
        )
        assert np.asarray(slices).shape == (1, IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
        pulled = adj(np.ones_like(np.asarray(slices)))[0]
        assert np.asarray(pulled).shape == (np.prod(VOLUME_SHAPE),)

    def test_compute_AtAv_shape(self):
        config = _make_config()
        volume = np.ones(np.prod(VOLUME_SHAPE), dtype=np.float32)
        rots = np.eye(3, dtype=np.float32)[None, ...]
        ctf_params = np.zeros((1, 9), dtype=np.float32)

        out = core_forward.compute_AtAv(
            config,
            volume,
            ctf_params,
            rots,
            noise_variance=1.0,
            skip_ctf=True,
        )
        assert isinstance(out, tuple)
        assert np.asarray(out[0]).shape == (np.prod(VOLUME_SHAPE),)
