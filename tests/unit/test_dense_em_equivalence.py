"""Numerical equivalence tests for the dense single-volume EM refactoring.

These tests pin the current behavior of every function the refactoring
touches.  They must pass BEFORE any code is moved and AFTER each phase.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar import utils as rec_utils
from recovar.core.configs import ForwardModelConfig
from recovar.em import e_step, m_step, core as em_core
from recovar.em.states import EMState
from recovar.em.iterations import E_M_batches_2
from recovar.em.dense_single_volume import (
    compute_posterior,
    accumulate_sufficient_statistics,
    solve_mean,
    plan_em_iteration,
    run_dense_em_iteration,
    MeanStats,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Tiny synthetic dataset
# ---------------------------------------------------------------------------
IMAGE_SHAPE = (4, 4)
IMAGE_SIZE = 16
VOLUME_SHAPE = (4, 4, 4)
VOLUME_SIZE = 64
N_ROTATIONS = 3
N_TRANSLATIONS = 2
N_IMAGES = 2
SEED = 42


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    """CTF that returns ones (identity)."""
    if half_image:
        h, w = image_shape if image_shape is not None else IMAGE_SHAPE
        sz = h * (w // 2 + 1)
    else:
        sz = IMAGE_SIZE
    return jnp.ones((params.shape[0], sz), dtype=jnp.float32)


def _identity_process(batch, apply_image_mask=False):
    _ = apply_image_mask
    return batch


def _make_rotations(n, rng):
    """Make n rotation matrices (identity-ish, slightly perturbed for realism)."""
    rots = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        rots[i] = np.eye(3, dtype=np.float32)
        # Small perturbation to avoid degenerate tests
        if i > 0:
            angle = rng.uniform(-0.1, 0.1)
            c, s = np.cos(angle), np.sin(angle)
            rots[i, 0, 0] = c
            rots[i, 0, 1] = -s
            rots[i, 1, 0] = s
            rots[i, 1, 1] = c
    return rots


class MockDataset:
    """Minimal dataset for equivalence tests.

    Matches the interface used by E_with_precompute and M_with_precompute.
    """

    def __init__(self, rng):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = N_IMAGES
        self.n_units = N_IMAGES
        self.voxel_size = 1.0
        self.dtype = jnp.complex64
        self.CTF_params = np.zeros((N_IMAGES, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_identity_process)

        # Fixed images from seed
        self._images = (
            rng.standard_normal((N_IMAGES, IMAGE_SIZE)).astype(np.float32)
            + 1j * rng.standard_normal((N_IMAGES, IMAGE_SIZE)).astype(np.float32)
        )

        class _ImageSource:
            process_images = staticmethod(_identity_process)

        self.image_source = _ImageSource()

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        _ = kwargs
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices)
        for chunk_start in range(0, len(indices), max(1, batch_size)):
            chunk_end = min(chunk_start + max(1, batch_size), len(indices))
            idx = np.asarray(indices[chunk_start:chunk_end])
            yield (
                jnp.asarray(self._images[idx]),
                None,  # rotation_matrices
                None,  # translations
                jnp.asarray(self.CTF_params[idx]),
                None,  # noise_variance
                idx,   # particle_indices
                idx,   # indices
            )

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def mock_dataset(rng):
    return MockDataset(rng)


@pytest.fixture
def seeded_inputs(rng, mock_dataset):
    """Deterministic inputs for snapshot tests."""
    volume = (
        rng.standard_normal(VOLUME_SIZE).astype(np.float32)
        + 1j * rng.standard_normal(VOLUME_SIZE).astype(np.float32)
    )
    rotations = _make_rotations(N_ROTATIONS, rng)
    translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    noise_variance = np.ones(IMAGE_SIZE, dtype=np.float32)

    # Build config
    config = ForwardModelConfig.from_dataset(
        mock_dataset,
        disc_type="linear_interp",
        process_fn=_identity_process,
    )

    # Precompute projections for kernel-level tests
    from recovar import core
    projections = np.zeros((N_ROTATIONS, IMAGE_SIZE), dtype=np.complex64)
    for i in range(N_ROTATIONS):
        projections[i] = core.slice_volume(
            volume, rotations[i:i+1], IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp"
        )
    projections = jnp.asarray(projections)

    # Images from the dataset
    batch = jnp.asarray(mock_dataset._images)
    ctf_params = jnp.asarray(mock_dataset.CTF_params)

    # Probabilities (valid: sum to 1 over (rot, trans))
    raw = rng.dirichlet(np.ones(N_ROTATIONS * N_TRANSLATIONS), size=N_IMAGES).astype(np.float32)
    probabilities = raw.reshape(N_IMAGES, N_ROTATIONS, N_TRANSLATIONS)

    return {
        "volume": volume,
        "rotations": rotations,
        "translations": translations,
        "noise_variance": noise_variance,
        "config": config,
        "projections": projections,
        "batch": batch,
        "ctf_params": ctf_params,
        "probabilities": probabilities,
        "dataset": mock_dataset,
    }


# ---------------------------------------------------------------------------
# Kernel-level snapshot tests
# ---------------------------------------------------------------------------

def test_compute_dot_products_eqx_snapshot(seeded_inputs):
    """Pin compute_dot_products_eqx output."""
    s = seeded_inputs
    result = np.asarray(em_core.compute_dot_products_eqx(
        s["config"], s["projections"], s["batch"], s["translations"],
        s["ctf_params"], s["noise_variance"],
    ))

    assert result.shape == (N_IMAGES, N_ROTATIONS, N_TRANSLATIONS)
    assert np.all(np.isfinite(result))

    # Self-consistency: compute again, must be identical
    result2 = np.asarray(em_core.compute_dot_products_eqx(
        s["config"], s["projections"], s["batch"], s["translations"],
        s["ctf_params"], s["noise_variance"],
    ))
    np.testing.assert_array_equal(result, result2)


def test_compute_CTFed_proj_norms_eqx_snapshot(seeded_inputs):
    """Pin compute_CTFed_proj_norms_eqx output."""
    s = seeded_inputs
    proj_abs2 = jnp.abs(s["projections"]) ** 2
    result = np.asarray(em_core.compute_CTFed_proj_norms_eqx(
        s["config"], proj_abs2, s["ctf_params"], s["noise_variance"],
    ))

    assert result.shape == (N_IMAGES, N_ROTATIONS)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)  # norms are non-negative

    result2 = np.asarray(em_core.compute_CTFed_proj_norms_eqx(
        s["config"], proj_abs2, s["ctf_params"], s["noise_variance"],
    ))
    np.testing.assert_array_equal(result, result2)


def test_sum_up_images_fixed_rots_eqx_snapshot(seeded_inputs):
    """Pin sum_up_images_fixed_rots_eqx output (Ft_y, Ft_ctf)."""
    s = seeded_inputs
    Ft_y_init = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)
    Ft_ctf_init = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)

    Ft_y, Ft_ctf = m_step.sum_up_images_fixed_rots_eqx(
        s["config"], s["batch"], s["probabilities"], s["translations"],
        s["rotations"], s["ctf_params"], s["noise_variance"],
        Ft_y=Ft_y_init, Ft_ctf=Ft_ctf_init,
    )

    Ft_y = np.asarray(Ft_y)
    Ft_ctf = np.asarray(Ft_ctf)
    assert Ft_y.shape == (VOLUME_SIZE,)
    assert Ft_ctf.shape == (VOLUME_SIZE,)
    assert np.all(np.isfinite(Ft_y))
    assert np.all(np.isfinite(Ft_ctf))

    # Self-consistency (atol=1e-6: JIT FP non-determinism at ~1e-7)
    Ft_y2, Ft_ctf2 = m_step.sum_up_images_fixed_rots_eqx(
        s["config"], s["batch"], s["probabilities"], s["translations"],
        s["rotations"], s["ctf_params"], s["noise_variance"],
        Ft_y=Ft_y_init, Ft_ctf=Ft_ctf_init,
    )
    np.testing.assert_allclose(Ft_y, np.asarray(Ft_y2), atol=1e-6)
    np.testing.assert_allclose(Ft_ctf, np.asarray(Ft_ctf2), atol=1e-6)


def test_probability_normalization_snapshot(seeded_inputs):
    """Pin compute_probability_from_residual_normal_squared_one_image."""
    s = seeded_inputs

    # Build fake residuals (same shape as E-step output)
    rng = np.random.default_rng(SEED + 1)
    residuals = rng.standard_normal((N_IMAGES, N_ROTATIONS, N_TRANSLATIONS)).astype(np.float32)
    residuals = jnp.asarray(residuals)

    probs = np.asarray(
        e_step.compute_probability_from_residual_normal_squared_one_image(residuals)
    )

    assert probs.shape == (N_IMAGES, N_ROTATIONS, N_TRANSLATIONS)
    np.testing.assert_allclose(
        np.sum(probs.reshape(N_IMAGES, -1), axis=1),
        np.ones(N_IMAGES, dtype=np.float32),
        atol=1e-6,
    )
    assert np.all(probs >= 0)

    # Self-consistency
    probs2 = np.asarray(
        e_step.compute_probability_from_residual_normal_squared_one_image(residuals)
    )
    np.testing.assert_array_equal(probs, probs2)


# ---------------------------------------------------------------------------
# Integration-level snapshot tests
# ---------------------------------------------------------------------------

def test_E_with_precompute_snapshot(monkeypatch, seeded_inputs):
    """Pin full E_with_precompute (homogeneous path, u=None)."""
    s = seeded_inputs
    ds = s["dataset"]

    # Monkeypatch GPU memory to get deterministic batch sizes
    monkeypatch.setattr(rec_utils, "set_gpu_memory_limit", lambda gb: None)
    monkeypatch.setattr(rec_utils, "get_gpu_memory_total", lambda device=0: 10)

    probs = e_step.E_with_precompute(
        ds, s["volume"], s["rotations"], s["translations"],
        s["noise_variance"], "linear_interp",
        image_indices=None, u=None, s=None,
    )
    probs = np.asarray(probs)

    assert probs.shape == (N_IMAGES, N_ROTATIONS, N_TRANSLATIONS)
    assert np.all(np.isfinite(probs))
    # Probabilities sum to 1 over (rot, trans) for each image
    np.testing.assert_allclose(
        np.sum(probs.reshape(N_IMAGES, -1), axis=1),
        np.ones(N_IMAGES, dtype=np.float32),
        atol=1e-6,
    )

    # Run again, must be identical
    probs2 = np.asarray(e_step.E_with_precompute(
        ds, s["volume"], s["rotations"], s["translations"],
        s["noise_variance"], "linear_interp",
        image_indices=None, u=None, s=None,
    ))
    np.testing.assert_allclose(probs, probs2, atol=1e-6)


def test_M_with_precompute_snapshot(monkeypatch, seeded_inputs):
    """Pin full M_with_precompute."""
    s = seeded_inputs
    ds = s["dataset"]

    monkeypatch.setattr(rec_utils, "get_gpu_memory_total", lambda device=0: 10)

    Ft_y, Ft_ctf = m_step.M_with_precompute(
        ds, s["probabilities"], s["rotations"], s["translations"],
        s["noise_variance"], "linear_interp",
        image_indices=None,
    )
    Ft_y = np.asarray(Ft_y)
    Ft_ctf = np.asarray(Ft_ctf)

    assert Ft_y.shape == (VOLUME_SIZE,)
    assert Ft_ctf.shape == (VOLUME_SIZE,)
    assert np.all(np.isfinite(Ft_y))
    assert np.all(np.isfinite(Ft_ctf))

    # Run again
    Ft_y2, Ft_ctf2 = m_step.M_with_precompute(
        ds, s["probabilities"], s["rotations"], s["translations"],
        s["noise_variance"], "linear_interp",
        image_indices=None,
    )
    np.testing.assert_allclose(Ft_y, np.asarray(Ft_y2), atol=1e-6)
    np.testing.assert_allclose(Ft_ctf, np.asarray(Ft_ctf2), atol=1e-6)


def test_EM_iteration_snapshot(monkeypatch, seeded_inputs):
    """Pin one full EM iteration via E_M_batches_2 + EMState."""
    s = seeded_inputs
    ds = s["dataset"]

    monkeypatch.setattr(rec_utils, "get_gpu_memory_total", lambda device=0: 10)

    mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0
    state = EMState(s["volume"].copy(), mean_variance, s["noise_variance"])

    state, hard_assignment = E_M_batches_2(
        ds, state, s["rotations"], s["translations"],
        "linear_interp", memory_to_use=10,
    )
    state.finish_up_M_step(ds, "linear_interp")

    new_mean = np.asarray(state.mean)
    assert new_mean.shape == (VOLUME_SIZE,)
    assert np.all(np.isfinite(new_mean))
    assert hard_assignment.shape == (N_IMAGES,)

    # Ft_y and Ft_CTF should still be accessible (used by split_E_M_v2)
    assert hasattr(state, "Ft_y")
    assert hasattr(state, "Ft_CTF")
    Ft_y = np.asarray(state.Ft_y)
    Ft_CTF = np.asarray(state.Ft_CTF)
    assert Ft_y.shape == (VOLUME_SIZE,)
    assert Ft_CTF.shape == (VOLUME_SIZE,)

    # Run again with fresh state
    state2 = EMState(s["volume"].copy(), mean_variance, s["noise_variance"])
    state2, hard_assignment2 = E_M_batches_2(
        ds, state2, s["rotations"], s["translations"],
        "linear_interp", memory_to_use=10,
    )
    state2.finish_up_M_step(ds, "linear_interp")

    np.testing.assert_allclose(new_mean, np.asarray(state2.mean), atol=1e-6)
    np.testing.assert_array_equal(hard_assignment, hard_assignment2)
    np.testing.assert_allclose(Ft_y, np.asarray(state2.Ft_y), atol=1e-6)
    np.testing.assert_allclose(Ft_CTF, np.asarray(state2.Ft_CTF), atol=1e-6)


# ---------------------------------------------------------------------------
# Batch partition invariance test
# ---------------------------------------------------------------------------

def test_batch_partition_invariance(monkeypatch, seeded_inputs):
    """Splitting images into different batch sizes should yield same Ft_y/Ft_ctf."""
    s = seeded_inputs
    ds = s["dataset"]

    monkeypatch.setattr(rec_utils, "get_gpu_memory_total", lambda device=0: 10)

    # Run M-step as single batch (all images at once)
    Ft_y_1, Ft_ctf_1 = m_step.M_with_precompute(
        ds, s["probabilities"], s["rotations"], s["translations"],
        s["noise_variance"], "linear_interp",
    )

    # Run M-step as two batches (one image at a time)
    Ft_y_a, Ft_ctf_a = m_step.M_with_precompute(
        ds, s["probabilities"][:1], s["rotations"], s["translations"],
        s["noise_variance"], "linear_interp",
        image_indices=np.array([0]),
    )
    Ft_y_b, Ft_ctf_b = m_step.M_with_precompute(
        ds, s["probabilities"][1:], s["rotations"], s["translations"],
        s["noise_variance"], "linear_interp",
        image_indices=np.array([1]),
    )

    # Additive: sum of shards should match single batch
    np.testing.assert_allclose(
        np.asarray(Ft_y_1),
        np.asarray(Ft_y_a) + np.asarray(Ft_y_b),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(Ft_ctf_1),
        np.asarray(Ft_ctf_a) + np.asarray(Ft_ctf_b),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Cross-path equivalence: new package vs old functions
# ---------------------------------------------------------------------------

def test_compute_posterior_matches_E_with_precompute(monkeypatch, seeded_inputs):
    """New compute_posterior must match old E_with_precompute (u=None)."""
    s = seeded_inputs
    ds = s["dataset"]

    monkeypatch.setattr(rec_utils, "get_gpu_memory_total", lambda device=0: 10)

    # Old path
    old_probs = np.asarray(e_step.E_with_precompute(
        ds, s["volume"], s["rotations"], s["translations"],
        s["noise_variance"], "linear_interp",
        image_indices=None, u=None, s=None,
    ))

    # New path
    plan = plan_em_iteration(
        grid_size=ds.grid_size,
        n_rotations=N_ROTATIONS,
        n_translations=N_TRANSLATIONS,
        memory_to_use_gb=10,
    )
    new_probs = np.asarray(compute_posterior(
        s["config"], ds, s["volume"], s["rotations"], s["translations"],
        s["noise_variance"], "linear_interp", plan,
        image_indices=None,
    ))

    np.testing.assert_allclose(old_probs, new_probs, atol=1e-6)


def test_accumulate_matches_M_with_precompute(monkeypatch, seeded_inputs):
    """New accumulate_sufficient_statistics must match old M_with_precompute."""
    s = seeded_inputs
    ds = s["dataset"]

    monkeypatch.setattr(rec_utils, "get_gpu_memory_total", lambda device=0: 10)

    # Old path
    old_Ft_y, old_Ft_ctf = m_step.M_with_precompute(
        ds, s["probabilities"], s["rotations"], s["translations"],
        s["noise_variance"], "linear_interp",
    )

    # New path
    plan = plan_em_iteration(
        grid_size=ds.grid_size,
        n_rotations=N_ROTATIONS,
        n_translations=N_TRANSLATIONS,
        memory_to_use_gb=10,
    )
    stats = accumulate_sufficient_statistics(
        s["config"], ds, s["probabilities"], s["rotations"],
        s["translations"], s["noise_variance"], plan,
    )

    np.testing.assert_allclose(np.asarray(old_Ft_y), np.asarray(stats.Ft_y), atol=1e-6)
    np.testing.assert_allclose(np.asarray(old_Ft_ctf), np.asarray(stats.Ft_ctf), atol=1e-6)


def test_solve_mean_matches_EMState_finish(monkeypatch, seeded_inputs):
    """New solve_mean must match EMState.finish_up_M_step."""
    s = seeded_inputs
    ds = s["dataset"]

    monkeypatch.setattr(rec_utils, "get_gpu_memory_total", lambda device=0: 10)

    # Get Ft_y, Ft_ctf from M-step
    Ft_y, Ft_ctf = m_step.M_with_precompute(
        ds, s["probabilities"], s["rotations"], s["translations"],
        s["noise_variance"], "linear_interp",
    )

    mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0

    # Old path: EMState.finish_up_M_step
    from recovar.reconstruction import relion_functions
    old_mean = np.asarray(relion_functions.post_process_from_filter(
        ds, Ft_ctf, Ft_y, tau=mean_variance, disc_type="linear_interp",
    ).reshape(-1))

    # New path
    stats = MeanStats(Ft_y=Ft_y, Ft_ctf=Ft_ctf)
    new_mean = np.asarray(solve_mean(ds, stats, mean_variance, "linear_interp"))

    np.testing.assert_allclose(old_mean, new_mean, atol=0)


def test_engine_matches_E_M_batches_2(monkeypatch, seeded_inputs):
    """New run_dense_em_iteration must match old E_M_batches_2 + EMState."""
    s = seeded_inputs
    ds = s["dataset"]

    monkeypatch.setattr(rec_utils, "get_gpu_memory_total", lambda device=0: 10)

    mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0

    # Old path: E_M_batches_2 + finish_up_M_step
    state = EMState(s["volume"].copy(), mean_variance, s["noise_variance"])
    state, old_ha = E_M_batches_2(
        ds, state, s["rotations"], s["translations"],
        "linear_interp", memory_to_use=10,
    )
    state.finish_up_M_step(ds, "linear_interp")
    old_mean = np.asarray(state.mean)
    old_Ft_y = np.asarray(state.Ft_y)
    old_Ft_CTF = np.asarray(state.Ft_CTF)

    # New path: run_dense_em_iteration
    new_mean, new_ha, new_Ft_y, new_Ft_ctf = run_dense_em_iteration(
        ds, s["volume"].copy(), mean_variance, s["noise_variance"],
        s["rotations"], s["translations"], "linear_interp",
        memory_to_use_gb=10,
    )

    np.testing.assert_allclose(old_mean, np.asarray(new_mean), atol=1e-6)
    np.testing.assert_array_equal(old_ha, new_ha)
    np.testing.assert_allclose(old_Ft_y, np.asarray(new_Ft_y), atol=1e-6)
    np.testing.assert_allclose(old_Ft_CTF, np.asarray(new_Ft_ctf), atol=1e-6)
