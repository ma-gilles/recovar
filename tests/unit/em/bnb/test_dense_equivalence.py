"""Phase-2 dense-equivalence gate: BnB with no pruning == direct local EM.

The BnB engine is a *support selector* on top of ``run_local_em_exact``. When
pruning is disabled (``posterior_tail_tol=1.0``, all caps at 1.0, large
floors) every (rotation, translation) candidate must survive, and the final
local-EM run must produce the exact same ``Ft_y``, ``Ft_ctf`` and
``hard_assignment`` as a directly-built ``LocalHypothesisLayout`` containing
every (rotation, translation) for every image.

We compare BnB against ``run_local_em_exact`` (rather than against
``run_em``) so the test isolates BnB's support-selection logic from any
half-volume/Wiener convention differences between the dense and local M-step
backends. A separate test in the test_refine_relion_mode suite already pins
``run_em`` vs ``run_local_em_exact`` equivalence at the layout level.

This test mirrors the synthetic-dataset pattern from
``tests/unit/test_half_spectrum_em.py`` (``MockDataset``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume.bnb import (
    BranchBoundOptions,
    build_bnb_local_layout,
    run_bnb_em_k1,
    select_bnb_support_fixed_grid_k1,
)
from recovar.em.dense_single_volume.local_em_engine import run_local_em_exact
from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512
H, W = IMAGE_SHAPE
N_HALF = H * (W // 2 + 1)
N_ROTATIONS = 5
N_TRANSLATIONS = 3
N_IMAGES = 4
SEED = 42


def _raw_real_image_2d(image_shape, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(image_shape).astype(np.float32)


def _hermitian_volume(volume_shape, seed=42):
    rng = np.random.default_rng(seed)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fftn(real_vol))
    return jnp.array(ft.ravel(), dtype=jnp.complex64)


def _make_rotations(n, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return jnp.array(q, dtype=jnp.float32)


def _identity_ctf_full(params, image_shape=None, voxel_size=None, *, half_image=False):
    if half_image:
        h, w = image_shape if image_shape is not None else IMAGE_SHAPE
        sz = h * (w // 2 + 1)
    else:
        sz = IMAGE_SIZE
    return jnp.ones((params.shape[0], sz), dtype=jnp.float32)


def _raw_real_process(batch, apply_image_mask=False):
    images = jnp.asarray(batch)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _raw_real_process_half(batch, apply_image_mask=False):
    images = jnp.asarray(batch)
    return ftu.get_dft2_real(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


class _MockDataset:
    """Minimal dataset for equivalence tests — same shape as tests/unit/test_half_spectrum_em.py."""

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
        self.ctf_evaluator = staticmethod(_identity_ctf_full)
        self.process_images = staticmethod(_raw_real_process)
        self.process_images_half = staticmethod(_raw_real_process_half)

        self._images = np.zeros((N_IMAGES, *IMAGE_SHAPE), dtype=np.float32)
        for i in range(N_IMAGES):
            self._images[i] = _raw_real_image_2d(IMAGE_SHAPE, seed=rng.integers(10000))

        class _ImageSource:
            process_images = staticmethod(_raw_real_process)
            process_images_half = staticmethod(_raw_real_process_half)

        self.image_source = _ImageSource()

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices)
        for chunk_start in range(0, len(indices), max(1, batch_size)):
            chunk_end = min(chunk_start + max(1, batch_size), len(indices))
            idx = np.asarray(indices[chunk_start:chunk_end])
            yield (
                jnp.asarray(self._images[idx]),
                None,
                None,
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def mock_dataset(rng):
    return _MockDataset(rng)


@pytest.fixture
def seeded_inputs(rng, mock_dataset):
    volume = _hermitian_volume(VOLUME_SHAPE, seed=42)
    rotations = _make_rotations(N_ROTATIONS, seed=12)
    translations = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=jnp.float32,
    )
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    return {
        "volume": volume,
        "rotations": rotations,
        "translations": translations,
        "noise_variance": noise_variance,
        "dataset": mock_dataset,
    }


def _no_pruning_options() -> BranchBoundOptions:
    """Configure BnB so every candidate survives every stage."""
    return BranchBoundOptions(
        enabled=True,
        # Tail-mass margin so loose that nothing gets dropped.
        posterior_tail_tol=1.0,
        # Caps disabled (fraction > 1 means no cap).
        max_orientation_fraction=1.0,
        max_shift_fraction=1.0,
        # Floors large enough to force restoration if anything sneaks past.
        min_orientations_per_image=N_ROTATIONS,
        min_shifts_per_image=N_TRANSLATIONS,
        min_joint_candidates_per_image=N_ROTATIONS * N_TRANSLATIONS,
        max_joint_candidates_per_image=N_ROTATIONS * N_TRANSLATIONS,
        return_diagnostics=True,
    )


def _build_full_layout(rotations: np.ndarray, translations: np.ndarray) -> LocalHypothesisLayout:
    """A LocalHypothesisLayout where every image has every (rotation, translation)."""
    rotations = np.asarray(rotations, dtype=np.float32)
    translations = np.asarray(translations, dtype=np.float32)
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])
    rotation_offsets = np.array(
        [i * n_rot for i in range(N_IMAGES + 1)], dtype=np.int64,
    )
    rotation_ids_flat = np.tile(np.arange(n_rot, dtype=np.int32), N_IMAGES)
    rotations_flat = np.tile(rotations[None, :], (N_IMAGES, 1, 1, 1)).reshape(
        N_IMAGES * n_rot, 3, 3,
    )
    rotation_log_priors_flat = np.zeros(N_IMAGES * n_rot, dtype=np.float32)
    rotation_counts = np.full(N_IMAGES, n_rot, dtype=np.int32)
    translation_log_priors = np.zeros((N_IMAGES, n_trans), dtype=np.float32)
    sample_mask_flat = np.ones((N_IMAGES * n_rot, n_trans), dtype=bool)
    return LocalHypothesisLayout(
        n_global_rotations=n_rot,
        n_pixels=0,
        n_psi=0,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=rotation_counts,
        translation_grid=translations,
        translation_log_priors=translation_log_priors,
        rotation_posterior_ids_flat=None,
        sample_mask_flat=sample_mask_flat,
    )


def test_bnb_support_no_pruning_includes_everything(seeded_inputs):
    """With pruning disabled, every (rot, trans) candidate survives."""
    s = seeded_inputs
    options = _no_pruning_options()
    support = select_bnb_support_fixed_grid_k1(
        s["dataset"],
        s["volume"],
        s["noise_variance"],
        np.asarray(s["rotations"]),
        s["translations"],
        current_size=None,
        options=options,
        image_batch_size=N_IMAGES,
        rotation_block_size=N_ROTATIONS,
    )
    assert support.sample_mask_per_image.shape == (N_IMAGES, N_ROTATIONS, N_TRANSLATIONS)
    assert np.all(support.sample_mask_per_image), "Some candidates were pruned despite no-pruning options"


def test_bnb_layout_no_pruning_matches_hand_built(seeded_inputs):
    """build_bnb_local_layout output == hand-built full layout (no pruning)."""
    s = seeded_inputs
    options = _no_pruning_options()
    support = select_bnb_support_fixed_grid_k1(
        s["dataset"],
        s["volume"],
        s["noise_variance"],
        np.asarray(s["rotations"]),
        s["translations"],
        current_size=None,
        options=options,
        image_batch_size=N_IMAGES,
        rotation_block_size=N_ROTATIONS,
    )
    bnb_layout = build_bnb_local_layout(
        support,
        np.asarray(s["rotations"]),
        np.asarray(s["translations"]),
    )
    full_layout = _build_full_layout(s["rotations"], s["translations"])

    np.testing.assert_array_equal(bnb_layout.rotation_offsets, full_layout.rotation_offsets)
    np.testing.assert_array_equal(bnb_layout.rotation_counts, full_layout.rotation_counts)
    np.testing.assert_array_equal(bnb_layout.rotation_ids_flat, full_layout.rotation_ids_flat)
    np.testing.assert_allclose(bnb_layout.rotations_flat, full_layout.rotations_flat, atol=1e-7)
    np.testing.assert_array_equal(bnb_layout.sample_mask_flat, full_layout.sample_mask_flat)


def test_bnb_no_pruning_matches_local_em_exact(seeded_inputs):
    """BnB(no pruning) == direct run_local_em_exact on the full layout.

    This is the load-bearing Phase-2 gate: it shows that the BnB engine is
    a faithful wrapper around the existing local engine — same Ft_y, same
    Ft_ctf, same hard assignment.
    """
    s = seeded_inputs
    ds = s["dataset"]
    volume = s["volume"]
    rotations = np.asarray(s["rotations"])
    translations = s["translations"]
    noise_variance = s["noise_variance"]
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0

    full_layout = _build_full_layout(rotations, np.asarray(translations))
    ref = run_local_em_exact(
        ds,
        volume,
        mean_variance,
        noise_variance,
        full_layout,
        "linear_interp",
        image_batch_size=N_IMAGES,
        rotation_block_size=N_ROTATIONS,
        current_size=None,
        accumulate_noise=True,
        return_best_pose_details=True,
        score_with_masked_images=False,
    )

    bnb_ret = run_bnb_em_k1(
        ds, volume, mean_variance, noise_variance,
        rotations, translations, "linear_interp",
        current_size=None,
        options=_no_pruning_options(),
        image_batch_size=N_IMAGES,
        rotation_block_size=N_ROTATIONS,
        return_best_pose_details=True,
        score_with_masked_images=False,
    )

    # Both tuples are: (Ft_y, Ft_ctf, hard_assignment, best_pose_rotations,
    # best_pose_translations, best_pose_rotation_ids, relion_stats,
    # noise_stats).
    np.testing.assert_allclose(np.asarray(bnb_ret[0]), np.asarray(ref[0]), rtol=1e-6, atol=1e-6,
                                err_msg="Ft_y mismatch")
    np.testing.assert_allclose(np.asarray(bnb_ret[1]), np.asarray(ref[1]), rtol=1e-6, atol=1e-6,
                                err_msg="Ft_ctf mismatch")
    np.testing.assert_array_equal(np.asarray(bnb_ret[2]), np.asarray(ref[2]),
                                   err_msg="hard_assignment mismatch")
