import jax.numpy as jnp
import numpy as np
import pytest

from recovar.core.configs import ForwardModelConfig
from recovar.core.ctf import as_ctf_evaluator
from recovar.em.ppca_refinement.dense_dataset import _project_augmented_half_volumes, iter_dense_ppca_dataset_blocks
from recovar.em.ppca_refinement.engine import dense_pose_ppca_E_step_blocked, dense_pose_ppca_logZ_blocked
from recovar.em.ppca_refinement.initialization import real_volume_to_centered_fourier_half
from recovar.em.dense_single_volume.helpers.preprocessing import half_translation_phase_table, preprocess_batch
from recovar.em.sampling import get_rotation_grid_at_order


pytestmark = pytest.mark.unit


def _complex_normal(rng, shape, *, scale=1.0):
    real = rng.normal(size=shape)
    imag = rng.normal(size=shape)
    return (scale * (real + 1j * imag)).astype(np.complex64)


def _weighted_score_inputs(raw_shifted_images, raw_image_norms, precision):
    """Convert raw half-images to the dense PPCA score convention."""
    return (
        jnp.asarray(float(precision) * raw_shifted_images, dtype=jnp.complex64),
        jnp.asarray(np.full((raw_shifted_images.shape[0], raw_shifted_images.shape[-1]), precision), dtype=jnp.float32),
        jnp.asarray(float(precision) * raw_image_norms, dtype=jnp.float32),
    )


def _pack_upper_tri_numpy(matrix):
    tri_i, tri_j = np.triu_indices(matrix.shape[-1])
    return matrix[..., tri_i, tri_j]


def _numpy_ppca_pose_reference(Y1, proj_aug, ctf2_over_noise, y_norm, pose_log_prior=None):
    Y1 = np.asarray(Y1)
    proj_aug = np.asarray(proj_aug)
    ctf2_over_noise = np.asarray(ctf2_over_noise)
    y_norm = np.asarray(y_norm)
    n_images, n_trans, _n_freq = Y1.shape
    n_rot, n_aug, _ = proj_aug.shape
    q = n_aug - 1

    score = np.empty((n_images, n_trans, n_rot), dtype=np.float64)
    alpha = np.empty((n_images, n_trans, n_rot, n_aug), dtype=np.complex128)
    G_tri = np.empty((n_images, n_trans, n_rot, n_aug * (n_aug + 1) // 2), dtype=np.complex128)
    for b in range(n_images):
        for t in range(n_trans):
            for r in range(n_rot):
                mu = proj_aug[r, 0]
                W = proj_aug[r, 1:]
                rho = (
                    y_norm[b]
                    - 2.0 * np.sum(np.conj(Y1[b, t]) * mu).real
                    + np.sum(ctf2_over_noise[b] * np.conj(mu) * mu).real
                )
                if q == 0:
                    score[b, t, r] = -0.5 * rho
                    alpha[b, t, r] = np.ones((1,), dtype=np.complex128)
                    G_tri[b, t, r] = np.ones((1,), dtype=np.complex128)
                    continue

                g_zx = np.einsum("f,qf->q", Y1[b, t], np.conj(W)).real
                h_zm = np.einsum("f,qf,f->q", ctf2_over_noise[b], np.conj(W), mu).real
                Hzz = np.einsum("f,qf,pf->qp", ctf2_over_noise[b], np.conj(W), W).real
                Hzz = 0.5 * (Hzz + np.swapaxes(np.conj(Hzz), -1, -2))
                M = np.eye(q, dtype=np.complex128) + Hzz
                chol = np.linalg.cholesky(M)
                b_vec = g_zx - h_zm
                z_bar = np.linalg.solve(M, b_vec)
                S_z = np.linalg.solve(M, np.eye(q, dtype=np.complex128))
                logdet_M = 2.0 * np.sum(np.log(np.real(np.diag(chol))))
                quad = np.sum(np.conj(b_vec) * z_bar).real
                score[b, t, r] = -0.5 * (rho - quad + logdet_M)

                alpha[b, t, r] = np.concatenate([np.ones((1,), dtype=np.complex128), z_bar])
                G = np.empty((n_aug, n_aug), dtype=np.complex128)
                G[0, 0] = 1.0
                G[0, 1:] = z_bar
                G[1:, 0] = z_bar
                G[1:, 1:] = S_z + z_bar[:, None] * z_bar[None, :]
                G_tri[b, t, r] = _pack_upper_tri_numpy(G)

    if pose_log_prior is not None:
        score = score + np.swapaxes(np.asarray(pose_log_prior), -1, -2)

    score_flat = score.reshape(n_images, n_trans * n_rot)
    score_max = np.max(score_flat, axis=1, keepdims=True)
    logZ = np.squeeze(score_max, axis=1) + np.log(np.sum(np.exp(score_flat - score_max), axis=1))
    gamma = np.exp(score - logZ[:, None, None])
    best_flat = np.argmax(score_flat, axis=1)
    return {
        "logZ": logZ,
        "pmax": np.max(gamma.reshape(n_images, n_trans * n_rot), axis=1),
        "best_rotation_idx": best_flat % n_rot,
        "best_translation_idx": best_flat // n_rot,
        "n_significant_per_image": np.sum(gamma > 1e-3, axis=(1, 2)),
        "alpha_aug_acc": np.einsum("btr,btrp->bp", gamma, alpha),
        "G_aug_tri_acc": np.einsum("btr,btrk->bk", gamma, G_tri),
    }


def test_dense_ppca_score_moments_and_prior_axes_match_numpy_reference():
    rng = np.random.default_rng(20260512)
    n_images, n_trans, n_rot, n_freq, q = 2, 3, 4, 5, 2
    Y1 = _complex_normal(rng, (n_images, n_trans, n_freq), scale=0.3)
    proj_aug = _complex_normal(rng, (n_rot, q + 1, n_freq), scale=0.2)
    proj_aug[:, 1:, :] *= 0.25
    ctf2_over_noise = rng.uniform(0.5, 2.0, size=(n_images, n_freq)).astype(np.float32)
    y_norm = rng.uniform(1.0, 2.0, size=(n_images,)).astype(np.float32)
    pose_log_prior = rng.normal(scale=0.07, size=(n_images, n_rot, n_trans)).astype(np.float32)

    stats, diagnostics = dense_pose_ppca_E_step_blocked(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2_over_noise),
        jnp.asarray(y_norm),
        pose_log_prior=jnp.asarray(pose_log_prior),
    )
    expected = _numpy_ppca_pose_reference(Y1, proj_aug, ctf2_over_noise, y_norm, pose_log_prior)

    np.testing.assert_allclose(np.asarray(diagnostics.logZ), expected["logZ"], rtol=2e-5, atol=2e-5)
    score_only_logZ = dense_pose_ppca_logZ_blocked(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2_over_noise),
        jnp.asarray(y_norm),
        pose_log_prior=jnp.asarray(pose_log_prior),
    )
    np.testing.assert_allclose(np.asarray(score_only_logZ), expected["logZ"], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(diagnostics.pmax), expected["pmax"], rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(np.asarray(diagnostics.best_rotation_idx), expected["best_rotation_idx"])
    np.testing.assert_array_equal(np.asarray(diagnostics.best_translation_idx), expected["best_translation_idx"])
    np.testing.assert_array_equal(
        np.asarray(diagnostics.n_significant_per_image),
        expected["n_significant_per_image"],
    )
    np.testing.assert_allclose(np.asarray(stats.alpha_aug_acc), expected["alpha_aug_acc"], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(np.asarray(stats.G_aug_tri_acc), expected["G_aug_tri_acc"], rtol=3e-5, atol=3e-5)


def test_dense_ppca_score_recovers_synthetic_rotation_translation_with_high_pmax():
    rng = np.random.default_rng(20260506)
    n_images, n_trans, n_rot, n_freq, q = 5, 4, 7, 12, 2
    precision = 30.0

    mean_proj = _complex_normal(rng, (n_rot, n_freq), scale=0.35)
    for r in range(n_rot):
        mean_proj[r, r % n_freq] += 3.0 + 0.25j * r
    W_proj = _complex_normal(rng, (n_rot, q, n_freq), scale=0.035)
    proj_aug = np.concatenate([mean_proj[:, None, :], W_proj], axis=1).astype(np.complex64)

    true_rot = np.asarray([0, 3, 6, 2, 5], dtype=np.int32)
    true_trans = np.asarray([1, 0, 3, 2, 1], dtype=np.int32)
    z_true = rng.normal(size=(n_images, q)).astype(np.float32) * 0.7

    raw_shifted = np.zeros((n_images, n_trans, n_freq), dtype=np.complex64)
    raw_norm = np.zeros((n_images,), dtype=np.float32)
    for b in range(n_images):
        r = int(true_rot[b])
        t = int(true_trans[b])
        signal = proj_aug[r, 0] + np.einsum("q,qf->f", z_true[b], proj_aug[r, 1:])
        raw_shifted[b, t] = signal
        raw_norm[b] = np.sum(np.abs(signal) ** 2).real

    Y1, ctf2_over_noise, y_norm = _weighted_score_inputs(raw_shifted, raw_norm, precision)
    _stats, diagnostics = dense_pose_ppca_E_step_blocked(
        Y1,
        jnp.asarray(proj_aug),
        ctf2_over_noise,
        y_norm,
    )

    np.testing.assert_array_equal(np.asarray(diagnostics.best_rotation_idx), true_rot)
    np.testing.assert_array_equal(np.asarray(diagnostics.best_translation_idx), true_trans)
    assert np.all(np.asarray(diagnostics.pmax) > 0.999)
    np.testing.assert_array_equal(np.asarray(diagnostics.n_significant_per_image), np.ones(n_images, dtype=np.int32))


def test_dense_ppca_moments_recover_latent_coordinates_at_identifiable_pose():
    n_images, n_trans, n_rot, n_freq, q = 4, 1, 5, 12, 2
    precision = 200.0
    proj_aug = np.zeros((n_rot, q + 1, n_freq), dtype=np.complex64)
    for r in range(n_rot):
        proj_aug[r, 0, r] = 4.0 + 0.2j * r
        proj_aug[r, 1, 8] = 1.0
        proj_aug[r, 2, 9] = 1.0

    true_rot = np.asarray([0, 2, 4, 1], dtype=np.int32)
    z_true = np.asarray([[0.7, -0.2], [-0.5, 0.4], [0.25, 0.6], [-0.8, -0.3]], dtype=np.float32)
    raw_shifted = np.zeros((n_images, n_trans, n_freq), dtype=np.complex64)
    raw_norm = np.zeros((n_images,), dtype=np.float32)
    for b, r in enumerate(true_rot):
        signal = proj_aug[int(r), 0] + np.einsum("q,qf->f", z_true[b], proj_aug[int(r), 1:])
        raw_shifted[b, 0] = signal
        raw_norm[b] = np.sum(np.abs(signal) ** 2).real

    Y1, ctf2_over_noise, y_norm = _weighted_score_inputs(raw_shifted, raw_norm, precision)
    stats, diagnostics = dense_pose_ppca_E_step_blocked(
        Y1,
        jnp.asarray(proj_aug),
        ctf2_over_noise,
        y_norm,
    )

    np.testing.assert_array_equal(np.asarray(diagnostics.best_rotation_idx), true_rot)
    assert np.all(np.asarray(diagnostics.pmax) > 0.999)
    expected_posterior_mean = (precision / (1.0 + precision)) * z_true
    np.testing.assert_allclose(
        np.asarray(stats.alpha_aug_acc[:, 1:]).real,
        expected_posterior_mean,
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(np.asarray(stats.alpha_aug_acc[:, 1:]).imag, 0.0, atol=1e-6)


def test_dense_ppca_complex_fourier_real_latent_posterior_matches_closed_form():
    mu = np.asarray([0.7 + 0.2j, -0.3 + 0.4j, 0.1 - 0.5j], dtype=np.complex64)
    W = np.asarray([[0.2 + 0.5j, -0.6 + 0.1j, 0.4 - 0.3j]], dtype=np.complex64)
    z_true = np.asarray([0.8], dtype=np.float32)
    precision = np.float32(3.0)
    observed = mu + z_true[0] * W[0]

    Y1 = jnp.asarray((precision * observed)[None, None, :], dtype=jnp.complex64)
    ctf2_over_noise = jnp.asarray(np.full((1, observed.size), precision, dtype=np.float32))
    y_norm = jnp.asarray([precision * np.sum(np.abs(observed) ** 2).real], dtype=jnp.float32)
    proj_aug = jnp.asarray(np.concatenate([mu[None, None, :], W[None, :, :]], axis=1))

    stats, _diagnostics = dense_pose_ppca_E_step_blocked(Y1, proj_aug, ctf2_over_noise, y_norm)

    H = np.einsum("f,qf,pf->qp", np.asarray(ctf2_over_noise[0]), np.conj(W), W).real
    b = np.einsum("qf,f->q", np.conj(W), precision * (observed - mu)).real
    expected_z = np.linalg.solve(np.eye(1, dtype=np.float32) + H, b)
    np.testing.assert_allclose(np.asarray(stats.alpha_aug_acc[0, 1:]), expected_z, rtol=2e-5, atol=2e-5)


def _asymmetric_real_volume_bank(rng, q, volume_shape):
    grid = np.indices(volume_shape, dtype=np.float32)
    x, y, z = [(axis - (size - 1.0) / 2.0) / float(size) for axis, size in zip(grid, volume_shape)]
    mean = (
        0.75 * np.exp(-80.0 * ((x - 0.10) ** 2 + (y + 0.05) ** 2 + (z - 0.02) ** 2))
        - 0.55 * np.exp(-65.0 * ((x + 0.12) ** 2 + (y - 0.08) ** 2 + (z + 0.07) ** 2))
        + 0.20 * rng.normal(size=volume_shape)
    ).astype(np.float32)
    volumes = [mean]
    for k in range(q):
        pc = (
            (0.25 + 0.05 * k)
            * np.exp(-70.0 * ((x + 0.08 * k) ** 2 + (y - 0.09) ** 2 + (z + 0.05 * k) ** 2))
            + 0.05 * rng.normal(size=volume_shape)
        ).astype(np.float32)
        volumes.append(pc)
    return np.stack(volumes, axis=0)


def test_healpix_grid_projected_q0_score_recovers_exact_grid_rotation():
    rng = np.random.default_rng(20260507)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rotations = get_rotation_grid_at_order(0, n_in_planes=3, matrices=True).astype(np.float32)
    true_rot = np.asarray([2, 11, 25], dtype=np.int32)
    precision = 12.0

    volume_bank = _asymmetric_real_volume_bank(rng, 0, volume_shape)
    augmented_half = np.stack([real_volume_to_centered_fourier_half(volume_bank[0])], axis=0)
    proj_aug = _project_augmented_half_volumes(
        jnp.asarray(augmented_half),
        jnp.asarray(rotations),
        image_shape,
        volume_shape,
        "linear_interp",
        max_r=None,
        relion_texture_interp=False,
    )
    raw = np.asarray(proj_aug)[true_rot, 0, :]
    raw_shifted = raw[:, None, :]
    raw_norm = np.sum(np.abs(raw) ** 2, axis=1).real.astype(np.float32)

    Y1, ctf2_over_noise, y_norm = _weighted_score_inputs(raw_shifted, raw_norm, precision)
    _stats, diagnostics = dense_pose_ppca_E_step_blocked(Y1, proj_aug, ctf2_over_noise, y_norm)

    np.testing.assert_array_equal(np.asarray(diagnostics.best_rotation_idx), true_rot)
    np.testing.assert_array_equal(np.asarray(diagnostics.best_translation_idx), np.zeros(true_rot.shape, dtype=np.int32))
    assert np.all(np.asarray(diagnostics.pmax) > 0.999)


def test_healpix_grid_projected_q0_score_recovers_rotation_and_translation_phase():
    rng = np.random.default_rng(20260508)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rotations = get_rotation_grid_at_order(0, n_in_planes=3, matrices=True).astype(np.float32)
    translations = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    true_rot = np.asarray([2, 11, 25], dtype=np.int32)
    true_trans = np.asarray([0, 2, 3], dtype=np.int32)
    precision = 12.0

    volume_bank = _asymmetric_real_volume_bank(rng, 0, volume_shape)
    augmented_half = np.stack([real_volume_to_centered_fourier_half(volume_bank[0])], axis=0)
    proj_aug = _project_augmented_half_volumes(
        jnp.asarray(augmented_half),
        jnp.asarray(rotations),
        image_shape,
        volume_shape,
        "linear_interp",
        max_r=None,
        relion_texture_interp=False,
    )
    proj_true = np.asarray(proj_aug)[true_rot, 0, :]
    phases = np.asarray(half_translation_phase_table(jnp.asarray(translations), image_shape))
    observed = proj_true / phases[true_trans]
    raw_shifted = observed[:, None, :] * phases[None, :, :]
    raw_norm = np.sum(np.abs(proj_true) ** 2, axis=1).real.astype(np.float32)

    Y1, ctf2_over_noise, y_norm = _weighted_score_inputs(raw_shifted, raw_norm, precision)
    _stats, diagnostics = dense_pose_ppca_E_step_blocked(Y1, proj_aug, ctf2_over_noise, y_norm)

    np.testing.assert_array_equal(np.asarray(diagnostics.best_rotation_idx), true_rot)
    np.testing.assert_array_equal(np.asarray(diagnostics.best_translation_idx), true_trans)
    assert np.all(np.asarray(diagnostics.pmax) > 0.999)


class _FakeHalfImageDataset:
    def __init__(self, images_half, image_shape, volume_shape):
        self._images_half = np.asarray(images_half, dtype=np.complex64)
        self.image_shape = tuple(image_shape)
        self.volume_shape = tuple(volume_shape)

    def process_images_half(self, batch, apply_image_mask):
        del apply_image_mask
        return jnp.asarray(self._images_half[np.asarray(batch, dtype=np.int32)])


class _FakeDenseDataset(_FakeHalfImageDataset):
    def __init__(self, images_half, image_shape, volume_shape):
        super().__init__(images_half, image_shape, volume_shape)
        self.grid_size = int(volume_shape[0])
        self.voxel_size = 1.0
        self.padding = 0
        self.n_images = int(self._images_half.shape[0])
        self.n_units = self.n_images
        self.ctf_evaluator = as_ctf_evaluator(_identity_ctf)

    def process_images(self, batch, apply_image_mask=False):
        del apply_image_mask
        raise AssertionError("dense PPCA block generation should use process_images_half")

    def iter_batches(self, image_batch_size, indices, by_image):
        del image_batch_size, by_image
        idx = np.asarray(indices, dtype=np.int32)
        yield (
            idx,
            None,
            None,
            jnp.zeros((idx.size, 11), dtype=jnp.float32),
            None,
            None,
            idx,
        )


def _identity_ctf(ctf_params, image_shape, voxel_size, *, half_image=False):
    del voxel_size
    n_pixels = image_shape[0] * (image_shape[1] // 2 + 1) if half_image else int(np.prod(image_shape))
    return jnp.ones((ctf_params.shape[0], n_pixels), dtype=jnp.float32)


def test_preprocess_batch_identity_ctf_feeds_dense_ppca_translation_score():
    rng = np.random.default_rng(20260510)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rotations = get_rotation_grid_at_order(0, n_in_planes=3, matrices=True).astype(np.float32)
    translations = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    true_rot = np.asarray([5, 19], dtype=np.int32)
    true_trans = np.asarray([2, 0], dtype=np.int32)

    volume_bank = _asymmetric_real_volume_bank(rng, 0, volume_shape)
    augmented_half = np.stack([real_volume_to_centered_fourier_half(volume_bank[0])], axis=0)
    proj_aug = _project_augmented_half_volumes(
        jnp.asarray(augmented_half),
        jnp.asarray(rotations),
        image_shape,
        volume_shape,
        "linear_interp",
        max_r=None,
        relion_texture_interp=False,
    )
    proj_true = np.asarray(proj_aug)[true_rot, 0, :]
    phases = np.asarray(half_translation_phase_table(jnp.asarray(translations), image_shape))
    observed_half = proj_true / phases[true_trans]

    dataset = _FakeHalfImageDataset(observed_half, image_shape, volume_shape)
    config = ForwardModelConfig(
        image_shape=image_shape,
        volume_shape=volume_shape,
        grid_size=volume_shape[0],
        voxel_size=1.0,
        padding=0,
        disc_type="linear_interp",
        ctf=as_ctf_evaluator(_identity_ctf),
    )
    shifted_half, batch_norm, ctf2_over_noise = preprocess_batch(
        dataset,
        np.arange(true_rot.shape[0], dtype=np.int32),
        jnp.zeros((true_rot.shape[0], 11), dtype=jnp.float32),
        jnp.ones((proj_true.shape[-1],), dtype=jnp.float32),
        jnp.asarray(translations),
        config,
    )
    expected_shifted = observed_half[:, None, :] * phases[None, :, :]
    np.testing.assert_allclose(
        np.asarray(shifted_half).reshape(true_rot.shape[0], translations.shape[0], proj_true.shape[-1]),
        expected_shifted,
        rtol=1e-6,
        atol=1e-6,
    )

    _stats, diagnostics = dense_pose_ppca_E_step_blocked(
        shifted_half.reshape(true_rot.shape[0], translations.shape[0], proj_true.shape[-1]),
        proj_aug,
        ctf2_over_noise,
        batch_norm.reshape(true_rot.shape[0]),
    )
    np.testing.assert_array_equal(np.asarray(diagnostics.best_rotation_idx), true_rot)
    np.testing.assert_array_equal(np.asarray(diagnostics.best_translation_idx), true_trans)
    assert np.all(np.asarray(diagnostics.pmax) > 0.999)


def test_iter_dense_ppca_dataset_blocks_recovers_healpix_pose_on_tiny_fake_dataset():
    rng = np.random.default_rng(20260511)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rotations = get_rotation_grid_at_order(0, n_in_planes=3, matrices=True).astype(np.float32)
    translations = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    true_rot = np.asarray([7, 21], dtype=np.int32)
    true_trans = np.asarray([1, 3], dtype=np.int32)

    volume_bank = _asymmetric_real_volume_bank(rng, 0, volume_shape)
    augmented_half = np.stack([real_volume_to_centered_fourier_half(volume_bank[0])], axis=0)
    proj_aug = _project_augmented_half_volumes(
        jnp.asarray(augmented_half),
        jnp.asarray(rotations),
        image_shape,
        volume_shape,
        "linear_interp",
        max_r=None,
        relion_texture_interp=False,
    )
    proj_true = np.asarray(proj_aug)[true_rot, 0, :]
    phases = np.asarray(half_translation_phase_table(jnp.asarray(translations), image_shape))
    observed_half = proj_true / phases[true_trans]
    dataset = _FakeDenseDataset(observed_half, image_shape, volume_shape)

    blocks = list(
        iter_dense_ppca_dataset_blocks(
            dataset,
            volume_bank[0],
            W=None,
            q=0,
            volume_domain="real",
            noise_variance=1.0,
            rotations=rotations,
            translations=translations,
            image_batch_size=16,
            rotation_block_size=len(rotations),
            relion_texture_interp=False,
        )
    )
    assert len(blocks) == 1
    block = blocks[0]
    _stats, diagnostics = dense_pose_ppca_E_step_blocked(
        block.Y1,
        block.proj_aug,
        block.ctf2_over_noise,
        block.y_norm,
        pose_log_prior=block.pose_log_prior,
    )

    np.testing.assert_array_equal(np.asarray(diagnostics.best_rotation_idx), true_rot)
    np.testing.assert_array_equal(np.asarray(diagnostics.best_translation_idx), true_trans)
    assert np.all(np.asarray(diagnostics.pmax) > 0.999)


def test_healpix_grid_projected_ppca_score_recovers_rotation_with_latent_signal():
    rng = np.random.default_rng(20260509)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    q = 2
    rotations = get_rotation_grid_at_order(0, n_in_planes=3, matrices=True).astype(np.float32)
    true_rot = np.asarray([4, 16, 29], dtype=np.int32)
    z_true = np.asarray([[0.8, -0.3], [-0.5, 0.7], [0.2, 0.4]], dtype=np.float32)
    precision = 18.0

    volume_bank = _asymmetric_real_volume_bank(rng, q, volume_shape)
    augmented_half = np.stack([real_volume_to_centered_fourier_half(volume) for volume in volume_bank], axis=0)
    proj_aug = _project_augmented_half_volumes(
        jnp.asarray(augmented_half),
        jnp.asarray(rotations),
        image_shape,
        volume_shape,
        "linear_interp",
        max_r=None,
        relion_texture_interp=False,
    )
    proj_aug_np = np.asarray(proj_aug)
    raw = np.stack(
        [
            proj_aug_np[int(rot), 0] + np.einsum("q,qf->f", z_true[b], proj_aug_np[int(rot), 1:])
            for b, rot in enumerate(true_rot)
        ],
        axis=0,
    ).astype(np.complex64)
    raw_shifted = raw[:, None, :]
    raw_norm = np.sum(np.abs(raw) ** 2, axis=1).real.astype(np.float32)

    Y1, ctf2_over_noise, y_norm = _weighted_score_inputs(raw_shifted, raw_norm, precision)
    _stats, diagnostics = dense_pose_ppca_E_step_blocked(Y1, proj_aug, ctf2_over_noise, y_norm)

    np.testing.assert_array_equal(np.asarray(diagnostics.best_rotation_idx), true_rot)
    np.testing.assert_array_equal(np.asarray(diagnostics.best_translation_idx), np.zeros(true_rot.shape, dtype=np.int32))
    assert np.all(np.asarray(diagnostics.pmax) > 0.99)
