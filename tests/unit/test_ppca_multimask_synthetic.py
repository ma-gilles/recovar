"""End-to-end test: multi-mask PPCA on CryoBench Ribosembly data.

Uses real Ribosembly volumes (4 states, downsampled to 64³) to build an
in-memory CryoEMDataset, then tests:
1. Multi-mask PCG localizes PCs to their assigned mask regions
2. Full PPCA EM with multi-mask runs and produces localized PCs
3. Gauss-Legendre contrast quadrature + multi-mask together
4. Single-mask baseline (regression guard)

The two masks split the molecule support along the z-axis median.
"""

import glob
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar import core
from recovar.data_io import cryoem_dataset as dataset_mod

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

_RIBOSEMBLY_VOLS = "/home/mg6942/mytigress/cryobench2/Ribosembly/vols/128_org"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RadialNoise:
    """Trivial noise model: unit variance everywhere."""

    def __init__(self, image_size, half_image_size):
        self._noise = np.ones((image_size,), dtype=np.float32)
        self._noise_half = np.ones((half_image_size,), dtype=np.float32)

    def get(self, indices):
        return np.tile(self._noise[None], (len(indices), 1))

    def get_half(self, indices):
        return np.tile(self._noise_half[None], (len(indices), 1))


class _FourierImageStack:
    """In-memory Fourier-domain image stack for PPCA testing."""

    def __init__(self, images_fourier, image_shape):
        self.n_images = images_fourier.shape[0]
        self.D = image_shape[0]
        self.unpadded_D = self.D
        self.padding = 0
        self.image_shape = image_shape
        self.grid_size = self.D
        self.mask = np.ones(image_shape, dtype=np.float32)
        self.Np = self.n_images
        self._images_fourier = images_fourier.astype(np.complex64)

    def get_dataset_generator(self, batch_size, num_workers=0, **kwargs):
        for start in range(0, self.n_images, batch_size):
            end = min(start + batch_size, self.n_images)
            idx = np.arange(start, end, dtype=np.int32)
            yield self._images_fourier[idx], idx, idx

    def get_image_generator(self, batch_size, num_workers=0):
        return self.get_dataset_generator(batch_size, num_workers=num_workers)

    def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
        subset_indices = np.asarray(subset_indices, dtype=np.int32)
        for start in range(0, subset_indices.size, batch_size):
            idx = subset_indices[start : start + batch_size]
            yield self._images_fourier[idx], idx, idx

    def get_image_subset_generator(self, batch_size, subset_indices, num_workers=0):
        return self.get_dataset_subset_generator(batch_size, subset_indices, num_workers=num_workers)

    def process_images(self, image, apply_image_mask=True):
        return image

    def process_images_half(self, image, apply_image_mask=True):
        return ftu.full_image_to_half_image(image, self.image_shape)


def _load_ribosembly_volumes(n_states=4, grid_size=64):
    """Load first n_states Ribosembly volumes, downsampled to grid_size."""
    import mrcfile
    from scipy.ndimage import zoom

    files = sorted(glob.glob(os.path.join(_RIBOSEMBLY_VOLS, "*.mrc")))
    if len(files) < n_states:
        pytest.skip(f"Need {n_states} Ribosembly volumes at {_RIBOSEMBLY_VOLS}")

    vols_real = []
    for f in files[:n_states]:
        with mrcfile.open(f, mode="r") as mrc:
            v = mrc.data.copy().astype(np.float32)
        if v.shape[0] != grid_size:
            factor = grid_size / v.shape[0]
            v = zoom(v, factor, order=1).astype(np.float32)
        vols_real.append(v)

    vols_real = np.array(vols_real)
    vol_shape = (grid_size, grid_size, grid_size)
    vols_fourier = np.array([ftu.get_dft3(v).ravel() for v in vols_real])

    return vols_real, vols_fourier, vol_shape


def _make_split_masks(vol_shape, vols_real):
    """Split molecule support along z-axis median into two masks."""
    from scipy.ndimage import binary_dilation, gaussian_filter

    mean_vol = vols_real.mean(axis=0)
    threshold = np.percentile(mean_vol[mean_vol > 0], 10)
    support = binary_dilation(mean_vol > threshold, iterations=2).astype(np.float32)

    z_boundary = int(np.median(np.where(support > 0.5)[2]))
    mask_left = support.copy()
    mask_left[:, :, z_boundary:] = 0.0
    mask_right = support.copy()
    mask_right[:, :, :z_boundary] = 0.0

    mask_left = (gaussian_filter(mask_left, sigma=1.5) > 0.3).astype(np.float32)
    mask_right = (gaussian_filter(mask_right, sigma=1.5) > 0.3).astype(np.float32)

    return mask_left, mask_right, support


def _simulate_dataset(vols_fourier, vol_shape, n_images=500, noise_level=1.0, contrast_std=0.0, seed=42):
    """Build an in-memory CryoEMDataset from Fourier volumes + random poses."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    grid_size = vol_shape[0]
    image_shape = (grid_size, grid_size)
    n_states = vols_fourier.shape[0]

    assignments = rng.integers(0, n_states, size=n_images)
    rotations = Rotation.random(n_images, random_state=seed).as_matrix().astype(np.float32)
    translations = np.zeros((n_images, 2), dtype=np.float32)

    # CTF: constant defocus, no astigmatism
    ctf_params = np.zeros((n_images, 9), dtype=np.float32)
    ctf_params[:, core.CTFParamIndex.DFU] = 15000.0
    ctf_params[:, core.CTFParamIndex.DFV] = 15000.0
    ctf_params[:, core.CTFParamIndex.DFANG] = 0.0
    ctf_params[:, core.CTFParamIndex.VOLT] = 300.0
    ctf_params[:, core.CTFParamIndex.CS] = 2.7
    ctf_params[:, core.CTFParamIndex.W] = 0.1
    ctf_params[:, core.CTFParamIndex.BFACTOR] = 0.0
    ctf_params[:, core.CTFParamIndex.CONTRAST] = 1.0

    # Per-image contrast
    if contrast_std > 0:
        contrasts = 1.0 + rng.normal(0, contrast_std, n_images).astype(np.float32)
        contrasts = np.clip(contrasts, 0.3, 2.0)
    else:
        contrasts = np.ones(n_images, dtype=np.float32)

    ctf_evaluator = core.CTFEvaluator()
    voxel_size = 1.0

    # Project volumes
    images_fourier = []
    for i in range(n_images):
        proj = core.slice_volume(
            jnp.array(vols_fourier[assignments[i]]),
            jnp.array(rotations[i : i + 1]),
            image_shape,
            vol_shape,
            "linear_interp",
        )
        ctf_val = ctf_evaluator(jnp.array(ctf_params[i : i + 1]), image_shape, voxel_size)
        proj = proj * ctf_val * contrasts[i]
        images_fourier.append(np.array(proj[0]))

    images_fourier = np.array(images_fourier)

    # Add noise
    signal_std = float(np.std(np.abs(images_fourier)))
    noise_std = noise_level * signal_std
    noise = (
        rng.normal(0, noise_std, images_fourier.shape) + 1j * rng.normal(0, noise_std, images_fourier.shape)
    ).astype(images_fourier.dtype)
    images_fourier += noise

    # Build CryoEMDataset
    image_stack = _FourierImageStack(images_fourier, image_shape)
    metadata = dataset_mod.ImageMetadata(rotations, translations, ctf_params)
    cryo = dataset_mod.CryoEMDataset(
        image_source=image_stack,
        voxel_size=voxel_size,
        metadata=metadata,
        ctf_evaluator=ctf_evaluator,
        dataset_indices=np.arange(n_images, dtype=np.int32),
        grid_size=grid_size,
    )
    half_image_shape = ftu.image_shape_to_half_image_shape(image_shape)
    cryo.noise = _RadialNoise(
        image_size=int(np.prod(image_shape)),
        half_image_size=int(np.prod(half_image_shape)),
    )

    # Half-set split (required by EM which calls materialize_halfset_datasets)
    all_idx = np.arange(n_images, dtype=np.int32)
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(all_idx)
    half = n_images // 2
    cryo.halfset_indices = (all_idx[:half], all_idx[half:])

    return cryo, assignments


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiMaskRibosembly:
    """Multi-mask PPCA on downsampled Ribosembly data."""

    @pytest.fixture(scope="class")
    def ribosembly_data(self):
        """Load Ribosembly volumes and create split masks."""
        vols_real, vols_fourier, vol_shape = _load_ribosembly_volumes(n_states=4, grid_size=64)
        mask_left, mask_right, support = _make_split_masks(vol_shape, vols_real)
        return {
            "vols_real": vols_real,
            "vols_fourier": vols_fourier,
            "vol_shape": vol_shape,
            "mask_left": mask_left,
            "mask_right": mask_right,
            "support": support,
        }

    def test_multi_mask_pcg_localizes(self, ribosembly_data):
        """PCG with multi-mask enforces zero outside assigned region."""
        from recovar.ppca.ppca import _pcg_hard_mstep, unpack_tri_to_full

        vol_shape = ribosembly_data["vol_shape"]
        mask_left = ribosembly_data["mask_left"]
        mask_right = ribosembly_data["mask_right"]
        q = 4

        half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
        half_vol = int(np.prod(half_vs))
        tri_sz = q * (q + 1) // 2

        # Diagonal-dominant LHS
        rng = np.random.default_rng(7)
        lhs_tri = np.zeros((half_vol, tri_sz), dtype=np.float32)
        idx = 0
        for i in range(q):
            for j in range(i, q):
                if i == j:
                    lhs_tri[:, idx] = 1.0 + rng.uniform(0, 0.1, half_vol).astype(np.float32)
                else:
                    lhs_tri[:, idx] = rng.normal(0, 0.01, half_vol).astype(np.float32)
                idx += 1

        # RHS from volume differences
        vols_fourier = ribosembly_data["vols_fourier"]
        rhs_full = np.zeros((half_vol, q), dtype=np.complex64)
        for k in range(q):
            diff = vols_fourier[(k + 1) % len(vols_fourier)] - vols_fourier[k]
            diff_half = ftu.full_volume_to_half_volume(jnp.array(diff), vol_shape)
            rhs_full[:, k] = np.array(diff_half) * 0.1

        reg_diag = np.ones((half_vol, q), dtype=np.float32) * 0.01
        masks = np.stack([mask_left, mask_right])
        assignment = np.array([0, 0, 1, 1], dtype=np.int32)
        union_mask = np.maximum(mask_left, mask_right)

        W_real = _pcg_hard_mstep(
            jnp.array(lhs_tri),
            jnp.array(rhs_full),
            jnp.array(reg_diag),
            union_mask,
            vol_shape,
            q,
            unpack_tri_to_full,
            maxiter=50,
            tol=1e-5,
            masks=masks,
            pc_mask_assignment=assignment,
        )
        W_np = np.array(W_real)

        for k in range(q):
            assigned_mask = masks[assignment[k]]
            outside = 1.0 - assigned_mask
            energy_outside = float(np.sum(W_np[k] ** 2 * outside))
            assert energy_outside < 1e-8, f"PC {k} (mask {assignment[k]}) has energy {energy_outside:.2e} outside"

    def test_full_em_multimask_runs(self, ribosembly_data):
        """Full PPCA EM with multi-mask completes and produces localized PCs."""
        from recovar.ppca import ppca as ppca_module

        vol_shape = ribosembly_data["vol_shape"]
        mask_left = ribosembly_data["mask_left"]
        mask_right = ribosembly_data["mask_right"]
        grid_size = vol_shape[0]
        basis_size = 4

        cryo, assignments = _simulate_dataset(
            ribosembly_data["vols_fourier"],
            vol_shape,
            n_images=500,
            noise_level=1.0,
            seed=42,
        )

        mean_fourier = ribosembly_data["vols_fourier"].mean(axis=0)
        half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
        half_vol = int(np.prod(half_vs))
        rng = np.random.default_rng(0)
        W_init = (
            rng.normal(size=(half_vol, basis_size)) * 0.01 + 1j * rng.normal(size=(half_vol, basis_size)) * 0.01
        ).astype(np.complex64)
        W_prior = np.ones((int(np.prod(vol_shape)), basis_size), dtype=np.float32)

        masks = np.stack([mask_left, mask_right])
        assignment = np.array([0, 0, 1, 1], dtype=np.int32)
        union_mask = np.maximum(mask_left, mask_right)

        U, S, W, ez, smz, iter_data, post_info = ppca_module.EM(
            cryo,
            mean_fourier,
            W_init,
            W_prior,
            EM_iter=5,
            volume_mask=union_mask,
            return_iteration_data=True,
            return_posterior_info=True,
            masks=masks,
            pc_mask_assignment=assignment,
        )

        assert not np.any(np.isnan(ez)), "NaN in embeddings"
        assert ez.shape == (500, basis_size)

        # Verify PCs are localized on W (raw M-step output), NOT U.
        # U = SVD(W) mixes columns and destroys per-PC mask localization.
        # W is in half-Fourier shape (half_vol, basis_size).
        for k in range(basis_size):
            w_half = jnp.array(W[:, k])
            w_full = ftu.half_volume_to_full_volume(w_half, vol_shape)
            w_real = np.array(ftu.get_idft3(w_full.reshape(vol_shape)).real)

            assigned_mask = masks[assignment[k]]
            outside = 1.0 - assigned_mask
            energy_outside = float(np.sum(w_real**2 * outside))
            energy_total = float(np.sum(w_real**2))

            if energy_total > 1e-10:
                frac_outside = energy_outside / energy_total
                assert frac_outside < 0.05, f"W col {k} has {frac_outside:.1%} energy outside its mask"

        print(f"\n  Multi-mask EM: eigenvalues = {S}")
        print(f"  Embedding std per PC: {np.std(ez, axis=0)}")

    def test_full_em_single_mask_baseline(self, ribosembly_data):
        """Single-mask EM (no multi-mask) as regression guard."""
        from recovar.ppca import ppca as ppca_module

        vol_shape = ribosembly_data["vol_shape"]
        mask_left = ribosembly_data["mask_left"]
        mask_right = ribosembly_data["mask_right"]
        basis_size = 4

        cryo, _ = _simulate_dataset(
            ribosembly_data["vols_fourier"],
            vol_shape,
            n_images=500,
            noise_level=1.0,
            seed=42,
        )

        mean_fourier = ribosembly_data["vols_fourier"].mean(axis=0)
        half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
        half_vol = int(np.prod(half_vs))
        rng = np.random.default_rng(0)
        W_init = (
            rng.normal(size=(half_vol, basis_size)) * 0.01 + 1j * rng.normal(size=(half_vol, basis_size)) * 0.01
        ).astype(np.complex64)
        W_prior = np.ones((int(np.prod(vol_shape)), basis_size), dtype=np.float32)
        union_mask = np.maximum(mask_left, mask_right)

        U, S, W, ez, smz, iter_data, post_info = ppca_module.EM(
            cryo,
            mean_fourier,
            W_init,
            W_prior,
            EM_iter=5,
            volume_mask=union_mask,
            return_iteration_data=True,
            return_posterior_info=True,
        )

        assert not np.any(np.isnan(ez)), "NaN in single-mask embeddings"
        assert ez.shape == (500, basis_size)
        print(f"\n  Single-mask EM: eigenvalues = {S}")
        print(f"  Embedding std per PC: {np.std(ez, axis=0)}")

    def test_full_em_contrast_marginalize_multimask(self, ribosembly_data):
        """EM with GL contrast marginalization + multi-mask."""
        from recovar.ppca import ppca as ppca_module
        from recovar.ppca.contrast_posterior import make_contrast_quadrature

        vol_shape = ribosembly_data["vol_shape"]
        mask_left = ribosembly_data["mask_left"]
        mask_right = ribosembly_data["mask_right"]
        basis_size = 4

        cryo, _ = _simulate_dataset(
            ribosembly_data["vols_fourier"],
            vol_shape,
            n_images=300,
            noise_level=1.0,
            contrast_std=0.3,
            seed=99,
        )

        mean_fourier = ribosembly_data["vols_fourier"].mean(axis=0)
        half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
        half_vol = int(np.prod(half_vs))
        rng = np.random.default_rng(0)
        W_init = (
            rng.normal(size=(half_vol, basis_size)) * 0.01 + 1j * rng.normal(size=(half_vol, basis_size)) * 0.01
        ).astype(np.complex64)
        W_prior = np.ones((int(np.prod(vol_shape)), basis_size), dtype=np.float32)

        masks = np.stack([mask_left, mask_right])
        assignment = np.array([0, 0, 1, 1], dtype=np.int32)
        union_mask = np.maximum(mask_left, mask_right)

        nodes, weights = make_contrast_quadrature(
            rule="gauss_legendre",
            interval=(0.0, 2.0),
            n_nodes=16,
        )

        U, S, W, ez, smz, iter_data, post_info = ppca_module.EM(
            cryo,
            mean_fourier,
            W_init,
            W_prior,
            EM_iter=5,
            volume_mask=union_mask,
            return_iteration_data=True,
            return_posterior_info=True,
            contrast_mode="marginalize",
            contrast_grid=np.array(nodes),
            contrast_weights=np.array(weights),
            masks=masks,
            pc_mask_assignment=assignment,
        )

        assert not np.any(np.isnan(ez)), "NaN in contrast+multimask embeddings"

        mean_c = post_info["mean_c"]
        median_c = float(np.median(mean_c))
        print(f"\n  Contrast+multimask EM: eigenvalues = {S}")
        print(f"  Median estimated contrast: {median_c:.3f}")
        assert 0.3 < median_c < 2.0, f"Median contrast {median_c} out of range"
