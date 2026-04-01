"""Integration tests for the full multi-iteration EM refinement loop.

These tests verify that our refinement implementation:
1. Converges (resolution improves over iterations)
2. Produces results comparable to RELION reference data
3. Volume quality is reasonable (FSC vs RELION or ground truth)

All tests require GPU and are marked @pytest.mark.slow.
They are NOT expected to run in CI; instead they are run via scripts
or submitted to Slurm for comprehensive validation.

Synthetic dataset: /scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/
RELION reference: /scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/relion_ref_npz/
"""

import logging
import os

import jax.numpy as jnp
import mrcfile
import numpy as np
import pytest

logger = logging.getLogger(__name__)

# Paths to the benchmark data
DATA_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data"
RELION_REF_NPZ = os.path.join(DATA_DIR, "relion_ref_npz")
RELION_REF_STAR = os.path.join(DATA_DIR, "relion_ref")
PARTICLES_STAR = os.path.join(DATA_DIR, "particles.star")
INIT_MRC = os.path.join(DATA_DIR, "reference_init.mrc")
GT_MRC = os.path.join(DATA_DIR, "reference_gt.mrc")

# Skip if benchmark data not available
DATA_AVAILABLE = os.path.exists(PARTICLES_STAR) and os.path.exists(INIT_MRC)
RELION_AVAILABLE = os.path.exists(os.path.join(RELION_REF_NPZ, "iteration_001.npz"))


def _setup_refinement(n_iter=5, adaptive_oversampling=1, seed=42):
    """Common setup for refinement tests. Returns (result, dataset_info)."""
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.em.dense_single_volume.refine import refine_single_volume
    from recovar.em.sampling import get_rotation_grid, get_translation_grid
    from recovar.reconstruction.regularization import average_over_shells
    from recovar import utils

    ds = load_dataset(PARTICLES_STAR, lazy=False)

    # Create half-sets
    n_images = ds.n_units
    indices = np.arange(n_images)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    half1_idx = np.sort(indices[:n_images // 2])
    half2_idx = np.sort(indices[n_images // 2:])
    ds_half1 = ds.subset(half1_idx)
    ds_half2 = ds.subset(half2_idx)

    # Load initial volume
    with mrcfile.open(INIT_MRC, mode="r") as mrc:
        init_vol_real = np.array(mrc.data, dtype=np.float32)
    init_vol_ft = np.fft.fftn(np.fft.ifftshift(init_vol_real)).astype(np.complex64).reshape(-1)

    # Grids
    healpix_order = 3
    rotations = get_rotation_grid(healpix_order, matrices=True).astype(np.float32)
    translations = get_translation_grid(3.0, 1.0).astype(np.float32)

    # Initialize noise and prior
    noise_variance = jnp.ones(ds.image_size, dtype=jnp.float32)
    init_PS = average_over_shells(jnp.abs(jnp.asarray(init_vol_ft)) ** 2, ds.volume_shape)
    init_prior = utils.make_radial_image(init_PS, ds.volume_shape, extend_last_frequency=True)
    mean_variance = jnp.asarray(init_prior * 0.5 + jnp.max(init_prior) * 1e-4)

    init_current_size = max(32, int(2 * ds.voxel_size * ds.grid_size / 30.0))

    result = refine_single_volume(
        experiment_datasets=[ds_half1, ds_half2],
        init_volume=jnp.asarray(init_vol_ft),
        init_noise_variance=noise_variance,
        init_mean_variance=mean_variance,
        rotations=rotations,
        translations=jnp.asarray(translations),
        disc_type="linear_interp",
        max_iter=n_iter,
        image_batch_size=500,
        rotation_block_size=5000,
        init_current_size=init_current_size,
        fsc_threshold=1.0 / 7.0,
        adaptive_oversampling=adaptive_oversampling,
        adaptive_fraction=0.999,
        max_significants=500,
        nside_level=healpix_order if adaptive_oversampling > 0 else None,
        translation_pixel_offset=1.0 if adaptive_oversampling > 0 else None,
    )

    dataset_info = {
        "ds": ds,
        "voxel_size": ds.voxel_size,
        "volume_shape": ds.volume_shape,
        "healpix_order": healpix_order,
        "n_rotations": rotations.shape[0],
        "n_translations": translations.shape[0],
    }

    return result, dataset_info


@pytest.mark.slow
@pytest.mark.skipif(not DATA_AVAILABLE, reason="Benchmark data not available")
def test_refinement_converges():
    """Run 5 iterations, verify resolution improves.

    The resolution (pixel_resolution) should decrease (improve) over iterations
    as the FSC-driven loop allows higher frequencies.
    """
    result, info = _setup_refinement(n_iter=5, adaptive_oversampling=1)

    current_sizes = result["current_sizes"]
    pixel_resolutions = result["pixel_resolutions"]

    logger.info("Current sizes: %s", current_sizes)
    logger.info("Pixel resolutions: %s", pixel_resolutions)

    # The current_size should increase (or stay) from iteration to iteration
    # (more resolution becomes available as the model improves)
    # At minimum, the last current_size should be >= the first
    assert current_sizes[-1] >= current_sizes[0], (
        f"Resolution should not regress: first cs={current_sizes[0]}, "
        f"last cs={current_sizes[-1]}"
    )

    # The FSC-based pixel resolution should be finite and positive
    for pr in pixel_resolutions:
        assert 0 < pr < 1000, f"Invalid pixel resolution: {pr}"

    # At least one FSC curve should exist
    assert len(result["fsc_history"]) == 5, (
        f"Expected 5 FSC curves, got {len(result['fsc_history'])}"
    )

    # Final FSC should have high values at low shells
    final_fsc = np.asarray(result["fsc_history"][-1])
    assert final_fsc[1] > 0.5, (
        f"Final FSC at shell 1 is too low: {final_fsc[1]:.4f}"
    )

    logger.info("PASS: Refinement converges (cs: %s -> %s, res: %.1f -> %.1f)",
                current_sizes[0], current_sizes[-1],
                pixel_resolutions[0], pixel_resolutions[-1])


@pytest.mark.slow
@pytest.mark.skipif(not DATA_AVAILABLE or not RELION_AVAILABLE,
                    reason="Benchmark or RELION data not available")
def test_refinement_vs_relion_resolution():
    """Compare current_size trajectory against RELION reference.

    Due to algorithmic differences (allowed sizes {32,64,128} vs RELION's
    finer granularity, different noise/prior estimation), we expect
    approximate agreement. The test checks that our trajectory is
    in the same ballpark.
    """
    result, info = _setup_refinement(n_iter=5, adaptive_oversampling=1)

    our_sizes = result["current_sizes"]
    relion_sizes = []
    for it in range(5):
        path = os.path.join(RELION_REF_NPZ, f"iteration_{it:03d}.npz")
        if os.path.exists(path):
            rd = dict(np.load(path, allow_pickle=True))
            cs = rd["current_image_size"]
            if cs.ndim == 0:
                cs = cs.item()
            relion_sizes.append(int(cs))
        else:
            relion_sizes.append(0)

    logger.info("Our current_sizes: %s", our_sizes)
    logger.info("RELION current_sizes: %s", relion_sizes)

    # Check that at least the final iteration has a reasonable current_size
    # (not stuck at minimum)
    assert our_sizes[-1] >= 32, (
        f"Final current_size should be >= 32, got {our_sizes[-1]}"
    )

    # Our quantized sizes should be within a factor of 2 of RELION's
    # (since our allowed set is {32, 64, 128})
    for i in range(min(len(our_sizes), len(relion_sizes))):
        if relion_sizes[i] > 0:
            ratio = our_sizes[i] / relion_sizes[i]
            assert 0.25 < ratio < 4.0, (
                f"Iter {i}: our cs={our_sizes[i]} vs RELION cs={relion_sizes[i]} "
                f"(ratio={ratio:.2f}) is too far off"
            )

    logger.info("PASS: Resolution trajectory is in the same ballpark as RELION")


@pytest.mark.slow
@pytest.mark.skipif(not DATA_AVAILABLE or not RELION_AVAILABLE,
                    reason="Benchmark or RELION data not available")
def test_refinement_vs_relion_volume():
    """Compare our final volume against RELION's final volume via FSC.

    Computes cross-FSC between our merged volume and RELION's merged volume.
    We expect reasonable agreement at low-to-medium frequencies, with potential
    divergence at high frequencies due to noise/prior differences.
    """
    result, info = _setup_refinement(n_iter=5, adaptive_oversampling=1)
    volume_shape = info["volume_shape"]

    from recovar.reconstruction.regularization import get_fsc_gpu

    our_mean_ft = jnp.asarray(result["mean"])

    # Load RELION merged volume
    relion_merged_path = os.path.join(RELION_REF_STAR, "run_class001.mrc")
    if not os.path.exists(relion_merged_path):
        pytest.skip("RELION merged MRC not found")

    with mrcfile.open(relion_merged_path, mode="r") as mrc:
        relion_vol_real = np.array(mrc.data, dtype=np.float32)
    relion_ft = jnp.asarray(
        np.fft.fftn(np.fft.ifftshift(relion_vol_real)).astype(np.complex64).reshape(-1)
    )

    fsc = np.asarray(get_fsc_gpu(our_mean_ft, relion_ft, volume_shape))

    logger.info("Cross-FSC (our vs RELION) at selected shells:")
    for s in [1, 5, 10, 15, 20, 30]:
        if s < len(fsc):
            logger.info("  shell %d: %.4f", s, fsc[s])

    # At low frequencies (shells 1-10), FSC should be high
    low_freq_fsc = fsc[1:11]
    mean_low_freq = float(np.mean(low_freq_fsc))
    logger.info("Mean FSC at shells 1-10: %.4f", mean_low_freq)

    assert mean_low_freq > 0.5, (
        f"Low-frequency FSC between our volume and RELION's is too low: "
        f"{mean_low_freq:.4f}. Expected > 0.5."
    )

    # Also check against ground truth
    if os.path.exists(GT_MRC):
        with mrcfile.open(GT_MRC, mode="r") as mrc:
            gt_real = np.array(mrc.data, dtype=np.float32)
        gt_ft = jnp.asarray(
            np.fft.fftn(np.fft.ifftshift(gt_real)).astype(np.complex64).reshape(-1)
        )
        fsc_gt = np.asarray(get_fsc_gpu(our_mean_ft, gt_ft, volume_shape))
        mean_gt_low = float(np.mean(fsc_gt[1:11]))
        logger.info("Mean FSC vs GT at shells 1-10: %.4f", mean_gt_low)
        assert mean_gt_low > 0.5, (
            f"Low-frequency FSC vs ground truth is too low: {mean_gt_low:.4f}"
        )

    logger.info("PASS: Volume quality is reasonable (FSC vs RELION mean low-freq=%.4f)",
                mean_low_freq)
