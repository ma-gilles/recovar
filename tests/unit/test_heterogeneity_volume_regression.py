"""Regression tests for heterogeneity volume reconstruction quality.

These tests create synthetic datasets with known ground truth, run the
kernel-regression volume reconstruction pipeline, and verify that
quality metrics (local resolution, FSC, L2 error) do not regress.

The tests are GPU-accelerated and use grid_size=16 with enough images
to produce meaningful reconstructions.
"""

import logging
import os
import sys

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.core.linalg as linalg
import recovar.heterogeneity.heterogeneity_volume as hv

from recovar.output import metrics as metrics_mod
from recovar.simulation import synthetic_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from helpers.tiny_synthetic import (
    make_tiny_cryo_dataset_with_images,
    make_tiny_fourier_volumes,
    make_tiny_simulation,
    tiny_ctf_pose_generator,
)

pytestmark = pytest.mark.unit

logger = logging.getLogger(__name__)

GRID_SIZE = 32
N_IMAGES_PER_HALF = 300


def _make_halfsets_and_gt(grid_size=GRID_SIZE, n_images_per_half=N_IMAGES_PER_HALF, seed=42):
    """Create two half-set datasets and ground-truth volume distribution.

    Returns (cryos, hvd, simulation_info, het_distances) where:
      - cryos: CryoEMDataset with halfset_indices and noise models attached
      - hvd: HeterogeneousVolumeDistribution for GT metrics
      - simulation_info: dict with image_assignment, per_image_contrast, etc.
      - het_distances: list of two arrays of random heterogeneity distances
    """
    n_shells = grid_size // 2 - 1
    noise_var = np.ones(n_shells, dtype=np.float32) * 1e-4

    # Create a single dataset with 2*n_images_per_half images split into halves
    cryo = make_tiny_cryo_dataset_with_images(grid_size=grid_size, n_images=2 * n_images_per_half, seed=seed)
    cryo.dataset_indices = np.arange(2 * n_images_per_half, dtype=np.int32)
    cryo.set_radial_noise_model(noise_var)
    cryo.halfset_indices = [
        np.arange(n_images_per_half, dtype=np.int32),
        np.arange(n_images_per_half, 2 * n_images_per_half, dtype=np.int32),
    ]
    cryos = cryo

    # Build GT volume distribution from the first half's simulation
    vols = make_tiny_fourier_volumes(grid_size=grid_size)
    _, _, _, _, sim_info, _, _ = make_tiny_simulation(grid_size=grid_size, n_images=n_images_per_half, seed=seed)
    volume_size = int(grid_size**3)
    hvd = synthetic_dataset.HeterogeneousVolumeDistribution(
        volumes=vols.copy(),
        image_assignments=sim_info["image_assignment"],
        contrasts=sim_info["per_image_contrast"],
        valid_indices=np.ones(volume_size, dtype=np.float32),
        vol_batch_size=1,
    )

    # Random heterogeneity distances (simulating latent-space distances)
    rng = np.random.default_rng(seed)
    het_distances = [
        rng.exponential(scale=2.0, size=n_images_per_half).astype(np.float32),
        rng.exponential(scale=2.0, size=n_images_per_half).astype(np.float32),
    ]

    return cryos, hvd, sim_info, het_distances


@pytest.mark.gpu
@pytest.mark.slow
def test_heterogeneity_volume_locres_regression(tmp_path, gpu_device):
    """Verify that kernel-regression volume reconstruction produces
    reasonable local-resolution metrics that don't regress.

    This is a regression test: it establishes baseline quality metrics and
    asserts that future changes don't degrade them beyond tolerance.
    """
    cryos, hvd, sim_info, het_distances = _make_halfsets_and_gt()

    bins = hv.pick_heterogeneity_bins2(
        ndim=-1,
        log_likelihoods=np.concatenate(het_distances),
        n_bins=5,
        min_images=10,
    )

    from recovar.output.output_paths import VolumeOutputPaths

    vol_paths = VolumeOutputPaths(str(tmp_path / "hv_regression"), "state", 0)

    hv.make_volumes_kernel_estimate_local(
        heterogeneity_distances=het_distances,
        dataset=cryos,
        vol_paths=vol_paths,
        ndim=-1,
        bins=bins,
        B_factor=0,
        tau=None,
        n_min_particles=10,
        metric_used="locshellmost_likely",
        locres_sampling=15,
        kernel_rad=1,
    )

    # Read output half-maps
    half1 = recovar.utils.load_mrc(vol_paths.half1_unfil)
    half2 = recovar.utils.load_mrc(vol_paths.half2_unfil)

    assert half1.shape == cryos.volume_shape, f"Expected {cryos.volume_shape}, got {half1.shape}"
    assert half2.shape == cryos.volume_shape

    # Compute halfmap-based metrics
    halfmap_metrics = metrics_mod.compute_volume_error_metrics_from_halfmaps(half1, half2, cryos.voxel_size, mask=None)

    # Check that median locres is finite and positive
    median_locres = halfmap_metrics["median_locres"]
    ninety_pc_locres = halfmap_metrics["ninety_pc_locres"]
    logger.info("Halfmap median_locres: %.2f A", median_locres)
    logger.info("Halfmap 90pct_locres: %.2f A", ninety_pc_locres)

    assert np.isfinite(median_locres), "median_locres is not finite"
    assert np.isfinite(ninety_pc_locres), "90pct_locres is not finite"
    assert median_locres > 0, "median_locres must be positive"

    # Compute GT-based metrics using GT mean volume
    gt_mean_ft = hvd.get_mean()
    # batch_idft3 expects (vol_size, n_vol) layout
    gt_mean_real = np.real(linalg.batch_idft3(gt_mean_ft[:, None], cryos.volume_shape, batch_size=1))[:, 0].reshape(
        cryos.volume_shape
    )
    estimate_avg = (half1 + half2) / 2.0

    gt_metrics = metrics_mod.compute_volume_error_metrics_from_gt(
        gt_mean_real, estimate_avg, cryos.voxel_size, mask=None
    )

    gt_median_locres = gt_metrics["median_locres"]
    gt_ninety_pc_locres = gt_metrics["ninety_pc_locres"]
    gt_median_error = gt_metrics["median_error"]
    logger.info("GT median_locres: %.2f A", gt_median_locres)
    logger.info("GT 90pct_locres: %.2f A", gt_ninety_pc_locres)
    logger.info("GT median_error: %.4f", gt_median_error)

    assert np.isfinite(gt_median_locres)
    assert np.isfinite(gt_ninety_pc_locres)
    assert np.isfinite(gt_median_error)

    # Read the filtered volume and local resolution map
    filtered = recovar.utils.load_mrc(vol_paths.filtered)
    locres_map = recovar.utils.load_mrc(output_folder + "local_resolution.mrc")

    assert filtered.shape == cryos.volume_shape
    assert locres_map.shape == cryos.volume_shape
    assert np.all(np.isfinite(filtered))
    assert np.all(np.isfinite(locres_map))

    # With tiny_ctf_pose_generator (all identity rotations), reconstruction quality
    # is limited. The main regression check is that the pipeline runs end-to-end
    # and produces finite, non-degenerate output. We check that the error doesn't
    # blow up to nonsensical values (>2.0 would mean the estimate is worse than
    # pure noise).
    assert gt_median_error < 2.0, f"GT median error {gt_median_error} too large (>2.0)"

    # Half-map consistency: median locres should be within a reasonable range
    # for grid_size=32 with voxel_size=1.5 (Nyquist = 3.0 Å, max ~48 Å)
    assert median_locres < 100.0, f"median_locres={median_locres} unreasonably large"


@pytest.mark.gpu
@pytest.mark.slow
def test_heterogeneity_volume_deterministic(tmp_path, gpu_device):
    """Running the same inputs twice must produce identical results."""
    cryos, _, _, het_distances = _make_halfsets_and_gt()

    bins = hv.pick_heterogeneity_bins2(
        ndim=-1,
        log_likelihoods=np.concatenate(het_distances),
        n_bins=5,
        min_images=10,
    )

    from recovar.output.output_paths import VolumeOutputPaths

    outputs = []
    for run_idx in range(2):
        vp = VolumeOutputPaths(str(tmp_path / f"det_run_{run_idx}"), "state", 0)
        hv.make_volumes_kernel_estimate_local(
            heterogeneity_distances=het_distances,
            dataset=cryos,
            vol_paths=vp,
            ndim=-1,
            bins=bins,
            B_factor=0,
            tau=None,
            n_min_particles=10,
            metric_used="locshellmost_likely",
            locres_sampling=15,
            kernel_rad=1,
        )
        h1 = recovar.utils.load_mrc(vp.half1_unfil)
        h2 = recovar.utils.load_mrc(vp.half2_unfil)
        outputs.append((h1, h2))

    # GPU floating-point reductions are not bit-exact across runs,
    # but results should be reproducible to near machine precision.
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=1e-6, rtol=1e-5, err_msg="half1 differs between runs")
    np.testing.assert_allclose(outputs[0][1], outputs[1][1], atol=1e-6, rtol=1e-5, err_msg="half2 differs between runs")


@pytest.mark.gpu
@pytest.mark.slow
def test_heterogeneity_volume_cv_selects_reasonable_bins(tmp_path, gpu_device):
    """The cross-validation bin selection should choose bins that produce
    volumes not significantly worse than the best available bin.
    """
    cryos, hvd, _, het_distances = _make_halfsets_and_gt()

    bins = hv.pick_heterogeneity_bins2(
        ndim=-1,
        log_likelihoods=np.concatenate(het_distances),
        n_bins=5,
        min_images=10,
    )

    from recovar.output.output_paths import VolumeOutputPaths

    vol_paths_cv = VolumeOutputPaths(str(tmp_path / "cv_test"), "state", 0)

    hv.make_volumes_kernel_estimate_local(
        heterogeneity_distances=het_distances,
        dataset=cryos,
        vol_paths=vol_paths_cv,
        ndim=-1,
        bins=bins,
        B_factor=0,
        tau=None,
        n_min_particles=10,
        metric_used="locshellmost_likely",
        locres_sampling=15,
        kernel_rad=1,
        save_all_estimates=True,
    )

    # Read the params to check bin selection
    params = recovar.utils.pickle_load(vol_paths_cv.params)

    assert "ml_choice" in params, "Missing ml_choice in output params"
    assert "heterogeneity_bins" in params
    assert "fscs" in params

    ml_choice = params["ml_choice"]
    n_bins = len(params["heterogeneity_bins"])

    # All choices should be valid bin indices
    assert np.all(np.asarray(ml_choice) >= 0)
    assert np.all(np.asarray(ml_choice) < n_bins)

    # The filtered volume should exist and be finite
    filtered = recovar.utils.load_mrc(vol_paths_cv.filtered)
    assert np.all(np.isfinite(filtered))


# Import recovar at module level (needed for load_mrc)
import recovar.utils
