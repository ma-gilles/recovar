"""Isolated regression tests for each pipeline function.

Each test exercises a single pipeline function (mean, noise, variance,
covariance columns, projected covariance, principal components, embedding
with/without contrast) in isolation, feeding standardized inputs from the
~/recovar baseline pipeline output.

This catches regressions at the function level — if a specific pipeline
step degrades, the test pinpoints exactly which function is responsible,
rather than only showing an end-to-end metric drop.

Baseline data:
    Pre-computed by scripts/generate_function_baselines.py using ~/recovar
    (published recovar at ma-gilles/recovar.git).  Intermediates are stored
    at FUNCTION_BASELINE_DIR (default: /scratch/gpfs/GILLES/mg6942/
    pdb_baseline_snr01/function_baselines/) and per-function scores in
    tests/baselines/pipeline_functions_isolated/pdb_5nrl/per_function_scores.json.

Environment variables:
    FUNCTION_BASELINE_DIR : str
        Directory with baseline intermediates (.npy files).
    FUNCTION_BASELINE_SCORES : str
        Path to per-function baseline scores JSON.
    FUNCTION_TEST_TOL_FRAC : float
        Tolerated relative degradation (default 0.10 = 10%).
    FUNCTION_TEST_DATASET_DIR : str
        Path to the shared test dataset.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers.metrics_regression import compare_metric, metric_direction

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.io,
    pytest.mark.long_test,
]

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_FUNCTION_BASELINE_DIR = (
    "/scratch/gpfs/GILLES/mg6942/pdb_baseline_snr01/function_baselines"
)
_DEFAULT_FUNCTION_BASELINE_SCORES = (
    _REPO_ROOT / "tests" / "baselines" / "pipeline_functions_isolated"
    / "pdb_5nrl" / "per_function_scores.json"
)
_DEFAULT_DATASET_DIR = (
    "/scratch/gpfs/GILLES/mg6942/pdb_baseline_snr01/test_dataset"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _env_or_default(var, default):
    return os.environ.get(var, str(default))


@pytest.fixture(scope="module")
def tol_frac():
    return float(os.environ.get("FUNCTION_TEST_TOL_FRAC", "0.10"))


@pytest.fixture(scope="module")
def baseline_dir():
    d = _env_or_default("FUNCTION_BASELINE_DIR", _DEFAULT_FUNCTION_BASELINE_DIR)
    p = Path(d)
    if not p.exists():
        pytest.skip(f"Baseline dir not found: {p}")
    return p


@pytest.fixture(scope="module")
def baseline_scores():
    p = Path(_env_or_default(
        "FUNCTION_BASELINE_SCORES", _DEFAULT_FUNCTION_BASELINE_SCORES
    ))
    if not p.exists():
        pytest.skip(f"Baseline scores not found: {p}")
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def dataset_dir():
    d = _env_or_default("FUNCTION_TEST_DATASET_DIR", _DEFAULT_DATASET_DIR)
    p = Path(d)
    if not p.exists():
        pytest.skip(f"Dataset dir not found: {p}")
    return p


@pytest.fixture(scope="module")
def intermediates(baseline_dir):
    """Load all baseline intermediate numpy arrays."""
    intdir = baseline_dir / "intermediates"
    if not intdir.exists():
        pytest.skip(f"Intermediates dir not found: {intdir}")
    data = {}
    for f in intdir.glob("*.npy"):
        data[f.stem] = np.load(str(f))
    return data


@pytest.fixture(scope="module")
def gt_data(dataset_dir):
    """Load ground truth from simulation_info.pkl."""
    from recovar.simulation import synthetic_dataset

    sim_path = str(dataset_dir / "simulation_info.pkl")
    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_path)
    gt_mean = gt.get_mean()
    gt_spatial_variance = gt.get_spatial_variances(contrasted=False)
    # GT Fourier variance: per-voxel power in Fourier space
    cov_sqrt_fourier = gt.get_covariance_square_root(contrasted=False)  # (vol_size, n_pcs)
    gt_fourier_variance = np.sum(np.abs(cov_sqrt_fourier) ** 2, axis=-1)  # (vol_size,)

    u_gt, s_gt, _ = gt.get_vol_svd(
        contrasted=False, real_space=True, random_svd_pcs=200
    )
    u_gt_fourier, s_gt_fourier, _ = gt.get_vol_svd(
        contrasted=False, real_space=False
    )

    with open(sim_path, "rb") as f:
        sim_info = pickle.load(f)

    return {
        "gt": gt,
        "gt_mean": gt_mean,
        "gt_spatial_variance": gt_spatial_variance,
        "gt_fourier_variance": gt_fourier_variance,
        "u_gt": u_gt,
        "s_gt": s_gt,
        "u_gt_fourier": u_gt_fourier,
        "s_gt_fourier": s_gt_fourier,
        "sim_info": sim_info,
        "pa": np.asarray(sim_info["image_assignment"]).ravel(),
        "gt_contrasts": np.asarray(sim_info["per_image_contrast"]).ravel(),
        "gt_noise": np.asarray(sim_info["noise_variance"]).ravel(),
        "grid_size": sim_info["grid_size"],
        "voxel_size": sim_info.get("voxel_size", 4.25),
    }


@pytest.fixture(scope="module")
def cryos(dataset_dir, intermediates):
    """Create CryoEMHalfsets from the shared dataset using current code."""
    from recovar.data_io import dataset

    grid_size = 128  # known for this dataset
    particles_file = str(dataset_dir / f"particles.{grid_size}.mrcs")
    poses_file = str(dataset_dir / "poses.pkl")
    ctf_file = str(dataset_dir / "ctf.pkl")

    # Use the same half-set split as the baseline for reproducibility
    ind0 = intermediates.get("halfset_ind0")
    ind1 = intermediates.get("halfset_ind1")
    if ind0 is not None and ind1 is not None:
        ind_split = [ind0, ind1]
    else:
        # Fall back to random split
        n_images = 50000  # known for this dataset
        rng = np.random.RandomState(0)
        perm = rng.permutation(n_images)
        ind_split = [perm[:n_images // 2], perm[n_images // 2:]]

    cryos = dataset.get_split_datasets(
        particles_file, poses_file, ctf_file,
        datadir=None, ind_split=ind_split, lazy=True,
    )
    # Initialize noise model (same as pipeline.py line 737)
    for cryo in cryos:
        cryo.set_radial_noise_model(None)
    return cryos


def _check_metric(key, current_val, baseline_scores, tol_frac):
    """Compare a single metric against baseline. Returns (ok, msg)."""
    if key not in baseline_scores:
        return True, f"no baseline for {key}"
    base = baseline_scores[key]
    direction = metric_direction(key)
    if direction == "ignore":
        return True, "ignored"
    ok, msg = compare_metric(float(current_val), float(base), direction, tol_frac)
    return ok, f"{key}: current={current_val:.6f} baseline={base:.6f} ({msg})"


def _assert_metrics(results: dict, baseline_scores: dict, tol_frac: float):
    """Assert all metrics in results pass against baseline."""
    failures = []
    for key, val in results.items():
        if not isinstance(val, (int, float)):
            continue
        ok, msg = _check_metric(key, val, baseline_scores, tol_frac)
        if not ok:
            failures.append(msg)
    assert not failures, "Metric regressions:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Test 1: compute_mean
# ---------------------------------------------------------------------------

def test_compute_mean(cryos, gt_data, intermediates, baseline_scores, tol_frac):
    """Test mean reconstruction in isolation.

    Uses raw images from the dataset. Compares FSC vs GT mean.
    """
    from recovar.reconstruction import homogeneous, noise
    from recovar.output import plot_utils

    batch_size = 512
    volume_shape = cryos.volume_shape

    # Initial noise from half-maps
    noise_var_from_hf, _ = noise.estimate_noise_variance(cryos[0], batch_size)

    means, mean_prior, _ = homogeneous.get_mean_conformation_relion(
        cryos, batch_size, noise_variance=noise_var_from_hf, use_regularization=False
    )

    _, mean_fsc_score = plot_utils.plot_fsc_new(
        gt_data["gt_mean"], means["combined"],
        np.array(volume_shape), gt_data["voxel_size"],
        threshold=0.5, name="Mean FSC"
    )

    results = {"mean_fsc": float(mean_fsc_score)}
    _assert_metrics(results, baseline_scores, tol_frac)


# ---------------------------------------------------------------------------
# Test 2: estimate_noise
# ---------------------------------------------------------------------------

def test_estimate_noise(cryos, gt_data, intermediates, baseline_scores, tol_frac):
    """Test noise estimation in isolation.

    Uses ~/recovar's mean and mask as standardized inputs to decouple
    from mean estimation quality.
    """
    from recovar.reconstruction import noise

    batch_size = 512
    mean_combined = intermediates["mean_combined"]
    dilated_volume_mask = intermediates["dilated_volume_mask"]
    gt_noise = gt_data["gt_noise"]

    # Outside mask
    masked_image_PS, _, _ = noise.estimate_noise_variance_from_outside_mask_v2(
        cryos[0], dilated_volume_mask, batch_size
    )
    # Upper bound inside mask
    radial_ub_noise_var, _, _ = noise.estimate_radial_noise_upper_bound_from_inside_mask_v2(
        cryos[0], mean_combined, dilated_volume_mask, batch_size
    )
    # Image PS for fallback
    _, _, image_PS, _ = noise.estimate_radial_noise_statistic_from_outside_mask(
        cryos[0], dilated_volume_mask, batch_size
    )

    noise_var_used = np.where(
        masked_image_PS > radial_ub_noise_var,
        radial_ub_noise_var, masked_image_PS
    )
    noise_var_used = np.where(noise_var_used < 0, image_PS / 10, noise_var_used)

    n_shells = min(len(gt_noise), len(noise_var_used))
    noise_corr = float(np.corrcoef(gt_noise[:n_shells], noise_var_used[:n_shells])[0, 1])
    rel_err = np.abs(gt_noise[:n_shells] - noise_var_used[:n_shells]) / (gt_noise[:n_shells] + 1e-12)

    results = {
        "noise_correlation": noise_corr,
        "noise_mean_relative_error": float(np.mean(rel_err)),
        "noise_median_relative_error": float(np.median(rel_err)),
    }
    _assert_metrics(results, baseline_scores, tol_frac)


# ---------------------------------------------------------------------------
# Test 3: compute_variance
# ---------------------------------------------------------------------------

def test_compute_variance(cryos, gt_data, intermediates, baseline_scores, tol_frac):
    """Test variance estimation in isolation.

    Uses ~/recovar's mean and noise as standardized inputs.

    Two metrics:
        variance_fourier_fsc: GT Fourier variance vs estimated Fourier variance.
            Both are per-Fourier-voxel power: sum_i |V_i(k)|^2.
        variance_spatial_fsc: Estimated spatial variance (IDFT of Fourier
            estimate) vs GT spatial variance, both DFT'd then FSC.

    NOTE: compute_variance was refactored (safe_div, half-image processing),
    so baseline scores reflect the current code, not ~/recovar.
    """
    from recovar.heterogeneity import covariance_estimation
    from recovar.reconstruction import noise
    from recovar.output import plot_utils
    from recovar.core import fourier_transform_utils
    from recovar import utils

    batch_size = 512
    volume_shape = cryos.volume_shape
    mean_combined = intermediates["mean_combined"]
    noise_var_used = intermediates["noise_var_used"]
    dilated_volume_mask = intermediates["dilated_volume_mask"]

    # Set noise model on cryos
    noise.update_noise_variance(noise_var_used, cryos)

    variance_est, _, _, _, _ = covariance_estimation.compute_variance(
        cryos, mean_combined, utils.safe_batch_size(batch_size // 2),
        dilated_volume_mask, use_regularization=True, disc_type="cubic"
    )

    est_fourier_var = variance_est["combined"]  # per-Fourier-voxel variance

    # Metric 1: Fourier variance FSC
    # GT Fourier variance (per-voxel power in Fourier space) vs estimated
    gt_fourier_var = gt_data["gt_fourier_variance"]
    _, fourier_var_fsc = plot_utils.plot_fsc_new(
        gt_fourier_var, est_fourier_var,
        np.array(volume_shape), gt_data["voxel_size"],
        threshold=0.5, name="Variance Fourier FSC"
    )

    # Metric 2: Spatial variance FSC
    # IDFT estimated Fourier variance → spatial estimate; DFT both → FSC
    gt_spatial_var = gt_data["gt_spatial_variance"]
    est_spatial_var = fourier_transform_utils.get_idft3(
        est_fourier_var.reshape(volume_shape)
    ).real.reshape(-1)
    gt_spatial_dft = fourier_transform_utils.get_dft3(
        gt_spatial_var.reshape(volume_shape)
    ).reshape(-1)
    est_spatial_dft = fourier_transform_utils.get_dft3(
        est_spatial_var.reshape(volume_shape)
    ).reshape(-1)
    _, spatial_var_fsc = plot_utils.plot_fsc_new(
        gt_spatial_dft, est_spatial_dft,
        np.array(volume_shape), gt_data["voxel_size"],
        threshold=0.5, name="Variance Spatial FSC"
    )

    print(f"  variance_fourier_fsc = {fourier_var_fsc:.6f}")
    print(f"  variance_spatial_fsc = {spatial_var_fsc:.6f}")

    results = {
        "variance_fourier_fsc": float(fourier_var_fsc),
        "variance_spatial_fsc": float(spatial_var_fsc),
    }
    _assert_metrics(results, baseline_scores, tol_frac)


# ---------------------------------------------------------------------------
# Test 4: covariance_columns
# ---------------------------------------------------------------------------

def test_covariance_columns(cryos, gt_data, intermediates, baseline_scores, tol_frac):
    """Test covariance column computation in isolation.

    Uses ~/recovar's mean, noise, and mask. Uses the same picked_frequencies
    from the baseline.
    """
    from recovar.heterogeneity import covariance_estimation, principal_components
    from recovar.reconstruction import noise
    from recovar.output import plot_utils

    batch_size = 512
    gpu_memory = 40.0
    volume_shape = cryos.volume_shape

    # Standardized inputs from baseline
    mean_combined = intermediates["mean_combined"]
    mean_prior = intermediates["mean_prior"]
    means_lhs = intermediates["means_lhs"]
    noise_var_used = intermediates["noise_var_used"]
    volume_mask = intermediates["volume_mask"]
    dilated_volume_mask = intermediates["dilated_volume_mask"]
    picked_frequencies = intermediates["picked_frequencies"]

    # Rebuild means dict
    means = {
        "combined": mean_combined,
        "prior": mean_prior,
        "lhs": means_lhs,
        "corrected0": intermediates.get("mean_corrected0", mean_combined),
        "corrected1": intermediates.get("mean_corrected1", mean_combined),
    }

    noise.update_noise_variance(noise_var_used, cryos)
    valid_idx = cryos.get_valid_frequency_indices()

    covariance_options = covariance_estimation.get_default_covariance_computation_options()

    cov_cols, _, col_fscs = covariance_estimation.compute_regularized_covariance_columns_in_batch(
        cryos, means, mean_prior, volume_mask, dilated_volume_mask,
        valid_idx, gpu_memory, covariance_options, picked_frequencies
    )

    # Compare against GT covariance columns
    gt_cov_cols = gt_data["gt"].get_covariance_columns(picked_frequencies, contrasted=False)
    est_cov_cols = cov_cols.get("est_mask", cov_cols.get("est"))

    if est_cov_cols is not None and gt_cov_cols is not None:
        col_fsc_scores = []
        n_cols = min(est_cov_cols.shape[1], gt_cov_cols.shape[1])
        for i in range(n_cols):
            _, col_fsc = plot_utils.plot_fsc_new(
                gt_cov_cols[:, i], est_cov_cols[:, i],
                np.array(volume_shape), gt_data["voxel_size"], threshold=0.5
            )
            col_fsc_scores.append(float(col_fsc))
        results = {
            "covariance_columns_mean_fsc": float(np.mean(col_fsc_scores)),
            "covariance_columns_median_fsc": float(np.median(col_fsc_scores)),
        }
        _assert_metrics(results, baseline_scores, tol_frac)


# ---------------------------------------------------------------------------
# Test 5: projected_covariance
# ---------------------------------------------------------------------------

def test_projected_covariance(cryos, gt_data, intermediates, baseline_scores, tol_frac):
    """Test projected covariance in isolation with GT eigenvectors as basis.

    Uses ~/recovar's mean and mask. GT eigenvectors (Fourier) are used as
    the projection basis, decoupling from covariance column estimation.

    Compares the current code's projected covariance matrix element-wise
    against the baseline's saved projected covariance (from ~/recovar).

    NOTE: compute_projected_covariance was deliberately changed (removed
    half-image path and Tikhonov regularization to fix catastrophic
    cancellation).  The baseline Frobenius error of ~0.27 reflects this
    intentional divergence from ~/recovar.
    """
    from recovar.heterogeneity import covariance_estimation
    from recovar.reconstruction import noise

    batch_size = 512
    mean_combined = intermediates["mean_combined"]
    noise_var_used = intermediates["noise_var_used"]
    volume_mask = intermediates["volume_mask"]

    noise.update_noise_variance(noise_var_used, cryos)

    # GT eigenvectors in Fourier space as basis — (vol_size, n_pcs)
    n_gt_pcs = min(10, gt_data["u_gt_fourier"].shape[1])
    gt_basis = gt_data["u_gt_fourier"][:, :n_gt_pcs]  # (vol_size, n_pcs)

    proj_cov = covariance_estimation.compute_projected_covariance(
        cryos, mean_combined, gt_basis, volume_mask,
        batch_size, disc_type="linear_interp", disc_type_u="linear_interp"
    )

    # Compare against baseline's projected covariance (same GT basis, old code)
    baseline_proj_cov = intermediates.get("projected_covariance_gt_basis")
    if baseline_proj_cov is not None:
        frob_err = float(
            np.linalg.norm(proj_cov - baseline_proj_cov)
            / (np.linalg.norm(baseline_proj_cov) + 1e-12)
        )

        # Also compare against GT theoretical projected covariance = diag(s^2)
        gt_proj_cov = np.diag(gt_data["s_gt_fourier"][:n_gt_pcs] ** 2)
        frob_err_vs_gt = float(
            np.linalg.norm(proj_cov - gt_proj_cov)
            / (np.linalg.norm(gt_proj_cov) + 1e-12)
        )

        print(f"  projected_covariance_relative_frobenius_error (vs old code) = {frob_err:.6f}")
        print(f"  projected_covariance_relative_frobenius_error (vs GT diag) = {frob_err_vs_gt:.6f}")
        print(f"  proj_cov diagonal: {np.diag(proj_cov)}")
        print(f"  GT diag(s^2):      {np.diag(gt_proj_cov)}")

        results = {"projected_covariance_relative_frobenius_error": frob_err}
        _assert_metrics(results, baseline_scores, tol_frac)
    else:
        pytest.skip("No baseline projected covariance available")


# ---------------------------------------------------------------------------
# Test 6: principal_components (full: columns → SVD → projected cov)
# ---------------------------------------------------------------------------

def test_principal_components(cryos, gt_data, intermediates, baseline_scores, tol_frac):
    """Test full principal component estimation in isolation.

    Uses ~/recovar's mean, noise, mask, and variance as standardized inputs.
    """
    from recovar.heterogeneity import principal_components, covariance_estimation
    from recovar.reconstruction import noise
    from recovar.output import metrics
    from recovar.core import linalg

    batch_size = 512
    gpu_memory = 40.0
    volume_shape = cryos.volume_shape
    volume_size = int(np.prod(volume_shape))
    vol_norm = np.sqrt(volume_size)

    # Standardized inputs
    mean_combined = intermediates["mean_combined"]
    mean_prior = intermediates["mean_prior"]
    means_lhs = intermediates["means_lhs"]
    noise_var_used = intermediates["noise_var_used"]
    volume_mask = intermediates["volume_mask"]
    dilated_volume_mask = intermediates["dilated_volume_mask"]
    variance_combined = intermediates["variance_combined"]

    means = {
        "combined": mean_combined,
        "prior": mean_prior,
        "lhs": means_lhs,
        "corrected0": intermediates.get("mean_corrected0", mean_combined),
        "corrected1": intermediates.get("mean_corrected1", mean_combined),
    }

    noise.update_noise_variance(noise_var_used, cryos)
    valid_idx = cryos.get_valid_frequency_indices()

    options = {
        "zs_dim_to_test": [4, 10],
        "contrast": "contrast_qr",
        "keep_intermediate": False,
        "ignore_zero_frequency": True,
        "use_combined_mean": True,
    }
    covariance_options = covariance_estimation.get_default_covariance_computation_options()

    u, s, _, _, _ = principal_components.estimate_principal_components(
        cryos, options, means, mean_prior, volume_mask, dilated_volume_mask,
        valid_idx, batch_size, gpu_memory_to_use=gpu_memory,
        covariance_options=covariance_options,
        variance_estimate=variance_combined,
        use_reg_mean_in_contrast=False
    )

    # Convert to real space for metrics
    n_pcs = min(20, u["rescaled"].shape[1])
    u_real = linalg.batch_idft3(
        u["rescaled"][:, :n_pcs], volume_shape, batch_size=2
    ).real
    u_est = np.array(u_real.reshape(volume_size, n_pcs)) * vol_norm
    _, rel_var, _ = metrics.get_all_variance_scores(
        u_est, gt_data["u_gt"], gt_data["s_gt"]
    )

    results = {}
    for k in [2, 4, 5, 10]:
        if rel_var.size > k:
            results[f"pcs_relative_variance_{k}"] = float(rel_var[k])

    print("  PCS relative variance:", {k: f"{v:.6f}" for k, v in results.items()})
    assert results, "No PCS metrics computed"
    _assert_metrics(results, baseline_scores, tol_frac)


# ---------------------------------------------------------------------------
# Test 7: embedding with contrast
# ---------------------------------------------------------------------------

def test_embedding_with_contrast(cryos, gt_data, intermediates, baseline_scores, tol_frac):
    """Test per-image embedding with contrast correction.

    Uses ~/recovar's mean, eigenvectors, and eigenvalues as standardized
    inputs — isolates embedding from PC estimation.
    """
    from recovar.heterogeneity import embedding
    from recovar.reconstruction import noise
    from recovar.output import metrics

    gpu_memory = 40.0
    mean_combined = intermediates["mean_combined"]
    noise_var_used = intermediates["noise_var_used"]
    volume_mask = intermediates["volume_mask"]
    u_rescaled = intermediates["u_rescaled"]
    s_rescaled = intermediates["s_rescaled"]

    noise.update_noise_variance(noise_var_used, cryos)

    results = {}
    for zdim in [4, 10]:
        zs, _, est_contrasts, _ = embedding.get_per_image_embedding(
            mean_combined, u_rescaled, s_rescaled, zdim,
            cryos, volume_mask, gpu_memory, "linear_interp",
            contrast_grid=None, contrast_option="contrast",
            ignore_zero_frequency=True
        )

        _, avg_var = metrics.variance_of_zs(zs, gt_data["pa"])
        results[f"embedding_squared_error_{zdim}"] = float(avg_var)

        contrast_mae = float(
            np.mean(np.abs(gt_data["gt_contrasts"] - est_contrasts.ravel()))
        )
        results[f"contrasts_{zdim}"] = contrast_mae

    _assert_metrics(results, baseline_scores, tol_frac)


# ---------------------------------------------------------------------------
# Test 8: embedding without contrast (no regularization)
# ---------------------------------------------------------------------------

def test_embedding_without_contrast(cryos, gt_data, intermediates, baseline_scores, tol_frac):
    """Test per-image embedding without eigenvalue regularization.

    Same as test 7 but s → inf (unregularized least-squares).
    """
    from recovar.heterogeneity import embedding
    from recovar.reconstruction import noise
    from recovar.output import metrics

    gpu_memory = 40.0
    mean_combined = intermediates["mean_combined"]
    noise_var_used = intermediates["noise_var_used"]
    volume_mask = intermediates["volume_mask"]
    u_rescaled = intermediates["u_rescaled"]
    s_rescaled = intermediates["s_rescaled"]

    noise.update_noise_variance(noise_var_used, cryos)

    results = {}
    for zdim in [4, 10]:
        zs_noreg, _, est_contrasts_noreg, _ = embedding.get_per_image_embedding(
            mean_combined, u_rescaled, s_rescaled * 0 + np.inf, zdim,
            cryos, volume_mask, gpu_memory, "linear_interp",
            contrast_grid=None, contrast_option="contrast",
            ignore_zero_frequency=True
        )

        _, avg_var = metrics.variance_of_zs(zs_noreg, gt_data["pa"])
        results[f"embedding_squared_error_{zdim}_noreg"] = float(avg_var)

        contrast_mae = float(
            np.mean(np.abs(gt_data["gt_contrasts"] - est_contrasts_noreg.ravel()))
        )
        results[f"contrasts_{zdim}_noreg"] = contrast_mae

    _assert_metrics(results, baseline_scores, tol_frac)
