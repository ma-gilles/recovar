"""Isolated checks for eight covariance pipeline stages.

The default case generates a deterministic 5NRL trajectory and particles in
pytest's temporary directory. Each stage receives known inputs derived from
the synthetic ground truth, so a failure in one stage cannot supply bad
inputs to the remaining tests.

Set both FUNCTION_TEST_DATASET_DIR and FUNCTION_BASELINE_DIR to run the
historical 128-cube experiment with its saved intermediates. In that mode,
FUNCTION_BASELINE_SCORES and FUNCTION_TEST_TOL_FRAC keep their original
meaning. Generated cases use truth-based checks and write per-test scores
next to the temporary data.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
from helpers.metrics_regression import log_comparison_table, metric_direction
from helpers.perf_regression import build_perf_record, check_perf_regression, perf_snapshot, stage_perf

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.io,
    pytest.mark.long_test,
]

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_FUNCTION_BASELINE_SCORES = (
    _REPO_ROOT / "tests" / "baselines" / "pipeline_functions_isolated" / "pdb_5nrl" / "per_function_scores.json"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def archive():
    dataset = os.environ.get("FUNCTION_TEST_DATASET_DIR")
    baseline = os.environ.get("FUNCTION_BASELINE_DIR")
    if dataset is None and baseline is None:
        return None
    if not dataset or not baseline:
        pytest.fail("Set both FUNCTION_TEST_DATASET_DIR and FUNCTION_BASELINE_DIR for the historical run")
    return Path(dataset), Path(baseline)


@pytest.fixture(scope="module")
def tol_frac():
    return float(os.environ.get("FUNCTION_TEST_TOL_FRAC", "0.10"))


@pytest.fixture(scope="module")
def baseline_scores(archive):
    if archive is None:
        return None
    path = Path(os.environ.get("FUNCTION_BASELINE_SCORES", str(_DEFAULT_FUNCTION_BASELINE_SCORES)))
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def dataset_dir(archive, tmp_path_factory):
    if archive is not None:
        dataset, _ = archive
        if not dataset.is_dir():
            pytest.fail(f"Archived dataset not found: {dataset}")
        return dataset

    from recovar.commands.run_test_all_metrics import make_big_test_dataset
    from recovar.simulation.trajectory_generation import generate_trajectory_volumes

    output_dir = tmp_path_factory.mktemp("isolated_5nrl")
    grid_size = 32
    prefix = generate_trajectory_volumes(
        output_dir, grid_size=grid_size, n_volumes=12, max_rotation_degrees=20, volume_workers=2
    )
    make_big_test_dataset(prefix, str(output_dir), grid_size=grid_size, n_images=2048, noise_level=0.025)
    return output_dir / "test_dataset"


@pytest.fixture(scope="module")
def intermediates(archive, gt_data):
    if archive is not None:
        _, baseline = archive
        intdir = baseline / "intermediates"
        paths = sorted(intdir.glob("*.npy"))
        if not paths:
            pytest.fail(f"No archived intermediate arrays found under {intdir}")
        return {path.stem: np.load(path, allow_pickle=False) for path in paths}

    from scipy.ndimage import binary_dilation

    from recovar.core.mask import soften_volume_mask
    from recovar.heterogeneity.covariance_core import get_picked_frequencies
    from recovar.output.metrics import make_union_gt_mask_from_hvd

    gt = gt_data["gt"]
    shape = gt.volume_shape
    volume_mask, binary_mask = make_union_gt_mask_from_hvd(gt, shape)
    dilated_volume_mask = soften_volume_mask(binary_dilation(binary_mask, iterations=2))
    mean = np.asarray(gt_data["gt_mean"], dtype=np.complex64)
    # Shell-independent prior scale from the known mean power. The low-frequency
    # column policy does not use the LHS to choose columns.
    mean_prior = np.full(mean.shape, np.mean(np.abs(mean) ** 2), dtype=np.float32)
    return {
        "mean_combined": mean,
        "mean_prior": mean_prior,
        "means_lhs": np.ones(mean.shape, dtype=np.float32),
        "noise_var_used": gt_data["gt_noise"].astype(np.float32),
        "volume_mask": np.asarray(volume_mask, dtype=np.float32),
        "dilated_volume_mask": np.asarray(dilated_volume_mask, dtype=np.float32),
        "picked_frequencies": np.asarray(get_picked_frequencies(shape, radius=3, use_half=True)),
        "variance_combined": gt_data["gt_fourier_variance"].astype(np.float32),
        "u_rescaled": gt_data["u_gt_fourier"].astype(np.complex64),
        "s_rescaled": (gt_data["s_gt_fourier"] ** 2).astype(np.float32),
    }


@pytest.fixture(scope="module")
def gt_data(dataset_dir, archive):
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
        contrasted=False, real_space=True, random_svd_pcs=200 if archive is not None else None
    )
    u_gt_fourier, s_gt_fourier, _ = gt.get_vol_svd(contrasted=False, real_space=False)

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


@pytest.fixture
def cryos(dataset_dir, intermediates, gt_data):
    """Create CryoEMDataset with halfset_indices from the shared dataset."""
    from recovar.data_io import halfsets

    grid_size = gt_data["grid_size"]
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
        n_images = gt_data["pa"].size
        rng = np.random.RandomState(0)
        perm = rng.permutation(n_images)
        ind_split = [perm[: n_images // 2], perm[n_images // 2 :]]

    dataset_spec = halfsets.HalfsetDatasetSpec(
        particles_file=particles_file,
        poses_file=poses_file,
        ctf_file=ctf_file,
        datadir=None,
    )
    ds = halfsets.load_halfset_dataset(
        dataset_spec,
        ind_split=ind_split,
        lazy=True,
    )
    # Initialize noise model (same as pipeline.py line 737)
    ds.set_radial_noise_model(None)
    return ds


def _assert_metrics(results, baseline_scores, tol_frac, tmp_path, minimums=None, maximums=None):
    if baseline_scores is not None:
        directions = metric_direction
        expected = set(results)
    else:
        minimums = minimums or {}
        maximums = maximums or {}
        assert minimums.keys().isdisjoint(maximums), "conflicting metric directions"
        baseline_scores = minimums | maximums

        def directions(key):
            return "higher" if key in minimums else "lower"

        tol_frac = 0.0
        expected = set(baseline_scores)

    assert expected <= results.keys(), f"missing scores: {expected - results.keys()}"
    assert expected <= baseline_scores.keys(), f"missing references: {expected - baseline_scores.keys()}"
    assert results and all(np.isfinite(v) for v in results.values()), results
    (tmp_path / "isolated_metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    checked, failures = log_comparison_table(results, baseline_scores, tol_frac, direction_fn=directions)
    assert checked, "no metrics compared"
    assert not failures, "regressions:\n" + "\n".join(failures)


def _covariance_options(portable):
    from recovar.heterogeneity.covariance_estimation import get_default_covariance_computation_options

    options = get_default_covariance_computation_options()
    if portable:
        options.update(
            column_sampling_scheme="low_freqs",
            column_radius=3,
            n_pcs_to_compute=12,
            randomized_sketch_size=16,
            column_batch_size=16,
        )
    return options


def _check_perf(baseline_scores, name, before):
    if baseline_scores is not None:
        path = _DEFAULT_FUNCTION_BASELINE_SCORES.with_name("perf_baseline.json")
        record = build_perf_record({name: stage_perf(before, perf_snapshot())})
        check_perf_regression(record, str(path), f"test_{name}")


# ---------------------------------------------------------------------------
# Test 1: compute_mean
# ---------------------------------------------------------------------------


def test_compute_mean(cryos, gt_data, intermediates, baseline_scores, tol_frac, tmp_path):
    """Test mean reconstruction in isolation.

    Uses raw images from the dataset. Compares FSC vs GT mean.
    """
    from recovar.output import plot_utils
    from recovar.reconstruction import homogeneous, noise

    snap_before = perf_snapshot()

    batch_size = 128 if baseline_scores is None else 512
    volume_shape = cryos.volume_shape

    # Initial noise from half-maps
    noise_var_from_hf, _ = noise.estimate_noise_variance(cryos, batch_size)

    means, mean_prior, _ = homogeneous.get_mean_conformation_relion(
        cryos, batch_size, noise_variance=noise_var_from_hf, use_regularization=False
    )

    _, mean_fsc_score = plot_utils.plot_fsc_new(
        gt_data["gt_mean"],
        means.combined,
        np.array(volume_shape),
        gt_data["voxel_size"],
        threshold=0.5,
        name="Mean FSC",
    )

    results = {"mean_fsc": float(mean_fsc_score)}
    _assert_metrics(
        results,
        baseline_scores,
        tol_frac,
        tmp_path,
        minimums={"mean_fsc": 6 / (gt_data["grid_size"] * gt_data["voxel_size"])},
    )

    _check_perf(baseline_scores, "compute_mean", snap_before)


# ---------------------------------------------------------------------------
# Test 2: estimate_noise
# ---------------------------------------------------------------------------


def test_estimate_noise(cryos, gt_data, intermediates, baseline_scores, tol_frac, tmp_path):
    """Test noise estimation in isolation.

    Uses ~/recovar's mean and mask as standardized inputs to decouple
    from mean estimation quality.
    """
    snap_before = perf_snapshot()

    from recovar.reconstruction import noise

    batch_size = 128 if baseline_scores is None else 512
    mean_combined = intermediates["mean_combined"]
    dilated_volume_mask = intermediates["dilated_volume_mask"]
    gt_noise = gt_data["gt_noise"]

    # Outside mask
    masked_image_PS, _, _ = noise.estimate_noise_variance_from_outside_mask_v2(cryos, dilated_volume_mask, batch_size)
    # Upper bound inside mask
    radial_ub_noise_var, _, _ = noise.estimate_radial_noise_upper_bound_from_inside_mask_v2(
        cryos, mean_combined, dilated_volume_mask, batch_size
    )
    # Image PS for fallback
    _, _, image_PS, _ = noise.estimate_radial_noise_statistic_from_outside_mask(cryos, dilated_volume_mask, batch_size)

    noise_var_used = np.where(masked_image_PS > radial_ub_noise_var, radial_ub_noise_var, masked_image_PS)
    noise_var_used = np.where(noise_var_used < 0, image_PS / 10, noise_var_used)

    n_shells = min(len(gt_noise), len(noise_var_used))
    noise_corr = float(np.corrcoef(gt_noise[:n_shells], noise_var_used[:n_shells])[0, 1])
    rel_err = np.abs(gt_noise[:n_shells] - noise_var_used[:n_shells]) / (gt_noise[:n_shells] + 1e-12)

    results = {
        "noise_correlation": noise_corr,
        "noise_mean_relative_error": float(np.mean(rel_err)),
        "noise_median_relative_error": float(np.median(rel_err)),
    }
    if baseline_scores is None:
        from recovar.core import fourier_transform_utils

        radii = np.asarray(fourier_transform_utils.get_grid_of_radial_distances(cryos.image_shape))
        n_pixels = np.bincount(radii.ravel(), minlength=n_shells)[:n_shells]
        # Integrate error across observed Fourier pixels. A single, masked DC
        # pixel contributes less to the noise model than an entire high shell.
        results["noise_pixel_mean_relative_error"] = float(np.average(rel_err, weights=n_pixels))
    _assert_metrics(
        results,
        baseline_scores,
        tol_frac,
        tmp_path,
        minimums={"noise_correlation": 0.90},
        maximums={"noise_pixel_mean_relative_error": 0.15, "noise_median_relative_error": 0.15},
    )

    _check_perf(baseline_scores, "estimate_noise", snap_before)


# ---------------------------------------------------------------------------
# Test 3: compute_variance
# ---------------------------------------------------------------------------


def test_compute_variance(cryos, gt_data, intermediates, baseline_scores, tol_frac, tmp_path):
    """Test variance estimation in isolation.

    Uses ~/recovar's mean and noise as standardized inputs.

    Metric: variance_fourier_fsc — GT Fourier variance vs estimated Fourier
    variance, both per-Fourier-voxel power: sum_i |V_i(k)|^2.

    The spatial variance metric (from eigendecomposition) is in
    test_principal_components, where the eigenvectors are available.

    NOTE: compute_variance was refactored (safe_div, half-image processing),
    so baseline scores reflect the current code, not ~/recovar.
    """
    snap_before = perf_snapshot()

    from recovar import utils
    from recovar.heterogeneity import covariance_estimation
    from recovar.output import plot_utils
    from recovar.reconstruction import noise

    batch_size = 128 if baseline_scores is None else 512
    volume_shape = cryos.volume_shape
    mean_combined = intermediates["mean_combined"]
    noise_var_used = intermediates["noise_var_used"]
    dilated_volume_mask = intermediates["dilated_volume_mask"]

    # Set noise model on cryos
    noise.update_noise_variance(noise_var_used, cryos)

    variance_est, _, _, _, _ = covariance_estimation.compute_variance(
        cryos,
        mean_combined,
        utils.safe_batch_size(batch_size // 2),
        dilated_volume_mask,
        use_regularization=True,
        disc_type="cubic",
    )

    est_fourier_var = variance_est["combined"]  # per-Fourier-voxel variance

    # Fourier variance FSC: GT vs estimated, both per-voxel power in Fourier space
    gt_fourier_var = gt_data["gt_fourier_variance"]
    _, fourier_var_fsc = plot_utils.plot_fsc_new(
        gt_fourier_var,
        est_fourier_var,
        np.array(volume_shape),
        gt_data["voxel_size"],
        threshold=0.5,
        name="Variance Fourier FSC",
    )

    print(f"  variance_fourier_fsc = {fourier_var_fsc:.6f}")

    results = {"variance_fourier_fsc": float(fourier_var_fsc)}
    _assert_metrics(
        results,
        baseline_scores,
        tol_frac,
        tmp_path,
        minimums={"variance_fourier_fsc": 2 / (gt_data["grid_size"] * gt_data["voxel_size"])},
    )

    _check_perf(baseline_scores, "compute_variance", snap_before)


# ---------------------------------------------------------------------------
# Test 4: covariance_columns
# ---------------------------------------------------------------------------


def test_covariance_columns(cryos, gt_data, intermediates, baseline_scores, tol_frac, tmp_path):
    """Test covariance column computation in isolation.

    Uses ~/recovar's mean, noise, and mask. Uses the same picked_frequencies
    from the baseline.
    """
    snap_before = perf_snapshot()

    from recovar.heterogeneity import covariance_estimation
    from recovar.output import plot_utils
    from recovar.reconstruction import noise

    batch_size = 128 if baseline_scores is None else 512
    gpu_memory = 0.25 if baseline_scores is None else 40.0
    volume_shape = cryos.volume_shape

    # Standardized inputs from baseline
    mean_combined = intermediates["mean_combined"]
    mean_prior = intermediates["mean_prior"]
    means_lhs = intermediates["means_lhs"]
    noise_var_used = intermediates["noise_var_used"]
    volume_mask = intermediates["volume_mask"]
    dilated_volume_mask = intermediates["dilated_volume_mask"]
    picked_frequencies = intermediates["picked_frequencies"]

    # Rebuild means as MeanEstimate
    from recovar.reconstruction.homogeneous import MeanEstimate

    means = MeanEstimate(
        combined=mean_combined,
        corrected0=intermediates.get("mean_corrected0", mean_combined),
        corrected1=intermediates.get("mean_corrected1", mean_combined),
        corrected0reg=intermediates.get("mean_corrected0", mean_combined),
        corrected1reg=intermediates.get("mean_corrected1", mean_combined),
        lhs=means_lhs,
        prior=mean_prior,
    )

    noise.update_noise_variance(noise_var_used, cryos)
    valid_idx = cryos.get_valid_frequency_indices()

    covariance_options = _covariance_options(baseline_scores is None)

    cov_cols, _, col_fscs = covariance_estimation.compute_regularized_covariance_columns_in_batch(
        cryos,
        means,
        mean_prior,
        volume_mask,
        dilated_volume_mask,
        valid_idx,
        gpu_memory,
        covariance_options,
        picked_frequencies,
    )

    # The isolated estimator sees the simulated per-image contrast variation.
    # The historical scores used the structural covariance as their reference.
    gt_cov_cols = gt_data["gt"].get_covariance_columns(picked_frequencies, contrasted=baseline_scores is None)
    est_cov_cols = cov_cols.get("est_mask", cov_cols.get("est"))

    assert est_cov_cols is not None and gt_cov_cols is not None
    assert est_cov_cols.shape == gt_cov_cols.shape
    if baseline_scores is None:
        # Compare whole column matrices, including low and high spatial frequencies.
        correlation = np.vdot(gt_cov_cols, est_cov_cols).real / (
            np.linalg.norm(gt_cov_cols) * np.linalg.norm(est_cov_cols)
        )
        _assert_metrics(
            {"column_correlation": float(correlation)},
            baseline_scores,
            tol_frac,
            tmp_path,
            minimums={"column_correlation": 0.5},
        )
    else:
        col_fsc_scores = []
        n_cols = min(est_cov_cols.shape[1], gt_cov_cols.shape[1])
        for i in range(n_cols):
            _, col_fsc = plot_utils.plot_fsc_new(
                gt_cov_cols[:, i], est_cov_cols[:, i], np.array(volume_shape), gt_data["voxel_size"], threshold=0.5
            )
            col_fsc_scores.append(float(col_fsc))
        results = {
            "covariance_columns_mean_fsc": float(np.mean(col_fsc_scores)),
            "covariance_columns_median_fsc": float(np.median(col_fsc_scores)),
        }
        _assert_metrics(results, baseline_scores, tol_frac, tmp_path)

    _check_perf(baseline_scores, "covariance_columns", snap_before)


# ---------------------------------------------------------------------------
# Test 5: projected_covariance
# ---------------------------------------------------------------------------


def test_projected_covariance(cryos, gt_data, intermediates, baseline_scores, tol_frac, tmp_path):
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
    snap_before = perf_snapshot()

    from recovar.heterogeneity import covariance_estimation
    from recovar.reconstruction import noise

    batch_size = 128 if baseline_scores is None else 512
    mean_combined = intermediates["mean_combined"]
    noise_var_used = intermediates["noise_var_used"]
    volume_mask = intermediates["volume_mask"]

    noise.update_noise_variance(noise_var_used, cryos)

    # GT eigenvectors in Fourier space as basis — (vol_size, n_pcs)
    n_gt_pcs = min(4 if baseline_scores is None else 10, gt_data["u_gt_fourier"].shape[1])
    gt_basis = gt_data["u_gt_fourier"][:, :n_gt_pcs]  # (vol_size, n_pcs)

    proj_cov = covariance_estimation.compute_projected_covariance(
        cryos,
        mean_combined,
        gt_basis,
        volume_mask,
        batch_size,
        disc_type="cubic" if baseline_scores is None else "linear_interp",
        disc_type_u="cubic" if baseline_scores is None else "linear_interp",
    )

    if baseline_scores is None:
        # The projected-covariance call does not correct per-image contrast.
        coefficients = gt_basis.conj().T @ gt_data["gt"].get_covariance_square_root(contrasted=True)
        gt_proj_cov = (coefficients @ coefficients.conj().T).real
    else:
        gt_proj_cov = np.diag(gt_data["s_gt_fourier"][:n_gt_pcs] ** 2)
    proj_cov = np.asarray(proj_cov)
    assert proj_cov.shape == gt_proj_cov.shape
    results = {
        "proj_cov_frobenius_error_vs_gt": float(np.linalg.norm(proj_cov - gt_proj_cov) / np.linalg.norm(gt_proj_cov)),
        "proj_cov_off_diagonal_relative_error": float(
            np.linalg.norm(proj_cov - np.diag(np.diag(proj_cov))) / np.linalg.norm(proj_cov)
        ),
    }
    if baseline_scores is not None:
        baseline_proj_cov = intermediates["projected_covariance_gt_basis"]
        results["projected_covariance_relative_frobenius_error"] = float(
            np.linalg.norm(proj_cov - baseline_proj_cov) / np.linalg.norm(baseline_proj_cov)
        )
    _assert_metrics(
        results,
        baseline_scores,
        tol_frac,
        tmp_path,
        maximums={"proj_cov_frobenius_error_vs_gt": 0.5, "proj_cov_off_diagonal_relative_error": 0.25},
    )

    _check_perf(baseline_scores, "projected_covariance", snap_before)


# ---------------------------------------------------------------------------
# Test 6: principal_components (full: columns → SVD → projected cov)
# ---------------------------------------------------------------------------


def test_principal_components(cryos, gt_data, intermediates, baseline_scores, tol_frac, tmp_path):
    """Test full principal component estimation in isolation.

    Uses ~/recovar's mean, noise, mask, and variance as standardized inputs.
    """
    snap_before = perf_snapshot()

    from recovar.core import linalg
    from recovar.heterogeneity import principal_components
    from recovar.output import metrics
    from recovar.reconstruction import noise

    batch_size = 128 if baseline_scores is None else 512
    gpu_memory = 0.25 if baseline_scores is None else 40.0
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

    from recovar.reconstruction.homogeneous import MeanEstimate

    means = MeanEstimate(
        combined=mean_combined,
        corrected0=intermediates.get("mean_corrected0", mean_combined),
        corrected1=intermediates.get("mean_corrected1", mean_combined),
        corrected0reg=intermediates.get("mean_corrected0", mean_combined),
        corrected1reg=intermediates.get("mean_corrected1", mean_combined),
        lhs=means_lhs,
        prior=mean_prior,
    )

    noise.update_noise_variance(noise_var_used, cryos)
    valid_idx = cryos.get_valid_frequency_indices()

    from recovar.utils.helpers import AlgorithmOptions

    options = AlgorithmOptions(
        volume_mask_option="none",
        zs_dim_to_test=[4, 10],
        contrast="contrast_qr",
        keep_intermediate=False,
        ignore_zero_frequency=True,
    )
    covariance_options = _covariance_options(baseline_scores is None)

    u, s, _, _, _ = principal_components.estimate_principal_components(
        cryos,
        options,
        means,
        mean_prior,
        volume_mask,
        dilated_volume_mask,
        valid_idx,
        batch_size,
        gpu_memory_to_use=gpu_memory,
        covariance_options=covariance_options,
        variance_estimate=variance_combined,
        use_reg_mean_in_contrast=False,
    )

    # Convert to real space for metrics
    n_pcs = min(20, u["rescaled"].shape[1])
    u_real = linalg.batch_idft3(u["rescaled"][:, :n_pcs], volume_shape, batch_size=2).real
    u_est = np.array(u_real.reshape(volume_size, n_pcs)) * vol_norm
    _, rel_var, _ = metrics.get_all_variance_scores(u_est, gt_data["u_gt"], gt_data["s_gt"])

    results = {}
    for k in [2, 4, 5, 10]:
        if rel_var.size > k:
            results[f"pcs_relative_variance_{k}"] = float(rel_var[k])

    print("  PCS relative variance:", {k: f"{v:.6f}" for k, v in results.items() if k.startswith("pcs_")})
    _assert_metrics(
        results,
        baseline_scores,
        tol_frac,
        tmp_path,
        minimums={f"pcs_relative_variance_{k}": 0.5 for k in (2, 4, 5, 10)},
    )

    _check_perf(baseline_scores, "principal_components", snap_before)


# ---------------------------------------------------------------------------
# Test 7: embedding with contrast
# ---------------------------------------------------------------------------


def test_embedding_with_contrast(cryos, gt_data, intermediates, baseline_scores, tol_frac, tmp_path):
    """Test per-image embedding with contrast correction.

    Uses ~/recovar's mean, eigenvectors, and eigenvalues as standardized
    inputs — isolates embedding from PC estimation.
    """
    snap_before = perf_snapshot()

    from recovar.heterogeneity import embedding
    from recovar.output import metrics
    from recovar.reconstruction import noise

    gpu_memory = 0.25 if baseline_scores is None else 40.0
    mean_combined = intermediates["mean_combined"]
    noise_var_used = intermediates["noise_var_used"]
    volume_mask = intermediates["volume_mask"]
    u_rescaled = intermediates["u_rescaled"]
    s_rescaled = intermediates["s_rescaled"]

    noise.update_noise_variance(noise_var_used, cryos)

    image_ids = cryos.original_image_indices_from_local()
    labels = gt_data["pa"][image_ids]
    true_contrasts = gt_data["gt_contrasts"][image_ids]
    rng = np.random.default_rng(0)
    shuffled_labels = rng.permutation(labels)
    null_contrast_error = float(np.mean(np.abs(true_contrasts - 1)))
    results = {}
    limits = {}
    for zdim in [4, 10]:
        zs, _, est_contrasts, _ = embedding.get_per_image_embedding(
            mean_combined,
            u_rescaled,
            s_rescaled,
            zdim,
            cryos,
            volume_mask,
            gpu_memory,
            "linear_interp",
            contrast_grid=None,
            contrast_option="contrast",
            ignore_zero_frequency=True,
        )

        _, avg_var = metrics.variance_of_zs(zs, labels)
        results[f"embedding_squared_error_{zdim}"] = float(avg_var)

        contrast_mae = float(np.mean(np.abs(true_contrasts - est_contrasts.ravel())))
        results[f"contrasts_{zdim}"] = contrast_mae

        _, null_embedding_error = metrics.variance_of_zs(zs, shuffled_labels)
        limits[f"embedding_squared_error_{zdim}"] = float(null_embedding_error)
        limits[f"contrasts_{zdim}"] = null_contrast_error

    _assert_metrics(results, baseline_scores, tol_frac, tmp_path, maximums=limits)

    _check_perf(baseline_scores, "embedding_with_contrast", snap_before)


# ---------------------------------------------------------------------------
# Test 8: embedding without contrast (no regularization)
# ---------------------------------------------------------------------------


def test_embedding_without_contrast(cryos, gt_data, intermediates, baseline_scores, tol_frac, tmp_path):
    """Test per-image embedding without eigenvalue regularization.

    Same as test 7 but s → inf (unregularized least-squares).
    """
    snap_before = perf_snapshot()

    from recovar.heterogeneity import embedding
    from recovar.output import metrics
    from recovar.reconstruction import noise

    gpu_memory = 0.25 if baseline_scores is None else 40.0
    mean_combined = intermediates["mean_combined"]
    noise_var_used = intermediates["noise_var_used"]
    volume_mask = intermediates["volume_mask"]
    u_rescaled = intermediates["u_rescaled"]
    s_rescaled = intermediates["s_rescaled"]

    noise.update_noise_variance(noise_var_used, cryos)

    image_ids = cryos.original_image_indices_from_local()
    labels = gt_data["pa"][image_ids]
    true_contrasts = gt_data["gt_contrasts"][image_ids]
    rng = np.random.default_rng(0)
    shuffled_labels = rng.permutation(labels)
    null_contrast_error = float(np.mean(np.abs(true_contrasts - 1)))
    results = {}
    limits = {}
    for zdim in [4, 10]:
        zs_noreg, _, est_contrasts_noreg, _ = embedding.get_per_image_embedding(
            mean_combined,
            u_rescaled,
            np.full_like(s_rescaled, np.inf),
            zdim,
            cryos,
            volume_mask,
            gpu_memory,
            "linear_interp",
            contrast_grid=None,
            contrast_option="contrast",
            ignore_zero_frequency=True,
        )

        _, avg_var = metrics.variance_of_zs(zs_noreg, labels)
        results[f"embedding_squared_error_{zdim}_noreg"] = float(avg_var)

        contrast_mae = float(np.mean(np.abs(true_contrasts - est_contrasts_noreg.ravel())))
        results[f"contrasts_{zdim}_noreg"] = contrast_mae

        _, null_embedding_error = metrics.variance_of_zs(zs_noreg, shuffled_labels)
        limits[f"embedding_squared_error_{zdim}_noreg"] = float(null_embedding_error)
        limits[f"contrasts_{zdim}_noreg"] = null_contrast_error

    _assert_metrics(results, baseline_scores, tol_frac, tmp_path, maximums=limits)

    _check_perf(baseline_scores, "embedding_without_contrast", snap_before)
