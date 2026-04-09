#!/usr/bin/env python
"""
PPCA Scale Parameter Sweep for CryoBench Datasets (L1 and L2)

This script runs both L1 (sparse) and L2 (ridge) PPCA with scale parameter sweeps.
Results help guide parameter selection for real data.

Scale parameter interpretation:
- L1: sigma = scale * sqrt(variance_per_level / 2) → larger scale = more sparsity
- L2: W_prior = scale * data_variance → larger scale = less regularization (more variance allowed)

Warmstarting: Uses covariance-based PCA for initialization.

Usage (from repo root or with PYTHONPATH set to recovar parent):
  PYTHONPATH=/path/to/PPCA-EM-Notes/recovar python recovar/ppca/ppca_scale_sweep.py --base-dir /path/to/cryobench2 [--dataset Ribosembly]

Output locations:
  Results (simulated data, .npy): --results-dir (default: next to data, base-dir/ppca_sweep_results or .../ppca_sweep_results_10pc)
  Plots (.png): --plots-dir (default: ppca_plots, relative to cwd)

Compare both whitening strategies (Cz vs proj_ls vs no whitening):
  ... ppca_scale_sweep.py --compare-whitening [same other args]
"""

import argparse
import glob
import os
from datetime import datetime
from types import SimpleNamespace

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import utils
from recovar.data_io import halfsets
from recovar.heterogeneity import covariance_estimation, principal_components
from recovar.output import metrics, output
from recovar.ppca import ppca
from recovar.ppca.test_utils import compute_level_sigma
from recovar.reconstruction import homogeneous, regularization
from recovar.simulation import simulator, synthetic_dataset


def _log_jax_devices():
    print(f"JAX devices: {jax.devices()}")


def _with_trailing_separator(path):
    return path if path.endswith("/") else path + "/"


def _build_halfset_indices(n_images):
    midpoint = n_images // 2
    return [np.arange(midpoint), np.arange(midpoint, n_images)]


def _default_dataset_options():
    return {
        "particles_file": None,
        "poses_file": None,
        "ctf_file": None,
        "datadir": None,
        "strip_prefix": None,
        "ind": None,
        "n_images": -1,
        "padding": 0,
        "uninvert_data": False,
        "tilt_series": False,
        "tilt_series_ctf": "cryoem",
        "angle_per_tilt": None,
        "dose_per_tilt": None,
        "premultiplied_ctf": False,
        "downsample": None,
    }


def _load_dataset_with_halfsets(dataset_options, ind_split, lazy=False):
    spec = halfsets.HalfsetDatasetSpec(
        particles_file=dataset_options["particles_file"],
        poses_file=dataset_options.get("poses_file"),
        ctf_file=dataset_options.get("ctf_file"),
        datadir=dataset_options.get("datadir"),
        uninvert_data=dataset_options.get("uninvert_data", False),
        padding=dataset_options.get("padding", 0),
        n_images=dataset_options.get("n_images", -1),
        tilt_series=dataset_options.get("tilt_series", False),
        tilt_series_ctf=dataset_options.get("tilt_series_ctf", "cryoem"),
        angle_per_tilt=dataset_options.get("angle_per_tilt"),
        dose_per_tilt=dataset_options.get("dose_per_tilt"),
        premultiplied_ctf=dataset_options.get("premultiplied_ctf", False),
        strip_prefix=dataset_options.get("strip_prefix"),
        downsample_D=dataset_options.get("downsample"),
    )
    return halfsets.load_halfset_dataset(spec, ind_split=ind_split, lazy=lazy)


def _build_simulated_dataset_options(sim_output, grid_size):
    sim_output = _with_trailing_separator(sim_output)
    dataset_options = _default_dataset_options()
    dataset_options.update(
        {
            "ctf_file": sim_output + "ctf.pkl",
            "poses_file": sim_output + "poses.pkl",
            "particles_file": f"{sim_output}particles.{grid_size}.mrcs",
        }
    )
    return dataset_options


def _load_simulated_dataset(sim_output, grid_size, n_images, lazy=False):
    sim_output = _with_trailing_separator(sim_output)
    sim_info = utils.pickle_load(sim_output + "simulation_info.pkl")
    gt_results = synthetic_dataset.load_heterogeneous_reconstruction(sim_info)
    noise_variance = sim_info["noise_variance"]
    cryos = _load_dataset_with_halfsets(
        _build_simulated_dataset_options(sim_output, grid_size),
        _build_halfset_indices(n_images),
        lazy=lazy,
    )
    cryos.set_radial_noise_model(noise_variance)
    return cryos, sim_info, gt_results, noise_variance


def find_cryobench_datasets(base_dir):
    """Find all valid CryoBench dataset directories."""
    datasets = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            vol_dir = os.path.join(item_path, "vols", "128_org")
            if os.path.exists(vol_dir):
                mrc_files = glob.glob(os.path.join(vol_dir, "*.mrc"))
                if len(mrc_files) > 0:
                    datasets.append({"name": item, "vol_dir": vol_dir, "n_volumes": len(mrc_files)})
    return datasets


def generate_dataset(dataset_info, grid_size, n_images, noise_level, output_folder, seed=42, batch_size=1000):
    """
    Generate simulated cryo-EM dataset from CryoBench volumes.
    Also estimates mean from data (used for all downstream analysis).
    """
    import mrcfile

    np.random.seed(seed)
    vol_dir = dataset_info["vol_dir"]

    output.mkdir_safe(output_folder)

    voxel_size = 4.25 * 128 / grid_size

    print(f"Generating dataset from {vol_dir}")
    print(f"  Grid size: {grid_size}, N images: {n_images}, Noise level: {noise_level}")

    fixed_vol_dir = os.path.join(output_folder, "volumes_fixed")
    output.mkdir_safe(fixed_vol_dir)

    mrc_files = sorted(glob.glob(os.path.join(vol_dir, "*.mrc")))
    for i, mrc_file in enumerate(mrc_files):
        fixed_path = os.path.join(fixed_vol_dir, f"{i}.mrc")
        if not os.path.exists(fixed_path):
            with mrcfile.open(mrc_file, permissive=True) as mrc:
                data = mrc.data.copy()
            with mrcfile.new(fixed_path, overwrite=True) as mrc_out:
                mrc_out.set_data(data)
                mrc_out.voxel_size = voxel_size

    print(f"  Fixed {len(mrc_files)} volumes with proper MRC headers")

    simulator.generate_synthetic_dataset(
        output_folder,
        voxel_size,
        fixed_vol_dir + "/",
        n_images,
        outlier_file_input=None,
        grid_size=grid_size,
        volume_distribution=None,
        dataset_params_option="uniform",
        noise_level=noise_level,
        noise_model="white",
        put_extra_particles=False,
        percent_outliers=0.00,
        volume_radius=0.7,
        trailing_zero_format_in_vol_name=False,
        noise_scale_std=0,
        contrast_std=0,
        disc_type="nufft",
        n_tilts=-1,
    )

    output_folder = _with_trailing_separator(output_folder)
    cryos, sim_info, gt_results, noise_variance = _load_simulated_dataset(
        output_folder, grid_size, n_images, lazy=False
    )
    gt_mean = gt_results.get_mean()

    print("  Estimating mean from data...")
    noise_var_image = utils.make_radial_image(noise_variance, cryos.image_shape)
    means, _mean_prior, _fsc = homogeneous.get_mean_conformation_relion(
        cryos, batch_size, noise_variance=noise_var_image, use_regularization=False
    )
    mean_estimate = means.combined.flatten()

    mean_error = float(np.linalg.norm(mean_estimate - gt_mean.flatten()) / np.linalg.norm(gt_mean))
    print(f"  Mean estimation error: {mean_error:.4f}")

    return cryos, mean_estimate, gt_results, sim_info, means, mean_error


def load_dataset_and_warmstart(results_dir, dataset_name, grid_size):
    """Load cryos from simulated_data and PCA warmstart from pca_warmstart.npz (after --setup-only)."""
    npz_path = os.path.join(results_dir, dataset_name, "pca_warmstart.npz")
    data = np.load(npz_path, allow_pickle=True)
    n_images = int(data["n_images"])
    sim_output = os.path.join(results_dir, dataset_name, "simulated_data")
    cryos, _sim_info, gt_results, _noise_variance = _load_simulated_dataset(sim_output, grid_size, n_images, lazy=False)
    W_init = jnp.array(data["W_init"])
    W_prior_base = jnp.array(data["W_prior_base"])
    U_gt = np.array(data["U_gt"])
    s_gt = np.array(data["s_gt"])
    volume_shape = tuple(data["volume_shape"])
    mean_estimate = np.array(data["mean_estimate"])
    pca_results = data["pca_results"].flat[0] if "pca_results" in data and data["pca_results"].size > 0 else {}
    return cryos, mean_estimate, gt_results, W_init, W_prior_base, U_gt, s_gt, volume_shape, pca_results


def run_setup_only(dataset_info, grid_size, n_images, noise_level, n_pcs, seed, results_dir):
    """Generate dataset, run PCA warmstart, save pca_warmstart.npz for single-run jobs."""
    dataset_name = dataset_info["name"]
    sim_output = os.path.join(results_dir, dataset_name, "simulated_data")
    results_output = os.path.join(results_dir, dataset_name)
    output.mkdir_safe(results_output)
    cryos, mean_estimate, gt_results, sim_info, means, mean_error = generate_dataset(
        dataset_info, grid_size, n_images, noise_level, sim_output, seed
    )
    W_init, W_prior_base, U_gt, s_gt, pca_results = warmstart_from_pca(cryos, means, gt_results, n_pcs)
    pca_results["mean_error"] = mean_error
    volume_shape = tuple(gt_results.volume_shape)
    npz_path = os.path.join(results_output, "pca_warmstart.npz")
    np.savez(
        npz_path,
        W_init=np.array(W_init),
        W_prior_base=np.array(W_prior_base),
        U_gt=U_gt,
        s_gt=s_gt,
        volume_shape=np.array(volume_shape),
        mean_estimate=mean_estimate,
        n_pcs=np.array(n_pcs),
        grid_size=np.array(grid_size),
        n_images=np.array(n_images),
        pca_results=np.array([pca_results], dtype=object),
    )
    print(f"  Saved {npz_path}")
    return npz_path


def run_single_scale(dataset_name, grid_size, n_pcs, strategy, scale, n_iter, results_dir, results_output=None):
    """Run one (strategy, scale) for a dataset; expects pca_warmstart.npz to exist."""
    if results_output is None:
        results_output = os.path.join(results_dir, dataset_name)
    cryos, mean_estimate, gt_results, W_init, W_prior_base, U_gt, s_gt, volume_shape, pca_results = (
        load_dataset_and_warmstart(results_dir, dataset_name, grid_size)
    )
    W_init = jnp.array(W_init)
    W_prior_base = jnp.array(W_prior_base)
    U_gt = jnp.array(U_gt)
    s_gt = jnp.array(s_gt)
    strategy_configs = {
        "l2_no_whiten": (run_l2_sweep, False, "cz"),
        "l2_cz": (run_l2_sweep, True, "cz"),
        "l2_proj_ls": (run_l2_sweep, True, "proj_ls"),
        "l1_no_whiten": (run_l1_sweep, False, "cz"),
        "l1_cz": (run_l1_sweep, True, "cz"),
        "l1_proj_ls": (run_l1_sweep, True, "proj_ls"),
    }
    try:
        runner, use_whitening, whitening_mode = strategy_configs[strategy]
    except KeyError:
        raise ValueError(f"Unknown strategy: {strategy}")
    results = runner(
        cryos,
        mean_estimate,
        W_init,
        W_prior_base,
        U_gt,
        s_gt,
        volume_shape,
        [scale],
        n_iter,
        use_whitening=use_whitening,
        whitening_mode=whitening_mode,
    )
    r = results[0]
    # Keep iteration_data so combine-and-plot can draw EM convergence (list of small dicts)
    out = {k: v for k, v in r.items() if k not in ("W", "C_z")}
    out["strategy"] = strategy
    out_path = os.path.join(results_output, f"single_{strategy}_{scale}.npy")
    np.save(out_path, out)
    print(f"  Saved {out_path} rel_var={r['rel_var']:.4f}")
    return out_path


def list_run_dirs(results_dir, dataset_name):
    """Return list of (run_id, path) for dataset_name/run_* sorted by run_id (newest last)."""
    dataset_dir = os.path.join(results_dir, dataset_name)
    if not os.path.isdir(dataset_dir):
        return []
    run_dirs = []
    for name in os.listdir(dataset_dir):
        if name.startswith("run_"):
            run_id = name[4:]
            path = os.path.join(dataset_dir, name)
            if os.path.isdir(path):
                run_dirs.append((run_id, path))

    # Sort by run_id (numeric if possible, else string)
    def key(x):
        try:
            return (0, int(x[0]))
        except ValueError:
            return (1, x[0])

    run_dirs.sort(key=key)
    return run_dirs


def load_single_run_results(results_dir, dataset_name, results_output=None):
    """Load all single_<strategy>_<scale>.npy from results_dir/dataset_name (or run_<id> subdir) and group into six lists."""
    if results_output is None:
        results_output = os.path.join(results_dir, dataset_name)
    pattern = os.path.join(results_output, "single_*.npy")
    files = sorted(glob.glob(pattern))
    groups = {
        "l1_cz": [],
        "l1_proj_ls": [],
        "l1_no_whiten": [],
        "l2_cz": [],
        "l2_proj_ls": [],
        "l2_no_whiten": [],
    }
    for path in files:
        base = os.path.basename(path)
        if not base.startswith("single_") or not base.endswith(".npy"):
            continue
        # single_<strategy>_<scale>.npy  (scale can be int or float, e.g. 0.1 or 1.0)
        rest = base[7:-4]  # strip 'single_' and '.npy'
        parts = rest.rsplit("_", 1)
        if len(parts) != 2:
            continue
        strategy, scale_str = parts
        try:
            scale = float(scale_str)
        except ValueError:
            continue
        if strategy not in groups:
            continue
        data = np.load(path, allow_pickle=True).flat[0]
        data = dict(data)
        data["scale"] = scale
        data["strategy"] = strategy
        groups[strategy].append(data)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda r: r["scale"])
    return (
        groups["l1_cz"],
        groups["l1_proj_ls"],
        groups["l1_no_whiten"],
        groups["l2_cz"],
        groups["l2_proj_ls"],
        groups["l2_no_whiten"],
    )


def combine_and_plot_single_runs(results_dir, dataset_name, plots_dir, n_pcs, results_output=None, run_id=None):
    """Load single_*.npy results and pca_warmstart.npz, then generate whitening + scale diagnostics + EM convergence plots.
    If run_id is set, results_output = results_dir/dataset_name/run_<run_id>; pca_warmstart is read from dataset dir (parent).
    """
    if results_output is None:
        results_output = os.path.join(results_dir, dataset_name)
    if run_id is not None:
        results_output = os.path.join(results_dir, dataset_name, f"run_{run_id}")
        print(f"Using run dir: {results_output}")
    l1_cz, l1_proj_ls, l1_no_whiten, l2_cz, l2_proj_ls, l2_no_whiten = load_single_run_results(
        results_dir, dataset_name, results_output
    )
    n_total = len(l1_cz) + len(l1_proj_ls) + len(l1_no_whiten) + len(l2_cz) + len(l2_proj_ls) + len(l2_no_whiten)
    if n_total == 0:
        print(f"No single_*.npy files found in {results_output}")
        return
    print(f"Loaded {n_total} single-run results for {dataset_name}")
    # pca_warmstart.npz lives in dataset dir (parent of run_*), not inside run_*
    dataset_dir = (
        os.path.dirname(results_output) if os.path.basename(results_output).startswith("run_") else results_output
    )
    npz_path = os.path.join(dataset_dir, "pca_warmstart.npz")
    if not os.path.isfile(npz_path):
        print(f"  Warning: {npz_path} not found; plots will lack PCA baseline.")
        pca_results = {"rel_var": 0.0}
    else:
        data = np.load(npz_path, allow_pickle=True)
        pca_results = (
            data["pca_results"].flat[0] if "pca_results" in data and data["pca_results"].size > 0 else {"rel_var": 0.0}
        )
        if n_pcs is None and "n_pcs" in data:
            n_pcs = int(data["n_pcs"].flat[0])
    if n_pcs is None:
        n_pcs = 10
    os.makedirs(plots_dir, exist_ok=True)
    plot_whitening_comparison(
        l1_cz, l1_proj_ls, l1_no_whiten, l2_cz, l2_proj_ls, l2_no_whiten, dataset_name, pca_results, plots_dir, n_pcs
    )
    plot_scale_diagnostics(l1_cz, l1_proj_ls, l1_no_whiten, l2_cz, l2_proj_ls, l2_no_whiten, dataset_name, plots_dir)
    path = plot_em_convergence_all_methods(
        l1_cz, l1_proj_ls, l1_no_whiten, l2_cz, l2_proj_ls, l2_no_whiten, dataset_name, plots_dir
    )
    if path is None:
        print("  (EM convergence plot skipped: no iteration_data in single-run results)")
    # EM convergence with one line per scale (all scale coeffs) for each strategy
    for results_list, label in [
        (l1_cz, "L1+Cz"),
        (l1_proj_ls, "L1+proj_ls"),
        (l1_no_whiten, "L1 no whiten"),
        (l2_cz, "L2+Cz"),
        (l2_proj_ls, "L2+proj_ls"),
        (l2_no_whiten, "L2 no whiten"),
    ]:
        p = plot_em_convergence_all_scales(results_list, label, dataset_name, plots_dir)
        if p:
            print(f"  Saved EM convergence (all scales): {p}")
    # Single PNG with all plots stacked vertically for easy viewing
    combined_path = make_combined_plot_png(dataset_name, plots_dir)
    if combined_path:
        print(f"  Combined plot: {combined_path}")
    print(f"Plots saved to {plots_dir}")


def make_combined_plot_png(dataset_name, plots_dir):
    """Load dataset_* comparison/diagnostics/convergence PNGs, stack, and save directly with PIL (no matplotlib re-render)."""
    from PIL import Image

    names = ["whitening_comparison", "scale_diagnostics", "em_convergence_all"]
    paths = [os.path.join(plots_dir, f"{dataset_name}_{n}.png") for n in names]
    existing_paths = [p for p in paths if os.path.isfile(p)]
    if not existing_paths:
        return None
    pils = [Image.open(p).convert("RGB") for p in existing_paths]
    target_w = max(p.size[0] for p in pils)
    scaled = []
    for pil in pils:
        w0, h = pil.size
        if w0 == target_w:
            scaled.append(pil)
        else:
            new_h = int(round(h * target_w / w0))
            scaled.append(pil.resize((target_w, new_h), Image.Resampling.LANCZOS))
    out_w, out_h = target_w, sum(p.size[1] for p in scaled)
    out = Image.new("RGB", (out_w, out_h))
    y = 0
    for p in scaled:
        out.paste(p, (0, y))
        y += p.size[1]
    out_path = os.path.join(plots_dir, f"{dataset_name}_combined.png")
    out.save(out_path)
    return out_path


def warmstart_from_pca(cryos, means, gt_results, basis_size, batch_size=100, gpu_memory=40):
    """Get warmstart initialization from covariance-based PCA."""
    import time

    volume_shape = gt_results.volume_shape

    U_gt, s_gt, _ = gt_results.get_vol_svd()
    W_gt = U_gt[:, :basis_size] * s_gt[:basis_size]

    w_gt_averaged = regularization.batch_average_over_shells(jnp.abs(W_gt.T) ** 2, volume_shape, 0)
    W_prior_base = utils.batch_make_radial_image(w_gt_averaged, volume_shape, True).T

    volume_mask = np.ones(volume_shape, dtype=np.float32)
    dilated_volume_mask = volume_mask.copy()
    valid_idx = cryos.get_valid_frequency_indices()

    options = SimpleNamespace(
        keep_intermediate=False,
        ignore_zero_frequency=False,
        contrast="none",
    )

    covariance_options = covariance_estimation.get_default_covariance_computation_options(cryos.grid_size)
    covariance_options["n_pcs_to_compute"] = basis_size + 5
    covariance_options["column_sampling_scheme"] = "high_snr"
    covariance_options["col"] = "high_snr"
    covariance_options["sampling_n_cols"] = 100
    covariance_options["randomized_sketch_size"] = 100

    print("  Running covariance-based PCA (estimate_principal_components)...")
    pca_start = time.time()
    u, s, _, _, _ = principal_components.estimate_principal_components(
        cryos,
        options,
        means,
        means.prior,
        volume_mask,
        dilated_volume_mask,
        valid_idx,
        batch_size,
        gpu_memory_to_use=gpu_memory / 4,
        covariance_options=covariance_options,
        variance_estimate=None,
    )
    pca_time = time.time() - pca_start
    print(f"  PCA completed in {pca_time:.1f}s")

    U_est = u["rescaled"][:, :basis_size]
    s_est = s["rescaled"][:basis_size]

    W_init = U_est * np.sqrt(np.maximum(s_est, 0))
    W_init = jnp.array(W_init)

    _, rel_var_pca, _ = metrics.get_all_variance_scores(U_est, U_gt[:, :basis_size], s_gt[:basis_size] ** 2)

    pca_results = {
        "rel_var": float(np.mean(rel_var_pca)),
        "rel_var_per_pc": list(rel_var_pca),
        "eigenvalues_est": list(s_est),
        "eigenvalues_gt": list(s_gt[:basis_size] ** 2),
        "pca_time": pca_time,
        "W_norm": float(np.linalg.norm(W_init)),
        "W_init": np.array(W_init),
    }

    print(f"  PCA RelVar: {pca_results['rel_var']:.4f}")

    return W_init, W_prior_base, U_gt[:, :basis_size], s_gt[:basis_size], pca_results


def compute_sparsity(W, volume_shape):
    """Compute wavelet sparsity of loading matrix W."""
    import jaxwt

    sparsity_scores = []
    for k in range(W.shape[-1]):
        vol_real = (ftu.get_idft3(W[:, k].reshape(volume_shape)) * np.sqrt(np.prod(volume_shape))).real
        coeffs_list = jaxwt.wavedec3(vol_real[None], wavelet="db1", mode="symmetric", axes=(-3, -2, -1))
        all_coeffs = [coeffs_list[0].flatten()]
        for level_dict in coeffs_list[1:]:
            for key in level_dict:
                all_coeffs.append(level_dict[key].flatten())
        coeffs = np.concatenate(all_coeffs)
        threshold = 0.01 * np.max(np.abs(coeffs))
        sparsity = np.mean(np.abs(coeffs) < threshold)
        sparsity_scores.append(sparsity)
    return np.mean(sparsity_scores)


def _run_em_for_scale(
    cryos,
    mean_estimate,
    W_init,
    W_prior,
    U_gt,
    s_gt,
    n_iter,
    *,
    use_whitening,
    whitening_mode,
    sparse_pca,
    l1_sigma=None,
):
    return ppca.EM(
        cryos,
        mean_estimate,
        W_init.copy(),
        W_prior,
        U_gt=U_gt,
        S_gt=s_gt**2,
        EM_iter=n_iter,
        use_whitening=use_whitening,
        whitening_mode=whitening_mode,
        sparse_PCA=sparse_pca,
        l1_sigma=l1_sigma,
        disc_type_mean="cubic",
        disc_type="linear_interp",
        return_iteration_data=True,
    )


def _summarize_scale_result(scale, em_output, U_gt, s_gt, volume_shape):
    u, _s, W, ez, sm, iteration_data = em_output
    C_z = np.mean(sm, axis=0)  # E[zz^T] = mean of second moments

    # ppca.EM now returns u in its natural half-Fourier shape (half_vol, q).
    # U_gt is full-Fourier. Convert u to full-Fourier for the variance score.
    from recovar.core import fourier_transform_utils as _ftu
    u_arr = np.asarray(u)
    half_vol_size = int(np.prod(_ftu.volume_shape_to_half_volume_shape(volume_shape)))
    if u_arr.shape[0] == half_vol_size:
        u = _ftu.half_volume_to_full_volume(u_arr.T, volume_shape).T

    _, rel_var, _ = metrics.get_all_variance_scores(u, U_gt, s_gt**2)
    rv = float(np.mean(rel_var))
    W_norm = float(jnp.linalg.norm(W))
    z_var = float(np.var(ez)) if ez is not None else 0.0
    avg_sparsity = compute_sparsity(W, volume_shape)
    trace_Cz = float(np.trace(C_z))
    constraint_violation = float(np.linalg.norm(C_z - np.eye(C_z.shape[0])))
    return {
        "scale": scale,
        "rel_var": rv,
        "rel_var_per_pc": list(rel_var),
        "W_norm": W_norm,
        "z_var": z_var,
        "sparsity": avg_sparsity,
        "W": np.array(W),
        "trace_Cz": trace_Cz,
        "constraint_violation": constraint_violation,
        "C_z": C_z,
        "iteration_data": iteration_data,
    }


def run_l1_sweep(
    cryos,
    mean_estimate,
    W_init,
    W_prior_base,
    U_gt,
    s_gt,
    volume_shape,
    scales,
    n_iter=20,
    use_whitening=True,
    whitening_mode="cz",
):
    """Run L1 PPCA with scale parameter sweep."""
    results = []
    for scale in scales:
        print(f"\n  L1 Scale = {scale:.6f}")
        sigma = compute_level_sigma(W_init, volume_shape, scale, mode="avg_all")
        out = _run_em_for_scale(
            cryos,
            mean_estimate,
            W_init,
            W_prior_base,
            U_gt,
            s_gt,
            n_iter,
            use_whitening=use_whitening,
            whitening_mode=whitening_mode,
            sparse_pca=True,
            l1_sigma=sigma,
        )
        result = _summarize_scale_result(scale, out, U_gt, s_gt, volume_shape)
        results.append(result)
        print(
            f"    RelVar: {result['rel_var']:.4f}, Sparsity: {result['sparsity']:.2%}, W_norm: {result['W_norm']:.2f}"
        )
    return results


def run_l2_sweep(
    cryos,
    mean_estimate,
    W_init,
    W_prior_base,
    U_gt,
    s_gt,
    volume_shape,
    scales,
    n_iter=20,
    use_whitening=True,
    whitening_mode="cz",
):
    """Run L2 PPCA with scale parameter sweep."""
    results = []
    for scale in scales:
        print(f"\n  L2 Scale = {scale:.6f}")
        W_prior = scale * W_prior_base
        out = _run_em_for_scale(
            cryos,
            mean_estimate,
            W_init,
            W_prior,
            U_gt,
            s_gt,
            n_iter,
            use_whitening=use_whitening,
            whitening_mode=whitening_mode,
            sparse_pca=False,
        )
        result = _summarize_scale_result(scale, out, U_gt, s_gt, volume_shape)
        results.append(result)
        print(
            f"    RelVar: {result['rel_var']:.4f}, Sparsity: {result['sparsity']:.2%}, W_norm: {result['W_norm']:.2f}"
        )
    return results


def plot_sweep_results(l1_whiten, l1_no_whiten, l2_whiten, l2_no_whiten, dataset_name, pca_results, plots_dir, n_pcs):
    """Create comprehensive comparison plots for L1 vs L2 with/without whitening. Saves to plots_dir."""
    pca_baseline = pca_results["rel_var"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{dataset_name}: L1 vs L2 PPCA Scale Sweep (With/Without Whitening)\n({n_pcs} PCs, warmstarted from Cov-PCA)",
        fontsize=14,
    )

    l1w_scales = [r["scale"] for r in l1_whiten]
    l1w_relvars = [r["rel_var"] for r in l1_whiten]
    l1n_relvars = [r["rel_var"] for r in l1_no_whiten]
    l2w_scales = [r["scale"] for r in l2_whiten]
    l2w_relvars = [r["rel_var"] for r in l2_whiten]
    l2n_relvars = [r["rel_var"] for r in l2_no_whiten]

    ax = axes[0, 0]
    ax.semilogx(l1w_scales, l1w_relvars, "bo-", linewidth=2, markersize=8, label="L1 + whiten")
    ax.semilogx(l1w_scales, l1n_relvars, "b^--", linewidth=2, markersize=6, label="L1 no whiten")
    ax.axhline(y=pca_baseline, color="r", linestyle="--", linewidth=2, label=f"Cov-PCA ({pca_baseline:.3f})")
    best_idx = np.argmax(l1w_relvars)
    ax.plot(l1w_scales[best_idx], l1w_relvars[best_idx], "r*", markersize=15)
    ax.set_xlabel("L1 Scale (σ multiplier)", fontsize=12)
    ax.set_ylabel("Relative Variance", fontsize=12)
    ax.set_title(f"L1 PPCA: Best whiten={l1w_relvars[best_idx]:.3f} @ scale={l1w_scales[best_idx]:.4f}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, min(1.1, max(l1w_relvars + l1n_relvars + [pca_baseline]) * 1.2)])

    ax = axes[0, 1]
    ax.semilogx(l2w_scales, l2w_relvars, "go-", linewidth=2, markersize=8, label="L2 + whiten")
    ax.semilogx(l2w_scales, l2n_relvars, "g^--", linewidth=2, markersize=6, label="L2 no whiten")
    ax.axhline(y=pca_baseline, color="r", linestyle="--", linewidth=2, label=f"Cov-PCA ({pca_baseline:.3f})")
    best_idx_l2 = np.argmax(l2w_relvars)
    ax.plot(l2w_scales[best_idx_l2], l2w_relvars[best_idx_l2], "r*", markersize=15)
    ax.set_xlabel("L2 Scale (W_prior multiplier)", fontsize=12)
    ax.set_ylabel("Relative Variance", fontsize=12)
    ax.set_title(f"L2 PPCA: Best whiten={l2w_relvars[best_idx_l2]:.3f} @ scale={l2w_scales[best_idx_l2]:.4f}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, min(1.1, max(l2w_relvars + l2n_relvars + [pca_baseline]) * 1.2)])

    ax = axes[0, 2]
    ax.semilogx(l1w_scales, l1w_relvars, "bo-", linewidth=2, markersize=6, label="L1 + whiten")
    ax.semilogx(l1w_scales, l1n_relvars, "b^--", linewidth=1.5, markersize=5, label="L1 no whiten")
    ax.semilogx(l2w_scales, l2w_relvars, "gs-", linewidth=2, markersize=6, label="L2 + whiten")
    ax.semilogx(l2w_scales, l2n_relvars, "g^--", linewidth=1.5, markersize=5, label="L2 no whiten")
    ax.axhline(y=pca_baseline, color="r", linestyle="--", linewidth=2, label="Cov-PCA")
    ax.set_xlabel("Scale Parameter", fontsize=12)
    ax.set_ylabel("Relative Variance", fontsize=12)
    ax.set_title("L1 vs L2: Whitening vs No Whitening")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogx(
        l1w_scales,
        [s * 100 for s in [r["sparsity"] for r in l1_whiten]],
        "bo-",
        linewidth=2,
        markersize=6,
        label="L1 + whiten",
    )
    ax.semilogx(
        l1w_scales,
        [s * 100 for s in [r["sparsity"] for r in l1_no_whiten]],
        "b^--",
        linewidth=1.5,
        label="L1 no whiten",
    )
    ax.semilogx(
        l2w_scales,
        [s * 100 for s in [r["sparsity"] for r in l2_whiten]],
        "gs-",
        linewidth=2,
        markersize=6,
        label="L2 + whiten",
    )
    ax.semilogx(
        l2w_scales,
        [s * 100 for s in [r["sparsity"] for r in l2_no_whiten]],
        "g^--",
        linewidth=1.5,
        label="L2 no whiten",
    )
    ax.set_xlabel("Scale Parameter", fontsize=12)
    ax.set_ylabel("Sparsity (% near-zero)", fontsize=12)
    ax.set_title("Wavelet Coefficient Sparsity")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogx(l1w_scales, [r["W_norm"] for r in l1_whiten], "bo-", linewidth=2, markersize=6, label="L1 + whiten")
    ax.semilogx(l1w_scales, [r["W_norm"] for r in l1_no_whiten], "b^--", linewidth=1.5, label="L1 no whiten")
    ax.semilogx(l2w_scales, [r["W_norm"] for r in l2_whiten], "gs-", linewidth=2, markersize=6, label="L2 + whiten")
    ax.semilogx(l2w_scales, [r["W_norm"] for r in l2_no_whiten], "g^--", linewidth=1.5, label="L2 no whiten")
    ax.axhline(y=pca_results["W_norm"], color="r", linestyle="--", linewidth=2, label="PCA init")
    ax.set_xlabel("Scale Parameter", fontsize=12)
    ax.set_ylabel("||W|| (Frobenius norm)", fontsize=12)
    ax.set_title("Loading Matrix Norm")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    pca_per_pc = pca_results["rel_var_per_pc"]
    l1_best = l1_whiten[best_idx]
    l2_best = l2_whiten[best_idx_l2]
    pc_indices = np.arange(1, len(pca_per_pc) + 1)
    width = 0.25
    ax.bar(pc_indices - width, pca_per_pc, width, label="Cov-PCA", color="salmon", edgecolor="black")
    ax.bar(
        pc_indices,
        l1_best["rel_var_per_pc"],
        width,
        label=f"L1+whiten (s={l1w_scales[best_idx]:.4f})",
        color="steelblue",
        edgecolor="black",
    )
    ax.bar(
        pc_indices + width,
        l2_best["rel_var_per_pc"],
        width,
        label=f"L2+whiten (s={l2w_scales[best_idx_l2]:.4f})",
        color="lightgreen",
        edgecolor="black",
    )
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Relative Variance", fontsize=12)
    ax.set_title("Per-PC: Cov-PCA vs L1 vs L2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.1])
    ax.set_xticks(pc_indices)

    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{dataset_name}_l1_l2_sweep.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {plot_path}")
    return plot_path


def plot_whitening_comparison(
    l1_cz, l1_proj_ls, l1_no_whiten, l2_cz, l2_proj_ls, l2_no_whiten, dataset_name, pca_results, plots_dir, n_pcs
):
    """Plot cz vs proj_ls vs no whitening for L1 and L2 scale sweeps. Saves to plots_dir."""
    pca_baseline = pca_results["rel_var"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{dataset_name}: Whitening Comparison (Cz vs proj_ls vs no whiten)\n({n_pcs} PCs)", fontsize=14)

    scales_l1 = [r["scale"] for r in l1_cz]
    ax = axes[0, 0]
    ax.semilogx(scales_l1, [r["rel_var"] for r in l1_cz], "b-o", linewidth=2, markersize=6, label="L1 + Cz")
    ax.semilogx(scales_l1, [r["rel_var"] for r in l1_proj_ls], "g-s", linewidth=2, markersize=6, label="L1 + proj_ls")
    ax.semilogx(
        scales_l1, [r["rel_var"] for r in l1_no_whiten], "k^--", linewidth=1.5, markersize=5, label="L1 no whiten"
    )
    ax.axhline(y=pca_baseline, color="r", linestyle=":", linewidth=1.5, label=f"Cov-PCA ({pca_baseline:.3f})")
    ax.set_xlabel("L1 Scale", fontsize=12)
    ax.set_ylabel("Relative Variance", fontsize=12)
    ax.set_title("L1 PPCA: Cz vs proj_ls vs no whitening")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    all_rv = [r["rel_var"] for r in l1_cz + l1_proj_ls + l1_no_whiten] + [pca_baseline]
    ax.set_ylim([0, min(1.1, max(all_rv) * 1.15)])

    scales_l2 = [r["scale"] for r in l2_cz]
    ax = axes[0, 1]
    ax.semilogx(scales_l2, [r["rel_var"] for r in l2_cz], "b-o", linewidth=2, markersize=6, label="L2 + Cz")
    ax.semilogx(scales_l2, [r["rel_var"] for r in l2_proj_ls], "g-s", linewidth=2, markersize=6, label="L2 + proj_ls")
    ax.semilogx(
        scales_l2, [r["rel_var"] for r in l2_no_whiten], "k^--", linewidth=1.5, markersize=5, label="L2 no whiten"
    )
    ax.axhline(y=pca_baseline, color="r", linestyle=":", linewidth=1.5, label=f"Cov-PCA ({pca_baseline:.3f})")
    ax.set_xlabel("L2 Scale", fontsize=12)
    ax.set_ylabel("Relative Variance", fontsize=12)
    ax.set_title("L2 PPCA: Cz vs proj_ls vs no whitening")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    all_rv2 = [r["rel_var"] for r in l2_cz + l2_proj_ls + l2_no_whiten] + [pca_baseline]
    ax.set_ylim([0, min(1.1, max(all_rv2) * 1.15)])

    # Best RelVar bar comparison
    ax = axes[1, 0]
    best_l1_cz = max(l1_cz, key=lambda r: r["rel_var"])["rel_var"]
    best_l1_ls = max(l1_proj_ls, key=lambda r: r["rel_var"])["rel_var"]
    best_l1_nw = max(l1_no_whiten, key=lambda r: r["rel_var"])["rel_var"]
    best_l2_cz = max(l2_cz, key=lambda r: r["rel_var"])["rel_var"]
    best_l2_ls = max(l2_proj_ls, key=lambda r: r["rel_var"])["rel_var"]
    best_l2_nw = max(l2_no_whiten, key=lambda r: r["rel_var"])["rel_var"]
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, [best_l1_cz, best_l1_ls, best_l1_nw], width, label="L1", color="steelblue", edgecolor="black")
    ax.bar(x + width / 2, [best_l2_cz, best_l2_ls, best_l2_nw], width, label="L2", color="seagreen", edgecolor="black")
    ax.axhline(y=pca_baseline, color="r", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Cz whiten", "proj_ls whiten", "no whiten"])
    ax.set_ylabel("Best RelVar", fontsize=12)
    ax.set_title("Best result per strategy")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.05])

    # Sparsity: L1 only (cz vs proj_ls vs no whiten)
    ax = axes[1, 1]
    ax.semilogx(scales_l1, [r["sparsity"] * 100 for r in l1_cz], "b-o", linewidth=2, markersize=5, label="L1 + Cz")
    ax.semilogx(
        scales_l1, [r["sparsity"] * 100 for r in l1_proj_ls], "g-s", linewidth=2, markersize=5, label="L1 + proj_ls"
    )
    ax.semilogx(scales_l1, [r["sparsity"] * 100 for r in l1_no_whiten], "k^--", linewidth=1.5, label="L1 no whiten")
    ax.set_xlabel("L1 Scale", fontsize=12)
    ax.set_ylabel("Sparsity (% near-zero)", fontsize=12)
    ax.set_title("L1 wavelet sparsity by strategy")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{dataset_name}_whitening_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved whitening comparison: {plot_path}")
    return plot_path


def plot_em_convergence_all_scales(results_list, strategy_label, dataset_name, plots_dir):
    """Plot EM convergence (Neg_LL, RelVar, ||W||, tr(Ĉ_z), tr(E[μμ^T])) with one line per scale value."""
    entries = [(r["scale"], r.get("iteration_data", [])) for r in sorted(results_list, key=lambda x: x["scale"])]
    entries = [(scale, idat) for scale, idat in entries if idat]
    if not entries:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"{dataset_name}: EM Convergence — {strategy_label} (all scales)", fontsize=12)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(entries)))
    for (scale, idat), color in zip(entries, colors):
        iters = [d["Iteration"] for d in idat]
        axes[0, 0].plot(iters, [d["Neg_LL_Total"] for d in idat], color=color, label=f"{scale}", markersize=2)
        relvars = [d.get("Rel_Var_Explained") for d in idat]
        if any(v is not None for v in relvars):
            axes[0, 1].plot(
                iters, [v if v is not None else 0 for v in relvars], color=color, label=f"{scale}", markersize=2
            )
        axes[0, 2].plot(iters, [d.get("W_norm", 0) for d in idat], color=color, label=f"{scale}", markersize=2)
        axes[1, 0].plot(iters, [d.get("trace_Cz", 0) for d in idat], color=color, label=f"{scale}", markersize=2)
        axes[1, 1].plot(
            iters, [d.get("trace_E_mean_outer", 0) for d in idat], color=color, label=f"{scale}", markersize=2
        )
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Neg LL Total")
    axes[0, 0].set_title("Negative Log-Likelihood")
    axes[0, 0].legend(loc="best", fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("RelVar")
    axes[0, 1].set_title("Relative Variance Explained")
    axes[0, 1].legend(loc="best", fontsize=7)
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("||W||")
    axes[0, 2].set_title("Loading Matrix Norm")
    axes[0, 2].legend(loc="best", fontsize=7)
    axes[0, 2].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("tr(Ĉ_z)")
    axes[1, 0].set_title("Latent Covariance Trace")
    axes[1, 0].legend(loc="best", fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("tr(E[μμ^T])")
    axes[1, 1].set_title("E[μμ^T] from expected_zs")
    axes[1, 1].legend(loc="best", fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 2].axis("off")
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    safe_label = strategy_label.replace(" ", "_").replace("+", "_")
    path = os.path.join(plots_dir, f"{dataset_name}_em_convergence_{safe_label}_all_scales.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_em_convergence_all_methods(
    l1_cz, l1_proj_ls, l1_no_whiten, l2_cz, l2_proj_ls, l2_no_whiten, dataset_name, plots_dir
):
    """Plot EM convergence (Neg_LL, RelVar, W_norm, trace_Cz, constraint_violation) for best run of each method on one figure."""
    methods = [
        (max(l1_cz, key=lambda r: r["rel_var"]), "L1+Cz", "b-o"),
        (max(l1_proj_ls, key=lambda r: r["rel_var"]), "L1+proj_ls", "g-s"),
        (max(l1_no_whiten, key=lambda r: r["rel_var"]), "L1 no whiten", "k^--"),
        (max(l2_cz, key=lambda r: r["rel_var"]), "L2+Cz", "c-d"),
        (max(l2_proj_ls, key=lambda r: r["rel_var"]), "L2+proj_ls", "m-p"),
        (max(l2_no_whiten, key=lambda r: r["rel_var"]), "L2 no whiten", "y*:"),
    ]
    data_list = [(r.get("iteration_data", []), label, style) for r, label, style in methods]
    data_list = [(idat, label, style) for idat, label, style in data_list if idat]
    if not data_list:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"{dataset_name}: EM Convergence (all methods, best scale each)", fontsize=12)
    for idat, label, style in data_list:
        iters = [d["Iteration"] for d in idat]
        ax = axes[0, 0]
        ax.plot(iters, [d["Neg_LL_Total"] for d in idat], style, markersize=3, label=label)
        ax = axes[0, 1]
        relvars = [d.get("Rel_Var_Explained") for d in idat]
        if any(v is not None for v in relvars):
            ax.plot(iters, [v if v is not None else 0 for v in relvars], style, markersize=3, label=label)
        ax = axes[0, 2]
        ax.plot(iters, [d.get("W_norm", 0) for d in idat], style, markersize=3, label=label)
        ax = axes[1, 0]
        ax.plot(iters, [d.get("trace_Cz", 0) for d in idat], style, markersize=3, label=label)
        ax = axes[1, 1]
        ax.plot(iters, [d.get("trace_E_mean_outer", 0) for d in idat], style, markersize=3, label=label)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Neg LL Total")
    axes[0, 0].set_title("Negative Log-Likelihood")
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("RelVar")
    axes[0, 1].set_title("Relative Variance Explained")
    axes[0, 1].legend(loc="best", fontsize=8)
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("||W||")
    axes[0, 2].set_title("Loading Matrix Norm")
    axes[0, 2].legend(loc="best", fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("tr(Ĉ_z)")
    axes[1, 0].set_title("Latent Covariance Trace")
    axes[1, 0].legend(loc="best", fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("tr(E[μμ^T])")
    axes[1, 1].set_title("E[μμ^T] from expected_zs")
    axes[1, 1].legend(loc="best", fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 2].axis("off")
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{dataset_name}_em_convergence_all.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved EM convergence (all methods): {path}")
    return path


def plot_em_convergence(iteration_data, title, plots_dir, prefix):
    """Plot Neg_LL, RelVar, W_norm, trace_Cz, constraint_violation vs iteration (single method)."""
    if not iteration_data:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title, fontsize=12)
    iters = [d["Iteration"] for d in iteration_data]
    ax = axes[0, 0]
    ax.plot(iters, [d["Neg_LL_Total"] for d in iteration_data], "b-o", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Neg LL Total")
    ax.set_title("EM Convergence: Negative Log-Likelihood")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    relvars = [d.get("Rel_Var_Explained") for d in iteration_data]
    if any(v is not None for v in relvars):
        ax.plot(iters, [v if v is not None else 0 for v in relvars], "r-o", markersize=4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RelVar")
        ax.set_title("Relative Variance Explained vs EM Iteration")
        ax.set_ylim([0, 1.05])
    else:
        ax.text(0.5, 0.5, "No ground truth", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(iters, [d.get("W_norm", 0) for d in iteration_data], "g-o", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||W||")
    ax.set_title("Loading Matrix Norm")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(iters, [d.get("trace_Cz", 0) for d in iteration_data], "m-o", markersize=4, label="tr(Ĉ_z)")
    ax.plot(iters, [d.get("constraint_violation", 0) for d in iteration_data], "c-s", markersize=4, label="||Ĉ_z - I||")
    ax.plot(
        iters,
        [d.get("trace_E_mean_outer", 0) for d in iteration_data],
        "orange",
        marker="o",
        markersize=4,
        label="tr(E[μμ^T])",
    )
    ax.plot(
        iters,
        [d.get("norm_E_mean_outer_minus_I", 0) for d in iteration_data],
        "brown",
        marker="s",
        markersize=4,
        label="||E[μμ^T]-I||",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title("Latent: Ĉ_z and E[μμ^T] from expected_zs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, f"{prefix}_em_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_scale_diagnostics(l1_cz, l1_proj_ls, l1_no_whiten, l2_cz, l2_proj_ls, l2_no_whiten, dataset_name, plots_dir):
    """Plot ||W||, tr(Ĉ_z), ||Ĉ_z - I|| vs scale for each strategy."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"{dataset_name}: Scale Diagnostics (||W||, tr(Ĉ_z), ||Ĉ_z-I||)", fontsize=14)
    scales_l1 = [r["scale"] for r in l1_cz]
    scales_l2 = [r["scale"] for r in l2_cz]

    # Row 0: L1 diagnostics
    ax = axes[0, 0]
    ax.semilogx(scales_l1, [r["W_norm"] for r in l1_cz], "b-o", linewidth=2, markersize=5, label="Cz")
    ax.semilogx(scales_l1, [r["W_norm"] for r in l1_proj_ls], "g-s", linewidth=2, markersize=5, label="proj_ls")
    ax.semilogx(scales_l1, [r["W_norm"] for r in l1_no_whiten], "k^--", linewidth=1.5, label="no whiten")
    ax.set_xlabel("L1 Scale")
    ax.set_ylabel("||W||")
    ax.set_title("L1: Loading Norm vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogx(scales_l1, [r.get("trace_Cz", 0) for r in l1_cz], "b-o", linewidth=2, markersize=5, label="Cz")
    ax.semilogx(
        scales_l1, [r.get("trace_Cz", 0) for r in l1_proj_ls], "g-s", linewidth=2, markersize=5, label="proj_ls"
    )
    ax.semilogx(scales_l1, [r.get("trace_Cz", 0) for r in l1_no_whiten], "k^--", linewidth=1.5, label="no whiten")
    ax.set_xlabel("L1 Scale")
    ax.set_ylabel("tr(Ĉ_z)")
    ax.set_title("L1: Latent Covariance Trace vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.semilogx(
        scales_l1, [r.get("constraint_violation", 0) for r in l1_cz], "b-o", linewidth=2, markersize=5, label="Cz"
    )
    ax.semilogx(
        scales_l1,
        [r.get("constraint_violation", 0) for r in l1_proj_ls],
        "g-s",
        linewidth=2,
        markersize=5,
        label="proj_ls",
    )
    ax.semilogx(
        scales_l1, [r.get("constraint_violation", 0) for r in l1_no_whiten], "k^--", linewidth=1.5, label="no whiten"
    )
    ax.set_xlabel("L1 Scale")
    ax.set_ylabel("||Ĉ_z - I||")
    ax.set_title("L1: Constraint Violation vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 1: L2 diagnostics
    ax = axes[1, 0]
    ax.semilogx(scales_l2, [r["W_norm"] for r in l2_cz], "b-o", linewidth=2, markersize=5, label="Cz")
    ax.semilogx(scales_l2, [r["W_norm"] for r in l2_proj_ls], "g-s", linewidth=2, markersize=5, label="proj_ls")
    ax.semilogx(scales_l2, [r["W_norm"] for r in l2_no_whiten], "k^--", linewidth=1.5, label="no whiten")
    ax.set_xlabel("L2 Scale")
    ax.set_ylabel("||W||")
    ax.set_title("L2: Loading Norm vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogx(scales_l2, [r.get("trace_Cz", 0) for r in l2_cz], "b-o", linewidth=2, markersize=5, label="Cz")
    ax.semilogx(
        scales_l2, [r.get("trace_Cz", 0) for r in l2_proj_ls], "g-s", linewidth=2, markersize=5, label="proj_ls"
    )
    ax.semilogx(scales_l2, [r.get("trace_Cz", 0) for r in l2_no_whiten], "k^--", linewidth=1.5, label="no whiten")
    ax.set_xlabel("L2 Scale")
    ax.set_ylabel("tr(Ĉ_z)")
    ax.set_title("L2: Latent Covariance Trace vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.semilogx(
        scales_l2, [r.get("constraint_violation", 0) for r in l2_cz], "b-o", linewidth=2, markersize=5, label="Cz"
    )
    ax.semilogx(
        scales_l2,
        [r.get("constraint_violation", 0) for r in l2_proj_ls],
        "g-s",
        linewidth=2,
        markersize=5,
        label="proj_ls",
    )
    ax.semilogx(
        scales_l2, [r.get("constraint_violation", 0) for r in l2_no_whiten], "k^--", linewidth=1.5, label="no whiten"
    )
    ax.set_xlabel("L2 Scale")
    ax.set_ylabel("||Ĉ_z - I||")
    ax.set_title("L2: Constraint Violation vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, f"{dataset_name}_scale_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scale diagnostics: {path}")
    return path


def plot_scale_diagnostics_simple(l1_whiten, l1_no_whiten, l2_whiten, l2_no_whiten, dataset_name, plots_dir):
    """Plot ||W||, tr(Ĉ_z), ||Ĉ_z-I|| vs scale for whiten vs no whiten (2 strategies)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"{dataset_name}: Scale Diagnostics (||W||, tr(Ĉ_z), ||Ĉ_z-I||)", fontsize=14)
    scales_l1 = [r["scale"] for r in l1_whiten]
    scales_l2 = [r["scale"] for r in l2_whiten]

    ax = axes[0, 0]
    ax.semilogx(scales_l1, [r["W_norm"] for r in l1_whiten], "b-o", linewidth=2, markersize=5, label="whiten")
    ax.semilogx(scales_l1, [r["W_norm"] for r in l1_no_whiten], "k^--", linewidth=1.5, label="no whiten")
    ax.set_xlabel("L1 Scale")
    ax.set_ylabel("||W||")
    ax.set_title("L1: Loading Norm vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogx(scales_l1, [r.get("trace_Cz", 0) for r in l1_whiten], "b-o", linewidth=2, markersize=5, label="whiten")
    ax.semilogx(scales_l1, [r.get("trace_Cz", 0) for r in l1_no_whiten], "k^--", linewidth=1.5, label="no whiten")
    ax.set_xlabel("L1 Scale")
    ax.set_ylabel("tr(Ĉ_z)")
    ax.set_title("L1: Latent Covariance Trace vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.semilogx(
        scales_l1,
        [r.get("constraint_violation", 0) for r in l1_whiten],
        "b-o",
        linewidth=2,
        markersize=5,
        label="whiten",
    )
    ax.semilogx(
        scales_l1, [r.get("constraint_violation", 0) for r in l1_no_whiten], "k^--", linewidth=1.5, label="no whiten"
    )
    ax.set_xlabel("L1 Scale")
    ax.set_ylabel("||Ĉ_z - I||")
    ax.set_title("L1: Constraint Violation vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogx(scales_l2, [r["W_norm"] for r in l2_whiten], "g-s", linewidth=2, markersize=5, label="whiten")
    ax.semilogx(scales_l2, [r["W_norm"] for r in l2_no_whiten], "k^--", linewidth=1.5, label="no whiten")
    ax.set_xlabel("L2 Scale")
    ax.set_ylabel("||W||")
    ax.set_title("L2: Loading Norm vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogx(scales_l2, [r.get("trace_Cz", 0) for r in l2_whiten], "g-s", linewidth=2, markersize=5, label="whiten")
    ax.semilogx(scales_l2, [r.get("trace_Cz", 0) for r in l2_no_whiten], "k^--", linewidth=1.5, label="no whiten")
    ax.set_xlabel("L2 Scale")
    ax.set_ylabel("tr(Ĉ_z)")
    ax.set_title("L2: Latent Covariance Trace vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.semilogx(
        scales_l2,
        [r.get("constraint_violation", 0) for r in l2_whiten],
        "g-s",
        linewidth=2,
        markersize=5,
        label="whiten",
    )
    ax.semilogx(
        scales_l2, [r.get("constraint_violation", 0) for r in l2_no_whiten], "k^--", linewidth=1.5, label="no whiten"
    )
    ax.set_xlabel("L2 Scale")
    ax.set_ylabel("||Ĉ_z - I||")
    ax.set_title("L2: Constraint Violation vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, f"{dataset_name}_scale_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scale diagnostics: {path}")
    return path


def plot_combined_summary(all_results, plots_dir):
    """Create combined summary plot across all datasets. Saves to plots_dir."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("PPCA Scale Sweep: All Datasets (L1 vs L2, With/Without Whitening)", fontsize=14)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for i, (name, data) in enumerate(all_results.items()):
        scales = [r["scale"] for r in data["l1_whiten"]]
        rel_w = [r["rel_var"] for r in data["l1_whiten"]]
        rel_n = [r["rel_var"] for r in data["l1_no_whiten"]]
        pca_baseline = data["pca_results"]["rel_var"]
        ax = axes[0]
        ax.semilogx(scales, rel_w, "o-", color=colors[i], linewidth=2, markersize=5, label=name)
        ax.semilogx(scales, rel_n, "^--", color=colors[i], linewidth=1, markersize=3, alpha=0.7)
        ax.axhline(y=pca_baseline, color=colors[i], linestyle=":", alpha=0.4)
    axes[0].set_xlabel("L1 Scale", fontsize=12)
    axes[0].set_ylabel("Relative Variance", fontsize=12)
    axes[0].set_title("L1 PPCA (solid=whiten, dashed=no whiten)")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i, (name, data) in enumerate(all_results.items()):
        scales = [r["scale"] for r in data["l2_whiten"]]
        rel_w = [r["rel_var"] for r in data["l2_whiten"]]
        rel_n = [r["rel_var"] for r in data["l2_no_whiten"]]
        pca_baseline = data["pca_results"]["rel_var"]
        ax = axes[1]
        ax.semilogx(scales, rel_w, "s-", color=colors[i], linewidth=2, markersize=5, label=name)
        ax.semilogx(scales, rel_n, "^--", color=colors[i], linewidth=1, markersize=3, alpha=0.7)
        ax.axhline(y=pca_baseline, color=colors[i], linestyle=":", alpha=0.4)
    axes[1].set_xlabel("L2 Scale", fontsize=12)
    axes[1].set_ylabel("Relative Variance", fontsize=12)
    axes[1].set_title("L2 PPCA (solid=whiten, dashed=no whiten)")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    names = list(all_results.keys())
    x = np.arange(len(names))
    width = 0.2
    pca_vals = [data["pca_results"]["rel_var"] for data in all_results.values()]
    l1_vals = [max(r["rel_var"] for r in data["l1_whiten"]) for data in all_results.values()]
    l1n_vals = [max(r["rel_var"] for r in data["l1_no_whiten"]) for data in all_results.values()]
    l2_vals = [max(r["rel_var"] for r in data["l2_whiten"]) for data in all_results.values()]
    l2n_vals = [max(r["rel_var"] for r in data["l2_no_whiten"]) for data in all_results.values()]
    ax = axes[2]
    ax.bar(x - 1.5 * width, pca_vals, width, label="Cov-PCA", color="salmon", edgecolor="black")
    ax.bar(x - 0.5 * width, l1_vals, width, label="L1+whiten", color="steelblue", edgecolor="black")
    ax.bar(x, l1n_vals, width, label="L1 no whiten", color="cornflowerblue", edgecolor="black")
    ax.bar(x + 0.5 * width, l2_vals, width, label="L2+whiten", color="lightgreen", edgecolor="black")
    ax.bar(x + 1.5 * width, l2n_vals, width, label="L2 no whiten", color="palegreen", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Relative Variance", fontsize=12)
    ax.set_title("Best Results")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.0])

    l1_imp = [(l1 - pca) / pca * 100 for l1, pca in zip(l1_vals, pca_vals)]
    l1n_imp = [(l1 - pca) / pca * 100 for l1, pca in zip(l1n_vals, pca_vals)]
    l2_imp = [(l2 - pca) / pca * 100 for l2, pca in zip(l2_vals, pca_vals)]
    l2n_imp = [(l2 - pca) / pca * 100 for l2, pca in zip(l2n_vals, pca_vals)]
    ax = axes[3]
    ax.bar(x - 1.5 * width, l1_imp, width, label="L1+whiten", color="steelblue", edgecolor="black")
    ax.bar(x - 0.5 * width, l1n_imp, width, label="L1 no whiten", color="cornflowerblue", edgecolor="black")
    ax.bar(x + 0.5 * width, l2_imp, width, label="L2+whiten", color="lightgreen", edgecolor="black")
    ax.bar(x + 1.5 * width, l2n_imp, width, label="L2 no whiten", color="palegreen", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% Improvement over Cov-PCA", fontsize=12)
    ax.set_title("Improvement")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "combined_l1_l2_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved combined summary: {plot_path}")
    return plot_path


def run_benchmark(
    dataset_info,
    grid_size,
    n_images,
    noise_level,
    n_pcs,
    l1_scales,
    l2_scales,
    n_iter,
    seed,
    results_dir,
    plots_dir,
    whitening_mode="cz",
):
    """Run full L1+L2 scale sweep benchmark for one dataset. Results (data, .npy) go to results_dir; plots to plots_dir."""
    dataset_name = dataset_info["name"]
    print(f"\n{'=' * 80}")
    print(f"BENCHMARKING: {dataset_name}")
    print(f"  Volumes: {dataset_info['n_volumes']}, Grid: {grid_size}, N: {n_images}, Noise: {noise_level}")
    print(f"  L1 Scales: {l1_scales}, L2 Scales: {l2_scales}, whitening_mode: {whitening_mode}")
    print(f"{'=' * 80}")

    sim_output = os.path.join(results_dir, dataset_name, "simulated_data")
    results_output = os.path.join(results_dir, dataset_name)
    output.mkdir_safe(results_output)

    print("\n[1/5] Generating simulated dataset...")
    cryos, mean_estimate, gt_results, sim_info, means, mean_error = generate_dataset(
        dataset_info, grid_size, n_images, noise_level, sim_output, seed
    )

    print("\n[2/7] Computing PCA warmstart...")
    W_init, W_prior_base, U_gt, s_gt, pca_results = warmstart_from_pca(cryos, means, gt_results, n_pcs)
    pca_results["mean_error"] = mean_error

    print("\n[3/7] Running L2 PPCA scale sweep (WITH whitening)...")
    l2_whiten = run_l2_sweep(
        cryos,
        mean_estimate,
        W_init,
        W_prior_base,
        U_gt,
        s_gt,
        gt_results.volume_shape,
        l2_scales,
        n_iter,
        use_whitening=True,
        whitening_mode=whitening_mode,
    )
    print("\n[4/7] Running L2 PPCA scale sweep (NO whitening)...")
    l2_no_whiten = run_l2_sweep(
        cryos,
        mean_estimate,
        W_init,
        W_prior_base,
        U_gt,
        s_gt,
        gt_results.volume_shape,
        l2_scales,
        n_iter,
        use_whitening=False,
    )

    W_l2_best_whiten = max(l2_whiten, key=lambda r: r["rel_var"])["W"]
    W_l2_best_no_whiten = max(l2_no_whiten, key=lambda r: r["rel_var"])["W"]
    W_l2_best_whiten = jnp.array(W_l2_best_whiten)
    W_l2_best_no_whiten = jnp.array(W_l2_best_no_whiten)
    print("\n  L1 warm-start: best L2 (whiten) W for L1+whiten, best L2 (no whiten) W for L1 no whiten.")

    print("\n[5/7] Running L1 PPCA scale sweep (WITH whitening, warmstart from L2)...")
    l1_whiten = run_l1_sweep(
        cryos,
        mean_estimate,
        W_l2_best_whiten,
        W_prior_base,
        U_gt,
        s_gt,
        gt_results.volume_shape,
        l1_scales,
        n_iter,
        use_whitening=True,
        whitening_mode=whitening_mode,
    )
    print("\n[6/7] Running L1 PPCA scale sweep (NO whitening, warmstart from L2)...")
    l1_no_whiten = run_l1_sweep(
        cryos,
        mean_estimate,
        W_l2_best_no_whiten,
        W_prior_base,
        U_gt,
        s_gt,
        gt_results.volume_shape,
        l1_scales,
        n_iter,
        use_whitening=False,
    )

    print("\n[7/7] Creating plots...")
    plot_sweep_results(l1_whiten, l1_no_whiten, l2_whiten, l2_no_whiten, dataset_name, pca_results, plots_dir, n_pcs)
    plot_scale_diagnostics_simple(l1_whiten, l1_no_whiten, l2_whiten, l2_no_whiten, dataset_name, plots_dir)
    best_l1 = max(l1_whiten, key=lambda r: r["rel_var"])
    idat = best_l1.get("iteration_data", [])
    if idat:
        path = plot_em_convergence(
            idat,
            f"{dataset_name}: EM Convergence (L1+whiten, scale={best_l1['scale']:.4f})",
            plots_dir,
            f"{dataset_name}_l1_whiten",
        )
        if path:
            print(f"  Saved EM convergence: {path}")

    l1_best_whiten = max(l1_whiten, key=lambda r: r["rel_var"])
    l1_best_no_whiten = max(l1_no_whiten, key=lambda r: r["rel_var"])
    l2_best_whiten = max(l2_whiten, key=lambda r: r["rel_var"])
    l2_best_no_whiten = max(l2_no_whiten, key=lambda r: r["rel_var"])
    volume_shape = tuple(gt_results.volume_shape)

    def strip_W(r_list):
        return [{k: v for k, v in r.items() if k != "W"} for r in r_list]

    summary = {
        "dataset": dataset_name,
        "n_volumes": dataset_info["n_volumes"],
        "grid_size": grid_size,
        "n_images": n_images,
        "noise_level": noise_level,
        "n_pcs": n_pcs,
        "seed": seed,
        "whitening_mode": whitening_mode,
        "pca_results": {k: v for k, v in pca_results.items() if k != "W_init"},
        "l1_whiten": strip_W(l1_whiten),
        "l1_no_whiten": strip_W(l1_no_whiten),
        "l2_whiten": strip_W(l2_whiten),
        "l2_no_whiten": strip_W(l2_no_whiten),
        "best_l1_scale": l1_best_whiten["scale"],
        "best_l1_relvar": l1_best_whiten["rel_var"],
        "best_l2_scale": l2_best_whiten["scale"],
        "best_l2_relvar": l2_best_whiten["rel_var"],
        "W_best_l1": l1_best_whiten["W"],
        "W_best_l2": l2_best_whiten["W"],
        "W_pca": np.array(W_init),
        "volume_shape": volume_shape,
    }

    results_path = os.path.join(results_output, "l1_l2_sweep_results.npy")
    np.save(results_path, summary)
    print(f"  Saved results: {results_path}")

    return {
        "l1_whiten": l1_whiten,
        "l1_no_whiten": l1_no_whiten,
        "l2_whiten": l2_whiten,
        "l2_no_whiten": l2_no_whiten,
        "pca_results": pca_results,
        "summary": summary,
    }


def run_benchmark_compare_whitening(
    dataset_info, grid_size, n_images, noise_level, n_pcs, l1_scales, l2_scales, n_iter, seed, results_dir, plots_dir
):
    """Run L1+L2 scale sweep with BOTH cz and proj_ls whitening, plus no whitening. Results to results_dir; plots to plots_dir."""
    dataset_name = dataset_info["name"]
    print(f"\n{'=' * 80}")
    print(f"BENCHMARKING (compare whitening): {dataset_name}")
    print("  Strategies: Cz, proj_ls, no whitening")
    print(f"{'=' * 80}")

    sim_output = os.path.join(results_dir, dataset_name, "simulated_data")
    results_output = os.path.join(results_dir, dataset_name)
    output.mkdir_safe(results_output)

    print("\n[1/8] Generating simulated dataset...")
    cryos, mean_estimate, gt_results, sim_info, means, mean_error = generate_dataset(
        dataset_info, grid_size, n_images, noise_level, sim_output, seed
    )

    print("\n[2/8] Computing PCA warmstart...")
    W_init, W_prior_base, U_gt, s_gt, pca_results = warmstart_from_pca(cryos, means, gt_results, n_pcs)
    pca_results["mean_error"] = mean_error

    print("\n[3/8] Running L2 PPCA scale sweep (NO whitening)...")
    l2_no_whiten = run_l2_sweep(
        cryos,
        mean_estimate,
        W_init,
        W_prior_base,
        U_gt,
        s_gt,
        gt_results.volume_shape,
        l2_scales,
        n_iter,
        use_whitening=False,
    )
    print("\n[4/8] Running L2 PPCA scale sweep (Cz whitening)...")
    l2_cz = run_l2_sweep(
        cryos,
        mean_estimate,
        W_init,
        W_prior_base,
        U_gt,
        s_gt,
        gt_results.volume_shape,
        l2_scales,
        n_iter,
        use_whitening=True,
        whitening_mode="cz",
    )
    print("\n[5/8] Running L2 PPCA scale sweep (proj_ls whitening)...")
    l2_proj_ls = run_l2_sweep(
        cryos,
        mean_estimate,
        W_init,
        W_prior_base,
        U_gt,
        s_gt,
        gt_results.volume_shape,
        l2_scales,
        n_iter,
        use_whitening=True,
        whitening_mode="proj_ls",
    )

    W_l2_best_cz = jnp.array(max(l2_cz, key=lambda r: r["rel_var"])["W"])
    W_l2_best_proj_ls = jnp.array(max(l2_proj_ls, key=lambda r: r["rel_var"])["W"])
    W_l2_best_no_whiten = jnp.array(max(l2_no_whiten, key=lambda r: r["rel_var"])["W"])

    print("\n[6/8] Running L1 PPCA scale sweep (NO whitening)...")
    l1_no_whiten = run_l1_sweep(
        cryos,
        mean_estimate,
        W_l2_best_no_whiten,
        W_prior_base,
        U_gt,
        s_gt,
        gt_results.volume_shape,
        l1_scales,
        n_iter,
        use_whitening=False,
    )
    print("\n[7/8] Running L1 PPCA scale sweep (Cz whitening)...")
    l1_cz = run_l1_sweep(
        cryos,
        mean_estimate,
        W_l2_best_cz,
        W_prior_base,
        U_gt,
        s_gt,
        gt_results.volume_shape,
        l1_scales,
        n_iter,
        use_whitening=True,
        whitening_mode="cz",
    )
    print("\n[8/8] Running L1 PPCA scale sweep (proj_ls whitening)...")
    l1_proj_ls = run_l1_sweep(
        cryos,
        mean_estimate,
        W_l2_best_proj_ls,
        W_prior_base,
        U_gt,
        s_gt,
        gt_results.volume_shape,
        l1_scales,
        n_iter,
        use_whitening=True,
        whitening_mode="proj_ls",
    )

    print("\nCreating whitening comparison plot...")
    plot_whitening_comparison(
        l1_cz, l1_proj_ls, l1_no_whiten, l2_cz, l2_proj_ls, l2_no_whiten, dataset_name, pca_results, plots_dir, n_pcs
    )

    # Scale diagnostics: ||W||, tr(Ĉ_z), ||Ĉ_z-I|| vs scale
    plot_scale_diagnostics(l1_cz, l1_proj_ls, l1_no_whiten, l2_cz, l2_proj_ls, l2_no_whiten, dataset_name, plots_dir)

    # EM convergence: all 6 methods on one figure (best scale each)
    plot_em_convergence_all_methods(
        l1_cz, l1_proj_ls, l1_no_whiten, l2_cz, l2_proj_ls, l2_no_whiten, dataset_name, plots_dir
    )

    volume_shape = tuple(gt_results.volume_shape)

    def strip_W(r_list):
        return [{k: v for k, v in r.items() if k not in ("W", "C_z", "iteration_data")} for r in r_list]

    best_l1_cz = max(l1_cz, key=lambda r: r["rel_var"])
    best_l1_ls = max(l1_proj_ls, key=lambda r: r["rel_var"])
    best_l2_cz = max(l2_cz, key=lambda r: r["rel_var"])
    best_l2_ls = max(l2_proj_ls, key=lambda r: r["rel_var"])

    summary = {
        "dataset": dataset_name,
        "n_volumes": dataset_info["n_volumes"],
        "grid_size": grid_size,
        "n_images": n_images,
        "noise_level": noise_level,
        "n_pcs": n_pcs,
        "seed": seed,
        "compare_whitening": True,
        "pca_results": {k: v for k, v in pca_results.items() if k != "W_init"},
        "l1_cz": strip_W(l1_cz),
        "l1_proj_ls": strip_W(l1_proj_ls),
        "l1_no_whiten": strip_W(l1_no_whiten),
        "l2_cz": strip_W(l2_cz),
        "l2_proj_ls": strip_W(l2_proj_ls),
        "l2_no_whiten": strip_W(l2_no_whiten),
        "best_l1_cz_relvar": best_l1_cz["rel_var"],
        "best_l1_cz_scale": best_l1_cz["scale"],
        "best_l1_ls_relvar": best_l1_ls["rel_var"],
        "best_l1_ls_scale": best_l1_ls["scale"],
        "best_l2_cz_relvar": best_l2_cz["rel_var"],
        "best_l2_cz_scale": best_l2_cz["scale"],
        "best_l2_ls_relvar": best_l2_ls["rel_var"],
        "best_l2_ls_scale": best_l2_ls["scale"],
        "volume_shape": volume_shape,
    }

    results_path = os.path.join(results_output, "whitening_comparison_results.npy")
    np.save(results_path, summary)
    print(f"  Saved results: {results_path}")

    return {
        "l1_cz": l1_cz,
        "l1_proj_ls": l1_proj_ls,
        "l1_no_whiten": l1_no_whiten,
        "l2_cz": l2_cz,
        "l2_proj_ls": l2_proj_ls,
        "l2_no_whiten": l2_no_whiten,
        "pca_results": pca_results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="PPCA Scale Parameter Sweep (L1 + L2) for CryoBench datasets. Run with PYTHONPATH set to recovar parent."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/mg6942/mytigress/cryobench2",
        help="Base directory containing CryoBench datasets (Ribosembly, IgG-1D, etc.)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for results (simulated data, .npy); default: next to data, base-dir/ppca_sweep_results or .../ppca_sweep_results_10pc",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="ppca_plots",
        help="Directory for plot images (default: ppca_plots, relative to cwd)",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Run only this dataset (default: all)")
    parser.add_argument("--grid-size", type=int, default=128, help="Volume grid size")
    parser.add_argument("--n-images", type=int, default=50000, help="Number of images")
    parser.add_argument("--noise-level", type=float, default=1.0, help="Noise level")
    parser.add_argument("--n-pcs", type=int, default=10, help="Number of principal components")
    parser.add_argument("--em-iters", type=int, default=20, help="EM iterations per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--whitening-mode",
        type=str,
        default="cz",
        choices=["cz", "proj_ls"],
        help="Whitening mode when use_whitening=True (default: cz); ignored if --compare-whitening",
    )
    parser.add_argument(
        "--compare-whitening",
        action="store_true",
        help="Run both Cz and proj_ls whitening and compare (L1/L2 sweeps for each + no whiten)",
    )
    parser.add_argument(
        "--l1-scales", type=str, default="0.01,0.03,0.1,0.3,1.0,3.0,10.0", help="Comma-separated L1 scale values"
    )
    parser.add_argument("--l2-scales", type=str, default="0.01,0.1,1.0,10.0", help="Comma-separated L2 scale values")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only generate dataset + PCA warmstart and save pca_warmstart.npz (for single-run jobs)",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run one (strategy, scale) job; requires --strategy and --scale, and pca_warmstart.npz",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy for single-run: l2_no_whiten, l2_cz, l2_proj_ls, l1_no_whiten, l1_cz, l1_proj_ls",
    )
    parser.add_argument("--scale", type=float, default=None, help="Scale value for single-run")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run id (e.g. setup job id); single-run results go to dataset/run_<run_id>/",
    )
    parser.add_argument(
        "--combine-and-plot",
        action="store_true",
        help="Load single_*.npy and generate comparison plots; use --run-id to combine a specific run (default: latest run_*)",
    )
    args = parser.parse_args()
    if args.run_id is None and os.environ.get("RUN_ID"):
        args.run_id = os.environ.get("RUN_ID")

    _log_jax_devices()

    if args.combine_and_plot:
        if not args.dataset:
            print("ERROR: --combine-and-plot requires --dataset")
            return
        if args.results_dir is not None:
            results_root = os.path.abspath(args.results_dir)
        else:
            results_root = os.path.join(args.base_dir, "ppca_sweep_results_10pc")
        plots_root = os.path.abspath(args.plots_dir)
        run_id = getattr(args, "run_id", None) or os.environ.get("RUN_ID")
        runs = list_run_dirs(results_root, args.dataset)
        if run_id is None and runs:
            run_id = runs[-1][0]
            print(f"No --run-id given; using latest run: {run_id} (available: {[r[0] for r in runs]})")
        elif run_id is None:
            print("No run_* dirs found; combining from dataset dir (legacy single_*.npy).")
        combine_and_plot_single_runs(
            results_root, args.dataset, plots_root, args.n_pcs, results_output=None, run_id=run_id
        )
        return

    if args.setup_only:
        if not args.dataset:
            print("ERROR: --setup-only requires --dataset")
            return
        datasets = find_cryobench_datasets(args.base_dir)
        datasets = [d for d in datasets if d["name"] == args.dataset]
        if not datasets:
            print(f"ERROR: Dataset '{args.dataset}' not found")
            return
        if args.results_dir is not None:
            results_root = os.path.abspath(args.results_dir)
        else:
            results_root = os.path.join(args.base_dir, "ppca_sweep_results_10pc")
        run_setup_only(
            datasets[0], args.grid_size, args.n_images, args.noise_level, args.n_pcs, args.seed, results_root
        )
        print("Setup done. Submit single-run jobs with --single-run --strategy STRAT --scale SCALE")
        return

    if args.single_run:
        if not args.dataset or not args.strategy or args.scale is None:
            print("ERROR: --single-run requires --dataset, --strategy, and --scale")
            return
        if args.results_dir is not None:
            results_root = os.path.abspath(args.results_dir)
        else:
            results_root = os.path.join(args.base_dir, "ppca_sweep_results_10pc")
        results_output = None
        if args.run_id:
            results_output = os.path.join(results_root, args.dataset, f"run_{args.run_id}")
            os.makedirs(results_output, exist_ok=True)
        run_single_scale(
            args.dataset,
            args.grid_size,
            args.n_pcs,
            args.strategy,
            args.scale,
            args.em_iters,
            results_root,
            results_output=results_output,
        )
        return

    if args.results_dir is not None:
        results_root = os.path.abspath(args.results_dir)
    else:
        if args.n_pcs == 10:
            results_root = os.path.join(args.base_dir, "ppca_sweep_results_10pc")
        else:
            results_root = os.path.join(args.base_dir, "ppca_sweep_results")
    plots_root = os.path.abspath(args.plots_dir)

    l1_scales = [float(s) for s in args.l1_scales.split(",")]
    l2_scales = [float(s) for s in args.l2_scales.split(",")]

    print("PPCA Scale Parameter Sweep (L1 + L2)")
    print("=" * 60)
    print(f"Base directory: {args.base_dir}")
    print(f"Results directory (data, .npy): {results_root}")
    print(f"Plots directory: {plots_root}")
    print(f"Grid size: {args.grid_size}, N images: {args.n_images}, Noise: {args.noise_level}")
    print(f"N PCs: {args.n_pcs}, EM iters: {args.em_iters}")
    if args.compare_whitening:
        print("Mode: compare whitening (Cz vs proj_ls vs no whiten)")
    else:
        print(f"Whitening mode: {args.whitening_mode}")
    print(f"L1 Scales: {l1_scales}")
    print(f"L2 Scales: {l2_scales}")
    print("=" * 60)

    datasets = find_cryobench_datasets(args.base_dir)
    print(f"\nFound {len(datasets)} datasets:")
    for d in datasets:
        print(f"  - {d['name']}: {d['n_volumes']} volumes")

    if args.dataset:
        datasets = [d for d in datasets if d["name"] == args.dataset]
        if not datasets:
            print(f"ERROR: Dataset '{args.dataset}' not found")
            return

    output.mkdir_safe(results_root)
    os.makedirs(plots_root, exist_ok=True)
    all_results = {}

    for dataset_info in datasets:
        try:
            if args.compare_whitening:
                result = run_benchmark_compare_whitening(
                    dataset_info,
                    args.grid_size,
                    args.n_images,
                    args.noise_level,
                    args.n_pcs,
                    l1_scales,
                    l2_scales,
                    args.em_iters,
                    args.seed,
                    results_root,
                    plots_root,
                )
            else:
                result = run_benchmark(
                    dataset_info,
                    args.grid_size,
                    args.n_images,
                    args.noise_level,
                    args.n_pcs,
                    l1_scales,
                    l2_scales,
                    args.em_iters,
                    args.seed,
                    results_root,
                    plots_root,
                    whitening_mode=args.whitening_mode,
                )
            all_results[dataset_info["name"]] = result
        except Exception as e:
            print(f"\nERROR on {dataset_info['name']}: {e}")
            import traceback

            traceback.print_exc()

    if len(all_results) > 1 and not args.compare_whitening:
        plot_combined_summary(all_results, plots_root)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for name, data in all_results.items():
        summary = data["summary"]
        pca_relvar = summary["pca_results"]["rel_var"]
        print(f"\n{name}:")
        print(f"  Covariance-based PCA: {pca_relvar:.4f}")
        if summary.get("compare_whitening"):
            print(f"  L1 best Cz:    {summary['best_l1_cz_relvar']:.4f} (scale={summary['best_l1_cz_scale']:.6f})")
            print(f"  L1 best proj_ls: {summary['best_l1_ls_relvar']:.4f} (scale={summary['best_l1_ls_scale']:.6f})")
            print(f"  L2 best Cz:    {summary['best_l2_cz_relvar']:.4f} (scale={summary['best_l2_cz_scale']:.4f})")
            print(f"  L2 best proj_ls: {summary['best_l2_ls_relvar']:.4f} (scale={summary['best_l2_ls_scale']:.4f})")
        else:
            print(f"  Best L1 PPCA: {summary['best_l1_relvar']:.4f} (scale={summary['best_l1_scale']:.6f})")
            print(f"  Best L2 PPCA: {summary['best_l2_relvar']:.4f} (scale={summary['best_l2_scale']:.4f})")
            l1_imp = (summary["best_l1_relvar"] - pca_relvar) / pca_relvar * 100
            l2_imp = (summary["best_l2_relvar"] - pca_relvar) / pca_relvar * 100
            print(f"  L1 vs PCA: {l1_imp:+.1f}%  |  L2 vs PCA: {l2_imp:+.1f}%")

    print(f"\nResults (data, .npy): {results_root}")
    print(f"Plots: {plots_root}")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
