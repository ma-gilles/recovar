"""Self-contained synthetic-data walkthrough for a user-supplied volume stack.

This is the spike-friendly cousin of ``benchmark_kernel_bandwidth_1d``. It
does NOT import any trajectory-generation / 5nrl machinery — it operates on
a stack of MRC volumes at ``<output_dir>/01_raw_volumes/vol####.mrc``,
either pre-rendered or rendered by this script via ``--pdb-dir``.

Pipeline stages
---------------

1. ``01_raw_volumes/`` — the input stack. If empty and ``--pdb-dir`` is
   given, this script renders ``morph_*.pdb`` files from that directory into
   ``vol####.mrc`` using ``render_spike_morph_volumes``. Voxel size is then
   read from the MRC header.
2. ``02_active_volumes/`` — the volumes actually used by the simulator.
   Either a verbatim copy of the input stack, or (with ``--pc-project k``)
   the same stack projected onto its top-``k`` real-space PCs.
3. ``03_dataset/`` — simulated particle stack, STAR file, poses, CTF,
   per-image state assignment.
4. ``04_ground_truth/`` — GT reconstructions emitted by the simulator.
5. ``05_masks/`` — union envelope mask + localized "focus" mask.
6. ``06_pipeline/`` — RECOVAR pipeline output. With ``--use-oracle-pipeline``
   (default), mean / PCs / eigenvalues / noise come from the simulator and
   only the per-image latent embedding is computed from the noisy images.
   Without that flag, the full ``recovar pipeline`` is run.
7. ``07_compute_state/`` — ``recovar compute_state`` at the target state's
   latent centroid (the mean of latents of images assigned to that state).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import mrcfile
import numpy as np

import recovar.jax_config  # noqa: F401 - initialize JAX config before using RECOVAR
from recovar import utils
from recovar.commands import compute_state as compute_state_cmd
from recovar.commands import pipeline
from recovar.commands import render_spike_morph_volumes as render_cmd
from recovar.commands import spike_kernel_report
from recovar.core import fourier_transform_utils as ftu
from recovar.heterogeneity import kernel_bandwidth_benchmark as kb
from recovar.output import output as o
from recovar.project.job_context import job_context
from recovar.simulation import oracle_pipeline, simulator, synthetic_dataset
from recovar.utils.helpers import RobustFileHandler, RobustStreamHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_volume_stack(prefix: Path, n_states: int) -> np.ndarray:
    volumes = []
    for idx in range(n_states):
        path = Path(f"{prefix}{idx:04d}.mrc")
        if not path.exists():
            raise FileNotFoundError(f"Missing expected volume {path}")
        volumes.append(np.asarray(utils.load_mrc(path), dtype=np.float32))
    return np.stack(volumes, axis=0)


def _count_raw_volumes(raw_dir: Path) -> int:
    return sum(1 for _ in raw_dir.glob("vol[0-9][0-9][0-9][0-9].mrc"))


def _read_voxel_size(mrc_path: Path, fallback: float | None = None) -> float:
    with mrcfile.open(mrc_path, permissive=True) as m:
        v = float(m.voxel_size.x)
    if v > 0:
        return v
    if fallback is not None and fallback > 0:
        logger.warning("MRC header at %s has voxel_size=0; using --voxel-size=%.4f", mrc_path, fallback)
        return float(fallback)
    raise ValueError(
        f"MRC voxel_size is unset in {mrc_path}. Pass --voxel-size explicitly or re-render the volumes "
        f"with a renderer that writes voxel_size into the header (e.g. render_spike_morph_volumes)."
    )


def _uniform_state_distribution(n_states: int) -> np.ndarray:
    if n_states <= 0:
        raise ValueError(f"n_states must be positive, got {n_states}")
    return np.full(n_states, 1.0 / n_states, dtype=np.float32)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Stage 02 — active volumes
# ---------------------------------------------------------------------------


def _write_active_volumes(
    raw_volumes: np.ndarray, pc_project: int, out: Path, voxel_size: float
) -> tuple[Path, np.ndarray, dict]:
    active_dir = out / "02_active_volumes"
    active_prefix = active_dir / "vol"
    active_dir.mkdir(parents=True, exist_ok=True)

    if pc_project > 0:
        active_volumes, pca_meta = kb.project_volume_trajectory(raw_volumes, pc_project)
        logger.info("Using volume stack projected onto the first %d PC(s)", pc_project)
    else:
        active_volumes = np.asarray(raw_volumes, dtype=np.float32)
        pca_meta = kb.project_volume_trajectory(raw_volumes, 0)[1]
        logger.info("Using raw volume stack verbatim")

    kb.write_volume_prefix(active_volumes, active_prefix, voxel_size)
    np.save(active_dir / "volumes.npy", active_volumes.astype(np.float32))
    np.save(active_dir / "pca_scores.npy", np.asarray(pca_meta["scores"], dtype=np.float32))
    np.save(active_dir / "pca_singular_values.npy", np.asarray(pca_meta["singular_values"], dtype=np.float32))
    np.save(active_dir / "pca_explained_energy.npy", np.asarray(pca_meta["explained_energy"], dtype=np.float32))
    _write_json(
        active_dir / "README.json",
        {
            "description": "Volumes used by the simulator (verbatim or PC-projected).",
            "pc_project": int(pc_project),
            "volume_files": "vol0000.mrc, vol0001.mrc, ...",
            "volumes_npy": str(active_dir / "volumes.npy"),
            "pca_scores_npy": str(active_dir / "pca_scores.npy"),
        },
    )
    return active_prefix, active_volumes, pca_meta


# ---------------------------------------------------------------------------
# Stage 03 — dataset simulation
# ---------------------------------------------------------------------------


def _simulate_dataset(args, out: Path, volume_prefix: Path, voxel_size: float) -> tuple[np.ndarray, dict]:
    dataset_dir = out / "03_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    particles_path = dataset_dir / f"particles.{args.grid_size}.mrcs"
    sim_info_path = dataset_dir / "simulation_info.pkl"

    if particles_path.exists() and sim_info_path.exists() and not args.overwrite:
        logger.info("Reusing existing simulated dataset in %s", dataset_dir)
        with sim_info_path.open("rb") as f:
            sim_info = pickle.load(f)
        return np.asarray(utils.load_mrc(particles_path), dtype=np.float32), sim_info

    logger.info("Simulating %d images in %s", args.n_images, dataset_dir)
    np.random.seed(args.seed)
    image_stack, sim_info = simulator.generate_synthetic_dataset(
        str(dataset_dir),
        voxel_size,
        str(volume_prefix),
        int(args.n_images),
        outlier_file_input=None,
        grid_size=args.grid_size,
        volume_distribution=_uniform_state_distribution(args.n_states),
        dataset_params_option="uniform",
        noise_level=args.noise_level,
        noise_model=args.noise_model,
        put_extra_particles=False,
        percent_outliers=0.0,
        volume_radius=0.7,
        trailing_zero_format_in_vol_name=True,
        noise_scale_std=0.0,
        contrast_std=0.0,
        per_particle_contrast=False,
        disc_type="cubic",
        n_tilts=-1,
        premultiplied_ctf=args.premultiplied_ctf,
    )
    np.save(dataset_dir / "state_assignment.npy", np.asarray(sim_info["image_assignment"], dtype=np.int32))
    _write_json(
        dataset_dir / "README.json",
        {
            "description": "Synthetic particle stack and metadata generated from 02_active_volumes.",
            "contrast_std": 0.0,
            "noise_scale_std": 0.0,
            "noise_level": float(args.noise_level),
            "particles": str(particles_path),
            "star": str(dataset_dir / "particles.star"),
            "poses": str(dataset_dir / "poses.pkl"),
            "ctf": str(dataset_dir / "ctf.pkl"),
            "state_assignment": str(dataset_dir / "state_assignment.npy"),
        },
    )
    return np.asarray(image_stack, dtype=np.float32), sim_info


# ---------------------------------------------------------------------------
# Stages 04+05 — GT volumes and masks
# ---------------------------------------------------------------------------


def _write_gt_masks_and_volumes(
    out: Path,
    sim_info: dict,
    grid_size: int,
    voxel_size: float,
    mask_dilation_iters: int | None = None,
    focus_mask_percentile: float = 95.0,
) -> dict[str, str]:
    from recovar.core import mask as core_mask

    gt_dir = out / "04_ground_truth"
    mask_dir = out / "05_masks"
    gt_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_info)
    gt_volumes = np.asarray(
        [ftu.get_idft3(vol.reshape((grid_size, grid_size, grid_size))).real for vol in gt.volumes],
        dtype=np.float32,
    )
    np.save(gt_dir / "gt_volumes_used_by_simulator.npy", gt_volumes)
    kb.write_volume_prefix(gt_volumes, gt_dir / "gt_vol", voxel_size)

    volume_shape = (grid_size, grid_size, grid_size)
    real_vols = [v.reshape(volume_shape) for v in gt_volumes]
    volume_mask, binary_volume_mask = core_mask.make_union_gt_mask(
        real_vols, volume_shape, dilation_iters=mask_dilation_iters
    )
    if focus_mask_percentile and focus_mask_percentile > 0:
        focus_mask, binary_focus_mask = core_mask.make_localized_moving_gt_mask(
            real_vols, volume_shape, percentile=focus_mask_percentile, envelope_mask=binary_volume_mask
        )
        focus_mask_method = f"localized@p{focus_mask_percentile:g}"
    else:
        focus_mask, binary_focus_mask = core_mask.make_moving_gt_mask(
            real_vols, volume_shape, dilation_iters=mask_dilation_iters
        )
        focus_mask_method = "legacy_anything_moves"
    utils.write_mrc(
        mask_dir / "volume_mask_union.mrc", np.asarray(volume_mask, dtype=np.float32), voxel_size=voxel_size
    )
    utils.write_mrc(mask_dir / "focus_mask_moving.mrc", np.asarray(focus_mask, dtype=np.float32), voxel_size=voxel_size)
    np.save(mask_dir / "volume_mask_union.npy", np.asarray(volume_mask, dtype=np.float32))
    np.save(mask_dir / "focus_mask_moving.npy", np.asarray(focus_mask, dtype=np.float32))
    _write_json(
        mask_dir / "README.json",
        {
            "description": "Masks passed to the pipeline. volume_mask_union is --mask; focus_mask_moving is --focus-mask.",
            "volume_mask_union_fraction": float(np.mean(binary_volume_mask)),
            "focus_mask_moving_fraction": float(np.mean(binary_focus_mask)),
            "dilation_iters": mask_dilation_iters,
            "focus_mask_method": focus_mask_method,
            "focus_mask_percentile": float(focus_mask_percentile) if focus_mask_percentile else 0.0,
        },
    )
    return {
        "gt_dir": str(gt_dir),
        "volume_mask": str(mask_dir / "volume_mask_union.mrc"),
        "focus_mask": str(mask_dir / "focus_mask_moving.mrc"),
    }


# ---------------------------------------------------------------------------
# Stage 06 — pipeline (full or oracle)
# ---------------------------------------------------------------------------


def _run_pipeline(args, out: Path, mask_paths: dict[str, str]) -> Path:
    pipeline_dir = out / "06_pipeline"
    if (pipeline_dir / "model" / "params.pkl").exists() and not args.overwrite:
        logger.info("Reusing existing pipeline output in %s", pipeline_dir)
        return pipeline_dir

    dataset_dir = out / "03_dataset"
    pipeline_cmd = [
        str(dataset_dir / "particles.star"),
        "--poses",
        str(dataset_dir / "poses.pkl"),
        "--ctf",
        str(dataset_dir / "ctf.pkl"),
        "-o",
        str(pipeline_dir),
        "--mask",
        mask_paths["volume_mask"],
        "--focus-mask",
        mask_paths["focus_mask"],
        "--no-correct-contrast",
        "--noise-model",
        "radial" if args.noise_model == "radial1" else args.noise_model,
    ]
    if args.lazy:
        pipeline_cmd.append("--lazy")
    if args.low_memory_option:
        pipeline_cmd.append("--low-memory-option")
    if args.very_low_memory_option:
        pipeline_cmd.append("--very-low-memory-option")
    if args.pipeline_gpu_memory is not None:
        pipeline_cmd.extend(["--gpu-gb", str(args.pipeline_gpu_memory)])
    if args.premultiplied_ctf:
        pipeline_cmd.append("--premultiplied-ctf")

    logger.info("Running recovar pipeline %s", " ".join(pipeline_cmd))
    parser = pipeline.add_args(argparse.ArgumentParser())
    pipeline.standard_recovar_pipeline(parser.parse_args(pipeline_cmd))
    return pipeline_dir


def _run_oracle_pipeline(
    args,
    out: Path,
    mask_paths: dict[str, str],
    sim_info: dict,
    voxel_size: float,
) -> Path:
    pipeline_dir = out / "06_pipeline"
    if (pipeline_dir / "model" / "params.pkl").exists() and not args.overwrite:
        logger.info("Reusing existing oracle pipeline output in %s", pipeline_dir)
        return pipeline_dir

    volume_mask = np.asarray(utils.load_mrc(mask_paths["volume_mask"]), dtype=np.float32)
    focus_mask = np.asarray(utils.load_mrc(mask_paths["focus_mask"]), dtype=np.float32)
    zdims = sorted({int(args.zdim), 1, 2})

    summary = oracle_pipeline.write_oracle_pipeline_output(
        pipeline_dir=pipeline_dir,
        dataset_dir=out / "03_dataset",
        voxel_size=voxel_size,
        volume_mask=volume_mask,
        sim_info=sim_info,
        zdims=zdims,
        gpu_memory=args.pipeline_gpu_memory,
        focus_mask=focus_mask,
        premultiplied_ctf=bool(args.premultiplied_ctf),
        noise_model="radial" if args.noise_model == "radial1" else args.noise_model,
        lazy=bool(args.lazy),
    )
    _write_json(pipeline_dir / "oracle_pipeline_info.json", summary)
    return pipeline_dir


# ---------------------------------------------------------------------------
# Stage 07 — compute_state at the target state's latent centroid
# ---------------------------------------------------------------------------


def _target_state_latent_point(
    pipeline_dir: Path,
    sim_info: dict,
    target_state: int,
    zdim: int,
    *,
    coords_entry: str = "latent_coords",
) -> np.ndarray:
    po = o.PipelineOutput(str(pipeline_dir))
    latent_by_zdim = po.get(coords_entry)
    key = zdim if zdim in latent_by_zdim else f"zdim_{zdim}"
    if key not in latent_by_zdim:
        raise KeyError(f"Pipeline output has no {coords_entry}/{key}. Available: {sorted(latent_by_zdim)}")
    zs = np.asarray(latent_by_zdim[key], dtype=np.float32)
    assignments = np.asarray(sim_info["image_assignment"], dtype=np.int32).reshape(-1)
    target_particles = assignments == int(target_state)
    if not np.any(target_particles):
        raise RuntimeError(f"No simulated particles were assigned to target state {target_state}")
    latent_point = np.mean(zs[target_particles], axis=0, dtype=np.float64).astype(np.float32)
    return latent_point.reshape(1, -1)


def _run_compute_state(args, out: Path, pipeline_dir: Path, latent_point: np.ndarray) -> Path:
    compute_state_dir = out / "07_compute_state"
    latent_path = out / "target_latent_point.txt"
    np.savetxt(latent_path, np.asarray(latent_point, dtype=np.float32))

    compute_state_args = argparse.Namespace(
        result_dir=str(pipeline_dir),
        outdir=str(compute_state_dir),
        project=None,
        Bfactor=float(args.bfactor),
        n_bins=int(args.n_bins),
        maskrad_fraction=float(args.compute_state_maskrad_fraction),
        n_min_particles=int(args.compute_state_n_min_particles),
        zdim1=(latent_point.shape[1] == 1),
        no_z_regularization=False,
        lazy=bool(args.lazy),
        particles=None,
        datadir=None,
        strip_prefix=None,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
        gpu_memory=args.pipeline_gpu_memory,
        latent_points=str(latent_path),
        save_all_estimates=bool(args.compute_state_save_all_estimates),
        kernel_regression_mode=str(args.compute_state_kernel_regression_mode),
        deconv_lambda_grid=args.compute_state_deconv_lambda_grid,
        local_poly_degree=int(args.compute_state_local_poly_degree),
        local_poly_bandwidth_multipliers=args.compute_state_local_poly_bandwidth_multipliers,
    )
    logger.info("Running recovar compute_state at %s", latent_path)
    compute_state_cmd.compute_state(compute_state_args)
    return compute_state_dir


def _resolve_kernel_report_roots(args, out: Path, compute_state_dir: Path) -> tuple[Path | None, Path | None]:
    standard_root = Path(args.kernel_report_standard_dir) if args.kernel_report_standard_dir else None
    deconvolved_root = Path(args.kernel_report_deconvolved_dir) if args.kernel_report_deconvolved_dir else None

    if standard_root is None:
        if args.compute_state_kernel_regression_mode == "standard":
            standard_root = compute_state_dir
        elif (out / "07_compute_state_standard").exists():
            standard_root = out / "07_compute_state_standard"

    if deconvolved_root is None:
        if args.compute_state_kernel_regression_mode == "deconvolved":
            deconvolved_root = compute_state_dir
        elif (out / "07_compute_state_deconvolved").exists():
            deconvolved_root = out / "07_compute_state_deconvolved"

    return standard_root, deconvolved_root


def _kernel_report_can_use_current_run(args, out: Path) -> bool:
    if not args.kernel_report:
        return False
    if args.compute_state_kernel_regression_mode == "standard":
        return bool(args.kernel_report_deconvolved_dir) or (out / "07_compute_state_deconvolved").exists()
    if args.compute_state_kernel_regression_mode == "deconvolved":
        return bool(args.kernel_report_standard_dir) or (out / "07_compute_state_standard").exists()
    return False


def _maybe_run_kernel_report(
    args,
    out: Path,
    pipeline_dir: Path,
    compute_state_dir: Path,
    mask_paths: dict[str, str],
    target_state: int,
) -> dict | None:
    if not args.kernel_report:
        return None

    standard_root, deconvolved_root = _resolve_kernel_report_roots(args, out, compute_state_dir)
    if standard_root is None or deconvolved_root is None:
        logger.info(
            "Skipping 08 kernel report: need both standard and deconvolved compute_state outputs. "
            "Resolved standard=%s deconvolved=%s. Use --kernel-report-standard-dir and "
            "--kernel-report-deconvolved-dir, or place paired outputs at 07_compute_state_standard "
            "and 07_compute_state_deconvolved.",
            standard_root,
            deconvolved_root,
        )
        return None

    target_volume = out / "04_ground_truth" / f"gt_vol{target_state:04d}.mrc"
    mask = Path(args.kernel_report_mask) if args.kernel_report_mask else Path(mask_paths["focus_mask"])
    target_point = out / "target_latent_point.txt"
    report_out = Path(args.kernel_report_out_dir) if args.kernel_report_out_dir else out / "08_kernel_report"

    cfg = spike_kernel_report.SpikeKernelReportConfig(
        standard_root=standard_root,
        deconvolved_root=deconvolved_root,
        target_volume=target_volume,
        mask=mask,
        out_dir=report_out,
        pipeline_root=pipeline_dir,
        target_point=target_point,
        state_label="state000",
        report_title=f"target state {target_state}",
        expected_candidates=args.kernel_report_expected_candidates,
        package_lowpass=bool(args.kernel_report_package_lowpass),
        lowpass_cutoff=float(args.kernel_report_lowpass_cutoff),
    )
    logger.info("Generating 08 kernel report in %s", report_out)
    return spike_kernel_report.generate_report(cfg)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_walkthrough(args, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "01_raw_volumes"
    n_raw = _count_raw_volumes(raw_dir)

    # Stage 01 — render volumes from PDBs if not already present.
    if n_raw == 0:
        if args.pdb_dir is None:
            raise SystemExit(
                f"No volumes in {raw_dir} and --pdb-dir not given. Pass --pdb-dir <DIR> to render "
                f"from PDB files, or pre-render with render_spike_morph_volumes."
            )
        logger.info(
            "Rendering volumes from PDBs at %s (B=%.1f, grid=%d, vox=%.3f)",
            args.pdb_dir,
            args.render_bfactor,
            args.grid_size,
            args.render_voxel_size,
        )
        render_cmd.render_stack(
            pdb_dir=Path(args.pdb_dir),
            out_dir=out,
            grid_size=args.grid_size,
            voxel_size=args.render_voxel_size,
            bfactor=args.render_bfactor,
            glob_pattern=args.render_glob,
        )
        n_raw = _count_raw_volumes(raw_dir)
        if n_raw == 0:
            raise SystemExit(f"Rendering produced no volumes in {raw_dir}")
    elif args.pdb_dir is not None:
        logger.info("Skipping render — %d volumes already exist in %s", n_raw, raw_dir)

    if args.n_states is None:
        args.n_states = n_raw
        logger.info("Auto-detected n_states=%d from %s", n_raw, raw_dir)
    elif args.n_states > n_raw:
        raise SystemExit(f"--n-states {args.n_states} > {n_raw} volumes available in {raw_dir}")

    # Read voxel_size from the first raw volume's MRC header (mandatory),
    # unless the user passed --voxel-size to override.
    voxel_size = float(args.voxel_size) if args.voxel_size is not None else _read_voxel_size(raw_dir / "vol0000.mrc")
    logger.info("voxel_size = %.4f Å (box edge = %.1f Å)", voxel_size, args.grid_size * voxel_size)

    target_state = args.target_state if args.target_state is not None else args.n_states // 2
    if not (0 <= target_state < args.n_states):
        raise ValueError(f"--target-state must be in [0, {args.n_states - 1}], got {target_state}")

    raw_volumes = _load_volume_stack(raw_dir / "vol", args.n_states)
    active_prefix, _active_volumes, pca_meta = _write_active_volumes(raw_volumes, args.pc_project, out, voxel_size)
    _image_stack, sim_info = _simulate_dataset(args, out, active_prefix, voxel_size)
    mask_paths = _write_gt_masks_and_volumes(
        out,
        sim_info,
        args.grid_size,
        voxel_size,
        mask_dilation_iters=args.mask_dilation_iters,
        focus_mask_percentile=args.focus_mask_percentile,
    )
    if args.use_oracle_pipeline:
        pipeline_dir = _run_oracle_pipeline(args, out, mask_paths, sim_info, voxel_size)
    else:
        pipeline_dir = _run_pipeline(args, out, mask_paths)
    target_coords_entry = (
        "latent_coords_noreg"
        if args.compute_state_kernel_regression_mode in ("deconvolved", "local_poly")
        else "latent_coords"
    )
    latent_point = _target_state_latent_point(
        pipeline_dir,
        sim_info,
        target_state,
        args.zdim,
        coords_entry=target_coords_entry,
    )
    if _kernel_report_can_use_current_run(args, out) and not args.compute_state_save_all_estimates:
        logger.info("Enabling compute_state save_all_estimates because --kernel-report is on by default.")
        args.compute_state_save_all_estimates = True
    compute_state_dir = _run_compute_state(args, out, pipeline_dir, latent_point)
    kernel_report_summary = _maybe_run_kernel_report(
        args,
        out,
        pipeline_dir,
        compute_state_dir,
        mask_paths,
        target_state,
    )

    summary = {
        "output_dir": str(out),
        "raw_volumes": str(raw_dir),
        "active_volumes": str(out / "02_active_volumes"),
        "dataset": str(out / "03_dataset"),
        "ground_truth": mask_paths["gt_dir"],
        "masks": str(out / "05_masks"),
        "pipeline": str(pipeline_dir),
        "compute_state": str(compute_state_dir),
        "target_state": int(target_state),
        "target_latent_point": latent_point.reshape(-1).tolist(),
        "grid_size": int(args.grid_size),
        "voxel_size": float(voxel_size),
        "box_edge_A": float(args.grid_size * voxel_size),
        "n_states": int(args.n_states),
        "n_images": int(args.n_images),
        "noise_level": float(args.noise_level),
        "pc_project": int(args.pc_project),
        "pca_explained_energy_head": np.asarray(pca_meta["explained_energy"])[:10].astype(float).tolist(),
        "raw_volume_prefix": str(raw_dir / "vol"),
        "active_volume_prefix": str(active_prefix),
        "use_oracle_pipeline": bool(args.use_oracle_pipeline),
        "compute_state_kernel_regression_mode": str(args.compute_state_kernel_regression_mode),
        "kernel_report": kernel_report_summary,
    }
    _write_json(out / "config.json", {"args": vars(args), "summary": summary})
    _write_readme(out, summary)
    return summary


def _write_readme(out: Path, summary: dict) -> None:
    text = f"""# RECOVAR Spike Walkthrough

Self-contained synthetic-data walkthrough on a user-supplied volume stack.
Generated by `recovar.commands.spike_walkthrough`.

1. `01_raw_volumes/` — pre-rendered or rendered here from `--pdb-dir`.
2. `02_active_volumes/` — verbatim or PCA-projected.
3. `03_dataset/` — simulated particle stack + STAR/poses/CTF.
4. `04_ground_truth/` — simulator GT reconstructions.
5. `05_masks/` — union envelope + localized focus mask.
6. `06_pipeline/` — RECOVAR pipeline output (oracle if `use_oracle_pipeline`).
7. `07_compute_state/` — compute_state at the target state's latent centroid.

Key settings:

- grid_size: {summary["grid_size"]}
- voxel_size (Å): {summary["voxel_size"]:.4f}
- box edge (Å):  {summary["box_edge_A"]:.1f}
- n_states: {summary["n_states"]}
- n_images: {summary["n_images"]}
- noise_level: {summary["noise_level"]}
- pc_project: {summary["pc_project"]}
- target_state: {summary["target_state"]}
- use_oracle_pipeline: {summary["use_oracle_pipeline"]}
- compute_state_kernel_regression_mode: {summary["compute_state_kernel_regression_mode"]}
- kernel_report: {summary["kernel_report"]["out_dir"] if summary["kernel_report"] else None}
"""
    (out / "README.md").write_text(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--output-dir",
        type=os.path.abspath,
        required=True,
        help="Run dir. Volumes are rendered to <output_dir>/01_raw_volumes/ if "
        "--pdb-dir is given and that dir doesn't already contain them.",
    )
    parser.add_argument(
        "--pdb-dir",
        type=str,
        default=None,
        help="Directory of morph_*.pdb files to render into 01_raw_volumes/. "
        "Skipped if 01_raw_volumes/ already has volumes. Required when not pre-rendered.",
    )
    parser.add_argument(
        "--render-glob",
        type=str,
        default="morph_*.pdb",
        help="PDB filename glob inside --pdb-dir (default: morph_*.pdb).",
    )
    parser.add_argument(
        "--render-voxel-size",
        type=float,
        default=2.0,
        help="Voxel size (Å) used when rendering from PDBs (default 2.0).",
    )
    parser.add_argument(
        "--render-bfactor",
        type=float,
        default=100.0,
        help="Rendering B-factor (Å²) applied to the PDB → density (default 100). "
        "Independent of --bfactor (which is the compute_state sharpening).",
    )
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Å per voxel. If unset, read from 01_raw_volumes/vol0000.mrc header.",
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=None,
        help="Number of conformational states. Defaults to the count of vol####.mrc files.",
    )
    parser.add_argument("--n-images", type=int, default=5000)
    parser.add_argument(
        "--noise-level", type=float, default=1.0, help="Simulator noise-level multiplier. 1.0 ~ typical cryo-EM SNR."
    )
    parser.add_argument("--noise-model", choices=["radial1", "radial"], default="radial1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pc-project", type=int, default=0, help="0 uses raw vols; 1 uses PC1-projected vols; k uses PCk."
    )
    parser.add_argument("--target-state", type=int, default=None, help="Default is the middle state (n_states // 2).")
    parser.add_argument(
        "--zdim", type=int, default=1, help="Pipeline latent dimension used to compute the target latent point."
    )
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--compute-state-maskrad-fraction", type=float, default=20.0)
    parser.add_argument("--compute-state-n-min-particles", type=int, default=100)
    parser.add_argument("--compute-state-save-all-estimates", action="store_true")
    parser.add_argument(
        "--kernel-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate an 08_kernel_report by default when both standard and deconvolved compute_state "
            "outputs are available. Use --no-kernel-report to disable."
        ),
    )
    parser.add_argument(
        "--kernel-report-standard-dir",
        type=str,
        default=None,
        help="Standard compute_state output dir for the 08 kernel report. Defaults to the current run if mode=standard, otherwise 07_compute_state_standard when present.",
    )
    parser.add_argument(
        "--kernel-report-deconvolved-dir",
        type=str,
        default=None,
        help="Deconvolved compute_state output dir for the 08 kernel report. Defaults to the current run if mode=deconvolved, otherwise 07_compute_state_deconvolved when present.",
    )
    parser.add_argument(
        "--kernel-report-mask",
        type=str,
        default=None,
        help="Mask for 08 report true-GT FSC/error metrics. Defaults to 05_masks/focus_mask_moving.mrc.",
    )
    parser.add_argument(
        "--kernel-report-out-dir",
        type=str,
        default=None,
        help="Output directory for the 08 report. Defaults to <output_dir>/08_kernel_report.",
    )
    parser.add_argument("--kernel-report-expected-candidates", type=int, default=None)
    parser.add_argument("--kernel-report-package-lowpass", action="store_true")
    parser.add_argument("--kernel-report-lowpass-cutoff", type=float, default=0.15)
    parser.add_argument(
        "--compute-state-kernel-regression-mode",
        choices=("standard", "deconvolved", "local_poly"),
        default="standard",
        help="Kernel-regression mode passed through to compute_state.",
    )
    parser.add_argument(
        "--compute-state-deconv-lambda-grid",
        type=str,
        default=None,
        help="Comma-separated lambda grid passed through to deconvolved compute_state.",
    )
    parser.add_argument(
        "--compute-state-local-poly-degree",
        type=int,
        default=3,
        help="Polynomial degree passed through to local_poly compute_state.",
    )
    parser.add_argument(
        "--compute-state-local-poly-bandwidth-multipliers",
        type=str,
        default=None,
        help="Comma-separated bandwidth multipliers passed through to local_poly compute_state.",
    )
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--low-memory-option", action="store_true")
    parser.add_argument("--very-low-memory-option", action="store_true")
    parser.add_argument("--pipeline-gpu-memory", type=float, default=None)
    parser.add_argument("--premultiplied-ctf", action="store_true")
    parser.add_argument(
        "--bfactor", type=float, default=0.0, help="compute_state sharpening B-factor. 0 = no sharpening (default)."
    )
    parser.add_argument("--mask-dilation-iters", type=int, default=None)
    parser.add_argument(
        "--focus-mask-percentile",
        type=float,
        default=95.0,
        help="0 = legacy 'anything moves' mask. Higher = tighter focus.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--use-oracle-pipeline",
        action="store_true",
        default=True,
        help="Use the simulator's GT mean/PCs as pipeline input (default). "
        "Pass --no-use-oracle-pipeline to run the full pipeline.",
    )
    parser.add_argument("--no-use-oracle-pipeline", dest="use_oracle_pipeline", action="store_false")
    return parser


def main() -> None:
    parser = add_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    with job_context(args, "spike_walkthrough") as ctx:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
            level=logging.INFO,
            handlers=[
                RobustFileHandler(os.path.join(ctx.output_dir, "run.log")),
                RobustStreamHandler(),
            ],
        )
        summary = run_walkthrough(args, Path(ctx.output_dir))
        logger.info("Finished walkthrough. Outputs: %s", summary)


if __name__ == "__main__":
    main()
