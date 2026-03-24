#!/usr/bin/env python3
"""Profile the compute_state reweighted-volume hot path on saved outputs.

This script avoids rerunning the entire pipeline. It loads a saved
``pipeline_output`` directory, reconstructs the exact inputs that
``compute_state`` would feed into ``compute_and_save_reweighted()``,
and profiles one or more reweighted state volumes.

Outputs:
- JSON summary of stage timings
- optional cProfile dump
- optional JAX trace directory
"""

from __future__ import annotations

import argparse
import cProfile
import contextlib
import io
import json
import pstats
import shutil
import time
from pathlib import Path

import numpy as np

from recovar.commands.compute_state import _build_reweighted_halfset_datasets
from recovar.heterogeneity import embedding, heterogeneity_volume, latent_density
from recovar.output import output as o


def _resolve_embedding_entries(no_z_regularization: bool) -> tuple[str, str, str]:
    if no_z_regularization:
        return "latent_coords_noreg", "latent_precision_noreg", "contrasts_noreg"
    return "latent_coords", "latent_precision", "contrasts"


def _load_embedding_components(po: o.PipelineOutput, zdim: int, no_z_regularization: bool):
    coords_entry, precision_entry, contrast_entry = _resolve_embedding_entries(no_z_regularization)
    if hasattr(po, "get_embedding_component"):
        contrasts = po.get_embedding_component(contrast_entry, zdim)
        zs = po.get_embedding_component(coords_entry, zdim)
        cov_zs = po.get_embedding_component(precision_entry, zdim)
    else:
        contrasts = po.get(contrast_entry)[zdim]
        zs = po.get(coords_entry)[zdim]
        cov_zs = po.get(precision_entry)[zdim]
    return (
        np.asarray(contrasts, dtype=np.float32),
        np.asarray(zs, dtype=np.float32),
        np.asarray(cov_zs, dtype=np.float32),
    )


def _compute_latent_distances(dataset, latent_point, zs, cov_zs, embedding_option: str):
    ndim = zs.shape[-1]
    latent_points = np.asarray(latent_point, dtype=np.float32).reshape(1, ndim)
    if embedding_option == "llh":
        log_likelihoods = latent_density.compute_latent_log_likelihood(latent_points, zs, cov_zs)[..., 0]
        heterogeneity_distances = log_likelihoods - np.min(log_likelihoods)
    elif embedding_option == "cov_dist":
        heterogeneity_distances = latent_density.compute_latent_quadratic_forms_in_batch(
            latent_points, zs, cov_zs
        )[..., 0]
    elif embedding_option == "dist":
        identity_cov = np.broadcast_to(np.eye(ndim, dtype=np.float32), cov_zs.shape)
        heterogeneity_distances = latent_density.compute_latent_log_likelihood(
            latent_points, zs, identity_cov
        )[..., 0]
    else:
        raise ValueError(f"Unknown embedding option: {embedding_option}")
    return dataset.split_halfset_array(heterogeneity_distances, per_particle=dataset.tilt_series_flag)


def _profile_one_volume(
    *,
    dataset,
    halfset_datasets,
    heterogeneity_distances,
    output_dir: Path,
    n_bins: int,
    bfactor: float,
    n_min_particles: int,
    maskrad_fraction: float,
    save_all_estimates: bool,
):
    locres_maskrad = dataset.grid_size * dataset.voxel_size / maskrad_fraction
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    from recovar.output.output_paths import VolumeOutputPaths
    vol_paths = VolumeOutputPaths(str(output_dir), "state", 0)
    heterogeneity_volume.make_volumes_kernel_estimate_local(
        heterogeneity_distances,
        dataset,
        vol_paths,
        -1,
        n_bins,
        bfactor,
        tau=None,
        n_min_particles=n_min_particles,
        locres_sampling=locres_maskrad,
        locres_maskrad=locres_maskrad,
        locres_edgwidth=0,
        upsampling_for_ests=1,
        use_mask_ests=False,
        grid_correct_ests=False,
        save_all_estimates=save_all_estimates,
        metric_used="locshellmost_likely",
        halfset_datasets=halfset_datasets,
    )
    return time.perf_counter() - t0


def _jax_trace_ctx(trace_dir: Path | None):
    if trace_dir is None:
        return contextlib.nullcontext()
    import jax.profiler

    trace_dir.mkdir(parents=True, exist_ok=True)
    return jax.profiler.trace(str(trace_dir), create_perfetto_link=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline_output", type=Path)
    parser.add_argument("--latent-points", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--zdim", type=int, default=None)
    parser.add_argument("--volume-index", type=int, default=0)
    parser.add_argument("--n-volumes", type=int, default=1)
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--bfactor", type=float, default=0.0)
    parser.add_argument("--n-min-particles", type=int, default=100)
    parser.add_argument("--maskrad-fraction", type=float, default=20.0)
    parser.add_argument("--embedding-option", default="cov_dist", choices=["cov_dist", "llh", "dist"])
    parser.add_argument("--no-z-regularization", action="store_true")
    parser.add_argument("--halfset-mode", choices=["direct", "subset", "auto"], default="auto")
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--trace-jax", action="store_true")
    parser.add_argument("--save-all-estimates", action="store_true")
    args = parser.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    po = o.PipelineOutput(str(args.pipeline_output.resolve()))
    dataset = po.get("lazy_dataset") if args.lazy else po.get("dataset")

    latent_points = np.loadtxt(args.latent_points)
    latent_points = np.atleast_2d(np.asarray(latent_points, dtype=np.float32))
    zdim = args.zdim if args.zdim is not None else int(latent_points.shape[-1])
    contrasts, zs, cov_zs = _load_embedding_components(po, zdim, args.no_z_regularization)
    embedding.set_contrasts_in_cryos(dataset, contrasts)
    if latent_points.shape[-1] != zdim:
        latent_points = latent_points[:, :zdim]
    selected_points = latent_points[args.volume_index : args.volume_index + args.n_volumes]
    if selected_points.size == 0:
        raise ValueError("No latent points selected for profiling.")

    halfset_construction_t0 = time.perf_counter()
    if args.halfset_mode == "direct":
        halfset_datasets = dataset.materialize_halfset_datasets(independent=True, lazy=args.lazy)
    elif args.halfset_mode == "subset":
        halfset_datasets = dataset.materialize_halfset_datasets(independent=False, lazy=args.lazy)
    else:
        halfset_datasets = _build_reweighted_halfset_datasets(dataset, lazy=args.lazy)
        if halfset_datasets is None:
            halfset_datasets = dataset.materialize_halfset_datasets(independent=False, lazy=args.lazy)
    halfset_construction_s = time.perf_counter() - halfset_construction_t0

    summary: dict[str, object] = {
        "pipeline_output": str(args.pipeline_output.resolve()),
        "latent_points": str(args.latent_points.resolve()),
        "zdim": zdim,
        "lazy": bool(args.lazy),
        "halfset_mode": args.halfset_mode,
        "halfset_construction_s": halfset_construction_s,
        "volumes": [],
    }

    trace_dir = outdir / "jax_trace" if args.trace_jax else None
    profile_path = outdir / "cprofile.prof"
    text_profile_path = outdir / "cprofile.txt"

    profiler = cProfile.Profile() if args.cprofile else None
    if profiler is not None:
        profiler.enable()

    try:
        with _jax_trace_ctx(trace_dir):
            for idx, latent_point in enumerate(selected_points, start=args.volume_index):
                volume_dir = outdir / f"volume_{idx:03d}"

                t_dist = time.perf_counter()
                heterogeneity_distances = _compute_latent_distances(
                    dataset, latent_point, zs, cov_zs, args.embedding_option
                )
                distance_s = time.perf_counter() - t_dist

                volume_s = _profile_one_volume(
                    dataset=dataset,
                    halfset_datasets=halfset_datasets,
                    heterogeneity_distances=heterogeneity_distances,
                    output_dir=volume_dir,
                    n_bins=args.n_bins,
                    bfactor=args.bfactor,
                    n_min_particles=args.n_min_particles,
                    maskrad_fraction=args.maskrad_fraction,
                    save_all_estimates=args.save_all_estimates,
                )
                summary["volumes"].append(
                    {
                        "volume_index": idx,
                        "distance_s": distance_s,
                        "volume_s": volume_s,
                    }
                )
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(profile_path))
            stats_stream = io.StringIO()
            pstats.Stats(profiler, stream=stats_stream).sort_stats("cumtime").print_stats(80)
            text_profile_path.write_text(stats_stream.getvalue())

    if trace_dir is not None and not any(trace_dir.iterdir()):
        shutil.rmtree(trace_dir, ignore_errors=True)

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
