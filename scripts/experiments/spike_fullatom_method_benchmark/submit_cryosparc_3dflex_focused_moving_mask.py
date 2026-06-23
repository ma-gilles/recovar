#!/usr/bin/env python3
"""Submit a focused 3DFlex branch using only the localized moving mask.

This script reuses the completed corrected-sign 300k CryoSPARC import,
homogeneous reconstruction, and 3DFlex Data Prep jobs. It creates/imports a
box-128 moving-region mask and starts a new Mesh Prep -> Train K=1 ->
Reconstruct branch without touching the previous default-mask outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from cryosparc.tools import CryoSPARC

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.spike_fullatom_method_benchmark.submit_cryosparc_3dflex_100k import (  # noqa: E501
    DEFAULT_NOTEBOOK,
    first_output_name,
    load_cryosparc_credentials,
)


DEFAULT_SOURCE_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_nonuniform_B70_noise1_b80_300k_20260604/"
    "n00300000/runs/n00300000_seed0000"
)
DEFAULT_BENCH_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_nonuniform_B70_noise1_b80_300k_20260604"
)


def load_mrc(path: Path) -> tuple[np.ndarray, float]:
    import mrcfile

    with mrcfile.open(path, permissive=True) as handle:
        data = np.asarray(handle.data, dtype=np.float32)
        voxel_size = float(handle.voxel_size.x)
    return data, voxel_size


def write_mrc(path: Path, data: np.ndarray, voxel_size: float) -> None:
    import mrcfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.asarray(data, dtype=np.float32))
        handle.voxel_size = voxel_size
        handle.header.origin = (0.0, 0.0, 0.0)
        handle.update_header_stats()


def block_max_downsample(mask: np.ndarray, factor: int) -> np.ndarray:
    if any(size % factor for size in mask.shape):
        raise ValueError(f"Mask shape {mask.shape} is not divisible by {factor}")
    new_shape = tuple(size // factor for size in mask.shape)
    reshaped = mask.reshape(
        new_shape[0],
        factor,
        new_shape[1],
        factor,
        new_shape[2],
        factor,
    )
    return reshaped.max(axis=(1, 3, 5)).astype(np.float32)


def prepare_focused_mask(
    *,
    source_mask: Path,
    out_path: Path,
    target_box: int,
    hard_threshold: float,
    existing_voxel_size: float | None,
    overwrite: bool,
) -> dict[str, object]:
    if out_path.exists() and not overwrite:
        try:
            data, voxel_size = load_mrc(out_path)
            return {
                "path": str(out_path),
                "created": False,
                "shape": list(data.shape),
                "voxel_size_A": voxel_size,
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "sum": float(data.sum()),
                "voxels_gt_0p5": int((data > 0.5).sum()),
            }
        except ModuleNotFoundError:
            if existing_voxel_size is None:
                raise RuntimeError(
                    "mrcfile is not available in this Python. Pass "
                    "--prepared-mask-voxel-size for an existing prepared mask, "
                    "or create the mask with a Python environment that has mrcfile."
                ) from None
            return {
                "path": str(out_path),
                "created": False,
                "voxel_size_A": float(existing_voxel_size),
                "stats_unavailable": "mrcfile_missing_in_submission_python",
            }

    mask, voxel_size = load_mrc(source_mask)
    if len(set(mask.shape)) != 1:
        raise ValueError(f"Expected cubic source mask, got {mask.shape}")
    factor = mask.shape[0] // target_box
    if factor < 1 or mask.shape[0] != target_box * factor:
        raise ValueError(f"Cannot downsample {mask.shape} to box {target_box}")
    focused = block_max_downsample((mask >= hard_threshold).astype(np.float32), factor)
    write_mrc(out_path, focused, voxel_size * factor)
    return {
        "path": str(out_path),
        "created": True,
        "source": str(source_mask),
        "source_shape": list(mask.shape),
        "source_voxel_size_A": voxel_size,
        "target_box": target_box,
        "downsample_factor": factor,
        "hard_threshold": hard_threshold,
        "shape": list(focused.shape),
        "voxel_size_A": voxel_size * factor,
        "min": float(focused.min()),
        "max": float(focused.max()),
        "mean": float(focused.mean()),
        "sum": float(focused.sum()),
        "voxels_gt_0p5": int((focused > 0.5).sum()),
    }


def queue(job, lane: str) -> None:
    job.queue(lane=lane)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--project", default="P587")
    parser.add_argument("--workspace", default="W14")
    parser.add_argument("--project-dir", type=Path, default=Path("/projects/CRYOEM/singerlab/mg6942/CS-testres"))
    parser.add_argument("--source-mask", type=Path, default=None)
    parser.add_argument("--existing-flex-prep-job", default="J551")
    parser.add_argument("--existing-import-job", default="J549")
    parser.add_argument("--existing-homo-job", default="J550")
    parser.add_argument("--import-lane", default="24hrs")
    parser.add_argument("--gpu-lane", default="48hrs-a100")
    parser.add_argument("--target-box", type=int, default=128)
    parser.add_argument("--mask-hard-threshold", type=float, default=0.5)
    parser.add_argument("--prepared-mask-voxel-size", type=float, default=None)
    parser.add_argument("--mask-in-lowpass-a", type=float, default=10.0)
    parser.add_argument("--mask-in-threshold-level", type=float, default=0.5)
    parser.add_argument("--mask-dilate-a", type=float, default=5.0)
    parser.add_argument("--mask-pad-a", type=float, default=5.0)
    parser.add_argument("--tetra-num-cells", type=int, default=40)
    parser.add_argument("--flex-k", type=int, default=1)
    parser.add_argument("--manifest-name", default="cryosparc_3dflex_noise1_b80_n00300000_focused_movingmask_k1_jobs.json")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--overwrite-mask", action="store_true")
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    bench_root = args.bench_root.resolve()
    source_mask = (
        args.source_mask.resolve()
        if args.source_mask is not None
        else source_run / "05_masks" / "focus_mask_moving.mrc"
    )
    if not source_mask.exists():
        raise FileNotFoundError(source_mask)

    out_dir = bench_root / "cryosparc_3dflex_focused_moving_mask"
    input_dir = out_dir / "inputs"
    mask_path = input_dir / "focus_mask_moving_box128_hard_blockmax.mrc"
    mask_info = prepare_focused_mask(
        source_mask=source_mask,
        out_path=mask_path,
        target_box=args.target_box,
        hard_threshold=args.mask_hard_threshold,
        existing_voxel_size=args.prepared_mask_voxel_size,
        overwrite=args.overwrite_mask,
    )

    creds = load_cryosparc_credentials(args.notebook)
    cs = CryoSPARC(
        license=str(creds["license"]),
        email=str(creds["email"]),
        password=str(creds["password"]),
        host=str(creds["host"]),
        base_port=int(creds["base_port"]),
    )
    cs.test_connection()

    jobs: dict[str, object] = {}
    jobs["import_focused_mask"] = cs.create_job(
        args.project,
        args.workspace,
        "import_volumes",
        params={
            "volume_blob_path": str(mask_path),
            "volume_out_name": "mask",
            "volume_psize": float(mask_info["voxel_size_A"]),
        },
        title="Import spike moving-region 3DFlex mesh mask box128",
    )

    import_mask_output = "mask"
    if args.submit:
        queue(jobs["import_focused_mask"], args.import_lane)
        status = jobs["import_focused_mask"].wait_for_done(error_on_incomplete=True)
        print(f"import_focused_mask {jobs['import_focused_mask'].uid} finished with {status}")
        import_mask_output = first_output_name(jobs["import_focused_mask"], "mask")

    jobs["flex_meshprep_focused"] = cs.create_job(
        args.project,
        args.workspace,
        "flex_meshprep",
        connections={
            "volume": (args.existing_flex_prep_job, "volume"),
            "mask": (jobs["import_focused_mask"].uid, import_mask_output),
        },
        params={
            "mask_in_lowpass_A": args.mask_in_lowpass_a,
            "mask_in_threshold_level": args.mask_in_threshold_level,
            "mask_dilate_A": args.mask_dilate_a,
            "mask_pad_A": args.mask_pad_a,
            "tetra_num_cells": args.tetra_num_cells,
        },
        title="3DFlex mesh prep focused moving mask noise1/B80 n=300000",
    )
    jobs["flex_train_focused_k1"] = cs.create_job(
        args.project,
        args.workspace,
        "flex_train",
        connections={
            "particles": (args.existing_flex_prep_job, "particles"),
            "flex_mesh": (jobs["flex_meshprep_focused"].uid, "flex_mesh"),
        },
        params={"flex_K": args.flex_k},
        title=f"3DFlex train focused moving mask noise1/B80 n=300000 K={args.flex_k}",
    )
    jobs["flex_highres_focused_k1"] = cs.create_job(
        args.project,
        args.workspace,
        "flex_highres",
        connections={
            "flex_model": (jobs["flex_train_focused_k1"].uid, "flex_model"),
            "particles": (args.existing_flex_prep_job, "particles"),
        },
        title=f"3DFlex reconstruction focused moving mask noise1/B80 n=300000 K={args.flex_k}",
    )

    manifest = {
        "project_uid": args.project,
        "workspace_uid": args.workspace,
        "project_dir": str(args.project_dir.resolve()),
        "source_run": str(source_run),
        "bench_root": str(bench_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_images": 300000,
        "particle_sign": 1,
        "consensus_source": "existing_homo_reconstruct",
        "existing_import_job": args.existing_import_job,
        "existing_homo_reconstruct_job": args.existing_homo_job,
        "existing_flex_prep_job": args.existing_flex_prep_job,
        "model": "focused_movingmask_k1",
        "train_job": jobs["flex_train_focused_k1"].uid,
        "highres_job": jobs["flex_highres_focused_k1"].uid,
        "k": args.flex_k,
        "mask_mode": "focused_moving_region_mask_box128_hard_blockmax",
        "source_moving_mask": str(source_mask),
        "prepared_mask": mask_info,
        "meshprep_params": {
            "mask_in_lowpass_A": args.mask_in_lowpass_a,
            "mask_in_threshold_level": args.mask_in_threshold_level,
            "mask_dilate_A": args.mask_dilate_a,
            "mask_pad_A": args.mask_pad_a,
            "tetra_num_cells": args.tetra_num_cells,
        },
        "train_params": {"flex_K": args.flex_k},
        "import_lane": args.import_lane,
        "gpu_lane": args.gpu_lane,
        "submitted": bool(args.submit),
        "jobs": {name: job.uid for name, job in jobs.items()},
        "dynamic_outputs": {
            "import_focused_mask": import_mask_output,
            "focused_mesh": "flex_mesh",
            "flex_train_k1": "flex_model",
            "flex_highres_k1": "volume_flex",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

    if args.submit:
        queue(jobs["flex_meshprep_focused"], args.gpu_lane)
        queue(jobs["flex_train_focused_k1"], args.gpu_lane)
        queue(jobs["flex_highres_focused_k1"], args.gpu_lane)
        print(f"Queued focused 3DFlex branch; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
