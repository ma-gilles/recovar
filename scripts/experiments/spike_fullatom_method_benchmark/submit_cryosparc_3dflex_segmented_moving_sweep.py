#!/usr/bin/env python3
"""Submit 3DFlex custom-segmentation variants for localized spike motion.

The focused-mask-only branch is useful diagnostically, but CryoSPARC's 3DFlex
guidance generally points to keeping the full particle in the solvent mask and
using custom mesh segmentation / rigid segments to encode localized motion.

This submitter prepares several downsampled segmentation MRCs at the 3DFlex
Data Prep box size:

* moving core: voxels inside the moving/focus mask
* interface/buffer: a shell around the moving core
* body: the rest of the particle solvent mask
* solvent: -1

It then queues Mesh Prep -> Train -> Highres branches for several topology and
training-regularization variants.  Run preparation with the RECOVAR/pixi
environment; run submission with the cryosparc-tools module.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_NOTEBOOK = Path("/home/mg6942/recovar/20231229_3dflex.ipynb")
DEFAULT_SOURCE_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_nonuniform_B70_noise1_b80_300k_20260604/"
    "n00300000/runs/n00300000_seed0000"
)
DEFAULT_BENCH_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_nonuniform_B70_noise1_b80_300k_20260604"
)
DEFAULT_PROJECT_DIR = Path("/projects/CRYOEM/singerlab/mg6942/CS-testres")


def load_cryosparc_credentials(notebook: Path) -> dict[str, object]:
    """Read the existing local notebook without printing secrets."""
    nb = json.loads(notebook.read_text())
    values: dict[str, object] = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        for name in ("license", "email", "password", "host", "base_port", "project_num"):
            match = re.search(rf"^\s*{name}\s*=\s*(.+?)\s*$", source, flags=re.M)
            if not match:
                continue
            try:
                values[name] = ast.literal_eval(match.group(1))
            except Exception:
                pass
    values.setdefault("host", "della-cryoem")
    values.setdefault("base_port", 39000)
    values.setdefault("project_num", "P587")
    required = ("license", "email", "password", "host", "base_port", "project_num")
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"Missing CryoSPARC credential fields in {notebook}: {missing}")
    return values


def first_output_name(job: Any, output_type: str) -> str:
    job.refresh()
    for name, output in job.outputs.items():
        if output.type == output_type:
            return name
    raise RuntimeError(
        f"Job {job.uid} has no {output_type!r} output. Current outputs: {list(job.outputs)}"
    )


@dataclass(frozen=True)
class MeshVariant:
    name: str
    segmentation: str
    fuse_list: str | None
    rigid_list: str | None
    desc: str


@dataclass(frozen=True)
class TrainVariant:
    name: str
    params: dict[str, Any]
    desc: str


def load_mrc(path: Path) -> tuple[np.ndarray, float]:
    import mrcfile

    with mrcfile.open(path, permissive=True) as handle:
        return np.asarray(handle.data), float(handle.voxel_size.x)


def write_mrc(path: Path, data: np.ndarray, voxel_size: float) -> None:
    import mrcfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.asarray(data))
        handle.voxel_size = voxel_size
        handle.header.origin = (0.0, 0.0, 0.0)
        handle.update_header_stats()


def block_reduce(mask: np.ndarray, factor: int, reducer: str) -> np.ndarray:
    if any(size % factor for size in mask.shape):
        raise ValueError(f"Shape {mask.shape} is not divisible by factor {factor}")
    new_shape = tuple(size // factor for size in mask.shape)
    reshaped = mask.reshape(new_shape[0], factor, new_shape[1], factor, new_shape[2], factor)
    if reducer == "max":
        return reshaped.max(axis=(1, 3, 5))
    if reducer == "mean":
        return reshaped.mean(axis=(1, 3, 5))
    raise ValueError(reducer)


def ball_structure(radius: int) -> np.ndarray:
    coords = np.arange(-radius, radius + 1)
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    return (x * x + y * y + z * z) <= radius * radius


def solidify(binary: np.ndarray, close_radius: int = 2, dilate_radius: int = 0) -> np.ndarray:
    from scipy import ndimage

    out = np.asarray(binary, dtype=bool)
    if close_radius > 0:
        structure = ball_structure(close_radius)
        out = ndimage.binary_closing(out, structure=structure)
    out = ndimage.binary_fill_holes(out)
    labels, n_labels = ndimage.label(out)
    if n_labels > 0:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        out = labels == int(counts.argmax())
    if dilate_radius > 0:
        out = ndimage.binary_dilation(out, structure=ball_structure(dilate_radius))
    return np.asarray(out, dtype=bool)


def cosine_soften(binary: np.ndarray, edge_voxels: float) -> np.ndarray:
    from scipy import ndimage

    binary = np.asarray(binary, dtype=bool)
    distance = ndimage.distance_transform_edt(~binary)
    soft = np.zeros(binary.shape, dtype=np.float32)
    soft[binary] = 1.0
    edge = (~binary) & (distance <= edge_voxels)
    soft[edge] = 0.5 * (1.0 + np.cos(np.pi * distance[edge] / edge_voxels))
    return soft


def prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    from scipy import ndimage

    source_run = args.source_run.resolve()
    out_dir = args.out_dir.resolve()
    input_dir = out_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    moving_path = args.moving_mask.resolve() if args.moving_mask else source_run / "05_masks/focus_mask_moving.mrc"
    solvent_path = args.solvent_mask.resolve() if args.solvent_mask else source_run / "05_masks/volume_mask_union.mrc"
    moving_256, moving_apix = load_mrc(moving_path)
    solvent_256, solvent_apix = load_mrc(solvent_path)
    if moving_256.shape != solvent_256.shape:
        raise ValueError(f"Moving mask shape {moving_256.shape} != solvent mask shape {solvent_256.shape}")
    if moving_256.shape[0] % args.target_box:
        raise ValueError(f"Cannot downsample {moving_256.shape} to {args.target_box}")
    factor = moving_256.shape[0] // args.target_box
    target_apix = moving_apix * factor

    moving_core = block_reduce((moving_256 >= args.moving_threshold).astype(np.float32), factor, "max") > 0.5
    moving_core = solidify(moving_core, close_radius=args.close_radius_voxels, dilate_radius=args.moving_dilate_voxels)

    solvent = block_reduce((solvent_256 >= args.solvent_threshold).astype(np.float32), factor, "max") > 0.5
    solvent = solidify(solvent, close_radius=args.close_radius_voxels, dilate_radius=args.solvent_dilate_voxels)
    solvent = solvent | moving_core
    solvent_soft = cosine_soften(solvent, args.solvent_soft_edge_voxels)
    solvent_out = input_dir / "full_solvent_mask_box128_solid_cosine_soft.mrc"
    write_mrc(solvent_out, solvent_soft.astype(np.float32), target_apix)

    distances = ndimage.distance_transform_edt(~moving_core)
    input_records: dict[str, Any] = {
        "moving_mask_source": str(moving_path),
        "solvent_mask_source": str(solvent_path),
        "source_shape": list(moving_256.shape),
        "source_apix_A": moving_apix,
        "target_box": args.target_box,
        "target_apix_A": target_apix,
        "downsample_factor": factor,
        "full_solvent_mask": str(solvent_out),
        "moving_threshold": args.moving_threshold,
        "solvent_threshold": args.solvent_threshold,
        "solvent_voxels": int(solvent.sum()),
        "moving_core_voxels": int((moving_core & solvent).sum()),
        "segmentations": {},
    }

    def write_seg(name: str, seg: np.ndarray) -> Path:
        path = input_dir / f"{name}_segments_box128.mrc"
        write_mrc(path, seg.astype(np.int16), target_apix)
        ids, counts = np.unique(seg, return_counts=True)
        input_records["segmentations"][name] = {
            "path": str(path),
            "ids": {str(int(i)): int(c) for i, c in zip(ids, counts)},
        }
        return path

    for buffer_voxels in args.buffer_voxels:
        moving = moving_core & solvent
        interface = (distances <= buffer_voxels) & solvent & ~moving
        body = solvent & ~moving & ~interface
        seg = np.full(solvent.shape, -1, dtype=np.int16)
        seg[moving] = 0
        seg[interface] = 1
        seg[body] = 2
        write_seg(f"seg3_moving_interface{buffer_voxels}_body", seg)

        moving_plus_buffer = (moving | interface) & solvent
        body2 = solvent & ~moving_plus_buffer
        seg2 = np.full(solvent.shape, -1, dtype=np.int16)
        seg2[moving_plus_buffer] = 0
        seg2[body2] = 1
        write_seg(f"seg2_movingplusinterface{buffer_voxels}_body", seg2)

    manifest_path = input_dir / "segmented_moving_inputs_manifest.json"
    input_records["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(input_records, indent=2) + "\n")
    return input_records


def mesh_variants(buffer_values: list[int]) -> list[MeshVariant]:
    variants: list[MeshVariant] = []
    for buffer_voxels in buffer_values:
        seg3 = f"seg3_moving_interface{buffer_voxels}_body"
        variants.extend(
            [
                MeshVariant(
                    name=f"{seg3}_chain_bodyrigid",
                    segmentation=seg3,
                    fuse_list="2>1,1>0",
                    rigid_list="2",
                    desc="moving core flexible, interface buffer flexible, rigid body fused through interface",
                ),
                MeshVariant(
                    name=f"{seg3}_chain_norigid",
                    segmentation=seg3,
                    fuse_list="2>1,1>0",
                    rigid_list=None,
                    desc="moving core, interface, and body fused through interface; no fully rigid segment",
                ),
                MeshVariant(
                    name=f"{seg3}_star_bodyrigid",
                    segmentation=seg3,
                    fuse_list="2>0,2>1",
                    rigid_list="2",
                    desc="body is the fusion hub for moving core and interface; body marked rigid",
                ),
                MeshVariant(
                    name=f"{seg3}_star_norigid",
                    segmentation=seg3,
                    fuse_list="2>0,2>1",
                    rigid_list=None,
                    desc="body is the fusion hub for moving core and interface; no fully rigid segment",
                ),
            ]
        )
    return variants


def train_variants() -> list[TrainVariant]:
    return [
        TrainVariant(
            name="default",
            params={"flex_K": 1},
            desc="CryoSPARC default K=1 training parameters",
        ),
        TrainVariant(
            name="centerloose",
            params={"flex_K": 1, "flex_latent_prior_lam": 2.0},
            desc="weaker latent centering for localized motion with narrow default latents",
        ),
        TrainVariant(
            name="lowrig_lownoise",
            params={"flex_K": 1, "flex_sv_lam": 0.5, "flex_latent_samp_std": 0.1},
            desc="weaker rigidity and lower latent noise for high-SNR localized one-dimensional motion",
        ),
        TrainVariant(
            name="lowrig_lownoise_centerloose",
            params={
                "flex_K": 1,
                "flex_sv_lam": 0.5,
                "flex_latent_samp_std": 0.1,
                "flex_latent_prior_lam": 2.0,
            },
            desc=(
                "weaker rigidity, lower latent noise, and weaker latent centering "
                "for high-SNR localized one-dimensional motion"
            ),
        ),
    ]


def queue(job: Any, lane: str) -> None:
    job.queue(lane=lane)


def submit_jobs(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from cryosparc.tools import CryoSPARC
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "cryosparc-tools is required for --submit. Run with: "
            "module load cryosparc-tools/5.0.3"
        ) from exc

    out_dir = args.out_dir.resolve()
    input_manifest = out_dir / "inputs/segmented_moving_inputs_manifest.json"
    if not input_manifest.exists():
        raise FileNotFoundError(f"Run --prepare first; missing {input_manifest}")
    inputs = json.loads(input_manifest.read_text())

    creds = load_cryosparc_credentials(args.notebook)
    cs = CryoSPARC(
        license=str(creds["license"]),
        email=str(creds["email"]),
        password=str(creds["password"]),
        host=str(creds["host"]),
        base_port=int(creds["base_port"]),
    )
    cs.test_connection()

    jobs: dict[str, Any] = {}
    if args.existing_mask_import_job:
        import_mask_job_uid = args.existing_mask_import_job
        import_mask_output = args.existing_mask_import_output
    else:
        jobs["import_full_solvent_mask"] = cs.create_job(
            args.project,
            args.workspace,
            "import_volumes",
            params={
                "volume_blob_path": str(inputs["full_solvent_mask"]),
                "volume_out_name": "mask",
                "volume_psize": float(inputs["target_apix_A"]),
            },
            title="Import full solvent mask for segmented 3DFlex sweep",
        )
        import_mask_job_uid = jobs["import_full_solvent_mask"].uid
        import_mask_output = "mask"
        if args.submit:
            queue(jobs["import_full_solvent_mask"], args.import_lane)
            status = jobs["import_full_solvent_mask"].wait_for_done(error_on_incomplete=True)
            print(f"import_full_solvent_mask {jobs['import_full_solvent_mask'].uid} finished with {status}")
            import_mask_output = first_output_name(jobs["import_full_solvent_mask"], "mask")

    model_records: list[dict[str, Any]] = []
    exclude_mesh_names = set(args.exclude_mesh_names or [])
    for mesh in mesh_variants(args.buffer_voxels):
        if mesh.name in exclude_mesh_names:
            continue
        seg_path = Path(inputs["segmentations"][mesh.segmentation]["path"])
        mesh_params: dict[str, Any] = {
            "mask_in_lowpass_A": args.mask_in_lowpass_a,
            "mask_in_threshold_level": args.mask_in_threshold_level,
            "mask_dilate_A": args.mask_dilate_a,
            "mask_pad_A": args.mask_pad_a,
            "tetra_num_cells": args.tetra_num_cells,
            "tetra_segments_path": str(seg_path),
        }
        if mesh.fuse_list is not None:
            mesh_params["tetra_segments_fuse_list"] = mesh.fuse_list
        if mesh.rigid_list is not None:
            mesh_params["tetra_rigid_list"] = mesh.rigid_list
        if args.rigidity_penalty_min is not None:
            mesh_params["rigidity_penalty_min"] = args.rigidity_penalty_min
        if args.stiffen_low_density:
            mesh_params["rigidity_penalty_stiffen_low_density"] = True

        mesh_job = cs.create_job(
            args.project,
            args.workspace,
            "flex_meshprep",
            connections={
                "volume": (args.existing_flex_prep_job, "volume"),
                "mask": (import_mask_job_uid, import_mask_output),
            },
            params=mesh_params,
            title=f"3DFlex meshprep segmented {mesh.name}",
            desc=mesh.desc,
        )
        jobs[f"meshprep_{mesh.name}"] = mesh_job
        if args.submit:
            queue(mesh_job, args.gpu_lane)

        for train in train_variants():
            model_name = f"{mesh.name}_{train.name}"
            train_job = cs.create_job(
                args.project,
                args.workspace,
                "flex_train",
                connections={
                    "particles": (args.existing_flex_prep_job, "particles"),
                    "flex_mesh": (mesh_job.uid, "flex_mesh"),
                },
                params=train.params,
                title=f"3DFlex train segmented {model_name}",
                desc=f"{mesh.desc}; {train.desc}",
            )
            highres_job = cs.create_job(
                args.project,
                args.workspace,
                "flex_highres",
                connections={
                    "flex_model": (train_job.uid, "flex_model"),
                    "particles": (args.existing_flex_prep_job, "particles"),
                },
                params={
                    "flex_do_noflex_recon": True,
                    "compute_use_ssd": True,
                    "compute_all_particles_in_ram": False,
                },
                title=f"3DFlex highres segmented {model_name}",
                desc=f"Highres reconstruction for segmented 3DFlex model {model_name}.",
            )
            jobs[f"train_{model_name}"] = train_job
            jobs[f"highres_{model_name}"] = highres_job
            if args.submit:
                queue(train_job, args.gpu_lane)
                queue(highres_job, args.gpu_lane)
            model_records.append(
                {
                    "model": model_name,
                    "mesh_variant": mesh.name,
                    "mesh_desc": mesh.desc,
                    "segmentation": mesh.segmentation,
                    "segmentation_path": str(seg_path),
                    "fuse_list": mesh.fuse_list,
                    "rigid_list": mesh.rigid_list,
                    "train_variant": train.name,
                    "train_desc": train.desc,
                    "train_params": train.params,
                    "meshprep_job": mesh_job.uid,
                    "train_job": train_job.uid,
                    "highres_job": highres_job.uid,
                    "k": int(train.params.get("flex_K", 1)),
                    "mask_mode": "full_solvent_custom_segmentation",
                }
            )

    manifest = {
        "project_uid": args.project,
        "workspace_uid": args.workspace,
        "project_dir": str(args.project_dir.resolve()),
        "source_run": str(args.source_run.resolve()),
        "bench_root": str(args.bench_root.resolve()),
        "out_dir": str(out_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_images": 300000,
        "particle_sign": 1,
        "existing_import_job": args.existing_import_job,
        "existing_homo_reconstruct_job": args.existing_homo_job,
        "existing_flex_prep_job": args.existing_flex_prep_job,
        "inputs": inputs,
        "meshprep_common_params": {
            "mask_in_lowpass_A": args.mask_in_lowpass_a,
            "mask_in_threshold_level": args.mask_in_threshold_level,
            "mask_dilate_A": args.mask_dilate_a,
            "mask_pad_A": args.mask_pad_a,
            "tetra_num_cells": args.tetra_num_cells,
            "rigidity_penalty_min": args.rigidity_penalty_min,
            "rigidity_penalty_stiffen_low_density": bool(args.stiffen_low_density),
        },
        "import_lane": args.import_lane,
        "gpu_lane": args.gpu_lane,
        "submitted": bool(args.submit),
        "jobs": {name: job.uid for name, job in jobs.items()},
        "models": model_records,
    }
    manifest_path = out_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--project", default="P587")
    parser.add_argument("--workspace", default="W14")
    parser.add_argument("--existing-flex-prep-job", default="J551")
    parser.add_argument("--existing-import-job", default="J549")
    parser.add_argument("--existing-homo-job", default="J550")
    parser.add_argument("--moving-mask", type=Path, default=None)
    parser.add_argument("--solvent-mask", type=Path, default=None)
    parser.add_argument("--target-box", type=int, default=128)
    parser.add_argument("--moving-threshold", type=float, default=0.5)
    parser.add_argument("--solvent-threshold", type=float, default=0.05)
    parser.add_argument("--buffer-voxels", type=int, nargs="+", default=[6, 10])
    parser.add_argument("--close-radius-voxels", type=int, default=2)
    parser.add_argument("--moving-dilate-voxels", type=int, default=1)
    parser.add_argument("--solvent-dilate-voxels", type=int, default=2)
    parser.add_argument("--solvent-soft-edge-voxels", type=float, default=4.0)
    parser.add_argument("--mask-in-lowpass-a", type=float, default=10.0)
    parser.add_argument("--mask-in-threshold-level", type=float, default=0.5)
    parser.add_argument("--mask-dilate-a", type=float, default=3.0)
    parser.add_argument("--mask-pad-a", type=float, default=8.0)
    parser.add_argument("--tetra-num-cells", type=int, default=40)
    parser.add_argument("--rigidity-penalty-min", type=float, default=0.5)
    parser.add_argument("--stiffen-low-density", action="store_true")
    parser.add_argument("--import-lane", default="24hrs")
    parser.add_argument("--gpu-lane", default="48hrs-a100")
    parser.add_argument("--existing-mask-import-job", default=None)
    parser.add_argument("--existing-mask-import-output", default="imported_mask_1")
    parser.add_argument("--exclude-mesh-names", nargs="*", default=[])
    parser.add_argument("--manifest-name", default="cryosparc_3dflex_noise1_b80_n00300000_segmented_moving_sweep_jobs.json")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    args.source_run = args.source_run.resolve()
    args.bench_root = args.bench_root.resolve()
    args.out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else args.bench_root / "cryosparc_3dflex_segmented_moving_sweep"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare:
        summary = prepare_inputs(args)
        print(json.dumps(summary, indent=2))
    if args.submit:
        submit_jobs(args)
    if not args.prepare and not args.submit:
        raise SystemExit("Pass --prepare and/or --submit")


if __name__ == "__main__":
    main()
