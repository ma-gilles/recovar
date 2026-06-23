#!/usr/bin/env python3
"""Submit focused follow-up 3DFlex segmented-mesh variants.

This is a second-stage sweep for the nonuniform spike benchmark after the
initial full-solvent custom-segmentation mesh sweep.  It keeps the same full
solvent mask and Data Prep particles, then adds lower-complexity meshes and
weaker latent-centering/rigidity train variants that are useful when a custom
3DFlex mesh under-represents a localized motion.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.spike_fullatom_method_benchmark.submit_cryosparc_3dflex_segmented_moving_sweep import (  # noqa: E501
    DEFAULT_BENCH_ROOT,
    DEFAULT_NOTEBOOK,
    DEFAULT_PROJECT_DIR,
    DEFAULT_SOURCE_RUN,
    load_cryosparc_credentials,
)


@dataclass(frozen=True)
class MeshSpec:
    name: str
    segmentation: str | None
    tetra_num_cells: int | None
    fuse_list: str | None
    rigid_list: str | None
    desc: str
    existing_mesh_job: str | None = None


@dataclass(frozen=True)
class TrainSpec:
    name: str
    params: dict[str, Any]
    desc: str


def train_specs() -> list[TrainSpec]:
    return [
        TrainSpec(
            name="centerloose",
            params={"flex_K": 1, "flex_latent_prior_lam": 2.0},
            desc="K=1 with weaker latent centering.",
        ),
        TrainSpec(
            name="centerveryloose",
            params={"flex_K": 1, "flex_latent_prior_lam": 0.2},
            desc="K=1 with very weak latent centering.",
        ),
        TrainSpec(
            name="looserig_centerloose",
            params={
                "flex_K": 1,
                "flex_sv_lam": 0.1,
                "flex_latent_samp_std": 0.1,
                "flex_latent_prior_lam": 2.0,
            },
            desc="K=1, weaker spatial rigidity, lower latent noise, weaker centering.",
        ),
        TrainSpec(
            name="k2_centerloose",
            params={"flex_K": 2, "flex_latent_prior_lam": 2.0},
            desc="K=2 diagnostic with weaker latent centering.",
        ),
    ]


def mesh_specs(inputs: dict[str, Any]) -> list[MeshSpec]:
    specs: list[MeshSpec] = [
        MeshSpec(
            name="seg3_i10_chain_bodyrigid_tetra40_existing",
            segmentation=None,
            tetra_num_cells=None,
            fuse_list=None,
            rigid_list=None,
            existing_mesh_job="J583",
            desc="Existing interface10 chain/body-rigid tetra40 mesh.",
        ),
        MeshSpec(
            name="seg3_i10_chain_norigid_tetra40_existing",
            segmentation=None,
            tetra_num_cells=None,
            fuse_list=None,
            rigid_list=None,
            existing_mesh_job="J588",
            desc="Existing interface10 chain/no-rigid tetra40 mesh.",
        ),
    ]

    for tetra in (20, 12):
        specs.extend(
            [
                MeshSpec(
                    name=f"seg3_i10_chain_bodyrigid_tetra{tetra}",
                    segmentation="seg3_moving_interface10_body",
                    tetra_num_cells=tetra,
                    fuse_list="2>1,1>0",
                    rigid_list="2",
                    desc=f"Three-segment interface10 chain/body-rigid mesh, tetra={tetra}.",
                ),
                MeshSpec(
                    name=f"seg3_i10_chain_norigid_tetra{tetra}",
                    segmentation="seg3_moving_interface10_body",
                    tetra_num_cells=tetra,
                    fuse_list="2>1,1>0",
                    rigid_list=None,
                    desc=f"Three-segment interface10 chain/no-rigid mesh, tetra={tetra}.",
                ),
                MeshSpec(
                    name=f"seg2_i10_bodyrigid_tetra{tetra}",
                    segmentation="seg2_movingplusinterface10_body",
                    tetra_num_cells=tetra,
                    fuse_list="1>0",
                    rigid_list="1",
                    desc=f"Two-segment moving+interface/body-rigid mesh, tetra={tetra}.",
                ),
                MeshSpec(
                    name=f"seg2_i10_norigid_tetra{tetra}",
                    segmentation="seg2_movingplusinterface10_body",
                    tetra_num_cells=tetra,
                    fuse_list="1>0",
                    rigid_list=None,
                    desc=f"Two-segment moving+interface/no-rigid mesh, tetra={tetra}.",
                ),
            ]
        )

    missing = [
        spec.segmentation
        for spec in specs
        if spec.segmentation is not None
        and spec.segmentation not in inputs.get("segmentations", {})
    ]
    if missing:
        raise KeyError(f"Missing prepared segmentations: {sorted(set(missing))}")
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_BENCH_ROOT / "cryosparc_3dflex_segmented_moving_followup",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_BENCH_ROOT / "cryosparc_3dflex_segmented_moving_sweep" / "inputs",
    )
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--project", default="P587")
    parser.add_argument("--workspace", default="W14")
    parser.add_argument("--existing-flex-prep-job", default="J551")
    parser.add_argument("--existing-mask-import-job", default="J561")
    parser.add_argument("--existing-mask-import-output", default="imported_mask_1")
    parser.add_argument("--gpu-lane", default="48hrs-a100")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--mesh-mask-lowpass-a", type=float, default=10.0)
    parser.add_argument("--mesh-mask-threshold", type=float, default=0.5)
    parser.add_argument("--mesh-mask-dilate-a", type=float, default=3.0)
    parser.add_argument("--mesh-mask-pad-a", type=float, default=8.0)
    parser.add_argument("--rigidity-penalty-min", type=float, default=0.5)
    return parser.parse_args()


def queue(job: Any, lane: str) -> None:
    job.queue(lane=lane)


def main() -> None:
    try:
        from cryosparc.tools import CryoSPARC
    except ModuleNotFoundError as exc:
        raise RuntimeError("Run with: module load cryosparc-tools/5.0.3") from exc

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = args.input_dir / "segmented_moving_inputs_manifest.json"
    inputs = json.loads(inputs_path.read_text())

    creds = load_cryosparc_credentials(args.notebook)
    cs = CryoSPARC(
        license=str(creds["license"]),
        email=str(creds["email"]),
        password=str(creds["password"]),
        host=str(creds["host"]),
        base_port=int(creds["base_port"]),
    )
    cs.test_connection()

    jobs: dict[str, str] = {}
    model_records: list[dict[str, Any]] = []

    for mesh in mesh_specs(inputs):
        if mesh.existing_mesh_job is None:
            assert mesh.segmentation is not None
            assert mesh.tetra_num_cells is not None
            seg_path = Path(inputs["segmentations"][mesh.segmentation]["path"])
            mesh_params: dict[str, Any] = {
                "mask_in_lowpass_A": args.mesh_mask_lowpass_a,
                "mask_in_threshold_level": args.mesh_mask_threshold,
                "mask_dilate_A": args.mesh_mask_dilate_a,
                "mask_pad_A": args.mesh_mask_pad_a,
                "tetra_num_cells": mesh.tetra_num_cells,
                "tetra_segments_path": str(seg_path),
                "tetra_segments_fuse_list": mesh.fuse_list,
            }
            if mesh.rigid_list is not None:
                mesh_params["tetra_rigid_list"] = mesh.rigid_list
            if args.rigidity_penalty_min is not None:
                mesh_params["rigidity_penalty_min"] = args.rigidity_penalty_min

            mesh_job = cs.create_job(
                args.project,
                args.workspace,
                "flex_meshprep",
                connections={
                    "volume": (args.existing_flex_prep_job, "volume"),
                    "mask": (args.existing_mask_import_job, args.existing_mask_import_output),
                },
                params=mesh_params,
                title=f"3DFlex followup meshprep {mesh.name}",
                desc=mesh.desc,
            )
            mesh_job_uid = mesh_job.uid
            jobs[f"meshprep_{mesh.name}"] = mesh_job_uid
            if args.submit:
                queue(mesh_job, args.gpu_lane)
        else:
            seg_path = None
            mesh_job_uid = mesh.existing_mesh_job
            jobs[f"meshprep_{mesh.name}"] = mesh_job_uid

        for train in train_specs():
            model_name = f"{mesh.name}_{train.name}"
            train_job = cs.create_job(
                args.project,
                args.workspace,
                "flex_train",
                connections={
                    "particles": (args.existing_flex_prep_job, "particles"),
                    "flex_mesh": (mesh_job_uid, "flex_mesh"),
                },
                params=train.params,
                title=f"3DFlex followup train {model_name}",
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
                title=f"3DFlex followup highres {model_name}",
                desc=f"Highres reconstruction for followup 3DFlex model {model_name}.",
            )
            jobs[f"train_{model_name}"] = train_job.uid
            jobs[f"highres_{model_name}"] = highres_job.uid
            if args.submit:
                queue(train_job, args.gpu_lane)
                queue(highres_job, args.gpu_lane)
            model_records.append(
                {
                    "model": model_name,
                    "mesh_variant": mesh.name,
                    "mesh_desc": mesh.desc,
                    "segmentation": mesh.segmentation,
                    "segmentation_path": str(seg_path) if seg_path is not None else None,
                    "fuse_list": mesh.fuse_list,
                    "rigid_list": mesh.rigid_list,
                    "tetra_num_cells": mesh.tetra_num_cells,
                    "meshprep_job": mesh_job_uid,
                    "train_variant": train.name,
                    "train_desc": train.desc,
                    "train_params": train.params,
                    "train_job": train_job.uid,
                    "highres_job": highres_job.uid,
                    "k": int(train.params.get("flex_K", 1)),
                    "mask_mode": "full_solvent_custom_segmentation_followup",
                }
            )

    manifest = {
        "project_uid": args.project,
        "workspace_uid": args.workspace,
        "project_dir": str(args.project_dir.resolve()),
        "source_run": str(args.source_run.resolve()),
        "bench_root": str(args.bench_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "input_manifest": str(inputs_path.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "existing_flex_prep_job": args.existing_flex_prep_job,
        "existing_mask_import_job": args.existing_mask_import_job,
        "existing_mask_import_output": args.existing_mask_import_output,
        "gpu_lane": args.gpu_lane,
        "submitted": bool(args.submit),
        "jobs": jobs,
        "models": model_records,
    }
    manifest_path = args.out_dir / "cryosparc_3dflex_noise1_b80_n00300000_segmented_followup_jobs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
