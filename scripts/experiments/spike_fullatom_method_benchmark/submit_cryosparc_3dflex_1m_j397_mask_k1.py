#!/usr/bin/env python3
"""Submit the 1M-image 3DFlex K=1 pipeline matching the best 100k branch."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cryosparc.tools import CryoSPARC

from scripts.experiments.spike_fullatom_method_benchmark.submit_cryosparc_3dflex_100k import (
    DEFAULT_NOTEBOOK,
    first_output_name,
    load_cryosparc_credentials,
)


DEFAULT_SOURCE_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/"
    "n01000000/runs/n01000000_seed0000"
)
DEFAULT_BENCH_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sanity_1m_noise10_b100_20260530"
)


def queue(job, lane: str) -> None:
    job.queue(lane=lane)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--project", default="P587")
    parser.add_argument("--workspace", default="W13")
    parser.add_argument("--import-lane", default="24hrs")
    parser.add_argument("--gpu-lane", default="48hrs-a100")
    parser.add_argument("--source-mask-job", default="J409")
    parser.add_argument("--manifest-name", default="cryosparc_3dflex_1m_j397_mask_k1_jobs.json")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    dataset = source_run / "03_dataset"
    bench_root = args.bench_root.resolve()
    out_dir = bench_root / "cryosparc_3dflex"
    out_dir.mkdir(parents=True, exist_ok=True)

    for required in (
        dataset / "particles.star",
        dataset / "particles.256.mrcs",
        source_run / "03_dataset/state_assignment.npy",
        source_run / "04_ground_truth/gt_vol0000.mrc",
        source_run / "04_ground_truth/gt_vol0025.mrc",
        source_run / "04_ground_truth/gt_vol0050.mrc",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

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
    jobs["import_particles"] = cs.create_job(
        args.project,
        args.workspace,
        "import_particles",
        params={
            "particle_meta_path": str(dataset / "particles.star"),
            "particle_blob_path": str(dataset),
            "ignore_pose": 0,
            "ignore_splits": 1,
            "sign": 1,
        },
        title="Import 1M spike particles sign-flipped",
    )
    if args.submit:
        queue(jobs["import_particles"], args.import_lane)
        status = jobs["import_particles"].wait_for_done(error_on_incomplete=True)
        print(f"import_particles {jobs['import_particles'].uid} finished with {status}")
        import_particles_output = first_output_name(jobs["import_particles"], "particle")
    else:
        manifest = {
            "project_uid": args.project,
            "workspace_uid": args.workspace,
            "source_run": str(source_run),
            "bench_root": str(bench_root),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "particle_sign": 1,
            "consensus_source": "homo_reconstruct",
            "mask_mode": "existing_cryosparc_mask_resampled_128_threshold_0.5",
            "source_solvent_mask_job": args.source_mask_job,
            "k_values": [1],
            "import_lane": args.import_lane,
            "gpu_lane": args.gpu_lane,
            "submitted": False,
            "jobs": {"import_particles": jobs["import_particles"].uid},
        }
        manifest_path = out_dir / args.manifest_name
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return

    jobs["homo_reconstruct"] = cs.create_job(
        args.project,
        args.workspace,
        "homo_reconstruct",
        connections={"particles": (jobs["import_particles"].uid, import_particles_output)},
        title="Homogeneous reconstruction only 1M sign-flipped",
    )
    jobs["flex_prep"] = cs.create_job(
        args.project,
        args.workspace,
        "flex_prep",
        connections={
            "particles": (jobs["import_particles"].uid, import_particles_output),
            "volume": (jobs["homo_reconstruct"].uid, "volume"),
        },
        params={"bin_size_pix": 128},
        title="3DFlex data prep 1M sign-flipped",
    )
    jobs["flex_meshprep"] = cs.create_job(
        args.project,
        args.workspace,
        "flex_meshprep",
        connections={
            "volume": (jobs["flex_prep"].uid, "volume"),
            "mask": (args.source_mask_job, "mask"),
        },
        params={
            "mask_in_lowpass_A": 10.0,
            "mask_in_threshold_level": 0.5,
            "mask_dilate_A": 2.0,
            "mask_pad_A": 5.0,
            "tetra_num_cells": 20,
        },
        title="3DFlex mesh prep 1M using J397/J409 box128 thresholded mask",
    )
    jobs["flex_train_k1"] = cs.create_job(
        args.project,
        args.workspace,
        "flex_train",
        connections={
            "particles": (jobs["flex_prep"].uid, "particles"),
            "flex_mesh": (jobs["flex_meshprep"].uid, "flex_mesh"),
        },
        params={"flex_K": 1},
        title="3DFlex train 1M K=1 using J397 box128 thresholded mask",
    )
    jobs["flex_highres_k1"] = cs.create_job(
        args.project,
        args.workspace,
        "flex_highres",
        connections={
            "flex_model": (jobs["flex_train_k1"].uid, "flex_model"),
            "particles": (jobs["flex_prep"].uid, "particles"),
        },
        title="3DFlex reconstruction 1M K=1 using J397 box128 thresholded mask",
    )

    manifest = {
        "project_uid": args.project,
        "workspace_uid": args.workspace,
        "source_run": str(source_run),
        "bench_root": str(bench_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "particle_sign": 1,
        "consensus_source": "homo_reconstruct",
        "mask_mode": "existing_cryosparc_mask_resampled_128_threshold_0.5",
        "source_solvent_mask_job": args.source_mask_job,
        "k_values": [1],
        "import_lane": args.import_lane,
        "gpu_lane": args.gpu_lane,
        "submitted": bool(args.submit),
        "jobs": {name: job.uid for name, job in jobs.items()},
        "dynamic_outputs": {
            "import_particles": [jobs["import_particles"].uid, import_particles_output],
            "homo_reconstruct": [jobs["homo_reconstruct"].uid, "volume"],
            "flex_prep_particles": [jobs["flex_prep"].uid, "particles"],
            "flex_prep_volume": [jobs["flex_prep"].uid, "volume"],
            "thresholded_solvent_mask": [args.source_mask_job, "mask"],
            "flex_train_k1": [jobs["flex_train_k1"].uid, "flex_model"],
            "flex_highres_k1": [jobs["flex_highres_k1"].uid, "volume_flex"],
        },
    }
    manifest_path = out_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

    if args.submit:
        queue(jobs["homo_reconstruct"], args.gpu_lane)
        queue(jobs["flex_prep"], args.gpu_lane)
        queue(jobs["flex_meshprep"], args.gpu_lane)
        queue(jobs["flex_train_k1"], args.gpu_lane)
        queue(jobs["flex_highres_k1"], args.gpu_lane)
        print(f"Queued 1M 3DFlex K=1 pipeline; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
