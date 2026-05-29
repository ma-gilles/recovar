#!/usr/bin/env python3
"""Submit the 100k spike 3DFlex zdim sweep through CryoSPARC lanes."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from cryosparc.tools import CryoSPARC


DEFAULT_SOURCE_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/"
    "n00100000/runs/n00100000_seed0000"
)
DEFAULT_BENCH_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_benchmark_100k_20260517"
)
DEFAULT_NOTEBOOK = Path("/home/mg6942/recovar/20231229_3dflex.ipynb")


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


def queue(job, lane: str | None) -> None:
    if lane:
        job.queue(lane=lane)
    else:
        job.queue()


def first_output_name(job, output_type: str) -> str:
    job.refresh()
    for name, output in job.outputs.items():
        if output.type == output_type:
            return name
    raise RuntimeError(
        f"Job {job.uid} has no {output_type!r} output. Current outputs: {list(job.outputs)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--project", default=None)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--workspace-title", default="spike fullatom 100k method benchmark 20260517")
    parser.add_argument("--train-mask", type=Path, default=None)
    parser.add_argument("--train-mask-psize", type=float, default=None)
    parser.add_argument("--import-lane", default="24hrs")
    parser.add_argument("--gpu-lane", default="48hrs-a100")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    bench_root = args.bench_root.resolve()
    dataset = source_run / "03_dataset"
    mean_map = source_run / "06_pipeline/output/volumes/mean.mrc"
    train_mask = args.train_mask
    if train_mask is None:
        binned_mask = bench_root / "manifests/volume_mask_union_box128.mrc"
        train_mask = binned_mask if binned_mask.exists() else source_run / "05_masks/volume_mask_union.mrc"
    train_mask_psize = args.train_mask_psize
    if train_mask_psize is None:
        train_mask_psize = 2.5 if train_mask.name.endswith("_box128.mrc") else 1.25
    out_dir = bench_root / "cryosparc_3dflex"
    out_dir.mkdir(parents=True, exist_ok=True)

    for required in (
        dataset / "particles.star",
        dataset / "particles.256.mrcs",
        mean_map,
        train_mask,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    creds = load_cryosparc_credentials(args.notebook)
    project_uid = args.project or str(creds["project_num"])
    cs = CryoSPARC(
        license=str(creds["license"]),
        email=str(creds["email"]),
        password=str(creds["password"]),
        host=str(creds["host"]),
        base_port=int(creds["base_port"]),
    )
    cs.test_connection()

    if args.workspace:
        workspace_uid = args.workspace
    else:
        workspace = cs.create_workspace(project_uid, args.workspace_title)
        workspace_uid = workspace.uid

    jobs: dict[str, object] = {}
    jobs["import_particles"] = cs.create_job(
        project_uid,
        workspace_uid,
        "import_particles",
        params={
            "particle_meta_path": str(dataset / "particles.star"),
            "particle_blob_path": str(dataset),
            "ignore_pose": 0,
            "ignore_splits": 1,
            "sign": -1,
        },
        title="Import 100k spike particles",
    )
    jobs["import_map"] = cs.create_job(
        project_uid,
        workspace_uid,
        "import_volumes",
        params={
            "volume_blob_path": str(mean_map),
            "volume_out_name": "map",
            "volume_psize": 1.25,
        },
        title="Import RECOVAR mean map",
    )
    jobs["import_mask"] = cs.create_job(
        project_uid,
        workspace_uid,
        "import_volumes",
        params={
            "volume_blob_path": str(train_mask),
            "volume_out_name": "mask",
            "volume_psize": train_mask_psize,
        },
        title="Import training mask",
    )
    if args.submit:
        for name in ("import_particles", "import_map", "import_mask"):
            queue(jobs[name], args.import_lane)
        for name in ("import_particles", "import_map", "import_mask"):
            status = jobs[name].wait_for_done(error_on_incomplete=True)
            print(f"{name} {jobs[name].uid} finished with {status}")
    else:
        print("Created import jobs only. Re-run with --submit to queue imports and build 3DFlex jobs.")
        manifest = {
            "project_uid": project_uid,
            "workspace_uid": workspace_uid,
            "source_run": str(source_run),
            "bench_root": str(bench_root),
            "import_lane": args.import_lane,
            "gpu_lane": args.gpu_lane,
            "train_mask": str(train_mask),
            "train_mask_psize": train_mask_psize,
            "submitted": False,
            "jobs": {name: job.uid for name, job in jobs.items()},
        }
        manifest_path = out_dir / "cryosparc_3dflex_jobs.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return

    import_particles_output = first_output_name(jobs["import_particles"], "particle")
    import_map_output = first_output_name(jobs["import_map"], "volume")
    import_mask_output = first_output_name(jobs["import_mask"], "mask")

    jobs["flex_prep"] = cs.create_job(
        project_uid,
        workspace_uid,
        "flex_prep",
        connections={
            "particles": (jobs["import_particles"].uid, import_particles_output),
            "volume": (jobs["import_map"].uid, import_map_output),
        },
        params={"bin_size_pix": 128},
        title="3DFlex data prep 100k",
    )
    jobs["flex_meshprep"] = cs.create_job(
        project_uid,
        workspace_uid,
        "flex_meshprep",
        connections={
            "volume": (jobs["flex_prep"].uid, "volume"),
            "mask": (jobs["import_mask"].uid, import_mask_output),
        },
        params={
            "mask_in_lowpass_A": 10.0,
            "mask_in_threshold_level": 0.5,
            "mask_dilate_A": 2.0,
            "mask_pad_A": 5.0,
            "tetra_num_cells": 20,
        },
        title="3DFlex mesh prep 100k",
    )
    for k in (1, 2):
        jobs[f"flex_train_k{k}"] = cs.create_job(
            project_uid,
            workspace_uid,
            "flex_train",
            connections={
                "particles": (jobs["flex_prep"].uid, "particles"),
                "flex_mesh": (jobs["flex_meshprep"].uid, "flex_mesh"),
            },
            params={"flex_K": k},
            title=f"3DFlex train 100k K={k}",
        )

    manifest = {
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "source_run": str(source_run),
        "bench_root": str(bench_root),
        "import_lane": args.import_lane,
        "gpu_lane": args.gpu_lane,
        "train_mask": str(train_mask),
        "train_mask_psize": train_mask_psize,
        "submitted": bool(args.submit),
        "jobs": {name: job.uid for name, job in jobs.items()},
        "dynamic_outputs": {
            "import_particles": import_particles_output,
            "import_map": import_map_output,
            "import_mask": import_mask_output,
        },
    }
    manifest_path = out_dir / "cryosparc_3dflex_jobs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

    queue(jobs["flex_prep"], args.gpu_lane)
    queue(jobs["flex_meshprep"], args.gpu_lane)
    queue(jobs["flex_train_k1"], args.gpu_lane)
    queue(jobs["flex_train_k2"], args.gpu_lane)
    print(f"Queued CryoSPARC jobs; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
