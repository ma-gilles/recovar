#!/usr/bin/env python3
"""Generate 3DFlex maps at class-mean latents and default motion series.

This is for the 100k spike full-atom sanity run.  It creates a small
CryoSPARC external job per trained 3DFlex model containing the mean latent
coordinates for GT states 0, 25, and 50, then feeds those coordinates to
3D Flex Generator.  It also queues the default 3D Flex Generator latent-axis
motion sweep for visual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cryosparc.dataset import Dataset
from cryosparc.spec import Slot
from cryosparc.tools import CryoSPARC

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.spike_fullatom_method_benchmark.submit_cryosparc_3dflex_100k import (
    DEFAULT_NOTEBOOK,
    DEFAULT_SOURCE_RUN,
    load_cryosparc_credentials,
)


DEFAULT_BENCH_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sanity_100k_noise10_b100_20260529"
)


@dataclass(frozen=True)
class FlexModel:
    name: str
    train_job: str
    highres_job: str
    k: int
    mask_mode: str


DEFAULT_MODELS = (
    FlexModel("default_mask_k1", "J392", "J393", 1, "cryosparc_default"),
    FlexModel("default_mask_k2", "J394", "J395", 2, "cryosparc_default"),
    FlexModel("j397_mask_k1", "J411", "J412", 1, "j397_thresholded"),
    FlexModel("j397_mask_k2", "J413", "J414", 2, "j397_thresholded"),
)


def latest_latents(project_dir: Path, job_uid: str) -> Path:
    paths = sorted((project_dir / job_uid).glob(f"{job_uid}_latents_*.cs"))
    if not paths:
        raise FileNotFoundError(f"No latent .cs files found for {job_uid} in {project_dir}")
    return paths[-1]


def compute_state_mean_latents(
    latents_path: Path,
    state_assignment_path: Path,
    k: int,
    states: tuple[int, ...],
) -> tuple[list[dict[str, float | int | str]], Dataset]:
    latents = Dataset.load(str(latents_path))
    state_assignment = np.load(state_assignment_path)
    if len(latents) != len(state_assignment):
        raise ValueError(
            f"Latent rows ({len(latents)}) do not match state assignments "
            f"({len(state_assignment)})"
        )

    records: list[dict[str, float | int | str]] = []
    out = Dataset.allocate(
        len(states),
        fields=[
            (f"components_mode_{mode}/component", "u4")
            for mode in range(k)
        ]
        + [
            (f"components_mode_{mode}/value", "f4")
            for mode in range(k)
        ],
    )
    for mode in range(k):
        out[f"components_mode_{mode}/component"] = np.full(
            len(states), mode, dtype=np.uint32
        )

    for row_idx, state in enumerate(states):
        mask = state_assignment == state
        if not np.any(mask):
            raise ValueError(f"No particles assigned to GT state {state}")
        rec: dict[str, float | int | str] = {
            "state": int(state),
            "n_particles": int(mask.sum()),
        }
        for mode in range(k):
            field = f"components_mode_{mode}/value"
            values = np.asarray(latents[field], dtype=np.float32)
            mean_value = float(values[mask].mean())
            std_value = float(values[mask].std())
            out[field][row_idx] = mean_value
            rec[f"mode_{mode}_mean"] = mean_value
            rec[f"mode_{mode}_std"] = std_value
        records.append(rec)

    return records, out


def write_records_csv(path: Path, records: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def make_latent_external_job(
    cs: CryoSPARC,
    project_uid: str,
    workspace_uid: str,
    model: FlexModel,
    latents: Dataset,
    title_suffix: str,
) -> str:
    job = cs.create_external_job(
        project_uid,
        workspace_uid,
        title=f"3DFlex class-mean latent coordinates {model.name} {title_suffix}",
        desc=(
            "Three custom latent coordinates: means of particles assigned to "
            "GT states 0, 25, and 50."
        ),
    )
    slots = [
        Slot(name=f"components_mode_{mode}", dtype="components", required=(mode == 0))
        for mode in range(model.k)
    ]
    job.add_output("particle", name="latents", slots=slots, title="Class-mean latents")
    job.start("running")
    job.save_output("latents", latents)
    job.log("Saved GT-state class-mean latent coordinates.")
    job.stop()
    return job.uid


def queue_flex_generate_jobs(
    cs: CryoSPARC,
    project_uid: str,
    workspace_uid: str,
    model: FlexModel,
    external_latents_job: str,
    lane: str,
    frames: int,
    title_suffix: str,
) -> dict[str, str]:
    mean_job = cs.create_job(
        project_uid,
        workspace_uid,
        "flex_generate",
        connections={
            "flex_model": (model.train_job, "flex_model"),
            "volume": (model.highres_job, "volume_flex"),
            "latents": (external_latents_job, "latents"),
        },
        title=f"3DFlex generate GT mean latents {model.name} {title_suffix}",
        desc=(
            "Generate maps at the mean latent coordinate of particles assigned "
            "to GT states 0, 25, and 50."
        ),
    )
    mean_job.queue(lane=lane)

    motion_job = cs.create_job(
        project_uid,
        workspace_uid,
        "flex_generate",
        connections={
            "flex_model": (model.train_job, "flex_model"),
            "volume": (model.highres_job, "volume_flex"),
        },
        params={"flex_gen_num_pts": frames},
        title=f"3DFlex generate latent-axis motion {model.name} {title_suffix}",
        desc=(
            "Default 3D Flex Generator motion series: equally spaced maps "
            "along each latent axis through the origin."
        ),
    )
    motion_job.queue(lane=lane)

    return {
        "mean_latents_generate": mean_job.uid,
        "motion_axis_generate": motion_job.uid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("/tigress/CRYOEM/singerlab/mg6942/CS-testres"),
    )
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--project", default="P587")
    parser.add_argument("--workspace", default="W13")
    parser.add_argument("--lane", default="8hrs")
    parser.add_argument("--frames", type=int, default=11)
    parser.add_argument("--states", type=int, nargs="+", default=[0, 25, 50])
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    states = tuple(int(x) for x in args.states)
    source_run = args.source_run.resolve()
    out_dir = args.bench_root.resolve() / "evaluation_3dflex_mean_latents"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_assignment_path = source_run / "03_dataset/state_assignment.npy"
    if not state_assignment_path.exists():
        raise FileNotFoundError(state_assignment_path)

    model_manifests: list[dict[str, object]] = []
    all_means: list[dict[str, object]] = []

    creds = load_cryosparc_credentials(args.notebook)
    cs = CryoSPARC(
        license=str(creds["license"]),
        email=str(creds["email"]),
        password=str(creds["password"]),
        host=str(creds["host"]),
        base_port=int(creds["base_port"]),
    )
    cs.test_connection()

    title_suffix = "20260530"
    for model in DEFAULT_MODELS:
        latents_path = latest_latents(args.project_dir.resolve(), model.train_job)
        means, mean_latents_dataset = compute_state_mean_latents(
            latents_path, state_assignment_path, model.k, states
        )
        for record in means:
            record.update(
                {
                    "model": model.name,
                    "train_job": model.train_job,
                    "highres_job": model.highres_job,
                    "mask_mode": model.mask_mode,
                }
            )
        all_means.extend(means)

        latent_cs_path = out_dir / f"{model.name}_state_mean_latents.cs"
        mean_latents_dataset.save(latent_cs_path)

        manifest: dict[str, object] = {
            "model": model.name,
            "train_job": model.train_job,
            "highres_job": model.highres_job,
            "k": model.k,
            "mask_mode": model.mask_mode,
            "latest_latents": str(latents_path),
            "local_mean_latents_cs": str(latent_cs_path),
            "states": list(states),
            "lane": args.lane,
            "frames": args.frames,
        }
        if args.submit:
            external_job = make_latent_external_job(
                cs,
                args.project,
                args.workspace,
                model,
                mean_latents_dataset,
                title_suffix,
            )
            generated = queue_flex_generate_jobs(
                cs,
                args.project,
                args.workspace,
                model,
                external_job,
                args.lane,
                args.frames,
                title_suffix,
            )
            manifest["external_latents_job"] = external_job
            manifest.update(generated)
        model_manifests.append(manifest)

    means_csv = out_dir / "3dflex_state_mean_latents.csv"
    write_records_csv(means_csv, all_means)

    manifest_path = out_dir / "3dflex_generate_mean_latents_manifest.json"
    manifest = {
        "project_uid": args.project,
        "workspace_uid": args.workspace,
        "source_run": str(source_run),
        "bench_root": str(args.bench_root.resolve()),
        "state_assignment": str(state_assignment_path),
        "submitted": bool(args.submit),
        "models": model_manifests,
        "outputs": {
            "mean_latents_csv": str(means_csv),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
