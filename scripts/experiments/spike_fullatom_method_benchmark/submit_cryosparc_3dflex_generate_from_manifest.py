#!/usr/bin/env python3
"""Submit 3DFlex map generation at GT-state mean latents from a job manifest.

This fills the reproducibility gap left by
``prepare_spike_trajectory_state_maps.py``: given an existing 3DFlex training
manifest, compute the mean trained 3DFlex latent value for each requested GT
state and queue one CryoSPARC ``flex_generate`` job that emits one map per
state.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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

from scripts.experiments.spike_fullatom_method_benchmark.submit_cryosparc_3dflex_100k import (  # noqa: E501
    DEFAULT_NOTEBOOK,
    load_cryosparc_credentials,
)


DEFAULT_PROJECT_DIR = Path("/projects/CRYOEM/singerlab/mg6942/CS-testres")


@dataclass(frozen=True)
class FlexModel:
    name: str
    train_job: str
    highres_job: str
    k: int
    mask_mode: str


def n_label_from_source_run(source_run: Path) -> str:
    match = re.search(r"(n\d{8})_seed\d+", source_run.name)
    if match:
        return match.group(1)
    for part in reversed(source_run.parts):
        if re.fullmatch(r"n\d{8}", part):
            return part
    return "nunknown"


def latest_latents(project_dir: Path, job_uid: str) -> Path:
    paths = sorted((project_dir / job_uid).glob(f"{job_uid}_latents_*.cs"))
    if not paths:
        raise FileNotFoundError(f"No latent .cs files found for {job_uid} under {project_dir}")
    return paths[-1]


def models_from_manifest(manifest: dict[str, object]) -> list[FlexModel]:
    if "models" in manifest:
        models = []
        for item in manifest["models"]:  # type: ignore[index]
            models.append(
                FlexModel(
                    name=str(item["model"]),
                    train_job=str(item["train_job"]),
                    highres_job=str(item["highres_job"]),
                    k=int(item.get("k", 1)),
                    mask_mode=str(item.get("mask_mode", manifest.get("mask_mode", "unknown"))),
                )
            )
        return models

    jobs = manifest.get("jobs", {})
    if not isinstance(jobs, dict):
        jobs = {}
    train_job = str(manifest.get("train_job") or jobs.get("flex_train_k1") or "")
    highres_job = str(manifest.get("highres_job") or jobs.get("flex_highres_k1") or "")
    if not train_job or not highres_job:
        raise ValueError("Manifest must provide train/highres jobs or jobs.flex_train_k1/flex_highres_k1")
    return [
        FlexModel(
            name=str(manifest.get("model", "corrected_k1")),
            train_job=train_job,
            highres_job=highres_job,
            k=int(manifest.get("k", 1)),
            mask_mode=str(manifest.get("mask_mode", "unknown")),
        )
    ]


def compute_state_mean_latents(
    *,
    latents_path: Path,
    state_assignment_path: Path,
    model: FlexModel,
    states: tuple[int, ...],
) -> tuple[list[dict[str, object]], Dataset]:
    latents = Dataset.load(str(latents_path))
    assignments = np.load(state_assignment_path).astype(np.int64).reshape(-1)
    if len(latents) != assignments.size:
        raise RuntimeError(
            f"Latent rows ({len(latents)}) do not match assignments ({assignments.size})"
        )

    fields = [
        (f"components_mode_{mode}/component", "u4")
        for mode in range(model.k)
    ] + [
        (f"components_mode_{mode}/value", "f4")
        for mode in range(model.k)
    ]
    out = Dataset.allocate(len(states), fields=fields)
    for mode in range(model.k):
        out[f"components_mode_{mode}/component"] = np.full(len(states), mode, dtype=np.uint32)

    rows: list[dict[str, object]] = []
    for row_idx, state in enumerate(states):
        state_mask = assignments == state
        if not np.any(state_mask):
            raise RuntimeError(f"No particles assigned to GT state {state}")
        row: dict[str, object] = {
            "model": model.name,
            "train_job": model.train_job,
            "highres_job": model.highres_job,
            "state": int(state),
            "n_particles": int(state_mask.sum()),
        }
        for mode in range(model.k):
            values = np.asarray(latents[f"components_mode_{mode}/value"], dtype=np.float32)
            mean_value = float(values[state_mask].mean())
            out[f"components_mode_{mode}/value"][row_idx] = mean_value
            row[f"mode_{mode}_mean"] = mean_value
            row[f"mode_{mode}_std"] = float(values[state_mask].std())
        rows.append(row)
    return rows, out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_latent_external_job(
    *,
    cs: CryoSPARC,
    project_uid: str,
    workspace_uid: str,
    model: FlexModel,
    latents: Dataset,
    states: tuple[int, ...],
    title_suffix: str,
) -> str:
    job = cs.create_external_job(
        project_uid,
        workspace_uid,
        title=f"3DFlex GT-state mean latents {model.name} {title_suffix}",
        desc=f"Mean latent coordinates for GT states: {', '.join(map(str, states))}.",
    )
    slots = [
        Slot(name=f"components_mode_{mode}", dtype="components", required=(mode == 0))
        for mode in range(model.k)
    ]
    job.add_output("particle", name="latents", slots=slots, title="GT-state mean latents")
    job.start("running")
    job.save_output("latents", latents)
    job.log(f"Saved {len(states)} GT-state mean latent coordinates.")
    job.stop()
    return job.uid


def queue_flex_generate(
    *,
    cs: CryoSPARC,
    project_uid: str,
    workspace_uid: str,
    model: FlexModel,
    external_job_uid: str,
    lane: str,
    states: tuple[int, ...],
    title_suffix: str,
) -> str:
    job = cs.create_job(
        project_uid,
        workspace_uid,
        "flex_generate",
        connections={
            "flex_model": (model.train_job, "flex_model"),
            "volume": (model.highres_job, "volume_flex"),
            "latents": (external_job_uid, "latents"),
        },
        title=f"3DFlex generate GT mean latents {model.name} {title_suffix}",
        desc=f"Generate one map per requested GT-state mean latent: {list(states)}.",
    )
    job.queue(lane=lane)
    return job.uid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--states", type=int, nargs="+", required=True)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--lane", default=None)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_manifest_path = args.manifest.resolve()
    source_manifest = json.loads(source_manifest_path.read_text())
    source_run = Path(str(source_manifest["source_run"])).resolve()
    bench_root = Path(str(source_manifest["bench_root"])).resolve()
    label = n_label_from_source_run(source_run)
    eval_root = (
        args.eval_root.resolve()
        if args.eval_root is not None
        else bench_root / "evaluation_3dflex_mean_latents_allstates" / label
    )
    eval_root.mkdir(parents=True, exist_ok=True)
    out_manifest = eval_root / "3dflex_generate_mean_latents_manifest.json"
    if out_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"{out_manifest} exists; pass --overwrite to replace")

    state_assignment_path = source_run / "03_dataset" / "state_assignment.npy"
    if not state_assignment_path.exists():
        raise FileNotFoundError(state_assignment_path)

    states = tuple(int(x) for x in args.states)
    project_uid = str(source_manifest["project_uid"])
    workspace_uid = str(source_manifest["workspace_uid"])
    lane = str(args.lane or source_manifest.get("gpu_lane") or source_manifest.get("lane") or "48hrs-a100")

    creds = load_cryosparc_credentials(args.notebook)
    cs = CryoSPARC(
        license=str(creds["license"]),
        email=str(creds["email"]),
        password=str(creds["password"]),
        host=str(creds["host"]),
        base_port=int(creds["base_port"]),
    )
    cs.test_connection()

    all_rows: list[dict[str, object]] = []
    model_manifests: list[dict[str, object]] = []
    for model in models_from_manifest(source_manifest):
        latents_path = latest_latents(args.project_dir.resolve(), model.train_job)
        rows, mean_latents = compute_state_mean_latents(
            latents_path=latents_path,
            state_assignment_path=state_assignment_path,
            model=model,
            states=states,
        )
        all_rows.extend(rows)
        local_cs = eval_root / f"{model.name}_state_mean_latents.cs"
        mean_latents.save(local_cs)

        record: dict[str, object] = {
            "model": model.name,
            "train_job": model.train_job,
            "highres_job": model.highres_job,
            "k": model.k,
            "mask_mode": model.mask_mode,
            "latest_latents": str(latents_path),
            "local_mean_latents_cs": str(local_cs),
            "states": list(states),
            "lane": lane,
        }
        if args.submit:
            suffix = f"{label} {len(states)}states"
            external_uid = make_latent_external_job(
                cs=cs,
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                model=model,
                latents=mean_latents,
                states=states,
                title_suffix=suffix,
            )
            generate_uid = queue_flex_generate(
                cs=cs,
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                model=model,
                external_job_uid=external_uid,
                lane=lane,
                states=states,
                title_suffix=suffix,
            )
            record["external_latents_job"] = external_uid
            record["mean_latents_generate"] = generate_uid
        model_manifests.append(record)

    means_csv = eval_root / "state_mean_latents.csv"
    write_csv(means_csv, all_rows)
    manifest = {
        "source_manifest": str(source_manifest_path),
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "source_run": str(source_run),
        "bench_root": str(bench_root),
        "project_dir": str(args.project_dir.resolve()),
        "state_assignment": str(state_assignment_path),
        "states": list(states),
        "lane": lane,
        "submitted": bool(args.submit),
        "models": model_manifests,
        "outputs": {
            "mean_latents_csv": str(means_csv),
            "manifest": str(out_manifest),
        },
    }
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
