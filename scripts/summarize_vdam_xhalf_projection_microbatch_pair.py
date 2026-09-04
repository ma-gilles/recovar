#!/usr/bin/env python3
"""Summarize one warmed VDAM x-half row-pixel cap pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from recovar.em.dense_single_volume.local_em_engine import (
    _exact_local_max_hypotheses_per_microbatch,
)
from scripts.summarize_em_completion_bench import (
    _load_relion_volume,
    normalized_fsc_auc,
    shell_fsc,
)

SCHEMA = "recovar.vdam_xhalf_projection_microbatch_pair.v2"
SCORED_LABELS = (
    "scored-control-1",
    "scored-control-2",
    "scored-candidate-1",
    "scored-candidate-2",
)
PAIR_LABELS = {
    "cross-1": ("scored-candidate-1", "scored-control-1"),
    "cross-2": ("scored-candidate-2", "scored-control-2"),
    "control-repeat": ("scored-control-1", "scored-control-2"),
    "candidate-repeat": ("scored-candidate-1", "scored-candidate-2"),
}
ARTIFACT_SUFFIXES = ("data.star", "model.star", "class001.mrc")


class XHalfPairError(RuntimeError):
    """Raised when a cap pair is incomplete or mixed."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise XHalfPairError(f"cannot read JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise XHalfPairError(f"expected a JSON object at {path}")
    return value


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _map_metric(lhs_path: Path, rhs_path: Path, *, with_fsc: bool) -> dict[str, Any]:
    lhs = _load_relion_volume(lhs_path)
    rhs = _load_relion_volume(rhs_path)
    if lhs.shape != rhs.shape:
        raise XHalfPairError(
            f"map shapes differ: {lhs_path} {lhs.shape} vs {rhs_path} {rhs.shape}"
        )
    delta = np.asarray(lhs - rhs, dtype=np.float64)
    denominator = float(np.linalg.norm(np.asarray(rhs, dtype=np.float64).reshape(-1)))
    result: dict[str, Any] = {
        "bytewise_array_equal": bool(np.array_equal(lhs, rhs)),
        "differing_voxel_count": int(np.count_nonzero(lhs != rhs)),
        "max_absolute_error": float(np.max(np.abs(delta))),
        "relative_l2_error": float(
            np.linalg.norm(delta.reshape(-1)) / max(denominator, 1e-300)
        ),
    }
    if with_fsc:
        result["fsc_auc"] = float(normalized_fsc_auc(shell_fsc(lhs, rhs)))
    return result


def _arms(root: Path) -> dict[str, int]:
    rows = {}
    for raw in (root / "provenance" / "arms.tsv").read_text().splitlines():
        arm, value = raw.split("\t")
        rows[arm] = int(value)
    if set(rows) != {"control", "candidate"} or rows["control"] == rows["candidate"]:
        raise XHalfPairError(f"invalid arm definitions: {rows}")
    return rows


def _execution_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in (root / "provenance" / "execution_order.tsv").read_text().splitlines():
        label, arm, row_pixels, elapsed_ns, gpu_uuid, cache_dir, run_status = raw.split("\t")
        rows.append(
            {
                "label": label,
                "arm": arm,
                "row_pixels": int(row_pixels),
                "external_wall_seconds": int(elapsed_ns) / 1.0e9,
                "gpu_uuid": gpu_uuid,
                "jax_cache_dir": cache_dir,
                "run_status": int(run_status),
            }
        )
    expected = [
        ("warmup-control", "control"),
        ("warmup-candidate", "candidate"),
        ("scored-control-1", "control"),
        ("scored-candidate-1", "candidate"),
        ("scored-candidate-2", "candidate"),
        ("scored-control-2", "control"),
    ]
    observed = [(row["label"], row["arm"]) for row in rows]
    if observed != expected:
        raise XHalfPairError(f"execution order differs: {observed!r}")
    if any(row["run_status"] != 0 for row in rows):
        raise XHalfPairError("one or more arms failed")
    if len({row["gpu_uuid"] for row in rows}) != 1:
        raise XHalfPairError("pair spans multiple physical GPUs")
    caches = {
        arm: {row["jax_cache_dir"] for row in rows if row["arm"] == arm}
        for arm in ("control", "candidate")
    }
    if any(len(values) != 1 for values in caches.values()) or caches["control"] == caches["candidate"]:
        raise XHalfPairError(f"arms do not have separate stable caches: {caches}")
    for row in rows:
        run = _load_json(root / row["label"] / "provenance" / "run.json")
        row["runner_wall_seconds"] = float(run["recovar_wall_s"])
    return rows


def _profile_absence(root: Path, target_iteration: int) -> dict[str, Any]:
    result = {}
    for label in ("warmup-control", "warmup-candidate", *SCORED_LABELS):
        rows = []
        for iteration in range(1, target_iteration + 1):
            meta = _load_json(
                root / label / "recovar_prefix" / f"run_it{iteration:03d}_recovar_meta.json"
            )
            keys = sorted(key for key in meta if "profile" in key.lower())
            rows.append({"iteration": iteration, "profile_keys": keys})
        result[label] = {
            "iteration_count": len(rows),
            "iterations_with_profile_metadata": sum(bool(row["profile_keys"]) for row in rows),
            "profile_unset_contract_pass": not any(row["profile_keys"] for row in rows),
            "rows": rows,
        }
    return result


def _memory(root: Path, execution: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for row in execution:
        path = root / row["label"] / "telemetry" / "gpu_memory.tsv"
        samples = []
        for raw in path.read_text().splitlines():
            timestamp_ns, used_mib, total_mib = raw.split("\t")
            samples.append(
                {
                    "timestamp_ns": int(timestamp_ns),
                    "used_mib": int(used_mib),
                    "total_mib": int(total_mib),
                }
            )
        if not samples:
            raise XHalfPairError(f"no GPU-memory samples for {row['label']}")
        result[row["label"]] = {
            "sample_count": len(samples),
            "peak_used_mib": max(sample["used_mib"] for sample in samples),
            "total_mib": max(sample["total_mib"] for sample in samples),
        }
    return result


def derive_it80_topology(layout_meta: dict[str, Any], row_pixels: int) -> dict[str, Any]:
    """Derive cap buckets from frozen layout evidence without enabling PROFILE."""

    profile = layout_meta.get("halfset_0_profile_summary")
    if not isinstance(profile, dict):
        raise XHalfPairError("layout evidence lacks halfset_0_profile_summary")
    sizes = [int(value) for value in profile["chunk_sizes"]]
    padded = [int(value) for value in profile["chunk_padded_rotations"]]
    if len(sizes) != len(padded) or not sizes or any(size <= 0 for size in sizes):
        raise XHalfPairError("layout evidence has invalid chunks")
    widths = [pad // size for pad, size in zip(padded, sizes, strict=True)]
    if any(pad % size for pad, size in zip(padded, sizes, strict=True)) or len(set(widths)) != 1:
        raise XHalfPairError("layout evidence does not have one reconstructible bucket width")
    width = widths[0]
    n_projection = int(profile["n_projection_windowed"])
    n_translations = int(layout_meta["n_translations"])
    upstream_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        n_projection,
        n_trans=n_translations,
        n_recon_windowed=n_projection,
        allow_high_memory_default=False,
    )
    projection_cap = max(width, int(row_pixels) // n_projection)
    effective_cap = min(int(upstream_cap), int(projection_cap))
    images_per_chunk = max(1, effective_cap // width)
    remaining = sum(sizes)
    predicted_sizes = []
    while remaining:
        take = min(remaining, images_per_chunk)
        predicted_sizes.append(take)
        remaining -= take
    return {
        "source_profile_chunk_sizes": sizes,
        "source_profile_chunk_padded_rotations": padded,
        "bucket_rotation_width": width,
        "particle_count": sum(sizes),
        "n_projection_windowed": n_projection,
        "n_translations": n_translations,
        "upstream_exact_local_cap": int(upstream_cap),
        "requested_projection_row_pixel_cap": int(row_pixels),
        "projection_hypothesis_cap": int(projection_cap),
        "effective_hypothesis_cap": int(effective_cap),
        "images_per_chunk": int(images_per_chunk),
        "predicted_chunk_sizes": predicted_sizes,
        "predicted_chunk_padded_rotations": [size * width for size in predicted_sizes],
        "predicted_big_jit_bucket_count": len(predicted_sizes),
    }


def _artifact_exactness(root: Path, target_iteration: int) -> dict[str, Any]:
    digests: dict[Path, str] = {}

    def digest(path: Path) -> str:
        if path not in digests:
            digests[path] = _sha256(path)
        return digests[path]

    result = {}
    for pair, (lhs, rhs) in PAIR_LABELS.items():
        rows = []
        for iteration in range(1, target_iteration + 1):
            for suffix in ARTIFACT_SUFFIXES:
                lhs_path = root / lhs / "recovar_prefix" / f"run_it{iteration:03d}_{suffix}"
                rhs_path = root / rhs / "recovar_prefix" / f"run_it{iteration:03d}_{suffix}"
                lhs_sha = digest(lhs_path)
                rhs_sha = digest(rhs_path)
                rows.append(
                    {
                        "iteration": iteration,
                        "suffix": suffix,
                        "lhs_sha256": lhs_sha,
                        "rhs_sha256": rhs_sha,
                        "byte_exact": lhs_sha == rhs_sha,
                    }
                )
        result[pair] = {
            "comparison_count": len(rows),
            "byte_mismatch_count": sum(not row["byte_exact"] for row in rows),
            "rows": rows,
        }
    return result


def _map_metrics(root: Path, relion_dir: Path, target_iteration: int) -> dict[str, Any]:
    checkpoint_candidates = (1, 2, 3, 4, 8, 20, 40, 60, target_iteration)
    checkpoints = tuple(sorted({value for value in checkpoint_candidates if value <= target_iteration}))
    pairwise = {}
    for pair, (lhs, rhs) in PAIR_LABELS.items():
        pairwise[pair] = [
            {
                "iteration": iteration,
                **_map_metric(
                    root / lhs / "recovar_prefix" / f"run_it{iteration:03d}_class001.mrc",
                    root / rhs / "recovar_prefix" / f"run_it{iteration:03d}_class001.mrc",
                    with_fsc=iteration == target_iteration,
                ),
            }
            for iteration in checkpoints
        ]
    versus_relion = {}
    for label in SCORED_LABELS:
        versus_relion[label] = [
            {
                "iteration": iteration,
                **_map_metric(
                    root / label / "recovar_prefix" / f"run_it{iteration:03d}_class001.mrc",
                    relion_dir / f"run_it{iteration:03d}_class001.mrc",
                    with_fsc=iteration == target_iteration,
                ),
            }
            for iteration in checkpoints
        ]
    repeat_final = [pairwise[name][-1] for name in ("control-repeat", "candidate-repeat")]
    cross_final = [pairwise[name][-1] for name in ("cross-1", "cross-2")]
    repeat_rel_l2_max = max(row["relative_l2_error"] for row in repeat_final)
    cross_rel_l2_max = max(row["relative_l2_error"] for row in cross_final)
    repeat_fsc_min = min(row["fsc_auc"] for row in repeat_final)
    cross_fsc_min = min(row["fsc_auc"] for row in cross_final)
    return {
        "checkpoints": checkpoints,
        "pairwise": pairwise,
        "versus_relion": versus_relion,
        "terminal_repeat_envelope": {
            "repeat_relative_l2_max": repeat_rel_l2_max,
            "cross_relative_l2_max": cross_rel_l2_max,
            "repeat_fsc_auc_min": repeat_fsc_min,
            "cross_fsc_auc_min": cross_fsc_min,
            "cross_within_2x_repeat_relative_l2": cross_rel_l2_max
            <= 2.0 * repeat_rel_l2_max + 1e-15,
            "cross_at_least_repeat_fsc_floor": cross_fsc_min >= repeat_fsc_min - 1e-12,
        },
    }


def _state_audits(root: Path) -> dict[str, Any]:
    reports = {
        pair: _load_json(root / "audits" / f"{pair}_state.json") for pair in PAIR_LABELS
    }
    reports.update(
        {
            f"{label}-vs-relion": _load_json(
                root / "audits" / f"{label}_vs_relion_state.json"
            )
            for label in SCORED_LABELS
        }
    )
    summaries = {
        name: {
            "first_divergent_iteration": report["first_divergent_iteration"],
            "maximum_divergent_particle_count": max(
                int(row["divergent_particle_count"]) for row in report["iterations"]
            ),
            "maximum_pmax_absolute_error": max(
                float(row["pmax_absolute_error"]["max"]) for row in report["iterations"]
            ),
        }
        for name, report in reports.items()
    }
    return {"summaries": summaries, "reports": reports}


def summarize(
    root: Path,
    relion_dir: Path,
    layout_evidence_meta: Path,
    target_iteration: int,
) -> dict[str, Any]:
    root = root.resolve()
    relion_dir = relion_dir.resolve()
    arms = _arms(root)
    execution = _execution_rows(root)
    profile_absence = _profile_absence(root, target_iteration)
    memory = _memory(root, execution)
    external = {
        arm: [
            row["external_wall_seconds"]
            for row in execution
            if row["label"].startswith("scored-") and row["arm"] == arm
        ]
        for arm in ("control", "candidate")
    }
    medians = {arm: float(statistics.median(values)) for arm, values in external.items()}
    runner = {
        arm: [
            row["runner_wall_seconds"]
            for row in execution
            if row["label"].startswith("scored-") and row["arm"] == arm
        ]
        for arm in ("control", "candidate")
    }
    runner_medians = {arm: float(statistics.median(values)) for arm, values in runner.items()}
    scored_memory = {
        arm: [
            memory[row["label"]]["peak_used_mib"]
            for row in execution
            if row["label"].startswith("scored-") and row["arm"] == arm
        ]
        for arm in ("control", "candidate")
    }
    topology_evidence = _load_json(layout_evidence_meta)
    topology = {
        arm: derive_it80_topology(topology_evidence, row_pixels)
        for arm, row_pixels in arms.items()
    }
    artifacts = _artifact_exactness(root, target_iteration)
    maps = _map_metrics(root, relion_dir, target_iteration)
    states = _state_audits(root)
    cross_states_exact = all(
        states["summaries"][name]["first_divergent_iteration"] is None
        for name in ("cross-1", "cross-2")
    )
    map_envelope_pass = bool(
        maps["terminal_repeat_envelope"]["cross_within_2x_repeat_relative_l2"]
        and maps["terminal_repeat_envelope"]["cross_at_least_repeat_fsc_floor"]
    )
    profile_unset = all(row["profile_unset_contract_pass"] for row in profile_absence.values())
    speedup = medians["control"] / medians["candidate"]
    pairwise_speedups = [
        external["control"][0] / external["candidate"][0],
        external["control"][1] / external["candidate"][1],
    ]
    repeatable_gain = speedup >= 1.02 and min(pairwise_speedups) >= 1.01
    science_pass = bool(cross_states_exact and map_envelope_pass)
    return {
        "schema": SCHEMA,
        "root": str(root),
        "relion_reference_dir": str(relion_dir),
        "layout_evidence_meta": str(layout_evidence_meta.resolve()),
        "layout_evidence_sha256": _sha256(layout_evidence_meta),
        "target_iteration": int(target_iteration),
        "arms": arms,
        "execution": execution,
        "runtime": {
            "scored_external_wall_seconds": external,
            "scored_external_median_wall_seconds": medians,
            "candidate_speedup": speedup,
            "pairwise_speedups": pairwise_speedups,
            "scored_runner_wall_seconds": runner,
            "scored_runner_median_wall_seconds": runner_medians,
        },
        "gpu_memory": {"per_run": memory, "scored_peak_used_mib": scored_memory},
        "profile_metadata": profile_absence,
        "derived_it80_topology": topology,
        "artifact_exactness": artifacts,
        "map_metrics": maps,
        "particle_state": states,
        "classification": {
            "benchmark_topology_valid": True,
            "profile_unset_contract_pass": profile_unset,
            "cross_particle_pose_translation_exact": cross_states_exact,
            "cross_terminal_map_within_repeat_envelope": map_envelope_pass,
            "science_within_repeat_envelope": science_pass,
            "repeatable_warmed_gain_at_least_2pct": repeatable_gain,
            "promote_candidate_factor": bool(science_pass and repeatable_gain),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--relion-dir", type=Path, required=True)
    parser.add_argument("--layout-evidence-meta", type=Path, required=True)
    parser.add_argument("--target-iteration", type=int, default=80)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = summarize(
        args.root,
        args.relion_dir,
        args.layout_evidence_meta,
        args.target_iteration,
    )
    output = args.root / "pair_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.check and not report["classification"]["profile_unset_contract_pass"]:
        raise SystemExit("PROFILE unexpectedly enabled in x-half cap benchmark")


if __name__ == "__main__":
    main()
