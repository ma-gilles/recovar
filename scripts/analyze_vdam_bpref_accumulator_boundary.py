#!/usr/bin/env python3
"""Compare captured VDAM BPref rows with production native/candidate accumulators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovar.em.bpref_contribution_replay import (
    BPrefAccumulatorReplay,
    accumulator_replay_metrics,
    load_bpref_contribution_bundle,
    replay_relion_double,
    summarize_bpref_contribution_bundle,
)
from scripts.analyze_vdam_mstep_boundary import _read_relion_array

SCHEMA = "recovar.vdam_bpref_accumulator_boundary.v1"


def _production_names(half: int) -> tuple[str, str, str, str]:
    if half == 1:
        return (
            "pipe_it1_c0_bp_data_pre_reweight.bin",
            "pipe_it1_c0_bp_weight.bin",
            "accum_h0_data.npy",
            "accum_h0_weight.npy",
        )
    if half == 2:
        return (
            "pipe_it1_c0_bp_data_h_pre_reweight.bin",
            "pipe_it1_c0_bp_weight_h.bin",
            "accum_h1_data.npy",
            "accum_h1_weight.npy",
        )
    raise ValueError(f"half must be 1 or 2, got {half}")


def _rank_particle_sources(rows: dict[str, np.ndarray]) -> list[dict[str, object]]:
    identities = np.asarray(rows["active_original_indices"], dtype=np.int64)
    data = np.asarray(rows["active_summed"])
    weight = np.asarray(rows["active_ctf_probs"], dtype=np.float64)
    if identities.shape != (data.shape[0],) or weight.shape != data.shape:
        raise ValueError("BPref source ranking received incompatible row arrays")
    ranking = []
    for original_index in np.unique(identities):
        selected = identities == original_index
        selected_data = data[selected]
        selected_weight = weight[selected]
        ranking.append(
            {
                "original_index": int(original_index),
                "row_count": int(np.count_nonzero(selected)),
                "data_l2": float(np.linalg.norm(selected_data.reshape(-1))),
                "data_l1": float(np.sum(np.abs(selected_data), dtype=np.float64)),
                "weight_l1": float(np.sum(np.abs(selected_weight), dtype=np.float64)),
            }
        )
    ranking.sort(key=lambda row: (-float(row["data_l2"]), int(row["original_index"])))
    return ranking


def _geometry(candidate: np.ndarray, native: np.ndarray, control: np.ndarray) -> dict[str, float]:
    candidate = np.asarray(candidate).reshape(-1)
    native = np.asarray(native).reshape(-1)
    control = np.asarray(control).reshape(-1)
    if candidate.shape != native.shape or candidate.shape != control.shape:
        raise ValueError("BPref comparison geometry requires equal shapes")
    native_delta = native - control
    candidate_delta = candidate - control
    native_norm2 = float(np.vdot(native_delta, native_delta).real)
    if native_norm2 == 0.0:
        return {
            "candidate_projection_on_native_delta": 0.0,
            "candidate_orthogonal_over_native_delta": float(np.linalg.norm(candidate_delta)),
        }
    projection = float(np.vdot(native_delta, candidate_delta).real / native_norm2)
    orthogonal = candidate_delta - projection * native_delta
    return {
        "candidate_projection_on_native_delta": projection,
        "candidate_orthogonal_over_native_delta": float(
            np.linalg.norm(orthogonal) / np.sqrt(native_norm2)
        ),
    }


def analyze(
    contribution_paths: list[Path],
    native_directory: Path,
    recovar_directory: Path,
    *,
    half: int,
) -> dict[str, object]:
    bundle = load_bpref_contribution_bundle(contribution_paths)
    boundary_half = int(np.asarray(bundle.boundary_values["half"]).item())
    if boundary_half != half:
        raise ValueError(f"captured half {boundary_half} does not match requested half {half}")
    native_data_name, native_weight_name, candidate_data_name, candidate_weight_name = (
        _production_names(half)
    )
    native = BPrefAccumulatorReplay(
        data=_read_relion_array(Path(native_directory) / native_data_name, complex_values=True),
        weight=_read_relion_array(Path(native_directory) / native_weight_name, complex_values=False),
        backend="native_relion_cuda",
        order="native_execution",
        precision="complex64/float32 accumulator promoted to binary64 dump",
        launch_topology="native_particle_launches",
    )
    candidate = BPrefAccumulatorReplay(
        data=np.load(Path(recovar_directory) / candidate_data_name, allow_pickle=False),
        weight=np.load(Path(recovar_directory) / candidate_weight_name, allow_pickle=False),
        backend="recovar_relion_fused_x_half",
        order="candidate_execution",
        precision="complex64/float32 accumulator promoted to binary64 dump",
        launch_topology="candidate_particle_grid_handler",
    )
    from recovar.relion_bind._relion_bind_core import TRILINEAR, get_backprojector_data

    deterministic = {
        order: replay_relion_double(
            bundle,
            order=order,
            get_backprojector_data=get_backprojector_data,
            interpolator=TRILINEAR,
        )
        for order in ("execution", "canonical")
    }
    comparisons = {
        "candidate_vs_native": accumulator_replay_metrics(candidate, native),
        "candidate_vs_relion_double_execution": accumulator_replay_metrics(
            candidate, deterministic["execution"]
        ),
        "candidate_vs_relion_double_canonical": accumulator_replay_metrics(
            candidate, deterministic["canonical"]
        ),
        "native_vs_relion_double_execution": accumulator_replay_metrics(
            native, deterministic["execution"]
        ),
        "native_vs_relion_double_canonical": accumulator_replay_metrics(
            native, deterministic["canonical"]
        ),
        "relion_double_execution_vs_canonical": accumulator_replay_metrics(
            deterministic["execution"], deterministic["canonical"]
        ),
    }
    execution = deterministic["execution"]
    geometry = {
        "data": _geometry(candidate.data, native.data, execution.data),
        "weight": _geometry(candidate.weight, native.weight, execution.weight),
    }
    ranking = _rank_particle_sources(bundle.concatenate("execution"))
    return {
        "schema": SCHEMA,
        "status": "complete",
        "half": int(half),
        "bundle": summarize_bpref_contribution_bundle(bundle),
        "comparisons": comparisons,
        "relion_double_execution_geometry": geometry,
        "particle_source_ranking": ranking,
        "top_particle_source_ranking": ranking[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--native-directory", required=True, type=Path)
    parser.add_argument("--recovar-directory", required=True, type=Path)
    parser.add_argument("--half", required=True, type=int, choices=(1, 2))
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        args.inputs,
        args.native_directory,
        args.recovar_directory,
        half=args.half,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
