#!/usr/bin/env python3
"""Compare captured VDAM BPref rows with production native/candidate accumulators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from recovar.em.diagnostics.bpref_contribution_replay import (
    BPrefAccumulatorReplay,
    accumulator_replay_metrics,
    load_bpref_contribution_bundle,
    replay_relion_double,
    summarize_bpref_contribution_bundle,
)
from recovar.em.local.local_backprojection import enforce_relion_half_volume_x0_hermitian_host
from recovar.em.vdam.layout import relion_bpref_frame_scales

if __package__:
    from scripts.analyze_vdam_mstep_boundary import _read_relion_array
else:
    from analyze_vdam_mstep_boundary import _read_relion_array

SCHEMA = "recovar.vdam_bpref_accumulator_boundary.v1"


def _production_names(half: int, *, iteration: int = 1) -> tuple[str, str, str, str]:
    if iteration < 1:
        raise ValueError(f"iteration must be positive, got {iteration}")
    if half == 1:
        return (
            f"pipe_it{iteration}_c0_bp_data_pre_reweight.bin",
            f"pipe_it{iteration}_c0_bp_weight.bin",
            "accum_h0_data.npy",
            "accum_h0_weight.npy",
        )
    if half == 2:
        return (
            f"pipe_it{iteration}_c0_bp_data_h_pre_reweight.bin",
            f"pipe_it{iteration}_c0_bp_weight_h.bin",
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


def _to_relion_bpref_frame(
    replay: BPrefAccumulatorReplay,
    *,
    ori_size: int,
) -> BPrefAccumulatorReplay:
    """Convert a raw scatter replay to RELION's stored BPref FFT frame."""

    data_scale, weight_scale = relion_bpref_frame_scales(ori_size)
    return BPrefAccumulatorReplay(
        data=np.asarray(replay.data) * data_scale,
        weight=np.asarray(replay.weight) * weight_scale,
        backend=f"{replay.backend}_relion_bpref_frame",
        order=replay.order,
        precision=replay.precision,
        launch_topology=replay.launch_topology,
    )


def _inline_projector_replays(
    bundle,
    *,
    reconstruction_group: int,
    ori_size: int,
) -> tuple[dict[str, BPrefAccumulatorReplay], dict[str, object]]:
    """Sum exact one-particle fused-projector outputs for one joint halfset."""

    data_parts = []
    weight_parts = []
    identities = []
    for shard in bundle.shards:
        values = shard.values
        data = np.asarray(values["inline_projector_data_volumes"])
        weight = np.asarray(values["inline_projector_weight_volumes"])
        original = np.asarray(values["inline_projector_original_indices"], dtype=np.int64)
        if data.size == 0:
            if weight.size or original.size:
                raise ValueError("incomplete inline-projector BPref capture")
            continue
        if data.ndim != 2 or weight.shape != data.shape or original.shape != (data.shape[0],):
            raise ValueError("inline-projector BPref contribution topology is malformed")
        particle_original = np.asarray(values["original_indices"], dtype=np.int64)
        particle_groups = np.asarray(values["reconstruction_group_ids"], dtype=np.int32)
        if particle_groups.shape != particle_original.shape:
            raise ValueError("inline-projector capture lacks joint-halfset group ownership")
        group_by_original = {
            int(original_index): int(group)
            for original_index, group in zip(particle_original, particle_groups)
        }
        try:
            inline_groups = np.asarray(
                [group_by_original[int(original_index)] for original_index in original],
                dtype=np.int32,
            )
        except KeyError as exc:
            raise ValueError(
                "inline-projector identity does not close against captured particles"
            ) from exc
        selected = inline_groups == int(reconstruction_group)
        data_parts.extend(np.asarray(data[selected], dtype=np.complex64))
        weight_parts.extend(np.asarray(weight[selected], dtype=np.float32))
        identities.extend(original[selected].tolist())
    if not data_parts:
        return {}, {"status": "not_captured"}
    if len(set(identities)) != len(identities):
        raise ValueError("inline-projector particle identities are duplicated")

    volume_shape = tuple(
        int(value) for value in np.asarray(bundle.boundary_values["volume_shape"])
    )
    half_shape = (*volume_shape[:2], volume_shape[2] // 2 + 1)
    expected_size = int(np.prod(half_shape))
    if any(np.asarray(part).size != expected_size for part in data_parts + weight_parts):
        raise ValueError("inline-projector accumulator size differs from capture boundary")

    def _sum(dtype_data, dtype_weight, precision):
        data_total = np.zeros((expected_size,), dtype=dtype_data)
        weight_total = np.zeros((expected_size,), dtype=dtype_weight)
        for data_part, weight_part in zip(data_parts, weight_parts):
            data_total += np.asarray(data_part, dtype=dtype_data).reshape(-1)
            weight_total += np.asarray(weight_part, dtype=dtype_weight).reshape(-1)
        data_total = enforce_relion_half_volume_x0_hermitian_host(
            data_total,
            volume_shape,
        )
        weight_total = enforce_relion_half_volume_x0_hermitian_host(
            weight_total,
            volume_shape,
        )
        return _to_relion_bpref_frame(
            BPrefAccumulatorReplay(
                data=np.asarray(data_total).reshape(half_shape),
                weight=np.asarray(weight_total).reshape(half_shape),
                backend="recovar_inline_vdam_projector_particle_sum",
                order="captured_particle_execution",
                precision=precision,
                launch_topology="one_fused_projector_launch_per_particle_then_host_sum",
            ),
            ori_size=ori_size,
        )

    replays = {
        "sequential_float32": _sum(
            np.complex64,
            np.float32,
            "per-particle complex64/float32; sequential complex64/float32 host sum",
        ),
        "sequential_float64": _sum(
            np.complex128,
            np.float64,
            "per-particle complex64/float32; sequential complex128/float64 host sum",
        ),
    }
    identity_bytes = np.asarray(identities, dtype=np.int64).tobytes(order="C")
    return replays, {
        "status": "complete",
        "reconstruction_group": int(reconstruction_group),
        "particle_count": len(identities),
        "particle_original_indices_sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "first_particle_original_index": int(identities[0]),
        "last_particle_original_index": int(identities[-1]),
    }


def analyze(
    contribution_paths: list[Path],
    native_directory: Path,
    recovar_directory: Path,
    *,
    half: int,
    iteration: int = 1,
    reconstruction_group: int | None = None,
) -> dict[str, object]:
    bundle = load_bpref_contribution_bundle(contribution_paths)
    boundary_half = int(np.asarray(bundle.boundary_values["half"]).item())
    if reconstruction_group is None and boundary_half != half:
        raise ValueError(f"captured half {boundary_half} does not match requested half {half}")
    if reconstruction_group is not None and int(reconstruction_group) != half - 1:
        raise ValueError(
            "InitialModel reconstruction groups are zero-based and must equal half - 1"
        )
    native_data_name, native_weight_name, candidate_data_name, candidate_weight_name = (
        _production_names(half, iteration=iteration)
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

    ori_size = int(np.asarray(bundle.boundary_values["image_shape"])[0])
    deterministic = {
        order: _to_relion_bpref_frame(
            replay_relion_double(
                bundle,
                order=order,
                get_backprojector_data=get_backprojector_data,
                interpolator=TRILINEAR,
                reconstruction_group=reconstruction_group,
            ),
            ori_size=ori_size,
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
    inline_replays = {}
    inline_summary = {"status": "not_requested"}
    if reconstruction_group is not None:
        inline_replays, inline_summary = _inline_projector_replays(
            bundle,
            reconstruction_group=reconstruction_group,
            ori_size=ori_size,
        )
        for name, replay in inline_replays.items():
            comparisons[f"candidate_vs_inline_projector_{name}"] = (
                accumulator_replay_metrics(candidate, replay)
            )
            comparisons[f"native_vs_inline_projector_{name}"] = (
                accumulator_replay_metrics(native, replay)
            )
        if len(inline_replays) == 2:
            comparisons["inline_projector_float32_vs_float64"] = (
                accumulator_replay_metrics(
                    inline_replays["sequential_float32"],
                    inline_replays["sequential_float64"],
                )
            )
    execution = deterministic["execution"]
    geometry = {
        "data": _geometry(candidate.data, native.data, execution.data),
        "weight": _geometry(candidate.weight, native.weight, execution.weight),
    }
    inline_geometry = {}
    for name, replay in inline_replays.items():
        inline_geometry[name] = {
            "data": _geometry(candidate.data, native.data, replay.data),
            "weight": _geometry(candidate.weight, native.weight, replay.weight),
        }
    ranking = _rank_particle_sources(
        bundle.concatenate(
            "execution",
            reconstruction_group=reconstruction_group,
        )
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "iteration": int(iteration),
        "half": int(half),
        "capture_context_half": boundary_half,
        "reconstruction_group": (
            None if reconstruction_group is None else int(reconstruction_group)
        ),
        "bundle": summarize_bpref_contribution_bundle(bundle),
        "comparisons": comparisons,
        "relion_double_execution_geometry": geometry,
        "inline_projector": inline_summary,
        "inline_projector_geometry": inline_geometry,
        "particle_source_ranking": ranking,
        "top_particle_source_ranking": ranking[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--native-directory", required=True, type=Path)
    parser.add_argument("--recovar-directory", required=True, type=Path)
    parser.add_argument("--half", required=True, type=int, choices=(1, 2))
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--reconstruction-group", type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        args.inputs,
        args.native_directory,
        args.recovar_directory,
        half=args.half,
        iteration=args.iteration,
        reconstruction_group=args.reconstruction_group,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
