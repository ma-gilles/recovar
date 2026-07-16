#!/usr/bin/env python3
"""Fail-closed gate for an aggregate production BPref capture.

The gate proves that the captured accumulator came from the ordinary flattened
production adjoint, that the recorded x=0 Hermitian and public-layout stages
are exact, and that enabling capture stays inside a same-GPU repeat envelope.
It deliberately does not run canonical or promoted-float64 replay; callers run
those controls only after this program reports ``PASS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar.em.dense_single_volume.helpers import half_volume_mstep  # noqa: E402

MAGIC = "RECOVAR_PRODUCTION_BPREF_ACCUMULATOR"
SCHEMA = "recovar-production-bpref-accumulator-v1"
REPORT_SCHEMA = "recovar.em.bpref-production-capture-closure.v1"
PRODUCTION_TOPOLOGY = "ordinary-flattened-production-adjoint"
PRODUCTION_OPERANDS = "authoritative-ordinary-translation-reduction"
PANEL_TOPOLOGY = "production-accumulator;diagnostic-per-particle-signature"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(archive, key: str):
    if key not in archive:
        raise ValueError(f"missing required field {key!r}")
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"field {key!r} must be scalar, got {value.shape}")
    return value.item()


def load_stage(path: Path, expected_stage: str) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if _scalar(values, "magic") != MAGIC or _scalar(values, "schema") != SCHEMA:
        raise ValueError(f"unknown production capture header: {path}")
    if int(_scalar(values, "schema_version")) != 1:
        raise ValueError(f"unsupported production capture version: {path}")
    if _scalar(values, "stage") != expected_stage:
        raise ValueError(
            f"production capture stage mismatch: {_scalar(values, 'stage')!r} "
            f"!= {expected_stage!r}"
        )
    if _scalar(values, "topology_claim") != PRODUCTION_TOPOLOGY:
        raise ValueError("capture does not claim the ordinary flattened production adjoint")
    if bool(_scalar(values, "arithmetic_mutated")):
        raise ValueError("capture reports that production arithmetic was mutated")
    return values


def array_metrics(lhs, rhs) -> dict[str, object]:
    left = np.asarray(lhs)
    right = np.asarray(rhs)
    if left.shape != right.shape:
        raise ValueError(f"array shape mismatch: {left.shape} != {right.shape}")
    metric_dtype = (
        np.complex128 if np.iscomplexobj(left) or np.iscomplexobj(right) else np.float64
    )
    left_metric = left.astype(metric_dtype, copy=False)
    right_metric = right.astype(metric_dtype, copy=False)
    if not np.all(np.isfinite(left_metric)) or not np.all(np.isfinite(right_metric)):
        raise ValueError("comparison contains nonfinite values")
    absolute = np.abs(left_metric - right_metric)
    lhs_rms = float(np.sqrt(np.mean(np.abs(left_metric) ** 2)))
    delta_rms = float(np.sqrt(np.mean(absolute**2)))
    return {
        "count": int(left.size),
        "array_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "lhs_dtype": str(left.dtype),
        "rhs_dtype": str(right.dtype),
        "lhs_rms": lhs_rms,
        "delta_rms_abs": delta_rms,
        "delta_rms_over_lhs_rms": float(
            delta_rms / max(lhs_rms, np.finfo(np.float64).tiny)
        ),
        "delta_mean_abs": float(np.mean(absolute)),
        "delta_max_abs": float(np.max(absolute, initial=0.0)),
    }


def calibrated_envelope(repeat_metrics: dict[str, object]) -> float:
    repeat = float(repeat_metrics["delta_rms_over_lhs_rms"])
    if not np.isfinite(repeat) or repeat < 0:
        raise ValueError("repeat metric must be finite and nonnegative")
    return max(5.0 * repeat, np.finfo(np.float64).tiny)


def validate_companion(panel_path: Path, contribution_dir: Path) -> dict[str, object]:
    with np.load(panel_path, allow_pickle=False) as panel:
        panel_fields = {key: panel[key] for key in panel.files}
    if _scalar(panel_fields, "operand_source") != PRODUCTION_OPERANDS:
        raise ValueError("signature companion does not contain production-reduced operands")
    if _scalar(panel_fields, "production_adjoint_topology") != PRODUCTION_TOPOLOGY:
        raise ValueError("signature companion production topology is missing or inconsistent")
    if _scalar(panel_fields, "topology_claim") != PANEL_TOPOLOGY:
        raise ValueError("panel topology does not distinguish its diagnostic geometry companion")
    paths = sorted(contribution_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"no contribution artifacts under {contribution_dir}")
    particle_count = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as contribution:
            if _scalar(contribution, "operand_source") != PRODUCTION_OPERANDS:
                raise ValueError(f"non-production operands in {path}")
            if _scalar(contribution, "production_adjoint_topology") != PRODUCTION_TOPOLOGY:
                raise ValueError(f"production topology mismatch in {path}")
            particle_count += int(np.asarray(contribution["original_indices"]).size)
    return {
        "panel_path": str(panel_path.resolve()),
        "panel_topology_claim": PANEL_TOPOLOGY,
        "operand_source": PRODUCTION_OPERANDS,
        "production_adjoint_topology": PRODUCTION_TOPOLOGY,
        "contribution_shards": len(paths),
        "particle_count": particle_count,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-x0", required=True, type=Path)
    parser.add_argument("--post-x0", required=True, type=Path)
    parser.add_argument("--post-public", required=True, type=Path)
    parser.add_argument("--live-data", required=True, type=Path)
    parser.add_argument("--live-weight", required=True, type=Path)
    parser.add_argument("--repeat-a-data", required=True, type=Path)
    parser.add_argument("--repeat-a-weight", required=True, type=Path)
    parser.add_argument("--repeat-b-data", required=True, type=Path)
    parser.add_argument("--repeat-b-weight", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--contribution-dir", required=True, type=Path)
    parser.add_argument("--capture-gpu-uuid", required=True)
    parser.add_argument("--repeat-a-gpu-uuid", required=True)
    parser.add_argument("--repeat-b-gpu-uuid", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    paths = (
        args.pre_x0,
        args.post_x0,
        args.post_public,
        args.live_data,
        args.live_weight,
        args.repeat_a_data,
        args.repeat_a_weight,
        args.repeat_b_data,
        args.repeat_b_weight,
        args.panel,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.contribution_dir.is_dir():
        raise FileNotFoundError(args.contribution_dir)
    gpu_uuids = {
        args.capture_gpu_uuid,
        args.repeat_a_gpu_uuid,
        args.repeat_b_gpu_uuid,
    }
    if len(gpu_uuids) != 1:
        raise ValueError(f"same-GPU calibration required, got {sorted(gpu_uuids)}")

    stages = {
        "pre_x0": load_stage(args.pre_x0, "pre_x0"),
        "post_x0": load_stage(args.post_x0, "post_x0"),
        "post_public_layout": load_stage(args.post_public, "post_public_layout"),
    }
    identity_fields = ("iteration", "half", "run_id", "current_size", "n_images")
    for field in identity_fields:
        values = {_scalar(stage, field) for stage in stages.values()}
        if len(values) != 1:
            raise ValueError(f"capture stage identity mismatch for {field}: {values}")
    shape = tuple(int(value) for value in stages["pre_x0"]["recon_volume_shape"])
    if len(shape) != 3 or len(set(shape)) != 1:
        raise ValueError(f"expected one cubic accumulator shape, got {shape}")

    pre_data = np.asarray(stages["pre_x0"]["Ft_y"])
    pre_weight = np.asarray(stages["pre_x0"]["Ft_ctf"])
    post_data = np.asarray(stages["post_x0"]["Ft_y"])
    post_weight = np.asarray(stages["post_x0"]["Ft_ctf"])
    public_data = np.asarray(stages["post_public_layout"]["Ft_y"]).reshape(shape)
    public_weight = np.asarray(stages["post_public_layout"]["Ft_ctf"]).reshape(shape).real

    expected_post_data = np.asarray(
        half_volume_mstep.enforce_relion_half_volume_x0_hermitian_host(pre_data, shape)
    )
    expected_post_weight = np.asarray(
        half_volume_mstep.enforce_relion_half_volume_x0_hermitian_host(pre_weight, shape)
    )
    expected_public_data = np.asarray(
        half_volume_mstep.relion_x_half_volume_to_full(post_data, shape)
    ).reshape(shape)
    expected_public_weight = np.asarray(
        half_volume_mstep.relion_x_half_volume_to_full(post_weight, shape)
    ).reshape(shape).real

    stage_metrics = {
        "data_pre_to_post_x0": array_metrics(expected_post_data, post_data),
        "weight_pre_to_post_x0": array_metrics(expected_post_weight, post_weight),
        "data_post_x0_to_public": array_metrics(expected_public_data, public_data),
        "weight_post_x0_to_public": array_metrics(expected_public_weight, public_weight),
    }
    stage_exact = all(metrics["array_equal"] for metrics in stage_metrics.values())

    live = {
        "data": np.load(args.live_data).reshape(shape),
        "weight": np.load(args.live_weight).reshape(shape).real,
    }
    repeats = {
        "data": array_metrics(
            np.load(args.repeat_a_data).reshape(shape),
            np.load(args.repeat_b_data).reshape(shape),
        ),
        "weight": array_metrics(
            np.load(args.repeat_a_weight).reshape(shape).real,
            np.load(args.repeat_b_weight).reshape(shape).real,
        ),
    }
    closure = {
        "data": array_metrics(public_data, live["data"]),
        "weight": array_metrics(public_weight, live["weight"]),
    }
    instrumented_vs_repeat_a = {
        "data": array_metrics(np.load(args.repeat_a_data).reshape(shape), live["data"]),
        "weight": array_metrics(
            np.load(args.repeat_a_weight).reshape(shape).real, live["weight"]
        ),
    }
    envelopes = {field: calibrated_envelope(repeats[field]) for field in ("data", "weight")}
    closure_pass = all(
        float(closure[field]["delta_rms_over_lhs_rms"]) <= envelopes[field]
        for field in ("data", "weight")
    )
    inertness_pass = all(
        float(instrumented_vs_repeat_a[field]["delta_rms_over_lhs_rms"])
        <= envelopes[field]
        for field in ("data", "weight")
    )
    companion = validate_companion(args.panel, args.contribution_dir)
    with np.load(args.panel, allow_pickle=False) as panel:
        panel_data = np.asarray(panel["data_accumulator"])
        panel_weight = np.asarray(panel["weight_accumulator"])
    panel_pre_x0 = {
        "data": array_metrics(pre_data, panel_data),
        "weight": array_metrics(pre_weight, panel_weight),
    }
    panel_post_data = np.asarray(
        half_volume_mstep.enforce_relion_half_volume_x0_hermitian_host(
            panel_data, shape
        )
    )
    panel_post_weight = np.asarray(
        half_volume_mstep.enforce_relion_half_volume_x0_hermitian_host(
            panel_weight, shape
        )
    )
    panel_post_x0 = {
        "data": array_metrics(post_data, panel_post_data),
        "weight": array_metrics(post_weight, panel_post_weight),
    }
    panel_closure_pass = all(
        float(panel_pre_x0[field]["delta_rms_over_lhs_rms"]) <= envelopes[field]
        and float(panel_post_x0[field]["delta_rms_over_lhs_rms"])
        <= envelopes[field]
        for field in ("data", "weight")
    )
    passed = bool(
        stage_exact
        and closure_pass
        and inertness_pass
        and panel_closure_pass
    )

    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if passed else "FAIL_PRODUCTION_CLOSURE",
        "canonical_replay_authorized": passed,
        "scope": "aggregate half-set production accumulator; no serial particle tracing",
        "metric_policy": "exact array metrics for intermediate accumulators; no correlation",
        "gpu_uuid": args.capture_gpu_uuid,
        "capture_identity": {
            field: _scalar(stages["pre_x0"], field) for field in identity_fields
        },
        "volume_shape": list(shape),
        "stage_metrics": stage_metrics,
        "stage_exact": stage_exact,
        "repeat_calibration": repeats,
        "normalized_rms_envelopes": envelopes,
        "captured_public_vs_live": closure,
        "instrumented_live_vs_repeat_a": instrumented_vs_repeat_a,
        "closure_pass": closure_pass,
        "capture_inertness_pass": inertness_pass,
        "signature_companion": companion,
        "production_panel_vs_captured_pre_x0": panel_pre_x0,
        "production_panel_vs_captured_post_x0": panel_post_x0,
        "production_panel_closure_pass": panel_closure_pass,
        "next_step": (
            "Run canonical float32 and promoted-float64 controls on the production-operand "
            "signature companion."
            if passed
            else "Reject canonical replay and repair the production capture boundary first."
        ),
        "input_sha256": {str(path.resolve()): sha256(path) for path in paths},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
