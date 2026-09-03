#!/usr/bin/env python3
"""Compare direct and candidate VDAM transitions from one exact live state.

The ordinary InitialModel artifacts intentionally omit the large VDAM
gradient-moment state, so independently restarted trajectories cannot prove
which implementation caused the first difference.  This diagnostic runs the
direct implementation through a requested checkpoint, retains that exact
in-memory state, and then executes a direct/candidate/candidate/direct (ABBA)
panel for precisely one next iteration. Every arm receives independent deep
copies of the same model, particle, and sampling state.

This is a diagnostic harness, not a production continuation interface.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

SCHEMA = "recovar.vdam_hybrid_same_state_transition.v3"
ARM_ORDER = ("direct_1", "hybrid_1", "hybrid_2", "direct_2")
FLAT_ROW_ARM_ORDER = ("direct_1", "flat_rows_1", "flat_rows_2", "direct_2")
PACKED_PROJECTION_ARM_ORDER = (
    "direct_1",
    "packed_projection_1",
    "packed_projection_2",
    "direct_2",
)
PACKED_DEFERRED_ARM_ORDER = (
    "direct_1",
    "packed_deferred_1",
    "packed_deferred_2",
    "direct_2",
)
HYBRID_PACKED_DEFERRED_ARM_ORDER = (
    "direct_1",
    "hybrid_packed_deferred_1",
    "hybrid_packed_deferred_2",
    "direct_2",
)
HYBRID_ENVIRONMENT = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE",
)
FLAT_ROW_ENVIRONMENT = "RECOVAR_INITIAL_MODEL_FLAT_LOCAL_ROWS"
PACKED_PROJECTION_ENVIRONMENT = "RECOVAR_INITIAL_MODEL_PACKED_LOCAL_PROJECTION"
PACKED_DEFERRED_ENVIRONMENT = "RECOVAR_INITIAL_MODEL_DEFER_PACKED_VDAM"
CANDIDATE_MODES = (
    "hybrid",
    "flat_rows",
    "packed_projection",
    "packed_deferred",
    "hybrid_packed_deferred",
)
META_ARRAY_KEYS = (
    "selected_particle_ids",
    "best_pose_rotation_ids",
    "pose_assignments",
    "class_assignments",
    "best_pose_translations",
    "max_posterior_per_image",
    "significant_counts",
    "cutoff_counts",
    "relion_f32_sum_weight",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--acceptance-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-iteration", type=int, default=34)
    parser.add_argument("--image-batch-size", type=int, default=500)
    parser.add_argument("--candidate-mode", choices=CANDIDATE_MODES, default="hybrid")
    return parser.parse_args(argv)


def _arm_order(candidate_mode: str) -> tuple[str, str, str, str]:
    if candidate_mode == "hybrid":
        return ARM_ORDER
    if candidate_mode == "flat_rows":
        return FLAT_ROW_ARM_ORDER
    if candidate_mode == "packed_projection":
        return PACKED_PROJECTION_ARM_ORDER
    if candidate_mode == "packed_deferred":
        return PACKED_DEFERRED_ARM_ORDER
    if candidate_mode == "hybrid_packed_deferred":
        return HYBRID_PACKED_DEFERRED_ARM_ORDER
    raise ValueError(f"unsupported same-state candidate mode: {candidate_mode}")


def _candidate_environment(candidate_mode: str, *, enabled: bool) -> dict[str, str]:
    values = {
        **{name: "0" for name in HYBRID_ENVIRONMENT},
        FLAT_ROW_ENVIRONMENT: "0",
        PACKED_PROJECTION_ENVIRONMENT: "0",
        PACKED_DEFERRED_ENVIRONMENT: "0",
    }
    if enabled:
        if candidate_mode == "hybrid":
            values.update({name: "1" for name in HYBRID_ENVIRONMENT})
        elif candidate_mode == "flat_rows":
            values[FLAT_ROW_ENVIRONMENT] = "1"
        elif candidate_mode == "packed_projection":
            values[FLAT_ROW_ENVIRONMENT] = "1"
            values[PACKED_PROJECTION_ENVIRONMENT] = "1"
        elif candidate_mode == "packed_deferred":
            values[FLAT_ROW_ENVIRONMENT] = "1"
            values[PACKED_PROJECTION_ENVIRONMENT] = "1"
            values[PACKED_DEFERRED_ENVIRONMENT] = "1"
        elif candidate_mode == "hybrid_packed_deferred":
            values.update({name: "1" for name in HYBRID_ENVIRONMENT})
            values[FLAT_ROW_ENVIRONMENT] = "1"
            values[PACKED_PROJECTION_ENVIRONMENT] = "1"
            values[PACKED_DEFERRED_ENVIRONMENT] = "1"
        else:
            raise ValueError(f"unsupported same-state candidate mode: {candidate_mode}")
    return values


def _candidate_uses_hybrid(candidate_mode: str) -> bool:
    return candidate_mode in {"hybrid", "hybrid_packed_deferred"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@contextmanager
def _temporary_environment(values: dict[str, str | None]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _array_manifest(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    result: dict[str, Any] = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
        "size": int(array.size),
    }
    if array.size and array.dtype.kind in "biufc":
        absolute = np.abs(array)
        result.update(
            finite=bool(np.all(np.isfinite(array))),
            max_abs=float(np.max(absolute)),
            l2_norm=float(np.linalg.norm(array.reshape(-1))),
        )
    return result


def _array_comparison(left: Any, right: Any) -> dict[str, Any]:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    result: dict[str, Any] = {
        "left_dtype": lhs.dtype.str,
        "right_dtype": rhs.dtype.str,
        "left_shape": list(lhs.shape),
        "right_shape": list(rhs.shape),
        "comparable": bool(lhs.shape == rhs.shape),
    }
    if lhs.shape != rhs.shape:
        result["exact_equal"] = False
        return result
    exact = np.array_equal(lhs, rhs, equal_nan=True)
    result["exact_equal"] = bool(exact)
    unequal = ~(lhs == rhs)
    if lhs.dtype.kind in "fc" or rhs.dtype.kind in "fc":
        unequal &= ~(np.isnan(lhs) & np.isnan(rhs))
    result["mismatch_count"] = int(np.count_nonzero(unequal))
    if np.any(unequal):
        result["first_mismatch_flat_indices"] = np.flatnonzero(unequal.reshape(-1))[:32].tolist()
    if lhs.dtype.kind in "biufc" and rhs.dtype.kind in "biufc" and lhs.size:
        delta = lhs.astype(np.complex128 if (lhs.dtype.kind == "c" or rhs.dtype.kind == "c") else np.float64) - rhs
        absolute = np.abs(delta)
        scale = max(
            float(np.linalg.norm(lhs.reshape(-1))),
            float(np.linalg.norm(rhs.reshape(-1))),
            float(np.finfo(np.float64).tiny),
        )
        result.update(
            max_abs_delta=float(np.max(absolute)),
            normalized_l2_delta=float(np.linalg.norm(delta.reshape(-1)) / scale),
        )
        if not np.iscomplexobj(delta):
            result["signed_mean_delta"] = float(np.mean(delta, dtype=np.float64))
    return result


def _dataclass_manifest(value: Any) -> dict[str, Any]:
    if not dataclasses.is_dataclass(value):
        raise TypeError("value must be a dataclass instance")
    fields: dict[str, Any] = {}
    for field in dataclasses.fields(value):
        item = getattr(value, field.name)
        fields[field.name] = (
            {"kind": "none"}
            if item is None
            else {"kind": "array", **_array_manifest(item)}
            if isinstance(item, np.ndarray)
            else {"kind": "scalar", "value": _json_ready(item)}
        )
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return {"fields": fields, "manifest_sha256": _sha256_bytes(encoded)}


def _dataclass_comparison(left: Any, right: Any) -> dict[str, Any]:
    if type(left) is not type(right) or not dataclasses.is_dataclass(left):
        raise TypeError("values must be matching dataclass instances")
    result: dict[str, Any] = {}
    for field in dataclasses.fields(left):
        lhs = getattr(left, field.name)
        rhs = getattr(right, field.name)
        if lhs is None or rhs is None:
            result[field.name] = {"exact_equal": lhs is None and rhs is None}
        elif isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray):
            result[field.name] = _array_comparison(lhs, rhs)
        else:
            result[field.name] = {"exact_equal": bool(lhs == rhs), "left": _json_ready(lhs), "right": _json_ready(rhs)}
    return result


def _accumulator_manifest(accumulators: list[Any]) -> list[dict[str, Any]]:
    return [_dataclass_manifest(value) for value in accumulators]


def _accumulator_comparison(left: list[Any], right: list[Any]) -> dict[str, Any]:
    if len(left) != len(right):
        return {"comparable": False, "left_count": len(left), "right_count": len(right)}
    return {
        "comparable": True,
        "entries": [
            _dataclass_comparison(lhs, rhs) for lhs, rhs in zip(left, right)
        ],
    }


def _meta_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in META_ARRAY_KEYS:
        if key in left or key in right:
            result[key] = (
                _array_comparison(left[key], right[key])
                if key in left and key in right
                else {"exact_equal": False, "left_present": key in left, "right_present": key in right}
            )
    left_ids = np.asarray(left.get("selected_particle_ids", []), dtype=np.int64)
    right_ids = np.asarray(right.get("selected_particle_ids", []), dtype=np.int64)
    if np.array_equal(left_ids, right_ids):
        for key in ("best_pose_rotation_ids", "pose_assignments", "significant_counts"):
            if key not in left or key not in right:
                continue
            lhs = np.asarray(left[key])
            rhs = np.asarray(right[key])
            if lhs.shape == rhs.shape == left_ids.shape:
                mask = lhs != rhs
                result[key]["mismatching_selected_particle_ids"] = left_ids[mask].tolist()
    return result


def _support_audits(meta: dict[str, Any]) -> dict[str, Any]:
    """Return every support audit, including audits nested in profile summaries."""
    result: dict[str, Any] = {}
    for key, value in meta.items():
        if "coarse_significance_support_audit" in key:
            result[key] = _json_ready(value)
        if isinstance(value, dict) and "coarse_significance_support_audit" in value:
            result[f"{key}.coarse_significance_support_audit"] = _json_ready(
                value["coarse_significance_support_audit"]
            )
    return result


def _support_audit_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    lhs_audits = _support_audits(left)
    rhs_audits = _support_audits(right)
    entries: dict[str, Any] = {}
    for key in sorted(set(lhs_audits) | set(rhs_audits)):
        lhs = lhs_audits.get(key)
        rhs = rhs_audits.get(key)
        entries[key] = {
            "exact_equal": lhs == rhs,
            "left_present": key in lhs_audits,
            "right_present": key in rhs_audits,
            "left_sha256": (
                _sha256_bytes(json.dumps(lhs, sort_keys=True, separators=(",", ":")).encode())
                if key in lhs_audits
                else None
            ),
            "right_sha256": (
                _sha256_bytes(json.dumps(rhs, sort_keys=True, separators=(",", ":")).encode())
                if key in rhs_audits
                else None
            ),
            "left_aggregate_support_sha256": (
                lhs.get("aggregate_support_sha256") if isinstance(lhs, dict) else None
            ),
            "right_aggregate_support_sha256": (
                rhs.get("aggregate_support_sha256") if isinstance(rhs, dict) else None
            ),
        }
    return {
        "exact_equal": bool(entries) and all(entry["exact_equal"] for entry in entries.values()),
        "left_count": len(lhs_audits),
        "right_count": len(rhs_audits),
        "entries": entries,
    }


def _capture_direct_checkpoint(
    *,
    fixture_dir: Path,
    acceptance: dict[str, Any],
    output_root: Path,
    checkpoint_iteration: int,
    image_batch_size: int,
) -> dict[str, Any]:
    import recovar.em.initial_model.driver as driver
    from scripts import run_ab_initio
    from scripts.run_vdam_relion_parity_case import build_recovar_command

    input_star = fixture_dir / "particles.star"
    definition = acceptance["science_contract"]["definition"]
    command = build_recovar_command(
        input_star=input_star,
        output_prefix=output_root / "checkpoint" / "run",
        fixture_dir=fixture_dir,
        definition=definition,
        image_batch_size=image_batch_size,
    )
    argv = list(command[3:])
    write_index = argv.index("--grad_write_iter") + 1
    argv[write_index] = str(checkpoint_iteration)
    argv.extend(("--diagnostic_stop_after_iteration", str(checkpoint_iteration)))

    captured: dict[str, Any] = {"argv": argv}
    original_expectation_factory = driver._native_expectation_step
    original_run = driver.run_native_initial_model

    def capture_expectation_factory(dataset, opts, noise_variance, particle_state, sampling_state=None, optics_state=None):
        captured.update(
            dataset=dataset,
            opts=opts,
            noise_variance=noise_variance,
            particle_state=particle_state,
            sampling_state=sampling_state,
            optics_state=optics_state,
        )
        return original_expectation_factory(
            dataset,
            opts,
            noise_variance,
            particle_state,
            sampling_state,
            optics_state,
        )

    def capture_run(opts):
        result = original_run(opts)
        captured["result"] = result
        return result

    driver._native_expectation_step = capture_expectation_factory
    driver.run_native_initial_model = capture_run
    try:
        with _temporary_environment(
            {
                **_candidate_environment("hybrid", enabled=False),
                "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT": "0",
                "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS": "0",
            }
        ):
            status = int(run_ab_initio.main(argv))
    finally:
        driver._native_expectation_step = original_expectation_factory
        driver.run_native_initial_model = original_run
    if status != 0:
        raise RuntimeError(f"direct checkpoint trajectory exited with status {status}")
    required = {
        "dataset",
        "opts",
        "noise_variance",
        "particle_state",
        "sampling_state",
        "optics_state",
        "result",
    }
    missing = required - set(captured)
    if missing:
        raise RuntimeError(f"direct checkpoint capture is incomplete: {sorted(missing)}")
    if int(captured["result"].state.iter) != checkpoint_iteration:
        raise RuntimeError("direct checkpoint stopped at the wrong iteration")
    captured["expectation_factory"] = original_expectation_factory
    return captured


def _run_transition_arm(
    checkpoint: dict[str, Any],
    *,
    label: str,
    candidate_mode: str,
    candidate_enabled: bool,
    checkpoint_iteration: int,
) -> dict[str, Any]:
    import recovar.em.initial_model.driver as driver
    from recovar.data_io.starfile import read_star
    from recovar.em.initial_model.schedules import default_subset_sizes_for_3d_initial_model

    state = copy.deepcopy(checkpoint["result"].state)
    particle_state = copy.deepcopy(checkpoint["particle_state"])
    sampling_state = copy.deepcopy(checkpoint["sampling_state"])
    initial_state_manifest = _dataclass_manifest(state)
    initial_particle_state_manifest = _dataclass_manifest(particle_state)
    initial_sampling_state_manifest = _dataclass_manifest(sampling_state)
    opts = checkpoint["opts"]
    dataset = checkpoint["dataset"]
    expectation_step = checkpoint["expectation_factory"](
        dataset,
        opts,
        checkpoint["noise_variance"],
        particle_state,
        sampling_state,
        checkpoint["optics_state"],
    )
    captured: dict[str, Any] = {}

    def capture_expectation(current, particle_ids, halfset_ids):
        accumulators, meta = expectation_step(current, particle_ids, halfset_ids)
        captured["accumulators"] = accumulators
        captured["estep_meta"] = copy.deepcopy(meta)
        return accumulators, meta

    def capture_artifact(_current, _iteration, meta):
        captured["post_iteration_meta"] = copy.deepcopy(meta)

    post_mstep_update = None
    if opts.do_solvent:
        solvent_mask = driver.relion_solvent_mask(
            ori_size=int(state.ori_size),
            pixel_size=float(state.pixel_size),
            particle_diameter_ang=float(opts.particle_diameter),
            width_mask_edge_px=float(opts.width_mask_edge_px),
        )

        def post_mstep_update(current, iteration, meta):
            current = driver.relion_solvent_flatten_state(current, mask=solvent_mask)
            return driver._maybe_replay_iteration_references(current, iteration=iteration, meta=meta)

    main_star, _optics_star = read_star(opts.fn_img)
    optics_group_by_particle = driver._optics_group_indices(main_star)
    particle_order = driver._micrograph_sort_order(main_star)
    grad_ini_subset_size, grad_fin_subset_size = default_subset_sizes_for_3d_initial_model(
        int(dataset.n_images)
    )
    started = time.perf_counter()
    with _temporary_environment(
        {
            **_candidate_environment(candidate_mode, enabled=candidate_enabled),
            "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT": "1",
            "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS": "1",
        }
    ):
        final_state = driver.run_vdam_iterations(
            state,
            nr_particles=int(dataset.n_images),
            optics_group_by_particle=optics_group_by_particle,
            grad_ini_subset_size=grad_ini_subset_size,
            grad_fin_subset_size=grad_fin_subset_size,
            tau2_fudge_arg=float(opts.tau2_fudge),
            grad_em_iters=int(opts.grad_em_iters),
            random_seed=int(opts.random_seed),
            rnd_unif_factory=driver._relion_rnd_unif_factory,
            expectation_step=capture_expectation,
            iter_artifact_sink=capture_artifact,
            post_mstep_update=post_mstep_update,
            particle_order=particle_order,
            grad_ini_frac=float(opts.grad_ini_frac),
            grad_fin_frac=float(opts.grad_fin_frac),
            grad_stepsize=float(opts.stepsize),
            mu=float(opts.mu),
            projector_padding_factor=int(opts.padding_factor),
            start_iteration=checkpoint_iteration,
            diagnostic_stop_after_iteration=checkpoint_iteration + 1,
        )
    wall_s = float(time.perf_counter() - started)
    if int(final_state.iter) != checkpoint_iteration + 1:
        raise RuntimeError(f"{label} stopped at the wrong iteration")
    if "accumulators" not in captured or "estep_meta" not in captured:
        raise RuntimeError(f"{label} did not capture its E-step boundary")
    return {
        "label": label,
        "candidate_mode": candidate_mode,
        "candidate_enabled": bool(candidate_enabled),
        "hybrid": bool(candidate_enabled and _candidate_uses_hybrid(candidate_mode)),
        "wall_s": wall_s,
        "initial_state_manifest": initial_state_manifest,
        "initial_particle_state_manifest": initial_particle_state_manifest,
        "initial_sampling_state_manifest": initial_sampling_state_manifest,
        "final_state": final_state,
        "particle_state": particle_state,
        "sampling_state": sampling_state,
        **captured,
    }


def _pair_report(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "left": left["label"],
        "right": right["label"],
        "estep_meta": _meta_comparison(left["estep_meta"], right["estep_meta"]),
        "support_audits": _support_audit_comparison(left["estep_meta"], right["estep_meta"]),
        "accumulators": _accumulator_comparison(left["accumulators"], right["accumulators"]),
        "particle_state": _dataclass_comparison(left["particle_state"], right["particle_state"]),
        "sampling_state": _dataclass_comparison(left["sampling_state"], right["sampling_state"]),
        "final_state": _dataclass_comparison(left["final_state"], right["final_state"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fixture_dir = args.fixture_dir.resolve(strict=True)
    acceptance_path = args.acceptance_config.resolve(strict=True)
    output_root = args.output_root.resolve()
    if args.checkpoint_iteration < 1:
        raise ValueError("checkpoint-iteration must be positive")
    if args.image_batch_size < 1:
        raise ValueError("image-batch-size must be positive")
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    acceptance = json.loads(acceptance_path.read_text())
    frozen_nr_iter = int(acceptance["science_contract"]["definition"]["nr_iter"])
    if args.checkpoint_iteration >= frozen_nr_iter:
        raise ValueError("checkpoint-iteration must leave one next iteration")

    checkpoint_started = time.perf_counter()
    checkpoint = _capture_direct_checkpoint(
        fixture_dir=fixture_dir,
        acceptance=acceptance,
        output_root=output_root,
        checkpoint_iteration=args.checkpoint_iteration,
        image_batch_size=args.image_batch_size,
    )
    checkpoint_wall_s = float(time.perf_counter() - checkpoint_started)
    checkpoint_manifest = {
        "state": _dataclass_manifest(checkpoint["result"].state),
        "particle_state": _dataclass_manifest(checkpoint["particle_state"]),
        "sampling_state": _dataclass_manifest(checkpoint["sampling_state"]),
    }
    arm_order = _arm_order(args.candidate_mode)
    arms: dict[str, dict[str, Any]] = {}
    for label in arm_order:
        arms[label] = _run_transition_arm(
            checkpoint,
            label=label,
            candidate_mode=args.candidate_mode,
            candidate_enabled=not label.startswith("direct"),
            checkpoint_iteration=args.checkpoint_iteration,
        )

    expected_manifests = {
        "initial_state_manifest": checkpoint_manifest["state"]["manifest_sha256"],
        "initial_particle_state_manifest": checkpoint_manifest["particle_state"]["manifest_sha256"],
        "initial_sampling_state_manifest": checkpoint_manifest["sampling_state"]["manifest_sha256"],
    }
    for label, arm in arms.items():
        for key, expected in expected_manifests.items():
            if arm[key]["manifest_sha256"] != expected:
                raise RuntimeError(f"{label} did not start from the exact shared {key}")

    candidate_1, candidate_2 = arm_order[1:3]
    pair_labels = (
        (arm_order[0], arm_order[3]),
        (candidate_1, candidate_2),
        (arm_order[0], candidate_1),
        (arm_order[0], candidate_2),
        (arm_order[3], candidate_1),
        (arm_order[3], candidate_2),
    )
    comparisons = {
        f"{left}__vs__{right}": _pair_report(arms[left], arms[right])
        for left, right in pair_labels
    }
    arm_dir = output_root / "arms"
    arm_dir.mkdir()
    arm_summaries: dict[str, Any] = {}
    for label, arm in arms.items():
        payload = {
            "label": label,
            "hybrid": arm["hybrid"],
            "candidate_mode": arm["candidate_mode"],
            "candidate_enabled": arm["candidate_enabled"],
            "wall_s": arm["wall_s"],
            "initial_state_manifest": arm["initial_state_manifest"],
            "initial_particle_state_manifest": arm["initial_particle_state_manifest"],
            "initial_sampling_state_manifest": arm["initial_sampling_state_manifest"],
            "accumulator_manifest": _accumulator_manifest(arm["accumulators"]),
            "final_state_manifest": _dataclass_manifest(arm["final_state"]),
            "particle_state_manifest": _dataclass_manifest(arm["particle_state"]),
            "sampling_state_manifest": _dataclass_manifest(arm["sampling_state"]),
            "estep_meta": _json_ready(arm["estep_meta"]),
            "post_iteration_meta": _json_ready(arm["post_iteration_meta"]),
        }
        arm_path = arm_dir / f"{label}.json"
        arm_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        arm_summaries[label] = {
            "hybrid": arm["hybrid"],
            "candidate_mode": arm["candidate_mode"],
            "candidate_enabled": arm["candidate_enabled"],
            "wall_s": arm["wall_s"],
            "artifact": str(arm_path.resolve()),
            "artifact_sha256": _sha256_bytes(arm_path.read_bytes()),
            "accumulator_manifest": payload["accumulator_manifest"],
            "final_state_manifest": payload["final_state_manifest"],
            "particle_state_manifest": payload["particle_state_manifest"],
            "sampling_state_manifest": payload["sampling_state_manifest"],
        }

    report = {
        "schema": SCHEMA,
        "classification": "diagnostic_same_in_memory_state_one_transition_only",
        "checkpoint_iteration": int(args.checkpoint_iteration),
        "profiled_iteration": int(args.checkpoint_iteration) + 1,
        "candidate_mode": args.candidate_mode,
        "frozen_nr_iter_schedule": frozen_nr_iter,
        "arm_order": list(arm_order),
        "checkpoint_wall_s": checkpoint_wall_s,
        "checkpoint_manifest": checkpoint_manifest,
        "fixture_dir": str(fixture_dir),
        "acceptance_config": str(acceptance_path),
        "acceptance_config_sha256": _sha256_bytes(acceptance_path.read_bytes()),
        "arms": arm_summaries,
        "comparisons": comparisons,
        "same_state_contract": {
            "model_state_exact_for_every_arm": True,
            "particle_state_exact_for_every_arm": True,
            "sampling_state_exact_for_every_arm": True,
            "baseline_trajectory_backend": "direct",
            "candidate_backend": args.candidate_mode,
            "transition_panel": f"direct/{args.candidate_mode}/{args.candidate_mode}/direct",
            "support_audit_ids_enabled_for_transition_arms": True,
        },
        "science_promotion_allowed": False,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(report_path.resolve()),
                "report_sha256": _sha256_bytes(report_path.read_bytes()),
                "checkpoint_wall_s": checkpoint_wall_s,
                "arm_walls_s": {label: arms[label]["wall_s"] for label in arm_order},
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
