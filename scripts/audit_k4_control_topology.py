#!/usr/bin/env python3
"""Fail-closed K=4 dispatch, schedule, convergence, and finalization audit.

This auditor deliberately does not load maps or compute correlation.  K=4 map
quality remains the responsibility of ``audit_k4_fsc_trajectory.py``, which
uses shellwise FSC and FSC-AUC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from recovar.em.dense_single_volume.relion_worker_scale import (
    load_relion_dispatch_schedule,
    relion_class3d_follower_owners_from_schedule,
    verify_relion_dispatch_schedule_oracle,
)

SCHEMA = "em_k4_control_topology_audit_v1"
N_CLASSES = 4


class AuditError(RuntimeError):
    """Raised when a required control artifact is missing or malformed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _particle_table(path: Path):
    if not path.is_file():
        raise AuditError(f"missing particle STAR: {path}")
    try:
        import starfile

        payload = starfile.read(path)
    except Exception as exc:
        raise AuditError(f"failed to read particle STAR {path}: {exc}") from exc
    if isinstance(payload, Mapping):
        for key in ("particles", "data_particles"):
            if key in payload and hasattr(payload[key], "columns"):
                return payload[key]
        tables = [value for value in payload.values() if hasattr(value, "columns")]
        if len(tables) == 1:
            return tables[0]
        raise AuditError(f"cannot identify one particle table in {path}")
    if not hasattr(payload, "columns"):
        raise AuditError(f"particle STAR {path} did not contain a table")
    return payload


def _column(table, name: str, *, source: Path) -> np.ndarray:
    for candidate in (name, f"_{name}"):
        if candidate in table.columns:
            return np.asarray(table[candidate])
    raise AuditError(f"missing {name} in {source}")


def _image_names(path: Path) -> np.ndarray:
    values = np.asarray(_column(_particle_table(path), "rlnImageName", source=path), dtype=str).reshape(-1)
    if values.size < 1 or np.any(values == ""):
        raise AuditError(f"empty rlnImageName identity in {path}")
    if np.unique(values).size != values.size:
        raise AuditError(f"duplicate rlnImageName identity in {path}")
    return values


def _star_scalar_occurrences(value: Any, key: str) -> list[Any]:
    if not isinstance(value, Mapping):
        return []
    found: list[Any] = []
    for current_key, current_value in value.items():
        if str(current_key).lstrip("_") == key and not isinstance(current_value, Mapping):
            found.append(current_value)
        if isinstance(current_value, Mapping):
            found.extend(_star_scalar_occurrences(current_value, key))
    return found


def _required_star_int(path: Path, key: str) -> int:
    if not path.is_file():
        raise AuditError(f"missing RELION STAR: {path}")
    try:
        import starfile

        payload = starfile.read(path)
    except Exception as exc:
        raise AuditError(f"failed to read RELION STAR {path}: {exc}") from exc
    occurrences = _star_scalar_occurrences(payload, key)
    if len(occurrences) != 1:
        raise AuditError(f"{path} must contain exactly one scalar {key}; found {len(occurrences)}")
    raw = occurrences[0]
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, float, np.integer, np.floating)):
        raise AuditError(f"{path} {key} must be an integer-valued numeric scalar; got {raw!r}")
    value = float(raw)
    if not np.isfinite(value) or value != np.floor(value):
        raise AuditError(f"{path} {key} must be finite and integer-valued; got {raw!r}")
    return int(value)


def _require_key(npz, key: str) -> np.ndarray:
    if key not in npz.files:
        raise AuditError(f"missing {key} in RECOVAR refinement results")
    return np.asarray(npz[key])


def _integer_scalar(npz, key: str, *, minimum: int | None = None) -> int:
    raw = _require_key(npz, key)
    if raw.size != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise AuditError(f"{key} must be an integer scalar; got dtype={raw.dtype} shape={raw.shape}")
    value = int(raw.reshape(()))
    if minimum is not None and value < minimum:
        raise AuditError(f"{key} must be >= {minimum}; got {value}")
    return value


def _boolean_scalar(npz, key: str) -> bool:
    raw = _require_key(npz, key)
    if raw.size != 1 or not np.issubdtype(raw.dtype, np.bool_):
        raise AuditError(f"{key} must be a boolean scalar; got dtype={raw.dtype} shape={raw.shape}")
    return bool(raw.reshape(()))


def _string_scalar(npz, key: str) -> str:
    raw = _require_key(npz, key)
    if raw.size != 1 or raw.dtype.kind not in {"S", "U"}:
        raise AuditError(f"{key} must be a string scalar; got dtype={raw.dtype} shape={raw.shape}")
    value = str(raw.reshape(()))
    if not value:
        raise AuditError(f"{key} must not be empty")
    return value


def _integer_vector(npz, key: str) -> np.ndarray:
    raw = _require_key(npz, key)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise AuditError(f"{key} must be a one-dimensional integer array; got {raw.dtype} {raw.shape}")
    return np.asarray(raw, dtype=np.int64)


def _finite_array(npz, key: str, *, ndim: int) -> np.ndarray:
    raw = _require_key(npz, key)
    if raw.ndim != ndim or not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise AuditError(f"{key} must be a {ndim}-D numeric array; got {raw.dtype} {raw.shape}")
    values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(values).all():
        raise AuditError(f"{key} contains non-finite values")
    return values


def _discover_numbered(prefix: Path, suffix: str) -> dict[int, Path]:
    expression = re.compile(rf"^{re.escape(prefix.name)}_it(\d+){re.escape(suffix)}$")
    found: dict[int, Path] = {}
    for path in prefix.parent.glob(f"{prefix.name}_it*{suffix}"):
        match = expression.fullmatch(path.name)
        if match is None:
            raise AuditError(f"malformed numbered RELION artifact name: {path}")
        iteration = int(match.group(1))
        if iteration == 0:
            continue
        if iteration in found:
            raise AuditError(f"duplicate RELION iteration {iteration} {suffix}")
        found[iteration] = path.resolve()
    return found


def _relion_prefix(relion_dir: Path, particle_relative_path: str) -> Path:
    particle_path = (relion_dir / particle_relative_path).resolve()
    try:
        particle_path.relative_to(relion_dir)
    except ValueError as exc:
        raise AuditError(f"dispatch particle STAR escapes RELION directory: {particle_path}") from exc
    match = re.fullmatch(r"(.+)_it000_data\.star", particle_path.name)
    if match is None:
        raise AuditError(
            "dispatch particle_star_relative_path must name a run_it000_data.star-style artifact; "
            f"got {particle_relative_path!r}"
        )
    return particle_path.with_name(match.group(1))


def _final_artifacts(prefix: Path) -> dict[str, Path]:
    return {name: Path(f"{prefix}_{name}.star").resolve() for name in ("data", "model", "sampling", "optimiser")}


def audit(
    *,
    recovar_results: Path,
    recovar_particles_star: Path,
    relion_dir: Path,
    dispatch_schedule: Path,
) -> dict[str, Any]:
    recovar_results = recovar_results.expanduser().resolve()
    recovar_particles_star = recovar_particles_star.expanduser().resolve()
    relion_dir = relion_dir.expanduser().resolve()
    dispatch_schedule = dispatch_schedule.expanduser().resolve()
    if not recovar_results.is_file():
        raise AuditError(f"missing RECOVAR refinement results: {recovar_results}")
    if not relion_dir.is_dir():
        raise AuditError(f"missing RELION directory: {relion_dir}")
    if not dispatch_schedule.is_file():
        raise AuditError(f"missing dispatch schedule: {dispatch_schedule}")

    try:
        schedule = load_relion_dispatch_schedule(dispatch_schedule)
        verify_relion_dispatch_schedule_oracle(schedule, relion_dir)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise AuditError(f"invalid or unbound RELION dispatch schedule: {exc}") from exc

    prefix = _relion_prefix(relion_dir, schedule.particle_star_relative_path)
    numbered_artifacts = {
        name: _discover_numbered(prefix, f"_{name}.star")
        for name in ("data", "model", "sampling", "optimiser")
    }
    models = numbered_artifacts["model"]
    optimisers = numbered_artifacts["optimiser"]
    manifested = set(schedule.oracle_artifact_paths)
    for artifacts in numbered_artifacts.values():
        for path in artifacts.values():
            relative = path.relative_to(relion_dir).as_posix()
            if relative not in manifested:
                raise AuditError(f"required consumed RELION control artifact is not manifest-bound: {relative}")
    final_paths = _final_artifacts(prefix)
    final_presence = {name: path.is_file() for name, path in final_paths.items()}
    if any(final_presence.values()) and not all(final_presence.values()):
        raise AuditError(f"partial RELION final all-data control artifact set: {final_presence}")
    relion_final_ran = all(final_presence.values())
    if relion_final_ran:
        for path in final_paths.values():
            relative = path.relative_to(relion_dir).as_posix()
            if relative not in manifested:
                raise AuditError(f"RELION final control artifact is not manifest-bound: {relative}")

    recovar_names = _image_names(recovar_particles_star)
    oracle_particles = (relion_dir / schedule.particle_star_relative_path).resolve()
    oracle_names = _image_names(oracle_particles)
    if set(recovar_names) != set(oracle_names):
        raise AuditError("RECOVAR and manifest-bound RELION particle identity sets differ")
    oracle_row = {identity: index for index, identity in enumerate(oracle_names.tolist())}
    particle_ids_by_recovar_row = np.asarray(
        [oracle_row[identity] for identity in recovar_names.tolist()], dtype=np.int64
    )

    failures: list[str] = []
    with np.load(recovar_results, allow_pickle=False) as npz:
        n_images = _integer_scalar(npz, "n_images", minimum=1)
        if n_images != recovar_names.size or n_images != schedule.owner_by_sorted_position.shape[1]:
            raise AuditError(
                "particle count mismatch across RECOVAR results/input and dispatch schedule: "
                f"results={n_images} input={recovar_names.size} schedule={schedule.owner_by_sorted_position.shape[1]}"
            )
        half1 = _integer_vector(npz, "half1_indices")
        half2 = _integer_vector(npz, "half2_indices")
        if not np.array_equal(np.sort(half1), np.arange(n_images)) or half2.size != 0:
            raise AuditError(
                "strict K=4 Class3D topology requires half1_indices to cover every image exactly once "
                "and half2_indices to be empty"
            )

        current_sizes = _integer_vector(npz, "current_sizes")
        if current_sizes.size < 1:
            raise AuditError("current_sizes must contain at least one numbered K=4 iteration")
        numbered_iterations = list(range(1, current_sizes.size + 1))
        recovar_final_ran = _boolean_scalar(npz, "final_all_data_ran")
        schedule_iterations = [int(value) for value in schedule.relion_iterations]
        expected_schedule_iterations = numbered_iterations + ([numbered_iterations[-1] + 1] if relion_final_ran else [])
        if schedule_iterations != expected_schedule_iterations:
            raise AuditError(
                "dispatch schedule does not exactly cover every numbered and final RECOVAR control boundary: "
                f"schedule={schedule_iterations} expected={expected_schedule_iterations}"
            )
        numbered_topology = {name: sorted(artifacts) for name, artifacts in numbered_artifacts.items()}
        if any(iterations != numbered_iterations for iterations in numbered_topology.values()):
            raise AuditError(
                "RELION numbered data/model/sampling/optimiser topology does not exactly match "
                "RECOVAR's numbered trajectory: "
                f"numbered={numbered_iterations} artifacts={numbered_topology}"
            )
        if np.any(current_sizes <= 0):
            raise AuditError(f"current_sizes must be positive; got {current_sizes.tolist()}")

        class_weights = _finite_array(npz, "class_weights", ndim=1)
        class_weight_trajectory = _finite_array(npz, "class_weight_trajectory", ndim=2)
        if class_weights.shape != (N_CLASSES,) or class_weight_trajectory.shape != (
            len(numbered_iterations),
            N_CLASSES,
        ):
            raise AuditError(
                "RECOVAR results are not a complete K=4 trajectory: "
                f"class_weights={class_weights.shape} trajectory={class_weight_trajectory.shape}"
            )

        result_hashes = {
            "oracle_id": _string_scalar(npz, "relion_dispatch_oracle_id"),
            "oracle_manifest_sha256": _string_scalar(npz, "relion_dispatch_oracle_manifest_sha256"),
            "particle_order_sha256": _string_scalar(npz, "relion_dispatch_particle_order_sha256"),
        }
        schedule_hashes = {
            "oracle_id": schedule.oracle_id,
            "oracle_manifest_sha256": schedule.oracle_manifest_sha256,
            "particle_order_sha256": schedule.particle_order_sha256,
        }
        for key, expected in schedule_hashes.items():
            if result_hashes[key] != expected:
                failures.append(f"dispatch {key} differs: RECOVAR={result_hashes[key]!r} schedule={expected!r}")

        owners_raw = _require_key(npz, "relion_scale_follower_owners_half1_trajectory")
        if owners_raw.shape != (len(numbered_iterations), n_images) or not np.issubdtype(owners_raw.dtype, np.integer):
            raise AuditError(
                "relion_scale_follower_owners_half1_trajectory must have integer shape "
                f"{(len(numbered_iterations), n_images)}; got {owners_raw.dtype} {owners_raw.shape}"
            )
        observed_owners = np.asarray(owners_raw, dtype=np.int64)
        expected_owners = np.stack(
            [
                relion_class3d_follower_owners_from_schedule(
                    schedule,
                    particle_ids_by_image=particle_ids_by_recovar_row[half1],
                    optics_group_ids_by_image=np.zeros(n_images, dtype=np.int64),
                    random_seed=schedule.random_seed,
                    relion_iteration=iteration,
                )
                for iteration in schedule_iterations
            ]
        )
        owner_checks = []
        for row, iteration in enumerate(numbered_iterations):
            differing = int(np.count_nonzero(observed_owners[row] != expected_owners[row]))
            owner_checks.append(
                {
                    "boundary": "numbered",
                    "relion_iteration": iteration,
                    "exact": differing == 0,
                    "different": differing,
                }
            )
            if differing:
                failures.append(f"dispatch owners differ at RELION iteration {iteration}: {differing} particles")
        final_owners = _integer_vector(npz, "relion_scale_follower_owners_half1")
        if final_owners.shape != (n_images,):
            raise AuditError(
                f"relion_scale_follower_owners_half1 must have shape {(n_images,)}; got {final_owners.shape}"
            )
        expected_final_owner_row = expected_owners[-1]
        final_owner_differing = int(np.count_nonzero(final_owners != expected_final_owner_row))
        owner_checks.append(
            {
                "boundary": "final_all_data" if relion_final_ran else "final_state",
                "relion_iteration": schedule_iterations[-1],
                "exact": final_owner_differing == 0,
                "different": final_owner_differing,
            }
        )
        if final_owner_differing:
            failures.append(
                "final follower-owner telemetry differs from the expected "
                f"RELION iteration {schedule_iterations[-1]} row: {final_owner_differing} particles"
            )

        scale_shapes = {}
        for key in (
            "relion_scale_follower_scales_numbered_pre_score_trajectory",
            "relion_scale_follower_scales_numbered_post_mstep_trajectory",
        ):
            values = _finite_array(npz, key, ndim=3)
            if values.shape[0] != len(numbered_iterations) or values.shape[1] != schedule.n_followers:
                raise AuditError(
                    f"{key} must cover every numbered iteration/follower; got {values.shape}, "
                    f"expected first axes {(len(numbered_iterations), schedule.n_followers)}"
                )
            scale_shapes[key] = list(values.shape)

        size_checks = []
        optimiser_states = []
        for row, iteration in enumerate(numbered_iterations):
            model_classes = _required_star_int(models[iteration], "rlnNrClasses")
            if model_classes != N_CLASSES:
                raise AuditError(f"{models[iteration]} rlnNrClasses={model_classes}, expected {N_CLASSES}")
            relion_size = _required_star_int(models[iteration], "rlnCurrentImageSize")
            size_equal = int(current_sizes[row]) == relion_size
            size_checks.append(
                {
                    "recovar_index": row,
                    "relion_iteration": iteration,
                    "recovar": int(current_sizes[row]),
                    "relion": relion_size,
                    "exact": size_equal,
                }
            )
            if not size_equal:
                failures.append(
                    f"current_size differs at RELION iteration {iteration}: "
                    f"RECOVAR={int(current_sizes[row])} RELION={relion_size}"
                )
            current_iteration = _required_star_int(optimisers[iteration], "rlnCurrentIteration")
            has_converged = _required_star_int(optimisers[iteration], "rlnHasConverged")
            if current_iteration != iteration or has_converged not in (0, 1):
                raise AuditError(
                    f"invalid optimiser control state at iteration {iteration}: "
                    f"current={current_iteration} converged={has_converged}"
                )
            optimiser_states.append({"relion_iteration": iteration, "has_converged": bool(has_converged)})
        if any(item["has_converged"] for item in optimiser_states[:-1]):
            raise AuditError("RELION numbered trajectory continues after an optimiser reports convergence")

        if relion_final_ran:
            final_iteration_value = _required_star_int(final_paths["optimiser"], "rlnCurrentIteration")
            final_has_converged = _required_star_int(final_paths["optimiser"], "rlnHasConverged")
            if final_iteration_value != -1 or final_has_converged != 1:
                raise AuditError(
                    "RELION final optimiser must record the converged unnumbered state "
                    f"(CurrentIteration=-1, HasConverged=1); got {final_iteration_value}, {final_has_converged}"
                )
            relion_has_converged = True
        else:
            relion_has_converged = optimiser_states[-1]["has_converged"]
            if relion_has_converged:
                raise AuditError("RELION reports numbered convergence but has no complete final all-data control set")

        recovar_convergence = {
            "iteration": _integer_scalar(npz, "convergence_iteration", minimum=0),
            "has_converged": _boolean_scalar(npz, "convergence_has_converged"),
        }
        relion_convergence = {
            "iteration": numbered_iterations[-1],
            "has_converged": bool(relion_has_converged),
        }
        for key in ("iteration", "has_converged"):
            if recovar_convergence[key] != relion_convergence[key]:
                failures.append(
                    f"convergence {key} differs: RECOVAR={recovar_convergence[key]} RELION={relion_convergence[key]}"
                )

        if recovar_final_ran != relion_final_ran:
            failures.append(
                "finalization path differs: "
                f"RECOVAR final_all_data_ran={recovar_final_ran} RELION final_all_data_ran={relion_final_ran}"
            )

    return {
        "schema": SCHEMA,
        "status": "pass" if not failures else "fail",
        "metric_policy": (
            "Exact K=4 control/topology only; no correlation or map metric is computed. "
            "Maps are gated separately by shellwise FSC and FSC-AUC."
        ),
        "sources": {
            "recovar_results": str(recovar_results),
            "recovar_particles_star": str(recovar_particles_star),
            "relion_dir": str(relion_dir),
            "dispatch_schedule": str(dispatch_schedule),
            "dispatch_schedule_sha256": _sha256_file(dispatch_schedule),
        },
        "k4_topology": {
            "n_classes": N_CLASSES,
            "n_images": n_images,
            "numbered_iterations": numbered_iterations,
            "relion_numbered_control_artifacts": numbered_topology,
            "dispatch_iterations": schedule_iterations,
        },
        "dispatch": {
            "oracle_content_verified": True,
            "schedule": schedule_hashes,
            "recovar": result_hashes,
            "hashes_exact": schedule_hashes == result_hashes,
            "owner_checks": owner_checks,
            "all_iterations_consumed_exactly": all(item["exact"] for item in owner_checks),
            "scale_trajectory_shapes": scale_shapes,
        },
        "current_size_schedule": size_checks,
        "convergence": {
            "recovar": recovar_convergence,
            "relion": relion_convergence,
            "exact": recovar_convergence == relion_convergence,
            "relion_numbered_states": optimiser_states,
        },
        "finalization": {
            "recovar_final_all_data_ran": recovar_final_ran,
            "relion_final_all_data_ran": relion_final_ran,
            "exact": recovar_final_ran == relion_final_ran,
            "relion_control_artifacts": {
                name: {"path": str(path), "present": final_presence[name]} for name, path in final_paths.items()
            },
        },
        "combined_control_pass": not failures,
        "failures": failures,
        "earliest_failure": failures[0] if failures else None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-results", required=True, type=Path)
    parser.add_argument("--recovar-particles-star", required=True, type=Path)
    parser.add_argument("--relion-dir", required=True, type=Path)
    parser.add_argument("--dispatch-schedule", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_json.expanduser().resolve()
    try:
        report = audit(
            recovar_results=args.recovar_results,
            recovar_particles_star=args.recovar_particles_star,
            relion_dir=args.relion_dir,
            dispatch_schedule=args.dispatch_schedule,
        )
        status = 0 if report["status"] == "pass" else 1
    except (AuditError, OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        report = {
            "schema": SCHEMA,
            "status": "error",
            "metric_policy": (
                "Exact K=4 control/topology only; no correlation or map metric is computed. "
                "Maps are gated separately by shellwise FSC and FSC-AUC."
            ),
            "error": str(exc),
            "combined_control_pass": False,
        }
        status = 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
