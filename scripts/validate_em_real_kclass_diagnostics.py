#!/usr/bin/env python3
"""Validate sealed, rejected real-data K-class InitialModel diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "recovar.em_real_kclass_initialmodel_diagnostics.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_CHECKPOINTS = list(range(1, 9))


class DiagnosticsValidationError(ValueError):
    """Raised when the rejected-run ledger is incomplete or misleading."""


def _require_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - value.keys())
    if missing:
        raise DiagnosticsValidationError(f"{label} is missing keys: {missing}")


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise DiagnosticsValidationError(f"{label} must be an absolute path")
    return Path(value)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise DiagnosticsValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _file_ref(value: dict[str, Any], label: str, *, require_size: bool) -> None:
    required = {"role", "path", "sha256"} | ({"size_bytes"} if require_size else set())
    _require_keys(value, required, label)
    if not isinstance(value["role"], str) or not value["role"]:
        raise DiagnosticsValidationError(f"{label}.role must be non-empty")
    _absolute_path(value["path"], f"{label}.path")
    _sha256(value["sha256"], f"{label}.sha256")
    if require_size and (not isinstance(value["size_bytes"], int) or value["size_bytes"] <= 0):
        raise DiagnosticsValidationError(f"{label}.size_bytes must be positive")


def _validate_performance(value: dict[str, Any], label: str) -> None:
    _require_keys(
        value,
        {"wall_s", "wall_s_source", "peak_hbm_mib", "max_rss_kib", "measurement_limitations"},
        label,
    )
    for key in ("wall_s", "peak_hbm_mib", "max_rss_kib"):
        metric = value[key]
        if not isinstance(metric, (int, float)) or metric <= 0:
            raise DiagnosticsValidationError(f"{label}.{key} must be positive")
    if not isinstance(value["wall_s_source"], str) or not value["wall_s_source"]:
        raise DiagnosticsValidationError(f"{label}.wall_s_source must be non-empty")
    if not isinstance(value["measurement_limitations"], list):
        raise DiagnosticsValidationError(f"{label}.measurement_limitations must be a list")


def _validate_slurm_execution(
    value: dict[str, Any],
    label: str,
    *,
    require_rejected_outcome: bool,
) -> None:
    _require_keys(
        value,
        {"job_id", "state", "exit_code", "elapsed_s", "node", "req_tres", "alloc_tres"},
        label,
    )
    if not str(value["job_id"]).isdigit():
        raise DiagnosticsValidationError(f"{label}.job_id must be numeric")
    if require_rejected_outcome and value["state"] == "COMPLETED" and value["exit_code"] == "0:0":
        raise DiagnosticsValidationError(f"{label} is not a rejected Slurm outcome")
    if value["req_tres"] != value["alloc_tres"]:
        raise DiagnosticsValidationError(f"{label} ReqTRES and AllocTRES differ")
    if not isinstance(value["elapsed_s"], int) or value["elapsed_s"] <= 0:
        raise DiagnosticsValidationError(f"{label}.elapsed_s must be positive")
    if not isinstance(value["node"], str) or not value["node"]:
        raise DiagnosticsValidationError(f"{label}.node must be non-empty")


def validate_diagnostics(data: dict[str, Any]) -> None:
    """Validate one checked diagnostic ledger without touching external files."""

    _require_keys(
        data,
        {"schema", "recorded_at", "admission_policy", "runs", "causal_diagnostics"},
        "ledger",
    )
    if data["schema"] != SCHEMA:
        raise DiagnosticsValidationError(f"unsupported diagnostics schema: {data['schema']!r}")
    if not isinstance(data["admission_policy"], str) or "not benchmark records" not in data[
        "admission_policy"
    ]:
        raise DiagnosticsValidationError("admission_policy must explicitly deny benchmark admission")
    if not isinstance(data["runs"], list) or not data["runs"]:
        raise DiagnosticsValidationError("runs must be a non-empty list")

    identifiers: set[str] = set()
    for index, run in enumerate(data["runs"]):
        label = f"runs[{index}]"
        _require_keys(
            run,
            {
                "diagnostic_id",
                "admission_status",
                "dataset",
                "scope",
                "source",
                "relion_source",
                "inputs",
                "command",
                "execution",
                "outcome",
                "quality",
                "performance",
                "artifacts",
                "rejection_reasons",
            },
            label,
        )
        identifier = run["diagnostic_id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise DiagnosticsValidationError(f"{label}.diagnostic_id must be unique and non-empty")
        identifiers.add(identifier)
        if run["admission_status"] != "REJECTED_DIAGNOSTIC":
            raise DiagnosticsValidationError(f"{identifier} cannot claim benchmark admission")
        if not isinstance(run["rejection_reasons"], list) or not run["rejection_reasons"]:
            raise DiagnosticsValidationError(f"{identifier} requires rejection reasons")
        if not all(isinstance(reason, str) and reason for reason in run["rejection_reasons"]):
            raise DiagnosticsValidationError(f"{identifier} rejection reasons must be non-empty strings")

        scope = run["scope"]
        _require_keys(
            scope,
            {
                "kind",
                "K",
                "particles",
                "checkpoints",
                "symmetry",
                "random_seed",
                "image_batch_size",
                "image_fourier_backend",
                "same_gpu_serial",
                "gold_standard_halfmaps_available",
            },
            f"{identifier}.scope",
        )
        if scope["K"] != 4 or scope["checkpoints"] != EXPECTED_CHECKPOINTS:
            raise DiagnosticsValidationError(f"{identifier} must retain the frozen K=4/it1--8 scope")
        if scope["particles"] != 10_000 or scope["same_gpu_serial"] is not True:
            raise DiagnosticsValidationError(f"{identifier} must retain the frozen 10k same-GPU scope")
        if scope["gold_standard_halfmaps_available"] is not False:
            raise DiagnosticsValidationError(f"{identifier} InitialModel diagnostic cannot claim half maps")

        source = run["source"]
        _require_keys(source, {"commit", "tree", "checkout", "clean"}, f"{identifier}.source")
        if GIT_SHA_RE.fullmatch(str(source["commit"])) is None or GIT_SHA_RE.fullmatch(
            str(source["tree"])
        ) is None:
            raise DiagnosticsValidationError(f"{identifier} source commit/tree must be full Git SHAs")
        _absolute_path(source["checkout"], f"{identifier}.source.checkout")
        if source["clean"] is not True:
            raise DiagnosticsValidationError(f"{identifier} source must be clean")

        relion = run["relion_source"]
        _require_keys(
            relion,
            {"commit", "tree", "source_dir", "executable", "executable_sha256"},
            f"{identifier}.relion_source",
        )
        if GIT_SHA_RE.fullmatch(str(relion["commit"])) is None or GIT_SHA_RE.fullmatch(
            str(relion["tree"])
        ) is None:
            raise DiagnosticsValidationError(f"{identifier} RELION commit/tree must be full Git SHAs")
        _absolute_path(relion["source_dir"], f"{identifier}.relion_source.source_dir")
        _absolute_path(relion["executable"], f"{identifier}.relion_source.executable")
        _sha256(relion["executable_sha256"], f"{identifier}.relion_source.executable_sha256")

        inputs = run["inputs"]
        if not isinstance(inputs, list) or len(inputs) < 3:
            raise DiagnosticsValidationError(f"{identifier} must seal STAR, stack, and indices inputs")
        for item_index, item in enumerate(inputs):
            _file_ref(item, f"{identifier}.inputs[{item_index}]", require_size=True)
        roles = [item["role"] for item in inputs]
        if len(roles) != len(set(roles)):
            raise DiagnosticsValidationError(f"{identifier} input roles must be unique")

        command = run["command"]
        _require_keys(command, {"shell", "manifest"}, f"{identifier}.command")
        if not isinstance(command["shell"], str) or not command["shell"].startswith("/"):
            raise DiagnosticsValidationError(f"{identifier}.command.shell must be an absolute command")
        _file_ref(command["manifest"], f"{identifier}.command.manifest", require_size=False)

        execution = run["execution"]
        _require_keys(
            execution,
            {
                "run_root",
                "pair_job",
                "audit_mode",
            },
            f"{identifier}.execution",
        )
        _absolute_path(execution["run_root"], f"{identifier}.execution.run_root")
        _validate_slurm_execution(
            execution["pair_job"],
            f"{identifier}.execution.pair_job",
            require_rejected_outcome=True,
        )
        audit_mode = execution["audit_mode"]
        if audit_mode == "independent_posthoc":
            if "posthoc_audit_job" not in execution:
                raise DiagnosticsValidationError(
                    f"{identifier} independent audit requires posthoc_audit_job"
                )
            _validate_slurm_execution(
                execution["posthoc_audit_job"],
                f"{identifier}.execution.posthoc_audit_job",
                require_rejected_outcome=True,
            )
        elif audit_mode != "embedded_pair_wrapper":
            raise DiagnosticsValidationError(f"{identifier} has unsupported audit_mode")

        outcome = run["outcome"]
        _require_keys(
            outcome,
            {"relion_native_completed", "recovar_native_completed", "wrapper_failure"},
            f"{identifier}.outcome",
        )
        if outcome["relion_native_completed"] is not True or outcome["recovar_native_completed"] is not True:
            raise DiagnosticsValidationError(f"{identifier} is not a sealed completed-engine pair")
        if not isinstance(outcome["wrapper_failure"], str) or not outcome["wrapper_failure"]:
            raise DiagnosticsValidationError(f"{identifier} must record its wrapper failure")

        quality = run["quality"]
        _require_keys(
            quality,
            {
                "audit_result",
                "minimum_matched_fsc_auc",
                "minimum_assignment_accuracy",
                "class_stability_result",
                "final_iteration",
                "final_permutation_candidate_to_reference",
                "final_matched_fsc_auc",
                "final_assignment_accuracy",
                "final_candidate_counts",
                "final_reference_counts",
                "gold_standard_halfmaps_available",
                "gold_standard_halfmaps_missing_reason",
            },
            f"{identifier}.quality",
        )
        if quality["audit_result"] != "fail":
            raise DiagnosticsValidationError(f"{identifier} rejected diagnostic must retain audit_result=fail")
        if len(quality["final_permutation_candidate_to_reference"]) != 4:
            raise DiagnosticsValidationError(f"{identifier} final permutation must contain four classes")
        if sorted(quality["final_permutation_candidate_to_reference"]) != [0, 1, 2, 3]:
            raise DiagnosticsValidationError(f"{identifier} final permutation must be a zero-based bijection")
        if quality["final_iteration"] != 8:
            raise DiagnosticsValidationError(f"{identifier} final iteration must be 8")
        for key in ("final_matched_fsc_auc", "final_candidate_counts", "final_reference_counts"):
            if len(quality[key]) != 4:
                raise DiagnosticsValidationError(f"{identifier}.quality.{key} must contain four values")
        if sum(quality["final_candidate_counts"]) != scope["particles"] or sum(
            quality["final_reference_counts"]
        ) != scope["particles"]:
            raise DiagnosticsValidationError(f"{identifier} final class counts must sum to all particles")
        if quality["gold_standard_halfmaps_available"] is not False or not isinstance(
            quality["gold_standard_halfmaps_missing_reason"], str
        ):
            raise DiagnosticsValidationError(f"{identifier} must retain the InitialModel half-map limitation")

        performance = run["performance"]
        _require_keys(performance, {"recovar", "relion", "formal_ratio"}, f"{identifier}.performance")
        _validate_performance(performance["recovar"], f"{identifier}.performance.recovar")
        _validate_performance(performance["relion"], f"{identifier}.performance.relion")
        if performance["formal_ratio"] is not None:
            raise DiagnosticsValidationError(f"{identifier} rejected run cannot publish a formal ratio")

        artifacts = run["artifacts"]
        if not isinstance(artifacts, list) or len(artifacts) < 4:
            raise DiagnosticsValidationError(f"{identifier} requires command, audit, shellwise, and log artifacts")
        for item_index, item in enumerate(artifacts):
            _file_ref(item, f"{identifier}.artifacts[{item_index}]", require_size=False)
        artifact_roles = [item["role"] for item in artifacts]
        if len(artifact_roles) != len(set(artifact_roles)):
            raise DiagnosticsValidationError(f"{identifier} artifact roles must be unique")
        required_artifact_roles = {
            "audit_json",
            "shellwise_fsc_npz",
            "relion_command",
            "recovar_command",
            "relion_gpu_monitor",
            "recovar_gpu_monitor",
            "relion_process_resources",
            "recovar_process_resources",
            "pair_stdout",
            "pair_stderr",
        }
        if audit_mode == "independent_posthoc":
            required_artifact_roles.update({"posthoc_audit_stdout", "posthoc_audit_stderr"})
        else:
            required_artifact_roles.add("pair_report")
        missing_artifact_roles = sorted(required_artifact_roles - set(artifact_roles))
        if missing_artifact_roles:
            raise DiagnosticsValidationError(
                f"{identifier} is missing required artifacts: {missing_artifact_roles}"
            )

    causal = data["causal_diagnostics"]
    if not isinstance(causal, list) or not causal:
        raise DiagnosticsValidationError("causal_diagnostics must be a non-empty list")
    causal_identifiers: set[str] = set()
    for index, diagnostic in enumerate(causal):
        label = f"causal_diagnostics[{index}]"
        _require_keys(
            diagnostic,
            {
                "diagnostic_id",
                "admission_status",
                "kind",
                "dataset",
                "source",
                "command",
                "execution",
                "coverage",
                "audit",
                "formal_ratio",
                "artifacts",
                "limitations",
            },
            label,
        )
        identifier = diagnostic["diagnostic_id"]
        if not isinstance(identifier, str) or not identifier or identifier in causal_identifiers:
            raise DiagnosticsValidationError(f"{label}.diagnostic_id must be unique and non-empty")
        causal_identifiers.add(identifier)
        if diagnostic["admission_status"] not in {"FORMAL_NEGATIVE", "DIAGNOSTIC_ONLY"}:
            raise DiagnosticsValidationError(f"{identifier} cannot claim benchmark admission")
        if diagnostic["formal_ratio"] is not None:
            raise DiagnosticsValidationError(f"{identifier} causal diagnostic cannot publish a formal ratio")

        source = diagnostic["source"]
        _require_keys(source, {"commit", "tree", "checkout", "clean"}, f"{identifier}.source")
        if GIT_SHA_RE.fullmatch(str(source["commit"])) is None or GIT_SHA_RE.fullmatch(
            str(source["tree"])
        ) is None:
            raise DiagnosticsValidationError(f"{identifier} source commit/tree must be full Git SHAs")
        _absolute_path(source["checkout"], f"{identifier}.source.checkout")
        if source["clean"] is not True:
            raise DiagnosticsValidationError(f"{identifier} source must be clean")

        command = diagnostic["command"]
        _require_keys(command, {"shell", "script"}, f"{identifier}.command")
        if not isinstance(command["shell"], str) or not command["shell"].startswith("sbatch "):
            raise DiagnosticsValidationError(f"{identifier}.command.shell must retain the sbatch command")
        _file_ref(command["script"], f"{identifier}.command.script", require_size=False)

        execution = diagnostic["execution"]
        _require_keys(execution, {"run_root", "job"}, f"{identifier}.execution")
        _absolute_path(execution["run_root"], f"{identifier}.execution.run_root")
        _validate_slurm_execution(
            execution["job"],
            f"{identifier}.execution.job",
            require_rejected_outcome=False,
        )

        coverage = diagnostic["coverage"]
        _require_keys(
            coverage,
            {
                "relion_assigned_particles",
                "recovar_assigned_particles",
                "shared_assigned_particles",
                "same_visited_particle_ids",
                "map_comparison_valid",
                "reason",
            },
            f"{identifier}.coverage",
        )
        for key in (
            "relion_assigned_particles",
            "recovar_assigned_particles",
            "shared_assigned_particles",
        ):
            if not isinstance(coverage[key], int) or coverage[key] <= 0:
                raise DiagnosticsValidationError(f"{identifier}.coverage.{key} must be positive")
        if coverage["map_comparison_valid"] is False and (
            coverage["same_visited_particle_ids"] is not False
            or not isinstance(coverage["reason"], str)
            or not coverage["reason"]
        ):
            raise DiagnosticsValidationError(
                f"{identifier} invalid map comparison must retain the coverage mismatch"
            )
        if coverage["map_comparison_valid"] is True and coverage["same_visited_particle_ids"] is not True:
            raise DiagnosticsValidationError(
                f"{identifier} valid map comparison requires identical visited particle IDs"
            )

        audit = diagnostic["audit"]
        _require_keys(
            audit,
            {"result", "minimum_matched_fsc_auc", "minimum_assignment_accuracy"},
            f"{identifier}.audit",
        )
        if audit["result"] not in {"pass", "fail"}:
            raise DiagnosticsValidationError(f"{identifier}.audit.result must be pass or fail")
        if not isinstance(diagnostic["limitations"], list) or not diagnostic["limitations"]:
            raise DiagnosticsValidationError(f"{identifier} requires explicit limitations")

        artifacts = diagnostic["artifacts"]
        if not isinstance(artifacts, list) or len(artifacts) < 3:
            raise DiagnosticsValidationError(f"{identifier} requires at least three hashed artifacts")
        for item_index, item in enumerate(artifacts):
            _file_ref(item, f"{identifier}.artifacts[{item_index}]", require_size=False)
        artifact_roles = [item["role"] for item in artifacts]
        if len(artifact_roles) != len(set(artifact_roles)):
            raise DiagnosticsValidationError(f"{identifier} artifact roles must be unique")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(data: dict[str, Any]) -> None:
    """Verify every diagnostic input/artifact hash; intended for explicit sealing."""

    cached: dict[Path, str] = {}
    for run in data["runs"]:
        references = [*run["inputs"], run["command"]["manifest"], *run["artifacts"]]
        for reference in references:
            path = Path(reference["path"])
            if not path.is_file():
                raise DiagnosticsValidationError(f"missing sealed file: {path}")
            if path not in cached:
                cached[path] = _hash_file(path)
            digest = cached[path]
            if digest != reference["sha256"]:
                raise DiagnosticsValidationError(f"checksum mismatch: {path}")
    for diagnostic in data["causal_diagnostics"]:
        references = [diagnostic["command"]["script"], *diagnostic["artifacts"]]
        for reference in references:
            path = Path(reference["path"])
            if not path.is_file():
                raise DiagnosticsValidationError(f"missing sealed file: {path}")
            if path not in cached:
                cached[path] = _hash_file(path)
            if cached[path] != reference["sha256"]:
                raise DiagnosticsValidationError(f"checksum mismatch: {path}")


def default_ledger_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "benchmarks"
        / "em"
        / "diagnostics"
        / "real-kclass-initialmodel-20260901.json"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", nargs="?", type=Path, default=default_ledger_path())
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.ledger.read_text())
        validate_diagnostics(data)
        if args.verify_files:
            verify_files(data)
    except (OSError, json.JSONDecodeError, DiagnosticsValidationError) as error:
        parser.exit(1, f"Real K-class diagnostics validation failed:\n{error}\n")
    print(
        f"Validated {len(data['runs'])} rejected pair(s) and "
        f"{len(data['causal_diagnostics'])} causal diagnostic(s): {args.ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
