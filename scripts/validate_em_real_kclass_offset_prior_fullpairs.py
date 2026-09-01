#!/usr/bin/env python3
"""Validate the sealed K=4 offset-prior real-data full-pair evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "recovar.em_real_kclass_offset_prior_fullpairs.v1"
EXPECTED_SOURCE_HEAD = "92438c285172998baf958ef46d7e3053a51052d2"
EXPECTED_SOURCE_TREE = "ef74c6c6069bd54e1ca5f898e886863bd7ca2ba9"
EXPECTED_DATASETS = {"EMPIAR-10076", "EMPIAR-10345"}
EXPECTED_CHECKPOINTS = list(range(1, 9))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FullPairEvidenceValidationError(ValueError):
    """Raised when the compact evidence could support a misleading claim."""


def _require_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - value.keys())
    if missing:
        raise FullPairEvidenceValidationError(f"{label} is missing keys: {missing}")


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise FullPairEvidenceValidationError(f"{label} must be an absolute path")
    return Path(value)


def _validate_sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FullPairEvidenceValidationError(f"{label} must be a lowercase SHA-256")


def _positive_number(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise FullPairEvidenceValidationError(f"{label} must be positive")


def _validate_file_ref(value: dict[str, Any], label: str, *, require_size: bool) -> None:
    required = {"path", "sha256"} | ({"size_bytes"} if require_size else set())
    _require_keys(value, required, label)
    _absolute_path(value["path"], f"{label}.path")
    _validate_sha(value["sha256"], f"{label}.sha256")
    if require_size:
        _positive_number(value["size_bytes"], f"{label}.size_bytes")


def _validate_iteration_1(value: dict[str, Any], label: str, *, dataset: str) -> None:
    _require_keys(
        value,
        {
            "assigned_particles",
            "raw_class_label_exact_count",
            "raw_class_label_accuracy",
            "recovar_counts",
            "relion_counts",
            "raw_pose_joint_le_1e_3_count",
            "map_hungarian_permutation_candidate_to_reference",
            "assignment_accuracy_after_map_hungarian_permutation",
            "matched_fsc_auc",
            "interpretation",
        },
        label,
    )
    assigned = value["assigned_particles"]
    if assigned != 200:
        raise FullPairEvidenceValidationError(f"{label} must retain the 200-particle boundary")
    if value["raw_class_label_exact_count"] != assigned or value["raw_class_label_accuracy"] != 1.0:
        raise FullPairEvidenceValidationError(f"{label} must retain exact raw iteration-1 labels")
    if value["recovar_counts"] != value["relion_counts"]:
        raise FullPairEvidenceValidationError(f"{label} raw class counts must match")
    if sum(value["recovar_counts"]) != assigned or len(value["recovar_counts"]) != 4:
        raise FullPairEvidenceValidationError(f"{label} must retain four counts summing to 200")
    permutation = value["map_hungarian_permutation_candidate_to_reference"]
    if sorted(permutation) != list(range(4)):
        raise FullPairEvidenceValidationError(f"{label} map permutation must be a K=4 permutation")
    map_assignment = value["assignment_accuracy_after_map_hungarian_permutation"]
    if not isinstance(map_assignment, (int, float)) or not 0 <= map_assignment <= 1:
        raise FullPairEvidenceValidationError(f"{label} map-Hungarian assignment must be in [0,1]")
    if dataset == "EMPIAR-10345" and map_assignment == value["raw_class_label_accuracy"]:
        raise FullPairEvidenceValidationError(
            f"{label} must distinguish raw-label agreement from map-Hungarian agreement"
        )
    fsc = value["matched_fsc_auc"]
    if len(fsc) != 4 or min(fsc) >= 0.999:
        raise FullPairEvidenceValidationError(f"{label} must retain the rejected K=4 FSC result")
    if not isinstance(value["interpretation"], str) or not value["interpretation"]:
        raise FullPairEvidenceValidationError(f"{label}.interpretation must be non-empty")


def _validate_case(value: dict[str, Any], index: int) -> None:
    label = f"cases[{index}]"
    _require_keys(
        value,
        {"diagnostic_id", "dataset", "run_root", "fixture", "slurm", "quality", "performance", "artifacts"},
        label,
    )
    dataset = value["dataset"]
    if dataset not in EXPECTED_DATASETS:
        raise FullPairEvidenceValidationError(f"{label}.dataset is not a sealed dataset")
    _absolute_path(value["run_root"], f"{label}.run_root")

    fixture = value["fixture"]
    _require_keys(fixture, {"directory", "particles_star", "particle_stack", "source_indices", "manifest"}, f"{label}.fixture")
    _absolute_path(fixture["directory"], f"{label}.fixture.directory")
    for role in ("particles_star", "particle_stack", "source_indices", "manifest"):
        _validate_file_ref(fixture[role], f"{label}.fixture.{role}", require_size=True)

    slurm = value["slurm"]
    _require_keys(
        slurm,
        {
            "job_id",
            "terminal_state",
            "exit_code",
            "elapsed_seconds",
            "node",
            "req_tres",
            "alloc_tres",
            "oversubscribe",
            "exclusive",
            "terminal_reason",
            "capture_command",
        },
        f"{label}.slurm",
    )
    if not str(slurm["job_id"]).isdigit():
        raise FullPairEvidenceValidationError(f"{label}.slurm.job_id must be numeric")
    if slurm["terminal_state"] != "FAILED" or slurm["exit_code"] != "1:0":
        raise FullPairEvidenceValidationError(f"{label} must retain the rejected terminal outcome")
    if slurm["req_tres"] != slurm["alloc_tres"]:
        raise FullPairEvidenceValidationError(f"{label} ReqTRES and AllocTRES differ")
    if "gres/gpu=1" not in slurm["req_tres"] or slurm["exclusive"] is not False:
        raise FullPairEvidenceValidationError(f"{label} must retain the exact one-GPU nonexclusive allocation")
    if slurm["oversubscribe"] != "OK":
        raise FullPairEvidenceValidationError(f"{label}.slurm.oversubscribe must be OK")
    _positive_number(slurm["elapsed_seconds"], f"{label}.slurm.elapsed_seconds")

    quality = value["quality"]
    _require_keys(
        quality,
        {
            "result",
            "trajectory_parity_result",
            "class_stability_result",
            "minimum_matched_fsc_auc",
            "minimum_map_hungarian_assignment_accuracy",
            "iteration_1",
            "iteration_2",
            "final_iteration_8",
        },
        f"{label}.quality",
    )
    if quality["result"] != "REJECTED" or quality["trajectory_parity_result"] != "fail":
        raise FullPairEvidenceValidationError(f"{label} must remain scientifically rejected")
    _validate_iteration_1(quality["iteration_1"], f"{label}.quality.iteration_1", dataset=dataset)
    iteration_2 = quality["iteration_2"]
    if iteration_2["assignment_accuracy_after_map_hungarian_permutation"] >= 0.995:
        raise FullPairEvidenceValidationError(f"{label} must retain the iteration-2 assignment divergence")
    if len(iteration_2["recovar_counts"]) != 4 or len(iteration_2["relion_counts"]) != 4:
        raise FullPairEvidenceValidationError(f"{label} iteration-2 counts must retain all four classes")
    final = quality["final_iteration_8"]
    for engine in ("recovar", "relion"):
        counts = final[f"{engine}_counts"]
        if len(counts) != 4 or sum(counts) != 10000:
            raise FullPairEvidenceValidationError(f"{label} final {engine} counts must sum to 10,000")
    if len(final["matched_fsc_auc"]) != 4:
        raise FullPairEvidenceValidationError(f"{label} final FSC must retain four matched classes")

    performance = value["performance"]
    _require_keys(
        performance,
        {
            "same_physical_gpu",
            "gpu",
            "recovar",
            "relion",
            "raw_recovar_over_relion_wall_ratio",
            "formal_ratio_admitted",
            "measurement_limitations",
        },
        f"{label}.performance",
    )
    if performance["same_physical_gpu"] is not True or performance["formal_ratio_admitted"] is not False:
        raise FullPairEvidenceValidationError(f"{label} cannot admit a formal performance ratio")
    for engine in ("recovar", "relion"):
        engine_metrics = performance[engine]
        if engine_metrics.get("exit_code") != 0:
            raise FullPairEvidenceValidationError(f"{label} {engine} engine must have exited successfully")
        for metric in ("wall_seconds", "peak_hbm_mib", "max_rss_kib"):
            _positive_number(engine_metrics.get(metric), f"{label}.performance.{engine}.{metric}")
    _positive_number(
        performance["raw_recovar_over_relion_wall_ratio"],
        f"{label}.performance.raw_recovar_over_relion_wall_ratio",
    )
    if not performance["measurement_limitations"]:
        raise FullPairEvidenceValidationError(f"{label} requires performance limitations")

    artifacts = value["artifacts"]
    required_roles = {
        "submission_manifest",
        "sbatch_script",
        "pair_report",
        "trajectory_audit",
        "trajectory_shellwise_fsc",
        "recovar_command",
        "relion_command",
        "recovar_gpu_monitor",
        "relion_gpu_monitor",
        "recovar_resources",
        "relion_resources",
        "slurm_stdout",
        "slurm_stderr",
        "iteration_1_model_state_discriminator",
    }
    _require_keys(artifacts, required_roles, f"{label}.artifacts")
    for role, reference in artifacts.items():
        _validate_file_ref(reference, f"{label}.artifacts.{role}", require_size=False)


def validate_evidence(data: dict[str, Any]) -> None:
    """Validate the compact full-pair evidence without reading external files."""

    _require_keys(data, {"schema", "source_contract", "experiment_contract", "reproduction", "scientific_scope", "cases"}, "ledger")
    if data["schema"] != SCHEMA:
        raise FullPairEvidenceValidationError(f"unexpected schema: {data['schema']!r}")
    source = data["source_contract"]
    if source.get("git_head") != EXPECTED_SOURCE_HEAD or source.get("git_tree") != EXPECTED_SOURCE_TREE:
        raise FullPairEvidenceValidationError("ledger source commit/tree is not the sealed treatment source")
    if source.get("tracked_dirty") is not False:
        raise FullPairEvidenceValidationError("ledger source must be clean")
    contract = data["experiment_contract"]
    if contract.get("K") != 4 or contract.get("checkpoints") != EXPECTED_CHECKPOINTS:
        raise FullPairEvidenceValidationError("experiment must retain K=4 checkpoints 1--8")
    if contract.get("particles") != 10000 or contract.get("image_batch_size") != 500:
        raise FullPairEvidenceValidationError("experiment must retain the matched 10,000-particle batch-500 workload")
    scope = data["scientific_scope"]
    if (
        scope.get("gold_standard_halfmaps_available") is not False
        or scope.get("halfmap_fsc_available") is not False
        or scope.get("formal_runtime_ratio_admissible") is not False
        or scope.get("benchmark_registry_admission") != "REJECTED"
    ):
        raise FullPairEvidenceValidationError("InitialModel evidence cannot claim half maps, a formal ratio, or admission")
    cases = data["cases"]
    if not isinstance(cases, list) or len(cases) != 2:
        raise FullPairEvidenceValidationError("ledger must contain exactly two independent real-data cases")
    if {case.get("dataset") for case in cases} != EXPECTED_DATASETS:
        raise FullPairEvidenceValidationError("ledger must contain distinct 10076 and 10345 cases")
    for index, case in enumerate(cases):
        _validate_case(case, index)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_compact_files(data: dict[str, Any]) -> None:
    """Verify compact fixture lineage and output evidence, excluding particle stacks."""

    for case in data["cases"]:
        references = [
            case["fixture"]["particles_star"],
            case["fixture"]["source_indices"],
            case["fixture"]["manifest"],
            *case["artifacts"].values(),
        ]
        for reference in references:
            path = Path(reference["path"])
            if not path.is_file():
                raise FullPairEvidenceValidationError(f"missing sealed file: {path}")
            if _hash_file(path) != reference["sha256"]:
                raise FullPairEvidenceValidationError(f"checksum mismatch: {path}")


def default_ledger_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "benchmarks"
        / "em"
        / "diagnostics"
        / "real-kclass-offset-prior-fullpairs-92438c285-20260901.json"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", nargs="?", type=Path, default=default_ledger_path())
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.ledger.read_text())
        validate_evidence(data)
        if args.verify_files:
            verify_compact_files(data)
    except (OSError, json.JSONDecodeError, FullPairEvidenceValidationError) as error:
        parser.exit(1, f"K=4 offset-prior full-pair evidence validation failed:\n{error}\n")
    print(f"Validated {len(data['cases'])} rejected real-data K=4 full pair(s): {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
