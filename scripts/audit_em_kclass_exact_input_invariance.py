#!/usr/bin/env python3
"""Audit K-class execution invariance on byte-identical inputs and oracles.

The run matrix is read from ``case_table.tsv``.  Each comparison declares one
allowed execution-axis change as
``BASE_CASE:VARIANT_CASE:CONFIG_KEY:BASE_VALUE:VARIANT_VALUE``.  Everything
else in the runtime configuration, the shared inputs, the RELION oracle,
controller topology, and requested/allocated resources must match exactly.

Numerical equality is reported separately from scientific equivalence.  The
formal science gate uses shellwise FSC/FSC-AUC, permutation-aware final class
matching, final particle-class agreement, and per-class GT FSC-AUC deltas.
Last-bit floating-point differences therefore remain visible without being
misclassified as a scientific failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
from scipy.optimize import linear_sum_assignment

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.summarize_em_completion_bench import normalized_fsc_auc, shell_fsc  # noqa: E402

SCHEMA = "recovar.em_kclass_exact_input_invariance.v1"
DEFAULT_CONFIG_METADATA_KEYS = frozenset(
    {
        "base_name",
        "case_root",
        "index",
        "name",
        "shared_input_role",
        "slurm_job_id",
    }
)
CONTROLLER_RE = re.compile(
    r"RELION Iteration (?P<iteration>\d+): current_size=(?P<current_size>\d+), "
    r"pixel_res=(?P<pixel_res>[^,]+), res=(?P<resolution>[^ ]+) A, "
    r"ave_Pmax=(?P<ave_pmax>[^,]+), healpix_order=(?P<healpix_order>\d+), "
    r"converged=(?P<converged>[^,]+), time=(?P<time>[^s]+)s"
)
NUMBERED_MAP_RE = re.compile(
    r"^it(?P<iteration>\d{3})_half(?P<half>[12])_class(?P<class_id>\d+)_reg\.mrc$"
)
ASSIGNMENT_RE = re.compile(r"^class_assignments_by_image_iter_(?P<iteration>\d+)$")
SBATCH_EXCLUSIVE_RE = re.compile(r"^\s*#SBATCH\s+--exclusive(?:\s|$)", re.MULTILINE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AuditError(RuntimeError):
    """Raised when required evidence is absent, inconsistent, or invalid."""


@dataclass(frozen=True)
class Thresholds:
    direct_fsc_auc_min: float = 0.995
    class_assignment_agreement_min: float = 0.99
    gt_fsc_auc_delta_min: float = -0.002


@dataclass(frozen=True)
class AxisExpectation:
    baseline_case: int
    variant_case: int
    key: str
    baseline_value: Any
    variant_value: Any

    @classmethod
    def parse(cls, raw: str) -> AxisExpectation:
        fields = raw.split(":", 4)
        if len(fields) != 5:
            raise argparse.ArgumentTypeError(
                "comparison must be BASE:VARIANT:KEY:BASE_VALUE:VARIANT_VALUE"
            )
        baseline, variant, key, baseline_raw, variant_raw = fields
        try:
            return cls(
                baseline_case=int(baseline),
                variant_case=int(variant),
                key=key,
                baseline_value=_json_scalar(baseline_raw),
                variant_value=_json_scalar(variant_raw),
            )
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc


def _json_scalar(raw: str) -> Any:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    if isinstance(value, (dict, list)):
        raise ValueError("axis values must be JSON scalars")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256_manifest(path: Path) -> dict[str, Any]:
    """Verify a two-space-delimited absolute-path SHA-256 manifest."""
    path = path.resolve()
    if not path.is_file():
        raise AuditError(f"missing manifest: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            expected, raw_path = line.split("  ", 1)
        except ValueError as exc:
            raise AuditError(f"malformed manifest line {line_number}: {path}") from exc
        artifact = Path(raw_path)
        if not SHA256_RE.fullmatch(expected) or not artifact.is_absolute():
            raise AuditError(f"invalid manifest line {line_number}: {path}")
        if not artifact.is_file():
            raise AuditError(f"manifest artifact is missing: {artifact}")
        observed = sha256_file(artifact)
        if observed != expected:
            raise AuditError(
                f"manifest checksum mismatch: {artifact}: expected {expected}, got {observed}"
            )
        rows.append({"path": str(artifact), "sha256": observed, "size_bytes": artifact.stat().st_size})
    if not rows:
        raise AuditError(f"manifest is empty: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "entry_count": len(rows),
        "entries": rows,
    }


def compare_configurations(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    expectation: AxisExpectation,
    *,
    metadata_keys: frozenset[str] = DEFAULT_CONFIG_METADATA_KEYS,
) -> dict[str, Any]:
    """Fail unless the declared execution axis is the only scientific change."""
    observed = (baseline.get(expectation.key), variant.get(expectation.key))
    expected = (expectation.baseline_value, expectation.variant_value)
    if observed != expected:
        raise AuditError(
            f"axis {expectation.key} is {observed!r}, expected {expected!r}"
        )
    ignored = set(metadata_keys) | {expectation.key}
    differences = {
        key: {"baseline": baseline.get(key), "variant": variant.get(key)}
        for key in sorted(set(baseline) | set(variant))
        if key not in ignored and baseline.get(key) != variant.get(key)
    }
    if differences:
        raise AuditError(f"undeclared configuration differences: {differences}")
    return {
        "axis": expectation.key,
        "baseline_value": observed[0],
        "variant_value": observed[1],
        "ignored_metadata_keys": sorted(metadata_keys),
        "undeclared_differences": {},
    }


def _tres_value(tres: str, key: str) -> int | None:
    values = []
    for item in tres.split(","):
        item_key, separator, raw_value = item.partition("=")
        if separator and (item_key == key or item_key.startswith(f"{key}:")):
            try:
                values.append(int(raw_value))
            except ValueError as exc:
                raise AuditError(f"invalid TRES component: {item}") from exc
    return sum(values) if values else None


def validate_slurm_allocation(
    record: dict[str, str],
    *,
    job_script_text: str,
    expected_cpus: int,
    expected_memory: str,
    expected_gpus: int,
) -> dict[str, Any]:
    """Validate completion, exact allocation, and nonexclusive submission."""
    failures = []
    if record.get("state") != "COMPLETED":
        failures.append(f"state={record.get('state')}")
    if record.get("exit_code") != "0:0":
        failures.append(f"exit_code={record.get('exit_code')}")
    if record.get("req_tres") != record.get("alloc_tres"):
        failures.append("ReqTRES != AllocTRES")
    if record.get("alloc_cpus") != str(expected_cpus):
        failures.append(f"alloc_cpus={record.get('alloc_cpus')}")
    if record.get("req_mem") != expected_memory:
        failures.append(f"req_mem={record.get('req_mem')}")
    if _tres_value(record.get("alloc_tres", ""), "gres/gpu") != expected_gpus:
        failures.append(f"gpu_count={_tres_value(record.get('alloc_tres', ''), 'gres/gpu')}")
    if SBATCH_EXCLUSIVE_RE.search(job_script_text):
        failures.append("job script requests --exclusive")
    if failures:
        raise AuditError("invalid Slurm allocation: " + "; ".join(failures))
    return {**record, "nonexclusive": True, "valid": True}


def evaluate_science_gate(
    *,
    complete: bool,
    controller_equal: bool,
    min_numbered_fsc_auc: float,
    min_final_fsc_auc: float,
    final_class_assignment_agreement: float,
    min_gt_fsc_auc_delta: float,
    thresholds: Thresholds,
) -> dict[str, Any]:
    """Apply the frozen science gate without conflating it with bitwise equality."""
    failures = []
    if not complete:
        failures.append("scientific artifact topology is incomplete")
    if not controller_equal:
        failures.append("controller topology differs")
    if min_numbered_fsc_auc < thresholds.direct_fsc_auc_min:
        failures.append(
            f"numbered FSC-AUC {min_numbered_fsc_auc} < {thresholds.direct_fsc_auc_min}"
        )
    if min_final_fsc_auc < thresholds.direct_fsc_auc_min:
        failures.append(f"final FSC-AUC {min_final_fsc_auc} < {thresholds.direct_fsc_auc_min}")
    if final_class_assignment_agreement < thresholds.class_assignment_agreement_min:
        failures.append(
            "final class-assignment agreement "
            f"{final_class_assignment_agreement} < {thresholds.class_assignment_agreement_min}"
        )
    if min_gt_fsc_auc_delta < thresholds.gt_fsc_auc_delta_min:
        failures.append(
            f"per-class GT FSC-AUC delta {min_gt_fsc_auc_delta} < "
            f"{thresholds.gt_fsc_auc_delta_min}"
        )
    return {"accepted": not failures, "failures": failures}


def _read_case_table(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    if not path.is_file():
        raise AuditError(f"missing case table: {path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    result = {}
    for row in rows:
        key = (int(row["index"]), int(row["seed"]))
        if key in result:
            raise AuditError(f"duplicate case-table row {key}")
        result[key] = row
    return result


def _slurm_accounting(job_id: str) -> dict[str, str]:
    command = [
        "sacct",
        "-X",
        "-j",
        job_id,
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,State,ElapsedRaw,ExitCode,AllocCPUS,ReqMem,ReqTRES,AllocTRES,NodeList",
    ]
    lines = subprocess.check_output(command, text=True).strip().splitlines()
    if not lines:
        raise AuditError(f"no Slurm accounting for job {job_id}")
    values = lines[0].rstrip("|").split("|")
    keys = (
        "job_id",
        "state",
        "elapsed_s",
        "exit_code",
        "alloc_cpus",
        "req_mem",
        "req_tres",
        "alloc_tres",
        "node",
    )
    if len(values) != len(keys):
        raise AuditError(f"malformed Slurm accounting for job {job_id}")
    return dict(zip(keys, values, strict=True))


def _controller(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AuditError(f"missing refinement log: {path}")
    rows = []
    for match in CONTROLLER_RE.finditer(path.read_text(errors="replace")):
        row = match.groupdict()
        row.pop("time")
        rows.append(row)
    if not rows:
        raise AuditError(f"no controller rows in {path}")
    return rows


def _load_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as volume:
        return np.asarray(volume.data, dtype=np.float64).copy()


def _fsc_auc(left: np.ndarray, right: np.ndarray) -> float:
    return float(normalized_fsc_auc(np.asarray(shell_fsc(left, right), dtype=np.float64)))


def _discover_numbered_maps(directory: Path, n_classes: int) -> dict[tuple[int, int, int], Path]:
    result = {}
    for path in directory.glob("*.mrc"):
        match = NUMBERED_MAP_RE.fullmatch(path.name)
        if match is None:
            continue
        key = tuple(int(match.group(name)) for name in ("iteration", "half", "class_id"))
        if key in result:
            raise AuditError(f"duplicate numbered map {key}: {directory}")
        result[key] = path
    if not result:
        raise AuditError(f"no numbered maps: {directory}")
    iterations = sorted({key[0] for key in result})
    expected = {
        (iteration, half, class_id)
        for iteration in range(len(iterations))
        for half in (1, 2)
        for class_id in range(1, n_classes + 1)
    }
    if set(result) != expected:
        raise AuditError(
            f"numbered map topology differs: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    return result


def _numbered_map_metrics(baseline: Path, variant: Path, n_classes: int) -> dict[str, Any]:
    baseline_maps = _discover_numbered_maps(baseline, n_classes)
    variant_maps = _discover_numbered_maps(variant, n_classes)
    if set(baseline_maps) != set(variant_maps):
        raise AuditError("baseline and variant numbered map identities differ")
    half_rows = []
    merged_rows = []
    cache: dict[tuple[str, int, int, int], np.ndarray] = {}

    def load(side: str, key: tuple[int, int, int]) -> np.ndarray:
        cache_key = (side, *key)
        if cache_key not in cache:
            paths = baseline_maps if side == "baseline" else variant_maps
            cache[cache_key] = _load_mrc(paths[key])
        return cache[cache_key]

    for iteration, half, class_id in sorted(baseline_maps):
        left = load("baseline", (iteration, half, class_id))
        right = load("variant", (iteration, half, class_id))
        half_rows.append(
            {
                "iteration_zero_based": iteration,
                "half": half,
                "class": class_id,
                "fsc_auc": _fsc_auc(left, right),
                "bitwise_equal": bool(np.array_equal(left, right)),
                "max_abs_difference": float(np.max(np.abs(left - right))),
            }
        )
    for iteration in sorted({key[0] for key in baseline_maps}):
        for class_id in range(1, n_classes + 1):
            left = 0.5 * (
                load("baseline", (iteration, 1, class_id))
                + load("baseline", (iteration, 2, class_id))
            )
            right = 0.5 * (
                load("variant", (iteration, 1, class_id))
                + load("variant", (iteration, 2, class_id))
            )
            merged_rows.append(
                {
                    "iteration_zero_based": iteration,
                    "class": class_id,
                    "fsc_auc": _fsc_auc(left, right),
                    "max_abs_difference": float(np.max(np.abs(left - right))),
                }
            )
    return {
        "iterations": len({key[0] for key in baseline_maps}),
        "half_map_rows": half_rows,
        "merged_map_rows": merged_rows,
        "minimum_half_map_fsc_auc": min(row["fsc_auc"] for row in half_rows),
        "minimum_merged_map_fsc_auc": min(row["fsc_auc"] for row in merged_rows),
        "minimum_numbered_map_fsc_auc": min(
            min(row["fsc_auc"] for row in half_rows),
            min(row["fsc_auc"] for row in merged_rows),
        ),
        "all_half_maps_bitwise_equal": all(row["bitwise_equal"] for row in half_rows),
        "maximum_half_map_abs_difference": max(
            row["max_abs_difference"] for row in half_rows
        ),
    }


def _final_map_metrics(baseline: Path, variant: Path, n_classes: int) -> dict[str, Any]:
    baseline_paths = sorted(baseline.glob("final_class*.mrc"))
    variant_paths = sorted(variant.glob("final_class*.mrc"))
    if len(baseline_paths) != n_classes or len(variant_paths) != n_classes:
        raise AuditError("final map collection is incomplete")
    baseline_maps = [_load_mrc(path) for path in baseline_paths]
    variant_maps = [_load_mrc(path) for path in variant_paths]
    scores = np.asarray(
        [[_fsc_auc(left, right) for right in variant_maps] for left in baseline_maps]
    )
    rows, columns = linear_sum_assignment(-scores)
    if not np.array_equal(rows, np.arange(n_classes)):
        raise AuditError("final Hungarian assignment is incomplete")
    permutation = [int(value) for value in columns]
    assigned = [float(scores[row, column]) for row, column in zip(rows, columns, strict=True)]
    return {
        "pairwise_fsc_auc": scores.tolist(),
        "baseline_to_variant_permutation_zero_based": permutation,
        "assigned_fsc_auc": assigned,
        "minimum_assigned_fsc_auc": min(assigned),
    }


def _class_assignment_metrics(
    baseline: Path,
    variant: Path,
    baseline_to_variant: list[int],
) -> dict[str, Any]:
    permutation = np.asarray(baseline_to_variant, dtype=np.int64)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    with np.load(baseline, allow_pickle=True) as left_npz, np.load(
        variant, allow_pickle=True
    ) as right_npz:
        keys = sorted(set(left_npz.files) & set(right_npz.files))
        rows = []
        for key in keys:
            match = ASSIGNMENT_RE.fullmatch(key)
            if match is None:
                continue
            left = np.asarray(left_npz[key])
            right = np.asarray(right_npz[key])
            if left.shape != right.shape or not right.size:
                raise AuditError(f"class-assignment shape mismatch: {key}")
            if np.any(right < 0) or np.any(right >= inverse.size):
                raise AuditError(f"class assignment is outside 0..K-1: {key}")
            mapped = inverse[right]
            rows.append(
                {
                    "iteration_zero_based": int(match.group("iteration")),
                    "agreement": float(np.mean(left == mapped)),
                    "disagreement_count": int(np.count_nonzero(left != mapped)),
                }
            )
    if not rows:
        raise AuditError("no by-image class-assignment trajectory")
    rows.sort(key=lambda row: row["iteration_zero_based"])
    return {
        "rows": rows,
        "final_agreement": rows[-1]["agreement"],
        "minimum_trajectory_agreement": min(row["agreement"] for row in rows),
    }


def _gt_deltas(baseline: Path, variant: Path, n_classes: int) -> dict[str, Any]:
    def load(path: Path) -> dict[int, float]:
        payload = json.loads(path.read_text())
        rows = payload["primary"]["per_class"]
        result = {
            int(row["matched_gt_class"]): float(row["fsc_auc"])
            for row in rows
        }
        if len(rows) != n_classes or set(result) != set(range(n_classes)):
            raise AuditError(f"GT association is not a K-class permutation: {path}")
        return result

    left = load(baseline)
    right = load(variant)
    rows = [
        {
            "gt_class_one_based": gt_class + 1,
            "baseline_fsc_auc": left[gt_class],
            "variant_fsc_auc": right[gt_class],
            "delta": right[gt_class] - left[gt_class],
        }
        for gt_class in range(n_classes)
    ]
    return {"rows": rows, "minimum_delta": min(row["delta"] for row in rows)}


def _wall_time(path: Path) -> int:
    return int(json.loads(path.read_text())["external_wall_s"])


def _gpu_peak(path: Path) -> int:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = [int(row[" memory.used [MiB]"].strip().removesuffix(" MiB")) for row in rows]
    if not values:
        raise AuditError(f"GPU monitor is empty: {path}")
    return max(values)


def _case_evidence(
    row: dict[str, str],
    *,
    expected_cpus: int,
    expected_memory: str,
    expected_gpus: int,
) -> dict[str, Any]:
    root = Path(row["case_root"]).resolve()
    config_path = root / "case_config.json"
    config = json.loads(config_path.read_text())
    script = Path(row["script"]).resolve()
    accounting = validate_slurm_allocation(
        _slurm_accounting(row["job_id"]),
        job_script_text=script.read_text(),
        expected_cpus=expected_cpus,
        expected_memory=expected_memory,
        expected_gpus=expected_gpus,
    )
    return {
        "root": str(root),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config": config,
        "job_script": str(script),
        "job_script_sha256": sha256_file(script),
        "slurm": accounting,
        "controller": _controller(root / "recovar" / "run_full_refinement.log"),
        "recovar_wall_s": _wall_time(root / "recovar" / "slurm_walltime.json"),
        "recovar_peak_hbm_mib": _gpu_peak(root / "recovar_gpu_monitor.csv"),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    suite_root = args.suite_root.resolve()
    case_table_path = suite_root / "case_table.tsv"
    rows = _read_case_table(case_table_path)
    thresholds = Thresholds(
        direct_fsc_auc_min=args.min_fsc_auc,
        class_assignment_agreement_min=args.min_class_agreement,
        gt_fsc_auc_delta_min=args.min_gt_delta,
    )
    expected_case_ids = {
        case_id
        for comparison in args.comparison
        for case_id in (comparison.baseline_case, comparison.variant_case)
    }
    cases: dict[tuple[int, int], dict[str, Any]] = {}
    for seed in args.seed:
        for case_id in sorted(expected_case_ids):
            key = (case_id, seed)
            if key not in rows:
                raise AuditError(f"missing case-table row {key}")
            cases[key] = _case_evidence(
                rows[key],
                expected_cpus=args.expected_cpus,
                expected_memory=args.expected_memory,
                expected_gpus=args.expected_gpus,
            )

    manifests: dict[str, dict[str, Any]] = {}
    comparisons = []
    all_bitwise = True
    for seed in args.seed:
        for expectation in args.comparison:
            baseline = cases[(expectation.baseline_case, seed)]
            variant = cases[(expectation.variant_case, seed)]
            config_result = compare_configurations(
                baseline["config"], variant["config"], expectation
            )
            for key in ("shared_input_manifest", "shared_relion_manifest"):
                baseline_path = Path(baseline["config"][key]).resolve()
                variant_path = Path(variant["config"][key]).resolve()
                if baseline_path != variant_path:
                    raise AuditError(f"seed {seed} does not share {key}")
                manifest_key = str(baseline_path)
                if manifest_key not in manifests:
                    manifests[manifest_key] = validate_sha256_manifest(baseline_path)
            for link, config_key in (("shared_data", "data_dir"), ("relion_ref", "shared_relion_dir")):
                for case in (baseline, variant):
                    resolved = (Path(case["root"]) / link).resolve()
                    expected = Path(case["config"][config_key]).resolve()
                    if resolved != expected:
                        raise AuditError(f"{case['root']} {link} does not resolve to {expected}")

            baseline_root = Path(baseline["root"])
            variant_root = Path(variant["root"])
            n_classes = int(baseline["config"]["n_classes"])
            numbered = _numbered_map_metrics(
                baseline_root / "recovar" / "intermediates",
                variant_root / "recovar" / "intermediates",
                n_classes,
            )
            final = _final_map_metrics(
                baseline_root / "recovar", variant_root / "recovar", n_classes
            )
            assignments = _class_assignment_metrics(
                baseline_root / "recovar" / "refinement_results.npz",
                variant_root / "recovar" / "refinement_results.npz",
                final["baseline_to_variant_permutation_zero_based"],
            )
            gt = _gt_deltas(
                baseline_root / "kclass_gt_fsc.json",
                variant_root / "kclass_gt_fsc.json",
                n_classes,
            )
            controller_equal = baseline["controller"] == variant["controller"]
            gate = evaluate_science_gate(
                complete=True,
                controller_equal=controller_equal,
                min_numbered_fsc_auc=numbered["minimum_numbered_map_fsc_auc"],
                min_final_fsc_auc=final["minimum_assigned_fsc_auc"],
                final_class_assignment_agreement=assignments["final_agreement"],
                min_gt_fsc_auc_delta=gt["minimum_delta"],
                thresholds=thresholds,
            )
            all_bitwise &= numbered["all_half_maps_bitwise_equal"]
            comparisons.append(
                {
                    "seed": seed,
                    "baseline_case": expectation.baseline_case,
                    "variant_case": expectation.variant_case,
                    "configuration": config_result,
                    "controller_equal": controller_equal,
                    "numbered_maps": numbered,
                    "final_maps": final,
                    "particle_class_assignments": assignments,
                    "gt_fsc_auc": gt,
                    "science_gate": gate,
                }
            )

    accepted = all(item["science_gate"]["accepted"] for item in comparisons)
    report = {
        "schema": SCHEMA,
        "suite_root": str(suite_root),
        "source_commit": args.expected_source_commit,
        "case_table": {
            "path": str(case_table_path),
            "sha256": sha256_file(case_table_path),
        },
        "seeds": args.seed,
        "thresholds": {
            "direct_fsc_auc_min": thresholds.direct_fsc_auc_min,
            "class_assignment_agreement_min": thresholds.class_assignment_agreement_min,
            "gt_fsc_auc_delta_min": thresholds.gt_fsc_auc_delta_min,
        },
        "manifests": list(manifests.values()),
        "cases": {
            f"{case_id}_seed{seed}": evidence
            for (case_id, seed), evidence in sorted(cases.items())
        },
        "comparisons": comparisons,
        "summary": {
            "trajectory_case_count": len(cases),
            "comparison_count": len(comparisons),
            "all_science_equivalent_at_frozen_thresholds": accepted,
            "all_numbered_half_maps_bitwise_equal": all_bitwise,
            "accepted": accepted,
        },
    }
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument(
        "--comparison", type=AxisExpectation.parse, action="append", required=True
    )
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-cpus", type=int, default=24)
    parser.add_argument("--expected-memory", default="192G")
    parser.add_argument("--expected-gpus", type=int, default=1)
    parser.add_argument("--min-fsc-auc", type=float, default=0.995)
    parser.add_argument("--min-class-agreement", type=float, default=0.99)
    parser.add_argument("--min-gt-delta", type=float, default=-0.002)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = run_audit(args)
    except AuditError as exc:
        report = {
            "schema": SCHEMA,
            "suite_root": str(args.suite_root.resolve()),
            "source_commit": args.expected_source_commit,
            "summary": {"accepted": False},
            "failures": [str(exc)],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output.resolve()}")
    return 0 if report["summary"]["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
