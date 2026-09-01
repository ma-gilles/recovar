#!/usr/bin/env python3
"""Seal completed RELION-collapse K-class runs as excluded diagnostic evidence.

This script is intentionally narrower than the accepted benchmark sealer.  It
records campaigns that stopped before RECOVAR because the RELION oracle had a
zero-population class.  It never converts such a run into a quality or
performance pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ref(path: Path, *, role: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    reference: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if role is not None:
        reference = {"role": role, **reference}
    return reference


def _manifest_source_structures(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text())
    class_numbers = [entry["class_number"] for entry in manifest]
    if class_numbers != list(range(1, len(manifest) + 1)):
        raise ValueError(f"class manifest is not ordered 1..K: {path}")
    return [
        _file_ref(Path(entry["pdb_path"]).resolve(), role=f"pdb_class_{entry['class_number']}")
        for entry in manifest
    ]


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def _git_source(checkout: Path) -> dict[str, Any]:
    checkout = checkout.resolve()
    if _run(["git", "status", "--short"], cwd=checkout):
        raise ValueError(f"source checkout is not clean: {checkout}")
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=checkout),
        "tree": _run(["git", "rev-parse", "HEAD^{tree}"], cwd=checkout),
        "checkout": str(checkout),
        "clean": True,
    }


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value, comments=False, posix=True)
        values[key] = parsed[0] if len(parsed) == 1 else raw_value
    return values


def _parse_case_table(path: Path, case_id: int) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    selected = [row for row in rows if int(row["index"]) == case_id]
    return sorted(selected, key=lambda row: int(row["seed"]))


def _parse_memory_to_kib(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    match = re.fullmatch(r"([0-9.]+)([KMGTP]?)", value)
    if match is None:
        raise ValueError(f"unsupported Slurm memory value: {value}")
    amount = float(match.group(1))
    scale = {"": 1 / 1024, "K": 1, "M": 1024, "G": 1024**2, "T": 1024**3, "P": 1024**4}
    return round(amount * scale[match.group(2)])


def _sacct_job(job_id: str) -> dict[str, Any]:
    fields = "JobIDRaw,State,ElapsedRaw,ExitCode,NodeList,ReqTRES,AllocTRES"
    output = _run(["sacct", "-X", "-n", "-P", "-j", job_id, f"--format={fields}"])
    rows = [line.split("|") for line in output.splitlines() if line.strip()]
    row = next((values for values in rows if values[0] == job_id), None)
    if row is None:
        raise ValueError(f"sacct returned no top-level row for job {job_id}")
    step_output = _run(["sacct", "-n", "-P", "-j", job_id, "--format=JobIDRaw,MaxRSS"])
    max_rss_values = [
        parsed
        for line in step_output.splitlines()
        if line.strip()
        for parsed in [_parse_memory_to_kib(line.split("|")[1])]
        if parsed is not None
    ]
    return {
        "job_id": row[0],
        "status": row[1].split()[0].rstrip("+"),
        "elapsed_s": int(row[2]),
        "exit_code": row[3],
        "node": row[4],
        "req_tres": row[5],
        "alloc_tres": row[6],
        "max_rss_kib": max(max_rss_values) if max_rss_values else None,
    }


def _gpu_identity(case_root: Path) -> tuple[str, str]:
    gpu_uuid = (case_root / "physical_gpu_uuid.txt").read_text().strip()
    with (case_root / "physical_gpu_inventory.csv").open(newline="") as stream:
        rows = list(csv.reader(stream, skipinitialspace=True))
    if len(rows) != 2:
        raise ValueError(f"expected one GPU inventory row: {case_root}")
    return rows[1][2].strip(), gpu_uuid


def _peak_hbm_mib(path: Path) -> int:
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream, skipinitialspace=True))
    memory_column = next(index for index, name in enumerate(rows[0]) if "memory.used" in name)
    return max(int(re.search(r"[0-9]+", row[memory_column]).group()) for row in rows[1:])


def _common_configuration(case_configs: list[dict[str, Any]], case_id: int) -> dict[str, Any]:
    excluded = {
        "name",
        "seed",
        "seed_replicate",
        "case_root",
        "slurm_job_id",
        "data_dir",
    }
    common: dict[str, Any] = {"case_id": case_id}
    for key, value in case_configs[0].items():
        if key not in excluded and all(config.get(key) == value for config in case_configs[1:]):
            common[key] = value
    return common


def _support_job(run_root: Path, role: str, job_id: str) -> dict[str, Any]:
    job = _sacct_job(job_id)
    if job["status"] != "COMPLETED" or job["exit_code"] != "0:0":
        raise ValueError(f"support job {job_id} was not successful: {job}")
    job["role"] = role
    job["logs"] = [
        str((run_root / f"em_kclass_matrix_{role}.out").resolve()),
        str((run_root / f"em_kclass_matrix_{role}.err").resolve()),
    ]
    return job


def _reproduction_environment(env: dict[str, str]) -> dict[str, str]:
    values = {
        "PYTHONNOUSERSITE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    for name in (
        "RELION_SRC_DIR",
        "EM_KCLASS_MATRIX_RELION_REFINE_MPI",
        "EM_KCLASS_MATRIX_PIXI_PY",
        "SBATCH_ACCOUNT",
        "SBATCH_PARTITION",
        "SBATCH_CONSTRAINT",
        "EM_KCLASS_MATRIX_SETUP_PARTITION",
        "EM_KCLASS_MATRIX_SETUP_CONSTRAINT",
        "EM_KCLASS_MATRIX_SUMMARY_PARTITION",
        "EM_KCLASS_MATRIX_SUMMARY_CONSTRAINT",
        "RELION_MODULE",
        "RELION_MPI_RANKS",
        "KCLASS_IMAGE_BATCH_SIZE",
        "KCLASS_ROTATION_BLOCK_SIZE",
        "EM_KCLASS_MATRIX_GT_ALIGN_REFINE_ORDERS",
    ):
        values[name] = env[name]
    return values


def _reproduction_command(env: dict[str, str], case_id: int) -> str:
    repo = env["REPO_ROOT"]
    python = env["EM_KCLASS_MATRIX_PIXI_PY"]
    scratch_parent = str(Path(env["SCRATCH_DIR"]).parent)
    replay_environment = _reproduction_environment(env)
    exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in replay_environment.items())
    return (
        f"cd {shlex.quote(repo)} && "
        "unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV "
        "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER "
        "EM_KCLASS_MATRIX_MAX_ITER EM_KCLASS_MATRIX_TIME_LIMIT EM_KCLASS_MATRIX_SEED "
        "EM_KCLASS_MATRIX_SEED_OFFSET EM_KCLASS_MATRIX_EXCLUSIVE && "
        f"export {exports} && "
        f"export FRESH_RUN_ROOT={shlex.quote(scratch_parent + f'/k4_case{case_id}_negative_replay')} && "
        "test ! -e \"$FRESH_RUN_ROOT\" && "
        f"{shlex.quote(python)} scripts/run_em_kclass_robustness_matrix_slurm.py "
        f"--scratch-dir \"$FRESH_RUN_ROOT\" --three-seed-suite --case {case_id} --watch"
    )


def _seal_run(run_root: Path, case_id: int) -> dict[str, Any]:
    run_root = run_root.resolve()
    submission_path = run_root / "submission.env"
    env = _parse_env(submission_path)
    rows = _parse_case_table(run_root / "case_table.tsv", case_id)
    expected_seeds = [41001, 41002, 41003]
    if [int(row["seed"]) for row in rows] != expected_seeds:
        raise ValueError(f"case {case_id} does not contain the frozen three-seed suite")

    recovar_source = _git_source(Path(env["REPO_ROOT"]))
    if recovar_source["commit"] != env["EXPECTED_GIT_HEAD"]:
        raise ValueError(f"case {case_id} source commit conflicts with submission.env")
    relion_checkout = Path(env["RELION_SRC_DIR"]).resolve().parent
    relion_source = {
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=relion_checkout),
        "tree": _run(["git", "rev-parse", "HEAD^{tree}"], cwd=relion_checkout),
        "checkout": str(relion_checkout),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=relion_checkout)
        ).hexdigest(),
        "executable": _file_ref(Path(env["EM_KCLASS_MATRIX_RELION_REFINE_MPI"])),
    }

    case_configs: list[dict[str, Any]] = []
    replicates: list[dict[str, Any]] = []
    for row in rows:
        case_root = Path(row["case_root"]).resolve()
        case_config_path = case_root / "case_config.json"
        case_config = json.loads(case_config_path.read_text())
        case_configs.append(case_config)
        audit_path = case_root / "relion_ref" / "class_population_audit.json"
        audit = json.loads(audit_path.read_text())
        if audit["passed"] or not audit["collapsed"]:
            raise ValueError(f"case is not a RELION-collapse negative: {case_root}")
        recovar_files = list((case_root / "recovar").rglob("*"))
        if any(path.is_file() for path in recovar_files):
            raise ValueError(f"RECOVAR produced output before the collapse gate: {case_root}")

        final_iteration = max(audit["numbered_iterations"])
        final_rows = sorted(
            (entry for entry in audit["rows"] if entry["iteration"] == final_iteration),
            key=lambda entry: entry["class"],
        )
        gpu_model, gpu_uuid = _gpu_identity(case_root)
        job = _sacct_job(row["job_id"])
        if job["status"] != "FAILED" or job["exit_code"] == "0:0":
            raise ValueError(f"collapse gate job was not recorded as a failed harness job: {job}")
        job.update(
            {
                "gpu_model": gpu_model,
                "gpu_uuid": gpu_uuid,
                "logs": [
                    str(
                        (
                            run_root
                            / f"em_kclass_matrix_{row['index']}_{row['name']}.out"
                        ).resolve()
                    ),
                    str(
                        (
                            run_root
                            / f"em_kclass_matrix_{row['index']}_{row['name']}.err"
                        ).resolve()
                    ),
                ],
            }
        )

        data_dir = case_root / "data"
        launcher_path = Path(row["script"]).resolve()
        walltime_path = case_root / "relion_ref" / "slurm_walltime.json"
        walltime = json.loads(walltime_path.read_text())
        relion_monitor = case_root / "relion_gpu_monitor.csv"
        stdout_path, stderr_path = (Path(path) for path in job["logs"])
        replicates.append(
            {
                "name": row["name"],
                "seed": int(row["seed"]),
                "case_root": str(case_root),
                "job": job,
                "launcher": _file_ref(launcher_path),
                "case_config": _file_ref(case_config_path),
                "inputs": [
                    _file_ref(data_dir / "particles.128.mrcs", role="particles"),
                    _file_ref(data_dir / "poses.pkl", role="poses"),
                    _file_ref(data_dir / "ctf.pkl", role="ctf"),
                    _file_ref(data_dir / "generation_config.json", role="generation_config"),
                    _file_ref(data_dir / "class_manifest.json", role="class_manifest"),
                    _file_ref(
                        data_dir / "reference_init_classes_relion.star",
                        role="relion_initial_reference_star",
                    ),
                ],
                "artifacts": [
                    _file_ref(audit_path, role="relion_class_population_audit"),
                    _file_ref(walltime_path, role="relion_walltime"),
                    _file_ref(relion_monitor, role="relion_gpu_monitor"),
                    _file_ref(case_root / "physical_gpu_inventory.csv", role="gpu_inventory"),
                    _file_ref(case_root / "physical_gpu_uuid.txt", role="gpu_uuid"),
                    _file_ref(stdout_path, role="job_stdout"),
                    _file_ref(stderr_path, role="job_stderr"),
                ],
                "relion_collapse": {
                    "status": "ZERO_CLASS",
                    "numbered_iterations": audit["numbered_iterations"],
                    "earliest_iteration": min(entry["iteration"] for entry in audit["collapsed"]),
                    "collapsed_events": audit["collapsed"],
                    "collapsed_event_count": len(audit["collapsed"]),
                    "final_class_distributions": [entry["class_distribution"] for entry in final_rows],
                    "final_orientation_masses": [entry["orientation_mass"] for entry in final_rows],
                    "zero_classes": [
                        entry["class"]
                        for entry in final_rows
                        if entry["class_distribution"] == 0 or entry["orientation_mass"] == 0
                    ],
                },
                "quality": {
                    "comparable_endpoint": False,
                    "fsc_evaluated": False,
                    "class_assignment_evaluated": False,
                    "missing_reason": (
                        "RELION had a zero-population class in numbered iteration 1; the fail-closed "
                        "harness stopped before RECOVAR, Hungarian matching, or FSC evaluation."
                    ),
                },
                "performance": {
                    "relion_wall_s": walltime["external_wall_s"],
                    "relion_peak_hbm_mib": _peak_hbm_mib(relion_monitor),
                    "recovar_wall_s": None,
                    "recovar_peak_hbm_mib": None,
                    "comparison_available": False,
                    "missing_reason": "RECOVAR was not launched after the RELION class-collapse gate failed.",
                },
                "outcome": {
                    "endpoint_status": "NOT_EVALUABLE",
                    "formal_status": "NOT_EVALUABLE",
                    "science_status": "BOUNDARY",
                    "classification": "NEGATIVE_RELION_CLASS_COLLAPSE",
                    "recovar_started": False,
                    "reason": (
                        "RELION assigned zero class and orientation mass to at least one class; "
                        "the paired RECOVAR comparison is scientifically undefined."
                    ),
                },
            }
        )

    custom_cuda = Path(env["RECOVAR_CUDA_LIB"])
    support_jobs = [
        _support_job(run_root, "setup", env["SETUP_JOB_ID"]),
        _support_job(run_root, "summary", env["SUMMARY_JOB_ID"]),
    ]
    return {
        "case_id": case_id,
        "name": case_configs[0]["base_name"],
        "run_root": str(run_root),
        "source": {
            "recovar": recovar_source,
            "relion": relion_source,
            "custom_cuda": _file_ref(custom_cuda),
        },
        "configuration": _common_configuration(case_configs, case_id),
        "reproduction": {
            "command": _reproduction_command(env, case_id),
            "environment": {
                "set": _reproduction_environment(env),
                "confirmed_unset": [
                    "PYTHONPATH",
                    "PYTHONHOME",
                    "CONDA_PREFIX",
                    "VIRTUAL_ENV",
                    "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT",
                    "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER",
                    "EM_KCLASS_MATRIX_MAX_ITER",
                    "EM_KCLASS_MATRIX_TIME_LIMIT",
                    "EM_KCLASS_MATRIX_SEED",
                    "EM_KCLASS_MATRIX_SEED_OFFSET",
                    "EM_KCLASS_MATRIX_EXCLUSIVE",
                ],
            },
            "fresh_root_required": True,
        },
        "support_jobs": support_jobs,
        "shared_artifacts": [
            _file_ref(run_root / "SAFE_TO_DELETE", role="safe_to_delete_marker"),
            _file_ref(run_root / "case_table.tsv", role="case_table"),
            _file_ref(submission_path, role="submission_environment"),
            _file_ref(run_root / "em_kclass_matrix_setup.out", role="setup_stdout"),
            _file_ref(run_root / "em_kclass_matrix_setup.err", role="setup_stderr"),
            _file_ref(run_root / "em_kclass_matrix_summary.out", role="summary_stdout"),
            _file_ref(run_root / "em_kclass_matrix_summary.err", role="summary_stderr"),
            _file_ref(run_root / "em_kclass_multiseed_summary.json", role="multiseed_summary_json"),
            _file_ref(run_root / "em_kclass_multiseed_summary.md", role="multiseed_summary_markdown"),
            _file_ref(run_root / "em_kclass_robustness_summary.json", role="matrix_summary_json"),
            _file_ref(run_root / "em_kclass_robustness_summary.md", role="matrix_summary_markdown"),
            _file_ref(
                run_root / "slurm_case_accounting.json",
                role="historical_accounting_snapshot_may_be_stale",
            ),
        ],
        "replicates": replicates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="CASE_ID:RUN_ROOT",
        help="Completed negative campaign root; repeat for every case.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_specs: list[tuple[int, Path]] = []
    for value in args.run:
        raw_case, separator, raw_root = value.partition(":")
        if not separator:
            parser.error(f"--run must be CASE_ID:RUN_ROOT, got {value!r}")
        run_specs.append((int(raw_case), Path(raw_root)))
    runs = [_seal_run(run_root, case_id) for case_id, run_root in run_specs]
    expected_seeds = [41001, 41002, 41003]

    first_manifest_path = (
        Path(runs[0]["replicates"][0]["case_root"]) / "data" / "class_manifest.json"
    )
    source_structures = _manifest_source_structures(first_manifest_path)
    for run in runs:
        if run["configuration"].get("n_classes") != 4 or run["configuration"].get("symmetry") != "C1":
            raise ValueError(f"case {run['case_id']} is not a C1 K=4 diagnostic")
        for replicate in run["replicates"]:
            manifest_path = Path(replicate["case_root"]) / "data" / "class_manifest.json"
            if _manifest_source_structures(manifest_path) != source_structures:
                raise ValueError(f"source structures differ across replicates: {manifest_path}")
    verification_python = runs[0]["reproduction"]["environment"]["set"][
        "EM_KCLASS_MATRIX_PIXI_PY"
    ]
    checked_at = datetime.now(timezone.utc).isoformat()
    diagnostic_id = args.output.stem
    diagnostic = {
        "schema_version": 1,
        "diagnostic_id": diagnostic_id,
        "recorded_at": checked_at,
        "record_type": "synthetic_kclass_negative_campaign_collection",
        "registry_disposition": "EXCLUDED_FROM_ACCEPTED_RESULTS",
        "subject": {
            "kind": "synthetic",
            "molecular_family": "CryoBench2 Ribosembly",
            "k": 4,
            "symmetry": "C1",
            "diagnostic_question": (
                "Can removing CTF variation, lowering noise, or improving the initial reference "
                "produce a non-collapsed paired K=4 trajectory?"
            ),
        },
        "source_structures": source_structures,
        "expected_seeds": expected_seeds,
        "runs": runs,
        "conclusion": {
            "accepted_result": False,
            "all_replicates_relion_zero_class": True,
            "recovar_was_evaluated": False,
            "classification": "NEGATIVE_RELION_CLASS_COLLAPSE",
            "interpretation": (
                "All nine RELION controls assigned exactly zero mass to class 3 by iteration 1; "
                "seed 41002 also ended with class 4 at zero. These cells reject the proposed "
                "no-CTF positive-control fixtures but provide no evidence for or against RECOVAR."
            ),
            "limitations": [
                "The fail-closed harness did not launch RECOVAR after the RELION oracle collapsed.",
                "No cross-engine FSC, GT FSC-AUC, class agreement, or timing ratio is defined.",
                "The checked external run roots are disposable and carry SAFE_TO_DELETE markers; "
                "reproduction is bound to hashed source structures, configs, inputs, and launchers.",
            ],
        },
        "verification": {
            "evidence_checked_at": checked_at,
            "commands": [
                f"{verification_python} scripts/validate_em_benchmark_registry.py",
                f"{verification_python} scripts/validate_em_benchmark_registry.py --verify-files",
                f"{verification_python} -m pytest tests/unit/test_validate_em_benchmark_registry.py",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostic, indent=2, sort_keys=False) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
