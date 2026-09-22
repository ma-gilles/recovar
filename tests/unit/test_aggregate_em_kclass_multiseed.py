import csv
import json
from pathlib import Path

import pytest

from scripts import aggregate_em_kclass_multiseed as aggregate

HEADER = [
    "index",
    "name",
    "n_classes",
    "n_images",
    "grid",
    "noise_level",
    "noise_model",
    "dataset_params_option",
    "seed",
    "pdb_bfactor",
    "init_radius",
    "noise_scale_std",
    "contrast_std",
    "volume_radius",
    "image_offset_n_std",
    "percent_outliers",
    "max_iter",
    "class_distribution",
    "time_limit",
    "mem",
    "image_batch_size_override",
    "rotation_block_size_override",
    "symmetry",
    "base_name",
    "base_seed",
    "seed_replicate",
    "shared_input_group",
    "shared_input_role",
    "pdb_dir",
    "case_root",
    "script",
    "job_id",
]


def _write_suite(root, *, seeds=(41001, 41002, 41003), changed_symmetry_seed=None, collapse_seed=None):
    rows = []
    summaries = []
    for replicate, seed in enumerate(seeds, start=1):
        name = f"ribo_k4_5k_g128_white_noise1_c4_uniform_seed{seed}"
        case_root = root / "cases" / f"31_{name}"
        relion = case_root / "relion_ref"
        relion.mkdir(parents=True)
        collapsed = (
            [{"iteration": 1, "class": 4, "class_distribution": 0.0, "orientation_mass": 0.0}]
            if seed == collapse_seed
            else []
        )
        (relion / "class_population_audit.json").write_text(
            json.dumps(
                {
                    "schema": "recovar.em.relion_class_population_audit.v1",
                    "passed": not collapsed,
                    "collapsed": collapsed,
                }
            )
        )
        symmetry = "D4" if seed == changed_symmetry_seed else "C4"
        job_id = str(70000 + replicate)
        rows.append(
            {
                "index": "31",
                "name": name,
                "n_classes": "4",
                "n_images": "5000",
                "grid": "128",
                "noise_level": "1",
                "noise_model": "white",
                "dataset_params_option": "uniform",
                "seed": str(seed),
                "pdb_bfactor": "80",
                "init_radius": "10",
                "noise_scale_std": "0",
                "contrast_std": "0",
                "volume_radius": "0.7",
                "image_offset_n_std": "0",
                "percent_outliers": "0",
                "max_iter": "5",
                "class_distribution": "uniform",
                "time_limit": "06:00:00",
                "mem": "256G",
                "image_batch_size_override": "",
                "rotation_block_size_override": "",
                "symmetry": symmetry,
                "base_name": "ribo_k4_5k_g128_white_noise1_c4_uniform",
                "base_seed": "41001",
                "seed_replicate": str(replicate),
                "shared_input_group": "",
                "shared_input_role": "",
                "pdb_dir": "/sealed/ribosembly/pdbs",
                "case_root": str(case_root),
                "script": str(root / "jobs" / f"case_{seed}.sh"),
                "job_id": job_id,
            }
        )
        (case_root / "case_config.json").write_text(
            json.dumps(
                {
                    "index": 31,
                    "name": name,
                    "pdb_dir": "/sealed/ribosembly/pdbs",
                    "n_classes": 4,
                    "n_images": 5000,
                    "grid_size": 128,
                    "noise_level": 1.0,
                    "noise_model": "white",
                    "dataset_params_option": "uniform",
                    "class_distribution": "uniform",
                    "seed": seed,
                    "pdb_bfactor": 80.0,
                    "init_radius": 10,
                    "noise_scale_std": 0.0,
                    "contrast_std": 0.0,
                    "volume_radius": 0.7,
                    "image_offset_n_std": 0.0,
                    "percent_outliers": 0.0,
                    "max_iter": 5,
                    "symmetry": symmetry,
                    "base_name": "ribo_k4_5k_g128_white_noise1_c4_uniform",
                    "base_seed": 41001,
                    "seed_replicate": replicate,
                    "shared_input_group": None,
                    "shared_input_role": None,
                    "case_root": str(case_root.resolve()),
                    "slurm_job_id": job_id,
                }
            )
        )
        summaries.append(
            {
                "case_root": str(case_root),
                "case_name": name,
                "job_id": job_id,
                "status": "failed" if collapsed else "ok",
                "slurm_state": "FAILED" if collapsed else "COMPLETED",
                "slurm_exit_code": "1:0" if collapsed else "0:0",
                "failure_reason": ("RELION class-collapse gate failed" if collapsed else None),
                "fsc_auc_vs_gt": 0.22 + replicate * 0.001,
                "relion_fsc_auc_vs_gt": 0.221 + replicate * 0.001,
                "fsc_auc_delta_vs_relion": -0.001,
                "wall_s": 1000 + replicate,
                "relion_wall_s": 100 + replicate,
                "recovar_peak_gpu_memory_mib": 22000 + replicate,
                "relion_peak_gpu_memory_mib": 19000 + replicate,
                "slurm_max_rss_mib": 3000 + replicate,
            }
        )

    case_table = root / "case_table.tsv"
    with case_table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADER, delimiter="|")
        writer.writeheader()
        writer.writerows(rows)
    matrix_summary = root / "em_kclass_robustness_summary.json"
    matrix_summary.write_text(json.dumps({"schema": "em_robustness_matrix_summary_v1", "cases": summaries}))
    return case_table, matrix_summary


def _trajectory_payload(
    *,
    n_classes: int,
    iterations: int,
    direct_value: float = 0.999,
    gt_delta: float = -0.0001,
    agreement: float = 0.999,
    status: str = "pass",
) -> dict:
    failure = "it001 class 1 strict trajectory gate failed"
    return {
        "schema": (
            "em_k4_fsc_trajectory_audit_v2"
            if n_classes == 4
            else "em_kclass_fsc_trajectory_audit_v1"
        ),
        "status": status,
        "n_classes": n_classes,
        "numbered_iteration_count": iterations,
        "quality_metric_policy": aggregate.TRAJECTORY_METRIC_POLICY,
        "thresholds": {
            "per_class_direct_fsc_auc_min": 0.995,
            "per_class_recovar_minus_relion_gt_fsc_auc_min": -0.002,
            "class_assignment_agreement_min_when_available": 0.99,
        },
        "failures": [] if status == "pass" else [failure],
        "earliest_failure": None if status == "pass" else failure,
        "numbered_iterations": [
            {
                "relion_iteration": iteration,
                "classes": [
                    {
                        "recovar_class": class_id,
                        "relion_class": class_id,
                        "gt_class": class_id,
                        "cross_engine": {"fsc_auc": direct_value},
                        "gt_fsc_auc_delta": gt_delta,
                        "vs_gt": {
                            "recovar": {"fsc_auc": 0.25 + gt_delta},
                            "relion": {"fsc_auc": 0.25},
                        },
                    }
                    for class_id in range(1, n_classes + 1)
                ],
                "class_agreement": {"status": "available", "agreement": agreement},
            }
            for iteration in range(1, iterations + 1)
        ],
        "final": {
            "classes": [
                {
                    "recovar_class": class_id,
                    "relion_class": class_id,
                    "gt_class": class_id,
                    "cross_engine": {"fsc_auc": direct_value},
                    "gt_fsc_auc_delta": gt_delta,
                    "vs_gt": {
                        "recovar": {"fsc_auc": 0.25 + gt_delta},
                        "relion": {"fsc_auc": 0.25},
                    },
                }
                for class_id in range(1, n_classes + 1)
            ]
        },
    }


def _write_trajectory_audits(
    root: Path,
    *,
    direct_value: float = 0.999,
    gt_delta: float = -0.0001,
    agreement: float = 0.999,
    status: str = "pass",
) -> None:
    with (root / "case_table.tsv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    for row in rows:
        case_root = Path(row["case_root"])
        n_classes = int(row["n_classes"])
        iterations = int(row["max_iter"])
        output = case_root / "trajectory_analysis"
        output.mkdir()
        payload = _trajectory_payload(
            n_classes=n_classes,
            iterations=iterations,
            direct_value=direct_value,
            gt_delta=gt_delta,
            agreement=agreement,
            status=status,
        )
        (output / f"k{n_classes}_fsc_trajectory.json").write_text(json.dumps(payload))


def test_three_seed_aggregate_requires_audited_completion_and_reports_metrics(tmp_path):
    case_table, matrix_summary = _write_suite(tmp_path)

    payload = aggregate.aggregate(
        tmp_path,
        matrix_summary=matrix_summary,
        case_table=case_table,
    )

    assert payload["outcome"] == "COMPLETE_ALL_CASES_ALL_SEEDS"
    assert payload["expected_seeds"] == [41001, 41002, 41003]
    assert payload["base_case_count"] == 1
    assert payload["replicate_count"] == 3
    case = payload["cases"][0]
    assert case["outcome"] == "COMPLETE_ALL_SEEDS"
    assert [row["outcome"] for row in case["replicates"]] == ["COMPLETE"] * 3
    assert case["metrics_across_seeds"]["worst_recovar_minus_relion_gt_fsc_auc"] == -0.001
    assert case["metrics_across_seeds"]["median_recovar_wall_s"] == 1002.0
    assert payload["formal_gate_claim"] is None
    assert len(payload["artifacts"]["case_table_sha256"]) == 64
    assert all(len(row["case_config_sha256"]) == 64 for row in case["replicates"])


def test_trajectory_v2_replays_every_seed_and_class_cell(tmp_path):
    case_table, matrix_summary = _write_suite(tmp_path)
    _write_trajectory_audits(tmp_path)

    payload = aggregate.aggregate(
        tmp_path,
        matrix_summary=matrix_summary,
        case_table=case_table,
        require_trajectory_audits=True,
    )

    assert payload["schema"] == aggregate.TRAJECTORY_SCHEMA
    assert payload["formal_gate_claim"] == {
        "status": "PASS",
        "audited_replicates": 3,
        "passing_replicates": 3,
        "evaluated_class_cells": 72,
        "passing_class_cells": 72,
        "audit_sha256": [
            row["trajectory_audit"]["sha256"]
            for row in payload["cases"][0]["replicates"]
        ],
    }
    assert payload["science_equivalence_claim"] == {
        "status": "PASS",
        "classification": "TRAJECTORY_EXACT",
        "audited_replicates": 3,
        "passing_replicates": 3,
        "evaluated_gt_class_cells": 72,
        "passing_gt_class_cells": 72,
    }
    metrics = payload["cases"][0]["metrics_across_seeds"]
    assert metrics["minimum_direct_fsc_auc"] == 0.999
    assert metrics["minimum_gt_fsc_auc_delta"] == -0.0001
    assert metrics["minimum_final_gt_fsc_auc_delta"] == -0.0001
    assert metrics["minimum_class_assignment_agreement"] == 0.999
    assert payload["cases"][0]["formal_trajectory_status"] == "PASS"
    assert payload["cases"][0]["science_quality_status"] == "PASS"
    assert payload["cases"][0]["quality_classification"] == "TRAJECTORY_EXACT"
    rendered = aggregate.render_markdown(payload, tmp_path / "summary.json")
    assert "formal trajectory claim: **PASS (3/3 replicates; 72/72 class cells)**" in rendered
    assert "science-equivalence claim: **PASS / TRAJECTORY_EXACT" in rendered
    assert "| ribo_k4_5k_g128_white_noise1_c4_uniform | 4 | PASS | PASS | TRAJECTORY_EXACT | 72/72 | 72/72 | 0.999000000 | -0.000100000 | 0.999000 |" in rendered
    assert "Max RELION HBM MiB" in rendered


def test_trajectory_v2_fails_closed_when_a_seed_audit_is_missing(tmp_path):
    case_table, matrix_summary = _write_suite(tmp_path)

    with pytest.raises(ValueError, match="required K=4 trajectory audit is missing"):
        aggregate.aggregate(
            tmp_path,
            matrix_summary=matrix_summary,
            case_table=case_table,
            require_trajectory_audits=True,
        )


def test_trajectory_v2_rejects_status_that_does_not_replay_metrics(tmp_path):
    case_table, matrix_summary = _write_suite(tmp_path)
    _write_trajectory_audits(tmp_path, direct_value=0.994, status="pass")

    with pytest.raises(ValueError, match="status does not replay"):
        aggregate.aggregate(
            tmp_path,
            matrix_summary=matrix_summary,
            case_table=case_table,
            require_trajectory_audits=True,
        )


@pytest.mark.parametrize("n_classes", [2, 8, 16])
def test_reads_generic_kclass_trajectory_schemas(tmp_path, n_classes):
    case_root = tmp_path / f"k{n_classes}"
    output = case_root / "trajectory_analysis"
    output.mkdir(parents=True)
    payload = _trajectory_payload(n_classes=n_classes, iterations=5)
    (output / f"k{n_classes}_fsc_trajectory.json").write_text(json.dumps(payload))

    observed = aggregate.read_trajectory_audit(
        case_root,
        n_classes=n_classes,
        expected_iterations=5,
    )

    assert observed["schema"] == "em_kclass_fsc_trajectory_audit_v1"
    assert observed["status"] == "PASS"
    assert observed["science_status"] == "PASS"
    assert observed["quality_classification"] == "TRAJECTORY_EXACT"
    assert observed["evaluated_class_cells"] == 6 * n_classes


def test_trajectory_v2_keeps_science_equivalence_separate_from_strict_failure(tmp_path):
    case_table, matrix_summary = _write_suite(tmp_path)
    _write_trajectory_audits(tmp_path, direct_value=0.994, agreement=0.98, status="fail")

    payload = aggregate.aggregate(
        tmp_path,
        matrix_summary=matrix_summary,
        case_table=case_table,
        require_trajectory_audits=True,
    )

    assert payload["formal_gate_claim"]["status"] == "FAIL"
    assert payload["science_equivalence_claim"] == {
        "status": "PASS",
        "classification": "SCIENCE_EQUIVALENT",
        "audited_replicates": 3,
        "passing_replicates": 3,
        "evaluated_gt_class_cells": 72,
        "passing_gt_class_cells": 72,
    }
    case = payload["cases"][0]
    assert case["formal_trajectory_status"] == "FAIL"
    assert case["science_quality_status"] == "PASS"
    assert case["quality_classification"] == "SCIENCE_EQUIVALENT"


def test_trajectory_v2_reports_science_gap_when_gt_delta_fails(tmp_path):
    case_table, matrix_summary = _write_suite(tmp_path)
    _write_trajectory_audits(tmp_path, gt_delta=-0.003, status="fail")

    payload = aggregate.aggregate(
        tmp_path,
        matrix_summary=matrix_summary,
        case_table=case_table,
        require_trajectory_audits=True,
    )

    assert payload["formal_gate_claim"]["status"] == "FAIL"
    assert payload["science_equivalence_claim"]["status"] == "FAIL"
    assert payload["science_equivalence_claim"]["classification"] == "SCIENCE_GAP"
    assert payload["science_equivalence_claim"]["passing_gt_class_cells"] == 0
    assert payload["cases"][0]["quality_classification"] == "SCIENCE_GAP"


def test_trajectory_v2_rejects_inconsistent_signed_gt_delta(tmp_path):
    case_root = tmp_path / "k2"
    output = case_root / "trajectory_analysis"
    output.mkdir(parents=True)
    payload = _trajectory_payload(n_classes=2, iterations=5)
    payload["final"]["classes"][0]["gt_fsc_auc_delta"] = 0.0
    (output / "k2_fsc_trajectory.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="signed GT FSC-AUC delta is inconsistent"):
        aggregate.read_trajectory_audit(
            case_root,
            n_classes=2,
            expected_iterations=5,
        )


def test_trajectory_v2_replays_intermediate_gt_delta_gate(tmp_path):
    case_root = tmp_path / "k2"
    output = case_root / "trajectory_analysis"
    output.mkdir(parents=True)
    payload = _trajectory_payload(n_classes=2, iterations=5)
    class_row = payload["numbered_iterations"][2]["classes"][0]
    class_row["vs_gt"]["recovar"]["fsc_auc"] = 0.247
    class_row["gt_fsc_auc_delta"] = -0.003
    (output / "k2_fsc_trajectory.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="status does not replay"):
        aggregate.read_trajectory_audit(
            case_root,
            n_classes=2,
            expected_iterations=5,
        )


def test_trajectory_v2_allows_explicitly_unavailable_class_agreement(tmp_path):
    case_root = tmp_path / "k2"
    output = case_root / "trajectory_analysis"
    output.mkdir(parents=True)
    payload = _trajectory_payload(n_classes=2, iterations=5)
    for row in payload["numbered_iterations"]:
        row["class_agreement"] = {"status": "unavailable", "reason": "sealed fixture"}
    (output / "k2_fsc_trajectory.json").write_text(json.dumps(payload))

    observed = aggregate.read_trajectory_audit(
        case_root,
        n_classes=2,
        expected_iterations=5,
    )

    assert observed["status"] == "PASS"
    assert observed["minimum_class_assignment_agreement"] is None
    assert observed["class_agreement_unavailable_iterations"] == [1, 2, 3, 4, 5]


def test_three_seed_aggregate_follows_shared_relion_oracle_symlink(tmp_path):
    case_table, matrix_summary = _write_suite(tmp_path)
    with case_table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    consumer_root = Path(rows[1]["case_root"])
    local_relion = consumer_root / "relion_ref"
    shared_relion = tmp_path / "shared_inputs" / "sealed_relion_ref"
    shared_relion.mkdir(parents=True)
    (local_relion / "class_population_audit.json").replace(shared_relion / "class_population_audit.json")
    local_relion.rmdir()
    local_relion.symlink_to(shared_relion, target_is_directory=True)

    payload = aggregate.aggregate(
        tmp_path,
        matrix_summary=matrix_summary,
        case_table=case_table,
    )

    assert payload["outcome"] == "COMPLETE_ALL_CASES_ALL_SEEDS"
    assert payload["cases"][0]["replicates"][1]["class_population_audit_sha256"]


def test_three_seed_aggregate_preserves_class_collapse_as_suite_failure(tmp_path):
    case_table, matrix_summary = _write_suite(tmp_path, collapse_seed=41002)

    payload = aggregate.aggregate(
        tmp_path,
        matrix_summary=matrix_summary,
        case_table=case_table,
    )

    assert payload["outcome"] == "FAIL_CLASS_COLLAPSE"
    assert payload["cases"][0]["outcome"] == "FAIL_CLASS_COLLAPSE"
    replicate = payload["cases"][0]["replicates"][1]
    assert replicate["outcome"] == "FAIL_CLASS_COLLAPSE"
    assert replicate["class_population_audit_sha256"] is not None


def test_three_seed_aggregate_rejects_missing_seed(tmp_path):
    case_table, matrix_summary = _write_suite(tmp_path, seeds=(41001, 41002))

    with pytest.raises(ValueError, match="must contain seeds"):
        aggregate.aggregate(
            tmp_path,
            matrix_summary=matrix_summary,
            case_table=case_table,
        )


def test_three_seed_aggregate_rejects_seed_replicate_reordering(tmp_path):
    case_table, matrix_summary = _write_suite(tmp_path, seeds=(41003, 41002, 41001))

    with pytest.raises(ValueError, match="must map seed_replicates.*in order"):
        aggregate.aggregate(
            tmp_path,
            matrix_summary=matrix_summary,
            case_table=case_table,
        )


def test_three_seed_aggregate_rejects_case_root_identity_swap(tmp_path):
    case_table, matrix_summary = _write_suite(tmp_path)
    with case_table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    rows[0]["case_root"], rows[2]["case_root"] = rows[2]["case_root"], rows[0]["case_root"]
    with case_table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADER, delimiter="|")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="case identity mismatch"):
        aggregate.aggregate(
            tmp_path,
            matrix_summary=matrix_summary,
            case_table=case_table,
        )


def test_three_seed_aggregate_rejects_scientific_axis_drift(tmp_path):
    case_table, matrix_summary = _write_suite(tmp_path, changed_symmetry_seed=41003)

    with pytest.raises(ValueError, match="changes frozen scientific columns.*symmetry"):
        aggregate.aggregate(
            tmp_path,
            matrix_summary=matrix_summary,
            case_table=case_table,
        )
