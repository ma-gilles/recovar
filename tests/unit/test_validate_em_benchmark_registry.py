"""Unit tests for the checked-in EM benchmark evidence registry."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import scripts.validate_em_benchmark_registry as registry_validator
from scripts.validate_em_benchmark_registry import (
    RegistryValidationError,
    validate_campaign,
    validate_diagnostic,
    validate_record,
    validate_registry,
    verify_campaign_files,
    verify_diagnostic_files,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = REPO_ROOT / "docs" / "benchmarks" / "em"
SCHEMA = json.loads((REGISTRY_ROOT / "schema_v1.json").read_text())
CAMPAIGN_SCHEMA = json.loads((REGISTRY_ROOT / "campaign_schema_v1.json").read_text())
DIAGNOSTIC_SCHEMA = json.loads((REGISTRY_ROOT / "diagnostic_schema_v1.json").read_text())
CANDIDATE = json.loads(
    (
        REGISTRY_ROOT
        / "entries"
        / "k4-ribosembly-100k256-1b9209cd8-h100.json"
    ).read_text()
)
CAMPAIGN = json.loads(
    (
        REGISTRY_ROOT
        / "campaigns"
        / "k4-expanded14-3466e7a32-h100"
    ).with_suffix(".json").read_text()
)
C4_SYMMETRY_CAMPAIGN = json.loads(
    (
        REGISTRY_ROOT
        / "campaigns"
        / "k4-c4-three-seed-c75cbfffc-h100"
    ).with_suffix(".json").read_text()
)
D4_SYMMETRY_CAMPAIGN = json.loads(
    (
        REGISTRY_ROOT
        / "campaigns"
        / "k4-d4-three-seed-c75cbfffc-h100"
    ).with_suffix(".json").read_text()
)
O_SYMMETRY_CAMPAIGN = json.loads(
    (
        REGISTRY_ROOT
        / "campaigns"
        / "k4-o-three-seed-22efd8065-h100"
    ).with_suffix(".json").read_text()
)
I1_SYMMETRY_CAMPAIGN = json.loads(
    (
        REGISTRY_ROOT
        / "campaigns"
        / "k4-i1-three-seed-22efd8065-h100"
    ).with_suffix(".json").read_text()
)
EXACT_INPUT_CAMPAIGN = json.loads(
    (
        REGISTRY_ROOT
        / "campaigns"
        / "k4-exact-input-invariance-91e8a30f4-h100"
    ).with_suffix(".json").read_text()
)
NEGATIVE_DIAGNOSTIC = json.loads(
    (
        REGISTRY_ROOT
        / "diagnostics"
        / "k4-noctf-collapse-cases30-35-36-h100"
    ).with_suffix(".json").read_text()
)
NATIVE_COARSE_SCORE_DIAGNOSTIC = json.loads(
    (
        REGISTRY_ROOT
        / "diagnostics"
        / "real-k4-native-coarse-score-a32cccccb-20260902"
    ).with_suffix(".json").read_text()
)
NATIVE_COARSE_COMPONENT_DIAGNOSTIC = json.loads(
    (
        REGISTRY_ROOT
        / "diagnostics"
        / "real-k4-native-coarse-components-8f9ebc9-20260902"
    ).with_suffix(".json").read_text()
)
NATIVE_COARSE_OPERAND_DIAGNOSTIC = json.loads(
    (
        REGISTRY_ROOT
        / "diagnostics"
        / "real-k4-native-coarse-operands-b061776-20260902"
    ).with_suffix(".json").read_text()
)


def test_checked_in_em_benchmark_registry_is_valid():
    paths = validate_registry(REGISTRY_ROOT)
    assert [path.stem for path in paths] == [
        "k4-igg-10k128-white1-uniform-0050dc54f-h100",
        "k4-ribosembly-100k256-1b9209cd8-h100",
        "k4-ribosembly-100k256-ac5177d2-a100",
        "k4-ribosembly-10k128-radial3-nonuniform-linear-0050dc54f-h100",
        "k4-ribosembly-10k128-radial3-nonuniform-outliers20-0050dc54f-h100",
        "k4-ribosembly-10k128-white1-uniform-0050dc54f-h100",
        "k4-c4-three-seed-c75cbfffc-h100",
        "k4-d4-three-seed-c75cbfffc-h100",
        "k4-exact-input-invariance-91e8a30f4-h100",
        "k4-expanded14-3466e7a32-h100",
        "k4-i1-three-seed-22efd8065-h100",
        "k4-o-three-seed-22efd8065-h100",
        "k4-noctf-collapse-cases30-35-36-h100",
    ]
    assert [case["case_id"] for case in CAMPAIGN["cases"]] == list(range(16, 30))
    classifications = {case["case_id"]: case["outcome"]["classification"] for case in CAMPAIGN["cases"]}
    assert classifications[17] == "NEGATIVE_ZERO_CLASS_BOUNDARY"
    assert classifications[20] == "TRAJECTORY_EXACT_NEAR_COLLAPSE"
    assert classifications[27] == "UNRESOLVED_TRAJECTORY_FAILURE"


def test_exact_input_campaign_does_not_promote_gt_only_equivalence():
    by_seed: dict[int, list[dict]] = {}
    for case in EXACT_INPUT_CAMPAIGN["cases"]:
        by_seed.setdefault(int(case["configuration"]["seed"]), []).append(case)

    assert {seed: len(cases) for seed, cases in by_seed.items()} == {
        41001: 3,
        41002: 3,
        41003: 3,
    }
    for case in by_seed[41002]:
        assert case["outcome"]["classification"] == "TRAJECTORY_EXACT"
        assert case["outcome"]["science_status"] == "PASS"
    for seed in (41001, 41003):
        for case in by_seed[seed]:
            assert case["outcome"]["classification"] == "UNRESOLVED_TRAJECTORY_FAILURE"
            assert case["outcome"]["science_status"] == "UNRESOLVED"
            assert all(
                row["gt_fsc_auc_delta"] >= -0.002
                for row in case["quality"]["final_classes"]
            )
            assert (
                min(
                    row["cross_engine_fsc_auc"]
                    for row in case["quality"]["final_classes"]
                )
                < 0.99
            )


def test_diagnostic_registry_routing_is_explicit_and_fail_closed():
    assert (
        registry_validator._diagnostic_registry_route(NEGATIVE_DIAGNOSTIC)
        == "synthetic_negative"
    )
    assert (
        registry_validator._diagnostic_registry_route(
            {"schema": "recovar.em_real_kclass_initialmodel_diagnostics.v1"}
        )
        == "separate"
    )
    assert (
        registry_validator._diagnostic_registry_route(
            {"schema": "recovar.em.real_kclass_selected_fine_diagnostic.v1"}
        )
        == "separate"
    )
    assert (
        registry_validator._diagnostic_registry_route(
            {"schema": "recovar.em_real_kclass_offset_prior_fullpairs.v1"}
        )
        == "separate"
    )
    assert (
        registry_validator._diagnostic_registry_route(NATIVE_COARSE_SCORE_DIAGNOSTIC)
        == "separate"
    )
    assert (
        registry_validator._diagnostic_registry_route(
            NATIVE_COARSE_COMPONENT_DIAGNOSTIC
        )
        == "separate"
    )
    assert (
        registry_validator._diagnostic_registry_route(
            NATIVE_COARSE_OPERAND_DIAGNOSTIC
        )
        == "separate"
    )
    with pytest.raises(RegistryValidationError, match="unrecognized diagnostic family"):
        registry_validator._diagnostic_registry_route({"schema": "unknown.v1"})


def test_native_coarse_score_diagnostic_pins_causal_and_admission_boundaries():
    diagnostic = NATIVE_COARSE_SCORE_DIAGNOSTIC
    assert diagnostic["admission_status"] == "DIAGNOSTIC_ONLY"
    assert diagnostic["accepted_result"] is False
    assert diagnostic["execution"]["req_tres_equals_alloc_tres"] is True
    assert diagnostic["execution"]["formal_engine_performance_result"] is False
    assert diagnostic["capture_inertness"]["status"] == "PASS"
    assert diagnostic["capture_inertness"]["bitwise_inert"] is False
    assert min(diagnostic["capture_inertness"]["iteration_1_class_map_fsc_auc"]) > 0.999999

    scores = diagnostic["score_comparison"]
    assert scores["classification"] == (
        "raw_likelihood_surface_is_first_material_support_difference"
    )
    assert scores["orientation_class_prior_centered_max_abs"] == 0
    assert scores["raw_likelihood_centered_max_abs"] > 6

    swaps = diagnostic["support_counterfactuals"]
    assert swaps["native_raw_plus_recovar_prior"]["exact_records"] == 16
    assert swaps["recovar_raw_plus_native_prior"]["exact_records"] == 5
    assert swaps["recovar_raw_plus_native_prior"] == swaps["recovar_combined"]
    assert swaps["boundary_tie_records"] == 0


def test_native_coarse_component_diagnostic_pins_causal_and_inertness_boundaries():
    diagnostic = NATIVE_COARSE_COMPONENT_DIAGNOSTIC
    assert diagnostic["admission_status"] == "DIAGNOSTIC_ONLY"
    assert diagnostic["accepted_result"] is False
    assert diagnostic["execution"]["native_replay"]["req_tres_equals_alloc_tres"]
    assert diagnostic["execution"]["recovar_components"]["req_tres_equals_alloc_tres"]
    assert diagnostic["capture_inertness"]["status"] == "PASS"
    assert diagnostic["capture_inertness"]["iteration_1_decision_fields_exact"]
    assert min(diagnostic["capture_inertness"]["iteration_1_class_map_fsc_auc"]) > 0.999999

    comparison = diagnostic["component_comparison"]
    assert comparison["classification"] == (
        "cross_term_is_dominant_component_of_first_material_likelihood_difference"
    )
    assert comparison["cross_term_pooled_centered_rms"] > (
        comparison["reference_norm_pooled_centered_rms"]
    )
    assert comparison["pooled_raw_residual_energy_remaining_after_native_cross"] < (
        comparison["pooled_raw_residual_energy_remaining_after_native_norm"]
    )

    swaps = diagnostic["support_counterfactuals"]
    assert swaps["native_combined"]["exact_records"] == 16
    assert swaps["recovar_combined"]["exact_records"] == 5
    assert swaps["native_cross_only"]["exact_records"] == 11
    assert swaps["native_norm_only"]["exact_records"] == 5
    assert swaps["boundary_tie_records"] == 0
    assert diagnostic["rejected_in_kernel_capture"]["accepted_evidence"] is False


def test_native_coarse_operand_diagnostic_pins_factorial_and_repeatability_boundaries():
    diagnostic = NATIVE_COARSE_OPERAND_DIAGNOSTIC
    assert diagnostic["admission_status"] == "DIAGNOSTIC_ONLY"
    assert diagnostic["accepted_result"] is False
    assert diagnostic["execution"]["all_req_tres_equal_alloc_tres"]
    assert diagnostic["execution"]["formal_engine_performance_result"] is False

    repeatability = diagnostic["capture_repeatability"]
    assert repeatability["status"] == "PASS"
    assert repeatability["repeat_count"] == 3
    assert repeatability["passed_repeats"] == 3
    assert repeatability["iteration_1_decision_exact_repeats"] == 3
    assert repeatability["class_map_fsc_auc_minimum"] > 0.999999
    assert repeatability["class_map_relative_l2_maximum"] < 1e-5

    comparison = diagnostic["operand_comparison"]
    assert comparison["classification"] == (
        "projected_reference_is_first_material_k4_coarse_likelihood_operand_difference"
    )
    energy = comparison["fraction_baseline_energy_remaining"]
    assert energy["native_projected_reference"] < 1e-6
    assert energy["native_shifted_image"] > 0.99
    assert energy["native_correction"] > 0.99
    assert comparison["projected_reference_relative_l2"] > 0.01
    assert comparison["shifted_image_relative_l2_maximum"] < 1e-5
    assert comparison["correction_relative_l2_maximum"] < 1e-5

    swaps = diagnostic["support_counterfactuals"]
    assert swaps["native_projected_reference"]["exact_records"] == 16
    assert swaps["native_shifted_image"]["exact_records"] == 5
    assert swaps["native_correction"]["exact_records"] == 5
    assert swaps["boundary_tie_records"] == 0
    assert diagnostic["rejected_synchronous_capture"]["accepted_evidence"] is False

    resolved = diagnostic["resolved_projection_boundary"]
    assert resolved["status"] == "PASS"
    assert resolved["classification"] == (
        "k4_coarse_projected_reference_and_significant_support_match_native"
    )
    assert resolved["allocation"]["req_tres_equal_alloc_tres"]
    assert resolved["allocation"]["formal_engine_performance_result"] is False
    assert resolved["exact_boundary"]["projected_reference_max_relative_l2"] == 0
    assert resolved["exact_boundary"]["euler_max_abs"] == 0
    assert resolved["exact_boundary"]["exact_probe_records"] == 16
    assert resolved["exact_boundary"]["selected_candidate_intersection"] == 1336
    assert resolved["exact_boundary"]["selected_candidate_union"] == 1336
    assert resolved["repeatability"]["parity_array_count"] == 11
    assert resolved["repeatability"]["parity_arrays_bitwise"]
    assert resolved["repeatability"]["map_fsc_auc_minimum"] > 0.999999
    assert resolved["repeatability"]["map_relative_l2_maximum"] < 1e-5


def test_negative_diagnostic_is_complete_but_never_an_accepted_result():
    assert NEGATIVE_DIAGNOSTIC["registry_disposition"] == "EXCLUDED_FROM_ACCEPTED_RESULTS"
    conclusion = NEGATIVE_DIAGNOSTIC["conclusion"]
    assert conclusion["accepted_result"] is False
    assert conclusion["all_replicates_relion_zero_class"] is True
    assert conclusion["recovar_was_evaluated"] is False
    assert conclusion["classification"] == "NEGATIVE_RELION_CLASS_COLLAPSE"
    assert [run["case_id"] for run in NEGATIVE_DIAGNOSTIC["runs"]] == [30, 35, 36]
    for run in NEGATIVE_DIAGNOSTIC["runs"]:
        assert [replicate["seed"] for replicate in run["replicates"]] == [41001, 41002, 41003]
        for replicate in run["replicates"]:
            assert replicate["relion_collapse"]["status"] == "ZERO_CLASS"
            assert replicate["relion_collapse"]["zero_classes"]
            assert replicate["quality"]["fsc_evaluated"] is False
            assert replicate["outcome"]["recovar_started"] is False
            assert replicate["performance"]["recovar_wall_s"] is None
            assert replicate["performance"]["recovar_peak_hbm_mib"] is None


def test_negative_diagnostic_cannot_claim_acceptance_or_recovar_metrics():
    diagnostic = copy.deepcopy(NEGATIVE_DIAGNOSTIC)
    diagnostic["conclusion"]["accepted_result"] = True

    with pytest.raises(RegistryValidationError, match="False was expected"):
        validate_diagnostic(diagnostic, DIAGNOSTIC_SCHEMA)

    diagnostic = copy.deepcopy(NEGATIVE_DIAGNOSTIC)
    diagnostic["runs"][0]["replicates"][0]["performance"]["recovar_wall_s"] = 1

    with pytest.raises(RegistryValidationError, match="not of type 'null'"):
        validate_diagnostic(diagnostic, DIAGNOSTIC_SCHEMA)


def test_negative_diagnostic_collapse_flags_and_allocations_are_mechanical():
    diagnostic = copy.deepcopy(NEGATIVE_DIAGNOSTIC)
    diagnostic["runs"][0]["replicates"][0]["relion_collapse"]["zero_classes"] = [4]

    with pytest.raises(RegistryValidationError, match="zero_classes must match"):
        validate_diagnostic(diagnostic, DIAGNOSTIC_SCHEMA)

    diagnostic = copy.deepcopy(NEGATIVE_DIAGNOSTIC)
    diagnostic["runs"][0]["replicates"][0]["job"]["alloc_tres"] = (
        "billing=48,cpu=24,gres/gpu=2,mem=192G,node=1"
    )

    with pytest.raises(RegistryValidationError, match="ReqTRES and AllocTRES differ"):
        validate_diagnostic(diagnostic, DIAGNOSTIC_SCHEMA)

    diagnostic = copy.deepcopy(NEGATIVE_DIAGNOSTIC)
    diagnostic["runs"][0]["replicates"][0]["job"]["req_tres"] = (
        "billing=24,cpu=24,gres/gpu=10,mem=192G,node=1"
    )
    diagnostic["runs"][0]["replicates"][0]["job"]["alloc_tres"] = (
        "billing=24,cpu=24,gres/gpu=10,mem=192G,node=1"
    )

    with pytest.raises(RegistryValidationError, match="must request exactly one GPU"):
        validate_diagnostic(diagnostic, DIAGNOSTIC_SCHEMA)


@pytest.mark.parametrize(
    ("campaign", "symmetry"),
    [
        (C4_SYMMETRY_CAMPAIGN, "C4"),
        (D4_SYMMETRY_CAMPAIGN, "D4"),
        (O_SYMMETRY_CAMPAIGN, "O"),
        (I1_SYMMETRY_CAMPAIGN, "I1"),
    ],
)
def test_three_seed_symmetry_campaigns_retain_complete_trajectory_evidence(campaign, symmetry):
    assert [case["case_id"] for case in campaign["cases"]] == [41001, 41002, 41003]
    assert {case["configuration"]["symmetry"] for case in campaign["cases"]} == {symmetry}
    for case in campaign["cases"]:
        assert case["outcome"]["classification"] == "TRAJECTORY_EXACT"
        assert case["quality"]["trajectory"]["status"] == "PASS"
        assert case["quality"]["trajectory"]["evaluated_class_cells"] == 20
        assert case["quality"]["trajectory"]["passing_class_cells"] == 20
        for class_result in case["quality"]["final_classes"]:
            assert class_result["recovar_gt_resolution_0_143_angstrom"] > 0
            assert class_result["relion_gt_resolution_0_143_angstrom"] > 0
            assert (
                class_result["recovar_gt_resolution_0_143_angstrom"]
                == class_result["relion_gt_resolution_0_143_angstrom"]
            )


def test_campaign_per_class_resolution_requires_both_engines():
    campaign = copy.deepcopy(C4_SYMMETRY_CAMPAIGN)
    del campaign["cases"][0]["quality"]["final_classes"][0][
        "relion_gt_resolution_0_143_angstrom"
    ]

    with pytest.raises(
        RegistryValidationError,
        match="per-class 0.143 GT resolutions must be recorded for both engines",
    ):
        validate_campaign(campaign, CAMPAIGN_SCHEMA)


def test_completed_job_requires_identical_requested_and_allocated_tres():
    record = copy.deepcopy(CANDIDATE)
    record["execution"]["jobs"][0]["alloc_tres"] = "cpu=4,gres/gpu=2,mem=500G,node=1"

    with pytest.raises(RegistryValidationError, match="ReqTRES and AllocTRES differ"):
        validate_record(record, SCHEMA)


def test_science_equivalent_cannot_claim_trajectory_exact():
    record = copy.deepcopy(CANDIDATE)
    record["gate"]["trajectory_exact"] = True

    with pytest.raises(RegistryValidationError, match="requires trajectory_exact=false"):
        validate_record(record, SCHEMA)


def test_kclass_matching_must_be_a_permutation_with_k_rows():
    record = copy.deepcopy(CANDIDATE)
    record["quality"]["matching"]["recovar_to_relion"] = [1, 1, 3, 4]
    record["quality"]["final_classes"].pop()

    with pytest.raises(RegistryValidationError, match="permutation of 1..K"):
        validate_record(record, SCHEMA)


def test_missing_halfmap_values_require_a_reason():
    record = copy.deepcopy(CANDIDATE)
    record["quality"]["halfmap_fsc"]["missing_reason"] = None

    with pytest.raises(RegistryValidationError, match="half-map values require missing_reason"):
        validate_record(record, SCHEMA)


def test_cross_hardware_record_cannot_report_formal_speedup():
    record = copy.deepcopy(CANDIDATE)
    record["performance"]["comparison"]["formal_speedup"] = 1.4

    with pytest.raises(RegistryValidationError, match="cannot report formal_speedup"):
        validate_record(record, SCHEMA)


def test_class_collapse_flags_must_match_hard_populations():
    record = copy.deepcopy(CANDIDATE)
    record["quality"]["class_collapse"] = {
        "metric": "hard_assignment_fraction_below_threshold",
        "threshold": 0.25,
        "recovar_classes": [],
        "relion_classes": [],
        "status": "CLEAR",
    }

    with pytest.raises(RegistryValidationError, match="must match the hard-population threshold"):
        validate_record(record, SCHEMA)


def test_missing_performance_values_require_a_reason():
    record = copy.deepcopy(CANDIDATE)
    record["performance"]["relion"]["missing_reason"] = None

    with pytest.raises(RegistryValidationError, match="missing values require missing_reason"):
        validate_record(record, SCHEMA)


def test_campaign_signed_gt_delta_must_match_per_engine_values():
    campaign = copy.deepcopy(CAMPAIGN)
    campaign["cases"][0]["quality"]["final_classes"][0]["gt_fsc_auc_delta"] += 1e-6

    with pytest.raises(RegistryValidationError, match="signed GT FSC-AUC delta is inconsistent"):
        validate_campaign(campaign, CAMPAIGN_SCHEMA)


def test_campaign_occupancies_must_sum_and_flags_are_mechanical():
    campaign = copy.deepcopy(CAMPAIGN)
    case20 = next(case for case in campaign["cases"] if case["case_id"] == 20)
    case20["quality"]["occupancy"]["relion_counts"][0] -= 1
    case20["quality"]["occupancy"]["relion_flagged_classes"] = []

    with pytest.raises(RegistryValidationError, match="RELION occupancy must contain K counts"):
        validate_campaign(campaign, CAMPAIGN_SCHEMA)


def test_campaign_failure_modes_remain_distinct():
    campaign = copy.deepcopy(CAMPAIGN)
    case17 = next(case for case in campaign["cases"] if case["case_id"] == 17)
    case17["outcome"]["classification"] = "NEGATIVE_RELION_CLASS_COLLAPSE"
    validate_campaign(campaign, CAMPAIGN_SCHEMA)

    case17["outcome"]["classification"] = "RECOVAR_IMPLEMENTATION_FAILURE"
    case17["outcome"]["science_status"] = "UNRESOLVED"
    case17["quality"]["occupancy"] = {
        "source_iteration": None,
        "recovar_metric": "not_evaluated",
        "recovar_counts": None,
        "recovar_fractions": None,
        "relion_counts": None,
        "collapse_threshold_fraction": 0.01,
        "recovar_flagged_classes": [],
        "relion_flagged_classes": [],
        "status": "NOT_EVALUATED",
        "note": "Run stopped before either engine produced comparable occupancy.",
    }
    validate_campaign(campaign, CAMPAIGN_SCHEMA)

    case17["quality"]["occupancy"]["status"] = "ZERO_CLASS"
    with pytest.raises(RegistryValidationError, match="unevaluated occupancy requires"):
        validate_campaign(campaign, CAMPAIGN_SCHEMA)


def test_differing_particle_hashes_cannot_support_execution_invariance():
    campaign = copy.deepcopy(CAMPAIGN)
    case25 = next(case for case in campaign["cases"] if case["case_id"] == 25)
    case25["input_equivalence"]["admissible_for_execution_invariance"] = True

    with pytest.raises(RegistryValidationError, match="differing particle hashes"):
        validate_campaign(campaign, CAMPAIGN_SCHEMA)


def test_campaign_external_paths_must_be_absolute():
    campaign = copy.deepcopy(CAMPAIGN)
    campaign["cases"][0]["inputs"][0]["path"] = "relative/input.mrcs"

    with pytest.raises(RegistryValidationError, match="does not match"):
        validate_campaign(campaign, CAMPAIGN_SCHEMA)


def test_campaign_external_verification_fails_closed(tmp_path, monkeypatch):
    existing = tmp_path / "evidence.json"
    existing.write_text("sealed evidence\n")
    digest = hashlib.sha256(existing.read_bytes()).hexdigest()
    reference = {
        "path": str(existing),
        "sha256": digest,
        "size_bytes": existing.stat().st_size,
    }
    monkeypatch.setattr(registry_validator, "_campaign_file_references", lambda _: [reference])
    verify_campaign_files(CAMPAIGN, {})

    reference["size_bytes"] += 1
    with pytest.raises(RegistryValidationError, match="size mismatch"):
        verify_campaign_files(CAMPAIGN, {})

    reference["size_bytes"] = existing.stat().st_size
    reference["sha256"] = "0" * 64
    with pytest.raises(RegistryValidationError, match="checksum mismatch"):
        verify_campaign_files(CAMPAIGN, {})

    reference["path"] = str(tmp_path / "missing.json")
    with pytest.raises(RegistryValidationError, match="missing file"):
        verify_campaign_files(CAMPAIGN, {})


def test_diagnostic_external_verification_fails_closed(tmp_path, monkeypatch):
    existing = tmp_path / "negative-evidence.json"
    existing.write_text("sealed negative evidence\n")
    digest = hashlib.sha256(existing.read_bytes()).hexdigest()
    reference = {
        "path": str(existing),
        "sha256": digest,
        "size_bytes": existing.stat().st_size,
    }
    monkeypatch.setattr(registry_validator, "_diagnostic_file_references", lambda _: [reference])
    verify_diagnostic_files(NEGATIVE_DIAGNOSTIC, {})

    reference["size_bytes"] += 1
    with pytest.raises(RegistryValidationError, match="size mismatch"):
        verify_diagnostic_files(NEGATIVE_DIAGNOSTIC, {})

    reference["size_bytes"] = existing.stat().st_size
    reference["sha256"] = "0" * 64
    with pytest.raises(RegistryValidationError, match="checksum mismatch"):
        verify_diagnostic_files(NEGATIVE_DIAGNOSTIC, {})

    reference["path"] = str(tmp_path / "missing.json")
    with pytest.raises(RegistryValidationError, match="missing file"):
        verify_diagnostic_files(NEGATIVE_DIAGNOSTIC, {})
