"""Guards for the checked rejected real K-class InitialModel ledger."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.validate_em_real_kclass_diagnostics import (
    DiagnosticsValidationError,
    default_ledger_path,
    validate_diagnostics,
)

pytestmark = pytest.mark.unit


def _ledger():
    return json.loads(default_ledger_path().read_text())


def test_checked_rejected_real_kclass_diagnostics_are_valid():
    validate_diagnostics(_ledger())


def test_rejected_diagnostic_cannot_claim_benchmark_admission():
    data = copy.deepcopy(_ledger())
    data["runs"][0]["admission_status"] = "QUALIFIED"

    with pytest.raises(DiagnosticsValidationError, match="cannot claim benchmark admission"):
        validate_diagnostics(data)


def test_rejected_diagnostic_requires_exact_matching_tres():
    data = copy.deepcopy(_ledger())
    data["runs"][0]["execution"]["pair_job"]["alloc_tres"] = (
        "cpu=8,gres/gpu=2,mem=192G,node=1"
    )

    with pytest.raises(DiagnosticsValidationError, match="ReqTRES and AllocTRES differ"):
        validate_diagnostics(data)


def test_rejected_diagnostic_retains_failed_science_result():
    data = copy.deepcopy(_ledger())
    data["runs"][0]["quality"]["audit_result"] = "pass"

    with pytest.raises(DiagnosticsValidationError, match="audit_result=fail"):
        validate_diagnostics(data)


def test_rejected_diagnostic_cannot_claim_gold_standard_halfmaps():
    data = copy.deepcopy(_ledger())
    data["runs"][0]["scope"]["gold_standard_halfmaps_available"] = True

    with pytest.raises(DiagnosticsValidationError, match="cannot claim half maps"):
        validate_diagnostics(data)


def test_rejected_diagnostic_keeps_all_four_class_counts():
    data = copy.deepcopy(_ledger())
    data["runs"][0]["quality"]["final_candidate_counts"][0] -= 1

    with pytest.raises(DiagnosticsValidationError, match="sum to all particles"):
        validate_diagnostics(data)


def test_causal_diagnostic_cannot_claim_benchmark_admission():
    data = copy.deepcopy(_ledger())
    data["causal_diagnostics"][0]["admission_status"] = "QUALIFIED"

    with pytest.raises(DiagnosticsValidationError, match="cannot claim benchmark admission"):
        validate_diagnostics(data)


def test_invalid_map_comparison_must_retain_coverage_mismatch():
    data = copy.deepcopy(_ledger())
    data["causal_diagnostics"][0]["coverage"]["same_visited_particle_ids"] = True

    with pytest.raises(DiagnosticsValidationError, match="retain the coverage mismatch"):
        validate_diagnostics(data)


def test_valid_map_comparison_requires_identical_visited_ids():
    data = copy.deepcopy(_ledger())
    data["causal_diagnostics"][1]["coverage"]["same_visited_particle_ids"] = False

    with pytest.raises(DiagnosticsValidationError, match="requires identical visited particle IDs"):
        validate_diagnostics(data)
