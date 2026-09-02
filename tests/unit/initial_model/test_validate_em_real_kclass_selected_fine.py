"""Guards for the checked historical K=4 selected-fine diagnostic."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.validate_em_real_kclass_selected_fine import (
    SelectedFineEvidenceValidationError,
    default_ledger_path,
    validate_evidence,
)

pytestmark = pytest.mark.unit


def _ledger():
    return json.loads(default_ledger_path().read_text())


def test_checked_selected_fine_evidence_is_valid():
    validate_evidence(_ledger())


def test_selected_fine_evidence_cannot_claim_benchmark_admission():
    data = copy.deepcopy(_ledger())
    data["admission_status"] = "QUALIFIED"

    with pytest.raises(SelectedFineEvidenceValidationError, match="cannot claim benchmark admission"):
        validate_evidence(data)


def test_selected_fine_evidence_retains_audited_hash_correction():
    data = copy.deepcopy(_ledger())
    data["provenance_correction"]["previous_value"] = data["provenance_correction"]["corrected_value"]

    with pytest.raises(SelectedFineEvidenceValidationError, match="audited coarse-report hash correction"):
        validate_evidence(data)


def test_selected_fine_evidence_forbids_nearest_row_join():
    data = copy.deepcopy(_ledger())
    data["scope"]["unmatched_policy"] = "map unmatched rotations to nearest row"

    with pytest.raises(SelectedFineEvidenceValidationError, match="forbid nearest-row joins"):
        validate_evidence(data)


def test_selected_fine_evidence_requires_exact_one_gpu_allocation():
    data = copy.deepcopy(_ledger())
    data["capture"]["live_scontrol_alloc_tres"] = "billing=16,cpu=8,gres/gpu=2,mem=64G,node=1"

    with pytest.raises(SelectedFineEvidenceValidationError, match="ReqTRES and AllocTRES differ"):
        validate_evidence(data)


def test_selected_fine_evidence_cannot_invent_missing_performance():
    data = copy.deepcopy(_ledger())
    data["capture"]["peak_hbm_mib"] = 1234

    with pytest.raises(SelectedFineEvidenceValidationError, match="unavailable HBM or RSS"):
        validate_evidence(data)


def test_selected_fine_evidence_retains_support_as_first_boundary():
    data = copy.deepcopy(_ledger())
    data["particles"][0]["first_observed_nonidentical_boundary"] = "raw_score"

    with pytest.raises(SelectedFineEvidenceValidationError, match="first observed boundary"):
        validate_evidence(data)


def test_selected_fine_evidence_winner_must_agree_with_masses():
    data = copy.deepcopy(_ledger())
    data["particles"][0]["recovar_class_winner_zero_based"] = 2

    with pytest.raises(SelectedFineEvidenceValidationError, match="winner disagrees with its masses"):
        validate_evidence(data)


def test_selected_fine_evidence_retains_one_matching_and_one_mismatching_winner():
    data = copy.deepcopy(_ledger())
    particle = data["particles"][1]
    particle["recovar_captured_class_probability_mass"] = particle["relion_class_probability_mass"]
    particle["recovar_class_winner_zero_based"] = particle["relion_class_winner_zero_based"]
    particle["classification"] = "fine_class_winner_exact_with_support_and_prior_differences"

    with pytest.raises(SelectedFineEvidenceValidationError, match="one matching and one mismatching"):
        validate_evidence(data)


def test_selected_fine_evidence_retains_invalid_output_exclusion():
    data = copy.deepcopy(_ledger())
    data["excluded_outputs"]["reason"] = "old output"

    with pytest.raises(SelectedFineEvidenceValidationError, match="explicitly excluded"):
        validate_evidence(data)


def test_selected_fine_evidence_retains_strict_claim_boundary():
    data = copy.deepcopy(_ledger())
    data["strict_claim_boundary"] = "This establishes production quality."

    with pytest.raises(SelectedFineEvidenceValidationError, match="strict claim boundary"):
        validate_evidence(data)
