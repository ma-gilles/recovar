"""Guards for the checked K=4 offset-prior real-data full-pair evidence."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.validate_em_real_kclass_offset_prior_fullpairs import (
    FullPairEvidenceValidationError,
    default_ledger_path,
    validate_evidence,
)

pytestmark = pytest.mark.unit


def _ledger():
    return json.loads(default_ledger_path().read_text())


def test_checked_k4_offset_prior_fullpairs_are_valid():
    validate_evidence(_ledger())


def test_fullpair_evidence_requires_exact_iteration_1_raw_labels():
    data = copy.deepcopy(_ledger())
    data["cases"][0]["quality"]["iteration_1"]["raw_class_label_exact_count"] = 199

    with pytest.raises(FullPairEvidenceValidationError, match="exact raw iteration-1 labels"):
        validate_evidence(data)


def test_fullpair_evidence_separates_raw_and_map_hungarian_labels():
    data = copy.deepcopy(_ledger())
    case = next(case for case in data["cases"] if case["dataset"] == "EMPIAR-10345")
    case["quality"]["iteration_1"]["assignment_accuracy_after_map_hungarian_permutation"] = 1.0

    with pytest.raises(FullPairEvidenceValidationError, match="raw-label agreement from map-Hungarian"):
        validate_evidence(data)


def test_fullpair_evidence_requires_exact_slurm_allocation():
    data = copy.deepcopy(_ledger())
    data["cases"][0]["slurm"]["alloc_tres"] = "billing=15,cpu=8,gres/gpu=2,mem=192G,node=1"

    with pytest.raises(FullPairEvidenceValidationError, match="ReqTRES and AllocTRES differ"):
        validate_evidence(data)


def test_rejected_fullpair_cannot_admit_formal_runtime_ratio():
    data = copy.deepcopy(_ledger())
    data["cases"][0]["performance"]["formal_ratio_admitted"] = True

    with pytest.raises(FullPairEvidenceValidationError, match="cannot admit a formal performance ratio"):
        validate_evidence(data)


def test_initialmodel_fullpair_cannot_claim_halfmaps():
    data = copy.deepcopy(_ledger())
    data["scientific_scope"]["gold_standard_halfmaps_available"] = True

    with pytest.raises(FullPairEvidenceValidationError, match="cannot claim half maps"):
        validate_evidence(data)
