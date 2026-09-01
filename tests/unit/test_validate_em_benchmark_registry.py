"""Unit tests for the checked-in EM benchmark evidence registry."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.validate_em_benchmark_registry import (
    RegistryValidationError,
    validate_record,
    validate_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = REPO_ROOT / "docs" / "benchmarks" / "em"
SCHEMA = json.loads((REGISTRY_ROOT / "schema_v1.json").read_text())
CANDIDATE = json.loads(
    (
        REGISTRY_ROOT
        / "entries"
        / "k4-ribosembly-100k256-1b9209cd8-h100.json"
    ).read_text()
)


def test_checked_in_em_benchmark_registry_is_valid():
    paths = validate_registry(REGISTRY_ROOT)
    assert [path.stem for path in paths] == [
        "k4-ribosembly-100k256-1b9209cd8-h100",
        "k4-ribosembly-100k256-ac5177d2-a100",
    ]


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
