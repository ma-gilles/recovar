"""Fail-closed tests for the rejected K=4 compact-pair threshold campaign."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.validate_em_real_kclass_compact_pair_threshold_campaign import (
    EXPECTED_PHYSICAL_ROTATION_DEG,
    CampaignValidationError,
    default_ledger_path,
    validate_campaign,
)

LEDGER_PATH = default_ledger_path()
LEDGER = json.loads(LEDGER_PATH.read_text())


def _mutate(path: tuple[str | int, ...], value: object) -> dict:
    mutated = copy.deepcopy(LEDGER)
    target = mutated
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    return mutated


def test_checked_in_campaign_is_valid_and_remains_rejected():
    validate_campaign(LEDGER)
    assert LEDGER["schema"] == "recovar.em.real_kclass_compact_pair_threshold_campaign.v1"
    assert LEDGER["accepted_result"] is False
    assert LEDGER["aggregate"]["formal"] == {
        "accept_pair_count": 0,
        "reject_pair_count": 3,
        "decision": "reject",
    }
    assert LEDGER["aggregate"]["scientific_equivalence"] == {
        "accept_pair_count": 2,
        "reject_pair_count": 1,
        "decision": "reject",
    }


def test_default_path_is_the_checked_diagnostic():
    assert LEDGER_PATH == (
        Path(__file__).resolve().parents[2] / "docs/benchmarks/em/diagnostics/"
        "real-k4-10345-native10k-compact-pair-threshold128-default512-"
        "campaign-2f6759608-20260903.json"
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("admission_status",), "ACCEPTED", "admission_status"),
        (("accepted_result",), True, "accepted_result"),
        (("status",), "complete_accepted", "status"),
        (("aggregate", "formal", "decision"), "accept", "aggregate.formal.decision"),
        (
            ("aggregate", "scientific_equivalence", "decision"),
            "accept",
            "aggregate.scientific_equivalence.decision",
        ),
        (
            ("pairs", 1, "scientific_equivalence_decision"),
            "accept",
            "scientific_equivalence_decision",
        ),
        (("pairs", 0, "formal_decision"), "accept", "formal_decision"),
    ],
    ids=[
        "admission",
        "accepted-result",
        "status",
        "formal-aggregate",
        "science-aggregate",
        "seed42002-science",
        "pair-formal",
    ],
)
def test_rejection_cannot_be_upgraded(path, value, message):
    with pytest.raises(CampaignValidationError, match=message):
        validate_campaign(_mutate(path, value))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("decision_contract", "aggregation_rule"),
            "mean_pair_metrics",
            "aggregation rule",
        ),
        (("aggregate", "aggregation_rule"), "majority_vote", "aggregate rule"),
        (("aggregate", "pair_count"), 2, "pair_count"),
        (("aggregate", "formal", "accept_pair_count"), 3, "accept_pair_count"),
        (
            ("aggregate", "scientific_equivalence", "reject_pair_count"),
            0,
            "reject_pair_count",
        ),
    ],
    ids=["contract-average", "aggregate-average", "pair-count", "formal-count", "science-count"],
)
def test_all_pair_conjunction_cannot_be_replaced_by_averaging(path, value, message):
    with pytest.raises(CampaignValidationError, match=message):
        validate_campaign(_mutate(path, value))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("pairs", 0, "scientific_equivalence_contract_timing"), "prospective"),
        (("pairs", 1, "scientific_equivalence_contract_timing"), "retrospective"),
        (("pairs", 2, "scientific_equivalence_contract_timing"), "retrospective"),
        (("pairs", 0, "formal_contract_timing"), "retrospective"),
    ],
    ids=["retrospective-laundering", "prospective-seed2", "prospective-seed3", "formal"],
)
def test_prospective_and_retrospective_contracts_cannot_be_conflated(path, value):
    with pytest.raises(CampaignValidationError, match="contract_timing"):
        validate_campaign(_mutate(path, value))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("treatment", "production_default_min_bucket_size"), 128, "production default"),
        (("treatment", "production_default_changed"), True, "production_default_changed"),
        (("treatment", "disposition"), "promote_threshold128", "disposition"),
        (("treatment", "baseline", "resolved_min_bucket_size"), 128, "treatment.baseline"),
        (("aggregate", "production_disposition"), "promote_threshold128", "aggregate disposition"),
    ],
    ids=["default", "changed", "treatment-disposition", "baseline", "aggregate-disposition"],
)
def test_production_default_512_cannot_be_changed(path, value, message):
    with pytest.raises(CampaignValidationError, match=message):
        validate_campaign(_mutate(path, value))


def test_science_decision_is_recomputed_instead_of_trusting_stored_acceptance():
    mutated = copy.deepcopy(LEDGER)
    mutated["pairs"][1]["summary"]["poses_exact"] = True
    mutated["pairs"][1]["scientific_equivalence_decision"] = "accept"
    with pytest.raises(CampaignValidationError, match="sealed scientific-equivalence decision"):
        validate_campaign(mutated)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("physical_pose_discriminator", "metric_provenance", "aggregate_field_present"),
            True,
            "aggregate provenance",
        ),
        (
            ("physical_pose_discriminator", "metric_provenance", "derivation_status"),
            "aggregate_reported",
            "derivation status",
        ),
        (
            ("physical_pose_discriminator", "physical_relative_rotation_deg"),
            0.0,
            "physical relative rotation",
        ),
        (
            ("physical_pose_discriminator", "differing_pose_row_count"),
            0,
            "differing_pose_row_count",
        ),
        (
            ("physical_pose_discriminator", "baseline_euler_deg"),
            [0.0, 0.0, 0.0],
            "baseline_euler_deg",
        ),
        (
            ("physical_pose_discriminator", "genuinely_distinct_physical_pose"),
            False,
            "genuinely_distinct_physical_pose",
        ),
    ],
    ids=["aggregate-source", "derived-status", "angle", "row-count", "raw-row", "distinct"],
)
def test_checked_physical_pose_discriminator_cannot_be_erased(path, value, message):
    with pytest.raises(CampaignValidationError, match=message):
        validate_campaign(_mutate(path, value))


def test_physical_rotation_is_checked_from_the_immutable_raw_euler_row():
    pose = LEDGER["physical_pose_discriminator"]
    assert pose["metric_provenance"]["aggregate_field_present"] is False
    assert pose["metric_provenance"]["pair_report_field_present"] is False
    assert pose["metric_provenance"]["raw_euler_row_is_immutable_source"] is True
    assert pose["metric_provenance"]["derivation_status"] == "checked_derived_diagnostic"
    assert pose["physical_relative_rotation_deg"] == pytest.approx(EXPECTED_PHYSICAL_ROTATION_DEG, abs=1e-12)


def test_wrong_dispatch_attempt_cannot_be_admitted_or_dropped():
    admitted = _mutate(("failed_first_attempt", "accepted_evidence"), True)
    with pytest.raises(CampaignValidationError, match="accepted_evidence"):
        validate_campaign(admitted)

    dropped = copy.deepcopy(LEDGER)
    dropped["failed_first_attempt"]["jobs"].pop()
    with pytest.raises(CampaignValidationError, match="four rows"):
        validate_campaign(dropped)


def test_exact_slurm_resources_and_source_hashes_are_fail_closed():
    resources = _mutate(
        ("pairs", 1, "jobs", "candidate", "alloc_tres"),
        "cpu=8,mem=128G,node=1,billing=10,gres/gpu=2",
    )
    with pytest.raises(CampaignValidationError, match="alloc_tres"):
        validate_campaign(resources)

    source_hash = _mutate(("sealed_evidence", "aggregate", "sha256"), "0" * 64)
    with pytest.raises(CampaignValidationError, match="aggregate.sha256"):
        validate_campaign(source_hash)


def test_canonical_seal_rejects_consistent_unreviewed_metric_rewrite():
    mutated = copy.deepcopy(LEDGER)
    performance = mutated["pairs"][0]["performance"]
    performance["baseline_external_wall_s"] *= 2
    performance["external_wall_ratio_descriptive"] = (
        performance["candidate_external_wall_s"] / performance["baseline_external_wall_s"]
    )
    with pytest.raises(CampaignValidationError, match="canonical campaign SHA-256"):
        validate_campaign(mutated)


def test_unknown_fields_are_rejected():
    mutated = copy.deepcopy(LEDGER)
    mutated["averaged_acceptance"] = True
    with pytest.raises(CampaignValidationError, match="extra=.*averaged_acceptance"):
        validate_campaign(mutated)
