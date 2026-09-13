from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_vdam_hybrid_fused_fallback_execution import (
    ARM_ORDER,
    FUSED_TARGET,
    FUSED_WRAPPER,
    FallbackExecutionAuditError,
    audit_fused_fallback_root,
    validate_fused_fallback_observation,
)

pytestmark = pytest.mark.unit


def _meta(*, stable: bool, rectangular_batches: int = 0) -> dict:
    return {
        "halfset_0_profile_summary": {
            "coarse_gaussian_gemm_hybrid": {
                "selected_block_capacity": 1,
                "all_full_dense_batches_used_fused": rectangular_batches == 0,
                "whole_batch_fail_closed_fallback": True,
                "full_fallback_backend_requested": "fused_projector",
                "full_fallback_backend_armed": "fused_projector",
                "full_fallback_backend_effective": (
                    "fused_projector" if rectangular_batches == 0 else "rectangular"
                ),
                "full_dense_batch_count": 2,
                "full_dense_image_count": 350,
                "fused_full_fallback_batch_count": 2 - rectangular_batches,
                "fused_full_fallback_image_count": 350 if rectangular_batches == 0 else 200,
                "rectangular_full_fallback_batch_count": rectangular_batches,
                "rectangular_full_fallback_image_count": 0 if rectangular_batches == 0 else 150,
                "selected_rescore_batch_count": 0,
                "fallback_reasons": {"block_capacity_overflow": 1},
                "overflow_latch_activation_count": 1,
                "overflow_latch_active_at_return": True,
                "coarse_square_layout": {
                    "stable_fourier_window_shapes_requested": stable,
                    "logical_square_pixels": 364,
                    "physical_square_pixels": 2112 if stable else 364,
                    "executed_square_pixels": 364,
                    "logical_issue_stream_is_prefix": True,
                    "physical_tail_skipped_by_runtime_count": stable,
                },
            },
            "coarse_selector_audit": {
                "requested_fused": True,
                "effective_fused": rectangular_batches == 0,
                "target": FUSED_TARGET if rectangular_batches == 0 else "rectangular",
                "wrapper": FUSED_WRAPPER if rectangular_batches == 0 else "rectangular",
                "counts": {
                    "actual_rows": 350,
                    "fused_calls": 2 - rectangular_batches,
                    "multistream_calls": 0,
                    "prehalf_selected_calls": 0,
                    "native_atomic_selected_calls": 0,
                },
            },
        },
    }


def test_observation_accepts_only_actual_mature_fused_fallback():
    result = validate_fused_fallback_observation(
        _meta(stable=True),
        arm="stable_on_1",
        iteration=2,
        expected_capacity=1,
    )

    assert result["full_dense_batch_count"] == 2
    assert result["fused_full_fallback_batch_count"] == 2
    assert result["rectangular_full_fallback_batch_count"] == 0
    assert result["logical_square_pixels"] == 364
    assert result["physical_square_pixels"] == 2112


def test_observation_rejects_any_rectangular_fallback():
    with pytest.raises(FallbackExecutionAuditError, match="did not route every full batch"):
        validate_fused_fallback_observation(
            _meta(stable=False, rectangular_batches=1),
            arm="stable_off_1",
            iteration=1,
            expected_capacity=1,
        )


def test_observation_rejects_spoofed_selector_target():
    meta = _meta(stable=False)
    meta["halfset_0_profile_summary"]["coarse_selector_audit"]["target"] = "wrong"

    with pytest.raises(FallbackExecutionAuditError, match="selector target differs"):
        validate_fused_fallback_observation(
            meta,
            arm="stable_off_2",
            iteration=1,
            expected_capacity=1,
        )


def test_observation_rejects_noncapacity_fallback_reason():
    meta = _meta(stable=False)
    hybrid = meta["halfset_0_profile_summary"]["coarse_gaussian_gemm_hybrid"]
    hybrid["fallback_reasons"] = {"invalid_selected_exact_output": 1}

    with pytest.raises(FallbackExecutionAuditError, match="not solely capacity overflow"):
        validate_fused_fallback_observation(
            meta,
            arm="stable_off_1",
            iteration=1,
            expected_capacity=1,
        )


def test_control_observation_rejects_missing_stable_request_field():
    meta = _meta(stable=False)
    layout = meta["halfset_0_profile_summary"]["coarse_gaussian_gemm_hybrid"][
        "coarse_square_layout"
    ]
    del layout["stable_fourier_window_shapes_requested"]

    with pytest.raises(FallbackExecutionAuditError, match="omitted its stable-shape request"):
        validate_fused_fallback_observation(
            meta,
            arm="stable_off_1",
            iteration=1,
            expected_capacity=1,
        )


def _write_root(root: Path) -> None:
    provenance = root / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "repo_head.txt").write_text("a" * 40 + "\n")
    (provenance / "coarse_mode.txt").write_text("hybrid_fused_fallback\n")
    (provenance / "hybrid_block_capacity.txt").write_text("1\n")
    for arm in ARM_ORDER:
        run = root / "runs" / arm
        output = run / "output"
        output.mkdir(parents=True)
        (run / "SCIENCE_COMPLETED").write_text("complete\n")
        (run / "command.json").write_text(json.dumps({"hybrid_block_capacity": 1}) + "\n")
        (output / "run_it001_recovar_meta.json").write_text(
            json.dumps(_meta(stable=arm.startswith("stable_on_"))) + "\n"
        )


def test_root_audit_seals_all_four_arms(tmp_path: Path):
    _write_root(tmp_path)

    report = audit_fused_fallback_root(
        tmp_path,
        iterations=(1,),
        expected_capacity=1,
        expected_head="a" * 40,
    )

    assert report["result"] == "pass"
    assert report["observation_count"] == 4
    assert {row["arm"] for row in report["observations"]} == set(ARM_ORDER)


def test_root_audit_rejects_command_capacity_mismatch(tmp_path: Path):
    _write_root(tmp_path)
    command = tmp_path / "runs/stable_on_2/command.json"
    command.write_text(json.dumps({"hybrid_block_capacity": 64}) + "\n")

    with pytest.raises(FallbackExecutionAuditError, match="command fallback capacity differs"):
        audit_fused_fallback_root(tmp_path, iterations=(1,), expected_capacity=1)
