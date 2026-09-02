"""Focused contracts for the fixed-capacity local score correctness gate."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

from recovar.em.dense_single_volume import local_big_jit
from scripts import run_fixed_capacity_local_score_gate as score_gate

pytestmark = pytest.mark.unit


def _captured_call() -> score_gate._CapturedCall:
    candidate_mask = np.asarray([[[True, True], [False, False]]], dtype=bool)
    scores = np.asarray([[[1.25, -0.5], [-np.inf, -np.inf]]], dtype=np.float32)
    probs = np.asarray([[[0.85, 0.15], [0.0, 0.0]]], dtype=np.float32)
    diagnostics = {
        "log_z": np.asarray([1.75], dtype=np.float32),
        "best_log_score": np.asarray([1.25], dtype=np.float32),
        "best_argmax": np.asarray([0], dtype=np.int32),
        "max_posterior": np.asarray([0.85], dtype=np.float32),
        "probs_sum_t": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "reconstruction_probs_sum_t": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "n_significant_samples": np.asarray([2], dtype=np.int32),
        "reconstruction_sample_mask": candidate_mask.copy(),
        "reconstruction_rotation_mask": np.asarray([[True, False]], dtype=bool),
        "reconstruction_row_count": np.asarray(1, dtype=np.int32),
        "debug_scores": scores,
        "debug_probs": probs,
        "candidate_mask": candidate_mask,
        "finite_score_mask": np.isfinite(scores),
        "posterior_support": probs > 0,
    }
    return score_gate._CapturedCall(
        diagnostics=diagnostics,
        inputs={
            "Ft_y": np.zeros(1, dtype=np.complex64),
            "Ft_ctf": np.zeros(1, dtype=np.float32),
            "sample_mask": None,
        },
        static_arguments={"score_only": True},
        donated_input_object_ids={"Ft_y": 1, "Ft_ctf": 2},
        prepared_call=None,
        initial_carry=(),
        replay_static_arguments={},
    )


def test_future_float64_companion_envelopes_are_at_least_three_orders_tighter():
    score_gate.validate_future_envelope_pair()
    f32 = score_gate.FUTURE_SPEED_QUALIFIED_ENVELOPES["float32"]
    f64 = score_gate.FUTURE_SPEED_QUALIFIED_ENVELOPES["float64"]
    assert f32.keys() == f64.keys()
    assert all(f32[name] / f64[name] >= 1_000 for name in f32)


def test_current_shared_seam_rejects_any_continuous_drift():
    expected = _captured_call()
    diagnostics = dict(expected.diagnostics)
    changed_scores = diagnostics["debug_scores"].copy()
    changed_scores[0, 0, 0] = np.nextafter(changed_scores[0, 0, 0], np.float32(np.inf))
    diagnostics["debug_scores"] = changed_scores
    actual = replace(expected, diagnostics=diagnostics)

    with pytest.raises(AssertionError, match="not bitwise equal"):
        score_gate.compare_captured_calls(
            actual,
            expected,
            label="one-ulp-drift",
            require_exact_current_seam=True,
        )


def test_current_shared_seam_rejects_discrete_decision_or_support_drift():
    expected = _captured_call()
    diagnostics = dict(expected.diagnostics)
    diagnostics["best_argmax"] = np.asarray([1], dtype=np.int32)
    actual = replace(expected, diagnostics=diagnostics)

    with pytest.raises(AssertionError, match="best_argmax"):
        score_gate.compare_captured_calls(
            actual,
            expected,
            label="argmax-drift",
            require_exact_current_seam=True,
        )


def test_fixture_has_two_authoritative_calls_with_a_default_off_selector():
    fixture = score_gate.build_gate_fixture()
    signature = inspect.signature(score_gate.local_em_engine.run_local_em_exact)

    np.testing.assert_array_equal(fixture.bucket_image_order, [1, 0, 2])
    assert fixture.bucket_image_capacity == 2
    assert fixture.bucket_radix == 16
    assert fixture.fixed_bundle.plan.valid_call_count == 2
    assert fixture.fixed_bundle.plan.physical_call_capacity == 3
    assert fixture.fixed_bundle.plan.valid_image_count == 3
    assert fixture.fixed_bundle.plan.valid_row_count == 8
    assert signature.parameters["_fixed_capacity_enabled"].default is False


def test_gate_donation_contract_tracks_signature_positions_and_fresh_objects():
    parameter_names = tuple(inspect.signature(local_big_jit.run_local_bucket_big_jit).parameters)

    assert parameter_names[7:9] == score_gate.CURRENT_DONATED_POSITIONAL_NAMES
    assert score_gate.CURRENT_DONATED_POSITIONAL_NAMES == ("Ft_y", "Ft_ctf")


def test_slurm_runner_is_fail_closed_and_forbids_speed_or_default_claims():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "scripts/run_fixed_capacity_local_score_gate.sbatch").read_text()

    required_fragments = (
        ': "${EXPECTED_REPO_HEAD:',
        ': "${EXPECTED_SOURCE_MANIFEST_SHA256:',
        ': "${EXPECTED_NODE_NAME:',
        ': "${TARGET_GPU_UUID:',
        ': "${EXPECTED_FOCUSED_TEST_COUNT:',
        'test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"',
        'vdam_select_target_gpu "${TARGET_GPU_UUID}" 0',
        'vdam_verify_selected_gpu "${selected_gpu_uuid}"',
        '--expected-gpu-uuid "${selected_gpu_uuid}"',
        "--run-mechanism-microbenchmark",
        'assert payload["classification"] == "correctness_only"',
        'assert payload["speed_claim_allowed"] is False',
        'assert payload["default_promotion_allowed"] is False',
        'cmp "${PROVENANCE}/source_manifest.sha256"',
        'touch "${ROOT}/COMPLETED"',
    )
    for fragment in required_fragments:
        assert fragment in source
    assert source.rindex('touch "${ROOT}/COMPLETED"') > source.rindex("vdam_verify_selected_gpu")


@pytest.mark.gpu
def test_fixed_capacity_local_score_gate_is_exact_on_gpu(tmp_path):
    devices = jax.devices("gpu")
    assert len(devices) == 1
    payload = score_gate.run_gate(
        output_dir=tmp_path / "gate",
        repeat_count=2,
        git_head="focused-test",
        gpu_uuid="focused-test-gpu",
    )

    assert payload["schema"] == score_gate.SCHEMA
    assert payload["classification"] == "correctness_only"
    assert payload["passed"] is True
    assert payload["current_seam_exact"] is True
    assert payload["speed_claim_allowed"] is False
    assert payload["default_promotion_allowed"] is False
    assert payload["production_whole_boundary_enabled"] is True
    assert payload["mechanism_microbenchmark"] is None
    assert all(item["passed"] for item in payload["comparisons"])
    assert all(item["passed"] for item in payload["production_comparisons"])
    assert any(
        "production-mature-vs-fixed-one-boundary" in item["label"]
        for item in payload["production_comparisons"]
    )
    assert all(item["passed"] for item in payload["whole_boundary_comparisons"])
    assert all(
        item["one_compiled_boundary"] and item["call_count"] == 2
        for item in payload["whole_boundary_comparisons"]
    )
    assert all(
        value == 0.0
        for comparison in payload["comparisons"]
        for value in comparison["metrics"].values()
    )
