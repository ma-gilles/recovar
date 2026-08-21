"""Tests for the bounded K=1 coarse operand report envelope."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts import analyze_k1_coarse_operand_boundary_v3 as analyzer


def _run_report(
    tmp_path: Path,
    monkeypatch,
    *,
    physical_iteration: int,
    validations: dict[str, bool],
    plural_validation_names: bool = False,
) -> tuple[dict, list[int]]:
    native_directory = tmp_path / "capture"
    recovar_directory = tmp_path / "coarse"
    analysis_directory = tmp_path / "analysis"
    native_directory.mkdir()
    recovar_directory.mkdir()
    analysis_directory.mkdir()

    stack_index = 79
    original_index = 78
    (native_directory / f"part2767_stack{stack_index}.p1-v2.bin").touch()
    (native_directory / f"part2767_stack{stack_index}.p1-op-v2.bin").touch()
    (native_directory / f"part2767_stack{stack_index}.p1-lane-v1.bin").touch()
    (native_directory / f"part2767_stack{stack_index}.p1-live-v1.bin").touch()
    (recovar_directory / f"significance_orig{original_index:06d}_it003_cs080.npz").touch()
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "case_id": 22,
                "physical_iteration": physical_iteration,
                "targets": [
                    {
                        "stack_index_one_based": stack_index,
                        "original_index_zero_based": original_index,
                    }
                ],
            }
        )
    )
    for label, ready in validations.items():
        filename = {
            "components": "components_validation.json",
            "operands": (
                "operands_validation.json"
                if plural_validation_names
                else "operand_validation.json"
            ),
            "lanes": (
                "lanes_validation.json"
                if plural_validation_names
                else "lane_validation.json"
            ),
        }[label]
        (analysis_directory / filename).write_text(
            json.dumps(
                {
                    "status": "accepted" if ready else "rejected",
                    "classification_ready": ready,
                }
            )
        )

    observed_boundaries: list[tuple[int, int]] = []

    def fake_compare(
        *_args,
        physical_iteration: int,
        physical_image_size: int,
        translation_pair_recovar: tuple[int, int] | None,
    ):
        assert translation_pair_recovar is None
        observed_boundaries.append((physical_iteration, physical_image_size))
        return {"stack_index_one_based": stack_index}

    output_path = tmp_path / "report.json"
    monkeypatch.setattr(analyzer, "_compare", fake_compare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_k1_coarse_operand_boundary_v3.py",
            "--native-directory",
            str(native_directory),
            "--recovar-directory",
            str(recovar_directory),
            "--selection-json",
            str(selection_path),
            "--physical-image-size",
            "256",
            "--output-json",
            str(output_path),
        ],
    )
    analyzer.main()
    return json.loads(output_path.read_text()), observed_boundaries


def test_report_uses_selection_physical_iteration_and_validation_gates(
    tmp_path: Path,
    monkeypatch,
):
    report, observed_boundaries = _run_report(
        tmp_path,
        monkeypatch,
        physical_iteration=3,
        validations={"components": False, "operands": True, "lanes": True},
    )

    assert observed_boundaries == [(3, 256)]
    assert report["physical_iteration"] == 3
    assert report["physical_image_size"] == 256
    assert report["classification_ready"] is False
    assert report["capture_validation"]["components"]["status"] == "rejected"
    assert report["capture_validation"]["operands"]["status"] == "accepted"
    assert report["capture_validation"]["lanes"]["status"] == "accepted"


def test_report_rejects_missing_capture_validations(tmp_path: Path, monkeypatch):
    report, _ = _run_report(
        tmp_path,
        monkeypatch,
        physical_iteration=3,
        validations={},
    )

    assert report["classification_ready"] is False
    assert report["capture_validation"]["components"]["status"] == "missing"
    assert report["capture_validation"]["operands"]["status"] == "missing"
    assert report["capture_validation"]["lanes"]["status"] == "missing"


def test_report_accepts_deployed_plural_validation_names(tmp_path: Path, monkeypatch):
    report, _ = _run_report(
        tmp_path,
        monkeypatch,
        physical_iteration=2,
        validations={"components": True, "operands": True, "lanes": True},
        plural_validation_names=True,
    )

    assert report["classification_ready"] is True
    assert report["capture_validation"]["operands"]["path"].endswith(
        "operands_validation.json"
    )
    assert report["capture_validation"]["lanes"]["path"].endswith(
        "lanes_validation.json"
    )


def test_operand_panel_uses_active_overlap_in_requested_order():
    selected, positions, operand_order, recovar_only, native_only = (
        analyzer._matched_operand_rotation_panel(
            np.asarray([534, 27288, 9], dtype=np.int64),
            np.asarray([19000, 534, 9], dtype=np.int64),
            np.asarray([True, True, False]),
        )
    )

    np.testing.assert_array_equal(selected, [534])
    np.testing.assert_array_equal(positions, [0])
    np.testing.assert_array_equal(operand_order, [1])
    assert recovar_only == [27288]
    assert native_only == [19000]


def test_native_coarse_image_size_distinguishes_model_and_scoring_sizes():
    component_header = np.zeros(32, dtype=np.int64)
    operand_header = np.zeros(32, dtype=np.int64)
    component_header[27] = 104
    operand_header[12] = 104
    operand_header[18] = 100

    assert analyzer._native_coarse_image_size(component_header, operand_header) == 100


def test_stage_only_recovar_capture_marks_projection_fields_unavailable(tmp_path: Path):
    path = tmp_path / "stage_only.npz"
    np.savez(
        path,
        original_index=np.asarray(7),
        current_size=np.asarray(10),
        scores_pre_prior_per_class=np.zeros((1, 2, 3)),
        scores_with_prior_per_class=np.zeros((1, 2, 3)),
        weights_per_class=np.ones((1, 6)),
        significant_mask=np.ones(6, dtype=bool),
        translations=np.zeros((3, 2)),
        window_indices=np.arange(4),
        shifted_data=np.zeros((3, 4), dtype=np.complex128),
        ctf2_data=np.zeros((1, 4)),
        half_weights=np.ones(4),
        coarse_gaussian_shifted_corrected=np.zeros((3, 6), dtype=np.complex64),
        coarse_gaussian_unshifted_corrected=np.zeros(6, dtype=np.complex64),
        coarse_gaussian_pixel_weight=np.ones(6, dtype=np.float32),
        coarse_gaussian_initial_diff2=np.asarray(0, dtype=np.float32),
        coarse_gaussian_score_indices=np.arange(6, dtype=np.int32),
    )

    capture = analyzer._load_recovar(path)

    assert capture["projection_capture_available"] is False
    assert capture["rotation_ids"] is None
    assert capture["missing_projection_fields"] == [
        "projected_cross_score_per_class",
        "projected_reference_norm_score_per_class",
        "projected_reference_per_class",
        "projected_reference_rotation_ids",
    ]


def test_counterfactual_accepts_one_rotation_with_multiple_translations():
    total = np.asarray([[0.0, 1.0, -1.0]], dtype=np.float64)
    counterfactual = np.zeros_like(total)

    report = analyzer._candidate_panel_counterfactual(total, counterfactual)

    assert report["baseline_centered_energy"] == 2.0
    assert report["swapped_centered_energy"] == 0.0
    assert report["counterfactual_energy_removal_fraction"] == 1.0


def test_support_mismatch_panel_preserves_native_score_sign():
    report = analyzer._support_mismatch_panel(
        rotation_ids=np.asarray([28138]),
        native_raw=np.asarray([[10.0, 11.0]]),
        native_norm=np.asarray([[4.0, 4.0]]),
        native_cross=np.asarray([[6.0, 7.0]]),
        native_significant=np.asarray([[False, True]]),
        recovar_raw=np.asarray([[-10.25, -10.75]]),
        recovar_norm=np.asarray([[-4.0, -4.0]]),
        recovar_cross=np.asarray([[-6.25, -6.75]]),
        recovar_significant=np.asarray([[True, False]]),
    )

    assert [row["translation_id"] for row in report] == [0, 1]
    assert report[0]["native"]["raw_score"] == -10.0
    assert report[0]["recovar_minus_native"]["raw_score"] == -0.25
    assert report[1]["recovar_minus_native"]["cross_score"] == 0.25


def test_operand_comparison_checks_complex_float32_bits():
    native = np.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    recovar = native.copy()
    recovar[1] = np.complex64(3.0 + np.nextafter(np.float32(4.0), np.float32(5.0)) * 1j)

    report = analyzer._operand_comparison(native, recovar)

    assert report["value_count"] == 2
    assert report["bitwise_equal_count"] == 1
    assert report["max_abs"] > 0.0
    assert report["diagnostic_least_squares_scalar"]["real"] > 1.0

    noncontiguous = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    strided_report = analyzer._operand_comparison(noncontiguous, noncontiguous.copy())
    assert strided_report["bitwise_equal_count"] == noncontiguous.size


def test_active_pixel_operand_comparison_ignores_zero_weight_pixels():
    native = np.asarray([[1.0 + 2.0j, 99.0 + 88.0j, 3.0 + 4.0j]], dtype=np.complex64)
    recovar = np.asarray([[1.0 + 2.0j, 0.0 + 0.0j, 3.0 + 4.0j]], dtype=np.complex64)

    report = analyzer._active_pixel_operand_comparison(
        native,
        recovar,
        np.asarray([True, False, True]),
    )

    assert report["shape"] == [1, 2]
    assert report["value_count"] == 2
    assert report["bitwise_equal_count"] == 2


def test_compact_to_native_full_order_is_a_scatter():
    score_indices = np.asarray(
        [20, 21, 22, 25, 26, 27, 10, 11, 12, 15, 16, 17],
        dtype=np.int64,
    )
    permutation = np.asarray([7, 2, 10, 0, 11, 4, 6, 1, 8, 5, 3, 9])
    values = np.arange(12, dtype=np.float32)

    restored = analyzer._compact_to_native_full_order(
        values[permutation],
        score_indices[permutation],
        physical_image_size=8,
        current_size=4,
    )

    np.testing.assert_array_equal(restored, values)


def test_log_scores_from_lane_partials_uses_thread_order():
    lanes = np.asarray([1.0, 10.0, 100.0, 2.0, 20.0, 200.0], dtype=np.float32)

    scores = analyzer._log_scores_from_lane_partials(lanes, translation_count=3)

    np.testing.assert_array_equal(scores, [-3.0, -30.0, -300.0])


def test_atomic_add_envelope_enumerates_lane_order_rounding():
    # Three lanes per translation, interleaved exactly like the native coarse
    # block.  Cancellation makes the final binary32 value order-dependent.
    lanes = np.asarray(
        [1.0e8, 1.0e8, -1.0e8, -1.0e8, 1.0, 2.0],
        dtype=np.float32,
    )

    first = analyzer._atomic_add_log_score_values(
        lanes,
        translation_count=2,
        translation=0,
        initial_diff2=np.float32(1.0),
    )
    report, margins = analyzer._atomic_relative_score_envelope(
        lanes,
        translation_count=2,
        first_translation=0,
        second_translation=1,
        initial_diff2=np.float32(1.0),
    )

    assert first.size > 1
    assert report["active_lane_count"] == 3
    assert report["relative_log_score_unique_count"] == margins.size
    assert report["relative_log_score_min"] < report["relative_log_score_max"]
