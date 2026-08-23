from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_em_k4_raw_diff2_strata as analyzer


def _summarize(native: np.ndarray, recovar: np.ndarray):
    count = native.size
    return analyzer.summarize_raw_diff2_strata(
        native_raw=native,
        recovar_raw=recovar,
        recovar_rotation=np.asarray([2, 2, 5, 5])[:count],
        translation=np.asarray([7, 8, 7, 9])[:count],
        native_candidate_index=np.asarray([11, 12, 13, 14])[:count],
    )


def test_exact_table_has_fixed_zero_mismatch_summary() -> None:
    native = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    report = _summarize(native, native.copy())

    assert report["active_count"] == 4
    assert report["bitwise_match_count"] == 4
    assert report["bitwise_mismatch_count"] == 0
    assert report["bitwise_exact"] is True
    assert report["absolute_delta_quantiles_nonzero"] is None
    assert report["float32_ulp_distance_quantiles_nonzero"] is None
    assert report["representative"] is None
    assert report["signed_mismatch_counts"] == {
        "recovar_lower": 0,
        "equal_numeric_nonbitwise": 0,
        "recovar_higher": 0,
    }


def test_stratifies_signed_mismatches_and_selects_fixed_representative() -> None:
    native = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    recovar = np.asarray([10.0, 19.5, 31.0, 40.5], dtype=np.float32)

    report = _summarize(native, recovar)

    assert report["bitwise_match_count"] == 1
    assert report["bitwise_mismatch_count"] == 3
    assert report["bitwise_exact"] is False
    assert report["signed_mismatch_counts"] == {
        "recovar_lower": 1,
        "equal_numeric_nonbitwise": 0,
        "recovar_higher": 2,
    }
    assert report["rotation_strata"] == {
        "mismatching_rotation_count": 2,
        "maximum_mismatches_in_one_rotation": 2,
    }
    assert report["translation_strata"] == {
        "mismatching_translation_count": 3,
        "maximum_mismatches_in_one_translation": 1,
    }
    assert report["absolute_delta_quantiles_nonzero"] == {
        "minimum": 0.5,
        "p50": 0.5,
        "p95": 0.95,
        "p99": 0.99,
        "maximum": 1.0,
    }
    assert report["float32_ulp_distance_quantiles_nonzero"] == {
        "minimum": 131072.0,
        "p50": 262144.0,
        "p95": 498073.6,
        "p99": 519045.12,
        "maximum": 524288.0,
    }
    assert report["representative"] == {
        "selection_rule": (
            "maximum_absolute_delta_then_lowest_native_candidate_index"
        ),
        "native_candidate_index": 13,
        "recovar_rotation_row": 5,
        "translation_id": 7,
        "native_raw_diff2": 30.0,
        "native_raw_diff2_bits": 1106247680,
        "recovar_raw_diff2": 31.0,
        "recovar_raw_diff2_bits": 1106771968,
        "delta_recovar_minus_native": 1.0,
        "absolute_delta": 1.0,
        "float32_ulp_distance": 524288,
    }


def test_representative_tie_breaks_by_native_candidate_index() -> None:
    native = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    recovar = np.asarray([11.0, 19.0, 30.0], dtype=np.float32)

    report = analyzer.summarize_raw_diff2_strata(
        native_raw=native,
        recovar_raw=recovar,
        recovar_rotation=np.asarray([3, 2, 1]),
        translation=np.asarray([4, 5, 6]),
        native_candidate_index=np.asarray([12, 7, 20]),
    )

    assert report["representative"]["native_candidate_index"] == 7
    assert report["representative"]["recovar_rotation_row"] == 2
    assert report["representative"]["translation_id"] == 5


def test_counts_equal_numeric_nonbitwise_signed_zero() -> None:
    native = np.asarray([0.0, 1.0], dtype=np.float32)
    recovar = np.asarray([-0.0, 1.0], dtype=np.float32)

    report = analyzer.summarize_raw_diff2_strata(
        native_raw=native,
        recovar_raw=recovar,
        recovar_rotation=np.asarray([0, 0]),
        translation=np.asarray([0, 1]),
        native_candidate_index=np.asarray([0, 1]),
    )

    assert report["bitwise_mismatch_count"] == 1
    assert report["signed_mismatch_counts"]["equal_numeric_nonbitwise"] == 1
    assert report["representative"]["absolute_delta"] == 0.0
    assert report["representative"]["float32_ulp_distance"] == 1


def _hashed_input(path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _build_report_fixture(tmp_path, monkeypatch):
    factor_path = tmp_path / "factor.bin"
    fine_score_path = tmp_path / "fine-score.bin"
    factor_path.write_bytes(b"factor")
    fine_score_path.write_bytes(b"score")
    recovar_pass2_path = tmp_path / "pass2.npz"
    rotations = np.zeros((2, 3, 3), dtype=np.float32)
    candidate_mask = np.zeros((2, 3), dtype=bool)
    candidate_mask[1, 0] = True
    candidate_mask[1, 1] = True
    candidate_mask[0, 1] = True
    candidate_mask[0, 2] = True
    raw_table = np.full((2, 3), np.nan, dtype=np.float32)
    raw_table[1, 0] = 10.0
    raw_table[1, 1] = 20.0
    raw_table[0, 1] = 31.0
    raw_table[0, 2] = 40.0
    np.savez(
        recovar_pass2_path,
        rotations=rotations,
        candidate_mask=candidate_mask,
        relion_raw_diff2=raw_table,
    )
    dtype = np.dtype(
        [
            ("flags", np.uint32),
            ("rotation_local", np.int32),
            ("translation_id", np.int32),
            ("raw_diff2", np.float32),
        ]
    )
    candidates = np.asarray(
        [
            (analyzer.ACTIVE, 0, 0, 10.0),
            (analyzer.ACTIVE, 0, 1, 20.0),
            (analyzer.ACTIVE, 1, 1, 30.0),
            (analyzer.ACTIVE, 1, 2, 40.0),
        ],
        dtype=dtype,
    )
    monkeypatch.setattr(analyzer, "EXPECTED_SUPPORT", 4)
    monkeypatch.setattr(analyzer, "EXPECTED_ROTATIONS", 2)
    monkeypatch.setattr(
        analyzer,
        "load_factor_capture",
        lambda _: SimpleNamespace(rotations={"matrix": rotations}),
    )
    monkeypatch.setattr(
        analyzer,
        "load_fine_score_capture",
        lambda _: SimpleNamespace(candidates=candidates),
    )
    monkeypatch.setattr(
        analyzer,
        "_rotation_permutation",
        lambda _native, _recovar: np.asarray([1, 0]),
    )
    raw_report_path = tmp_path / "raw-report.json"
    raw_report_path.write_text(
        json.dumps(
            {
                "schema": analyzer.RAW_REPORT_SCHEMA,
                "status": "complete",
                "classification_ready": True,
                "support": {"exact": True},
                "raw_diff2": {
                    "count": 4,
                    "bitwise_mismatch_count": 1,
                    "bitwise_exact": False,
                },
                "inputs": {
                    "factor": _hashed_input(factor_path),
                    "fine_score": _hashed_input(fine_score_path),
                    "recovar_pass2": _hashed_input(recovar_pass2_path),
                },
            }
        )
    )
    return raw_report_path


def test_build_report_replays_parent_metric_and_hashes(
    tmp_path,
    monkeypatch,
) -> None:
    raw_report_path = _build_report_fixture(tmp_path, monkeypatch)

    report = analyzer.build_report(raw_report_path=raw_report_path)

    assert report["classification"] == analyzer.MISMATCH_CLASSIFICATION
    assert report["accepted"] is False
    assert report["scorecard_change_admissible"] is False
    assert report["strata"]["active_count"] == 4
    assert report["strata"]["bitwise_mismatch_count"] == 1
    assert report["strata"]["representative"]["native_candidate_index"] == 2
    assert report["inputs"]["raw_report"]["sha256"] == hashlib.sha256(
        raw_report_path.read_bytes()
    ).hexdigest()


def test_build_report_rejects_parent_metric_that_does_not_replay(
    tmp_path,
    monkeypatch,
) -> None:
    raw_report_path = _build_report_fixture(tmp_path, monkeypatch)
    raw_report = json.loads(raw_report_path.read_text())
    raw_report["raw_diff2"]["bitwise_mismatch_count"] = 2
    raw_report_path.write_text(json.dumps(raw_report))

    with pytest.raises(ValueError, match="do not replay"):
        analyzer.build_report(raw_report_path=raw_report_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("recovar_raw", np.asarray([1.0]), "one-dimensional and equal length"),
        (
            "recovar_rotation",
            np.asarray([-1, 0]),
            "identities must be nonnegative",
        ),
        (
            "native_candidate_index",
            np.asarray([1, 1]),
            "indices must be unique",
        ),
        (
            "native_raw",
            np.asarray([-1.0, 2.0]),
            "costs must be nonnegative",
        ),
    ],
)
def test_rejects_invalid_aligned_inputs(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    kwargs = {
        "native_raw": np.asarray([1.0, 2.0], dtype=np.float32),
        "recovar_raw": np.asarray([1.0, 2.0], dtype=np.float32),
        "recovar_rotation": np.asarray([0, 0]),
        "translation": np.asarray([0, 1]),
        "native_candidate_index": np.asarray([0, 1]),
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        analyzer.summarize_raw_diff2_strata(**kwargs)
