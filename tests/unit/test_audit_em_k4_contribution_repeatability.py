from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import audit_em_k4_contribution_repeatability as auditor


def _write_archives(
    root: Path,
    *,
    observed_value: np.ndarray,
    reference_value: np.ndarray,
    observed_companion: str = "/observed/contribution.npz",
    reference_companion: str = "/reference/contribution.npz",
) -> dict[str, tuple[Path, Path]]:
    paths = {}
    for group in auditor.GROUPS:
        observed_path = root / f"observed_{group}.npz"
        reference_path = root / f"reference_{group}.npz"
        observed = {"value": observed_value}
        reference = {"value": reference_value}
        if group == "device_signature":
            observed["companion_contribution_path"] = np.asarray(observed_companion)
            reference["companion_contribution_path"] = np.asarray(reference_companion)
        np.savez(observed_path, **observed)
        np.savez(reference_path, **reference)
        paths[group] = (observed_path, reference_path)
    return paths


def test_exact_archives_pass_all_fixed_gates(tmp_path: Path) -> None:
    values = np.asarray([1.0, 2.0], dtype=np.float32)
    report = auditor.audit_repeatability(
        _write_archives(
            tmp_path,
            observed_value=values,
            reference_value=values.copy(),
        ),
        owner_job_id=123,
    )

    assert report["status"] == "complete"
    assert report["accepted"] is True
    assert report["fixed_metric"] == {
        "passing": 3,
        "evaluated": 3,
        "gates": {
            "pass2": True,
            "contribution": True,
            "device_signature": True,
        },
    }
    assert report["owner_job_id"] == 123
    assert report["comparisons"]["device_signature"]["arrays"]["companion_contribution_path"]["byte_equal"] is False


def test_signed_zero_difference_is_not_bitwise_equal(tmp_path: Path) -> None:
    observed = np.asarray([0.0, -0.0], dtype=np.float32)
    reference = np.asarray([0.0, 0.0], dtype=np.float32)
    report = auditor.audit_repeatability(
        _write_archives(
            tmp_path,
            observed_value=observed,
            reference_value=reference,
        )
    )

    assert report["accepted"] is False
    comparison = report["comparisons"]["pass2"]["arrays"]["value"]
    assert comparison["mismatch_count"] == 1
    assert comparison["first_mismatch"]["index"] == [1]
    assert comparison["first_mismatch"]["observed_bytes_hex"] != comparison["first_mismatch"]["reference_bytes_hex"]
    assert comparison["max_abs_difference"] == 0.0


def test_distinct_nan_payloads_are_not_bitwise_equal(tmp_path: Path) -> None:
    observed = np.asarray([0x7FC00001], dtype=np.uint32).view(np.float32)
    reference = np.asarray([0x7FC00002], dtype=np.uint32).view(np.float32)
    report = auditor.audit_repeatability(
        _write_archives(
            tmp_path,
            observed_value=observed,
            reference_value=reference,
        )
    )

    assert report["accepted"] is False
    comparison = report["comparisons"]["contribution"]["arrays"]["value"]
    assert comparison["mismatch_count"] == 1
    assert comparison["max_abs_difference"] is None


def test_shape_and_dtype_mismatches_are_recorded(tmp_path: Path) -> None:
    paths = _write_archives(
        tmp_path,
        observed_value=np.asarray([1, 2], dtype=np.int32),
        reference_value=np.asarray([1, 2], dtype=np.int64),
    )
    observed_pass2, reference_pass2 = paths["pass2"]
    np.savez(observed_pass2, value=np.asarray([[1, 2]], dtype=np.float32))
    np.savez(reference_pass2, value=np.asarray([1, 2], dtype=np.float32))

    report = auditor.audit_repeatability(paths)

    pass2 = report["comparisons"]["pass2"]["arrays"]["value"]
    contribution = report["comparisons"]["contribution"]["arrays"]["value"]
    assert pass2["shape_equal"] is False
    assert pass2["mismatch_count"] is None
    assert contribution["dtype_equal"] is False
    assert contribution["mismatch_count"] is None


def test_missing_observed_archive_emits_incomplete_report(
    tmp_path: Path,
) -> None:
    values = np.asarray([1.0], dtype=np.float32)
    paths = _write_archives(
        tmp_path,
        observed_value=values,
        reference_value=values,
    )
    paths["pass2"][0].unlink()

    report = auditor.audit_repeatability(paths)

    assert report["status"] == "incomplete"
    assert report["accepted"] is False
    assert "FileNotFoundError" in report["errors"]["pass2"]
    assert report["fixed_metric"]["gates"]["pass2"] is False


def test_cli_writes_failure_report_before_returning_nonzero(
    tmp_path: Path,
) -> None:
    paths = _write_archives(
        tmp_path,
        observed_value=np.asarray([1.0], dtype=np.float32),
        reference_value=np.asarray([2.0], dtype=np.float32),
    )
    output = tmp_path / "report.json"
    argv: list[str] = []
    for group, (observed, reference) in paths.items():
        option = group.replace("_", "-")
        argv.extend(
            [
                f"--observed-{option}",
                str(observed),
                f"--reference-{option}",
                str(reference),
            ]
        )
    argv.extend(["--output", str(output), "--require-accepted"])

    status = auditor.main(argv)

    assert status == 1
    report = json.loads(output.read_text())
    assert report["status"] == "complete"
    assert report["accepted"] is False
