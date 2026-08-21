from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_bpref_factor_split as factors
from scripts import validate_relion_bpref_prescatter as summary

pytestmark = pytest.mark.unit


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_pair(directory: Path, *, truncate_factor: bool = False) -> tuple[Path, Path]:
    rotations = np.zeros(1, dtype=summary.ROTATION_DTYPE)
    rotations["orientation_class_key"] = 9
    rotations["matrix"][0] = np.eye(3, dtype=np.float32).reshape(-1)
    rows = np.zeros(2, dtype=summary.ROW_DTYPE)
    rows["state"] = 1
    rows["orientation_local"] = 0
    rows["pixel"] = [1, 2]
    rows["flags"] = summary.ROW_FLAG_FWEIGHT_POSITIVE | summary.ROW_FLAG_RADIUS_SUPPORT
    rows["x"] = [1, 2]
    rows["source_re"] = [6.0, 7.0]
    rows["source_im"] = [-8.0, -9.0]
    rows["source_weight"] = [10.0, 11.0]
    values = [0] * 40
    values[:5] = [1, summary.HEADER_STRUCT.size, summary.ROW_DTYPE.itemsize, summary.ROTATION_DTYPE.itemsize, summary.FOOTER_STRUCT.size]
    values[5:20] = [1, 1, 4, 101, 4, 2, 0, 129, 256, 1, 33024, 1, 2, 24, 576]
    values[20:25] = [_float_bits(2.0), _float_bits(0.1), _float_bits(1.0), 0, 0]
    values[25:30] = [1, 1, 1, 10_000_000, 1_000_000]
    values[30:38] = [123, 456, 99, 99, 99, 0, 0, 1]
    values[38:40] = [2, 0]
    summary_path = directory / "part4_stack101_img4_class1.bpre-v1.bin"
    summary_path.write_bytes(
        summary.HEADER_STRUCT.pack(summary.HEADER_MAGIC, *values)
        + rotations.tobytes()
        + rows.tobytes()
        + summary.FOOTER_STRUCT.pack(summary.FOOTER_MAGIC, rows.size, rotations.size)
    )

    factor_rows = np.zeros(2, dtype=factors.FACTOR_ROW_DTYPE)
    factor_rows["state"] = 1
    factor_rows["orientation_local"] = 0
    factor_rows["translation_local"] = 3
    factor_rows["pixel"] = [1, 2]
    factor_rows["flags"] = factors.ROW_FLAG_FWEIGHT_POSITIVE | factors.ROW_FLAG_RADIUS_SUPPORT
    factor_rows["x"] = [1, 2]
    factor_rows["posterior"] = 1
    factor_rows["posterior_over_weight_norm"] = 1
    factor_rows["minvsigma2"] = 2
    factor_rows["ctf"] = 3
    factor_rows["weight_term"] = [10, 11]
    factor_rows["running_re"] = [6, 7]
    factor_rows["running_im"] = [-8, -9]
    factor_rows["running_weight"] = [10, 11]
    factor_header = [0] * 40
    factor_header[:4] = [1, factors.HEADER_STRUCT.size, factors.FACTOR_ROW_DTYPE.itemsize, factors.FOOTER_STRUCT.size]
    factor_header[4:19] = [1, 1, 4, 101, 4, 2, 0, 129, 256, 1, 33024, 1, 116, 1, 2]
    factor_header[19:30] = [_float_bits(1.0), _float_bits(0.1), 0, 1, 1, 10_000_000, 1_000_000, 123, 456, 1, 1]
    factor_path = directory / "part4_stack101_img4_class1.bpf-v1.bin"
    payload = (
        factors.HEADER_STRUCT.pack(factors.HEADER_MAGIC, *factor_header)
        + factor_rows.tobytes()
        + factors.FOOTER_STRUCT.pack(factors.FOOTER_MAGIC, factor_rows.size, 1)
    )
    factor_path.write_bytes(payload[:-1] if truncate_factor else payload)
    return summary_path, factor_path


def test_validate_factor_panel_and_exact_summary_closure(tmp_path: Path):
    _write_pair(tmp_path)

    artifacts, report = factors.validate_factor_directory(
        tmp_path, expected_stack_indices=np.asarray([101], dtype=np.int64)
    )

    assert len(artifacts) == 1
    assert report["classification_ready"] is True
    assert report["device_running_sum_exact_summary_closure"] is True
    assert report["active_pair_count_min"] == report["active_pair_count_max"] == 1


def test_load_factor_rejects_truncation(tmp_path: Path):
    _, factor_path = _write_pair(tmp_path, truncate_factor=True)

    with pytest.raises(ValueError, match="factor artifact byte count mismatch"):
        factors.load_factor_artifact(factor_path)


def test_validate_factor_rejects_running_sum_mismatch(tmp_path: Path):
    _, factor_path = _write_pair(tmp_path)
    payload = bytearray(factor_path.read_bytes())
    running_re_offset = factors.HEADER_STRUCT.size + factors.FACTOR_ROW_DTYPE.fields["running_re"][1]
    struct.pack_into("<f", payload, running_re_offset, 999.0)
    factor_path.write_bytes(payload)

    with pytest.raises(ValueError, match="running data does not exactly close"):
        factors.validate_factor_directory(
            tmp_path, expected_stack_indices=np.asarray([101], dtype=np.int64)
        )


def test_validate_factor_allows_exact_zero_weight_radius_support(tmp_path: Path):
    _, factor_path = _write_pair(tmp_path)
    payload = factor_path.read_bytes()
    magic, *header = factors.HEADER_STRUCT.unpack_from(payload)
    rows = np.frombuffer(
        payload,
        dtype=factors.FACTOR_ROW_DTYPE,
        count=header[18],
        offset=factors.HEADER_STRUCT.size,
    ).copy()
    expanded = np.zeros(3, dtype=factors.FACTOR_ROW_DTYPE)
    expanded[:2] = rows
    expanded[2]["state"] = 1
    expanded[2]["orientation_local"] = 0
    expanded[2]["translation_local"] = 3
    expanded[2]["pixel"] = 3
    expanded[2]["flags"] = factors.ROW_FLAG_FWEIGHT_POSITIVE | factors.ROW_FLAG_RADIUS_SUPPORT
    expanded[2]["x"] = 3
    expanded[2]["posterior"] = 1
    expanded[2]["posterior_over_weight_norm"] = 1
    expanded[2]["minvsigma2"] = 0
    expanded[2]["ctf"] = 1
    header[18] = expanded.size
    factor_path.write_bytes(
        factors.HEADER_STRUCT.pack(magic, *header)
        + expanded.tobytes()
        + factors.FOOTER_STRUCT.pack(factors.FOOTER_MAGIC, expanded.size, 1)
    )

    _, report = factors.validate_factor_directory(
        tmp_path, expected_stack_indices=np.asarray([101], dtype=np.int64)
    )

    assert report["per_particle_closure"][0]["zero_weight_radius_support_count"] == 1
