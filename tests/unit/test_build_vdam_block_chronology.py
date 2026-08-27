from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.build_vdam_block_chronology import (
    HEADER_DTYPE,
    MAGIC,
    RECORD_DTYPE,
    load_capture,
    validate_capture,
)


def _records() -> np.ndarray:
    records = np.zeros(5, dtype=RECORD_DTYPE)
    records["launch_sequence"] = [0, 0, 0, 1, 1]
    records["particle_id"] = [7, 7, 7, 11, 11]
    records["block_start_globaltimer"] = [100, 102, 101, 110, 111]
    records["first_atomic_globaltimer"] = [105, 107, 106, 115, 116]
    records["block_end_globaltimer"] = [140, 142, 141, 130, 131]
    records["orientation_row"] = [0, 1, 2, 0, 1]
    records["worker_id"] = [2, 2, 2, 5, 5]
    records["class_id"] = 0
    records["sm_id"] = [4, 6, 5, 9, 10]
    records["image_count"] = [3, 3, 3, 2, 2]
    records["iteration"] = 1
    records["flags"] = 4
    return records


def _write_capture(path: Path, records: np.ndarray, *, trailing: bytes = b"") -> None:
    header = np.zeros(1, dtype=HEADER_DTYPE)
    header["magic"] = MAGIC
    header["schema_version"] = 1
    header["header_size"] = HEADER_DTYPE.itemsize
    header["record_size"] = RECORD_DTYPE.itemsize
    header["iteration"] = 1
    header["record_count"] = records.size
    header["capacity"] = 100
    path.write_bytes(header.tobytes() + records.tobytes() + trailing)


@pytest.mark.unit
def test_block_chronology_validates_and_orders_native_timestamps(tmp_path):
    capture = tmp_path / "trace.bin"
    _write_capture(capture, _records())
    header, records = load_capture(capture)
    result = validate_capture(
        header,
        records,
        iteration=1,
        n_particles=2,
        n_threads=8,
        n_classes=1,
        sm_count=132,
    )
    assert result["launch_count"] == 2
    assert result["block_start_order"].tolist() == [0, 2, 1, 3, 4]
    assert result["first_atomic_order"].tolist() == [0, 2, 1, 3, 4]
    assert result["sm_ids"].tolist() == [4, 5, 6, 9, 10]


@pytest.mark.unit
def test_block_chronology_rejects_truncation_and_trailing_bytes(tmp_path):
    capture = tmp_path / "trace.bin"
    _write_capture(capture, _records(), trailing=b"x")
    with pytest.raises(ValueError, match="trailing bytes"):
        load_capture(capture)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("worker_id", 8, "worker IDs"),
        ("sm_id", 132, "SM IDs"),
        ("first_atomic_globaltimer", 0, "first-atomic"),
        ("orientation_row", 9, "orientation row"),
    ],
)
def test_block_chronology_rejects_invalid_record_fields(tmp_path, field, value, message):
    capture = tmp_path / "trace.bin"
    records = _records()
    records[field][0] = value
    _write_capture(capture, records)
    header, loaded = load_capture(capture)
    with pytest.raises(ValueError, match=message):
        validate_capture(
            header,
            loaded,
            iteration=1,
            n_particles=2,
            n_threads=8,
            n_classes=1,
            sm_count=132,
        )


@pytest.mark.unit
def test_block_chronology_rejects_incomplete_launch(tmp_path):
    capture = tmp_path / "trace.bin"
    records = _records()[:-1]
    _write_capture(capture, records)
    header, loaded = load_capture(capture)
    with pytest.raises(ValueError, match="launch 1 is incomplete"):
        validate_capture(
            header,
            loaded,
            iteration=1,
            n_particles=2,
            n_threads=8,
            n_classes=1,
            sm_count=132,
        )
