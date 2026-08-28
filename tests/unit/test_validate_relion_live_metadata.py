from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "validate_relion_live_metadata.py"
SPEC = importlib.util.spec_from_file_location("validate_relion_live_metadata", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_capture(path: Path, *, stage: int, order: list[int], metadata: np.ndarray) -> None:
    particle_count, metadata_width = metadata.shape
    values = [0] * 16
    values[:10] = [1, 144, 64, 8, 32, 7, particle_count, metadata_width, stage, 1]
    with path.open("wb") as stream:
        stream.write(struct.pack("<16s16Q", b"RLNLMDV1HEADER", *values))
        for sorted_position in order:
            record = struct.pack(
                "<QqqqQ3Q",
                sorted_position,
                100 + sorted_position,
                1 + sorted_position,
                1 + sorted_position % 2,
                metadata_width,
                0,
                0,
                0,
            )
            stream.write(record)
            stream.write(np.asarray(metadata[sorted_position], dtype="<f8").tobytes())
        stream.write(struct.pack("<16sQQ", b"RLNLMDV1FOOTER", particle_count, particle_count * metadata_width))


def test_pair_is_sorted_and_reports_exact_changes(tmp_path: Path) -> None:
    incoming = np.arange(75, dtype=np.float64).reshape(3, 25)
    outgoing = incoming.copy()
    outgoing[1, 8] = np.nextafter(outgoing[1, 8], np.inf)
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"
    _write_capture(input_path, stage=1, order=[2, 0, 1], metadata=incoming)
    _write_capture(output_path, stage=2, order=[1, 2, 0], metadata=outgoing)

    report, input_capture, output_capture = MODULE.validate_pair(input_path, output_path)

    assert input_capture.particle_id.tolist() == [100, 101, 102]
    assert output_capture.follower_rank.tolist() == [1, 2, 1]
    assert report["particle_count"] == 3
    assert report["columns"]["pmax"]["changed_particles_bitwise"] == 1
    assert report["columns"]["rot"]["changed_particles_bitwise"] == 0


def test_duplicate_sorted_position_is_rejected(tmp_path: Path) -> None:
    metadata = np.zeros((2, 25), dtype=np.float64)
    path = tmp_path / "duplicate.bin"
    _write_capture(path, stage=1, order=[0, 0], metadata=metadata)
    with pytest.raises(ValueError, match="complete zero-based range"):
        MODULE.load_live_metadata(path)


def test_trailing_or_truncated_bytes_are_rejected(tmp_path: Path) -> None:
    metadata = np.zeros((1, 25), dtype=np.float64)
    path = tmp_path / "capture.bin"
    _write_capture(path, stage=1, order=[0], metadata=metadata)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="size"):
        MODULE.load_live_metadata(path)
