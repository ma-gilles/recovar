from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import validate_relion_k4_coarse_operand_capture as validator


def test_direct_script_entrypoint_can_resolve_repository_imports() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(validator.__file__).resolve()), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--capture-dir" in result.stdout


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_operand(directory: Path, class_one_based: int) -> Path:
    rotation_keys = np.arange(
        (class_one_based - 1) * 2, class_one_based * 2, dtype=np.uint64
    )
    local_indices = np.arange(2, dtype=np.uint64)
    matrices = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    reference_real = np.asarray(
        [[1.0, 0.5, -0.25, 0.75], [0.25, -0.5, 1.0, 0.125]],
        dtype=np.float32,
    ) * np.float32(class_one_based)
    reference_imag = np.asarray(
        [[0.0, 0.25, 0.5, -0.25], [0.75, 0.0, -0.125, 0.5]],
        dtype=np.float32,
    )
    image_real = np.asarray([0.5, 1.0, -0.5, 0.25], dtype=np.float32)
    image_imag = np.asarray([0.25, -0.5, 0.75, 1.0], dtype=np.float32)
    correction = np.asarray([2.0, 1.0, 0.5, 3.0], dtype=np.float32)
    translations = np.asarray(
        [[0.0, 0.2, -0.3], [0.0, -0.1, 0.25], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    pixels = np.arange(4)
    x = (pixels % 2).astype(np.float32)
    y = np.where(pixels // 2 > 1, pixels // 2 - 2, pixels // 2).astype(
        np.float32
    )
    phase = (
        translations[0, :, None] * x[None]
        + translations[1, :, None] * y[None]
    ).astype(np.float32)
    sine = np.sin(phase).astype(np.float32)
    cosine = np.cos(phase).astype(np.float32)
    shifted_real = (
        cosine * image_real[None] - sine * image_imag[None]
    ).astype(np.float32)
    shifted_imag = (
        cosine * image_imag[None] + sine * image_real[None]
    ).astype(np.float32)
    artifact_bytes = (
        validator.HEADER_STRUCT.size
        + rotation_keys.nbytes
        + local_indices.nbytes
        + matrices.nbytes
        + reference_real.nbytes
        + reference_imag.nbytes
        + image_real.nbytes
        + image_imag.nbytes
        + correction.nbytes
        + translations.nbytes
        + shifted_real.nbytes
        + shifted_imag.nbytes
        + validator.FOOTER_STRUCT.size
    )
    header = [0] * 64
    header[:5] = [
        1,
        validator.HEADER_STRUCT.size,
        validator.FLOAT_DTYPE.itemsize,
        validator.UINT64_DTYPE.itemsize,
        validator.FOOTER_STRUCT.size,
    ]
    header[5:17] = [2, 4, 101, 1, 0, class_one_based, 0, 8, 4, 3, 2, 2]
    header[17:24] = [2, 2, 1, 1, 8, 8, 8]
    header[24] = _bits(1.0)
    header[25:32] = [1, 2, 2 * artifact_bytes, artifact_bytes, 123, 456, 789]
    header[32:42] = [1, 1, 1, 1, 1, 1, 8, 2, 2, 2]
    payload = bytearray(validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header))
    for values in (
        rotation_keys,
        local_indices,
        matrices,
        reference_real,
        reference_imag,
        image_real,
        image_imag,
        correction,
        translations,
        shifted_real,
        shifted_imag,
    ):
        payload.extend(values.tobytes())
    payload.extend(validator.FOOTER_STRUCT.pack(validator.FOOTER_MAGIC, 2, 3, 4))
    path = directory / (
        f"part4_stack101_class{class_one_based}.coarse-operands-v1.bin"
    )
    path.write_bytes(payload)
    return path


def _paired_tables(artifacts: list[validator.K4CoarseOperandCapture]) -> tuple[object, object]:
    component_dtype = np.dtype(
        [("reference_norm", "<f8"), ("cross_term", "<f8")]
    )
    score_dtype = np.dtype([("raw_diff2", "<f4")])
    components = np.zeros(12, dtype=component_dtype)
    scores = np.zeros(12, dtype=score_dtype)
    for artifact in artifacts:
        shared = validator._as_replay_capture(artifact)
        reference, cross = validator.replay.replay_components(shared)
        diff2 = validator.replay.replay_production_diff2(shared)
        flat = (
            artifact.rotation_keys[:, None] * np.uint64(3)
            + np.arange(3, dtype=np.uint64)[None]
        )
        components["reference_norm"][flat] = reference[:, None]
        components["cross_term"][flat] = cross
        scores["raw_diff2"][flat] = diff2 + np.float32(5.0)
    component_header = [0] * 64
    component_header[4:19] = [2, 4, 101, 1, 0, 0, 2, 2, 1, 3, 12, 123, 456, 789, 1]
    score_header = [0] * 64
    score_header[5:7] = [4, 101]
    return (
        SimpleNamespace(header=tuple(component_header), candidates=components),
        SimpleNamespace(header=tuple(score_header), candidates=scores),
    )


def test_validates_complete_k4_style_operand_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = [validator.load_artifact(_write_operand(tmp_path, cls)) for cls in (1, 2)]
    component, score = _paired_tables(artifacts)
    monkeypatch.setattr(
        validator.component_validator,
        "validate_directory",
        lambda *_args, **_kwargs: {"schema": "paired-test"},
    )
    monkeypatch.setattr(
        validator.component_validator,
        "load_coarse_component_capture",
        lambda _path: component,
    )
    monkeypatch.setattr(
        validator.score_validator,
        "load_coarse_score_capture",
        lambda _path: score,
    )
    report = validator.validate_directory(
        tmp_path, (101,), expected_iteration=2
    )
    assert report["status"] == "pass"
    assert report["class_count"] == 2
    assert report["fixed_metric"] == {
        "evaluated_artifacts": 2,
        "passed_artifacts": 2,
    }


def test_rejects_incomplete_particle_class_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_operand(tmp_path, 1)
    monkeypatch.setattr(
        validator.component_validator,
        "validate_directory",
        lambda *_args, **_kwargs: {"schema": "paired-test"},
    )
    with pytest.raises(ValueError, match="panel is incomplete"):
        validator.validate_directory(tmp_path, (101,), expected_iteration=2)


def test_rejects_truncated_k4_operand_artifact(tmp_path: Path) -> None:
    path = _write_operand(tmp_path, 1)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        validator.load_artifact(path)
