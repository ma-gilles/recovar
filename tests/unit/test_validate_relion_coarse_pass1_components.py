from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts import validate_relion_coarse_pass1_components as validator

PATCH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "patches"
    / "0001-RELION-coarse-diff2-components-combined-capture.patch"
)
PATCH_SHA256 = "84a41aa04d8eae11512fe7728b482eb539844c06ac5be62e39218ee863e17631"


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_artifact(
    directory: Path,
    *,
    perturbation: float = 0.0,
    schema: int = 2,
    inactive_row: bool = False,
    finite_inactive_component: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    reference = np.repeat(
        np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32), 3, axis=1
    )
    cross = np.arange(12, dtype=np.float32).reshape(4, 3) / 16
    raw = reference + cross + np.float32(9.0)
    raw[0, 1] += np.float32(perturbation)
    if inactive_row:
        raw[2] = validator.RELION_INVALID_DIFF2
        reference[2] = np.nan
        cross[2] = np.nan
        if finite_inactive_component:
            reference[2, 0] = 0.0
    weights = np.asarray(
        [[0.8, 0.1, 0], [0.3, 0.2, 0], [0, 0, 0], [0.4, 0, 0]],
        dtype=np.float32,
    )
    mask = (weights >= np.float32(0.3)).astype(np.uint8)
    translations = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    header = [0] * 40
    header[0] = schema
    header[1] = validator.HEADER_STRUCT.size
    header[2] = validator.FLOAT_DTYPE.itemsize
    header[3] = validator.MASK_DTYPE.itemsize
    header[4] = validator.FOOTER_STRUCT.size
    header[5:16] = [2, 4, 101, 1, 0, 2, 2, 3, 12, 3, 0]
    header[16] = _bits(0.3)
    header[17] = _bits(float(np.sum(weights)))
    header[18] = _bits(float(np.max(weights)))
    header[19] = _bits(float(np.min(raw)))
    header[20:23] = [1, 1, 1]
    artifact_bytes = (
        validator.HEADER_STRUCT.size
        + raw.nbytes
        + weights.nbytes
        + reference.nbytes
        + cross.nbytes
        + mask.nbytes
        + translations.nbytes
        + validator.FOOTER_STRUCT.size
    )
    header[23:25] = [artifact_bytes, artifact_bytes]
    header[25:28] = [123, 456, 60]
    header[28] = _bits(0.999)
    header[29] = np.iinfo(np.uint64).max
    header[30] = translations.size
    header[31:37] = [1, 1, 1, 1, 1, schema]
    payload = bytearray(validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header))
    for values in (raw, weights, reference, cross, mask, translations):
        payload.extend(values.tobytes())
    payload.extend(
        validator.FOOTER_STRUCT.pack(validator.FOOTER_MAGIC, 12, 3)
    )
    path = directory / "part4_stack101.p1-v2.bin"
    path.write_bytes(payload)
    return path


def test_validates_component_replay_and_fixed_metric(tmp_path: Path) -> None:
    _write_artifact(tmp_path)
    artifacts, report = validator.validate_directory(
        tmp_path,
        expected_particles=1,
        expected_stack_indices=np.asarray([101]),
        expected_mpi_rank=1,
    )
    assert artifacts[0].reference_norms.shape == (4, 3)
    assert report["status"] == "pass"
    assert report["fixed_metric"] == {
        "evaluated_particles": 1,
        "expected_particles": 1,
        "replay_p95_passed": 1,
        "replay_max_passed": 1,
        "reference_translation_invariance_passed": 1,
    }


def test_replay_gate_rejects_nonconstant_residual(tmp_path: Path) -> None:
    _write_artifact(tmp_path, perturbation=0.01)
    _, report = validator.validate_directory(tmp_path)
    assert report["status"] == "rejected"
    assert report["fixed_metric"]["replay_max_passed"] == 0


def test_rejects_wrong_schema(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, schema=1)
    with pytest.raises(ValueError, match="schema mismatch"):
        validator.load_artifact(path)


def test_rejects_truncated_artifact(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        validator.load_artifact(path)


def test_accepts_nan_components_exactly_on_relion_invalid_rows(
    tmp_path: Path,
) -> None:
    path = _write_artifact(tmp_path, inactive_row=True)
    artifact = validator.load_artifact(path)
    assert np.all(np.isnan(artifact.reference_norms[2]))
    assert np.all(np.isnan(artifact.cross_terms[2]))


def test_rejects_finite_component_on_relion_invalid_row(tmp_path: Path) -> None:
    path = _write_artifact(
        tmp_path,
        inactive_row=True,
        finite_inactive_component=True,
    )
    with pytest.raises(ValueError, match="inactive reference norms"):
        validator.load_artifact(path)


def test_combined_relion_patch_is_frozen_and_keeps_production_diff2() -> None:
    payload = PATCH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PATCH_SHA256
    text = payload.decode()
    assert "RELION_P1_CAPTURE_SCHEMA must be exactly 1 or 2" in text
    assert 'has_components ? "RLNP1V2HEADER" : "RLNP1V1HEADER"' in text
    assert "Coarse diff2 component capture requires both output arrays" in text
    assert (
        "diff2s[j] += (diff_real * diff_real + diff_imag * diff_imag) * "
        "s_corr[i + init_pixel % block_sz];"
    ) in text
    assert "+\t\t\t\tdiff2s[j] +=" not in text
