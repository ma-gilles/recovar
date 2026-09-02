from __future__ import annotations

import struct

import numpy as np
import pytest

from scripts import validate_relion_coarse_component_capture as validator
from scripts import validate_relion_coarse_score_capture as score_validator


def _bits(value: np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_score(path, *, stack: int, selected_text: str, expected_particles: int):
    class_count, nr_dir, nr_psi, nr_trans = 2, 2, 1, 2
    candidate_count = class_count * nr_dir * nr_psi * nr_trans
    min_diff2 = np.float32(8.0)
    orientation_prior = np.repeat(
        np.asarray([-0.1, -0.2, -0.3, -0.4], dtype=np.float32), nr_trans
    )
    translation_prior = np.tile(
        np.asarray([-0.5, -0.75], dtype=np.float32), candidate_count // nr_trans
    )
    raw = np.asarray([8.0, 9.0, 8.5, 10.0, 9.0, 10.0, 8.2, 11.0], dtype=np.float32)
    flags = np.zeros(candidate_count, dtype=np.uint32)
    flags[1::2] |= score_validator.TRANSLATION_ZERO
    flags[4:6] |= score_validator.ORIENTATION_ZERO
    active = (flags & score_validator.REJECTION_FLAGS) == 0
    flags[active] |= score_validator.ACTIVE
    combined = np.full(candidate_count, -np.inf, dtype=np.float32)
    combined[active] = (
        orientation_prior[active]
        + translation_prior[active]
        + min_diff2
        - raw[active]
    )
    weights_max = np.max(combined[active])
    shift = np.float32(50.0) - weights_max
    shifted = combined + shift
    post = np.zeros(candidate_count, dtype=np.float32)
    post[active] = np.exp(shifted[active], dtype=np.float32)
    active_order = np.flatnonzero(active)[np.argsort(-post[active], kind="stable")]
    significant_weight = post[active_order[1]]
    significant = post >= significant_weight
    flags[significant] |= score_validator.SIGNIFICANT

    candidates = np.zeros(candidate_count, dtype=score_validator.CANDIDATE_DTYPE)
    flat = np.arange(candidate_count, dtype=np.uint64)
    orientation = flat // nr_trans
    class_direction = orientation // nr_psi
    candidates["flat_index"] = flat
    candidates["class_one_based"] = class_direction // nr_dir + 1
    candidates["direction"] = class_direction % nr_dir
    candidates["psi"] = orientation % nr_psi
    candidates["translation"] = flat % nr_trans
    candidates["flags"] = flags
    candidates["raw_diff2"] = raw
    candidates["orientation_log_prior"] = orientation_prior
    candidates["translation_log_prior"] = translation_prior
    candidates["combined_preexponent"] = combined
    candidates["shifted_log_weight"] = shifted
    candidates["post_exponent_weight"] = post

    header = [0] * 64
    header[:4] = [
        1,
        score_validator.HEADER_STRUCT.size,
        score_validator.CANDIDATE_DTYPE.itemsize,
        score_validator.FOOTER_STRUCT.size,
    ]
    header[4:15] = [
        1,
        stack + 100,
        stack,
        2,
        0,
        0,
        class_count,
        nr_dir,
        nr_psi,
        nr_trans,
        candidate_count,
    ]
    header[15:20] = [
        _bits(min_diff2),
        _bits(weights_max),
        _bits(shift),
        _bits(significant_weight),
        _bits(np.sum(post, dtype=np.float32)),
    ]
    header[20:28] = [
        2,
        int(np.count_nonzero(significant)),
        int(np.count_nonzero(post > 0)),
        1,
        int(np.argmax(post)),
        _bits(np.max(post)),
        _bits(np.float32(0.999)),
        (1 << 64) - 1,
    ]
    header[28:32] = [
        11,
        12,
        score_validator.fnv1a64(selected_text),
        expected_particles,
    ]
    expected_bytes = (
        score_validator.HEADER_STRUCT.size
        + candidates.nbytes
        + score_validator.FOOTER_STRUCT.size
    )
    header[32:41] = [expected_bytes, 1_000_000, 4, 1, 0, 13, 1, 1, 1]
    footer = score_validator.FOOTER_STRUCT.pack(
        score_validator.FOOTER_MAGIC,
        candidate_count,
        int(np.count_nonzero(flags & score_validator.ACTIVE)),
        int(np.count_nonzero(significant)),
    )
    path.write_bytes(
        score_validator.HEADER_STRUCT.pack(score_validator.HEADER_MAGIC, *header)
        + candidates.tobytes()
        + footer
    )


def _write_component(
    path,
    *,
    stack: int,
    selected_text: str,
    expected_particles: int,
    corrupt_flat_index: bool = False,
    nonfinite: bool = False,
):
    class_count, nr_dir, nr_psi, nr_trans = 2, 2, 1, 2
    candidate_count = class_count * nr_dir * nr_psi * nr_trans
    candidates = np.zeros(candidate_count, dtype=validator.CANDIDATE_DTYPE)
    candidates["flat_index"] = np.arange(candidate_count, dtype=np.uint64)
    if corrupt_flat_index:
        candidates["flat_index"][3] = 99
    candidates["reference_norm"] = np.repeat(
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64), nr_trans
    )
    candidates["cross_term"] = np.arange(candidate_count, dtype=np.float64) / 16.0
    if nonfinite:
        candidates["cross_term"][3] = np.nan
    header = [0] * 64
    header[:4] = [
        1,
        validator.HEADER_STRUCT.size,
        validator.CANDIDATE_DTYPE.itemsize,
        validator.FOOTER_STRUCT.size,
    ]
    header[4:15] = [
        1,
        stack + 100,
        stack,
        2,
        0,
        0,
        class_count,
        nr_dir,
        nr_psi,
        nr_trans,
        candidate_count,
    ]
    expected_bytes = (
        validator.HEADER_STRUCT.size
        + candidates.nbytes
        + validator.FOOTER_STRUCT.size
    )
    header[15:27] = [
        11,
        12,
        score_validator.fnv1a64(selected_text),
        expected_particles,
        expected_bytes,
        1_000_000,
        8,
        0,
        13,
        1,
        1,
        1,
    ]
    finite_count = int(
        np.count_nonzero(
            np.isfinite(candidates["reference_norm"])
            & np.isfinite(candidates["cross_term"])
        )
    )
    footer = validator.FOOTER_STRUCT.pack(
        validator.FOOTER_MAGIC, candidate_count, finite_count
    )
    path.write_bytes(
        validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header)
        + candidates.tobytes()
        + footer
    )


def _write_pair(directory, *, stack: int, selected_text: str, expected_particles: int):
    stem = f"part{stack + 100}_stack{stack}"
    _write_score(
        directory / f"{stem}.coarse-score-v1.bin",
        stack=stack,
        selected_text=selected_text,
        expected_particles=expected_particles,
    )
    component_path = directory / f"{stem}.coarse-components-v1.bin"
    _write_component(
        component_path,
        stack=stack,
        selected_text=selected_text,
        expected_particles=expected_particles,
    )
    return component_path


def test_paired_component_capture_round_trips_complete_boundary(tmp_path):
    _write_pair(tmp_path, stack=17, selected_text="17,23", expected_particles=2)
    _write_pair(tmp_path, stack=23, selected_text="17,23", expected_particles=2)

    report = validator.validate_directory(tmp_path, (17, 23), expected_iteration=1)

    assert report["capture_ready"] is True
    assert report["particle_count"] == 2
    assert report["candidate_count"] == 16
    assert all(
        row["metrics"]["reference_norm_translation_spread_max"] == 0
        for row in report["files"]
    )


def test_component_capture_rejects_order_and_nonfinite_values(tmp_path):
    ordered = tmp_path / "part117_stack17.coarse-components-v1.bin"
    _write_component(
        ordered,
        stack=17,
        selected_text="17",
        expected_particles=1,
        corrupt_flat_index=True,
    )
    with pytest.raises(ValueError, match="flattened order"):
        validator.load_coarse_component_capture(ordered)

    _write_component(
        ordered,
        stack=17,
        selected_text="17",
        expected_particles=1,
        nonfinite=True,
    )
    with pytest.raises(ValueError, match="non-finite"):
        validator.load_coarse_component_capture(ordered)


def test_component_directory_requires_identity_matched_score(tmp_path):
    _write_component(
        tmp_path / "part117_stack17.coarse-components-v1.bin",
        stack=17,
        selected_text="17",
        expected_particles=1,
    )
    with pytest.raises(ValueError, match="paired coarse-score artifact is missing"):
        validator.validate_directory(tmp_path, (17,))

    _write_score(
        tmp_path / "part117_stack17.coarse-score-v1.bin",
        stack=17,
        selected_text="23",
        expected_particles=1,
    )
    with pytest.raises(ValueError, match="selected-stack hash"):
        validator.validate_directory(tmp_path, (17,))


def test_component_capture_rejects_truncation(tmp_path):
    path = _write_pair(tmp_path, stack=17, selected_text="17", expected_particles=1)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        validator.load_coarse_component_capture(path)
