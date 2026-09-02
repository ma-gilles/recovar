from __future__ import annotations

import struct

import numpy as np
import pytest

from scripts import validate_relion_coarse_score_capture as validator


def _bits(value: np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_capture(
    path,
    *,
    stack: int,
    selected_text: str = "17,23",
    corrupt_algebra: bool = False,
    corrupt_support: bool = False,
):
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
    flags[1::2] |= validator.TRANSLATION_ZERO
    flags[4:6] |= validator.ORIENTATION_ZERO
    active = (flags & validator.REJECTION_FLAGS) == 0
    flags[active] |= validator.ACTIVE

    combined = np.full(candidate_count, -np.inf, dtype=np.float32)
    combined[active] = (
        orientation_prior[active]
        + translation_prior[active]
        + min_diff2
        - raw[active]
    )
    if corrupt_algebra:
        combined[0] += np.float32(0.01)
    weights_max = np.max(combined[active])
    shift = np.float32(50.0) - weights_max
    shifted = combined + shift
    post = np.zeros(candidate_count, dtype=np.float32)
    post[active] = np.exp(shifted[active], dtype=np.float32)
    active_order = np.flatnonzero(active)[np.argsort(-post[active], kind="stable")]
    significant_weight = post[active_order[1]]
    significant = post >= significant_weight
    if corrupt_support:
        significant[active_order[-1]] = True
    flags[significant] |= validator.SIGNIFICANT

    candidates = np.zeros(candidate_count, dtype=validator.CANDIDATE_DTYPE)
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
    header[:4] = [1, validator.HEADER_STRUCT.size, validator.CANDIDATE_DTYPE.itemsize, 40]
    header[4:15] = [1, stack + 100, stack, 2, 0, 0, class_count, nr_dir, nr_psi, nr_trans, candidate_count]
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
    header[28:32] = [1, 2, validator.fnv1a64(selected_text), 2]
    expected_bytes = validator.HEADER_STRUCT.size + candidates.nbytes + validator.FOOTER_STRUCT.size
    header[32:41] = [expected_bytes, 1_000_000, 4, 1, 0, 3, 1, 1, 1]
    footer = validator.FOOTER_STRUCT.pack(
        validator.FOOTER_MAGIC,
        candidate_count,
        int(np.count_nonzero(flags & validator.ACTIVE)),
        int(np.count_nonzero(significant)),
    )
    path.write_bytes(
        validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header)
        + candidates.tobytes()
        + footer
    )


def test_coarse_score_capture_round_trips_complete_score_boundary(tmp_path):
    path = tmp_path / "part117_stack17.coarse-score-v1.bin"
    _write_capture(path, stack=17)

    capture = validator.load_coarse_score_capture(path)

    assert capture.stack_index == 17
    assert capture.candidates.size == 8
    assert np.count_nonzero(capture.candidates["flags"] & validator.ACTIVE) == 3
    assert np.count_nonzero(capture.candidates["flags"] & validator.SIGNIFICANT) == 2
    assert capture.algebra_max_abs == 0
    assert capture.shift_max_abs == 0
    assert capture.exponent_max_rel < 3e-6


def test_coarse_score_capture_rejects_algebra_and_support_drift(tmp_path):
    algebra = tmp_path / "part117_stack17.coarse-score-v1.bin"
    _write_capture(algebra, stack=17, corrupt_algebra=True)
    with pytest.raises(ValueError, match="prior/diff2 algebra"):
        validator.load_coarse_score_capture(algebra)

    support = tmp_path / "part123_stack23.coarse-score-v1.bin"
    _write_capture(support, stack=23, corrupt_support=True)
    with pytest.raises(ValueError, match="threshold support"):
        validator.load_coarse_score_capture(support)


def test_coarse_score_directory_requires_exact_hash_bound_stack_panel(tmp_path):
    _write_capture(tmp_path / "part117_stack17.coarse-score-v1.bin", stack=17)
    _write_capture(tmp_path / "part123_stack23.coarse-score-v1.bin", stack=23)

    report = validator.validate_directory(tmp_path, (17, 23), expected_iteration=1)

    assert report["capture_ready"] is True
    assert report["particle_count"] == 2
    assert report["candidate_count"] == 16
    assert report["significant_candidate_count"] == 4

    with pytest.raises(ValueError, match="stack set"):
        validator.validate_directory(tmp_path, (17, 24))


def test_coarse_score_capture_rejects_truncation_and_unknown_flags(tmp_path):
    path = tmp_path / "part117_stack17.coarse-score-v1.bin"
    _write_capture(path, stack=17)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        validator.load_coarse_score_capture(path)

    _write_capture(path, stack=17)
    payload = bytearray(path.read_bytes())
    flags_offset = validator.HEADER_STRUCT.size + 24
    original_flags = struct.unpack_from("<I", payload, flags_offset)[0]
    struct.pack_into("<I", payload, flags_offset, original_flags | (1 << 31))
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="unknown coarse-score"):
        validator.load_coarse_score_capture(path)
