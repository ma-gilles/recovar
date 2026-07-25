import json
import struct

import numpy as np
import pytest

from scripts import validate_relion_fine_score_capture as validator


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_capture(path, *, stack, rank=2, selected_text="17,23", corrupt_algebra=False):
    raw_diff2 = np.asarray([10.0, 150.25, 9.0], dtype=np.float32)
    orientation_prior = np.asarray([-0.5, -0.5, -0.5], dtype=np.float32)
    translation_prior = np.asarray([-0.25, -0.75, -0.25], dtype=np.float32)
    min_diff2 = np.float32(9.5)
    combined = orientation_prior + translation_prior + min_diff2 - raw_diff2
    combined[2] = -np.inf
    weights_max = np.float32(-1.0)
    shift = np.float32(50.0) - weights_max
    shifted = combined + shift
    post = np.exp(shifted, dtype=np.float32)
    post[shifted < np.float32(-88.0)] = 0
    post[2] = 0
    if corrupt_algebra:
        combined[0] += np.float32(0.01)

    candidates = np.zeros(3, dtype=validator.CANDIDATE_DTYPE)
    candidates["sparse_index"] = np.arange(3)
    candidates["rotation_id"] = [0, 1, 2]
    candidates["rotation_local"] = [0, 1, 2]
    candidates["translation_id"] = [0, 2, 0]
    candidates["coarse_translation"] = [0, 1, 0]
    candidates["flags"] = [validator.ACTIVE, validator.ACTIVE, validator.DIFF2_BELOW_MIN]
    candidates["raw_diff2"] = raw_diff2
    candidates["orientation_log_prior"] = orientation_prior
    candidates["translation_log_prior"] = translation_prior
    candidates["combined_preexponent"] = combined
    candidates["shifted_log_weight"] = shifted
    candidates["post_exponent_weight"] = post

    header = [0] * 48
    header[:4] = [1, 400, 64, 32]
    header[4:10] = [10, 2, stack + 100, stack, rank, 0]
    header[10:18] = [8, 2, 3, 1, 2, 2, 3, 2]
    header[18:21] = [_bits(min_diff2), _bits(weights_max), _bits(shift)]
    header[21:29] = [
        2,
        2,
        2,
        1_000_000,
        400 + 3 * 64 + 32,
        1,
        2,
        validator.fnv1a64(selected_text),
    ]
    header[29:32] = [1, 1, 1]
    footer = validator.FOOTER_STRUCT.pack(validator.FOOTER_MAGIC, 3, 2)
    payload = validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header)
    path.write_bytes(payload + candidates.tobytes() + footer)


def _selection(path, *, ranks=(2, 2)):
    path.write_text(
        json.dumps(
            {
                "schema": "bpref-factor-stratification-v1",
                "selected": [
                    {"stack_index_1based": 17, "expected_mpi_rank": ranks[0]},
                    {"stack_index_1based": 23, "expected_mpi_rank": ranks[1]},
                ],
            }
        )
    )


def test_fine_score_capture_directory_is_complete_and_algebra_checked(tmp_path):
    selection = tmp_path / "selection.json"
    _selection(selection)
    _write_capture(tmp_path / "part117_stack17_class2.fine-score-v1.bin", stack=17)
    _write_capture(tmp_path / "part123_stack23_class2.fine-score-v1.bin", stack=23)

    report = validator.validate_directory(tmp_path, selection, expected_rank=2)

    assert report["capture_ready"] is True
    assert report["particle_count"] == 2
    assert report["candidate_count"] == 6
    assert report["active_candidate_count"] == 4
    assert report["iteration"] == 10
    assert report["class_one_based"] == 2
    assert report["algebra_max_abs"] == 0
    assert report["shift_max_abs"] == 0
    assert report["exponent_max_rel"] == 0
    assert report["underflow_candidate_count"] == 2


def test_fine_score_capture_uses_per_particle_mpi_ranks(tmp_path):
    selection = tmp_path / "selection.json"
    _selection(selection, ranks=(1, 2))
    _write_capture(tmp_path / "part117_stack17_class2.fine-score-v1.bin", stack=17, rank=1)
    _write_capture(tmp_path / "part123_stack23_class2.fine-score-v1.bin", stack=23, rank=2)

    report = validator.validate_directory(tmp_path, selection)

    assert report["mpi_rank"] is None
    assert report["mpi_rank_counts"] == {"1": 1, "2": 1}
    assert report["mpi_rank_by_stack"] == {"17": 1, "23": 2}


def test_fine_score_capture_rejects_score_algebra_drift(tmp_path):
    capture = tmp_path / "part117_stack17_class2.fine-score-v1.bin"
    _write_capture(capture, stack=17, corrupt_algebra=True)

    with pytest.raises(ValueError, match="prior/diff2 algebra"):
        validator.load_fine_score_capture(capture)


def test_fine_score_capture_rejects_wrong_rank_and_truncation(tmp_path):
    selection = tmp_path / "selection.json"
    _selection(selection, ranks=(1, 2))
    first = tmp_path / "part117_stack17_class2.fine-score-v1.bin"
    second = tmp_path / "part123_stack23_class2.fine-score-v1.bin"
    _write_capture(first, stack=17, rank=2)
    _write_capture(second, stack=23, rank=2)

    with pytest.raises(ValueError, match="MPI rank changed"):
        validator.validate_directory(tmp_path, selection)

    second.write_bytes(second.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        validator.load_fine_score_capture(second)
