import json
import struct

import numpy as np
import pytest

from scripts import validate_relion_firstiter_cc_capture as validator


def _bits(value: np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_capture(path, *, pass_index: int, stack: int = 17, corrupt_winner_bits: bool = False):
    if pass_index == 0:
        raw_scores = np.asarray([4.0, 3.0, 5.0, 4.5, 2.0, 1.0, 3.5, 2.5], dtype=np.float32)
        class_ids = np.asarray([1] * 4 + [2] * 4, dtype=np.uint32)
        rotation_ids = np.asarray([0, 0, 1, 1] * 2, dtype=np.uint64)
        translation_ids = np.asarray([0, 1, 0, 1] * 2, dtype=np.uint64)
        hidden = np.arange(8, dtype=np.uint64)
        base_flag = validator.DENSE_COARSE
    else:
        raw_scores = np.asarray([2.0, 0.5, 1.5], dtype=np.float32)
        class_ids = np.asarray([2, 2, 2], dtype=np.uint32)
        rotation_ids = np.asarray([4, 5, 6], dtype=np.uint64)
        translation_ids = np.asarray([0, 1, 0], dtype=np.uint64)
        hidden = np.asarray([40, 41, 42], dtype=np.uint64)
        base_flag = validator.SPARSE_FINE
    winner_index = int(np.argmin(raw_scores))
    candidates = np.zeros(raw_scores.size, dtype=validator.CANDIDATE_DTYPE)
    candidates["candidate_index"] = np.arange(raw_scores.size)
    candidates["ihidden_overs"] = hidden
    candidates["rotation_id"] = rotation_ids
    candidates["rotation_local"] = rotation_ids
    candidates["translation_id"] = translation_ids
    candidates["class_one_based"] = class_ids
    candidates["flags"] = base_flag
    candidates["flags"][winner_index] |= validator.WINNER
    candidates["raw_score"] = raw_scores

    header = [0] * 64
    header[:4] = [1, validator.HEADER_STRUCT.size, validator.CANDIDATE_DTYPE.itemsize, validator.FOOTER_STRUCT.size]
    header[4:10] = [1, pass_index, 7, stack, 0, 3]
    header[10:18] = [2, 0, 1, 2, 1, 2, 1, 1]
    header[18:21] = [raw_scores.size, winner_index, _bits(raw_scores[winner_index])]
    if corrupt_winner_bits:
        header[20] ^= 1
    file_bytes = validator.HEADER_STRUCT.size + candidates.nbytes + validator.FOOTER_STRUCT.size
    header[21:29] = [1, 1, 1, 1_000_000, file_bytes, 11, 12, 13]
    header[29:32] = [1, 1, 1]
    header[32:34] = [int(pass_index == 0), int(pass_index == 1)]
    footer = validator.FOOTER_STRUCT.pack(
        validator.FOOTER_MAGIC,
        raw_scores.size,
        winner_index,
        header[20],
    )
    path.write_bytes(
        validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header)
        + candidates.tobytes()
        + footer
    )


def _selection(path):
    path.write_text(
        json.dumps(
            {
                "schema": "bpref-factor-stratification-v1",
                "selected": [{"stack_index_1based": 17}],
            }
        )
    )


def test_firstiter_cc_capture_validates_both_passes(tmp_path):
    selection = tmp_path / "selection.json"
    _selection(selection)
    _write_capture(tmp_path / "part7_stack17_pass0.firstiter-cc-v1.bin", pass_index=0)
    _write_capture(tmp_path / "part7_stack17_pass1.firstiter-cc-v1.bin", pass_index=1)

    report = validator.validate_directory(tmp_path, selection, expected_rank=0)

    assert report["capture_ready"] is True
    assert report["particle_count"] == 1
    assert report["file_count"] == 2
    assert report["coarse_candidate_count"] == 8
    assert report["fine_candidate_count"] == 3
    assert report["rows"][0]["winner_class_one_based"] == 2
    assert report["rows"][0]["coarse_runner_up_margin"] == 1.0


def test_firstiter_cc_capture_rejects_changed_winner_bits(tmp_path):
    path = tmp_path / "part7_stack17_pass0.firstiter-cc-v1.bin"
    _write_capture(path, pass_index=0, corrupt_winner_bits=True)

    with pytest.raises(ValueError, match="winner score bits"):
        validator.load_firstiter_cc_capture(path)


def test_firstiter_cc_capture_rejects_truncation(tmp_path):
    path = tmp_path / "part7_stack17_pass1.firstiter-cc-v1.bin"
    _write_capture(path, pass_index=1)
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="byte count"):
        validator.load_firstiter_cc_capture(path)


def test_firstiter_cc_capture_requires_fine_support_in_coarse_winner_class(tmp_path):
    selection = tmp_path / "selection.json"
    _selection(selection)
    coarse = tmp_path / "part7_stack17_pass0.firstiter-cc-v1.bin"
    fine = tmp_path / "part7_stack17_pass1.firstiter-cc-v1.bin"
    _write_capture(coarse, pass_index=0)
    _write_capture(fine, pass_index=1)
    payload = bytearray(fine.read_bytes())
    first_class_offset = validator.HEADER_STRUCT.size + validator.CANDIDATE_DTYPE.fields["class_one_based"][1]
    struct.pack_into("<I", payload, first_class_offset, 1)
    fine.write_bytes(payload)

    with pytest.raises(ValueError, match="fine support escaped"):
        validator.validate_directory(tmp_path, selection)
