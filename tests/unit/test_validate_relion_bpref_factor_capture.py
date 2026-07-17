import json
import struct

import numpy as np
import pytest

from scripts import validate_relion_bpref_factor_capture as validator
from scripts.validate_relion_bpref_prescatter import ROTATION_DTYPE, ROW_DTYPE


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_capture(path, *, stack, selected_text="17,23", rank=2):
    rotations = np.zeros(2, dtype=ROTATION_DTYPE)
    rotations["orientation_class_key"] = [10, 11]
    rotations["oversampled_rotation"] = [1, 2]
    rotations["orientation_local"] = [0, 1]
    rotations["matrix"] = np.eye(3, dtype=np.float32).reshape(1, 9)
    translations = np.zeros(2, dtype=validator.TRANSLATION_DTYPE)
    translations["translation"] = [0, 1]
    translations["x"] = [0.0, 0.5]
    translations["y"] = [0.0, -0.5]
    hypotheses = np.zeros(4, dtype=validator.HYPOTHESIS_DTYPE)
    hypotheses["orientation_local"] = [0, 0, 1, 1]
    hypotheses["translation"] = [0, 1, 0, 1]
    hypotheses["posterior"] = [0.2, 1.0, 0.1, 0.3]
    hypotheses["posterior_over_weight_norm"] = hypotheses["posterior"]
    hypotheses["flags"][1] = 1
    pixels = np.zeros(3, dtype=validator.PIXEL_DTYPE)
    pixels["pixel"] = [0, 1, 2]
    pixels["x"] = [0, 1, 2]
    pixels["image_re"] = [1, 2, 3]
    pixels["ctf"] = 1
    pixels["minvsigma2"] = 1
    summaries = np.zeros(2, dtype=ROW_DTYPE)
    summaries["state"] = 1
    summaries["orientation_local"] = 0
    summaries["pixel"] = [0, 1]
    summaries["flags"] = 3
    summaries["x"] = [0, 1]
    summaries["source_re"] = [1, 2]
    summaries["source_weight"] = 1
    terms = np.zeros(3, dtype=validator.TERM_DTYPE)
    terms["state"] = 1
    terms["orientation_local"] = 0
    terms["translation"] = 1
    terms["pixel"] = [0, 1, 2]
    terms["flags"] = 1
    terms["translated_re"] = [1, 2, 3]
    terms["posterior_over_weight_norm"] = 1
    terms["weighted_ctf"] = 1
    terms["term_re"] = [1, 2, 3]
    terms["weight_term"] = 1

    header = [0] * 64
    header[:9] = [2, 528, 64, 24, 24, 40, 40, 56, 64]
    header[9:16] = [1, 1, stack + 100, stack, 0, rank, 0]
    header[16:22] = [3, 1, 1, 3, 2, 2]
    header[22:29] = [1, 1, _bits(2.0), _bits(0.999), _bits(1.0), 0, 0]
    header[29:37] = [2, 2, 2, 10_000_000, 1_000_000, 1, 2, validator.fnv1a64(selected_text)]
    header[37:43] = [5, 9, 9, (-4) & 0xFFFFFFFFFFFFFFFF, (-4) & 0xFFFFFFFFFFFFFFFF, 1]
    header[43:53] = [3, 1, 1, 2, 2, 4, 3, 2, 3, 1]
    magic = validator.HEADER_MAGIC
    footer = validator.FOOTER_STRUCT.pack(validator.FOOTER_MAGIC, 2, 2, 4, 3, 2, 3)
    payload = validator.HEADER_STRUCT.pack(magic, *header)
    payload += rotations.tobytes() + translations.tobytes() + hypotheses.tobytes()
    payload += pixels.tobytes() + summaries.tobytes() + terms.tobytes() + footer
    path.write_bytes(payload)


def _selection(path):
    path.write_text(
        json.dumps(
            {
                "schema": "bpref-factor-stratification-v1",
                "selected": [
                    {"stack_index_1based": 17},
                    {"stack_index_1based": 23},
                ],
            }
        )
    )


def test_factor_capture_directory_is_complete_and_hash_bound(tmp_path):
    selection = tmp_path / "selection.json"
    _selection(selection)
    _write_capture(tmp_path / "part117_stack17_img0_class1.bpre-v2.bin", stack=17)
    _write_capture(tmp_path / "part123_stack23_img0_class1.bpre-v2.bin", stack=23)

    report = validator.validate_directory(tmp_path, selection, expected_rank=2)

    assert report["capture_ready"] is True
    assert report["particle_count"] == 2
    assert report["accepted_hypotheses_per_particle"] == [1, 1]


def test_factor_capture_rejects_missing_selected_stack_and_truncation(tmp_path):
    selection = tmp_path / "selection.json"
    _selection(selection)
    first = tmp_path / "part117_stack17_img0_class1.bpre-v2.bin"
    _write_capture(first, stack=17)
    with pytest.raises(ValueError, match="file count"):
        validator.validate_directory(tmp_path, selection, expected_rank=2)

    second = tmp_path / "part123_stack23_img0_class1.bpre-v2.bin"
    _write_capture(second, stack=23)
    second.write_bytes(second.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        validator.validate_directory(tmp_path, selection, expected_rank=2)
