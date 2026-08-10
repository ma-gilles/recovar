import json
import struct

import numpy as np
import pytest

from scripts import validate_relion_bpref_factor_capture as validator
from scripts.validate_relion_bpref_prescatter import ROTATION_DTYPE, ROW_DTYPE


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_capture(
    path,
    *,
    stack,
    selected_text="17,23",
    rank=2,
    orientation_count=2,
    significant_weight=0.999,
    weight_norm=1.0,
    accepted=True,
    geometry_only=False,
):
    rotations = np.zeros(orientation_count, dtype=ROTATION_DTYPE)
    rotations["orientation_class_key"] = np.arange(10, 10 + orientation_count)
    rotations["oversampled_rotation"] = np.arange(1, 1 + orientation_count)
    rotations["orientation_local"] = np.arange(orientation_count)
    rotations["matrix"] = np.eye(3, dtype=np.float32).reshape(1, 9)
    translations = np.zeros(2, dtype=validator.TRANSLATION_DTYPE)
    translations["translation"] = [0, 1]
    translations["x"] = [0.0, 0.5]
    translations["y"] = [0.0, -0.5]
    hypotheses = np.zeros(orientation_count * 2, dtype=validator.HYPOTHESIS_DTYPE)
    hypotheses["orientation_local"] = np.repeat(np.arange(orientation_count), 2)
    hypotheses["translation"] = np.tile([0, 1], orientation_count)
    hypotheses["posterior"] = 0.1
    if accepted:
        hypotheses["posterior"][1] = 1.0
    hypotheses["posterior_over_weight_norm"] = hypotheses["posterior"] / weight_norm
    if accepted:
        hypotheses["flags"][1] = 1
    pixels = np.zeros(12, dtype=validator.PIXEL_DTYPE)
    pixels["pixel"] = np.arange(12)
    pixels["x"] = np.tile(np.arange(3), 4)
    pixels["y"] = np.repeat([0, 1, 2, -1], 3)
    pixels["image_re"] = np.arange(1, 13)
    pixels["ctf"] = 1
    pixels["minvsigma2"] = 1
    summaries = np.zeros(2 if accepted else 0, dtype=ROW_DTYPE)
    terms = np.zeros(12 if accepted else 0, dtype=validator.TERM_DTYPE)
    if accepted:
        summaries["state"] = 1
        summaries["orientation_local"] = 0
        summaries["pixel"] = [0, 1]
        summaries["flags"] = 3
        summaries["x"] = [0, 1]
        summaries["source_re"] = [1, 2]
        summaries["source_weight"] = 1
        terms["state"] = 1
        terms["orientation_local"] = 0
        terms["translation"] = 1
        terms["pixel"] = np.arange(12)
        terms["flags"] = 1
        terms["translated_re"] = np.arange(1, 13)
        terms["posterior_over_weight_norm"] = 1
        terms["weighted_ctf"] = 1
        terms["term_re"] = np.arange(1, 13)
        terms["weight_term"] = 1
    if geometry_only:
        hypotheses = hypotheses[:0]
        pixels = pixels[:0]
        summaries = summaries[:0]
        terms = terms[:0]

    header = [0] * 64
    header[:9] = [2, 528, 64, 24, 24, 40, 40, 56, 64]
    header[9:16] = [1, 1, stack + 100, stack, 0, rank, 0]
    header[16:22] = [3, 4, 1, 12, orientation_count, 2]
    header[22:29] = [1, 1, _bits(2.0), _bits(significant_weight), _bits(weight_norm), 0, 0]
    header[29:37] = [
        2,
        2,
        2,
        10_000_000,
        900_000 + 50_000 * orientation_count,
        1,
        2,
        validator.fnv1a64(selected_text),
    ]
    header[37:43] = [5, 9, 9, (-4) & 0xFFFFFFFFFFFFFFFF, (-4) & 0xFFFFFFFFFFFFFFFF, 1]
    header[43:53] = [
        0 if geometry_only else 3 if accepted else 0,
        0 if geometry_only else 1 if accepted else 0,
        1 if accepted else 0,
        orientation_count,
        2,
        0 if geometry_only else orientation_count * 2,
        0 if geometry_only else 12,
        summaries.size,
        terms.size,
        0 if geometry_only else 1,
    ]
    header[53] = int(geometry_only)
    magic = validator.HEADER_MAGIC
    footer = validator.FOOTER_STRUCT.pack(
        validator.FOOTER_MAGIC,
        orientation_count,
        2,
        hypotheses.size,
        pixels.size,
        summaries.size,
        terms.size,
    )
    payload = validator.HEADER_STRUCT.pack(magic, *header)
    payload += rotations.tobytes() + translations.tobytes() + hypotheses.tobytes()
    payload += pixels.tobytes() + summaries.tobytes() + terms.tobytes() + footer
    path.write_bytes(payload)


def _selection(path, *, ranks=None):
    if ranks is None:
        ranks = (2, 2)
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


def _k1_selection(path, *, ranks=(2, 2), schema="recovar.em.k1_bpref_factor_panel.v1"):
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "targets": [
                    {"stack_index_one_based": 17, "expected_mpi_rank": ranks[0]},
                    {"stack_index_one_based": 23, "expected_mpi_rank": ranks[1]},
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
    assert report["mpi_rank"] == 2
    assert report["mpi_rank_counts"] == {"2": 2}
    assert report["mpi_rank_by_stack"] == {"17": 2, "23": 2}


def test_factor_capture_directory_uses_per_particle_mpi_ranks(tmp_path):
    selection = tmp_path / "selection.json"
    _selection(selection, ranks=(1, 2))
    _write_capture(tmp_path / "part117_stack17_img0_class1.bpre-v2.bin", stack=17, rank=1)
    _write_capture(tmp_path / "part123_stack23_img0_class1.bpre-v2.bin", stack=23, rank=2)

    report = validator.validate_directory(tmp_path, selection)

    assert report["capture_ready"] is True
    assert report["mpi_rank"] is None
    assert report["mpi_rank_counts"] == {"1": 1, "2": 1}
    assert report["mpi_rank_by_stack"] == {"17": 1, "23": 2}


def test_factor_capture_directory_accepts_k1_boundary_panel_schema(tmp_path):
    selection = tmp_path / "selection.json"
    _k1_selection(selection, ranks=(1, 2))
    _write_capture(tmp_path / "part117_stack17_img0_class1.bpre-v2.bin", stack=17, rank=1)
    _write_capture(tmp_path / "part123_stack23_img0_class1.bpre-v2.bin", stack=23, rank=2)

    report = validator.validate_directory(tmp_path, selection)

    assert report["capture_ready"] is True
    assert report["mpi_rank_by_stack"] == {"17": 1, "23": 2}


def test_factor_capture_directory_accepts_k1_fine_score_panel_schema(tmp_path):
    selection = tmp_path / "selection.json"
    _k1_selection(selection, schema="recovar.em.k1_fine_score_panel.v1")
    _write_capture(tmp_path / "part117_stack17_img0_class1.bpre-v2.bin", stack=17)
    _write_capture(tmp_path / "part123_stack23_img0_class1.bpre-v2.bin", stack=23)

    report = validator.validate_directory(tmp_path, selection, expected_rank=2)

    assert report["capture_ready"] is True


def test_factor_capture_accepts_explicit_zero_accepted_hypotheses(tmp_path):
    """Geometry remains valid when one selected class has no accepted pose."""

    selection = tmp_path / "selection.json"
    _selection(selection)
    _write_capture(
        tmp_path / "part117_stack17_img0_class1.bpre-v2.bin",
        stack=17,
        accepted=False,
    )
    _write_capture(tmp_path / "part123_stack23_img0_class1.bpre-v2.bin", stack=23)

    report = validator.validate_directory(tmp_path, selection, expected_rank=2)

    assert report["capture_ready"] is True
    assert report["accepted_hypotheses_per_particle"] == [0, 1]
    capture = validator.load_factor_capture(
        tmp_path / "part117_stack17_img0_class1.bpre-v2.bin"
    )
    assert capture.rotations.size == 2
    assert capture.translations.size == 2
    assert capture.summaries.size == 0
    assert capture.terms.size == 0


def test_factor_capture_accepts_explicit_geometry_only_panel(tmp_path):
    selection = tmp_path / "selection.json"
    _selection(selection)
    _write_capture(
        tmp_path / "part117_stack17_img0_class1.bpre-v2.bin",
        stack=17,
        geometry_only=True,
    )
    _write_capture(
        tmp_path / "part123_stack23_img0_class1.bpre-v2.bin",
        stack=23,
        accepted=False,
        geometry_only=True,
    )

    report = validator.validate_directory(tmp_path, selection, expected_rank=2)

    assert report["capture_ready"] is True
    assert report["accepted_hypotheses_per_particle"] == [1, 0]
    capture = validator.load_factor_capture(
        tmp_path / "part117_stack17_img0_class1.bpre-v2.bin"
    )
    assert capture.geometry_only is True
    assert capture.rotations.size == 2
    assert capture.translations.size == 2
    assert capture.hypotheses.size == 0
    assert capture.pixels.size == 0
    assert capture.summaries.size == 0
    assert capture.terms.size == 0


def test_factor_capture_directory_rejects_particle_on_wrong_mpi_rank(tmp_path):
    selection = tmp_path / "selection.json"
    _selection(selection, ranks=(1, 2))
    _write_capture(tmp_path / "part117_stack17_img0_class1.bpre-v2.bin", stack=17, rank=2)
    _write_capture(tmp_path / "part123_stack23_img0_class1.bpre-v2.bin", stack=23, rank=2)

    with pytest.raises(ValueError, match="MPI rank changed"):
        validator.validate_directory(tmp_path, selection)


def test_factor_capture_allows_particle_local_fine_support_and_normalization(tmp_path):
    selection = tmp_path / "selection.json"
    _selection(selection)
    _write_capture(tmp_path / "part117_stack17_img0_class1.bpre-v2.bin", stack=17)
    _write_capture(
        tmp_path / "part123_stack23_img0_class1.bpre-v2.bin",
        stack=23,
        orientation_count=1,
        significant_weight=0.5,
        weight_norm=2.0,
    )

    report = validator.validate_directory(tmp_path, selection, expected_rank=2)

    assert report["capture_ready"] is True
    assert report["orientation_count"] is None
    assert report["orientation_counts_per_particle"] == [2, 1]
    assert report["translation_count"] == 2
    assert report["translation_counts_per_particle"] == [2, 2]


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
