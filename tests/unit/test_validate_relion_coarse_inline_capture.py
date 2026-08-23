from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import validate_relion_coarse_inline_capture as validator


def _fixture(tmp_path: Path):
    image_size = 8
    translation_count = 2
    block_size = 8
    prefetch_fraction = 2
    rotation_key = 3
    reference_real = np.linspace(0.25, 2.0, image_size, dtype=np.float32)
    reference_imag = np.linspace(-1.0, 0.75, image_size, dtype=np.float32)
    shifted_real = np.stack((reference_real * 0.5, reference_real * -0.25)).astype(np.float32)
    shifted_imag = np.stack((reference_imag * 0.75, reference_imag * -0.5)).astype(np.float32)
    correction = np.linspace(0.5, 1.25, image_size, dtype=np.float32)
    fields = np.empty((len(validator.FIELD_NAMES), translation_count, image_size), dtype=np.float32)
    fields[0] = reference_real
    fields[1] = reference_imag
    fields[2] = shifted_real
    fields[3] = shifted_imag
    fields[4] = correction / np.float32(2.0)
    fields[5] = np.float32(fields[0] - fields[2])
    fields[6] = np.float32(fields[1] - fields[3])
    fields[7:] = np.float32(0.0)
    lane_partials = np.zeros((1, block_size), dtype=np.float32)
    stride = block_size // translation_count
    for translation in range(translation_count):
        for lane_group in range(stride):
            total = np.float32(0.0)
            for pixel in validator._lane_pixel_order(
                image_size,
                block_size,
                prefetch_fraction,
                lane_group,
                stride,
            ):
                fields[7, translation, pixel] = total
                square_real = np.float32(fields[5, translation, pixel] ** 2)
                square_imag = np.float32(fields[6, translation, pixel] ** 2)
                term = np.float32(np.float32(square_real + square_imag) * fields[4, translation, pixel])
                total = np.float32(total + term)
                fields[8, translation, pixel] = total
            lane_partials[0, translation + lane_group * translation_count] = total
    inline_header = [0] * 40
    inline_header[5] = 2
    inline_header[6] = 17
    inline_header[7] = 23
    inline_header[13] = image_size
    inline_header[14] = translation_count
    inline_header[15] = rotation_key
    inline_header[16] = rotation_key
    inline_header[18] = block_size
    inline_header[20] = prefetch_fraction
    inline = validator.CoarseInlineCapture(
        path=tmp_path / "part17_stack23.p1-inline-v1.bin",
        sha256="inline",
        header=tuple(inline_header),
        fields=fields,
    )
    operand_header = [0] * 40
    operand_header[5] = 2
    operand_header[13] = image_size
    operand_header[14] = translation_count
    operand = SimpleNamespace(
        part_id=17,
        stack_index=23,
        header=tuple(operand_header),
        rotation_keys=np.asarray([rotation_key], dtype=np.uint64),
        local_rotation_indices=np.asarray([rotation_key], dtype=np.uint64),
        reference_real=reference_real[None],
        reference_imag=reference_imag[None],
        shifted_real=shifted_real,
        shifted_imag=shifted_imag,
        correction=correction,
    )
    lane_header = [0] * 32
    lane_header[5] = 2
    lane_header[13] = image_size
    lane_header[14] = translation_count
    lane = SimpleNamespace(
        part_id=17,
        stack_index=23,
        header=tuple(lane_header),
        rotation_keys=np.asarray([rotation_key], dtype=np.uint64),
        lane_partials=lane_partials,
    )
    return inline, operand, lane


def test_inline_capture_closes_native_lane_trajectory(tmp_path):
    inline, operand, lane = _fixture(tmp_path)
    report = validator.validate_capture(inline, operand, lane)

    assert report["status"] == "pass"
    assert report["classification"] == "native_inline_operands_and_separate_float32_replay_are_exact"
    assert report["native_accumulation"]["accumulator_continuity"]["exact"]
    assert report["native_accumulation"]["final_lane_partial"]["exact"]


def test_inline_capture_localizes_projection_difference(tmp_path):
    inline, operand, lane = _fixture(tmp_path)
    operand.reference_real = operand.reference_real.copy()
    operand.reference_real[0, 0] = np.nextafter(operand.reference_real[0, 0], np.float32(np.inf))

    report = validator.validate_capture(inline, operand, lane)

    assert report["status"] == "pass"
    assert report["classification"] == "native_projection_differs_from_passive_projection_capture"
    assert not report["passive_operand_comparisons"]["projection_real_vs_passive"]["exact"]
