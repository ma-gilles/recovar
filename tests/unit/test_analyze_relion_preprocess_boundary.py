from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from scripts import analyze_relion_preprocess_boundary as analyzer
from scripts.validate_relion_preprocess_capture import RelionPreprocessCapture


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _capture(
    *,
    part_id: int,
    stack_index: int,
    raw: np.ndarray,
    normalized: np.ndarray,
    masked: np.ndarray,
    unmasked_fourier: np.ndarray,
    masked_fourier: np.ndarray,
) -> RelionPreprocessCapture:
    header = [0] * 40
    header[5] = 2
    header[6] = part_id
    header[7] = stack_index
    header[8] = 1
    header[20] = _bits(1.0)
    header[21] = _bits(0.0)
    header[22] = _bits(0.0)
    header[23] = _bits(0.0)
    header[24] = _bits(2.0)
    header[25] = _bits(3.0)
    header[26] = _bits(1.0)
    header[27] = _bits(0.25)
    return RelionPreprocessCapture(
        path=Path(f"/capture/part{part_id}.bin"),
        sha256=f"sha-{part_id}",
        header=tuple(header),
        raw_input_real=raw[np.newaxis],
        normalized_shifted_real=normalized[np.newaxis],
        unmasked_fourier_pre_optics=unmasked_fourier[np.newaxis],
        unmasked_fourier_post_optics=unmasked_fourier[np.newaxis],
        masked_real=masked[np.newaxis],
        masked_fourier_pre_optics=masked_fourier[np.newaxis],
        masked_fourier_post_optics=masked_fourier[np.newaxis],
    )


def _relion_layout(values: np.ndarray, full_size: int) -> np.ndarray:
    return np.fft.ifftshift(
        values / np.float32(full_size**2),
        axes=(-2,),
    ).astype(np.complex64)


def test_relion_fourier_mapping_restores_centered_units() -> None:
    centered = (
        np.arange(12, dtype=np.float32).reshape(1, 4, 3)
        + np.complex64(1j) * np.float32(2)
    ).astype(np.complex64)
    relion = _relion_layout(centered, 4)
    restored = analyzer.relion_fourier_to_recovar_centered(
        relion,
        full_image_size=4,
    )
    assert np.array_equal(restored, centered)


def test_relion_fourier_mapping_accepts_cropped_current_size() -> None:
    full_size = 8
    current_size = 4
    centered = (
        np.arange(12, dtype=np.float32).reshape(1, current_size, 3)
        + np.complex64(1j) * np.float32(2)
    ).astype(np.complex64)
    relion = np.fft.ifftshift(
        centered / np.float32(full_size**2),
        axes=(-2,),
    ).astype(np.complex64)
    restored = analyzer.relion_fourier_to_recovar_centered(
        relion,
        full_image_size=full_size,
    )
    assert np.array_equal(restored, centered)


def test_crop_centered_rfft_preserves_relion_positive_nyquist_row() -> None:
    full_size = 8
    values = np.arange(
        full_size * (full_size // 2 + 1),
        dtype=np.float32,
    ).reshape(1, full_size, full_size // 2 + 1)
    cropped = analyzer.crop_centered_rfft(values, current_size=4)
    assert np.array_equal(cropped, values[:, [6, 3, 4, 5], :3])


def test_scale_sensitive_metrics_do_not_accept_opposite_sign() -> None:
    target = np.ones((2, 4), dtype=np.float32)
    metrics = analyzer._per_particle_metrics(-target, target)
    assert metrics["relative_l2"] == [2.0, 2.0]
    assert metrics["material_relative_l2_count"] == 2
    assert metrics["bitwise_equal_count"] == 0


def test_classification_uses_earliest_fixed_material_boundary() -> None:
    stages = {}
    for name in (
        "normalized_shifted_real",
        "masked_real",
        "unmasked_fourier_pre_optics",
        "masked_fourier_pre_optics",
        "masked_fourier_post_optics",
    ):
        stages[name] = {
            "evaluated_particles": 2,
            "material_relative_l2_count": 0,
        }
    stages["masked_real"]["material_relative_l2_count"] = 1
    stages["masked_fourier_pre_optics"]["material_relative_l2_count"] = 2
    assert (
        analyzer.classify_earliest_boundary(stages, expected_particles=2)
        == "material_gap_begins_at_softmask"
    )


def test_classification_orders_unmasked_fft_before_softmask_branch() -> None:
    stages = {
        name: {
            "evaluated_particles": 2,
            "material_relative_l2_count": int(
                name in {"unmasked_fourier_pre_optics", "masked_real"}
            ),
        }
        for name in (
            "normalized_shifted_real",
            "masked_real",
            "unmasked_fourier_pre_optics",
            "masked_fourier_pre_optics",
            "masked_fourier_post_optics",
        )
    }
    assert (
        analyzer.classify_earliest_boundary(stages, expected_particles=2)
        == "material_gap_begins_at_unmasked_fft"
    )


def test_build_report_closes_exact_synthetic_boundaries(tmp_path: Path) -> None:
    full_size = 4
    raw = np.arange(2 * full_size * full_size, dtype=np.float32).reshape(
        2, full_size, full_size
    )
    normalized = raw + np.float32(1)
    masked = raw + np.float32(2)
    unmasked_fourier = (
        np.arange(2 * full_size * 3, dtype=np.float32).reshape(2, full_size, 3)
        + np.complex64(1j)
    ).astype(np.complex64)
    masked_fourier = (unmasked_fourier + np.complex64(3 + 2j)).astype(
        np.complex64
    )
    captures = tuple(
        _capture(
            part_id=10 + index,
            stack_index=100 + index,
            raw=raw[index],
            normalized=normalized[index],
            masked=masked[index],
            unmasked_fourier=_relion_layout(
                unmasked_fourier[index],
                full_size,
            ),
            masked_fourier=_relion_layout(
                masked_fourier[index],
                full_size,
            ),
        )
        for index in range(2)
    )
    particle_stack = tmp_path / "particles.mrcs"
    particle_stack.write_bytes(b"synthetic")
    report = analyzer.build_report_from_arrays(
        captures=captures,
        disk_raw=raw,
        strict_normalized=normalized,
        strict_masked=masked,
        strict_unmasked_fourier=unmasked_fourier,
        strict_masked_fourier=masked_fourier,
        repeat_normalized=(normalized, normalized, normalized),
        repeat_masked=(masked, masked, masked),
        capture_validation={"status": "pass"},
        particle_stack=particle_stack,
        gpu_uuid="GPU-test",
    )
    assert (
        report["classification"]
        == "all_preprocessing_boundaries_within_fixed_material_threshold"
    )
    assert report["fixed_metric"] == {
        "evaluated_particles": 2,
        "expected_particles": 2,
        "disk_raw_bitwise_equal": 2,
        "normalized_material_gap": 0,
        "masked_material_gap": 0,
        "unmasked_fft_material_gap": 0,
        "masked_fft_material_gap": 0,
        "masked_post_optics_material_gap": 0,
    }
