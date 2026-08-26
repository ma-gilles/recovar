from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.analyze_vdam_native_bpref_prefix import analyze


def _write_flat(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values)
    path.write_bytes(np.asarray([values.size], dtype="<u8").tobytes() + values.tobytes())


def _write_prefix(
    directory: Path,
    *,
    part_id: int,
    original_index: int,
    data: np.ndarray,
    weight: np.ndarray,
) -> None:
    shape = data.shape
    prefix = directory / f"half0_part{part_id}_stack{original_index + 1}_bpref_prefix_"
    metadata = np.asarray(
        [1, 1, 0, part_id, original_index + 1, 0, 1, shape[2], shape[1], shape[0], 0, 0, 2, 4, data.size],
        dtype="<u8",
    )
    _write_flat(Path(str(prefix) + "metadata.bin"), metadata)
    _write_flat(Path(str(prefix) + "real.bin"), data.real.astype("<f4"))
    _write_flat(Path(str(prefix) + "imag.bin"), data.imag.astype("<f4"))
    _write_flat(Path(str(prefix) + "weight.bin"), weight.astype("<f4"))


def test_native_bpref_prefix_matches_sequential_recovar_contributions(tmp_path: Path) -> None:
    native_directory = tmp_path / "native"
    native_directory.mkdir()
    shape = (3, 3, 2)
    first_data = np.full(shape, 2.0 + 3.0j, dtype=np.complex64)
    second_data = np.full(shape, -0.5 + 1.0j, dtype=np.complex64)
    first_weight = np.full(shape, 4.0, dtype=np.float32)
    second_weight = np.full(shape, 0.25, dtype=np.float32)
    data_scale = np.float32(-(128.0**-2))
    weight_scale = np.float32(128.0**-4)
    _write_prefix(
        native_directory,
        part_id=1,
        original_index=9,
        data=(first_data / data_scale).astype(np.complex64),
        weight=(first_weight / weight_scale).astype(np.float32),
    )
    _write_prefix(
        native_directory,
        part_id=3,
        original_index=999,
        data=((first_data + second_data) / data_scale).astype(np.complex64),
        weight=((first_weight + second_weight) / weight_scale).astype(np.float32),
    )
    recovar_capture = tmp_path / "recovar.npz"
    np.savez(
        recovar_capture,
        inline_projector_original_indices=np.asarray([9, 999], dtype=np.int64),
        inline_projector_data_volumes=np.stack([first_data, second_data]),
        inline_projector_weight_volumes=np.stack([first_weight, second_weight]),
        image_shape=np.asarray([128, 128], dtype=np.int32),
    )

    report = analyze(native_directory, [recovar_capture])

    assert report["identity"]["particle_count"] == 2
    assert report["identity"]["first_native_part_id"] == 1
    assert report["identity"]["last_original_index"] == 999
    assert report["identity"]["unmatched_recovar_original_indices"] == []
    assert report["summary"]["final_prefix_data"]["relative_l2"] == 0.0
    assert report["summary"]["final_prefix_weight"]["relative_l2"] == 0.0
    assert report["summary"]["increment_data_relative_l2"]["max"] == 0.0
    assert report["summary"]["increment_weight_relative_l2"]["max"] == 0.0


def test_native_bpref_prefix_reports_one_particle_increment_error(tmp_path: Path) -> None:
    native_directory = tmp_path / "native"
    native_directory.mkdir()
    shape = (2, 2, 2)
    contribution = np.ones(shape, dtype=np.complex64)
    weight = np.ones(shape, dtype=np.float32)
    data_scale = np.float32(-(128.0**-2))
    weight_scale = np.float32(128.0**-4)
    _write_prefix(
        native_directory,
        part_id=1,
        original_index=9,
        data=(contribution / data_scale).astype(np.complex64),
        weight=(weight / weight_scale).astype(np.float32),
    )
    recovar_capture = tmp_path / "recovar.npz"
    np.savez(
        recovar_capture,
        inline_projector_original_indices=np.asarray([9], dtype=np.int64),
        inline_projector_data_volumes=np.asarray([contribution * np.complex64(2.0)]),
        inline_projector_weight_volumes=np.asarray([weight]),
        image_shape=np.asarray([128, 128], dtype=np.int32),
    )

    report = analyze(native_directory, [recovar_capture])

    assert report["summary"]["final_prefix_data"]["relative_l2"] == 1.0
    assert report["summary"]["final_prefix_weight"]["relative_l2"] == 0.0
