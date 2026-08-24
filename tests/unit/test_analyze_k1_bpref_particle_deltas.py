from __future__ import annotations

import json
import sys

import numpy as np

from scripts import analyze_k1_bpref_particle_deltas as analyzer


def _write_counted(path, values):
    values = np.asarray(values)
    with path.open("wb") as stream:
        np.asarray([values.size], dtype=np.uint64).tofile(stream)
        values.tofile(stream)


def _write_native_particle(root, *, rank, part_id, stack_index, data, weight):
    stem = (
        f"rank{rank}_thread0_part{part_id}_stack{stack_index}_class1_bpref_"
    )
    metadata = np.asarray(
        [1, 1, part_id, stack_index, rank, 0, 1, 2, 3, 3, -1, -1, 1, 1, 18, 0, 0, 0],
        dtype=np.int64,
    ).view(np.uint64)
    _write_counted(root / f"{stem}shadow_real.bin", np.asarray(data.real, np.float32))
    _write_counted(root / f"{stem}shadow_imag.bin", np.asarray(data.imag, np.float32))
    _write_counted(root / f"{stem}shadow_weight.bin", np.asarray(weight, np.float32))
    _write_counted(root / f"{stem}shadow_metadata.bin", metadata)


def _write_recovar_particle(root, *, half, original_index, data, weight):
    np.savez(
        root / f"bpref_accumulator_delta_it001_h{half}_orig{original_index:06d}.npz",
        schema=np.asarray("recovar-bpref-accumulator-delta-v2"),
        iteration=np.int64(1),
        half=np.int64(half),
        original_index=np.int64(original_index),
        stack_index_1based=np.int64(original_index + 1),
        image_identity=np.asarray(f"{original_index + 1}@stack.mrcs"),
        particle_launch_ordinal=np.int64(0),
        particle_rotation_count=np.int64(1),
        volume_shape=np.asarray([3, 3, 3], dtype=np.int64),
        flat_accumulator_size=np.int64(data.size),
        max_r=np.float64(1),
        before_data=np.zeros(data.size, dtype=np.complex64),
        before_weight=np.zeros(weight.size, dtype=np.float32),
        after_data=np.asarray(data, dtype=np.complex64).reshape(-1),
        after_weight=np.asarray(weight, dtype=np.float32).reshape(-1),
        isolated_data=np.asarray(data, dtype=np.complex64).reshape(-1),
        isolated_weight=np.asarray(weight, dtype=np.float32).reshape(-1),
    )


def test_particle_delta_analyzer_closes_exact_scaled_panel(tmp_path, monkeypatch):
    native = tmp_path / "native"
    half1 = tmp_path / "half1"
    half2 = tmp_path / "half2"
    native.mkdir()
    half1.mkdir()
    half2.mkdir()
    native_data = (
        np.arange(18, dtype=np.float32) + 1j * np.arange(18, dtype=np.float32)[::-1]
    ).astype(np.complex64)
    native_weight = np.arange(1, 19, dtype=np.float32)
    for half, original_index, part_id in ((1, 0, 7), (2, 3, 11)):
        _write_native_particle(
            native,
            rank=half,
            part_id=part_id,
            stack_index=original_index + 1,
            data=native_data,
            weight=native_weight,
        )
        _write_recovar_particle(
            half1 if half == 1 else half2,
            half=half,
            original_index=original_index,
            data=-native_data / np.float32(4**2),
            weight=native_weight / np.float32(4**4),
        )

    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_k1_bpref_particle_deltas.py",
            "--native-directory",
            str(native),
            "--recovar-directory-half1",
            str(half1),
            "--recovar-directory-half2",
            str(half2),
            "--grid-size",
            "4",
            "--output-json",
            str(output),
        ],
    )
    analyzer.main()
    report = json.loads(output.read_text())
    assert report["fixed_panel_score"] == {
        "denominator_passes": 2,
        "joint_passes": 2,
        "numerator_passes": 2,
        "total": 2,
    }
    assert report["classification"].endswith("reduction_or_launch_order_remains")
    assert all(
        particle["particle_delta"][field]["relative_l2"] == 0.0
        for particle in report["particles"]
        for field in ("numerator", "denominator")
    )
    assert all(
        particle["production_increment"][field]["relative_l2"] == 0.0
        for particle in report["particles"]
        for field in ("numerator", "denominator")
    )


def test_particle_delta_analyzer_measures_recovar_repeat_floor(tmp_path, monkeypatch):
    native = tmp_path / "native"
    half1 = tmp_path / "half1"
    half2 = tmp_path / "half2"
    repeat1 = tmp_path / "repeat1"
    repeat2 = tmp_path / "repeat2"
    for path in (native, half1, half2, repeat1, repeat2):
        path.mkdir()
    native_data = np.ones(18, dtype=np.complex64)
    native_weight = np.ones(18, dtype=np.float32)
    for half, original_index in ((1, 0), (2, 3)):
        _write_native_particle(
            native,
            rank=half,
            part_id=original_index,
            stack_index=original_index + 1,
            data=native_data,
            weight=native_weight,
        )
        data = -native_data / np.float32(4**2)
        weight = native_weight / np.float32(4**4)
        _write_recovar_particle(
            half1 if half == 1 else half2,
            half=half,
            original_index=original_index,
            data=data,
            weight=weight,
        )
        repeat_data = data.copy()
        repeat_data[0] = np.complex64(
            np.nextafter(repeat_data[0].real, np.float32(0.0))
            + 1j * repeat_data[0].imag
        )
        _write_recovar_particle(
            repeat1 if half == 1 else repeat2,
            half=half,
            original_index=original_index,
            data=repeat_data,
            weight=weight,
        )

    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_k1_bpref_particle_deltas.py",
            "--native-directory",
            str(native),
            "--recovar-directory-half1",
            str(half1),
            "--recovar-directory-half2",
            str(half2),
            "--recovar-repeat-directory-half1",
            str(repeat1),
            "--recovar-repeat-directory-half2",
            str(repeat2),
            "--grid-size",
            "4",
            "--output-json",
            str(output),
        ],
    )
    analyzer.main()
    repeat = json.loads(output.read_text())["recovar_repeat"]["summary"]
    assert repeat["isolated_numerator"]["bit_exact_count"] == 0
    assert repeat["isolated_numerator"]["relative_l2_max"] > 0.0
    assert repeat["isolated_denominator"]["bit_exact_count"] == 2
    assert repeat["isolated_denominator"]["relative_l2_max"] == 0.0
