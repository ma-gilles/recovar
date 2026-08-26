from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyze_k1_bpref_prefixes import analyze


def _write_flat(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values)
    with path.open("wb") as stream:
        np.asarray([values.size], dtype=np.uint64).tofile(stream)
        values.tofile(stream)


def _write_native(root: Path, half: int, internal_index: int, stack_index: int, data, weight) -> None:
    prefix = root / f"half{half}_part{internal_index}_stack{stack_index}_bpref_prefix_"
    metadata = np.asarray(
        [1, 1, half, internal_index, stack_index, 0, 1, 2, 1, 1, 0, 0, 1, 1, 2],
        dtype=np.uint64,
    )
    _write_flat(Path(f"{prefix}metadata.bin"), metadata)
    native_data = -np.asarray(data, dtype=np.complex64)
    _write_flat(Path(f"{prefix}real.bin"), native_data.real.astype(np.float32))
    _write_flat(Path(f"{prefix}imag.bin"), native_data.imag.astype(np.float32))
    _write_flat(Path(f"{prefix}weight.bin"), np.asarray(weight, dtype=np.float32))


def _write_recovar(root: Path, half: int, original_index: int, ordinal: int, data, weight) -> None:
    data = np.asarray(data, dtype=np.complex64)
    weight = np.asarray(weight, dtype=np.float32)
    np.savez(
        root / f"bpref_accumulator_delta_it001_h{half}_orig{original_index:06d}.npz",
        schema=np.asarray("recovar-bpref-accumulator-delta-v2"),
        iteration=np.int64(1),
        half=np.int64(half),
        original_index=np.int64(original_index),
        particle_launch_ordinal=np.int64(ordinal),
        after_data=data,
        after_weight=weight,
        isolated_data=data,
        isolated_weight=weight,
    )


def test_prefix_analyzer_reports_first_unequal_prefix(tmp_path):
    native = tmp_path / "native"
    recovar = tmp_path / "recovar"
    native.mkdir()
    recovar.mkdir()
    selection = {
        "schema": "recovar-k1-bpref-prefix-selection-v2",
        "half1": {
            "half_local_ordinals": [0],
            "original_indices": [10],
            "native_internal_indices": [91],
            "stack_indices_1based": [11],
        },
        "half2": {
            "half_local_ordinals": [0],
            "original_indices": [20],
            "native_internal_indices": [82],
            "stack_indices_1based": [21],
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))

    _write_native(native, 1, 91, 11, [1 + 2j, 3 + 4j], [5, 6])
    _write_recovar(recovar, 1, 10, 0, [1 + 2j, 3 + 4j], [5, 6])
    _write_native(native, 2, 82, 21, [1 + 2j, 3 + 4j], [5, 6])
    _write_recovar(recovar, 2, 20, 0, [1 + 2j, 3 + 5j], [5, 6])

    report = analyze(selection_path, native, recovar)
    assert report["halves"]["half1"]["bitwise_equal_prefixes"] == 1
    assert report["halves"]["half1"]["first_unequal_prefix"] is None
    first = report["halves"]["half2"]["first_unequal_prefix"]
    assert first["prefix_particle_count"] == 1
    assert first["original_index"] == 20
    assert first["data"]["unequal_count"] == 1
    assert first["weight"]["bitwise_equal"]
