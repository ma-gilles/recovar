from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyze_k1_bpref_prefix_repeat import analyze


def _write_flat(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values)
    with path.open("wb") as stream:
        np.asarray([values.size], dtype=np.uint64).tofile(stream)
        values.tofile(stream)


def _write_native(
    root: Path, half: int, internal_index: int, stack_index: int, data, weight
) -> None:
    prefix = root / f"half{half}_part{internal_index}_stack{stack_index}_bpref_prefix_"
    metadata = np.asarray(
        [1, 1, half, internal_index, stack_index, 0, 1, 2, 1, 1, 0, 0, 1, 1, 2],
        dtype=np.uint64,
    )
    _write_flat(Path(f"{prefix}metadata.bin"), metadata)
    data = np.asarray(data, dtype=np.complex64)
    _write_flat(Path(f"{prefix}real.bin"), data.real.astype(np.float32))
    _write_flat(Path(f"{prefix}imag.bin"), data.imag.astype(np.float32))
    _write_flat(Path(f"{prefix}weight.bin"), np.asarray(weight, dtype=np.float32))


def _write_recovar(
    root: Path, half: int, original_index: int, ordinal: int, native_data, weight
) -> None:
    native_data = np.asarray(native_data, dtype=np.complex64)
    weight = np.asarray(weight, dtype=np.float32)
    np.savez(
        root / f"bpref_accumulator_delta_it001_h{half}_orig{original_index:06d}.npz",
        schema=np.asarray("recovar-bpref-accumulator-delta-v2"),
        iteration=np.int64(1),
        half=np.int64(half),
        original_index=np.int64(original_index),
        particle_launch_ordinal=np.int64(ordinal),
        after_data=-native_data,
        after_weight=weight,
    )


def test_repeat_analyzer_reports_cross_to_repeat_ratio(tmp_path):
    primary = tmp_path / "primary"
    repeat = tmp_path / "repeat"
    recovar = tmp_path / "recovar"
    for path in (primary, repeat, recovar):
        path.mkdir()
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

    for half, original, internal, stack in ((1, 10, 91, 11), (2, 20, 82, 21)):
        _write_native(primary, half, internal, stack, [1 + 2j, 3 + 4j], [5, 6])
        _write_native(repeat, half, internal, stack, [1 + 2j, 3 + 5j], [5, 8])
        _write_recovar(recovar, half, original, 0, [1 + 2j, 3 + 6j], [5, 10])

    report = analyze(selection_path, primary, repeat, recovar)
    for half in (1, 2):
        row = report["halves"][f"half{half}"]["last_prefix"]
        assert np.isclose(row["cross_to_repeat_relative_l2_ratio"]["data"], 2.0)
        assert np.isclose(row["cross_to_repeat_relative_l2_ratio"]["weight"], 2.0)
