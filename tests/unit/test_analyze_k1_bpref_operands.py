from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyze_k1_bpref_operands import analyze


def _write_flat(path: Path, values) -> None:
    values = np.asarray(values)
    with path.open("wb") as stream:
        np.asarray([values.size], dtype=np.uint64).tofile(stream)
        values.tofile(stream)


def _write_native(root: Path, half: int, internal: int, stack: int) -> None:
    prefix = root / f"half{half}_part{internal}_stack{stack}_bpref_prefix_"
    metadata = np.asarray([1, 1, half, internal, stack, 0, 1, 2, 1, 1, 0, 0, 1, 1, 2], dtype=np.uint64)
    arrays = {
        "metadata": metadata,
        "operand_image_real": np.asarray([-1, -3, -5, -7], dtype=np.float32),
        "operand_image_imag": np.asarray([-2, -4, -6, -8], dtype=np.float32),
        "operand_translation_x": np.asarray([0.1, 0.2], dtype=np.float32),
        "operand_translation_y": np.asarray([0.3, 0.4], dtype=np.float32),
        "operand_translation_z": np.zeros(2, dtype=np.float32),
        "operand_weights": np.asarray([1, 2, 3, 4], dtype=np.float32),
        "operand_minvsigma2": np.asarray([5, 6, 7, 8], dtype=np.float32),
        "operand_ctf": np.asarray([7, 8, 9, 10], dtype=np.float32),
        "operand_eulers": np.arange(18, dtype=np.float32),
        "operand_controls": np.asarray([2, 10], dtype=np.float32),
    }
    for suffix, values in arrays.items():
        _write_flat(Path(f"{prefix}{suffix}.bin"), values)


def _write_recovar(root: Path, half: int, original: int) -> None:
    np.savez(
        root / f"bpref_accumulator_delta_it001_h{half}_orig{original:06d}.npz",
        operand_source_image=np.asarray([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex64),
        operand_ctf=np.asarray([7, 8, 9, 10], dtype=np.float32),
        operand_minvsigma2=np.asarray([5, 6, 7, 8], dtype=np.float32),
        operand_posterior=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        operand_translation_angles=np.asarray([[0.1, 0.3], [0.2, 0.4]], dtype=np.float32),
        operand_eulers=np.arange(18, dtype=np.float32).reshape(2, 3, 3),
        operand_threshold=np.asarray([0.2], dtype=np.float32),
        operand_weight_norm=np.asarray([1], dtype=np.float32),
        max_r=np.float64(1.0),
    )


def test_operand_analyzer_aligns_native_conventions(tmp_path):
    native = tmp_path / "native"
    h1 = tmp_path / "h1"
    h2 = tmp_path / "h2"
    native.mkdir()
    h1.mkdir()
    h2.mkdir()
    selection = {
        "schema": "recovar-k1-bpref-prefix-selection-v2",
        "half1": {"half_local_ordinals": [0], "original_indices": [10], "native_internal_indices": [91], "stack_indices_1based": [11]},
        "half2": {"half_local_ordinals": [0], "original_indices": [20], "native_internal_indices": [82], "stack_indices_1based": [21]},
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    _write_native(native, 1, 91, 11)
    _write_native(native, 2, 82, 21)
    _write_recovar(h1, 1, 10)
    _write_recovar(h2, 2, 20)

    report = analyze(selection_path, native, {1: h1, 2: h2})
    for row in report["rows"]:
        assert row["image"]["best"] == "negative"
        assert row["image"]["variants"]["negative"]["bitwise_equal"]
        assert row["ctf"]["best"] == "direct"
        assert row["ctf"]["variants"]["direct"]["bitwise_equal"]
        assert row["minvsigma2"]["bitwise_equal"]
        assert row["posterior_normalized_active"]["bitwise_equal"]
        assert row["support_mask"]["bitwise_equal"]
        assert row["normalized_threshold"]["bitwise_equal"]
        assert row["eulers"]["best"] == "direct"
