from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import analyze_em_real_k4_ppref_sources as analysis
from scripts.build_k1_ppref_capture_from_map import write_ppref_capture


def _write_panel(directory: Path, *, perturb_model: int | None = None) -> None:
    for model in range(4):
        values = (
            np.arange(75, dtype=np.float32).reshape(5, 5, 3)
            + np.float32(model + 1)
            + 1j * np.arange(75, 150, dtype=np.float32).reshape(5, 5, 3)
        ).astype(np.complex64)
        if model == perturb_model:
            values[1, 1, 1] += np.complex64(1.0e-2)
        write_ppref_capture(
            directory / f"ppref_iter001_rank000_model{model:03d}.bin",
            values,
            iteration=1,
            rank=0,
            model=model,
            current_size=4,
            r_max=2,
            padding_factor=1,
        )


def test_classification_gate() -> None:
    assert analysis.classify(0.0) == "map_to_ppref_matches_native"
    assert analysis.classify(analysis.SOURCE_RELATIVE_L2_CEILING) == (
        "map_to_ppref_matches_native"
    )
    assert analysis.classify(2 * analysis.SOURCE_RELATIVE_L2_CEILING) == (
        "map_to_ppref_difference_remains"
    )


def test_complete_exact_source_panel(tmp_path: Path) -> None:
    native = tmp_path / "native"
    rebuilt = tmp_path / "rebuilt"
    native.mkdir()
    rebuilt.mkdir()
    _write_panel(native)
    _write_panel(rebuilt)
    maps = []
    for model in range(4):
        path = tmp_path / f"class{model}.mrc"
        path.write_bytes(bytes([model + 1]))
        maps.append(path)
    report = analysis.build_report(
        native_dir=native,
        rebuilt_dir=rebuilt,
        source_maps=maps,
        expected_iteration=1,
        expected_rank=0,
        n_classes=4,
        expected_current_size=4,
    )
    assert report["classification"] == "map_to_ppref_matches_native"
    assert report["summary"]["bitwise_equal_classes"] == 4
    assert report["summary"]["maximum_relative_l2"] == 0


def test_detects_source_difference(tmp_path: Path) -> None:
    native = tmp_path / "native"
    rebuilt = tmp_path / "rebuilt"
    native.mkdir()
    rebuilt.mkdir()
    _write_panel(native)
    _write_panel(rebuilt, perturb_model=2)
    maps = []
    for model in range(4):
        path = tmp_path / f"class{model}.mrc"
        path.write_bytes(bytes([model + 1]))
        maps.append(path)
    report = analysis.build_report(
        native_dir=native,
        rebuilt_dir=rebuilt,
        source_maps=maps,
        expected_iteration=1,
        expected_rank=0,
        n_classes=4,
        expected_current_size=4,
    )
    assert report["classification"] == "map_to_ppref_difference_remains"
    assert report["summary"]["bitwise_equal_classes"] == 3
