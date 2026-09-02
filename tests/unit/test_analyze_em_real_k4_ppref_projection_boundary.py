from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest

from scripts import analyze_em_real_k4_ppref_projection_boundary as analysis

pytestmark = pytest.mark.unit


def test_classify_texture_projection_boundary() -> None:
    assert analysis.classify(
        native_eulers_texture_vs_native_l2=6.0e-3,
        recovar_eulers_texture_vs_native_l2=6.0e-3,
        recovar_eulers_texture_vs_recovar_l2=1.0e-9,
        native_vs_recovar_l2=6.0e-3,
    ) == (
        "native_vs_recovar_texture_projection_is_first_material_ppref_downstream_difference"
    )


def test_classify_recovar_wiring_boundary() -> None:
    assert analysis.classify(
        native_eulers_texture_vs_native_l2=1.0e-9,
        recovar_eulers_texture_vs_native_l2=1.0e-9,
        recovar_eulers_texture_vs_recovar_l2=6.0e-3,
        native_vs_recovar_l2=6.0e-3,
    ) == "recovar_projected_reference_wiring_differs_after_matching_native_ppref_and_eulers"


def test_classify_euler_shell_boundary() -> None:
    assert analysis.classify(
        native_eulers_texture_vs_native_l2=2.0e-6,
        recovar_eulers_texture_vs_native_l2=6.0e-3,
        recovar_eulers_texture_vs_recovar_l2=5.0e-3,
        native_vs_recovar_l2=6.0e-3,
    ) == "recovar_euler_values_trigger_material_projection_shell_boundary_difference"


def test_classify_nonmaterial_and_mixed() -> None:
    assert analysis.classify(
        native_eulers_texture_vs_native_l2=1.0e-8,
        recovar_eulers_texture_vs_native_l2=1.0e-8,
        recovar_eulers_texture_vs_recovar_l2=1.0e-8,
        native_vs_recovar_l2=1.0e-8,
    ) == "captured_projected_reference_difference_is_not_material"
    assert analysis.classify(
        native_eulers_texture_vs_native_l2=3.0e-3,
        recovar_eulers_texture_vs_native_l2=3.0e-3,
        recovar_eulers_texture_vs_recovar_l2=2.0e-3,
        native_vs_recovar_l2=6.0e-3,
    ) == "ppref_projection_boundary_is_mixed_or_unresolved"


def _write_mrc(path: Path, values: np.ndarray) -> None:
    with mrcfile.new(path) as stream:
        stream.set_data(np.asarray(values, dtype=np.float32))


def test_source_map_panel_ignores_header_only_differences(tmp_path: Path) -> None:
    source_paths = []
    operand_paths = []
    for model in range(4):
        values = np.arange(27, dtype=np.float32).reshape(3, 3, 3) + model
        source = tmp_path / f"source_{model}.mrc"
        operand = tmp_path / f"operand_{model}.mrc"
        _write_mrc(source, values)
        _write_mrc(operand, values)
        with mrcfile.open(operand, mode="r+") as stream:
            stream.header.nlabl = 1
            stream.header.label[0] = b"different header"
        source_paths.append(source)
        operand_paths.append(operand)
    records = analysis.compare_source_map_panels(source_paths, operand_paths)
    assert len(records) == 4
    assert all(record["payload_bitwise_equal"] for record in records)
    assert any(
        record["ppref_source_map"]["sha256"]
        != record["operand_source_map"]["sha256"]
        for record in records
    )


def test_source_map_panel_rejects_payload_difference(tmp_path: Path) -> None:
    source = tmp_path / "source.mrc"
    operand = tmp_path / "operand.mrc"
    _write_mrc(source, np.zeros((3, 3, 3), dtype=np.float32))
    changed = np.zeros((3, 3, 3), dtype=np.float32)
    changed[1, 1, 1] = 1
    _write_mrc(operand, changed)
    with pytest.raises(analysis.AnalysisError, match="map payload differs"):
        analysis.compare_source_map_panels([source], [operand])
