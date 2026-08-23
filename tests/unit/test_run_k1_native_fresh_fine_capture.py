from pathlib import Path

import pytest


LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_k1_native_fresh_fine_capture.sbatch"
)


@pytest.mark.unit
def test_native_fresh_launcher_has_single_particle_fine_operand_probe() -> None:
    source = LAUNCHER.read_text()
    assert "FINE_OPERAND_ROTATION_LOCAL=${FINE_OPERAND_ROTATION_LOCAL:-}" in source
    assert "FINE_OPERAND_TRANSLATIONS=${FINE_OPERAND_TRANSLATIONS:-}" in source
    assert "FINE_OPERAND_TRANSLATIONS=${FINE_OPERAND_TRANSLATIONS//:/,}" in source
    assert 'test "${TARGET_COUNT}" -eq 1' in source
    assert "RELION_FINE_OPERAND_CAPTURE_EXPECTED_CANDIDATES" in source
    assert "*.fine-operand-v1.bin" in source


@pytest.mark.unit
def test_native_fresh_launcher_supports_sealed_later_iteration_prefixes() -> None:
    source = LAUNCHER.read_text()
    assert 'test "${TARGET_ITERATION}" -eq 2' not in source
    assert '[[ "${TARGET_ITERATION}" =~ ^[1-9][0-9]*$ ]]' in source
    assert "run_it${target_iteration_label}_data.star" in source
    assert '--iterations "${audit_iterations[@]}"' in source
    assert "expected_iteration = int(sys.argv[3])" in source


@pytest.mark.unit
def test_native_fresh_launcher_can_capture_same_run_ppref() -> None:
    source = LAUNCHER.read_text()
    assert "CAPTURE_PPREF=${CAPTURE_PPREF:-0}" in source
    assert 'case "${CAPTURE_PPREF}" in 0|1)' in source
    assert "RELION_PPREF_CAPTURE_DIR=${CAPTURE}" in source
    assert "RELION_PPREF_CAPTURE_ITER=${TARGET_ITERATION}" in source
    assert "ppref_iter*_rank*_model*.bin" in source
    assert 'assert {int(item["rank"]) for item in ppref} == {1, 2}' in source
