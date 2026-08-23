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
