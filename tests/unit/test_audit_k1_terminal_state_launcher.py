from pathlib import Path

import pytest


LAUNCHER = Path(__file__).resolve().parents[2] / "scripts" / "audit_k1_terminal_state.sbatch"


@pytest.mark.unit
def test_terminal_state_launcher_is_fixed_boundary_and_fail_closed() -> None:
    source = LAUNCHER.read_text()
    assert 'test "$((RECOVAR_ITERATION + 1))" -eq "${RELION_ITERATION}"' in source
    assert '--relion-star "${RELION_ITERATION}:${RELION_NUMBERED}"' in source
    assert '--recovar-iteration "${RECOVAR_ITERATION}"' in source
    assert '--relion-final-star "${RELION_FINAL}"' in source
    assert 'test ! -e "${ANALYSIS}/particle_state.json"' in source
    assert 'test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"' in source
