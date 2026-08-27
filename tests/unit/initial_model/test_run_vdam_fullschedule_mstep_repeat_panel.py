from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_repeat_panel_can_fail_before_output_on_wrong_physical_gpu():
    source = (
        REPO_ROOT / "scripts/run_vdam_fullschedule_mstep_repeat_panel.sbatch"
    ).read_text()

    assert "TARGET_GPU_UUID=${TARGET_GPU_UUID:-}" in source
    assert "VDAM_TARGET_GPU_MISS" in source
    assert "exit 75" in source
    assert source.index("VDAM_TARGET_GPU_MISS") < source.index("for arm in a b")
    assert "test-full" not in source
    assert "long-test" not in source
