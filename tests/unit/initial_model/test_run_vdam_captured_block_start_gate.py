from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_captured_block_start_gate_is_h100_focused_and_fail_closed():
    source = (REPO_ROOT / "scripts/run_vdam_captured_block_start_gate.sbatch").read_text()

    assert "#SBATCH --constraint=h100" in source
    assert "EXPECTED_REPO_HEAD" in source
    assert 'test -z "$(git -C "${REPO_ROOT}" status' in source
    assert "CUDA_ARCH=\"-gencode arch=compute_90,code=sm_90\"" in source
    assert "test_vdam_block_start_replay.py" in source
    assert "test_relion_vdam_captured_rotation_order_matches_reverse_replay" in source
    assert "test-full" not in source
    assert "long-test" not in source
    assert "SAFE_TO_DELETE" in source
    assert "COMPLETED" in source
