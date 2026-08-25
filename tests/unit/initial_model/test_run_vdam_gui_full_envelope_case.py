from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_gui_full_envelope_runner_reuses_one_candidate_for_native_repeats():
    source = (REPO_ROOT / "scripts/run_vdam_gui_full_envelope_case.sbatch").read_text()

    assert "#SBATCH --array=2-22%4" in source
    assert "VDAM_REPEAT_COUNT=${VDAM_REPEAT_COUNT:-4}" in source
    assert source.count("scripts.run_vdam_relion_parity_case") == 1
    assert "scripts.run_vdam_relion_native_repeat" in source
    assert 'for repeat_index in $(seq 2 "${VDAM_REPEAT_COUNT}")' in source
    assert "EXPECTED_REPO_HEAD" in source
    assert "status --porcelain=v1 --untracked-files=no" in source
    assert "SCIENCE_COMPLETED" in source
    assert "science_manifest.json" in source


def test_native_repeat_runner_is_strictly_same_gpu_and_preserves_audits():
    source = (REPO_ROOT / "scripts/run_vdam_relion_native_repeat.py").read_text()

    assert "native repeat was not allocated on the completed candidate GPU" in source
    assert "physical GPU changed during native RELION repeat" in source
    assert "audit(" in source
    assert "_write_particle_state_audit(" in source
    assert '"trajectory_audit.json"' in source
    assert '"particle_state_trajectory_audit.json"' in source
    assert '"NATIVE_SCIENCE_COMPLETED"' in source
    assert '"strict_point_reference_result": report["result"]' in source
