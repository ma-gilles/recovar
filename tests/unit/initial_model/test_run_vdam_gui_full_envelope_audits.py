from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_gui_full_envelope_audit_array_is_fail_closed_and_self_calibrated():
    source = (REPO_ROOT / "scripts/run_vdam_gui_full_envelope_audits.sbatch").read_text()

    assert "#SBATCH --partition=cpu" in source
    assert "#SBATCH --array=2-22%4" in source
    assert "#SBATCH --cpus-per-task=1" in source
    assert "#SBATCH --mem=8G" in source
    assert 'SLURM_ARRAY_TASK_ID > 999' in source
    assert 'case.get("id") == sys.argv[2]' in source
    assert 'case_match_count' in source
    assert "status --porcelain=v1 --untracked-files=no" in source
    assert "EXPECTED_REPO_HEAD" in source
    assert "SCIENCE_COMPLETED" in source
    assert "science_manifest.json" in source
    assert "rlnImagePixelSize" in source
    assert "scripts/run_vdam_candidate_envelope_audits.sbatch" in source
    assert "candidate_native_envelope.json" in source
    assert "candidate_state_envelope.json" in source
    assert "candidate_envelope_status.json" in source
    assert "evidence.sha256" in source
