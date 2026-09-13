from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_candidate_envelope_runner_is_fail_closed_and_runs_both_gates():
    source = (REPO_ROOT / "scripts/run_vdam_candidate_envelope_audits.sbatch").read_text()
    assert "--partition=cpu" in source
    assert "status --porcelain=v1 --untracked-files=no" in source
    assert "EXPECTED_REPO_HEAD" in source
    assert "scripts.audit_vdam_candidate_native_envelope" in source
    assert "scripts.audit_vdam_candidate_state_envelope" in source
    assert 'native_args+=(--native-root "${native_root}")' in source
    assert "NATIVE_REFERENCE_ROOT" in source
    assert 'native_reference_args+=(--native-reference-root "${NATIVE_REFERENCE_ROOT}")' in source
    assert 'test -f "${native_root}/SCIENCE_COMPLETED"' in source
    assert "candidate_native_envelope_shells.npz" in source
    assert "candidate_state_envelope.json" in source
    assert "JAX_PLATFORMS=cpu" in source
    assert "JAX_COMPILATION_CACHE_DIR" in source
    assert "VDAM_MARK_CANDIDATE_COMPLETE" in source
    assert 'touch "${CANDIDATE_ROOT}/COMPLETED"' in source
    assert "if (( map_status != 0 || state_status != 0 )); then" in source
