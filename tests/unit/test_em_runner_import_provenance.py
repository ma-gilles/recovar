"""Static and direct checks for EM runner checkout provenance."""

from pathlib import Path

import pytest

from scripts import run_full_refinement


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_RUNNERS = (
    REPO_ROOT / "scripts/run_em_completion_bench_slurm.sh",
    REPO_ROOT / "scripts/run_em_k1_robustness_matrix_slurm.sh",
    REPO_ROOT / "scripts/run_em_kclass_robustness_matrix_slurm.py",
)


@pytest.mark.unit
def test_active_em_runners_use_module_invocations():
    for runner in ACTIVE_RUNNERS:
        source = runner.read_text()
        assert '}" scripts/' not in source, runner
        assert "python scripts/" not in source, runner
        assert '}" "${REPO_ROOT}/scripts/' not in source, runner

    combined = "\n".join(runner.read_text() for runner in ACTIVE_RUNNERS)
    assert "-m scripts.run_full_refinement" in combined
    assert "RECOVAR_EXPECTED_REPO_ROOT" in combined


@pytest.mark.unit
def test_concrete_em_imports_are_bound_to_expected_repo(monkeypatch):
    monkeypatch.setenv("RECOVAR_EXPECTED_REPO_ROOT", str(REPO_ROOT))
    run_full_refinement._assert_expected_repo_imports()


@pytest.mark.unit
def test_concrete_em_imports_reject_wrong_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("RECOVAR_EXPECTED_REPO_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="RECOVAR import provenance failure"):
        run_full_refinement._assert_expected_repo_imports()
