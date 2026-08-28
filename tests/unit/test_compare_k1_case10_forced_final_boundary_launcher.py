from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "compare_k1_case10_forced_final_boundary.sbatch"
)


def test_launcher_requires_explicit_repo_under_slurm() -> None:
    source = LAUNCHER.read_text()

    assert ': "${REPO:?REPO is required}"' in source
    assert "BASH_SOURCE" not in source
    assert 'git -C "${REPO}" rev-parse HEAD' in source
