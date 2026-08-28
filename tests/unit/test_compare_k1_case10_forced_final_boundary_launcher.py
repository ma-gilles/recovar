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


def test_launcher_compares_native_grid_off_and_grid_on_replays() -> None:
    source = LAUNCHER.read_text()

    assert "native_accumulator_replay.mrc" in source
    assert "native_accumulator_replay_grid_on.mrc" in source
    assert "--grid-correct on" in source
    assert "native_relion_grid_on_replay_vs_source_capture" in source
