from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest

from scripts import launch_em_real_kclass_initialmodel_slurm as launcher

pytestmark = pytest.mark.unit


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_fixture_validation_pins_star_and_selected_indices(tmp_path: Path):
    star = tmp_path / "particles.star"
    indices = tmp_path / "source_indices.npy"
    star.write_bytes(b"star")
    indices.write_bytes(b"indices")
    (tmp_path / "fixture_manifest.json").write_text("{}")
    spec = launcher.FixtureSpec(
        dataset="test",
        fixture_dir=tmp_path,
        particles_star_sha256=_digest(b"star"),
        source_indices_sha256=_digest(b"indices"),
    )

    launcher.validate_fixture(spec)
    star.write_bytes(b"changed")
    with pytest.raises(SystemExit, match="particles.star drift"):
        launcher.validate_fixture(spec)


def _args(tmp_path: Path) -> argparse.Namespace:
    return launcher.parse_args(
        [
            "--dataset",
            "10345",
            "--output-root",
            str(tmp_path / "run"),
            "--relion-source-dir",
            str(tmp_path / "relion/src"),
            "--pixi-python",
            str(tmp_path / "pixi/envs/default/bin/python"),
        ]
    )


def test_pair_command_targets_current_runner_and_emitted_checkpoint_range(tmp_path: Path):
    args = _args(tmp_path)
    command = launcher.build_pair_command(args, launcher.FIXTURES["10345"], tmp_path / "pair")

    assert command[1:3] == ["-m", "scripts.run_em_real_kclass_initialmodel_pair"]
    checkpoints = [int(command[index + 1]) for index, value in enumerate(command) if value == "--checkpoint"]
    assert checkpoints == list(range(1, 9))
    assert command[command.index("--minimum-class-fraction") + 1] == "0.01"


def test_explicit_legacy_iteration_zero_checkpoint_is_retained(tmp_path: Path):
    args = launcher.parse_args(
        [
            "--dataset",
            "10345",
            "--output-root",
            str(tmp_path / "run"),
            "--relion-source-dir",
            str(tmp_path / "relion/src"),
            "--pixi-python",
            str(tmp_path / "pixi/envs/default/bin/python"),
            "--checkpoint",
            "0",
            "--checkpoint",
            "8",
        ]
    )

    assert args.checkpoint == [0, 8]


def test_sbatch_is_one_gpu_nonexclusive_and_builds_sealed_runtime(tmp_path: Path):
    args = _args(tmp_path)
    command = launcher.build_pair_command(args, launcher.FIXTURES["10345"], tmp_path / "pair")

    text = launcher.render_sbatch(args, expected_head="a" * 40, pair_command=command)

    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --exclusive" not in text
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in text
    assert "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/real_kclass_10345_" in text
    assert "RECOVAR_RELION_BIND_BUILD_DIR=" in text
    assert "RECOVAR_CUDA_LIB=" in text
    assert "include-system-site-packages = true" in text
    assert "CUDA_ARCH='-gencode arch=compute_90,code=sm_90" in text
    assert "scripts.run_em_real_kclass_initialmodel_pair" in text
    assert "\\\n+  " not in text


def test_frozen_mode_is_explicit_in_pair_command(tmp_path: Path):
    args = _args(tmp_path)
    args.reference_pair_report = (tmp_path / "oracle" / "pair_report.json").resolve()

    command = launcher.build_pair_command(args, launcher.FIXTURES["10345"], tmp_path / "pair")

    assert command[command.index("--reference-pair-report") + 1] == str(args.reference_pair_report)
