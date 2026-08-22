from __future__ import annotations

from pathlib import Path

from scripts import run_vdam_kclass_parity_pair as runner


def test_pair_commands_share_scientific_parameters(tmp_path: Path):
    args = runner._parse_args(
        [
            "--fixture-dir",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "out"),
            "--K",
            "3",
            "--nr-iter",
            "17",
            "--random-seed",
            "9",
            "--tau2-fudge",
            "2.5",
            "--healpix-order",
            "2",
            "--oversampling",
            "0",
            "--offset-range",
            "4",
            "--offset-step",
            "1",
            "--padding-factor",
            "2",
        ]
    )
    relion = runner.build_relion_command(args, tmp_path / "relion/run")
    recovar = runner.build_recovar_command(args, tmp_path / "recovar/run")

    shared = {
        "--K": "3",
        "--random_seed": "9",
        "--tau2_fudge": "2.5",
        "--healpix_order": "2",
        "--oversampling": "0",
        "--offset_range": "4.0",
        "--offset_step": "1.0",
        "--pad": "2",
    }
    aliases = {
        "--random_seed": "--random-seed",
        "--tau2_fudge": "--tau2-fudge",
        "--healpix_order": "--healpix-order",
        "--offset_range": "--offset-range",
        "--offset_step": "--offset-step",
        "--pad": "--padding-factor",
    }
    for relion_flag, value in shared.items():
        recovar_flag = aliases.get(relion_flag, relion_flag)
        assert relion[relion.index(relion_flag) + 1] == value
        assert recovar[recovar.index(recovar_flag) + 1] == value


def test_default_checkpoints_include_final_iteration(tmp_path: Path):
    args = runner._parse_args(
        ["--fixture-dir", str(tmp_path), "--output-root", str(tmp_path / "out"), "--nr-iter", "8"]
    )

    assert args.checkpoint == [0, 1, 2, 4, 8]
    assert args.minimum_assignment_accuracy == 0.995
