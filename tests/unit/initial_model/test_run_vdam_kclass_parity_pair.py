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


def test_default_checkpoints_cover_every_written_iteration(tmp_path: Path):
    args = runner._parse_args(
        ["--fixture-dir", str(tmp_path), "--output-root", str(tmp_path / "out"), "--nr-iter", "8"]
    )

    assert args.checkpoint == list(range(9))
    assert args.minimum_assignment_accuracy == 0.995


def test_pair_report_records_only_recovar_environment():
    captured = runner._recovar_environment(
        {
            "PATH": "/bin",
            "RECOVAR_Z_OVERRIDE": "z",
            "RECOVAR_RELION_FINE_DIFF2_FUSED_FFI": "1",
        }
    )

    assert captured == {
        "RECOVAR_RELION_FINE_DIFF2_FUSED_FFI": "1",
        "RECOVAR_Z_OVERRIDE": "z",
    }


def test_required_fixture_paths_follow_star_stack_references(tmp_path: Path):
    nested_stack = tmp_path / "nested" / "particles.mrcs"
    nested_stack.parent.mkdir()
    nested_stack.touch()
    grid_stack = tmp_path / "particles.256.mrcs"
    grid_stack.touch()
    data_star = tmp_path / "particles.star"
    data_star.write_text(
        "\n".join(
            [
                "data_particles",
                "loop_",
                "_rlnImageName #1",
                "1@particles.256.mrcs",
                "2@nested/particles.mrcs",
            ]
        )
        + "\n"
    )

    assert runner._required_fixture_paths(tmp_path) == [
        data_star,
        nested_stack.resolve(),
        grid_stack.resolve(),
    ]
    assert runner._fixture_source_name(nested_stack, tmp_path) == "nested/particles.mrcs"


def test_required_fixture_paths_reject_star_without_stack_reference(tmp_path: Path):
    (tmp_path / "particles.star").write_text("data_particles\n")

    try:
        runner._required_fixture_paths(tmp_path)
    except runner.PairRunError as error:
        assert "no particle stacks" in str(error)
    else:
        raise AssertionError("expected a stack-free particle STAR to be rejected")
