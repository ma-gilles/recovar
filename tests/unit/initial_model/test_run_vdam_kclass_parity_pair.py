from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    assert args.reference_pair_report is None


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


def _write_frozen_reference(tmp_path: Path):
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    stack = fixture_dir / "particles.mrcs"
    stack.write_bytes(b"stack")
    star = fixture_dir / "particles.star"
    star.write_text("data_particles\nloop_\n_rlnImageName #1\n1@particles.mrcs\n")
    executable = tmp_path / "relion_refine"
    executable.write_bytes(b"relion")
    executable.chmod(0o755)
    pair_root = tmp_path / "frozen-pair"
    reference_dir = pair_root / "relion"
    reference_dir.mkdir(parents=True)
    report_path = pair_root / "pair_report.json"
    args = runner._parse_args(
        [
            "--fixture-dir",
            str(fixture_dir),
            "--output-root",
            str(tmp_path / "candidate"),
            "--relion-refine",
            str(executable),
            "--reference-pair-report",
            str(report_path),
            "--K",
            "4",
            "--nr-iter",
            "2",
        ]
    )
    (reference_dir / "command.json").write_text(
        json.dumps(runner.build_relion_command(args, reference_dir / "run"))
    )
    required = runner._required_fixture_paths(fixture_dir)
    report = {
        "schema": "recovar.vdam_kclass_pair.v1",
        "git_dirty": False,
        "physical_gpu_uuid": "GPU-frozen",
        "fixture_sha256": {
            runner._fixture_source_name(path, fixture_dir): runner._sha256(path)
            for path in required
        },
        "relion_executable": str(executable),
        "relion_sha256": runner._sha256(executable),
        "relion_timing": {"wall_s": 12.0, "exit_code": 0},
        "audit": {
            "K": 4,
            "checkpoints": [0, 1, 2],
            "thresholds": {
                "minimum_per_class_fsc_auc": 0.999,
                "minimum_class_assignment_accuracy": 0.995,
            },
        },
    }
    report_path.write_text(json.dumps(report))
    return args, report["fixture_sha256"], report, reference_dir


def test_frozen_reference_validates_full_scientific_contract(tmp_path: Path):
    args, fixture_sha256, expected_report, expected_dir = _write_frozen_reference(tmp_path)

    report, reference_dir = runner._validated_frozen_reference(args, fixture_sha256)

    assert report == expected_report
    assert reference_dir == expected_dir


def test_frozen_reference_rejects_threshold_drift(tmp_path: Path):
    args, fixture_sha256, report, _reference_dir = _write_frozen_reference(tmp_path)
    report["audit"]["thresholds"]["minimum_class_assignment_accuracy"] = 0.99
    args.reference_pair_report.write_text(json.dumps(report))

    with pytest.raises(runner.PairRunError, match="audit contract differs"):
        runner._validated_frozen_reference(args, fixture_sha256)


def test_frozen_reference_rejects_fixture_drift(tmp_path: Path):
    args, fixture_sha256, _report, _reference_dir = _write_frozen_reference(tmp_path)
    (args.fixture_dir / "particles.mrcs").write_bytes(b"changed")
    changed_fixture_sha256 = {
        runner._fixture_source_name(path, args.fixture_dir): runner._sha256(path)
        for path in runner._required_fixture_paths(args.fixture_dir)
    }

    with pytest.raises(runner.PairRunError, match="fixture hashes differ"):
        runner._validated_frozen_reference(args, changed_fixture_sha256)

    assert changed_fixture_sha256 != fixture_sha256
