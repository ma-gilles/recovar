from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import run_em_real_kclass_initialmodel_pair as runner

pytestmark = pytest.mark.unit


def test_git_branch_records_detached_head(tmp_path: Path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "config", "user.name", "RECOVAR test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("sealed\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "sealed"], cwd=repo, check=True)

    branch = runner._git_branch(repo)
    assert branch not in {"", "<detached>"}

    subprocess.run(["git", "switch", "--detach", "-q", "HEAD"], cwd=repo, check=True)
    assert runner._git_branch(repo) == "<detached>"


def test_pair_commands_share_scientific_parameters_and_current_entrypoint(tmp_path: Path):
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

    assert recovar[1:3] == ["-m", "scripts.run_ab_initio"]
    assert "recovar.commands.initial_model" not in recovar
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
    aliases = {"--pad": "--padding_factor"}
    for relion_flag, value in shared.items():
        recovar_flag = aliases.get(relion_flag, relion_flag)
        assert relion[relion.index(relion_flag) + 1] == value
        assert recovar[recovar.index(recovar_flag) + 1] == value
    assert recovar[recovar.index("--do_run_C1") + 1] == "0"


def test_default_contract_covers_emitted_iterations_and_flags_collapsed_classes(tmp_path: Path):
    args = runner._parse_args(
        ["--fixture-dir", str(tmp_path), "--output-root", str(tmp_path / "out"), "--nr-iter", "8"]
    )

    assert args.checkpoint == list(range(1, 9))
    assert args.minimum_assignment_accuracy == 0.995
    assert args.minimum_class_fraction == 0.01
    assert args.reference_pair_report is None
    assert args.image_batch_size == 500
    assert args.image_fourier_backend == "relion_cuda"
    recovar = runner.build_recovar_command(args, tmp_path / "recovar/run")
    assert recovar[recovar.index("--image_batch_size") + 1] == "500"
    assert recovar[recovar.index("--image_fourier_backend") + 1] == "relion_cuda"


def test_explicit_legacy_iteration_zero_checkpoint_is_retained(tmp_path: Path):
    args = runner._parse_args(
        [
            "--fixture-dir",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "out"),
            "--checkpoint",
            "0",
            "--checkpoint",
            "8",
        ]
    )

    assert args.checkpoint == [0, 8]


def test_required_fixture_paths_and_lineage_are_separate(tmp_path: Path):
    nested_stack = tmp_path / "nested" / "particles.mrcs"
    nested_stack.parent.mkdir()
    nested_stack.touch()
    grid_stack = tmp_path / "particles.256.mrcs"
    grid_stack.touch()
    (tmp_path / "fixture_manifest.json").write_text("{}")
    (tmp_path / "source_indices.npy").touch()
    data_star = tmp_path / "particles.star"
    data_star.write_text(
        "data_particles\nloop_\n_rlnImageName #1\n"
        "1@particles.256.mrcs\n2@nested/particles.mrcs\n"
    )

    assert runner._required_fixture_paths(tmp_path) == [
        data_star,
        nested_stack.resolve(),
        grid_stack.resolve(),
    ]
    assert runner._fixture_lineage_paths(tmp_path) == [
        tmp_path / "fixture_manifest.json",
        tmp_path / "source_indices.npy",
    ]


def test_gpu_and_rss_parsers_keep_engine_resources(tmp_path: Path):
    monitor = tmp_path / "gpu.csv"
    monitor.write_text(
        "2026/09/01 10:00:00, 0, GPU-test, 100, 80000, 4\n"
        "2026/09/01 10:00:01, 0, GPU-test, 4567, 80000, 90\n"
    )
    resources = tmp_path / "resources.txt"
    resources.write_text("RECOVAR_EM_MAX_RSS_KIB=123456\n")

    assert runner._parse_gpu_monitor(monitor) == {
        "path": str(monitor),
        "sample_count": 2,
        "peak_hbm_mib": 4567.0,
        "gpu_memory_total_mib": 80000.0,
    }
    assert runner._parse_max_rss(resources) == 123456


def test_slurm_field_parser_exposes_exact_nonexclusive_one_gpu_contract():
    text = (
        "JobId=42 JobState=RUNNING NumNodes=1 NodeList=della-h1 "
        "ReqTRES=cpu=8,gres/gpu=1,mem=192G,node=1 "
        "AllocTRES=cpu=8,gres/gpu=1,mem=192G,node=1 "
        "Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* "
        "TresPerNode=gres/gpu:h100:1 OverSubscribe=OK"
    )

    fields = runner._parse_scontrol_fields(text)

    assert fields["ReqTRES"] == fields["AllocTRES"]
    assert fields["AllocTRES"] == "cpu=8,gres/gpu=1,mem=192G,node=1"
    assert fields["Socks/Node"] == "*"
    assert fields["NtasksPerN:B:S:C"] == "0:0:*:*"
    assert runner._gpu_count_from_tres(fields["ReqTRES"]) == 1
    assert fields["OverSubscribe"] == "OK"
    assert runner._gpu_count_from_tres("gres/gpu:h100=1") == 1
    assert runner._gpu_count_from_tres("gres/gpu=1,gres/gpu:h100=1") == 1


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
            "--checkpoint",
            "0",
            "--checkpoint",
            "1",
            "--checkpoint",
            "2",
        ]
    )
    (reference_dir / "command.json").write_text(
        json.dumps(runner.build_relion_command(args, reference_dir / "run"))
    )
    required = runner._required_fixture_paths(fixture_dir)
    fixture_sha256 = {
        runner._fixture_source_name(path, fixture_dir): runner._sha256(path) for path in required
    }
    report = {
        "schema": "recovar.vdam_kclass_pair.v1",
        "git_dirty": False,
        "physical_gpu_uuid": "GPU-frozen",
        "fixture_sha256": fixture_sha256,
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
    return args, fixture_sha256, report, reference_dir


def test_legacy_frozen_reference_remains_a_supported_oracle(tmp_path: Path):
    args, fixture_sha256, expected_report, expected_dir = _write_frozen_reference(tmp_path)

    report, reference_dir = runner._validated_frozen_reference(args, fixture_sha256)
    resources = runner._frozen_relion_resources(report)

    assert report == expected_report
    assert reference_dir == expected_dir
    assert resources["wall_s"] == 12.0
    assert resources["max_rss_kib"] is None
    assert resources["gpu"]["peak_hbm_mib"] is None


def test_legacy_frozen_reference_source_is_not_backfilled_from_current_source():
    current_binding_source = {"git_head": "current"}

    source = runner._relion_oracle_source(
        {"schema": "recovar.vdam_kclass_pair.v1"},
        current_binding_source,
    )

    assert source["available"] is False
    assert "legacy frozen pair report" in source["missing_reason"]
    assert runner._relion_oracle_source(None, current_binding_source) == current_binding_source


def test_frozen_reference_rejects_fixture_drift(tmp_path: Path):
    args, fixture_sha256, _report, _reference_dir = _write_frozen_reference(tmp_path)
    (args.fixture_dir / "particles.mrcs").write_bytes(b"changed")
    changed = {
        runner._fixture_source_name(path, args.fixture_dir): runner._sha256(path)
        for path in runner._required_fixture_paths(args.fixture_dir)
    }

    with pytest.raises(runner.PairRunError, match="fixture hashes differ"):
        runner._validated_frozen_reference(args, changed)
    assert changed != fixture_sha256


def test_build_artifacts_record_binaries_and_explicit_gaps(tmp_path: Path):
    cuda = tmp_path / "libcuda.so"
    cuda.write_bytes(b"cuda")
    binding = tmp_path / "binding"
    binding.mkdir()
    (binding / "_relion_bind_core.test.so").write_bytes(b"binding")

    complete = runner._build_artifacts(
        {"RECOVAR_CUDA_LIB": str(cuda), "RECOVAR_RELION_BIND_BUILD_DIR": str(binding)}
    )
    missing = runner._build_artifacts({})

    assert complete["cuda_library"]["sha256"] == runner._sha256(cuda)
    assert complete["relion_binding"][0]["sha256"] == runner._sha256(
        binding / "_relion_bind_core.test.so"
    )
    assert missing["cuda_library"] is None
    assert missing["relion_binding"] == []
