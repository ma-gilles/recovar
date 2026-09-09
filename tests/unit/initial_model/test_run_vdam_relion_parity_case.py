from __future__ import annotations

import hashlib
import json
import subprocess
import types
from pathlib import Path

import pytest

from scripts import run_ab_initio
from scripts import run_vdam_relion_parity_case as runner

pytestmark = pytest.mark.unit


DEFINITION = {
    "nr_classes": 1,
    "nr_iter": 8,
    "random_seed": 0,
    "tau2_fudge": 4.0,
    "healpix_order": 1,
    "oversampling": 1,
    "offset_range_px": 6.0,
    "offset_step_px": 2.0,
    "padding_factor": 1,
}

SBATCH_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_vdam_relion_parity_case.sbatch"


def _value(argv: list[str], flag: str) -> str:
    assert argv.count(flag) == 1
    return argv[argv.index(flag) + 1]


def test_relion_command_is_the_frozen_gui_initialmodel_contract():
    argv = runner.build_relion_command(
        input_star=Path("/data/particles.star"),
        output_prefix=Path("/out/relion/run"),
        definition=DEFINITION,
        relion_refine=Path("/opt/relion_refine"),
        threads=8,
    )

    assert argv[0] == "/opt/relion_refine"
    assert {"--grad", "--denovo_3dref", "--flatten_solvent", "--zero_mask", "--auto_sampling"} <= set(argv)
    assert _value(argv, "--iter") == "8"
    assert _value(argv, "--grad_write_iter") == "1"
    assert _value(argv, "--K") == "1"
    assert _value(argv, "--sym") == "C1"
    assert _value(argv, "--particle_diameter") == "200.0"
    assert _value(argv, "--random_seed") == "0"
    assert _value(argv, "--tau2_fudge") == "4.0"
    assert _value(argv, "--healpix_order") == "1"
    assert _value(argv, "--oversampling") == "1"
    assert _value(argv, "--offset_range") == "6.0"
    assert _value(argv, "--offset_step") == "2.0"
    assert _value(argv, "--pad") == "1"
    assert _value(argv, "--gpu") == "0"


def test_recovar_command_maps_the_same_frozen_definition(monkeypatch):
    monkeypatch.setattr(runner.sys, "executable", "/env/python")

    argv = runner.build_recovar_command(
        input_star=Path("/data/particles.star"),
        output_prefix=Path("/out/recovar/run"),
        fixture_dir=Path("/data"),
        definition=DEFINITION,
    )

    assert argv[:3] == ["/env/python", "-m", "scripts.run_ab_initio"]
    assert _value(argv, "--nr_iter") == "8"
    assert _value(argv, "--grad_write_iter") == "1"
    assert _value(argv, "--K") == "1"
    assert _value(argv, "--sym") == "C1"
    assert _value(argv, "--particle_diameter") == "200.0"
    assert _value(argv, "--random_seed") == "0"
    assert _value(argv, "--tau2_fudge") == "4.0"
    assert _value(argv, "--healpix_order") == "1"
    assert _value(argv, "--oversampling") == "1"
    assert _value(argv, "--offset_range") == "6.0"
    assert _value(argv, "--offset_step") == "2.0"
    assert _value(argv, "--padding_factor") == "1"
    assert _value(argv, "--image_batch_size") == "500"
    assert _value(argv, "--gpu") == "0"
    assert "--require_custom_cuda" in argv


def test_command_builders_map_configurable_symmetry_and_particle_diameter(monkeypatch):
    monkeypatch.setattr(runner.sys, "executable", "/env/python")
    definition = {
        **DEFINITION,
        "symmetry": "C2",
        "particle_diameter_angstrom": 144.0,
    }

    relion = runner.build_relion_command(
        input_star=Path("/data/particles.star"),
        output_prefix=Path("/out/relion/run"),
        definition=definition,
        relion_refine=Path("/opt/relion_refine"),
        threads=8,
    )
    recovar = runner.build_recovar_command(
        input_star=Path("/data/particles.star"),
        output_prefix=Path("/out/recovar/run"),
        fixture_dir=Path("/data"),
        definition=definition,
    )

    assert _value(relion, "--sym") == "C1"
    assert _value(recovar, "--sym") == "C2"
    assert _value(recovar, "--do_run_C1") == "1"
    for argv in (relion, recovar):
        assert _value(argv, "--particle_diameter") == "144.0"


def test_command_builders_can_refine_directly_in_requested_symmetry(monkeypatch):
    monkeypatch.setattr(runner.sys, "executable", "/env/python")
    definition = {**DEFINITION, "symmetry": "C2", "do_run_C1": False}

    relion = runner.build_relion_command(
        input_star=Path("/data/particles.star"),
        output_prefix=Path("/out/relion/run"),
        definition=definition,
        relion_refine=Path("/opt/relion_refine"),
        threads=8,
    )
    recovar = runner.build_recovar_command(
        input_star=Path("/data/particles.star"),
        output_prefix=Path("/out/recovar/run"),
        fixture_dir=Path("/data"),
        definition=definition,
    )

    assert _value(relion, "--sym") == "C2"
    assert _value(recovar, "--sym") == "C2"
    assert _value(recovar, "--do_run_C1") == "0"


def test_recovar_command_accepts_resource_only_batch_override(monkeypatch):
    monkeypatch.setattr(runner.sys, "executable", "/env/python")

    argv = runner.build_recovar_command(
        input_star=Path("/data/particles.star"),
        output_prefix=Path("/out/recovar/run"),
        fixture_dir=Path("/data"),
        definition=DEFINITION,
        image_batch_size=200,
    )

    assert _value(argv, "--image_batch_size") == "200"
    with pytest.raises(ValueError, match="must be positive"):
        runner.build_recovar_command(
            input_star=Path("/data/particles.star"),
            output_prefix=Path("/out/recovar/run"),
            fixture_dir=Path("/data"),
            definition=DEFINITION,
            image_batch_size=0,
        )


def test_iteration_override_is_explicit_bounded_and_non_mutating(monkeypatch):
    monkeypatch.setenv("VDAM_NR_ITER_OVERRIDE", "2")

    result = runner._definition_with_iteration_override(DEFINITION)

    assert result["nr_iter"] == 2
    assert DEFINITION["nr_iter"] == 8
    monkeypatch.setenv("VDAM_NR_ITER_OVERRIDE", "9")
    with pytest.raises(runner.RunError, match="must be in"):
        runner._definition_with_iteration_override(DEFINITION)
    monkeypatch.setenv("VDAM_NR_ITER_OVERRIDE", "not-an-integer")
    with pytest.raises(runner.RunError, match="must be an integer"):
        runner._definition_with_iteration_override(DEFINITION)


def test_particle_state_audit_covers_every_numbered_checkpoint(monkeypatch, tmp_path):
    captured = {}

    def fake_audit(recovar_dir, relion_dir, *, iterations):
        captured.update(
            recovar_dir=recovar_dir,
            relion_dir=relion_dir,
            iterations=iterations,
        )
        return {
            "schema": "recovar.vdam_particle_state_trajectory_audit.v1",
            "first_divergent_iteration": 2,
            "iterations": [
                {"iteration": 1, "divergent_particle_count": 0},
                {"iteration": 2, "divergent_particle_count": 3},
                {"iteration": 3, "divergent_particle_count": 4},
            ],
        }

    monkeypatch.setattr(runner, "audit_particle_states", fake_audit)
    summary = runner._write_particle_state_audit(
        recovar_dir=tmp_path / "recovar",
        relion_dir=tmp_path / "relion",
        case_root=tmp_path,
        nr_iter=3,
    )

    assert captured["iterations"] == (1, 2, 3)
    assert summary == {
        "artifact": "particle_state_trajectory_audit.json",
        "audited_iteration_count": 3,
        "first_divergent_iteration": 2,
        "first_divergent_particle_count": 3,
    }
    payload = json.loads((tmp_path / summary["artifact"]).read_text())
    assert payload["first_divergent_iteration"] == 2


def test_bounded_report_is_marked_as_schedule_changing():
    source = (Path(__file__).resolve().parents[3] / "scripts" / "run_vdam_relion_parity_case.py").read_text()

    assert '"scientific_schedule_changed_by_nr_iter_override": True' in source
    assert '"frozen_nr_iter": int(case["definition"]["nr_iter"])' in source
    assert "not default-trajectory parity evidence" in source


def test_gpu_capture_requires_exactly_one_visible_uuid(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "GPU-one\nGPU-two\n", ""),
    )

    with pytest.raises(runner.RunError, match="exactly one visible"):
        runner._physical_gpu_uuid()


def test_gpu_capture_rejects_slurm_uuid_mismatch(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_GPUS", "GPU-slurm")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "GPU-visible\n", ""),
    )

    with pytest.raises(runner.RunError, match="differs from visible UUID"):
        runner._physical_gpu_uuid()


def test_relion_reference_provenance_fingerprints_exact_binary(tmp_path):
    executable = tmp_path / "relion_refine"
    payload = b"pinned-relion-reference\n"
    executable.write_bytes(payload)
    executable.chmod(0o755)

    report = runner._relion_reference_provenance(executable)

    assert report == {
        "executable": str(executable.resolve()),
        "executable_sha256": hashlib.sha256(payload).hexdigest(),
        "executable_size_bytes": len(payload),
    }


def test_git_source_guard_accepts_clean_head_and_rejects_tracked_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "vdam-test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "VDAM Test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("frozen\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "frozen"], cwd=repo, check=True)

    head, dirty = runner._git_source_state(repo)
    assert len(head) == 40
    assert dirty is False
    runner._assert_git_source_unchanged(repo, head)

    tracked.write_text("changed\n")
    with pytest.raises(runner.RunError, match="source changed during execution"):
        runner._assert_git_source_unchanged(repo, head)


def test_recovar_native_extension_provenance_fingerprints_exact_binaries(tmp_path):
    repo = tmp_path / "repo"
    bind_dir = repo / "recovar" / "relion_bind"
    bind_dir.mkdir(parents=True)
    bind = bind_dir / "_relion_bind_core.cpython-test.so"
    cuda = tmp_path / "libcuda_backproject.so"
    bind.write_bytes(b"relion-bind-test\n")
    cuda.write_bytes(b"cuda-backproject-test\n")

    report = runner._recovar_native_extension_provenance(
        repo,
        {"RECOVAR_CUDA_LIB": str(cuda)},
    )

    assert report["cuda_backproject"] == {
        "path": str(cuda.resolve()),
        "sha256": hashlib.sha256(cuda.read_bytes()).hexdigest(),
        "size_bytes": cuda.stat().st_size,
    }
    assert report["relion_bind_core"] == {
        "path": str(bind.resolve()),
        "sha256": hashlib.sha256(bind.read_bytes()).hexdigest(),
        "size_bytes": bind.stat().st_size,
    }


def test_recovar_native_extension_provenance_fails_closed_without_cuda(tmp_path):
    with pytest.raises(runner.RunError, match="RECOVAR_CUDA_LIB"):
        runner._recovar_native_extension_provenance(tmp_path, {})


def test_recovar_child_environment_requires_cuda_without_legacy_platform_override():
    repo = Path("/active/worktree")
    env = runner._recovar_gpu_env(
        {"JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu", "KEEP": "yes"},
        repo=repo,
    )

    assert env["JAX_PLATFORMS"] == "cuda,cpu"
    assert "JAX_PLATFORM_NAME" not in env
    assert env["KEEP"] == "yes"
    assert env["PYTHONPATH"] == str(repo.resolve())
    assert env["RECOVAR_EXPECTED_REPO_ROOT"] == str(repo.resolve())


def test_initial_model_import_provenance_accepts_active_checkout(monkeypatch):
    repo = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("RECOVAR_EXPECTED_REPO_ROOT", str(repo))

    imported = run_ab_initio._assert_expected_repo_imports()

    assert set(imported) == set(run_ab_initio._CONCRETE_RECOVAR_PROVENANCE_MODULES)


def test_initial_model_import_provenance_rejects_sibling_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("RECOVAR_EXPECTED_REPO_ROOT", str(tmp_path))

    with pytest.raises(RuntimeError, match="InitialModel import provenance failure"):
        run_ab_initio._assert_expected_repo_imports()


def test_parity_cuda_environment_supports_explicit_deterministic_launches():
    env = runner._qualification_cuda_environment(
        {"CUDA_LAUNCH_BLOCKING": "0", "KEEP": "yes"},
        deterministic_cuda=True,
    )

    assert env["CUDA_LAUNCH_BLOCKING"] == "1"
    assert env["KEEP"] == "yes"


def test_parity_cuda_environment_defaults_to_stock_async_mode():
    env = runner._qualification_cuda_environment({}, deterministic_cuda=False)

    assert env["CUDA_LAUNCH_BLOCKING"] == "0"


def test_parity_sbatch_sets_launch_mode_before_gpu_provenance_gate():
    text = SBATCH_PATH.read_text()

    launch_mode = text.index("export CUDA_LAUNCH_BLOCKING=1")
    gpu_gate = text.index('"${PIXI_PY}" - <<\'PY\'')
    assert launch_mode < gpu_gate


def test_runner_records_native_atomic_softmask_counterfactual():
    source = (Path(__file__).resolve().parents[3] / "scripts" / "run_vdam_relion_parity_case.py").read_text()

    assert '"RECOVAR_RELION_NATIVE_ATOMIC_SOFTMASK_REDUCTION"' in source


def test_native_cli_custom_cuda_gate_primes_shared_slicing_dispatch(monkeypatch):
    import jax

    import recovar.cuda_backproject as cuda_backproject
    from recovar.core import slicing

    monkeypatch.setattr(jax, "devices", lambda: [types.SimpleNamespace(platform="cuda")])
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda_backproject, "custom_cuda_requested", lambda: True)
    monkeypatch.setattr(cuda_backproject, "cuda_available", lambda: True)

    report = run_ab_initio._require_custom_cuda_runtime()

    assert report == {
        "default_backend": "gpu",
        "device_platforms": ["cuda"],
        "slicing_on_gpu": True,
        "custom_cuda_requested": True,
        "cuda_available": True,
    }
    assert slicing._on_gpu() is True
    slicing._on_gpu.cache_clear()


def test_native_cli_custom_cuda_gate_rejects_cpu_only_runtime(monkeypatch):
    import jax

    import recovar.cuda_backproject as cuda_backproject
    from recovar.core import slicing

    monkeypatch.setattr(jax, "devices", lambda: [types.SimpleNamespace(platform="cpu")])
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(cuda_backproject, "custom_cuda_requested", lambda: True)
    monkeypatch.setattr(cuda_backproject, "cuda_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires a visible GPU"):
        run_ab_initio._require_custom_cuda_runtime()
    slicing._on_gpu.cache_clear()
