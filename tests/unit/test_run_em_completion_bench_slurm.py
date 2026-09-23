from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "run_em_completion_bench_slurm.sh"
DEFAULT_K1_RELION_DIR = Path(
    "/scratch/gpfs/GILLES/mg6942/em_relion_proj/pdb_k1_g256_n100000_noise1_bf80_20260516/relion_autorefine_k1_it015_os1"
)


def _launcher_env(tmp_path, scratch):
    """Launcher environment shared by the dry-run tests."""
    env = os.environ.copy()
    env.update(
        {
            "EM_COMPLETION_SCRATCH_DIR": str(scratch),
            "EM_COMPLETION_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
            "EM_COMPLETION_SETUP_PARTITION": "cpu",
            "EM_COMPLETION_SETUP_CONSTRAINT": "",
            "EM_COMPLETION_SUMMARY_PARTITION": "cpu",
            "EM_COMPLETION_SUMMARY_CONSTRAINT": "",
            "EM_COMPLETION_SUMMARY_GRES": "",
            "K1_MEM": "128G",
            "K1_TIME_LIMIT": "04:00:00",
            "RELION_SRC_DIR": str(tmp_path / "relion_src"),
            "RELION_REFINE_MPI": "/bin/true",
        }
    )
    return env


def _run_launcher(env, scope):
    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", scope],
        cwd=REPO_ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert proc.returncode == 0, proc.stdout


def test_completion_jobs_reuse_setup_relion_binding_build_dir(tmp_path):
    scratch = tmp_path / "scratch"
    runtime = tmp_path / "runtime"
    env = _launcher_env(tmp_path, scratch)
    env.update({
        'RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP': 'pair_sparse',
        'RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES': '4294967296',
        'RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES': '2147483648',
        'RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES': '1073741824',
        'RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS': '500',
        'RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS': '120',
        'RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET': '805306368',
    })

    _run_launcher(env, "--k1-only")
    setup_script = scratch / "jobs" / "em_completion_setup.sh"
    k1_script = scratch / "jobs" / "em_completion_k1_100k256.sh"
    summary_script = scratch / "jobs" / "em_completion_summary.sh"
    submission_env = scratch / "submission.env"
    assert setup_script.exists()
    assert k1_script.exists()
    assert summary_script.exists()
    assert submission_env.exists()

    setup_text = setup_script.read_text()
    k1_text = k1_script.read_text()
    summary_text = summary_script.read_text()
    submission_env_text = submission_env.read_text()
    submission_fingerprint = (
        scratch / "provenance" / "submission" / "git_worktree_fingerprint.sha256"
    ).read_text().strip()
    shared_export = f'export RECOVAR_RELION_BIND_BUILD_DIR="{scratch}/relion_bind_build/shared"'
    jax_cache_export = f'export RECOVAR_JAX_CACHE_DIR="{scratch}/jax_cache"'
    assert shared_export in setup_text
    assert shared_export in k1_text
    assert f'export RELION_SRC_DIR="{tmp_path / "relion_src"}"' in setup_text
    assert f"RELION_SRC_DIR={tmp_path / 'relion_src'}" in submission_env_text
    assert jax_cache_export in setup_text
    assert jax_cache_export in k1_text
    assert jax_cache_export in summary_text
    assert "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP=pair_sparse" in k1_text
    assert "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP=pair_sparse" in submission_env_text
    assert "RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES=4294967296" in k1_text
    assert "RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES=4294967296" in submission_env_text
    assert "RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES=2147483648" in k1_text
    assert "RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES=2147483648" in submission_env_text
    assert "RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES=1073741824" in k1_text
    assert "RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES=1073741824" in submission_env_text
    assert "RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0" in k1_text
    assert "RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0" in submission_env_text
    assert "RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS=500" in k1_text
    assert "RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS=500" in submission_env_text
    assert "RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS=120" in k1_text
    assert "RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS=120" in submission_env_text
    assert "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET=805306368" in k1_text
    assert "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET=805306368" in submission_env_text
    assert '--image_batch_size "187"' in k1_text
    assert '--rotation_block_size "8192"' in k1_text
    assert "K1_IMAGE_BATCH_SIZE=187" in submission_env_text
    assert "K1_ROTATION_BLOCK_SIZE=8192" in submission_env_text
    assert "K1_TRAJECTORY_MODE=autonomous" in submission_env_text
    assert "K1_SAVE_INTERMEDIATES=1" in submission_env_text
    assert 'mkdir -p "${OUTPUT_DIR}/intermediates"' in k1_text
    assert '--save_intermediates_dir "${OUTPUT_DIR}/intermediates"' in k1_text
    assert "--save_intermediates_skip_unregularized" in k1_text
    assert "--local_search_profile off" in k1_text
    assert "--local-search-profile" not in k1_text
    assert "RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0" in submission_env_text
    assert "TRAJECTORY_ARGS=(--relion_init_dir" in k1_text
    assert 'if [[ "autonomous" == "relion-replay" ]]' in k1_text
    assert "export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40" in k1_text
    assert "RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40" in submission_env_text
    assert "#SBATCH --mem=128G" in k1_text
    assert "#SBATCH --time=04:00:00" in k1_text
    assert "K1_MEM=128G" in submission_env_text
    assert "K1_TIME_LIMIT=04:00:00" in submission_env_text
    assert 'export JAX_COMPILATION_CACHE_DIR="${RECOVAR_JAX_CACHE_DIR}"' in setup_text
    assert 'export JAX_COMPILATION_CACHE_DIR="${RECOVAR_JAX_CACHE_DIR}"' in k1_text
    assert 'export JAX_COMPILATION_CACHE_DIR="${RECOVAR_JAX_CACHE_DIR}"' in summary_text
    assert "unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA" in setup_text
    assert "unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA" in k1_text
    completion_python = scratch / "venv" / "bin" / "python"
    python_export = f'export PIXI_PY="{completion_python}"'
    for generated_text in (setup_text, k1_text, summary_text):
        assert python_export in generated_text
        assert "pixi run" not in generated_text
        assert "git symbolic-ref --short HEAD ||" not in generated_text
    assert "export PIP_NO_INDEX=1" in setup_text
    base_pixi_python = Path(
        env.get(
            "EM_COMPLETION_PIXI_PY",
            REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "python",
        )
    ).resolve()
    pixi_root = base_pixi_python.parent.parent
    assert (
        f'export CMAKE_INCLUDE_PATH="{pixi_root}/include/fftw:'
        f'{pixi_root}/include:${{CMAKE_INCLUDE_PATH:-}}"'
    ) in setup_text
    assert (
        f'export CMAKE_LIBRARY_PATH="{pixi_root}/lib:${{CMAKE_LIBRARY_PATH:-}}"'
    ) in setup_text
    assert '-m venv --system-site-packages "${EM_COMPLETION_VENV}"' in setup_text
    assert (
        '"${PIXI_PY}" -m pip install -e . --no-deps --no-build-isolation '
        "--ignore-installed"
    ) in setup_text
    assert '"${PIXI_PY}" recovar/relion_bind/build.py' in setup_text
    assert "export JAX_PLATFORMS=cpu" in setup_text
    assert "export JAX_PLATFORM_NAME=cpu" in setup_text
    assert "export RECOVAR_DISABLE_CUDA=1" in setup_text
    assert 'export CUDA_VISIBLE_DEVICES=""' in setup_text
    assert 'rm -rf "${RECOVAR_RELION_BIND_BUILD_DIR:?}"' in setup_text
    assert 'rm -rf "${RECOVAR_RELION_BIND_BUILD_DIR:?}"' not in k1_text
    assert "recovar/relion_bind/build.py" not in k1_text
    assert (scratch / "SAFE_TO_DELETE").exists()
    assert (runtime / "SAFE_TO_DELETE").exists()
    for generated_text in (setup_text, k1_text, summary_text):
        assert f'export TMPDIR="{runtime}/' in generated_text
        assert "queued-job Git HEAD drift" in generated_text
        assert "queued-job worktree fingerprint drift" in generated_text
        assert submission_fingerprint in generated_text
        assert (
            'nvidia-smi -q > "${JOB_GIT_PROVENANCE_DIR}/nvidia_smi.txt"'
            in generated_text
        )
    assert "RELION_REFINE_MPI_SHA256=" in setup_text
    assert 'RELION_REFINE_MPI_BIN="/usr/bin/true"' in setup_text
    assert 'module load "relion/5.0.1/gcc-11.5.0-gpu"' not in setup_text
    assert 'nvidia-smi -q > "${OUTPUT_DIR}/nvidia_smi.txt"' in k1_text
    assert "ERROR: completion summary requires a clean worktree" in summary_text
    assert "ERROR: upstream job ${job_id} state is" in summary_text
    assert 'summarizer_status="$?"' in summary_text
    assert 'exit "${SUMMARY_STATUS}"' in summary_text
    assert f"RUNTIME_ROOT={runtime}" in submission_env_text
    assert f"EM_COMPLETION_VENV={scratch / 'venv'}" in submission_env_text
    assert f"PIXI_PY={completion_python}" in submission_env_text
    assert (
        f"SUBMISSION_GIT_WORKTREE_FINGERPRINT_SHA256={submission_fingerprint}"
        in submission_env_text
    )


def test_completion_jobs_preread_stacks_and_record_io_placement(tmp_path):
    """Particle stacks are read into host memory once, and every job says so.

    Without the preread each iteration re-reads its subset from the shared
    filesystem, which the VDAM workstream measured as roughly half the K=1
    100k/256 wall. The loader caps the per-file allocation, so leaving the flag
    on is safe for stacks above the cap.
    """
    scratch = tmp_path / "scratch"
    env = _launcher_env(tmp_path, scratch)
    _run_launcher(env, "--k1-only")
    k1_text = (scratch / "jobs" / "em_completion_k1_100k256.sh").read_text()
    assert 'export RECOVAR_PREREAD_IMAGES="${RECOVAR_PREREAD_IMAGES:-1}"' in k1_text
    assert 'export RECOVAR_PREREAD_MAX_GB="${RECOVAR_PREREAD_MAX_GB:-64}"' in k1_text
    assert 'echo "RECOVAR_PREREAD_IMAGES=${RECOVAR_PREREAD_IMAGES} RECOVAR_PREREAD_MAX_GB=${RECOVAR_PREREAD_MAX_GB}"' in k1_text
    assert 'echo "RECOVAR_CACHE_DIR=${RECOVAR_CACHE_DIR:-<staging disabled>}"' in k1_text
    assert 'echo "JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR}"' in k1_text


def test_completion_k1_relion_replay_mode_is_explicit(tmp_path):
    scratch = tmp_path / "scratch"
    runtime = tmp_path / "runtime"
    env = os.environ.copy()
    env.update(
        {
            "EM_COMPLETION_SCRATCH_DIR": str(scratch),
            "EM_COMPLETION_RUNTIME_ROOT": str(runtime),
            "K1_TRAJECTORY_MODE": "relion-replay",
        }
    )

    _run_launcher(env, "--k1-only")
    k1_text = (scratch / "jobs" / "em_completion_k1_100k256.sh").read_text()
    submission_env_text = (scratch / "submission.env").read_text()
    assert 'if [[ "relion-replay" == "relion-replay" ]]' in k1_text
    assert "--relion_init_dir" in k1_text
    assert "--perturb_replay_relion_dir" in k1_text
    assert "K1_TRAJECTORY_MODE=relion-replay" in submission_env_text
    assert "RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=1" in submission_env_text


def test_completion_k1_intermediates_can_be_disabled(tmp_path):
    scratch = tmp_path / "scratch"
    runtime = tmp_path / "runtime"
    env = os.environ.copy()
    env.update(
        {
            "EM_COMPLETION_SCRATCH_DIR": str(scratch),
            "EM_COMPLETION_RUNTIME_ROOT": str(runtime),
            "K1_SAVE_INTERMEDIATES": "0",
        }
    )

    _run_launcher(env, "--k1-only")
    k1_text = (scratch / "jobs" / "em_completion_k1_100k256.sh").read_text()
    submission_env_text = (scratch / "submission.env").read_text()
    assert "K1_SAVE_INTERMEDIATES=0" in submission_env_text
    assert "--save_intermediates_dir" not in k1_text
    assert "--save_intermediates_skip_unregularized" not in k1_text
    assert "--local_search_profile off" not in k1_text


def test_completion_k4_resource_overrides_are_written(tmp_path):
    scratch = tmp_path / "scratch"
    runtime = tmp_path / "runtime"
    dispatch_schedule = tmp_path / "dispatch_schedule.npz"
    dispatch_schedule.write_bytes(b"placeholder")
    env = os.environ.copy()
    env.update(
        {
            "EM_COMPLETION_SCRATCH_DIR": str(scratch),
            "EM_COMPLETION_RUNTIME_ROOT": str(runtime),
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
            "EM_COMPLETION_SETUP_PARTITION": "cpu",
            "EM_COMPLETION_SETUP_CONSTRAINT": "",
            "EM_COMPLETION_SUMMARY_PARTITION": "cpu",
            "EM_COMPLETION_SUMMARY_CONSTRAINT": "",
            "EM_COMPLETION_SUMMARY_GRES": "",
            "K4_MEM": "128G",
            "K4_TIME_LIMIT": "04:00:00",
            "K4_RELION_DISPATCH_SCHEDULE": str(dispatch_schedule),
            "RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES": "3221225472",
            "RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES": "1610612736",
            "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO": "0.5",
        }
    )

    _run_launcher(env, "--k4-only")
    k4_script = scratch / "jobs" / "em_completion_k4_100k256.sh"
    submission_env = scratch / "submission.env"
    assert k4_script.exists()
    assert submission_env.exists()

    k4_text = k4_script.read_text()
    submission_env_text = submission_env.read_text()
    assert "#SBATCH --mem=128G" in k4_text
    assert "#SBATCH --time=04:00:00" in k4_text
    assert '--image_batch_size "50"' in k4_text
    assert '--rotation_block_size "2000"' in k4_text
    assert "K4_IMAGE_BATCH_SIZE=50" in submission_env_text
    assert "K4_ROTATION_BLOCK_SIZE=2000" in submission_env_text
    assert "K4_MEM=128G" in submission_env_text
    assert "K4_TIME_LIMIT=04:00:00" in submission_env_text
    assert f'--relion-dispatch-schedule "{dispatch_schedule}"' in k4_text
    assert f"K4_RELION_DISPATCH_SCHEDULE={dispatch_schedule}" in submission_env_text
    assert 'mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/intermediates"' in k4_text
    assert '--save_intermediates_dir "${OUTPUT_DIR}/intermediates"' in k4_text
    assert "--save_intermediates_skip_unregularized" in k4_text
    assert "RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES=3221225472" in k4_text
    assert "RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES=3221225472" in submission_env_text
    assert "RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES=1610612736" in k4_text
    assert "RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES=1610612736" in submission_env_text
    assert "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO=0.5" in k4_text
    assert (
        "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO=0.5"
        in submission_env_text
    )


def test_completion_setup_defaults_to_cpu_partition(tmp_path):
    scratch = tmp_path / "scratch"
    runtime = tmp_path / "runtime"
    dispatch_schedule = tmp_path / "dispatch_schedule.npz"
    dispatch_schedule.write_bytes(b"placeholder")
    env = os.environ.copy()
    env.update(
        {
            "EM_COMPLETION_SCRATCH_DIR": str(scratch),
            "EM_COMPLETION_RUNTIME_ROOT": str(runtime),
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
            "EM_COMPLETION_SUMMARY_PARTITION": "cpu",
            "EM_COMPLETION_SUMMARY_CONSTRAINT": "",
            "K4_RELION_DISPATCH_SCHEDULE": str(dispatch_schedule),
        }
    )
    env.pop("EM_COMPLETION_SETUP_PARTITION", None)
    env.pop("EM_COMPLETION_SETUP_CONSTRAINT", None)
    env.pop("EM_COMPLETION_SETUP_GRES", None)

    _run_launcher(env, "--k4-only")
    setup_script = scratch / "jobs" / "em_completion_setup.sh"
    submission_env = scratch / "submission.env"
    assert setup_script.exists()
    assert submission_env.exists()
    assert "#SBATCH --partition=cpu" in setup_script.read_text()
    assert "EM_COMPLETION_SETUP_PARTITION=cpu" in submission_env.read_text()


def test_completion_jobs_pin_production_jax_memory_fraction(tmp_path):
    """GPU jobs do not inherit the submitting shell's JAX memory cap.

    The della login-node profile exports ``XLA_PYTHON_CLIENT_MEM_FRACTION=.50``
    and sbatch propagates it; K=1 100k/256 completions then planned against a
    42 GB limit on 80 GB GPUs. Every job shares the preamble, so K=1 covers K=4.
    """
    scratch = tmp_path / "scratch"
    env = _launcher_env(tmp_path, scratch)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
    _run_launcher(env, "--k1-only")
    k1_text = (scratch / "jobs" / "em_completion_k1_100k256.sh").read_text()
    assert "export XLA_PYTHON_CLIENT_MEM_FRACTION=.90\n" in k1_text
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION=.50" not in k1_text


def _fake_k1_fixture(tmp_path, version_line):
    """A K=1 fixture tree the launcher accepts, with a chosen optimiser header."""
    data = tmp_path / "k1_data"
    relion = data / "relion"
    relion.mkdir(parents=True)
    for name in ("particles.star", "reference_gt.mrc"):
        (data / name).write_text("x\n")
    names = [
        "run_it000_data.star", "run_it000_half1_model.star", "run_it000_half2_model.star",
        "run_it015_half1_class001.mrc", "run_it015_half2_class001.mrc", "run_it016_data.star",
        "run_it016_half1_model.star", "run_it016_half2_model.star", "run_it016_optimiser.star",
        "run_sampling.star", "run_optimiser.star", "run_it016_half1_class001.mrc",
        "run_it016_half2_class001.mrc",
    ]
    for name in names:
        (relion / name).write_text("x\n")
    (relion / "run_it000_half1_model.star").write_text("_rlnTau2FudgeFactor 1.000000\n")
    (relion / "run_it000_data.star").write_text("loop_\n_rlnRandomSubset #1\n1\n2\n")
    # The real optimiser header with only its version line replaced, so every other launcher check still sees it.
    real = (DEFAULT_K1_RELION_DIR / "run_it000_optimiser.star").read_text().splitlines(keepends=True)
    (relion / "run_it000_optimiser.star").write_text(f"# RELION optimiser; version {version_line}\n" + "".join(real[1:]))
    return data, relion


def test_completion_k1_particle_order_follows_the_oracle_build(tmp_path):
    """Autonomous K=1 runs own RELION's fresh order, whose first 100 rows are the accuracy trials.

    The default fixture's oracle was written by RELION 5.0.1 f2c1a3 (mt19937/std::shuffle);
    running it with the legacy libc order picks different accuracy trial particles, biases the
    expected-accuracy estimate low and flips knife-edge angular-sampling decisions.
    """
    scratch = tmp_path / "scratch"
    env = _launcher_env(tmp_path, scratch)
    _run_launcher(env, "--k1-only")
    k1_text = (scratch / "jobs" / "em_completion_k1_100k256.sh").read_text()
    assert "TRAJECTORY_ARGS+=(--relion-particle-shuffle mt19937)\n" in k1_text

    for version, expected in (("5.0.1-commit-d476e6", "legacy"), ("5.0.1-commit-f2c1a3", "mt19937")):
        case = tmp_path / expected
        data, relion = _fake_k1_fixture(case, version)
        env = _launcher_env(case, case / "scratch")
        env.update({"K1_DATA_DIR": str(data), "K1_RELION_DIR": str(relion)})
        _run_launcher(env, "--k1-only")
        text = (case / "scratch" / "jobs" / "em_completion_k1_100k256.sh").read_text()
        assert f"TRAJECTORY_ARGS+=(--relion-particle-shuffle {expected})\n" in text


def test_completion_k1_particle_order_fails_closed_on_an_unknown_build(tmp_path):
    data, relion = _fake_k1_fixture(tmp_path, "5.0.1-commit-000000")
    env = _launcher_env(tmp_path, tmp_path / "scratch")
    env.update({"K1_DATA_DIR": str(data), "K1_RELION_DIR": str(relion)})
    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", "--k1-only"],
        cwd=REPO_ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert proc.returncode == 2, proc.stdout
    assert "cannot infer the K=1 RELION particle order" in proc.stdout


def test_completion_k1_replay_does_not_request_a_fresh_order(tmp_path):
    scratch = tmp_path / "scratch"
    env = _launcher_env(tmp_path, scratch)
    env["K1_TRAJECTORY_MODE"] = "relion-replay"
    _run_launcher(env, "--k1-only")
    k1_text = (scratch / "jobs" / "em_completion_k1_100k256.sh").read_text()
    assert "--relion-particle-shuffle" not in k1_text
