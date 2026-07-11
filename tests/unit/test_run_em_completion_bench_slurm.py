from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "run_em_completion_bench_slurm.sh"


def test_completion_jobs_reuse_setup_relion_binding_build_dir(tmp_path):
    scratch = tmp_path / "scratch"
    env = os.environ.copy()
    env.update(
        {
            "EM_COMPLETION_SCRATCH_DIR": str(scratch),
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
            "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP": "pair_sparse",
            "RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES": "4294967296",
            "RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES": "2147483648",
            "RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES": "1073741824",
            "RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS": "500",
            "RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS": "120",
            "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET": "805306368",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", "--k1-only"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
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
    shared_export = f'export RECOVAR_RELION_BIND_BUILD_DIR="{scratch}/relion_bind_build/shared"'
    jax_cache_export = f'export RECOVAR_JAX_CACHE_DIR="{scratch}/jax_cache"'
    assert shared_export in setup_text
    assert shared_export in k1_text
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
    assert 'pixi run --frozen python recovar/relion_bind/build.py' in setup_text
    assert 'rm -rf "${RECOVAR_RELION_BIND_BUILD_DIR:?}"' in setup_text
    assert 'rm -rf "${RECOVAR_RELION_BIND_BUILD_DIR:?}"' not in k1_text
    assert "recovar/relion_bind/build.py" not in k1_text


def test_completion_k4_resource_overrides_are_written(tmp_path):
    scratch = tmp_path / "scratch"
    env = os.environ.copy()
    env.update(
        {
            "EM_COMPLETION_SCRATCH_DIR": str(scratch),
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
            "RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES": "3221225472",
            "RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES": "1610612736",
            "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO": "0.5",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", "--k4-only"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
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
    env = os.environ.copy()
    env.update(
        {
            "EM_COMPLETION_SCRATCH_DIR": str(scratch),
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
            "EM_COMPLETION_SUMMARY_PARTITION": "cpu",
            "EM_COMPLETION_SUMMARY_CONSTRAINT": "",
        }
    )
    env.pop("EM_COMPLETION_SETUP_PARTITION", None)
    env.pop("EM_COMPLETION_SETUP_CONSTRAINT", None)
    env.pop("EM_COMPLETION_SETUP_GRES", None)

    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", "--k4-only"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    setup_script = scratch / "jobs" / "em_completion_setup.sh"
    submission_env = scratch / "submission.env"
    assert setup_script.exists()
    assert submission_env.exists()
    assert "#SBATCH --partition=cpu" in setup_script.read_text()
    assert "EM_COMPLETION_SETUP_PARTITION=cpu" in submission_env.read_text()
