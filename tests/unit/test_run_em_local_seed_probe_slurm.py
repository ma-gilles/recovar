from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "run_em_local_seed_probe_slurm.sh"


def test_local_seed_probe_defaults_stay_memory_safe_for_exact_local(tmp_path):
    output_root = tmp_path / "probe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\necho 12345\n")
    sbatch.chmod(0o755)

    env = os.environ.copy()
    # Exercise the launcher's own explicit qualification default rather than
    # inheriting recovar.jax_config's process-wide user default from pytest.
    env.pop("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EM_LOCAL_PROBE_OUTPUT_ROOT": str(output_root),
            "EM_LOCAL_PROBE_PROFILE_NAME": "unit_probe",
            "EM_LOCAL_PROBE_JAX_CACHE_DIR": str(output_root / "shared_jax_cache"),
            "EM_LOCAL_PROBE_PROJECTOR_CACHE_DIR": str(output_root / "projector_cache"),
            "EM_LOCAL_PROBE_NATIVE_BUILD_ROOT": str(output_root / "native_build"),
            "EM_LOCAL_PROBE_DATA_DIR": str(tmp_path / "data"),
            "EM_LOCAL_PROBE_SEED_NPZ": str(tmp_path / "seed.npz"),
            "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET": "805306368",
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    job_script = output_root / "jobs" / "unit_probe.sbatch"
    assert job_script.exists()
    text = job_script.read_text()
    assert '--image_batch_size "64"' in text
    assert '--rotation_block_size "8192"' in text
    assert f'export RECOVAR_JAX_CACHE_DIR="{output_root / "shared_jax_cache"}"' in text
    assert f'export RECOVAR_RELION_PROJECTOR_CACHE_DIR="{output_root / "projector_cache"}"' in text
    assert 'export RECOVAR_INITIAL_NOISE_CACHE_DIR="' in text
    assert 'touch "${RECOVAR_INITIAL_NOISE_CACHE_DIR}/SAFE_TO_DELETE" || true' in text
    assert '--initial_noise_cache_dir "${RECOVAR_INITIAL_NOISE_CACHE_DIR}"' in text
    assert 'export JAX_COMPILATION_CACHE_DIR="${RECOVAR_JAX_CACHE_DIR}"' in text
    assert 'export JAX_ENABLE_COMPILATION_CACHE="1"' in text
    assert 'export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0"' in text
    assert 'export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0"' in text
    assert 'touch "${JAX_COMPILATION_CACHE_DIR}/SAFE_TO_DELETE" || true' in text
    assert 'touch "${RECOVAR_RELION_PROJECTOR_CACHE_DIR}/SAFE_TO_DELETE" || true' in text
    assert f'export RECOVAR_EM_LOCAL_NATIVE_BUILD_ROOT="{output_root / "native_build"}"' in text
    assert f'export RECOVAR_CUDA_LIB="{output_root / "native_build" / "cuda" / "libcuda_backproject.so"}"' in text
    assert f'export RECOVAR_RELION_BIND_BUILD_DIR="{output_root / "native_build" / "relion_bind"}"' in text
    assert "EM_LOCAL_PROBE_FORCE_NATIVE_REBUILD=0" in text
    assert "EM_LOCAL_PROBE_FORCE_INSTALL=0" in text
    assert 'touch "${RECOVAR_EM_LOCAL_NATIVE_BUILD_ROOT}/SAFE_TO_DELETE" || true' in text
    assert "Reusing editable RECOVAR install bound to this checkout" in text
    assert 'flock 8' in text
    assert "NEED_RELION_BIND_BUILD=0" in text
    assert "Reusing RELION binding:" in text
    assert 'make -C recovar/cuda LIB="${RECOVAR_CUDA_LIB}" clean all' not in text
    assert 'make -C recovar/cuda LIB="${RECOVAR_CUDA_LIB}" all' in text
    assert 'export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION="0.20"' in text
    assert 'export RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET="805306368"' in text
    assert (
        "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET="
        "${RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET:-<unset>}"
    ) in text
    assert 'export RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST=""' in text
    assert "RECOVAR_LOCAL_XHALF_BATCH_GUARD=${RECOVAR_LOCAL_XHALF_BATCH_GUARD:-<unset>}" in text
    assert (
        "RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT="
        "${RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT:-<unset>}"
    ) in text
    assert (
        "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB="
        "${RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB:-<unset>}"
    ) in text
    assert "Stop after profile: 0" in text
    assert "Stop after local search score-only: 1" in text
    assert '--local_search_profile "off"' in text
    assert 'if [[ "1" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then' in text
    assert 'elif [[ "0" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then' in text
    assert "EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search_score_only)" in text
    assert "--skip-large-outputs" in text
    assert (output_root / "jobs" / "unit_probe.jobid").read_text().strip() == "12345"


def test_local_seed_probe_can_opt_into_full_profile_mstep(tmp_path):
    output_root = tmp_path / "probe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\necho 12345\n")
    sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EM_LOCAL_PROBE_OUTPUT_ROOT": str(output_root),
            "EM_LOCAL_PROBE_PROFILE_NAME": "unit_probe",
            "EM_LOCAL_PROBE_DATA_DIR": str(tmp_path / "data"),
            "EM_LOCAL_PROBE_SEED_NPZ": str(tmp_path / "seed.npz"),
            "EM_LOCAL_PROBE_STOP_AFTER_LOCAL_SEARCH_SCORE_ONLY": "0",
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    text = (output_root / "jobs" / "unit_probe.sbatch").read_text()
    assert "Stop after profile: 1" in text
    assert "Stop after local search score-only: 0" in text
    assert '--local_search_profile "on"' in text
    assert 'if [[ "0" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then' in text
    assert 'elif [[ "1" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then' in text
    assert text.index('if [[ "0" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then') < text.index(
        "EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search_score_only)"
    )
    assert "--stop_after_local_search_profile" in text


def test_local_seed_probe_save_intermediates_skips_unregularized_by_default(tmp_path):
    output_root = tmp_path / "probe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\necho 12345\n")
    sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EM_LOCAL_PROBE_OUTPUT_ROOT": str(output_root),
            "EM_LOCAL_PROBE_PROFILE_NAME": "unit_probe",
            "EM_LOCAL_PROBE_DATA_DIR": str(tmp_path / "data"),
            "EM_LOCAL_PROBE_SEED_NPZ": str(tmp_path / "seed.npz"),
            "EM_LOCAL_PROBE_SAVE_INTERMEDIATES": "1",
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    text = (output_root / "jobs" / "unit_probe.sbatch").read_text()
    assert '--save_intermediates_dir "${OUT_DIR}/intermediates"' in text
    assert "--save_intermediates_skip_unregularized" in text


def test_local_seed_probe_profile_off_uses_fast_local_search_stop(tmp_path):
    output_root = tmp_path / "probe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\necho 12345\n")
    sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EM_LOCAL_PROBE_OUTPUT_ROOT": str(output_root),
            "EM_LOCAL_PROBE_PROFILE_NAME": "unit_probe",
            "EM_LOCAL_PROBE_DATA_DIR": str(tmp_path / "data"),
            "EM_LOCAL_PROBE_SEED_NPZ": str(tmp_path / "seed.npz"),
            "EM_LOCAL_PROBE_LOCAL_SEARCH_PROFILE": "off",
            "EM_LOCAL_PROBE_STOP_AFTER_LOCAL_SEARCH_SCORE_ONLY": "0",
            "EM_LOCAL_PROBE_STOP_AFTER_PROFILE": "1",
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    text = (output_root / "jobs" / "unit_probe.sbatch").read_text()
    assert '--local_search_profile "off"' in text
    assert 'if [[ "off" == "off" ]]; then\n    EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search)' in text


def test_local_seed_probe_can_use_score_only_local_search_stop(tmp_path):
    output_root = tmp_path / "probe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\necho 12345\n")
    sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EM_LOCAL_PROBE_OUTPUT_ROOT": str(output_root),
            "EM_LOCAL_PROBE_PROFILE_NAME": "unit_probe",
            "EM_LOCAL_PROBE_DATA_DIR": str(tmp_path / "data"),
            "EM_LOCAL_PROBE_SEED_NPZ": str(tmp_path / "seed.npz"),
            "EM_LOCAL_PROBE_STOP_AFTER_LOCAL_SEARCH_SCORE_ONLY": "1",
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    text = (output_root / "jobs" / "unit_probe.sbatch").read_text()
    assert "Stop after local search score-only: 1" in text
    assert "Local search profile mode: off" in text
    assert '--local_search_profile "off"' in text
    assert "EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search_score_only)" in text
    assert text.index("EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search_score_only)") < text.index(
        "EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search_profile)"
    )


def test_local_seed_probe_can_use_diagnostic_single_half(tmp_path):
    output_root = tmp_path / "probe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\necho 12345\n")
    sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EM_LOCAL_PROBE_OUTPUT_ROOT": str(output_root),
            "EM_LOCAL_PROBE_PROFILE_NAME": "unit_probe",
            "EM_LOCAL_PROBE_DATA_DIR": str(tmp_path / "data"),
            "EM_LOCAL_PROBE_SEED_NPZ": str(tmp_path / "seed.npz"),
            "EM_LOCAL_PROBE_STOP_AFTER_LOCAL_SEARCH_SCORE_ONLY": "1",
            "EM_LOCAL_PROBE_DIAGNOSTIC_SINGLE_HALF": "1",
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    text = (output_root / "jobs" / "unit_probe.sbatch").read_text()
    assert "Local search profile mode: off" in text
    assert "Diagnostic single half: 1" in text
    assert "EXTRA_REFINEMENT_ARGS+=(--diagnostic_single_half)" in text
    assert text.index("EXTRA_REFINEMENT_ARGS+=(--diagnostic_single_half)") < text.index(
        "EXTRA_REFINEMENT_ARGS+=(--skip-large-outputs)"
    )


def test_local_seed_probe_single_half_defaults_to_profile_off(tmp_path):
    output_root = tmp_path / "probe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\necho 12345\n")
    sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EM_LOCAL_PROBE_OUTPUT_ROOT": str(output_root),
            "EM_LOCAL_PROBE_PROFILE_NAME": "unit_probe",
            "EM_LOCAL_PROBE_DATA_DIR": str(tmp_path / "data"),
            "EM_LOCAL_PROBE_SEED_NPZ": str(tmp_path / "seed.npz"),
            "EM_LOCAL_PROBE_DIAGNOSTIC_SINGLE_HALF": "1",
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    text = (output_root / "jobs" / "unit_probe.sbatch").read_text()
    assert "Local search profile mode: off" in text
    assert '--local_search_profile "off"' in text
    assert "EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search)" in text
    assert "EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search_profile)" in text
    assert text.index("EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search)") < text.index(
        "EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search_profile)"
    )


def test_local_seed_probe_can_reuse_seed_noise_for_diagnostics(tmp_path):
    output_root = tmp_path / "probe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\necho 12345\n")
    sbatch.chmod(0o755)

    seed_npz = tmp_path / "seed.npz"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EM_LOCAL_PROBE_OUTPUT_ROOT": str(output_root),
            "EM_LOCAL_PROBE_PROFILE_NAME": "unit_probe",
            "EM_LOCAL_PROBE_DATA_DIR": str(tmp_path / "data"),
            "EM_LOCAL_PROBE_SEED_NPZ": str(seed_npz),
            "EM_LOCAL_PROBE_POSE_ITER": "7",
            "EM_LOCAL_PROBE_INIT_NOISE_FROM_SEED_NPZ": "1",
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    text = (output_root / "jobs" / "unit_probe.sbatch").read_text()
    assert "Init noise from seed NPZ: 1" in text
    assert "Init noise iter: 7" in text
    assert f'EXTRA_REFINEMENT_ARGS+=(--init_noise_from_npz "{seed_npz}" --init_noise_iter "7")' in text
    assert (
        text.index(f'EXTRA_REFINEMENT_ARGS+=(--init_noise_from_npz "{seed_npz}" --init_noise_iter "7")')
        < text.index('elif [[ -n "${RECOVAR_INITIAL_NOISE_CACHE_DIR}" ]]; then')
        < text.index('--initial_noise_cache_dir "${RECOVAR_INITIAL_NOISE_CACHE_DIR}"')
    )


def test_local_seed_probe_can_disable_initial_noise_cache(tmp_path):
    output_root = tmp_path / "probe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\necho 12345\n")
    sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EM_LOCAL_PROBE_OUTPUT_ROOT": str(output_root),
            "EM_LOCAL_PROBE_PROFILE_NAME": "unit_probe",
            "EM_LOCAL_PROBE_DATA_DIR": str(tmp_path / "data"),
            "EM_LOCAL_PROBE_SEED_NPZ": str(tmp_path / "seed.npz"),
            "EM_LOCAL_PROBE_INITIAL_NOISE_CACHE_DIR": "",
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    text = (output_root / "jobs" / "unit_probe.sbatch").read_text()
    assert 'export RECOVAR_INITIAL_NOISE_CACHE_DIR=""' in text
    assert 'elif [[ -n "${RECOVAR_INITIAL_NOISE_CACHE_DIR}" ]]; then' in text
    assert '--initial_noise_cache_dir "${RECOVAR_INITIAL_NOISE_CACHE_DIR}"' in text


def test_local_seed_probe_records_auto_microbatch_boost(tmp_path):
    output_root = tmp_path / "probe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\necho 12345\n")
    sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EM_LOCAL_PROBE_OUTPUT_ROOT": str(output_root),
            "EM_LOCAL_PROBE_PROFILE_NAME": "unit_probe",
            "EM_LOCAL_PROBE_DATA_DIR": str(tmp_path / "data"),
            "EM_LOCAL_PROBE_SEED_NPZ": str(tmp_path / "seed.npz"),
            "RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST": "3",
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    text = (output_root / "jobs" / "unit_probe.sbatch").read_text()
    assert "RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST=3" in text
    assert 'export RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST="3"' in text


def test_local_seed_probe_can_force_native_rebuild(tmp_path):
    output_root = tmp_path / "probe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\necho 12345\n")
    sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EM_LOCAL_PROBE_OUTPUT_ROOT": str(output_root),
            "EM_LOCAL_PROBE_PROFILE_NAME": "unit_probe",
            "EM_LOCAL_PROBE_DATA_DIR": str(tmp_path / "data"),
            "EM_LOCAL_PROBE_SEED_NPZ": str(tmp_path / "seed.npz"),
            "EM_LOCAL_PROBE_FORCE_NATIVE_REBUILD": "1",
            "EM_LOCAL_PROBE_FORCE_INSTALL": "1",
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    text = (output_root / "jobs" / "unit_probe.sbatch").read_text()
    assert "EM_LOCAL_PROBE_FORCE_NATIVE_REBUILD=1" in text
    assert "EM_LOCAL_PROBE_FORCE_INSTALL=1" in text
    assert 'make -C recovar/cuda LIB="${RECOVAR_CUDA_LIB}" clean' in text
    assert 'make -C recovar/cuda LIB="${RECOVAR_CUDA_LIB}" all' in text
