from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import starfile

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "run_em_k1_robustness_matrix_slurm.sh"
IDENTITY_CTF_STAR = REPO_ROOT / "scripts" / "make_relion_identity_ctf_star.py"


def _relion_src_fixture(tmp_path: Path) -> Path:
    relion_src = tmp_path / "relion_src"
    relion_src.mkdir(exist_ok=True)
    (relion_src / "projector.h").write_text("// unit-test fixture\n")
    return relion_src


def _dry_run_launcher(tmp_path, *, case: str, extra_env: dict[str, str] | None = None) -> tuple[subprocess.CompletedProcess, Path]:
    scratch = tmp_path / "scratch"
    relion_src = _relion_src_fixture(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "EM_K1_MATRIX_SCRATCH_DIR": str(scratch),
            "EM_K1_MATRIX_CASES": case,
            "EM_K1_MATRIX_RUN_RELION": "1",
            "EM_K1_MATRIX_SETUP_PARTITION": "cpu",
            "EM_K1_MATRIX_SUMMARY_PARTITION": "cpu",
            "EM_K1_MATRIX_SUMMARY_CONSTRAINT": "",
            "EM_K1_MATRIX_SUMMARY_GRES": "",
            "SBATCH_ACCOUNT": "gilles",
            "SBATCH_PARTITION": "cryoem",
            "SBATCH_CONSTRAINT": "",
            "RELION_SRC_DIR": str(relion_src),
        }
    )
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc, scratch


def test_relion_replay_stars_are_selected_as_matched_pairs(tmp_path):
    proc, scratch = _dry_run_launcher(tmp_path, case="15")

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_15_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    assert 'try_relion_replay_pair "${RELION_DIR}/run_data.star" "${RELION_DIR}/run_optimiser.star"' in text
    assert (
        'try_relion_replay_pair "${RELION_DIR}/run_it${RELION_ITER_PADDED}_data.star" '
        '"${RELION_DIR}/run_it${RELION_ITER_PADDED}_optimiser.star"'
    ) in text
    assert 'try_relion_replay_pair "${RELION_DIR}/run_it000_data.star" "${RELION_DIR}/run_it000_optimiser.star"' in text
    assert '"${RELION_DIR}/run_it000_optimiser.star" \\\n    "${RELION_DIR}/run_optimiser.star"' not in text
    assert '--relion_init_dir "${RELION_DIR}"' in text
    assert '--perturb_replay_relion_dir "${RELION_DIR}"' in text
    assert "LATEST_RELION_SAMPLING_ITER=" in text
    assert "Strict RELION replay: capping RECOVAR max_iter" in text
    assert "EM_K1_MATRIX_TRAJECTORY_MODE=controlled" in text
    assert "EM_K1_MATRIX_TRAJECTORY_MODE=controlled" in (scratch / "submission.env").read_text()


def test_autonomous_trajectory_uses_only_iter0_boundary_and_never_emits_replay_logic(tmp_path):
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="15",
        extra_env={"EM_K1_MATRIX_TRAJECTORY_MODE": "autonomous"},
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_15_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    submission = (scratch / "submission.env").read_text()

    assert 'RELION_HALF_SET_STAR="${RELION_DIR}/run_it000_data.star"' in text
    assert 'RELION_OPTIMISER_STAR="${RELION_DIR}/run_it000_optimiser.star"' in text
    assert '--relion_half_sets "${RELION_HALF_SET_STAR}"' in text
    assert '--relion_optimiser "${RELION_OPTIMISER_STAR}"' in text
    assert '--relion_init_dir "${RELION_DIR}"' in text
    assert '--perturb_replay_relion_dir' not in text
    assert "try_relion_replay_pair" not in text
    assert "LATEST_RELION_DATA_ITER" not in text
    assert "LATEST_RELION_SAMPLING_ITER" not in text
    assert "Strict RELION replay: capping RECOVAR max_iter" not in text
    assert "perturb-replay-restart-state-iterations" not in text
    assert "EM_K1_MATRIX_TRAJECTORY_MODE=autonomous" in text
    assert "EM_K1_MATRIX_TRAJECTORY_MODE=autonomous" in submission
    assert "RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0" in submission
    config_text = next((scratch / "jobs").glob("em_k1_matrix_15_*.sh")).read_text()
    assert '"trajectory_mode": "autonomous"' in config_text


def test_noctf_simulator_cases_use_sanitized_relion_ctf_by_default(tmp_path):
    proc, scratch = _dry_run_launcher(tmp_path, case="14")

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_14_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    assert 'if [[ "noctf" == "noctf" ]]; then' in text
    assert 'if [[ "1" == "1" ]]; then' in text
    assert 'RELION_CTF_ARGS=(--ctf)' in text
    assert 'RELION_INPUT_STAR="particles_relion_identity_ctf.star"' in text
    assert '"${PIXI_PY}" -m scripts.make_relion_identity_ctf_star \\' in text
    assert f'cd "{REPO_ROOT}"' in text
    assert '--input-star "${DATA_DIR}/particles.star" \\' in text
    assert 'particles_relion_identity_ctf.json' in text
    assert '--phase-shift-deg 180.0' in text
    assert 'RELION_INPUT_STAR=${RELION_INPUT_STAR}' in text
    assert '--i "${RELION_INPUT_STAR}" \\' in text
    assert '"${RELION_CTF_ARGS[@]}" \\' in text


def test_noctf_simulator_relion_noctf_diagnostic_can_disable_ctf(tmp_path):
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="14",
        extra_env={"EM_K1_NOCTF_RELION_USE_CTF": "0"},
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_14_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    assert 'if [[ "noctf" == "noctf" ]]; then' in text
    assert 'if [[ "0" == "1" ]]; then' in text
    assert 'RELION_CTF_ARGS=(--ctf)' in text
    assert 'RELION_INPUT_STAR="particles.star"' in text
    assert 'RELION_CTF_ARGS=()' in text
    assert '--i "${RELION_INPUT_STAR}" \\' in text


def test_identity_ctf_star_helper_rewrites_relion_metadata_without_touching_raw_star(tmp_path):
    raw_star = tmp_path / "particles.star"
    out_star = tmp_path / "particles_relion_identity_ctf.star"
    manifest = tmp_path / "particles_relion_identity_ctf.json"
    optics = pd.DataFrame(
        {
            "rlnOpticsGroup": [1],
            "rlnOpticsGroupName": ["opticsGroup1"],
            "rlnAmplitudeContrast": [-1.0],
            "rlnSphericalAberration": [2.7],
            "rlnVoltage": [300.0],
            "rlnImagePixelSize": [4.25],
            "rlnImageSize": [128],
            "rlnImageDimensionality": [2],
        }
    )
    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.128.mrcs", "2@particles.128.mrcs"],
            "rlnDefocusU": [12345.0, 23456.0],
            "rlnDefocusV": [12300.0, 23500.0],
            "rlnDefocusAngle": [7.0, 8.0],
            "rlnPhaseShift": [0.0, 0.0],
            "rlnOpticsGroup": [1, 1],
        }
    )
    starfile.write({"optics": optics, "particles": particles}, raw_star)

    proc = subprocess.run(
        [
            sys.executable,
            str(IDENTITY_CTF_STAR),
            "--input-star",
            str(raw_star),
            "--output-star",
            str(out_star),
            "--manifest",
            str(manifest),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    raw = starfile.read(raw_star)
    rewritten = starfile.read(out_star)
    assert float(raw["optics"]["rlnAmplitudeContrast"].iloc[0]) == -1.0
    assert float(rewritten["optics"]["rlnAmplitudeContrast"].iloc[0]) == 1.0
    assert float(rewritten["optics"]["rlnSphericalAberration"].iloc[0]) == 0.0
    assert set(rewritten["particles"]["rlnDefocusU"]) == {0.0}
    assert set(rewritten["particles"]["rlnDefocusV"]) == {0.0}
    assert set(rewritten["particles"]["rlnDefocusAngle"]) == {0.0}
    assert set(rewritten["particles"]["rlnPhaseShift"]) == {180.0}
    data = json.loads(manifest.read_text())
    assert data["raw_star_left_unchanged"] is True
    assert data["relion_identity_ctf"]["rlnAmplitudeContrast"] == 1.0
    assert data["relion_identity_ctf"]["rlnPhaseShift"] == 180.0


def test_identity_ctf_phase_shift_matches_recovar_noctf_sign():
    import jax.numpy as jnp
    import numpy as np

    from recovar.core.ctf import evaluate_ctf

    freqs = jnp.array([[0.0, 0.0], [0.01, 0.02], [0.1, 0.0]], dtype=jnp.float32)
    recovar_noctf = jnp.array([[0.0, 0.0, 0.0, 300.0, 0.0, -1.0, 0.0, 0.0, 1.0]], dtype=jnp.float32)
    relion_identity = jnp.array([[0.0, 0.0, 0.0, 300.0, 0.0, 1.0, 180.0, 0.0, 1.0]], dtype=jnp.float32)

    assert np.allclose(np.asarray(evaluate_ctf(freqs, recovar_noctf)), 1.0)
    assert np.allclose(np.asarray(evaluate_ctf(freqs, relion_identity)), 1.0)


def test_case_jobs_reuse_setup_relion_binding_build_dir(tmp_path):
    proc, scratch = _dry_run_launcher(tmp_path, case="14")

    assert proc.returncode == 0, proc.stdout
    setup_script = scratch / "jobs" / "em_k1_matrix_setup.sh"
    summary_script = scratch / "jobs" / "em_k1_matrix_summary.sh"
    case_scripts = list((scratch / "jobs").glob("em_k1_matrix_14_*.sh"))
    assert setup_script.exists()
    assert summary_script.exists()
    assert len(case_scripts) == 1

    setup_text = setup_script.read_text()
    case_text = case_scripts[0].read_text()
    summary_text = summary_script.read_text()
    shared_export = f'export RECOVAR_RELION_BIND_BUILD_DIR="{scratch}/relion_bind_build/shared"'
    jax_cache_export = f'export RECOVAR_JAX_CACHE_DIR="{scratch}/jax_cache"'
    assert shared_export in setup_text
    assert shared_export in case_text
    assert jax_cache_export in setup_text
    assert jax_cache_export in case_text
    assert jax_cache_export in summary_text
    assert 'export JAX_COMPILATION_CACHE_DIR="${RECOVAR_JAX_CACHE_DIR}"' in setup_text
    assert 'export JAX_COMPILATION_CACHE_DIR="${RECOVAR_JAX_CACHE_DIR}"' in case_text
    assert 'export JAX_COMPILATION_CACHE_DIR="${RECOVAR_JAX_CACHE_DIR}"' in summary_text
    assert "unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA" in setup_text
    assert "unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA" in case_text
    assert 'export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"' in case_text
    assert 'export RECOVAR_CACHE_DIR="${RECOVAR_CACHE_DIR-}"' in case_text
    assert 'RECOVAR_CACHE_DIR=${RECOVAR_CACHE_DIR:-<disabled>}' in case_text
    matrix_python = scratch / "venv" / "bin" / "python"
    python_export = f'export PIXI_PY="{matrix_python}"'
    for text in (setup_text, case_text, summary_text):
        assert python_export in text
        assert "pixi run" not in text
        assert "git symbolic-ref --short HEAD ||" not in text
    assert "export PIP_NO_INDEX=1" in setup_text
    base_pixi_python = Path(
        os.environ.get(
            "EM_K1_MATRIX_PIXI_PY",
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
    for text in (setup_text, case_text):
        assert "if command -v nvidia-smi >/dev/null 2>&1; then" in text
    assert '-m venv --system-site-packages "${EM_K1_MATRIX_VENV}"' in setup_text
    assert (
        '"${PIXI_PY}" -m pip install -e . --no-deps --no-build-isolation '
        "--ignore-installed"
    ) in setup_text
    assert '"${PIXI_PY}" recovar/relion_bind/build.py' in setup_text
    setup_gate = setup_text.split("# The default setup partition is CPU-only.", maxsplit=1)[1]
    assert "export JAX_PLATFORMS=cpu" in setup_gate
    assert "export JAX_PLATFORM_NAME=cpu" in setup_gate
    assert "export RECOVAR_DISABLE_CUDA=1" in setup_gate
    assert 'export CUDA_VISIBLE_DEVICES=""' in setup_gate
    assert 'rm -rf "${RECOVAR_RELION_BIND_BUILD_DIR:?}"' in setup_text
    assert 'rm -rf "${RECOVAR_RELION_BIND_BUILD_DIR:?}"' not in case_text
    assert "recovar/relion_bind/build.py" not in case_text
    submission_env = (scratch / "submission.env").read_text()
    assert f"EM_K1_MATRIX_VENV={scratch / 'venv'}" in submission_env
    assert f"PIXI_PY={matrix_python}" in submission_env


def test_queued_jobs_unset_inherited_non_matrix_refinement_overrides(tmp_path):
    inherited_overrides = {
        "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER": "1",
        "RECOVAR_DISABLE_RELION_EXACT_FINE_GAUSSIAN": "1",
        "RECOVAR_USE_FLOAT64_SCORING": "1",
        "RECOVAR_USE_FLOAT64_PROJECTIONS": "1",
    }
    proc, scratch = _dry_run_launcher(tmp_path, case="14", extra_env=inherited_overrides)

    assert proc.returncode == 0, proc.stdout
    job_texts = [path.read_text() for path in sorted((scratch / "jobs").glob("*.sh"))]
    assert len(job_texts) == 3
    for text in job_texts:
        assert (
            "unset RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER "
            "RECOVAR_DISABLE_RELION_EXACT_FINE_GAUSSIAN"
        ) in text
        assert "unset RECOVAR_USE_FLOAT64_SCORING RECOVAR_USE_FLOAT64_PROJECTIONS" in text
        for name, value in inherited_overrides.items():
            assert f"export {name}={value}" not in text


def test_case_jobs_build_cuda_lib_atomically_under_lock(tmp_path):
    proc, scratch = _dry_run_launcher(tmp_path, case="14")

    assert proc.returncode == 0, proc.stdout
    case_scripts = list((scratch / "jobs").glob("em_k1_matrix_14_*.sh"))
    assert len(case_scripts) == 1
    text = case_scripts[0].read_text()
    assert 'CUDA_LIB_TMP="${RECOVAR_CUDA_LIB}.${SLURM_JOB_ID:-$$}.tmp"' in text
    assert "export CUDA_LIB_TMP PIXI_PY" in text
    assert f'flock "{scratch}/cuda/build.lock" bash -lc' in text
    assert 'rm -f "${CUDA_LIB_TMP}"' in text
    assert 'make -C recovar/cuda LIB="${CUDA_LIB_TMP}" all' in text
    assert 'mv -f "${CUDA_LIB_TMP}" "${RECOVAR_CUDA_LIB}"' in text


def test_final_bpref_accumulator_dump_dir_is_case_scoped(tmp_path):
    dump_root = tmp_path / "bpref_dumps"
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="28",
        extra_env={"RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR": str(dump_root)},
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_28_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    assert f"export RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR={dump_root}" in text
    assert (
        'export RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR="${RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR%/}/'
        '28_small_kent_extra_offset_3k_g128_noise3_bf80"'
    ) in text
    assert 'mkdir -p "${RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR}"' in text
    submission = (scratch / "submission.env").read_text()
    assert f"RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR={dump_root}" in submission


def test_bpref_and_relion_mstep_diagnostic_dump_dirs_are_case_scoped(tmp_path):
    bpref_root = tmp_path / "bpref_iter"
    pass2_root = tmp_path / "pass2"
    relion_root = tmp_path / "relion_mstep"
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="11",
        extra_env={
            "RECOVAR_BPREF_ACCUM_DUMP_DIR": str(bpref_root),
            "RECOVAR_PASS2_DUMP_DIR": str(pass2_root),
            "RECOVAR_PASS2_DUMP_ORIGINAL_INDICES": "7,19",
            "RECOVAR_PASS2_DUMP_CURRENT_SIZE": "56",
            "RECOVAR_MSTEP_DUMP_DIR": str(relion_root),
            "RECOVAR_MSTEP_DUMP_MAX_CALLS": "6",
            "RECOVAR_MSTEP_DUMP_RAW": "1",
        },
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_11_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    case_name = "11_small_baseline_3k_g128_white_noise1_bf80"
    assert f"export RECOVAR_BPREF_ACCUM_DUMP_DIR={bpref_root}" in text
    assert f"export RECOVAR_PASS2_DUMP_DIR={pass2_root}" in text
    assert "export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=7\\,19" in text
    assert "export RECOVAR_PASS2_DUMP_CURRENT_SIZE=56" in text
    assert f"export RECOVAR_MSTEP_DUMP_DIR={relion_root}" in text
    assert "export RECOVAR_MSTEP_DUMP_MAX_CALLS=6" in text
    assert "export RECOVAR_MSTEP_DUMP_RAW=1" in text
    assert f'export RECOVAR_BPREF_ACCUM_DUMP_DIR="${{RECOVAR_BPREF_ACCUM_DUMP_DIR%/}}/{case_name}"' in text
    assert f'export RECOVAR_PASS2_DUMP_DIR="${{RECOVAR_PASS2_DUMP_DIR%/}}/{case_name}"' in text
    assert f'export RECOVAR_MSTEP_DUMP_DIR="${{RECOVAR_MSTEP_DUMP_DIR%/}}/{case_name}"' in text
    assert 'mkdir -p "${RECOVAR_BPREF_ACCUM_DUMP_DIR}"' in text
    assert 'mkdir -p "${RECOVAR_PASS2_DUMP_DIR}"' in text
    assert 'mkdir -p "${RECOVAR_MSTEP_DUMP_DIR}"' in text
    submission = (scratch / "submission.env").read_text()
    assert f"RECOVAR_BPREF_ACCUM_DUMP_DIR={bpref_root}" in submission
    assert f"RECOVAR_PASS2_DUMP_DIR={pass2_root}" in submission
    assert "RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=7,19" in submission
    assert "RECOVAR_PASS2_DUMP_CURRENT_SIZE=56" in submission
    assert f"RECOVAR_MSTEP_DUMP_DIR={relion_root}" in submission
    assert "RECOVAR_MSTEP_DUMP_MAX_CALLS=6" in submission
    assert "RECOVAR_MSTEP_DUMP_RAW=1" in submission


def test_local_fused_posterior_diagnostic_dump_dir_is_forwarded_and_case_scoped(tmp_path):
    dump_root = tmp_path / "local_fused"
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="11",
        extra_env={
            "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR": str(dump_root),
            "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES": "428,2814",
            "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_CURRENT_SIZE": "128",
            "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION": "11",
            "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL": "final_probe",
        },
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_11_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    case_name = "11_small_baseline_3k_g128_white_noise1_bf80"
    assert f"export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR={dump_root}" in text
    assert "export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES=428\\,2814" in text
    assert "export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_CURRENT_SIZE=128" in text
    assert "export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION=11" in text
    assert "export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL=final_probe" in text
    assert (
        'export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR="${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR%/}/'
        f'{case_name}"'
    ) in text
    assert 'mkdir -p "${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR}"' in text
    submission = (scratch / "submission.env").read_text()
    assert f"RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR={dump_root}" in submission
    assert "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES=428,2814" in submission
    assert "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_CURRENT_SIZE=128" in submission
    assert "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION=11" in submission
    assert "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL=final_probe" in submission


def test_relion_projector_diagnostic_dump_dir_is_forwarded_and_case_scoped(tmp_path):
    dump_root = tmp_path / "projector"
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="22",
        extra_env={"RECOVAR_RELION_PROJECTOR_DUMP_DIR": str(dump_root)},
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_22_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    case_name = "22_small_severe_outliers_3k_g128_radial_noise5_bf80"
    assert f"export RECOVAR_RELION_PROJECTOR_DUMP_DIR={dump_root}" in text
    assert (
        'export RECOVAR_RELION_PROJECTOR_DUMP_DIR="${RECOVAR_RELION_PROJECTOR_DUMP_DIR%/}/'
        f'{case_name}"'
    ) in text
    assert 'mkdir -p "${RECOVAR_RELION_PROJECTOR_DUMP_DIR}"' in text
    submission = (scratch / "submission.env").read_text()
    assert f"RECOVAR_RELION_PROJECTOR_DUMP_DIR={dump_root}" in submission


def test_final_all_data_grid_correct_defaults_to_quality_mode(tmp_path):
    proc, scratch = _dry_run_launcher(tmp_path, case="28")

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_28_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    assert "export RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=" not in text
    assert "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=\n" in (scratch / "submission.env").read_text()


def test_launcher_records_submission_and_runtime_git_fingerprints(tmp_path):
    proc, scratch = _dry_run_launcher(tmp_path, case="28")

    assert proc.returncode == 0, proc.stdout
    provenance_dir = scratch / "provenance" / "submission"
    assert (provenance_dir / "git_status_porcelain.txt").exists()
    assert (provenance_dir / "git_diff.patch").exists()
    assert (provenance_dir / "git_diff.sha256").exists()
    fingerprint = (provenance_dir / "git_worktree_fingerprint.sha256").read_text().strip()
    assert len(fingerprint) == 64
    components = (provenance_dir / "git_component_sha256.txt").read_text().splitlines()
    assert len(components) == 3
    assert all(len(component) == 64 and component.isalnum() for component in components)
    submission = (scratch / "submission.env").read_text()
    assert f"SUBMISSION_GIT_PROVENANCE_DIR={provenance_dir}" in submission
    assert f"SUBMISSION_GIT_WORKTREE_FINGERPRINT_SHA256={fingerprint}" in submission

    scripts = list((scratch / "jobs").glob("em_k1_matrix_28_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    assert "JOB_GIT_PROVENANCE_DIR=" in text
    assert "git_diff.patch" in text
    assert "git_untracked_file_hashes.tsv" in text
    assert "Git worktree fingerprint SHA256:" in text
    assert "EXPECTED_GIT_HEAD=" in text
    assert "EXPECTED_GIT_WORKTREE_FINGERPRINT_SHA256=" in text
    assert "ERROR: queued-job Git HEAD drift" in text
    assert "ERROR: queued-job worktree fingerprint drift" in text
    assert "Queued-job Git provenance gate ok" in text
    assert "git_status_porcelain.txt\" 2>/dev/null | awk '{print $1}'" in text


def test_runtime_caches_use_separate_em_runtime_root(tmp_path):
    runtime = tmp_path / "runtime"
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="32",
        extra_env={"EM_K1_MATRIX_RUNTIME_ROOT": str(runtime)},
    )

    assert proc.returncode == 0, proc.stdout
    case_script = next((scratch / "jobs").glob("em_k1_matrix_32_*.sh")).read_text()
    summary_script = (scratch / "jobs" / "em_k1_matrix_summary.sh").read_text()
    assert f'export TMPDIR="{runtime}/em_k1_matrix_32_' in case_script
    assert f'export PIXI_HOME="{runtime}/em_k1_matrix_32_' in case_script
    assert f'export RATTLER_CACHE_DIR="{runtime}/em_k1_matrix_32_' in case_script
    assert f'export TMPDIR="{runtime}/em_k1_matrix_summary_' in summary_script
    assert (runtime / "SAFE_TO_DELETE").exists()
    assert f"RUNTIME_ROOT={runtime}" in (scratch / "submission.env").read_text()


def test_case_and_matrix_summary_failures_propagate_to_slurm_status(tmp_path):
    proc, scratch = _dry_run_launcher(tmp_path, case="32")

    assert proc.returncode == 0, proc.stdout
    case_script = next((scratch / "jobs").glob("em_k1_matrix_32_*.sh")).read_text()
    summary_script = (scratch / "jobs" / "em_k1_matrix_summary.sh").read_text()
    assert 'STATUS="${SUMMARY_STATUS}"' in case_script
    assert 'MATRIX_SUMMARY_STATUS="${summary_status}"' in summary_script
    assert 'exit "${MATRIX_SUMMARY_STATUS}"' in summary_script
    assert "ERROR: queued-summary Git HEAD drift" in summary_script
    assert "ERROR: queued-summary worktree is dirty" in summary_script


def test_relion_binary_identity_is_recorded_in_case_job(tmp_path):
    proc, scratch = _dry_run_launcher(tmp_path, case="32")

    assert proc.returncode == 0, proc.stdout
    case_script = next((scratch / "jobs").glob("em_k1_matrix_32_*.sh")).read_text()
    assert "RELION_REFINE_MPI_RESOLVED=" in case_script
    assert "RELION_REFINE_MPI_SHA256=" in case_script
    assert 'CASE_GPU_UUID="$(capture_physical_gpu_uuid)"' in case_script
    assert 'RELION_GPU_UUID="$(capture_physical_gpu_uuid)"' in case_script
    assert 'RECOVAR_GPU_UUID="$(capture_physical_gpu_uuid)"' in case_script
    assert "paired_gpu_uuid.json" in case_script
    assert "physical_gpu_inventory.csv" in case_script
    assert 'mapfile -t visible_uuids < <(nvidia-smi --query-gpu=uuid' in case_script
    assert 'nvidia-smi --id="${gpu_token}"' not in case_script
    assert 'nvidia-smi --id="${slurm_gpu_token}"' not in case_script
    assert 'if [[ "${slurm_gpu_token}" == GPU-* && "${slurm_gpu_token}" != "${gpu_uuid}" ]]' in case_script


def test_k1_dense_pass2_diagnostic_env_is_forwarded(tmp_path):
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="32",
        extra_env={"RECOVAR_K1_DENSE_PASS2": "1"},
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_32_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    assert "export RECOVAR_K1_DENSE_PASS2=1" in text
    assert "RECOVAR_K1_DENSE_PASS2=1" in (scratch / "submission.env").read_text()


def test_k1_skip_significance_pruning_diagnostic_env_is_forwarded(tmp_path):
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="32",
        extra_env={"RECOVAR_K1_SKIP_SIGNIFICANCE_PRUNING": "1"},
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_32_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    assert "export RECOVAR_K1_SKIP_SIGNIFICANCE_PRUNING=1" in text
    assert "RECOVAR_K1_SKIP_SIGNIFICANCE_PRUNING=1" in (scratch / "submission.env").read_text()


def test_k1_relion_x_half_mstep_diagnostic_env_is_forwarded(tmp_path):
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="32",
        extra_env={"RECOVAR_K1_RELION_X_HALF_MSTEP": "0"},
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_32_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    assert "export RECOVAR_K1_RELION_X_HALF_MSTEP=0" in text
    assert "RECOVAR_K1_RELION_X_HALF_MSTEP=0" in (scratch / "submission.env").read_text()


def test_exact_local_packed_noise_chunk_target_env_is_forwarded(tmp_path):
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="26",
        extra_env={
            "RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS": "16000000",
            "RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS": "250",
            "RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS": "90",
        },
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_26_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    assert "export RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS=16000000" in text
    assert "RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS=16000000" in (
        scratch / "submission.env"
    ).read_text()
    assert "export RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS=250" in text
    assert "export RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS=90" in text
    submission = (scratch / "submission.env").read_text()
    assert "RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS=250" in submission
    assert "RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS=90" in submission


def test_sparse_pass2_memory_cap_envs_are_forwarded(tmp_path):
    extra_env = {
        "RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES": "123456",
        "RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES": "234567",
        "RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES": "345678",
        "RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES": "456789",
        "RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES": "567890",
        "RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS": "64",
    }
    proc, scratch = _dry_run_launcher(tmp_path, case="11", extra_env=extra_env)

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_11_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    submission = (scratch / "submission.env").read_text()
    for name, value in extra_env.items():
        assert f"export {name}={value}" in text
        assert f"{name}={value}" in submission


def test_local_adaptive_pass2_defaults_to_relion_pruned_parent_in_jobs(tmp_path):
    proc, scratch = _dry_run_launcher(tmp_path, case="32")

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_32_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    submission = (scratch / "submission.env").read_text()
    assert "export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0" in text
    assert "RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0" in submission


def test_k1_batch_defaults_are_forwarded_to_jobs_and_provenance(tmp_path):
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="32",
        extra_env={"RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET": "805306368"},
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_32_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    submission = (scratch / "submission.env").read_text()
    assert '--image_batch_size "187"' in text
    assert '--rotation_block_size "8192"' in text
    assert "K1_IMAGE_BATCH_SIZE=187" in submission
    assert "K1_ROTATION_BLOCK_SIZE=8192" in submission
    assert "export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40" in text
    assert "RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40" in submission
    assert "export RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET=805306368" in text
    assert "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET=805306368" in submission


def test_exact_local_auto_microbatch_boost_is_forwarded(tmp_path):
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="32",
        extra_env={"RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST": "3"},
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_32_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    submission = (scratch / "submission.env").read_text()
    assert "export RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST=3" in text
    assert "RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST=3" in submission


def test_save_intermediates_skip_unregularized_is_forwarded(tmp_path):
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="32",
        extra_env={
            "RECOVAR_SAVE_INTERMEDIATES_DIR": "auto",
        },
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_32_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    submission = (scratch / "submission.env").read_text()
    assert 'export RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED="1"' in text
    assert "RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED=1" in submission
    assert '--save_intermediates_dir "${RECOVAR_INTERMEDIATES_DIR}"' in text
    assert "--save_intermediates_skip_unregularized" in text


def test_save_intermediates_unregularized_maps_can_be_reenabled(tmp_path):
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="32",
        extra_env={
            "RECOVAR_SAVE_INTERMEDIATES_DIR": "auto",
            "RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED": "0",
        },
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_32_*.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    submission = (scratch / "submission.env").read_text()
    assert 'export RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED="0"' in text
    assert "RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED=0" in submission
    assert '--save_intermediates_dir "${RECOVAR_INTERMEDIATES_DIR}"' in text
    assert "--save_intermediates_skip_unregularized" in text
    assert 'if [[ "${RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]' in text


def test_setup_and_summary_default_to_cpu_without_gpu_constraint(tmp_path):
    scratch = tmp_path / "scratch"
    relion_src = _relion_src_fixture(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "EM_K1_MATRIX_SCRATCH_DIR": str(scratch),
            "EM_K1_MATRIX_CASES": "32",
            "EM_K1_MATRIX_RUN_RELION": "1",
                "SBATCH_ACCOUNT": "gilles",
                "SBATCH_PARTITION": "cryoem",
                "SBATCH_CONSTRAINT": "h100",
                "RELION_SRC_DIR": str(relion_src),
            }
    )
    for name in (
        "EM_K1_MATRIX_SETUP_PARTITION",
        "EM_K1_MATRIX_SETUP_CONSTRAINT",
        "EM_K1_MATRIX_SUMMARY_PARTITION",
        "EM_K1_MATRIX_SUMMARY_CONSTRAINT",
    ):
        env.pop(name, None)

    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    setup_text = (scratch / "jobs" / "em_k1_matrix_setup.sh").read_text()
    summary_text = (scratch / "jobs" / "em_k1_matrix_summary.sh").read_text()
    submission = (scratch / "submission.env").read_text()
    assert "#SBATCH --partition=cpu" in setup_text
    assert "#SBATCH --partition=cpu" in summary_text
    assert "#SBATCH --constraint=h100" not in setup_text
    assert "#SBATCH --constraint=h100" not in summary_text
    assert "EM_K1_MATRIX_SETUP_PARTITION=cpu" in submission
    assert "EM_K1_MATRIX_SUMMARY_PARTITION=cpu" in submission
    assert "EM_K1_MATRIX_SETUP_CONSTRAINT=" in submission
    assert "EM_K1_MATRIX_SUMMARY_CONSTRAINT=" in submission


def test_case_time_limit_override_updates_script_and_case_table(tmp_path):
    proc, scratch = _dry_run_launcher(
        tmp_path,
        case="2",
        extra_env={"EM_K1_MATRIX_TIME_LIMIT": "36:00:00"},
    )

    assert proc.returncode == 0, proc.stdout
    scripts = list((scratch / "jobs").glob("em_k1_matrix_2_*.sh"))
    assert len(scripts) == 1
    assert "#SBATCH --time=36:00:00" in scripts[0].read_text()

    selected = pd.read_csv(scratch / "selected_cases.tsv", sep="|")
    assert selected["time_limit"].tolist() == ["36:00:00"]
    submission = (scratch / "submission.env").read_text()
    assert "EM_K1_MATRIX_TIME_LIMIT=36:00:00" in submission
