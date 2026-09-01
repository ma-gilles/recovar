import sys

import pytest

from scripts import run_em_kclass_robustness_matrix_slurm as launcher


def _set_relion_src(tmp_path, monkeypatch):
    relion_src = tmp_path / "relion_src"
    relion_src.mkdir()
    (relion_src / "projector.h").write_text("// fixture\n")
    monkeypatch.setenv("RELION_SRC_DIR", str(relion_src))
    return relion_src


def _set_dispatch_capture_executable(tmp_path, monkeypatch):
    executable = tmp_path / "relion_refine_mpi_dispatch_v2"
    executable.write_text("#!/bin/sh\n# RELION_DISPATCH_LOG_SCHEMA_V2\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setenv("EM_KCLASS_MATRIX_RELION_REFINE_MPI", str(executable))
    return executable


def test_default_cases_cover_all_available_cryobench_pdb_families():
    pdb_dirs = {case.pdb_dir for case in launcher.DEFAULT_CASES}

    assert len(launcher.DEFAULT_CASES) == 34
    assert len({case.index for case in launcher.DEFAULT_CASES}) == 34
    assert len({case.name for case in launcher.DEFAULT_CASES}) == 34
    assert launcher.DEFAULT_RIBO_PDB_DIR in pdb_dirs
    assert launcher.DEFAULT_IGG_PDB_DIR in pdb_dirs
    assert launcher.DEFAULT_TOMOTWIN_PDB_DIR in pdb_dirs
    assert launcher.DEFAULT_IGG_RL_PDB_DIR in pdb_dirs


def test_default_cases_cover_k1_stress_axes_and_k4_invariance_controls():
    cases = {case.index: case for case in launcher.DEFAULT_CASES}

    assert cases[16].noise_level == 10.0
    assert cases[17].dataset_params_option == "noctf"
    assert (cases[18].noise_scale_std, cases[18].contrast_std) == (0.5, 0.5)
    assert cases[19].image_offset_n_std == 1.0
    assert cases[20].percent_outliers == 0.5
    assert cases[21].noise_level == 0.2
    assert cases[22].dataset_params_option == "kent"
    assert cases[23].class_distribution == "custom:0.80,0.10,0.07,0.03"
    assert (cases[24].grid_size, cases[24].pdb_bfactor) == (256, 0.0)
    assert {cases[28].seed, cases[29].seed} == {3802, 4802}

    invariance_cases = [cases[index] for index in (25, 26, 27)]
    scientific_fields = (
        "pdb_dir",
        "n_classes",
        "n_images",
        "grid_size",
        "noise_level",
        "noise_model",
        "dataset_params_option",
        "class_distribution",
        "seed",
        "pdb_bfactor",
        "init_radius",
        "noise_scale_std",
        "contrast_std",
        "volume_radius",
        "image_offset_n_std",
        "percent_outliers",
        "max_iter",
    )
    assert all(
        tuple(getattr(case, field) for field in scientific_fields)
        == tuple(getattr(invariance_cases[0], field) for field in scientific_fields)
        for case in invariance_cases[1:]
    )
    assert [(case.image_batch_size, case.rotation_block_size) for case in invariance_cases] == [
        (50, 8192),
        (17, 8192),
        (50, 257),
    ]
    assert len({case.shared_input_group for case in invariance_cases}) == 1
    assert invariance_cases[0].shared_input_producer is True
    assert [case.shared_input_producer for case in invariance_cases[1:]] == [False, False]


def test_default_cases_add_positive_noctf_and_k4_symmetry_trajectories():
    cases = {case.index: case for case in launcher.DEFAULT_CASES}

    assert cases[17].dataset_params_option == "noctf"
    assert cases[17].noise_level == 3.0
    assert cases[30].dataset_params_option == "noctf"
    assert (cases[30].noise_model, cases[30].noise_level) == ("white", 1.0)
    assert cases[30].n_classes == 4

    symmetry_cases = [cases[index] for index in (31, 32, 33, 34)]
    assert [case.symmetry for case in symmetry_cases] == ["C4", "D4", "O", "I1"]
    assert all(
        (case.n_classes, case.n_images, case.grid_size, case.max_iter) == (4, 5_000, 128, 5) for case in symmetry_cases
    )
    assert {case.seed for case in symmetry_cases} == {41001}


def test_extra_pdb_family_case_can_be_selected_by_name(monkeypatch):
    monkeypatch.setenv("EM_KCLASS_MATRIX_CASES", "tomotwin_k8_10k_g128_radial_noise3_kent_headheavy")
    args = type(
        "Args",
        (),
        {
            "case": [],
            "max_iter_override": None,
            "time_limit_override": None,
            "seed_override": None,
            "seed_offset": None,
        },
    )()

    case = launcher.selected_cases(args)[0]

    assert case.index == 13
    assert case.pdb_dir == launcher.DEFAULT_TOMOTWIN_PDB_DIR
    assert case.n_classes == 8
    assert case.dataset_params_option == "kent"
    assert case.class_distribution == "head-heavy"


def _write_numbered_class_maps(root, *, iterations, n_classes=4):
    root.mkdir(parents=True, exist_ok=True)
    for iteration in iterations:
        for half in (1, 2):
            for class_number in range(1, n_classes + 1):
                (root / f"it{iteration:03d}_half{half}_class{class_number}_reg.mrc").write_bytes(
                    f"{iteration}:{half}:{class_number}".encode()
                )


def test_numbered_class_map_audit_accepts_early_convergence(tmp_path):
    relion_dir = tmp_path / "relion"
    intermediates = tmp_path / "recovar" / "intermediates"
    relion_dir.mkdir()
    for iteration in (0, 1, 2):
        (relion_dir / f"run_it{iteration:03d}_model.star").write_text("model\n")
    _write_numbered_class_maps(intermediates, iterations=(0, 1))

    report = launcher.audit_numbered_class_maps(
        recovar_intermediates_dir=intermediates,
        relion_dir=relion_dir,
        n_classes=4,
    )

    assert report["relion_numbered_iterations"] == [1, 2]
    assert report["recovar_numbered_iterations"] == [0, 1]
    assert report["maps_per_iteration"] == 8
    assert report["map_count"] == 16
    assert (intermediates / "numbered_class_map_audit.json").is_file()
    assert len((intermediates / "numbered_class_maps.sha256").read_text().splitlines()) == 16


def test_numbered_class_map_audit_rejects_iteration_gaps_and_map_mismatches(tmp_path):
    relion_dir = tmp_path / "relion"
    intermediates = tmp_path / "recovar" / "intermediates"
    relion_dir.mkdir()
    for iteration in (0, 1, 3):
        (relion_dir / f"run_it{iteration:03d}_model.star").write_text("model\n")
    _write_numbered_class_maps(intermediates, iterations=(0, 1, 2))

    with pytest.raises(ValueError, match="RELION numbered iterations are not contiguous"):
        launcher.audit_numbered_class_maps(
            recovar_intermediates_dir=intermediates,
            relion_dir=relion_dir,
            n_classes=4,
        )

    (relion_dir / "run_it003_model.star").unlink()
    (intermediates / "it001_half2_class4_reg.mrc").unlink()
    with pytest.raises(ValueError, match="do not match the actual RELION trajectory"):
        launcher.audit_numbered_class_maps(
            recovar_intermediates_dir=intermediates,
            relion_dir=relion_dir,
            n_classes=4,
        )


def _write_relion_population_star(path, distributions, orientation_masses):
    import pandas as pd
    import starfile

    payload = {
        "model_classes": pd.DataFrame({"rlnClassDistribution": distributions}),
    }
    for class_number, mass in enumerate(orientation_masses, start=1):
        payload[f"model_pdf_orient_class_{class_number}"] = pd.DataFrame({"rlnOrientationDistribution": [mass]})
    starfile.write(payload, path, overwrite=True)


def test_relion_class_population_audit_passes_only_positive_classes(tmp_path):
    relion = tmp_path / "relion"
    relion.mkdir()
    _write_relion_population_star(
        relion / "run_it001_model.star",
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    )
    _write_relion_population_star(
        relion / "run_it002_model.star",
        [0.2, 0.3, 0.1, 0.4],
        [0.2, 0.3, 0.1, 0.4],
    )

    report = launcher.audit_relion_class_populations(relion_dir=relion, n_classes=4)

    assert report["passed"] is True
    assert report["collapsed"] == []
    assert len(report["rows"]) == 8
    assert (relion / "class_population_audit.json").is_file()


def test_relion_class_population_audit_records_then_fails_on_collapse(tmp_path):
    relion = tmp_path / "relion"
    relion.mkdir()
    _write_relion_population_star(
        relion / "run_it001_model.star",
        [0.25, 0.75, 0.0, 0.0],
        [0.25, 0.75, 0.0, 0.0],
    )

    with pytest.raises(ValueError, match="class-collapse gate failed"):
        launcher.audit_relion_class_populations(relion_dir=relion, n_classes=4)

    payload = __import__("json").loads((relion / "class_population_audit.json").read_text())
    assert payload["passed"] is False
    assert [(row["iteration"], row["class"]) for row in payload["collapsed"]] == [
        (1, 3),
        (1, 4),
    ]


def test_noise_rng_batch_size_generates_clean_prepare_command(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setenv("RECOVAR_FINAL_ALL_DATA_GRID_CORRECT", "1")
    monkeypatch.setenv("RECOVAR_K_CLASS_DENSE_PASS2", "1")
    monkeypatch.setenv("RECOVAR_K_CLASS_DENSE_PASS2_MEAN_SUPPORT_FRACTION", "0")
    monkeypatch.setenv("RECOVAR_K_CLASS_RELION_X_HALF_MSTEP", "1")
    monkeypatch.setenv("RECOVAR_K_CLASS_FULL_VOLUME_MSTEP", "1")
    monkeypatch.setenv("RECOVAR_K_CLASS_HALF_VOLUME_MSTEP", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "0")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "8192")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO", "0.5")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES", "2147483648")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES", "1073741824")
    monkeypatch.setenv("RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET", "805306368")
    monkeypatch.setenv("RECOVAR_KCLASS_DUMP_DIR", str(tmp_path / "kclass_dumps"))
    script = launcher.write_case_script(
        case=launcher.DEFAULT_CASES[0],
        scratch_dir=tmp_path,
        jobs_dir=jobs_dir,
        cuda_lib=tmp_path / "librecovar_cuda.so",
        account="gilles",
        partition="cryoem",
        constraint="a100",
        exclusive=False,
        cuda_module="cudatoolkit/12.8",
        relion_src_dir=tmp_path / "relion_src",
        relion_module="relion/5.0.1/gcc-11.5.0-gpu",
        relion_refine_mpi="/instrumented/relion_refine_mpi",
        relion_mpi_ranks=3,
        relion_pool=3,
        particle_diameter=380.0,
        image_batch_size=50,
        rotation_block_size=2000,
        gt_align_refine_orders="",
        noise_rng_batch_size="256",
    )

    text = script.read_text()
    assert "\n+" not in text
    assert "  --noise-rng-batch-size 256 \\\n  --relion-normalize \\" in text
    assert f"export RECOVAR_JAX_CACHE_DIR={tmp_path}/jax_cache" in text
    assert 'export JAX_COMPILATION_CACHE_DIR="${RECOVAR_JAX_CACHE_DIR}"' in text
    assert 'RECOVAR_*|RELION_*|JAX_*|XLA_*) unset "${ENV_NAME}"' in text
    assert "unset TF_GPU_ALLOCATOR" in text
    assert "export XLA_PYTHON_CLIENT_PREALLOCATE=false" in text
    assert "export RECOVAR_FINAL_ALL_DATA_GRID_CORRECT" not in text
    assert "export RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER" not in text
    assert "export RECOVAR_K_CLASS_DENSE_PASS2" not in text
    assert "export RECOVAR_K_CLASS_RELION_X_HALF_MSTEP" not in text
    assert "export RECOVAR_KCLASS_DUMP_DIR" not in text
    assert 'external_bind_dir = os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR")' in text
    assert 'str(relion_bind_file).startswith(str(external_bind_root) + "/")' in text
    assert "      --firstiter_cc \\\n" in text
    assert "  --firstiter_cc \\\n" in text
    assert f'"initial_resolution_ang": {launcher.KCLASS_INITIAL_RESOLUTION_ANG}' in text
    assert f"      --ini_high {launcher.KCLASS_INITIAL_RESOLUTION_ANG:g} \\\n" in text
    assert f"  --init_resolution {launcher.KCLASS_INITIAL_RESOLUTION_ANG:g} \\\n" in text
    assert "  --apply-initial-lowpass \\\n" in text
    assert "  --symmetry C1 \\\n" in text
    assert "      --sym C1 \\\n" in text
    assert "  --sym C1 \\\n" in text
    assert "  --init_resolution 30" not in text
    assert "  --image-fourier-backend relion_cuda \\\n" in text
    assert 'RECOVAR_INTERMEDIATES_DIR="${RECOVAR_DIR}/intermediates"' in text
    assert '--save_intermediates_dir "${RECOVAR_INTERMEDIATES_DIR}"' in text
    assert "from scripts.run_em_kclass_robustness_matrix_slurm import audit_numbered_class_maps" in text
    assert '"${RECOVAR_INTERMEDIATES_DIR}" "${RELION_DIR}" 2' in text
    assert "for iteration in $(seq 0" not in text
    assert "Numbered class-map audit ok" in text
    assert 'RELION_GPU_UUID="$(capture_physical_gpu_uuid)"' in text
    assert 'RECOVAR_GPU_UUID="$(capture_physical_gpu_uuid)"' in text
    assert "paired_gpu_uuid.json" in text
    assert 'start_engine_gpu_monitor "${CASE_ROOT}/relion_gpu_monitor.csv"' in text
    assert 'start_engine_gpu_monitor "${CASE_ROOT}/recovar_gpu_monitor.csv"' in text
    assert 'nvidia-smi --query-gpu="${GPU_MONITOR_QUERY}" --format=csv -l 5' in text
    assert '"gpu_monitor_interval_s": 5' in text
    assert "COMBINED_MONITOR_PID" in text
    assert "trap cleanup_gpu_monitors EXIT" in text
    assert "mapfile -t visible_uuids < <(nvidia-smi --query-gpu=uuid" in text
    assert 'nvidia-smi --id="${gpu_token}"' not in text
    assert 'nvidia-smi --id="${slurm_gpu_token}"' not in text
    assert 'if [[ "${slurm_gpu_token}" == GPU-* && "${slurm_gpu_token}" != "${gpu_uuid}" ]]' in text
    assert "Queued-job Git provenance gate ok" in text
    assert f"RUNTIME_ROOT={launcher.DEFAULT_RUNTIME_ROOT}/em_kclass_matrix_1_" in text
    assert "export RELION_DISPATCH_LOG" in text
    assert "RELION_DISPATCH_LOG_SCHEMA_V2" in text
    assert 'grep -aFq -- "${RELION_DISPATCH_SCHEMA_MARKER}"' in text
    assert "legacy four-column range capture is rejected" in text
    assert "-m scripts.build_relion_dispatch_schedule" in text
    assert '--relion-dispatch-schedule "${RELION_DISPATCH_SCHEDULE}"' in text
    assert "from scripts.run_em_kclass_robustness_matrix_slurm import audit_relion_class_populations" in text
    assert '"${RELION_DIR}" 2' in text
    assert "RELION class-population audit ok" in text


def test_case_jobs_only_verify_setup_sealed_cuda_lib(tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    script = launcher.write_case_script(
        case=launcher.DEFAULT_CASES[0],
        scratch_dir=tmp_path,
        jobs_dir=jobs_dir,
        cuda_lib=tmp_path / "librecovar_cuda.so",
        account="gilles",
        partition="cryoem",
        constraint="a100",
        exclusive=False,
        cuda_module="cudatoolkit/12.8",
        relion_src_dir=tmp_path / "relion_src",
        relion_module="relion/5.0.1/gcc-11.5.0-gpu",
        relion_refine_mpi="/instrumented/relion_refine_mpi",
        relion_mpi_ranks=3,
        relion_pool=3,
        particle_diameter=380.0,
        image_batch_size=50,
        rotation_block_size=2000,
        gt_align_refine_orders="",
        noise_rng_batch_size="",
    )

    text = script.read_text()
    assert f"export RECOVAR_CUDA_LIB={tmp_path}/librecovar_cuda.so" in text
    assert 'setup did not seal the shared CUDA library' in text
    assert 'sha256sum --check "${RECOVAR_CUDA_LIB}.sha256"' in text
    assert 'make -C recovar/cuda' not in text
    assert 'flock "$(dirname "${RECOVAR_CUDA_LIB}")/build.lock"' not in text


def test_case_job_uses_case_specific_batch_invariance_overrides(tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    case = launcher.DEFAULT_CASES[25]

    script = launcher.write_case_script(
        case=case,
        scratch_dir=tmp_path,
        jobs_dir=jobs_dir,
        cuda_lib=tmp_path / "librecovar_cuda.so",
        account="gilles",
        partition="cryoem",
        constraint="h100",
        exclusive=False,
        cuda_module="cudatoolkit/12.8",
        relion_src_dir=tmp_path / "relion_src",
        relion_module="relion/5.0.1/gcc-11.5.0-gpu",
        relion_refine_mpi="/instrumented/relion_refine_mpi",
        relion_mpi_ranks=3,
        relion_pool=3,
        particle_diameter=380.0,
        image_batch_size=50,
        rotation_block_size=2000,
        gt_align_refine_orders="",
        noise_rng_batch_size="",
    )

    text = script.read_text()
    assert '"image_batch_size": 17' in text
    assert '"rotation_block_size": 8192' in text
    assert "--image_batch_size 17" in text
    assert "--rotation_block_size 8192" in text
    assert f"DATA_DIR={tmp_path}/shared_inputs/" in text
    assert "GENERATE_INPUT=0" in text
    assert "RUN_RELION=0" in text
    assert 'sha256sum --check "${SHARED_INPUT_MANIFEST}"' in text
    assert 'sha256sum --check "${SHARED_RELION_MANIFEST}"' in text


def test_shared_input_group_rejects_consumer_without_selected_producer():
    cases = {case.index: case for case in launcher.DEFAULT_CASES}

    with pytest.raises(SystemExit, match="requires exactly one selected producer"):
        launcher.validate_shared_input_groups([cases[26], cases[27]])


def test_shared_input_group_rejects_generator_axis_drift():
    cases = {case.index: case for case in launcher.DEFAULT_CASES}
    changed_consumer = launcher.replace(cases[26], noise_level=2.0)

    with pytest.raises(SystemExit, match="changes generator fields.*noise_level"):
        launcher.validate_shared_input_groups([cases[25], changed_consumer])


def test_shared_input_group_dependencies_follow_one_sealed_producer(tmp_path, monkeypatch):
    _set_relion_src(tmp_path, monkeypatch)
    _set_dispatch_capture_executable(tmp_path, monkeypatch)
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    cases_by_index = {case.index: case for case in launcher.DEFAULT_CASES}
    selected = tuple(launcher.replace(cases_by_index[index], pdb_dir=pdb_dir) for index in (25, 26, 27))
    monkeypatch.setattr(launcher, "DEFAULT_CASES", selected)
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_em_kclass_robustness_matrix_slurm.py",
            "--scratch-dir",
            str(scratch),
        ],
    )
    submissions = []

    def fake_submit(script, *, dry_run, extra_args=None):
        job_id = str(80000 + len(submissions))
        submissions.append((script.name, list(extra_args or []), job_id))
        return job_id

    monkeypatch.setattr(launcher, "submit", fake_submit)

    launcher.main()

    assert submissions[1][0].startswith("em_kclass_matrix_25_")
    assert submissions[1][1] == ["--dependency=afterok:80000"]
    assert submissions[2][0].startswith("em_kclass_matrix_26_")
    assert submissions[2][1] == ["--dependency=afterok:80000:80001"]
    assert submissions[3][0].startswith("em_kclass_matrix_27_")
    assert submissions[3][1] == ["--dependency=afterok:80000:80001"]
    scripts = [
        (scratch / "jobs" / f"em_kclass_matrix_{index}_{selected[offset].name}.sh").read_text()
        for offset, index in enumerate((25, 26, 27))
    ]
    data_lines = [next(line for line in text.splitlines() if line.startswith("DATA_DIR=")) for text in scripts]
    relion_lines = [next(line for line in text.splitlines() if line.startswith("RELION_DIR=")) for text in scripts]
    relion_manifest_lines = [
        next(line for line in text.splitlines() if line.startswith("SHARED_RELION_MANIFEST=")) for text in scripts
    ]
    assert len(set(data_lines)) == 1
    assert len(set(relion_lines)) == 1
    assert len(set(relion_manifest_lines)) == 1
    assert "GENERATE_INPUT=1" in scripts[0]
    assert all("GENERATE_INPUT=0" in text for text in scripts[1:])
    assert "RUN_RELION=1" in scripts[0]
    assert all("RUN_RELION=0" in text for text in scripts[1:])
    assert 'xargs -0 sha256sum > "${MANIFEST_TMP}"' in scripts[0]
    assert 'xargs -0 sha256sum > "${RELION_MANIFEST_TMP}"' in scripts[0]
    assert all('sha256sum --check "${SHARED_INPUT_MANIFEST}"' in text for text in scripts)
    assert all('sha256sum --check "${SHARED_RELION_MANIFEST}"' in text for text in scripts)
    assert all('ln -sfn "${RELION_DIR}" "${CASE_ROOT}/relion_ref"' in text for text in scripts)
    assert all("shared_relion_ref" not in text for text in scripts)
    assert "refusing to reuse an existing case root" in scripts[0]
    assert "refusing to regenerate or reseal an existing dataset" in scripts[0]
    assert "refusing to regenerate or reseal an existing shared RELION oracle" in scripts[0]


@pytest.mark.parametrize("case_index,symmetry", [(31, "C4"), (32, "D4"), (33, "O"), (34, "I1")])
def test_symmetry_case_wires_identical_label_to_generator_relion_and_recovar(
    tmp_path,
    case_index,
    symmetry,
):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    case = {case.index: case for case in launcher.DEFAULT_CASES}[case_index]

    script = launcher.write_case_script(
        case=case,
        scratch_dir=tmp_path,
        jobs_dir=jobs_dir,
        cuda_lib=tmp_path / "librecovar_cuda.so",
        account="gilles",
        partition="cryoem",
        constraint="h100",
        exclusive=False,
        cuda_module="cudatoolkit/12.8",
        relion_src_dir=tmp_path / "relion_src",
        relion_module="relion/5.0.1/gcc-11.5.0-gpu",
        relion_refine_mpi="/instrumented/relion_refine_mpi",
        relion_mpi_ranks=3,
        relion_pool=3,
        particle_diameter=380.0,
        image_batch_size=50,
        rotation_block_size=2000,
        gt_align_refine_orders="",
        noise_rng_batch_size="",
    )

    text = script.read_text()
    assert f'"symmetry": "{symmetry}"' in text
    assert f"  --symmetry {symmetry} \\\n" in text
    assert f"      --sym {symmetry} \\\n" in text
    assert f"  --sym {symmetry} \\\n" in text


def test_setup_script_allows_external_relion_bind_build_dir(tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    script = launcher.write_setup_script(
        scratch_dir=tmp_path,
        jobs_dir=jobs_dir,
        cuda_lib=tmp_path / "librecovar_cuda.so",
        account="gilles",
        partition="cpu",
        constraint="",
        setup_gres="",
        cuda_module="cudatoolkit/12.8",
        relion_src_dir=tmp_path / "relion_src",
    )

    text = script.read_text()
    matrix_python = tmp_path / "venv" / "bin" / "python"
    shared_bind = tmp_path / "relion_bind_build" / "shared"
    assert f"export PIXI_PY={matrix_python}" in text
    assert f"export RECOVAR_RELION_BIND_BUILD_DIR={shared_bind}" in text
    assert '-m venv --system-site-packages "${EM_KCLASS_MATRIX_VENV}"' in text
    assert '"${PIXI_PY}" -m pip install -e . --no-deps --no-build-isolation --ignore-installed' in text
    assert '"${PIXI_PY}" recovar/relion_bind/build.py' in text
    assert 'CUDA_LIB_TMP="${RECOVAR_CUDA_LIB}.${SLURM_JOB_ID:-$$}.tmp"' in text
    assert 'flock "$(dirname "${RECOVAR_CUDA_LIB}")/build.lock"' in text
    assert 'make -C recovar/cuda LIB="${CUDA_LIB_TMP}" all' in text
    assert 'sha256sum "${RECOVAR_CUDA_LIB}" > "${RECOVAR_CUDA_LIB}.sha256"' in text
    assert "pixi run" not in text
    assert f"sha256sum --check {tmp_path}/relion_bind_build/shared.sha256" in text
    assert 'external_bind_dir = os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR")' in text
    assert 'str(relion_bind_file).startswith(str(external_bind_root) + "/")' in text
    assert 'assert str(pathlib.Path(relion_bind.__file__).resolve()).startswith(str(repo) + "/")' not in text


def test_main_rejects_exclusive_kclass_jobs(tmp_path, monkeypatch):
    _set_relion_src(tmp_path, monkeypatch)
    monkeypatch.setenv("EM_KCLASS_MATRIX_EXCLUSIVE", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_em_kclass_robustness_matrix_slurm.py",
            "--dry-run",
            "--scratch-dir",
            str(tmp_path / "scratch"),
            "--case",
            "1",
        ],
    )
    _set_dispatch_capture_executable(tmp_path, monkeypatch)

    with pytest.raises(SystemExit, match="must be non-exclusive"):
        launcher.main()


def test_main_rejects_nonempty_reused_scratch_root(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "stale_evidence.json").write_text("{}\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_em_kclass_robustness_matrix_slurm.py",
            "--dry-run",
            "--scratch-dir",
            str(scratch),
            "--case",
            "1",
        ],
    )

    with pytest.raises(SystemExit, match="must be new or empty"):
        launcher.main()


def test_setup_and_summary_default_to_cpu_without_gpu_constraint(tmp_path, monkeypatch):
    relion_src = _set_relion_src(tmp_path, monkeypatch)
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    case = launcher.replace(launcher.DEFAULT_CASES[0], pdb_dir=pdb_dir)
    second_case = launcher.replace(case, index=2, name="second_case")
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(launcher, "DEFAULT_CASES", (case, second_case))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_em_kclass_robustness_matrix_slurm.py",
            "--dry-run",
            "--scratch-dir",
            str(scratch),
            "--case",
            "1",
            "--case",
            "2",
        ],
    )
    monkeypatch.setenv("SBATCH_PARTITION", "cryoem")
    monkeypatch.setenv("SBATCH_ACCOUNT", "gilles")
    monkeypatch.setenv("SBATCH_CONSTRAINT", "h100")
    _set_dispatch_capture_executable(tmp_path, monkeypatch)
    for name in (
        "EM_KCLASS_MATRIX_SETUP_PARTITION",
        "EM_KCLASS_MATRIX_SETUP_CONSTRAINT",
        "EM_KCLASS_MATRIX_SUMMARY_PARTITION",
        "EM_KCLASS_MATRIX_SUMMARY_CONSTRAINT",
    ):
        monkeypatch.delenv(name, raising=False)

    launcher.main()

    setup_text = (scratch / "jobs" / "em_kclass_matrix_setup.sh").read_text()
    summary_text = (scratch / "jobs" / "em_kclass_matrix_summary.sh").read_text()
    submission = (scratch / "submission.env").read_text()
    assert "#SBATCH --partition=cpu" in setup_text
    assert "#SBATCH --partition=cpu" in summary_text
    assert "#SBATCH --constraint=h100" not in setup_text
    assert "#SBATCH --constraint=h100" not in summary_text
    assert "EM_KCLASS_MATRIX_SETUP_PARTITION=cpu" in submission
    assert "EM_KCLASS_MATRIX_SUMMARY_PARTITION=cpu" in submission
    assert "EM_KCLASS_MATRIX_SETUP_CONSTRAINT=" in submission
    assert "EM_KCLASS_MATRIX_SUMMARY_CONSTRAINT=" in submission
    expected_head = launcher.git_text("rev-parse", "HEAD")
    assert f"EXPECTED_GIT_HEAD={expected_head}" in setup_text
    assert f"EXPECTED_GIT_HEAD={expected_head}" in summary_text
    assert "git status --short --untracked-files=no" in setup_text
    assert "git status --short --untracked-files=no" in summary_text
    assert f"RUNTIME_ROOT={launcher.DEFAULT_RUNTIME_ROOT}/em_kclass_matrix_setup_" in setup_text
    assert "export BASE_PIXI_PY" in setup_text
    assert f"export RELION_SRC_DIR={relion_src}" in setup_text
    assert f"RUNTIME_ROOT={launcher.DEFAULT_RUNTIME_ROOT}/em_kclass_matrix_summary_" in summary_text
    assert f"MATRIX_PY={scratch / 'venv' / 'bin' / 'python'}" in summary_text
    assert f"BASE_PIXI_PY={launcher.base_pixi_python()}" in summary_text
    assert 'touch "${RUNTIME_ROOT}/SAFE_TO_DELETE"' in setup_text
    assert 'touch "${RUNTIME_ROOT}/SAFE_TO_DELETE"' in summary_text
    assert "pixi run" not in summary_text
    assert f"--slurm-accounting-json-out {scratch / 'slurm_case_accounting.json'}" in summary_text
    assert f"EXPECTED_GIT_HEAD={expected_head}" in submission
    assert "CASE_JOB_IDS='DRYRUN DRYRUN'" in submission
    assert f"RUNTIME_ROOT={launcher.DEFAULT_RUNTIME_ROOT}" in submission
    assert f"export RELION_SRC_DIR={relion_src}" in setup_text
    assert f"RELION_SRC_DIR={relion_src}" in submission


def test_main_fails_closed_without_dispatch_capture_relion(tmp_path, monkeypatch):
    _set_relion_src(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_em_kclass_robustness_matrix_slurm.py",
            "--dry-run",
            "--scratch-dir",
            str(tmp_path / "scratch"),
            "--case",
            "1",
        ],
    )
    monkeypatch.delenv("EM_KCLASS_MATRIX_RELION_REFINE_MPI", raising=False)

    with pytest.raises(SystemExit, match="must name an absolute, executable RELION build"):
        launcher.main()


def test_main_fails_closed_for_legacy_dispatch_capture_schema(tmp_path, monkeypatch):
    _set_relion_src(tmp_path, monkeypatch)
    executable = tmp_path / "legacy_relion_refine_mpi"
    executable.write_text("#!/bin/sh\n# legacy four-column dispatch ranges\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setenv("EM_KCLASS_MATRIX_RELION_REFINE_MPI", str(executable))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_em_kclass_robustness_matrix_slurm.py",
            "--dry-run",
            "--scratch-dir",
            str(tmp_path / "scratch"),
            "--case",
            "1",
        ],
    )

    with pytest.raises(SystemExit, match="RELION_DISPATCH_LOG_SCHEMA_V2") as error:
        launcher.main()

    assert "five-column identity records" in str(error.value)
    assert "legacy four-column range capture is rejected" in str(error.value)


def test_main_fails_closed_without_relion_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_em_kclass_robustness_matrix_slurm.py",
            "--dry-run",
            "--scratch-dir",
            str(tmp_path / "scratch"),
            "--case",
            "1",
        ],
    )
    monkeypatch.delenv("RELION_SRC_DIR", raising=False)

    with pytest.raises(SystemExit, match="RELION_SRC_DIR must name"):
        launcher.main()


def test_selected_cases_support_iteration_and_time_limit_overrides(monkeypatch):
    monkeypatch.setenv("EM_KCLASS_MATRIX_CASES", "1")
    monkeypatch.setenv("EM_KCLASS_MATRIX_MAX_ITER", "8")
    monkeypatch.setenv("EM_KCLASS_MATRIX_TIME_LIMIT", "10:00:00")

    args = type("Args", (), {"case": [], "max_iter_override": None, "time_limit_override": None})()
    cases = launcher.selected_cases(args)

    assert len(cases) == 1
    assert cases[0].max_iter == 8
    assert cases[0].time_limit == "10:00:00"
    assert launcher.DEFAULT_CASES[0].max_iter == 5


def test_max_iter_override_is_recorded_in_case_rows(monkeypatch):
    monkeypatch.setenv("EM_KCLASS_MATRIX_MAX_ITER", "9")
    args = type("Args", (), {"case": ["1"], "max_iter_override": None, "time_limit_override": None})()

    case = launcher.selected_cases(args)[0]

    assert case.row_fields[15] == "0"
    assert case.row_fields[16] == "9"


def test_seed_offset_renames_case_and_updates_generated_commands(tmp_path, monkeypatch):
    _set_relion_src(tmp_path, monkeypatch)
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    case = launcher.replace(launcher.DEFAULT_CASES[1], pdb_dir=pdb_dir)
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(launcher, "DEFAULT_CASES", (case,))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_em_kclass_robustness_matrix_slurm.py",
            "--dry-run",
            "--scratch-dir",
            str(scratch),
            "--case",
            "2",
            "--seed-offset",
            "100",
        ],
    )
    monkeypatch.setenv("SBATCH_PARTITION", "cryoem")
    monkeypatch.setenv("SBATCH_ACCOUNT", "gilles")
    _set_dispatch_capture_executable(tmp_path, monkeypatch)

    launcher.main()

    selected = (scratch / "case_table.tsv").read_text()
    script = next((scratch / "jobs").glob("em_kclass_matrix_2_*_seed2902.sh")).read_text()
    submission = (scratch / "submission.env").read_text()
    assert "ribo_k4_10k_g128_white_noise1_uniform_seed2902" in selected
    assert "|2902|" in selected
    assert '"seed": 2902' in script
    assert "  --seed 2902 \\" in script
    assert "      --random_seed 2902 \\" in script
    assert "EM_KCLASS_MATRIX_SEED_OFFSET=100" in submission
    assert "EM_KCLASS_MATRIX_SEED=" in submission


def test_seed_offset_applies_when_all_default_cases_are_selected(monkeypatch):
    monkeypatch.delenv("EM_KCLASS_MATRIX_CASES", raising=False)
    args = type(
        "Args",
        (),
        {
            "case": [],
            "max_iter_override": None,
            "time_limit_override": None,
            "seed_override": None,
            "seed_offset": 10_000,
        },
    )()

    cases = launcher.selected_cases(args)

    assert len(cases) == len(launcher.DEFAULT_CASES)
    assert all(
        updated.seed == original.seed + 10_000 for updated, original in zip(cases, launcher.DEFAULT_CASES, strict=True)
    )
    assert all(updated.name.endswith(f"_seed{updated.seed}") for updated in cases)


def test_three_seed_suite_expands_each_case_over_frozen_seed_set(monkeypatch):
    monkeypatch.delenv("EM_KCLASS_MATRIX_CASES", raising=False)
    args = type(
        "Args",
        (),
        {
            "case": ["31"],
            "max_iter_override": None,
            "time_limit_override": None,
            "seed_override": None,
            "seed_offset": None,
            "three_seed_suite": True,
        },
    )()

    cases = launcher.selected_cases(args)

    assert [case.seed for case in cases] == list(launcher.THREE_SEED_VALUES)
    assert [case.seed_replicate for case in cases] == [1, 2, 3]
    assert {case.base_name for case in cases} == {"ribo_k4_5k_g128_white_noise1_c4_uniform"}
    assert {case.base_seed for case in cases} == {41001}
    assert all(case.symmetry == "C4" for case in cases)


def test_three_seed_suite_dry_run_writes_one_shared_setup_and_aggregate(tmp_path, monkeypatch):
    _set_relion_src(tmp_path, monkeypatch)
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    case = launcher.replace(launcher.DEFAULT_CASES[30], pdb_dir=pdb_dir)
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(launcher, "DEFAULT_CASES", (case,))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_em_kclass_robustness_matrix_slurm.py",
            "--dry-run",
            "--scratch-dir",
            str(scratch),
            "--three-seed-suite",
        ],
    )
    _set_dispatch_capture_executable(tmp_path, monkeypatch)

    launcher.main()

    rows = (scratch / "case_table.tsv").read_text().splitlines()
    assert len(rows) == 4
    assert all(f"|{seed}|" in rows[index] for index, seed in enumerate(launcher.THREE_SEED_VALUES, 1))
    assert len(list((scratch / "jobs").glob("em_kclass_matrix_31_*_seed*.sh"))) == 3
    summary = (scratch / "jobs" / "em_kclass_matrix_summary.sh").read_text()
    assert "-m scripts.aggregate_em_kclass_multiseed" in summary
    assert "--expected-seeds 41001,41002,41003" in summary
    assert f"--case-table {scratch / 'case_table.tsv'}" in summary
    submission = (scratch / "submission.env").read_text()
    assert "EM_KCLASS_MATRIX_THREE_SEED_SUITE=1" in submission
    assert "EM_KCLASS_MATRIX_THREE_SEED_VALUES=41001,41002,41003" in submission
    assert "CASE_JOB_IDS='DRYRUN DRYRUN DRYRUN'" in submission


def test_seed_override_and_seed_offset_are_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("EM_KCLASS_MATRIX_CASES", "1")
    monkeypatch.setenv("EM_KCLASS_MATRIX_SEED", "9")
    monkeypatch.setenv("EM_KCLASS_MATRIX_SEED_OFFSET", "1")
    args = type(
        "Args",
        (),
        {
            "case": [],
            "max_iter_override": None,
            "time_limit_override": None,
            "seed_override": None,
            "seed_offset": None,
        },
    )()

    try:
        launcher.selected_cases(args)
    except SystemExit as exc:
        assert "Use either EM_KCLASS_MATRIX_SEED" in str(exc)
    else:
        raise AssertionError("selected_cases should reject simultaneous seed override and offset")


def test_three_seed_suite_rejects_single_seed_override(monkeypatch):
    monkeypatch.setenv("EM_KCLASS_MATRIX_CASES", "1")
    monkeypatch.setenv("EM_KCLASS_MATRIX_SEED", "9")
    args = type(
        "Args",
        (),
        {
            "case": [],
            "max_iter_override": None,
            "time_limit_override": None,
            "seed_override": None,
            "seed_offset": None,
            "three_seed_suite": True,
        },
    )()

    with pytest.raises(SystemExit, match="cannot be combined"):
        launcher.selected_cases(args)


def test_outlier_kclass_case_uses_holdout_pdb_and_disables_streaming_mmap(tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    script = launcher.write_case_script(
        case=launcher.DEFAULT_CASES[9],
        scratch_dir=tmp_path,
        jobs_dir=jobs_dir,
        cuda_lib=tmp_path / "librecovar_cuda.so",
        account="gilles",
        partition="cryoem",
        constraint="",
        exclusive=False,
        cuda_module="cudatoolkit/12.8",
        relion_src_dir=tmp_path / "relion_src",
        relion_module="relion/5.0.1/gcc-11.5.0-gpu",
        relion_refine_mpi="/instrumented/relion_refine_mpi",
        relion_mpi_ranks=3,
        relion_pool=3,
        particle_diameter=380.0,
        image_batch_size=50,
        rotation_block_size=2000,
        gt_align_refine_orders="3",
        noise_rng_batch_size="",
    )

    text = script.read_text()
    assert "Need 5 PDB files under" in text
    assert 'OUTLIER_PDB="${PDBS[4]}"' in text
    assert "Using holdout outlier PDB" in text
    assert '"percent_outliers": 0.2' in text
    assert "  --percent-outliers 0.2 \\" in text
    assert '  --outlier-pdb-path "${OUTLIER_PDB}" \\' in text
    assert "  --no-streaming-mmap \\" in text
