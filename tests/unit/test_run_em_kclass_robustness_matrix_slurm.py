import sys

import pytest

from scripts import run_em_kclass_robustness_matrix_slurm as launcher


def test_default_cases_cover_all_available_cryobench_pdb_families():
    pdb_dirs = {case.pdb_dir for case in launcher.DEFAULT_CASES}

    assert launcher.DEFAULT_RIBO_PDB_DIR in pdb_dirs
    assert launcher.DEFAULT_IGG_PDB_DIR in pdb_dirs
    assert launcher.DEFAULT_TOMOTWIN_PDB_DIR in pdb_dirs
    assert launcher.DEFAULT_IGG_RL_PDB_DIR in pdb_dirs


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
    assert "Queued-job Git provenance gate ok" in text
    assert f"RUNTIME_ROOT={launcher.DEFAULT_RUNTIME_ROOT}/em_kclass_matrix_1_" in text
    assert "export RELION_DISPATCH_LOG" in text
    assert "-m scripts.build_relion_dispatch_schedule" in text
    assert '--relion-dispatch-schedule "${RELION_DISPATCH_SCHEDULE}"' in text


def test_case_jobs_build_cuda_lib_atomically_under_lock(tmp_path):
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
    assert 'CUDA_LIB_TMP="${RECOVAR_CUDA_LIB}.${SLURM_JOB_ID:-$$}.tmp"' in text
    assert "export CUDA_LIB_TMP PIXI_PY" in text
    assert 'flock "$(dirname "${RECOVAR_CUDA_LIB}")/build.lock"' in text
    assert f"export RECOVAR_CUDA_LIB={tmp_path}/${{SLURM_JOB_ID}}/librecovar_cuda.so" in text
    assert 'rm -f "${CUDA_LIB_TMP}"' in text
    assert 'make -C recovar/cuda LIB="${CUDA_LIB_TMP}" all' in text
    assert 'mv -f "${CUDA_LIB_TMP}" "${RECOVAR_CUDA_LIB}"' in text


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
    )

    text = script.read_text()
    assert 'external_bind_dir = os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR")' in text
    assert 'str(relion_bind_file).startswith(str(external_bind_root) + "/")' in text
    assert 'assert str(pathlib.Path(relion_bind.__file__).resolve()).startswith(str(repo) + "/")' not in text


def test_setup_and_summary_default_to_cpu_without_gpu_constraint(tmp_path, monkeypatch):
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
    monkeypatch.setenv("EM_KCLASS_MATRIX_RELION_REFINE_MPI", "/bin/true")
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
    assert f"RUNTIME_ROOT={launcher.DEFAULT_RUNTIME_ROOT}/em_kclass_matrix_summary_" in summary_text
    assert f"EXPECTED_GIT_HEAD={expected_head}" in submission
    assert "CASE_JOB_IDS='DRYRUN DRYRUN'" in submission
    assert f"RUNTIME_ROOT={launcher.DEFAULT_RUNTIME_ROOT}" in submission


def test_main_fails_closed_without_dispatch_capture_relion(tmp_path, monkeypatch):
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
    monkeypatch.setenv("EM_KCLASS_MATRIX_RELION_REFINE_MPI", "/bin/true")

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
