from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from recovar.em.initial_model.gt_metrics import (
    DEFAULT_GT_ALIGN_HEALPIX_ORDER,
    DEFAULT_GT_ALIGN_MAX_SHELL,
    relion_alignment_rotations,
)
from scripts.run_vdam_abinitio_merge_guard import _default_output_root, build_guard_commands, run_guard

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_long_guard_module():
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    module_path = REPO_ROOT / "tests/long_test/test_em_parity_long.py"
    spec = importlib.util.spec_from_file_location("em_parity_long_guard_for_unit_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vdam_alignment_defaults_are_merge_guarded():
    assert DEFAULT_GT_ALIGN_HEALPIX_ORDER == 2
    assert DEFAULT_GT_ALIGN_MAX_SHELL == 8
    assert relion_alignment_rotations(DEFAULT_GT_ALIGN_HEALPIX_ORDER).shape == (4608, 3, 3)


def test_gt_alignment_cli_wiring_survives_merges():
    expected_tokens = [
        "--gt_align",
        "--gt_align_healpix_order",
        "DEFAULT_GT_ALIGN_HEALPIX_ORDER",
        "--gt_align_max_shell",
        "DEFAULT_GT_ALIGN_MAX_SHELL",
        "aligned_fsc_vs_gt",
        "gt_align_rotation_index",
    ]
    for relative_path in [
        "scripts/run_multi_iter_parity.py",
        "scripts/postprocess_multi_iter_gt.py",
    ]:
        text = (REPO_ROOT / relative_path).read_text()
        missing = [token for token in expected_tokens if token not in text]
        assert not missing, f"{relative_path} lost GT-alignment wiring: {missing}"


def test_standalone_abinitio_evaluator_cli_wiring_survives_merges():
    text = (REPO_ROOT / "scripts/evaluate_ab_initio_gt.py").read_text()
    expected_tokens = [
        "--volume_frame",
        "--gt_frame",
        "choices=(\"relion\", \"recovar\")",
        "--gt_align",
        "aligned_prefix = f\"{label}_aligned\"",
        "_fsc_vs_gt",
        "Native InitialModel writes RELION-frame",
        "load_relion_volume",
        "output_json",
        "output_npz",
    ]
    missing = [token for token in expected_tokens if token not in text]
    assert not missing, f"standalone evaluator lost merge-guarded contract: {missing}"


def test_merge_guard_plan_contains_cpu_and_gpu_gates():
    cpu_names = [command.name for command in build_guard_commands("cpu")]
    assert cpu_names == [
        "py_compile",
        "vdam_abinitio_contracts",
        "vdam_frozen_scorecard",
        "initial_model_vdam_unit_slice",
        "initial_model_unit_suite",
        "em_fast_guard",
    ]

    quick_names = [command.name for command in build_guard_commands("cpu", quick=True)]
    assert "initial_model_unit_suite" not in quick_names
    assert "initial_model_vdam_unit_slice" in quick_names
    assert "em_fast_guard" in quick_names

    all_names = [command.name for command in build_guard_commands("all", quick=True)]
    assert "native_initialmodel_k2_smoke_gpu" in all_names
    assert "em_parity_fast_gpu" in all_names
    assert "extract_em_parity_fast_tables" in all_names

    gpu_names = [command.name for command in build_guard_commands("gpu")]
    assert gpu_names[0] == "native_initialmodel_k2_smoke_gpu"


def test_merge_guard_output_root_can_use_overflow_scratch(monkeypatch, tmp_path):
    overflow = tmp_path / "overflow"
    monkeypatch.setenv("VDAM_ABINITIO_GUARD_OUTPUT_ROOT", str(overflow))
    monkeypatch.delenv("RECOVAR_AGENT_SCRATCH_ROOT", raising=False)
    assert _default_output_root() == overflow

    agent_root = tmp_path / "agent-root"
    monkeypatch.delenv("VDAM_ABINITIO_GUARD_OUTPUT_ROOT", raising=False)
    monkeypatch.setenv("RECOVAR_AGENT_SCRATCH_ROOT", str(agent_root))
    assert _default_output_root() == agent_root


def test_merge_guard_redirects_runtime_caches_to_output_dir():
    guard = (REPO_ROOT / "scripts/run_vdam_abinitio_merge_guard.py").read_text()
    expected_tokens = [
        'pytest_cache_dir = output_dir / "pytest_cache"',
        '"-o cache_dir={pytest_cache_dir}',
        '"PYTHONPYCACHEPREFIX"',
        'pycache_dir = output_dir / "pycache"',
    ]
    missing = [token for token in expected_tokens if token not in guard]
    assert not missing, f"VDAM merge guard lost output-dir cache redirection: {missing}"


def test_long_native_quality_guard_uses_relion_initialmodel_reference():
    long_test = (REPO_ROOT / "tests/long_test/test_em_parity_long.py").read_text()
    slurm_launcher = (REPO_ROOT / "scripts/run_em_parity_long_slurm.sh").read_text()

    expected_long_tokens = [
        "K1_NATIVE_RELION_DIR",
        "relion_initialmodel_k1_it008",
        "_assert_relion_initialmodel_reference",
        "--grad",
        "--denovo_3dref",
        "_rlnDoGradientRefine",
        "_rlnDoAutoRefine",
        "run_it008_class001.mrc",
        "relion_initialmodel_it008",
        "_relion_frame_map_similarity",
        "k1_native_initialmodel_vdam_vs_relion_it008_corr",
        ">= 0.999",
        "direct VDAM vs RELION it008",
    ]
    missing = [token for token in expected_long_tokens if token not in long_test]
    assert not missing, f"native InitialModel long guard lost RELION --grad reference checks: {missing}"

    expected_launcher_tokens = [
        "em_parity_long_k1_native_ref",
        "relion_initialmodel_k1_it008",
        "--grad",
        "--denovo_3dref",
        "--dependency=afterok:${K1_NATIVE_REF_JOB}",
        "_rlnDoGradientRefine[[:space:]]*1",
        "_rlnDoAutoRefine[[:space:]]*0",
    ]
    missing = [token for token in expected_launcher_tokens if token not in slurm_launcher]
    assert not missing, f"EM-long launcher no longer prepares the RELION InitialModel reference: {missing}"


def test_native_vdam_subset_order_uses_relion_sorted_idx_base_order():
    driver = (REPO_ROOT / "recovar/em/initial_model/driver.py").read_text()
    iteration_loop = (REPO_ROOT / "recovar/em/initial_model/iteration_loop.py").read_text()

    expected_driver_tokens = [
        "_micrograph_sort_order(main_star)",
        "_experiment_read_order(main_star)",
        "particle_order=particle_order",
    ]
    missing = [token for token in expected_driver_tokens if token not in driver]
    assert not missing, f"native InitialModel driver lost RELION sorted_idx base-order wiring: {missing}"

    expected_loop_tokens = [
        "particle_order: Sequence[int] | None = None",
        "base_order = np.asarray(particle_order",
        "shuffled = base_order.copy()",
        "shuffled = base_order[permutation]",
    ]
    missing = [token for token in expected_loop_tokens if token not in iteration_loop]
    assert not missing, f"VDAM iteration loop lost RELION sorted_idx base-order handling: {missing}"


def test_native_vdam_solvent_flattening_is_separate_from_zero_mask():
    driver = (REPO_ROOT / "recovar/em/initial_model/driver.py").read_text()
    iteration_loop = (REPO_ROOT / "recovar/em/initial_model/iteration_loop.py").read_text()
    run_ab_initio = (REPO_ROOT / "scripts/run_ab_initio.py").read_text()

    expected_tokens = [
        "do_solvent: bool = True",
        "if opts.do_solvent",
        "relion_solvent_mask",
        "relion_solvent_flatten_state",
        "do_solvent=opts.do_solvent",
    ]
    haystack = "\n".join([driver, iteration_loop, run_ab_initio])
    missing = [token for token in expected_tokens if token not in haystack]
    assert not missing, f"native InitialModel lost RELION --flatten_solvent post-M-step wiring: {missing}"


def test_native_vdam_writes_auditable_iteration_zero_checkpoint():
    driver = (REPO_ROOT / "recovar/em/initial_model/driver.py").read_text()
    tests = (REPO_ROOT / "tests/unit/initial_model/test_native_driver.py").read_text()
    expected_tokens = [
        '{"checkpoint_iteration": 0, "phase": "bootstrap"}',
        "test_iteration_zero_artifacts_use_the_normal_iteration_writer",
        'run_it000_class001.mrc',
        'run_it000_model.star',
        'run_it000_data.star',
        'run_it000_recovar_meta.json',
    ]
    missing = [token for token in expected_tokens if token not in driver + tests]
    assert not missing, f"native InitialModel lost iteration-zero checkpoint wiring: {missing}"


def test_vdam_frozen_trajectory_runner_and_fsc_auditor_are_merge_guarded():
    guard = (REPO_ROOT / "scripts/run_vdam_abinitio_merge_guard.py").read_text()
    runner = (REPO_ROOT / "scripts/run_vdam_relion_parity_case.py").read_text()
    auditor = (REPO_ROOT / "scripts/audit_vdam_fsc_trajectory.py").read_text()
    sbatch = (REPO_ROOT / "scripts/run_vdam_relion_parity_case.sbatch").read_text()
    expected_tokens = [
        "test_audit_vdam_fsc_trajectory.py",
        "test_run_vdam_relion_parity_case.py",
        "build_relion_command",
        "build_recovar_command",
        "--require_custom_cuda",
        'env["JAX_PLATFORMS"] = "cuda,cpu"',
        "runtime_environment.json",
        "materialize(",
        "cwd=fixture_dir",
        "paired_gpu_uuid.json",
        "signed shellwise FSC and normalized non-DC FSC-AUC only",
        "CHECKPOINTS = (0, 1, 2, 4, 8)",
        "artifact_topology_exact",
        "correlation_used",
        "--gres=gpu:1",
        "RECOVAR_CUDA_LIB",
        "cuda_backproject.cuda_available()",
        "VDAM parity provenance/GPU/CUDA-FFI gate passed",
        '--threads "${RELION_THREADS:-8}"',
        'mkdir -p "${RELION_ACC_DUMP_DIR}"',
    ]
    haystack = "\n".join([guard, runner, auditor, sbatch])
    missing = [token for token in expected_tokens if token not in haystack]
    assert not missing, f"VDAM trajectory runner/auditor lost required wiring: {missing}"


def test_vdam_fixed12_matrix_maps_array_tasks_to_frozen_case_ids():
    matrix = (REPO_ROOT / "scripts/run_vdam_relion_parity_matrix.sbatch").read_text()
    expected_tokens = [
        "#SBATCH --array=1-12%4",
        "SLURM_ARRAY_TASK_ID < 1",
        "SLURM_ARRAY_TASK_ID > 12",
        "printf 'vdam-%02d'",
        "run_vdam_relion_parity_case.sbatch",
        "OUTPUT_ROOT",
    ]
    missing = [token for token in expected_tokens if token not in matrix]
    assert not missing, f"VDAM fixed12 matrix lost task-to-case wiring: {missing}"
    assert "RECOVAR_VDAM_IMAGE_BATCH_SIZE" not in matrix
    assert "RECOVAR_EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB" not in matrix


def test_vdam_parameter_suite_freezes_user_facing_variants_and_default_resources():
    suite_path = REPO_ROOT / "docs/math/vdam_k1_parameter_suite_v1.json"
    suite = json.loads(suite_path.read_text())
    matrix = (REPO_ROOT / "scripts/run_vdam_relion_parameter_suite.sbatch").read_text()
    case_runner = (REPO_ROOT / "scripts/run_vdam_relion_parity_case.sbatch").read_text()

    assert suite["schema"] == "recovar.vdam_relion_parity_suite.v1"
    assert suite["suite_id"] == "vdam-k1-parameter-v1"
    assert len(suite["cases"]) == 11
    assert [case["id"] for case in suite["cases"]] == [f"vdam-p{index:02d}" for index in range(1, 12)]
    definitions = [case["definition"] for case in suite["cases"]]
    assert {definition["tau2_fudge"] for definition in definitions} >= {2, 4, 8}
    assert {definition["healpix_order"] for definition in definitions} >= {0, 1, 2}
    assert {definition["oversampling"] for definition in definitions} >= {0, 1, 2}
    assert {definition["padding_factor"] for definition in definitions} >= {1, 2}
    assert {definition["symmetry"] for definition in definitions} >= {"C1", "C2"}
    assert "#SBATCH --array=1-11%4" in matrix
    assert "VDAM_SCORECARD" in matrix and "VDAM_SCORECARD" in case_runner
    assert "RECOVAR_VDAM_IMAGE_BATCH_SIZE" not in matrix


def test_vdam_case_runner_uses_job_scoped_runtime_roots():
    runner = (REPO_ROOT / "scripts/run_vdam_relion_parity_case.sbatch").read_text()
    expected_tokens = [
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/",
        'export TMPDIR="${RUNTIME_ROOT}/tmp"',
        'export PIXI_HOME="${RUNTIME_ROOT}/pixi_home"',
        'export RATTLER_CACHE_DIR="${RUNTIME_ROOT}/rattler_cache"',
    ]
    missing = [token for token in expected_tokens if token not in runner]
    assert not missing, f"VDAM case runner lost job-scoped runtime roots: {missing}"


def test_native_vdam_tau2_refresh_and_ssnr_diagnostics_are_merge_guarded():
    """Protect the K=1 current-size parity fix.

    RELION refreshes tau2_class from Projector::computeFourierTransformMap
    before each E-step. Native InitialModel must keep that behavior, and must
    write sigma2/coverage diagnostics so DVP/current-size regressions are
    visible in model.star.
    """
    driver = (REPO_ROOT / "recovar/em/initial_model/driver.py").read_text()
    iteration_loop = (REPO_ROOT / "recovar/em/initial_model/iteration_loop.py").read_text()
    m_step = (REPO_ROOT / "recovar/em/initial_model/m_step.py").read_text()
    state = (REPO_ROOT / "recovar/em/initial_model/state.py").read_text()
    bind = (REPO_ROOT / "recovar/relion_bind/initialmodel_bind.cpp").read_text()
    tests = (REPO_ROOT / "tests/unit/initial_model/test_iteration_loop.py").read_text()

    haystack = "\n".join([driver, iteration_loop, m_step, state, bind, tests])
    expected_tokens = [
        "def refresh_tau2_from_projector_power",
        "vdam_projector_power_spectrum",
        "refresh_tau2_from_projector: bool = True",
        "projector_padding_factor=int(opts.padding_factor)",
        "current = refresh_tau2_from_projector_power",
        "sigma2_class:  (K, S)",
        "fourier_coverage_class: (K, S)",
        "new_sigma2_class[k] = np.asarray(sigma2",
        "new_fourier_coverage_class[k] = np.asarray(fourier_coverage",
        "state.sigma2_class[k]",
        "state.fourier_coverage_class[k]",
        "test_iteration_loop_refreshes_tau2_before_estep",
    ]
    missing = [token for token in expected_tokens if token not in haystack]
    assert not missing, f"native InitialModel lost tau2 refresh/SSNR diagnostic wiring: {missing}"


def test_native_vdam_postmerge_parity_fixes_are_merge_guarded():
    """Guard the load-bearing parity fixes from codex/vdam-postmerge-20260507.

    These string contracts complement functional unit tests. They make merge
    conflict mistakes fail early when branches touch the same InitialModel,
    dense K-class, or RELION binding code.
    """
    driver = (REPO_ROOT / "recovar/em/initial_model/driver.py").read_text()
    dense_adapter = (REPO_ROOT / "recovar/em/initial_model/dense_adapter.py").read_text()
    iteration_loop = (REPO_ROOT / "recovar/em/initial_model/iteration_loop.py").read_text()
    state = (REPO_ROOT / "recovar/em/initial_model/state.py").read_text()
    init = (REPO_ROOT / "recovar/em/initial_model/init.py").read_text()
    k_class = (REPO_ROOT / "recovar/em/dense_single_volume/k_class.py").read_text()
    local_em = (REPO_ROOT / "recovar/em/dense_single_volume/local_em_engine.py").read_text()
    bind = (REPO_ROOT / "recovar/relion_bind/initialmodel_bind.cpp").read_text()
    unit_tests = "\n".join(
        [
            (REPO_ROOT / "tests/unit/initial_model/test_dense_adapter.py").read_text(),
            (REPO_ROOT / "tests/unit/initial_model/test_iteration_loop.py").read_text(),
            (REPO_ROOT / "tests/unit/initial_model/test_native_driver.py").read_text(),
        ]
    )
    guard_scripts = "\n".join(
        [
            (REPO_ROOT / "scripts/run_vdam_abinitio_merge_guard.py").read_text(),
            (REPO_ROOT / "scripts/run_em_merge_guard_slurm.sh").read_text(),
        ]
    )

    expected_by_area = {
        "bootstrap_fft_order": [
            "windowFourierTransform(Faux, Fimg",
            "CenterFFTbySign(Fimg)",
        ],
        "sigma_offset_and_noise_momentum": [
            "sigma2_offset: float = 100.0",
            "sigma2_offset=100.0",
            "MIN_SIGMA2_OFFSET_ANGSTROM2",
            "wsum_sigma2_offset / (2.0 * sum_weight)",
            "def update_noise_from_estep_meta",
            "normalize_wsum_to_sigma2_noise",
            "int(state.ori_size) ** 4",
            "current = update_noise_from_estep_meta(current, meta, do_grad=do_grad, mu=mu)",
        ],
        "direction_and_translation_priors": [
            "relion_sigma_offset_prior_center",
            "translation_prior_centers",
            "def _class_direction_rotation_log_prior",
            "class_rotation_log_prior",
            "values[positive] / mean_pdf",
            "rotation_log_prior=_class_pass2_rotation_log_prior(group_kwargs, class_index)",
            "local_layout = tuple(local_layouts)",
            "allow_empty=True",
            "rotation_log_prior=group_kwargs.get(\"class_rotation_log_prior\"",
            "stats_use_reconstruction_probs=True",
            "class_posterior_sums_from_noise=False",
            "class_reconstruction_support_sums",
            "class_bpref_weight_sums",
            "class_posterior_sums_override",
            "reconstruction_probs_sum_t if stats_use_reconstruction_probs else probs_sum_t",
            "relion_f32_fine_posterior=bool(",
            "sparse_diagnostics.relion_x_half_f32_fine_posterior_enabled()",
            "use_relion_f32_fine_posterior=use_relion_f32_fine_posterior",
            "_relion_f32_fine_reconstruction_probs(",
        ],
        "relion_model_star_contract": [
            "data_model_pdf_orient_class_",
            "_rlnOrientationDistribution #1",
            "_rlnSigmaOffsetsAngst",
        ],
        "kclass_profile_and_unit_guards": [
            "profile_summary",
            "return_profile",
            "initial_model_vdam_unit_slice",
            "tests/unit/test_k_class_joint_semantics.py",
        ],
        "functional_tests": [
            "test_updates_sigma2_noise_with_vdam_momentum_on_subset_iterations",
            "test_iteration_loop_feeds_updated_sigma2_noise_to_next_estep",
            "test_model_star_uses_relion_model_blocks",
            "test_dense_initial_model_estep_sparse_pass2_preserves_k_class_state",
        ],
    }
    haystack = "\n".join(
        [driver, dense_adapter, iteration_loop, state, init, k_class, local_em, bind, unit_tests, guard_scripts]
    )
    missing = {
        area: [token for token in tokens if token not in haystack]
        for area, tokens in expected_by_area.items()
    }
    missing = {area: tokens for area, tokens in missing.items() if tokens}
    assert not missing, f"VDAM post-merge parity guard lost required wiring: {missing}"


def test_relion_initialmodel_reference_checker_rejects_autorefine(tmp_path):
    mod = _load_long_guard_module()

    initialmodel_dir = tmp_path / "initialmodel"
    initialmodel_dir.mkdir()
    (initialmodel_dir / "run_it008_optimiser.star").write_text(
        "\n".join(
            [
                "# RELION optimiser; version test",
                (
                    "# --o out/run --iter 8 --grad --denovo_3dref --i particles.star "
                    "--ctf --K 1 --sym C1 --flatten_solvent --zero_mask "
                    "--dont_combine_weights_via_disc --pool 3 --pad 1 "
                    "--particle_diameter 200 --oversampling 1 --healpix_order 1 "
                    "--offset_range 6 --offset_step 2 --auto_sampling --tau2_fudge 4 "
                    "--j 4 --random_seed 0"
                ),
                "",
                "data_optimiser_general",
                "_rlnCurrentIteration 8",
                "_rlnNumberOfIterations 8",
                "_rlnDoSplitRandomHalves 0",
                "_rlnDoGradientRefine 1",
                "_rlnDoAutoRefine 0",
            ]
        )
    )
    mod._assert_relion_initialmodel_reference(initialmodel_dir, expected_iter=8)

    autorefine_dir = tmp_path / "autorefine"
    autorefine_dir.mkdir()
    (autorefine_dir / "run_it008_optimiser.star").write_text(
        "\n".join(
            [
                "# RELION optimiser; version test",
                "# --i particles.star --ref reference_init_relion.mrc --o out/run --auto_refine --split_random_halves --pad 2 --iter 8",
                "",
                "data_optimiser_general",
                "_rlnCurrentIteration 8",
                "_rlnNumberOfIterations 999",
                "_rlnDoSplitRandomHalves 1",
                "_rlnDoGradientRefine 0",
                "_rlnDoAutoRefine 1",
            ]
        )
    )
    try:
        mod._assert_relion_initialmodel_reference(autorefine_dir, expected_iter=8)
    except AssertionError as exc:
        assert "--grad" in str(exc)
        assert "--denovo_3dref" in str(exc)
    else:
        raise AssertionError("auto-refine fixture was accepted as native InitialModel reference")


def test_merge_guard_dry_run_writes_reproducibility_ledger(tmp_path):
    ledger = run_guard(tier="cpu", quick=True, output_dir=tmp_path, dry_run=True)

    assert ledger["schema"] == "vdam_abinitio_merge_guard.v1"
    assert ledger["ok"] is True
    assert ledger["dry_run"] is True
    assert ledger["provenance"]["ok"] is True
    assert [command["name"] for command in ledger["commands"]] == [
        "py_compile",
        "vdam_abinitio_contracts",
        "vdam_frozen_scorecard",
        "initial_model_vdam_unit_slice",
        "em_fast_guard",
    ]

    summary_path = Path(ledger["summary_path"])
    assert summary_path.exists()
    stored = json.loads(summary_path.read_text())
    assert stored["git"]["commit"]
    assert stored["commands"][0]["skipped"] is True
