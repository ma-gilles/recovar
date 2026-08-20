from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EM_DIR = ROOT / "recovar" / "em"
CLAUDE = EM_DIR / "CLAUDE.md"
AGENTS = EM_DIR / "AGENTS.md"
LEDGER = ROOT / "docs" / "math" / "em_parity_best_metrics.md"
PARALLEL_TEST_RUNNER = ROOT / "scripts" / "run_tests_parallel.sh"


def test_em_agent_guides_stay_in_sync():
    assert CLAUDE.read_text() == AGENTS.read_text()


def test_em_agent_guides_pin_validation_policy():
    guide = CLAUDE.read_text()
    required = [
        "cmp recovar/em/CLAUDE.md recovar/em/AGENTS.md",
        "at most once every 3-4 hours",
        "test-em-parity-fast",
        "both K=1 and K=4",
        "at least 100k particles",
        "at least 256x256 images",
        "better, worse, or same",
        "docs/math/em_parity_best_metrics.md",
        "scripts/prepare_pdb_k1_relion_sanity_benchmark.py",
        "scripts/prepare_cryobench_pdb_multiclass_relion_parity_benchmark.py",
        "K=15 run is useful stress coverage but is not the K=4",
        "./scripts/run_tests_parallel.sh long-test",
        "pixi run test-full",
    ]
    for text in required:
        assert text in guide


def test_em_best_metrics_ledger_has_quality_and_perf_contract():
    ledger = LEDGER.read_text()
    required = [
        ">=100k, >=256x256",
        "final_half1_FSC_AUC_vs_RELION",
        "final_half2_FSC_AUC_vs_RELION",
        "merged_FSC_AUC_vs_RELION",
        "recovar_FSC_AUC_vs_GT",
        "relion_FSC_AUC_vs_GT",
        "minimum_non_DC_shell_FSC_vs_RELION",
        "FSC_0.5_shell_RECOVAR",
        "FSC_0.143_shell_RECOVAR",
        "Pmax_gap_RECOVAR_minus_RELION",
        "pose_angle_error_vs_RELION",
        "translation_error_vs_RELION",
        "K4_class_assignment_or_map_match",
        "RECOVAR_end_to_end_walltime",
        "RELION_end_to_end_walltime",
        "RECOVAR_images_per_second",
        "RELION_images_per_second",
        "RECOVAR_peak_gpu_memory",
        "RELION_peak_gpu_memory",
        "Accepted as new best",
    ]
    for text in required:
        assert text in ledger
    assert "Correlation values in legacy" in ledger
    assert "cannot accept or reject" in ledger


def test_parallel_test_runner_accepts_external_runtime_root():
    runner = PARALLEL_TEST_RUNNER.read_text()
    assert 'RUNTIME_ROOT="${RECOVAR_TEST_RUNTIME_ROOT:-${WORKDIR}/.tmp}"' in runner
    assert runner.count('${RUNTIME_ROOT}/slurm_\\${SLURM_JOB_ID}') == 2
    assert runner.count('${RUNTIME_ROOT}/pixi_home_\\${SLURM_JOB_ID}') == 2
    assert runner.count('${RUNTIME_ROOT}/rattler_cache_\\${SLURM_JOB_ID}') == 2


def test_parallel_test_runner_binds_workers_to_one_slurm_gpu():
    runner = PARALLEL_TEST_RUNNER.read_text()
    inherited = 'CUDA_FIRST_GPU="\\${CUDA_VISIBLE_DEVICES:-}"'
    slurm_fallback = 'SLURM_VISIBLE_GPUS="\\${SLURM_STEP_GPUS:-\\${SLURM_JOB_GPUS:-}}"'
    assert inherited in runner
    assert slurm_fallback in runner
    assert runner.index(inherited) < runner.index(slurm_fallback)
    assert 'CUDA_FIRST_GPU="\\${SLURM_VISIBLE_GPUS%%,*}"' in runner
    assert 'export CUDA_VISIBLE_DEVICES="\\${CUDA_FIRST_GPU}"' in runner
    assert "len(devices) == 1 and devices[0].platform == 'gpu'" in runner


def test_parallel_test_runner_supports_external_relion_binding():
    runner = PARALLEL_TEST_RUNNER.read_text()
    assert 'RELION_BIND_BUILD_DIR="${RECOVAR_TEST_RELION_BIND_BUILD_DIR:-}"' in runner
    assert 'export RECOVAR_RELION_BIND_BUILD_DIR="${RELION_BIND_BUILD_DIR}"' in runner
    assert "if os.environ.get('RECOVAR_RELION_BIND_BUILD_DIR')" in runner
    assert "from recovar.relion_bind import _relion_bind_core" in runner
