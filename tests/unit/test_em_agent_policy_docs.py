from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EM_DIR = ROOT / "recovar" / "em"
CLAUDE = EM_DIR / "CLAUDE.md"
AGENTS = EM_DIR / "AGENTS.md"
LEDGER = ROOT / "docs" / "math" / "em_parity_best_metrics.md"


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
        "final_half1_corr_vs_RELION",
        "final_half2_corr_vs_RELION",
        "merged_corr_vs_RELION",
        "recovar_corr_vs_GT",
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
