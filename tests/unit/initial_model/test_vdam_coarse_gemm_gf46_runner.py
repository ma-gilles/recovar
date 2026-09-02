from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "run_vdam_coarse_gemm_gf46_gate.sbatch"


def _source() -> str:
    return RUNNER.read_text()


def _array(name: str) -> list[str]:
    match = re.search(
        rf"^{name}=\(\n(?P<body>.*?)^\)$",
        _source(),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, name
    return [line.strip().strip('"') for line in match.group("body").splitlines() if line.strip()]


def test_runner_pins_clean_committed_head_tree_and_exact_h100() -> None:
    source = _source()

    assert ': "${EXPECTED_REPO_HEAD:?pin the committed harness head}"' in source
    assert ': "${EXPECTED_REPO_TREE:?pin the committed harness tree}"' in source
    assert ': "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the benchmark source manifest}"' in source
    assert "#SBATCH --nodelist=della-h21g4" in source
    assert "readonly TARGET_NODE=della-h21g4" in source
    assert "readonly TARGET_GPU_UUID=GPU-099c0d77-bb85-f2e9-f628-148b733c9176" in source
    assert 'status --porcelain=v1 --untracked-files=all' in source
    assert 'vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}"' in source
    assert 'vdam_select_target_gpu "${TARGET_GPU_UUID}" 0' in source
    assert 'test "${selected_gpu_uuid}" = "${TARGET_GPU_UUID}"' in source
    assert '[[ "${gpu_name}" == *H100* ]]' in source


def test_runner_preloads_cusparse_for_cpu_analyzer() -> None:
    source = _source()
    assert (
        'env LD_PRELOAD="${CUSPARSE_LIBRARY}" JAX_PLATFORMS=cpu '
        'JAX_PLATFORM_NAME=cpu' in source
    )


def test_runner_reuses_the_hash_pinned_qualified_runtime_without_building() -> None:
    source = _source()

    assert (
        "readonly QUALIFIED_GPU_GATE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "vdam_coarse_prehalf_h100_13315513"
    ) in source
    for digest in (
        "9ab443af3a90f63bc0fd6eeac90f9c15f84d7f667c6c19171682a11f0c168cc8",
        "2af7bf1e4cbdc10705948d907c087d1662db612fe8d57362f1390033ac6c047b",
        "48556a44c0dd1570866beb838e6fcbea771bce93d413acd4197bb2f254b72d23",
        "9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980",
        "58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf",
    ):
        assert digest in source
    assert 'cp --reflink=auto "${QUALIFIED_CUDA}" "${CUDA_BINARY}"' in source
    assert 'cp --reflink=auto "${RELION_BIND_SOURCE}" "${RELION_BIND_BINARY}"' in source
    assert "make -C" not in source
    assert "nvcc" not in source.lower()
    assert "pytest" not in source
    assert "nsys" not in source.lower()


def test_runner_pins_the_exact_gf46_inputs_and_manifests() -> None:
    source = _source()

    assert (
        "readonly GF46_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "vdam_full_expansion_v3_984637b7d_87274be_20260826/vdam-gf46/repeat-01/vdam-gf46"
    ) in source
    assert "de224471a690d1faaae4067217dbcc90b632269d62b0b3372b20aafa69157d91" in source
    assert "804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9" in source
    assert len(_array("INPUT_FILES")) == 9
    assert 'sha256sum "${INPUT_FILES[@]}" > "${PROVENANCE}/input_manifest.sha256"' in source
    assert source.count("verify_frozen_inputs") >= 3
    source_files = _array("SOURCE_FILES")
    for required in (
        "recovar/em/dense_single_volume/helpers/scoring.py",
        "recovar/em/dense_single_volume/helpers/significance.py",
        "recovar/em/initial_model/dense_adapter.py",
        "scripts/run_vdam_late_iteration_profile.py",
        "scripts/analyze_vdam_coarse_gemm_gf46_gate.py",
        "scripts/run_vdam_coarse_gemm_gf46_gate.sbatch",
        "tests/unit/initial_model/test_vdam_coarse_gemm_gf46_analyzer.py",
        "tests/unit/initial_model/test_vdam_coarse_gemm_gf46_runner.py",
    ):
        assert required in source_files
    assert 'cmp "${PROVENANCE}/source_manifest.sha256"' in source


def test_runner_uses_a_one_shot_raw_pair_for_two_pinned_selected_targets() -> None:
    source = _source()
    raw_start = source.index("raw_command=(")
    raw_end = source.index('touch "${ROOT}/RAW_COMPLETED"')
    raw = source[raw_start:raw_end]

    assert "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=1" in raw
    assert '"RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR=${RAW_DIAGNOSTIC_DIR}"' in raw
    assert "readonly RAW_TARGETS_CSV=1,2160" in source
    assert '"RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES=${RAW_TARGETS_CSV}"' in raw
    assert '"${PIXI_PY}" -m scripts.run_ab_initio' in raw
    assert "scripts.run_vdam_late_iteration_profile" not in raw
    assert '--diagnostic_continue_optimiser "${CHECKPOINT}"' in raw
    assert '--diagnostic_stop_after_iteration "${PROFILED_ITERATION}"' in raw
    assert 'test ! -e "${RAW_OUTPUT}/run_it182_recovar_meta.json"' in raw
    assert "'coarse_gemm_manifest_*.json'" in raw
    assert "'coarse_gemm_scope_*.json'" in raw
    assert "'coarse_gemm_ab_*.npz'" in raw
    assert "'coarse_gemm_scope_*.json' | wc -l)\" -eq 1" in raw
    assert "'coarse_gemm_ab_*.npz' | wc -l)\" -eq 2" in raw


def test_runner_preflights_the_native_frozen_subset_and_pseudo_halfsets() -> None:
    source = _source()

    assert "EXPECTED_TARGETS = (1, 2160)" in source
    assert "EXPECTED_POSITIONS = (72, 999)" in source
    assert "EXPECTED_PART_IDS = (1, 2160)" in source
    assert "EXPECTED_HALFSETS = (1, 0)" in source
    assert "c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be" in source
    assert "select_subset_for_iter(" in source
    assert "_micrograph_sort_order(main_star)" in source
    assert "_optics_group_indices(main_star)" in source
    assert '"rlnSgdSubsetSize"' in source
    assert 'raise RuntimeError("native RELION shuffle binding was not used")' in source
    assert 'set(target_halfsets) != {0, 1}' in source
    assert 'requested_batch_indices != (0, 1)' in source
    assert '"joint_halfset_particle_stream": True' in source
    assert '"${PROVENANCE}/raw_target_preflight.json"' in source


def test_raw_and_timing_arms_explicitly_disable_every_competing_selector() -> None:
    source = _source()
    for selector in (
        "RECOVAR_K1_COARSE_FUSED_PROJECTOR=0",
        "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION=0",
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION=0",
        "RECOVAR_K1_COARSE_PREHALF_WEIGHT=0",
        "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL=0",
        "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS=0",
        "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=0",
    ):
        assert source.count(selector) >= 2
    for prerequisite in (
        "RECOVAR_K1_COARSE_GAUSSIAN_FFI=1",
        "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1",
        "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1",
        "RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1",
    ):
        assert source.count(prerequisite) >= 2


def test_runner_executes_only_the_diagnostic_unset_a_b_b_a_timing_panel() -> None:
    source = _source()

    assert "RUN_LABELS=(direct_1 gemm_1 gemm_2 direct_2)" in source
    assert "RUN_GEMM=(0 1 1 0)" in source
    assert "-u RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR" in source
    assert "-u RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES" in source
    assert '"RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=${gemm}"' in source
    assert source.count('"${PIXI_PY}" -m scripts.run_vdam_late_iteration_profile') == 1
    assert "--cuda-profiler-range" not in source
    assert "--audit-raw-image-cache" not in source
    assert 'test -z "$(find "${run_root}" -type f -name \'coarse_gemm*\'' in source


def test_all_timing_arms_share_the_frozen_late_profile_contract() -> None:
    source = _source()

    for argument in (
        '--checkpoint-optimiser "${CHECKPOINT}"',
        '--input-star "${DATA_STAR}"',
        '--data-dir "${DATA_DIR}"',
        '--checkpoint-iteration "${CHECKPOINT_ITERATION}"',
        '--nr-iter "${NR_ITER_SCHEDULE}"',
        "--random-seed 29",
        "--image-batch-size 500",
        "--exact-local-bucket-radix 4",
        "--exact-local-physical-order-chunk-size 0",
    ):
        assert argument in source
    assert "RECOVAR_EM_RAW_IMAGE_CACHE=auto" in source
    assert "RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB=16" in source
    assert '/usr/bin/time -v -o "${run_root}/process.time"' in source
    assert 'test ! -e "${run_root}/profile/warm/run_it182_recovar_meta.json"' in source


def test_runner_predeclares_point_nine_runtime_gate_and_seals_pass_only_on_success() -> None:
    source = _source()

    assert "readonly MATERIAL_END_TO_END_RATIO=0.90" in source
    assert '"material_end_to_end_ratio": 0.90' in source
    assert '"science_promotion_allowed": False' in source
    assert '"default_enablement_allowed": False' in source
    assert '"${PIXI_PY}" -m scripts.analyze_vdam_coarse_gemm_gf46_gate' in source
    assert 'if [[ "${analyzer_status}" == 0 ]]' in source
    assert 'touch "${ROOT}/PASSED"' in source
    assert 'touch "${ROOT}/COMPLETED"' in source
    assert ') > "${ROOT}/SHA256SUMS"' in source
    assert 'sha256sum "${ROOT}/SHA256SUMS" > "${ROOT}/SHA256SUMS.sha256"' in source
    assert 'find "${ROOT}" -type f -exec chmod a-w {} +' in source
    assert 'find "${ROOT}" -depth -type d -exec chmod a-w {} +' in source
    assert source.index('touch "${ROOT}/PASSED"') < source.index('touch "${ROOT}/COMPLETED"')
