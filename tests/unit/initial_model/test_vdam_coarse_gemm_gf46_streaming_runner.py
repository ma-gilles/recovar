from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "run_vdam_coarse_gemm_gf46_streaming_selector.sbatch"


def _source() -> str:
    return RUNNER.read_text()


def _array(name: str) -> list[str]:
    match = re.search(
        rf"^{name}=\(\n(?P<body>.*?)^\)$",
        _source(),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, name
    return [
        line.strip().strip('"')
        for line in match.group("body").splitlines()
        if line.strip()
    ]


def test_streaming_runner_pins_clean_committed_head_tree_and_exact_h100() -> None:
    source = _source()

    assert ': "${EXPECTED_REPO_HEAD:?pin the committed streaming-diagnostic head}"' in source
    assert ': "${EXPECTED_REPO_TREE:?pin the committed streaming-diagnostic tree}"' in source
    assert ': "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the compact source manifest}"' in source
    assert ': "${PIXI_PY_OVERRIDE:?pin the qualified RECOVAR pixi interpreter}"' in source
    assert "#SBATCH --nodelist=della-h21g4" in source
    assert "readonly TARGET_NODE=della-h21g4" in source
    assert "readonly TARGET_GPU_UUID=GPU-099c0d77-bb85-f2e9-f628-148b733c9176" in source
    assert 'status --porcelain=v1 --untracked-files=all' in source
    assert 'vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}"' in source
    assert 'vdam_select_target_gpu "${TARGET_GPU_UUID}" 0' in source
    assert 'test "${selected_gpu_uuid}" = "${TARGET_GPU_UUID}"' in source
    assert '[[ "${gpu_name}" == *H100* ]]' in source


def test_streaming_runner_reuses_the_qualified_selected_block_runtime() -> None:
    source = _source()

    assert (
        "readonly QUALIFIED_GPU_GATE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/"
        "codex/vdam_coarse_rotation_blocks_h100_695a629fa_20260901"
    ) in source
    for value in (
        "695a629fa70ee951734d98728bb3daffcc53bd88",
        "04a3ca49be3547cc8329d6c27c7b1aa2acf30c3e",
        "13328717",
        "664c741946c16df13b59ae41c6a98949c5463beddbe33193e66948ae87a91de0",
        "3e05f86750f7f3bdb3df1b956cd8cd9b3ae11c655d0455ca4c71b714bc8745d8",
        "9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980",
        "58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf",
    ):
        assert value in source
    assert 'cp --reflink=auto "${QUALIFIED_CUDA}" "${CUDA_BINARY}"' in source
    assert 'cp --reflink=auto "${RELION_BIND_SOURCE}" "${RELION_BIND_BINARY}"' in source
    assert 'callable(getattr(cuda_backproject, "relion_coarse_diff2_rotation_blocks_f32", None))' in source
    assert "make -C" not in source
    assert "nvcc" not in source.lower()
    assert "nsys" not in source.lower()


def test_streaming_runner_pins_frozen_gf46_inputs_and_complete_source_scope() -> None:
    source = _source()

    assert (
        "readonly GF46_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "vdam_full_expansion_v3_984637b7d_87274be_20260826/vdam-gf46/"
        "repeat-01/vdam-gf46"
    ) in source
    assert "de224471a690d1faaae4067217dbcc90b632269d62b0b3372b20aafa69157d91" in source
    assert "804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9" in source
    assert len(_array("INPUT_FILES")) == 9
    source_files = _array("SOURCE_FILES")
    for required in (
        "recovar/cuda/cuda_backproject.cu",
        "recovar/cuda_backproject.py",
        "recovar/em/dense_single_volume/helpers/coarse_gemm_streaming.py",
        "recovar/em/dense_single_volume/helpers/scoring.py",
        "recovar/em/dense_single_volume/helpers/significance.py",
        "recovar/em/initial_model/iteration_loop.py",
        "scripts/run_vdam_coarse_gemm_gf46_streaming_selector.sbatch",
        "tests/unit/test_coarse_gaussian_gemm_streaming.py",
        "tests/unit/initial_model/test_vdam_coarse_gemm_gf46_streaming_runner.py",
    ):
        assert required in source_files
    assert 'sha256sum "${INPUT_FILES[@]}" > "${PROVENANCE}/input_manifest.sha256"' in source
    assert source.count('cmp "${PROVENANCE}/source_manifest.sha256"') >= 2
    assert source.count('cmp "${PROVENANCE}/runtime_manifest.sha256"') >= 2


def test_streaming_runner_preflights_the_exact_all_particle_subset_once() -> None:
    source = _source()

    assert "readonly EXPECTED_PARTICLE_COUNT=3000" in source
    assert "readonly EXPECTED_SUBSET_SIZE=1000" in source
    assert "EXPECTED_TARGETS = (1, 2160)" in source
    assert "EXPECTED_POSITIONS = (72, 999)" in source
    assert "EXPECTED_PART_IDS = (1, 2160)" in source
    assert "EXPECTED_HALFSETS = (1, 0)" in source
    assert "c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be" in source
    assert "select_subset_for_iter(" in source
    assert "_micrograph_sort_order(main_star)" in source
    assert "_optics_group_indices(main_star)" in source
    assert 'set(target_halfsets) != {0, 1}' in source
    assert '"joint_halfset_particle_stream": True' in source
    assert 'scope_ids == expected_list' in source
    assert 'artifact_ids == expected_list' in source
    assert 'len(set(artifact_ids)) == expected_count' in source
    assert 'aggregate.get("all_particles_captured_exactly_once") is True' in source


def test_streaming_runner_executes_one_transition_without_timing_arms_or_score_cube() -> None:
    source = _source()
    command_start = source.index("stream_command=(")
    command_end = source.index('test -s "${OUTPUT}/run_it181_recovar_meta.json"')
    command = source[command_start:command_end]

    assert "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=1" in command
    assert '"RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR=${DIAGNOSTIC_DIR}"' in command
    assert '"RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_TOPK=${STREAM_TOPK}"' in command
    assert '"${PIXI_PY}" -m scripts.run_ab_initio' in command
    assert '--diagnostic_continue_optimiser "${CHECKPOINT}"' in command
    assert '--diagnostic_stop_after_iteration "${PROFILED_ITERATION}"' in command
    assert "scripts.run_vdam_late_iteration_profile" not in source
    assert "RUN_LABELS=" not in source
    assert "RUN_GEMM=" not in source
    assert "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR=${" not in command
    assert "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES=${" not in command
    assert "coarse_gemm_ab_*.npz" in source
    assert 'require(not scalar_bool(artifact, "stores_score_cube")' in source
    assert '"trajectory_timing_arms_enabled": False' in source
    assert '"clean_timing_eligible": False' in source


def test_streaming_runner_requires_dual_raw_and_posterior_v2_certificates() -> None:
    source = _source()

    assert "readonly STREAM_TOPK=2048" in source
    assert 'int(aggregate.get("schema_version", 0)) >= 2' in source
    assert 'int(summary.get("schema_version", 0)) >= 2' in source
    for required in (
        '"pre_prior_all_candidate"',
        '"relion_nonzero_rescore"',
        '"relion_raw_max_rescore"',
        '"relion_rescore_union"',
        '"pre_prior_finite_pair_count"',
        '"raw_max_error_safe_source_rotation_block_ids"',
        '"relion_rescore_source_rotation_block_union_ids"',
        '"posterior_streaming_state_bytes"',
        '"pre_prior_streaming_state_bytes"',
        '"persistent_streaming_state_bytes"',
    ):
        assert required in source
    assert "persistent_bytes == posterior_bytes + pre_prior_bytes" in source
    assert 'require(raw_or_pre_prior_fields' in source
    assert 'require(posterior_or_support_fields' in source


def test_streaming_runner_disables_every_competing_coarse_selector() -> None:
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
        assert selector in source
    for prerequisite in (
        "RECOVAR_K1_COARSE_GAUSSIAN_FFI=1",
        "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1",
        "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1",
        "RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1",
    ):
        assert prerequisite in source


def test_streaming_runner_seals_immutable_diagnostic_only_results() -> None:
    source = _source()

    assert 'touch "${ROOT}/IDENTITY_VALIDATED"' in source
    assert '"science_promotion_allowed": False' in source
    assert '"default_enablement_allowed": False' in source
    assert 'touch "${ROOT}/COMPLETED"' in source
    assert ') > "${ROOT}/SHA256SUMS"' in source
    assert 'sha256sum "${ROOT}/SHA256SUMS" > "${ROOT}/SHA256SUMS.sha256"' in source
    assert 'find "${ROOT}" -type f -exec chmod a-w {} +' in source
    assert 'find "${ROOT}" -depth -type d -exec chmod a-w {} +' in source
    assert source.index('touch "${ROOT}/IDENTITY_VALIDATED"') < source.index(
        'touch "${ROOT}/COMPLETED"'
    )
