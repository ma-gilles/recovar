from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "run_vdam_raw_cache_abba.sbatch"


def _source() -> str:
    return RUNNER.read_text()


def _array(name: str) -> list[str]:
    match = re.search(rf"^{name}=\(\n(?P<body>.*?)^\)$", _source(), re.MULTILINE | re.DOTALL)
    assert match is not None, name
    return [line.strip() for line in match.group("body").splitlines() if line.strip()]


def test_runner_pins_qualified_gate_runtime_and_exact_h100() -> None:
    source = _source()

    assert "unset PYTHONOPTIMIZE" in source
    assert source.count("sys.flags.optimize != 0") == 2
    assert "#SBATCH --nodelist=della-h21g4" in source
    assert "readonly TARGET_NODE=della-h21g4" in source
    assert "readonly TARGET_GPU_UUID=GPU-099c0d77-bb85-f2e9-f628-148b733c9176" in source
    assert (
        "readonly QUALIFIED_GPU_GATE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "vdam_coarse_prehalf_h100_13315513"
    ) in source
    assert "readonly QUALIFIED_CUDA=${QUALIFIED_GPU_GATE_ROOT}/build/libcuda_backproject.so" in source
    assert "runtime/cuda/libcuda_backproject.so" not in source.split("readonly QUALIFIED_CUDA=", 1)[1].splitlines()[0]
    for digest in (
        "9ab443af3a90f63bc0fd6eeac90f9c15f84d7f667c6c19171682a11f0c168cc8",
        "2af7bf1e4cbdc10705948d907c087d1662db612fe8d57362f1390033ac6c047b",
        "48556a44c0dd1570866beb838e6fcbea771bce93d413acd4197bb2f254b72d23",
        "9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980",
        "9b32b4e9beee469bc8c26640db228b04583234505c315e23f7e622efd61a68ab",
        "58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf",
    ):
        assert digest in source
    assert 'cd "${QUALIFIED_GPU_GATE_ROOT}"' in source
    assert "sha256sum -c SHA256SUMS" in source
    assert 'vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}"' in source
    assert 'vdam_select_target_gpu "${TARGET_GPU_UUID}" 0' in source
    assert 'test "${selected_gpu_uuid}" = "${TARGET_GPU_UUID}"' in source
    assert '[[ "${gpu_name}" == *H100* ]]' in source


def test_runner_allows_only_declared_benchmark_overlay() -> None:
    source = _source()

    assert ': "${EXPECTED_REPO_HEAD:?pin the committed harness head}"' in source
    assert ': "${EXPECTED_REPO_TREE:?pin the committed harness tree}"' in source
    assert ': "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the benchmark source manifest}"' in source
    assert "readonly SCIENCE_BASE_HEAD=77e09c292e438a265a6d157414b2a0fe525710e6" in source
    assert "readonly SCIENCE_BASE_TREE=384448a3ce6639e571cad53d2b82c1b18df1979a" in source
    assert 'status --porcelain=v1 --untracked-files=all' in source
    assert 'merge-base --is-ancestor "${SCIENCE_BASE_HEAD}" "${EXPECTED_REPO_HEAD}"' in source
    assert "diff --name-only --diff-filter=ACDMRTUXB" in source
    assert "science-base overlay contains an unapproved benchmark path" in source
    assert _array("BENCHMARK_DIFF_ALLOWLIST") == [
        "recovar/data_io/image_loader.py",
        "recovar/em/vdam/iteration_loop.py",
        "scripts/analyze_vdam_raw_cache_abba.py",
        "scripts/probe_vdam_raw_cache_memory.py",
        "scripts/run_vdam_late_iteration_profile.py",
        "scripts/run_vdam_raw_cache_abba.sbatch",
        "tests/unit/test_image_loader.py",
        "tests/unit/initial_model/test_iteration_loop.py",
        "tests/unit/initial_model/test_vdam_late_iteration_profile.py",
        "tests/unit/initial_model/test_vdam_raw_cache_memory_probe.py",
        "tests/unit/initial_model/test_vdam_raw_cache_abba_analyzer.py",
        "tests/unit/initial_model/test_vdam_raw_cache_abba_runner.py",
    ]
    assert 'test "$(validate_benchmark_overlay)" = "${benchmark_overlay}"' in source
    assert '"${PROVENANCE}/benchmark_overlay.txt"' in source


def test_runner_pins_source_and_exact_gf46_input_manifests() -> None:
    source = _source()
    source_files = _array("SOURCE_FILES")

    for required in (
        "recovar/data_io/image_loader.py",
        "recovar/data_io/staging.py",
        "recovar/em/helpers/batch_planning.py",
        "recovar/em/vdam/driver.py",
        "tests/unit/test_image_loader.py",
        "tests/unit/initial_model/test_iteration_loop.py",
        "scripts/run_ab_initio.py",
        "scripts/run_vdam_late_iteration_profile.py",
        "scripts/probe_vdam_raw_cache_memory.py",
        "scripts/summarize_vdam_nsys_sqlite.py",
        "scripts/analyze_vdam_raw_cache_abba.py",
        "scripts/run_vdam_raw_cache_abba.sbatch",
        "tests/unit/initial_model/test_vdam_raw_cache_abba_runner.py",
        "tests/unit/initial_model/test_vdam_raw_cache_memory_probe.py",
    ):
        assert required in source_files
    assert "write_source_manifest > \"${PROVENANCE}/source_manifest.sha256\"" in source
    assert 'cmp "${PROVENANCE}/source_manifest.sha256"' in source
    assert (
        "readonly GF46_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "vdam_full_expansion_v3_984637b7d_87274be_20260826/vdam-gf46/repeat-01/vdam-gf46"
    ) in source
    assert "de224471a690d1faaae4067217dbcc90b632269d62b0b3372b20aafa69157d91" in source
    assert "804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9" in source
    assert len(_array("INPUT_FILES")) == 9
    assert 'sha256sum "${INPUT_FILES[@]}" > "${PROVENANCE}/input_manifest.sha256"' in source
    assert source.count("verify_frozen_inputs") >= 3


def test_runner_executes_exact_off_auto_auto_off_panel_without_force() -> None:
    source = _source()

    assert (
        "RUN_LABELS=(cache_off_1 cache_auto_1 cache_auto_2 cache_off_2 "
        "cache_auto_3 cache_off_3 cache_off_4 cache_auto_4)"
    ) in source
    assert "RUN_MODES=(off auto auto off auto off off auto)" in source
    assert '"RECOVAR_EM_RAW_IMAGE_CACHE=${mode}"' in source
    assert '"RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB=${RAW_IMAGE_CACHE_MAX_GB}"' in source
    assert '"RECOVAR_CACHE_DIR="' in source
    assert 'assert os.environ.get("RECOVAR_CACHE_DIR") == ""' in source
    assert "readonly RAW_IMAGE_CACHE_MAX_GB=16" in source
    assert "readonly RAW_IMAGE_CACHE_EXPECTED_BYTES=196608000" in source
    assert 'RECOVAR_EM_RAW_IMAGE_CACHE=force' not in source
    assert 'RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB=force' not in source
    assert "--audit-raw-image-cache" in source
    assert 'assert mode == "auto", mode' in source
    assert 'assert len(events) == 1, events' in source
    assert 'assert event["num_images"] == 3000, event' in source
    assert 'assert event["image_size"] == 128, event' in source
    assert 'assert event["cached_before"] is False, event' in source
    assert 'assert event["cached_after"] is True, event' in source
    assert 'assert event["cached_nbytes"] == expected_bytes, event' in source
    assert 'assert event["loader_type"] == "recovar.data_io.image_loader.StarLoader", event' in source
    assert '"loader_type": "recovar.data_io.image_loader.MRCLoader"' in source
    assert '"io_path": particle_stack' in source
    assert '"mapping_mrc_indices_sha256": mapping_sha256' in source
    assert '"leaf_cached_after": [False]' in source
    assert 'assert events == [], events' in source
    assert '"schema": "recovar.vdam_raw_cache_admission.v2"' in source


def test_runner_keeps_the_qualified_atomic_multistream_profile_fixed() -> None:
    source = _source()

    for selector in (
        "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS=8",
        "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL=0",
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION=1",
    ):
        assert selector in source
    assert "-u RECOVAR_RELION_COARSE_CANONICAL_REDUCTION" in source
    for argument in (
        "--checkpoint-iteration \"${CHECKPOINT_ITERATION}\"",
        "--nr-iter \"${NR_ITER_SCHEDULE}\"",
        "--random-seed 29",
        "--image-batch-size 500",
        "--exact-local-bucket-radix 4",
        "--exact-local-physical-order-chunk-size 0",
        "--cuda-profiler-range",
    ):
        assert argument in source
    assert 'test -s "${run_root}/profile/warm/run_it181_recovar_meta.json"' in source
    assert 'test ! -e "${run_root}/profile/warm/run_it182_recovar_meta.json"' in source


def test_runner_uses_shared_profile_nsys_summarizer_and_no_broad_tests() -> None:
    source = _source()

    assert "-m scripts.run_vdam_late_iteration_profile" in source
    assert "-m scripts.summarize_vdam_nsys_sqlite" in source
    assert "pytest" not in source
    assert "--trace=cuda,nvtx,osrt" in source
    assert "--sample=none --cpuctxsw=none" in source
    assert "--capture-range=cudaProfilerApi" in source
    assert "--capture-range-end=stop" in source
    assert '"${NSYS}" export --type=sqlite' in source
    assert '/usr/bin/time -v -o "${run_root}/process.time"' in source
    assert 'jax_cache_files.txt' in source
    assert 'jax_cache_file_count.txt' in source


def test_runner_invokes_future_analyzer_then_seals_all_artifacts() -> None:
    source = _source()

    assert 'touch "${ROOT}/ARMS_COMPLETED"' in source
    assert '"${PIXI_PY}" -m scripts.probe_vdam_raw_cache_memory' in source
    assert '--output-json "${PROVENANCE}/raw_cache_memory_probe.json"' in source
    assert '--comparison-batch-size 500' in source
    assert 'RECOVAR_CACHE_DIR=' in source
    assert 'test -s "${PROVENANCE}/raw_cache_memory_probe.json"' in source
    assert '"schema": "recovar.vdam_raw_cache_abba.v1"' in source
    assert '"cache_auto_3", "cache_off_3", "cache_off_4", "cache_auto_4"' in source
    assert '"raw_image_cache_modes": ["off", "auto", "auto", "off", "auto", "off", "off", "auto"]' in source
    assert '"science_promotion_allowed": False' in source
    assert "env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu" in source
    assert '"${PIXI_PY}" -m scripts.analyze_vdam_raw_cache_abba' in source
    assert '--output-json "${ANALYSIS}/report.json"' in source
    assert '--output-markdown "${ANALYSIS}/report.md"' in source
    assert source.count("trap - ERR") >= 2
    assert "trap on_error ERR" in source
    assert 'if [[ "${analyzer_status}" != 0 && "${analyzer_status}" != 1 ]]' in source
    assert 'printf \'%s\\n\' "${analyzer_message}" > "${PROVENANCE}/failure.txt"' in source
    assert 'touch "${ROOT}/COMPLETED"' in source
    assert ') > "${ROOT}/SHA256SUMS"' in source
    assert 'sha256sum "${ROOT}/SHA256SUMS" > "${ROOT}/SHA256SUMS.sha256"' in source
    assert 'find "${ROOT}" -type f -exec chmod a-w {} +' in source
    assert 'find "${ROOT}" -depth -type d -exec chmod a-w {} +' in source
    assert 'if [[ "${analyzer_status}" == 1 ]]' in source
    arms_completed = source.index('touch "${ROOT}/ARMS_COMPLETED"')
    memory_probe = source.index("memory_probe_command=(")
    run_json = source.index('"${PIXI_PY}" - "${PROVENANCE}/run.json"')
    analyzer_call = source.index('"${PIXI_PY}" -m scripts.analyze_vdam_raw_cache_abba')
    completed = source.index('touch "${ROOT}/COMPLETED"')
    no_go_exit = source.index('if [[ "${analyzer_status}" == 1 ]]')
    assert arms_completed < memory_probe < run_json < analyzer_call < completed < no_go_exit
