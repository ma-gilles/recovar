from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "run_vdam_coarse_prehalf_h100_gate.sbatch"


def _source() -> str:
    return RUNNER.read_text()


def test_runner_pins_clean_committed_primitive_and_exact_h100() -> None:
    source = _source()

    assert ': "${EXPECTED_REPO_HEAD:?pin the committed runner head}"' in source
    assert ': "${TARGET_GPU_UUID:?pin the physical H100 UUID}"' in source
    assert "PRIMITIVE_HEAD=3c1ab9990b33496697a894e617f81fc0b0a2d0d9" in source
    assert "PRIMITIVE_TREE=95c4a7cd4c40878beb87380e43d7e310748586fe" in source
    assert "status --porcelain=v1 --untracked-files=all" in source
    assert 'merge-base --is-ancestor "${PRIMITIVE_HEAD}" "${EXPECTED_REPO_HEAD}"' in source
    assert 'diff --quiet "${PRIMITIVE_HEAD}" -- "${PRIMITIVE_FILES[@]}"' in source
    assert 'vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}"' in source
    assert 'vdam_select_target_gpu "${TARGET_GPU_UUID}" 0' in source
    assert 'test "${selected_gpu_uuid}" = "${TARGET_GPU_UUID}"' in source
    assert '[[ "${gpu_name}" == *H100* ]]' in source


def test_runner_builds_sm90_outside_repo_and_audits_frozen_binary() -> None:
    source = _source()

    assert "CUDA_TOOLKIT=/usr/local/cuda-12.6" in source
    assert (
        "CUOBJDUMP=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "cuda_binary_tools_12.6_login_20260901/cuobjdump" in source
    )
    assert "CUOBJDUMP_SHA256=f0299c666460f826a01444f29f3c5ff889765199fc68069e8756c69d4deb661b" in source
    assert 'sha256sum "${CUOBJDUMP}"' in source
    assert '"${PROVENANCE}/cuobjdump.sha256"' in source
    assert "CUDA_ARCH_FLAGS='-gencode arch=compute_90,code=sm_90'" in source
    assert "CANDIDATE_BINARY=${BUILD}/libcuda_backproject.so" in source
    assert "result root must be outside the repository" in source
    assert '"NVCC=${NVCC}"' in source
    assert '"CUDA_ARCH=${CUDA_ARCH_FLAGS}"' in source
    assert '"LIB=${CANDIDATE_BINARY}"' in source
    assert "BASELINE_SHA256=fcd9f03383ec42483016cf2b686adc998e61f1cc6fdf338b3a6713bbc3554d56" in source
    assert "scripts/audit_vdam_coarse_prehalf_binary.py" in source
    assert '--baseline-binary "${BASELINE_BINARY}"' in source
    assert '--candidate-binary "${CANDIDATE_BINARY}"' in source
    assert '--cuobjdump "${CUOBJDUMP}"' in source
    assert "baseline_stat_before=$(stat" in source
    assert 'stat -c \'%d:%i:%s:%Y:%Z\' "${BASELINE_BINARY}")" = "${baseline_stat_before}"' in source


def test_runner_uses_only_focused_gpu_test_then_exact_default_benchmark() -> None:
    source = _source()
    test_node = (
        "tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_vdam_prehalf_source_order_across_dispatchers"
    )

    assert f"FOCUSED_GPU_TEST={test_node}" in source
    assert '"${PIXI_PY}" -m pytest -vv --run-gpu' in source
    assert "assert len(cases) == 1" in source
    assert source.count('"${PIXI_PY}" scripts/benchmark_vdam_coarse_prehalf.py') == 2
    for override in (
        "--current-size",
        "--model-max-r",
        "--rotation-count",
        "--translation-count",
        "--physical-batch-size",
        "--actual-batch-size",
        "--repetitions",
        "--top-support-size",
        "--seed",
    ):
        assert override not in source
    assert 'configuration["profile"] == "gf46_it181_geometry"' in source
    assert 'report["protocol"]["alternation"] == [False, True]' in source
    assert 'set(report["apis"]) == {"serial", "multistream"}' in source


def test_runner_rechecks_inputs_and_seals_content_hashes() -> None:
    source = _source()

    assert 'cmp "${PROVENANCE}/source_manifest.sha256"' in source
    assert source.count('sha256sum "${BASELINE_BINARY}"') >= 3
    assert source.count('sha256sum "${CANDIDATE_BINARY}"') >= 3
    assert "after_build %s\\n" in source
    assert "after_focused_gpu_test %s\\n" in source
    assert "after_benchmark %s\\n" in source
    assert '"${PROVENANCE}/execution_environment.txt"' in source
    assert 'touch "${RESULT_ROOT}/COMPLETED"' in source
    assert ') > "${RESULT_ROOT}/SHA256SUMS"' in source
    assert 'sha256sum "${RESULT_ROOT}/SHA256SUMS"' in source
    assert 'find "${RESULT_ROOT}" -type f -exec chmod a-w {} +' in source
    assert 'find "${RESULT_ROOT}" -depth -type d -exec chmod a-w {} +' in source
