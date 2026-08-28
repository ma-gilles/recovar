from __future__ import annotations

from pathlib import Path

RUNNER = Path("scripts/run_vdam_matched_live_posterior_panel.sbatch")


def _runner_text() -> str:
    return RUNNER.read_text()


def test_runner_is_one_allocation_matched_panel():
    text = _runner_text()
    assert "one allocation, one physical H100, sequential native and live candidate arms" in text
    assert "for repeat in $(seq 1 \"${REPEAT_COUNT}\")" in text
    assert "VDAM_NATIVE_ONLY=1" in text
    assert "candidate-repeat-${repeat_tag}" in text


def test_runner_captures_production_fused_scores_for_native_particle_set():
    text = _runner_text()
    assert "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES=${target_indices}" in text
    assert "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_SCORES=1" in text
    assert 'test "${target_count}" -eq 200' in text
    assert "posterior identity closure failed" in text


def test_runner_stops_after_iteration_one_and_reuses_compile_cache():
    text = _runner_text()
    assert "--diagnostic_stop_after_iteration 1" in text
    assert "shared_jax_cache=${VDAM_PANEL_ROOT}/jax_cache" in text
    assert "JAX_COMPILATION_CACHE_DIR=${shared_jax_cache}" in text


def test_runner_seals_source_cuda_gpu_and_report():
    text = _runner_text()
    assert 'test "$(git rev-parse HEAD)" = "${VDAM_EXPECTED_HEAD}"' in text
    assert "VDAM_EXPECTED_CUDA_SHA256" in text
    assert "VDAM_EXPECTED_RELION_BIND_SHA256" in text
    assert "scripts.audit_vdam_live_posterior_repeat_panel" in text
    assert "matched_live_posterior_evidence.sha256" in text
    assert "MATCHED_LIVE_POSTERIOR_SUCCESS" in text
