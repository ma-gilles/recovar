from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts" / "cuda_runtime_context_probe.cu"
SBATCH = ROOT / "scripts" / "run_vdam_cuda_context_probe.sbatch"


def test_cuda_runtime_context_probe_is_passive_and_reports_primary_state():
    source = SOURCE.read_text()

    assert "cudaGetDeviceFlags" in source
    assert "cuDevicePrimaryCtxGetState" in source
    assert "cuCtxGetCurrent" in source
    assert "cudaStreamGetFlags" in source
    assert "cudaStreamGetPriority" in source
    assert "cudaSetDeviceFlags" not in source
    assert "cudaDeviceReset" not in source
    assert "<<<" not in source


def test_cuda_runtime_context_probe_supports_shared_and_executable_builds():
    source = SOURCE.read_text()

    assert 'extern "C" const char* recovar_cuda_runtime_context_probe_json()' in source
    assert "#ifdef RECOVAR_CUDA_CONTEXT_PROBE_MAIN" in source


def test_cuda_runtime_context_probe_job_is_fail_closed_and_compares_jax():
    source = SBATCH.read_text()

    assert "EXPECTED_REPO_HEAD" in source
    assert "EXPECTED_PROBE_EXECUTABLE_SHA256" in source
    assert "EXPECTED_PROBE_LIBRARY_SHA256" in source
    assert "TARGET_GPU_UUID" in source
    assert 'jax.devices("gpu")' in source
    assert "all_observable_context_fields_match" in source
