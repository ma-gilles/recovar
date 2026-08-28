from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CUDA_SOURCE = ROOT / "recovar" / "cuda" / "cuda_backproject.cu"
PYTHON_WRAPPER = ROOT / "recovar" / "cuda_backproject.py"
HELPER = ROOT / "scripts" / "run_vdam_exact_native_host_replay.py"


def test_exact_native_host_replay_has_one_shared_abi():
    cuda_source = CUDA_SOURCE.read_text()
    helper_source = HELPER.read_text()

    symbol = "recovar_relion_vdam_exact_native_host_replay"
    assert f'extern "C" int {symbol}' in cuda_source
    assert symbol in helper_source
    for field in (
        "projector_full",
        "posterior_over_weight_norm",
        "worker_lane_ids",
        "rotation_replay_order",
        "data_real_volume",
        "denominator_sum",
        "reconstruction_group_count",
        "parallel_worker_replay",
    ):
        assert field in cuda_source
        assert f'("{field}",' in helper_source


def test_external_host_replay_is_fresh_process_and_ordered_callback():
    wrapper = PYTHON_WRAPPER.read_text()

    assert "run_vdam_exact_native_host_replay.py" in wrapper
    assert "subprocess.run(" in wrapper
    assert "jax.experimental.io_callback(" in wrapper
    assert "ordered=True" in wrapper
    assert "external exact-native host replay cannot mix" in wrapper


def test_external_host_replay_requires_exact_ptx_and_protects_outputs():
    helper = HELPER.read_text()

    assert 'os.environ.get("RECOVAR_VDAM_EXACT_NATIVE_PTX"' in helper
    assert "refusing to overwrite" in helper
    assert "allow_pickle=False" in helper
    assert "clean-process CUDA replay failed" in helper


def test_external_host_replay_report_names_include_parent_pid():
    wrapper = PYTHON_WRAPPER.read_text()

    assert 'f"pid-{os.getpid()}-call-{call:04d}.json"' in wrapper


def test_exact_wavg_predecessor_uses_relion_ptx_and_same_particle_stream():
    cuda_source = CUDA_SOURCE.read_text()

    assert "RECOVAR_VDAM_EXACT_WAVG_PREDECESSOR" in cuda_source
    assert "_Z16cuda_kernel_wavgILb1ELb1ELb0ELi256E" in cuda_source
    assert "cuModuleGetFunction(wavg)" in cuda_source
    assert "cuLaunchKernel(wavg)" in cuda_source
    assert "reinterpret_cast<CUstream>(particle_streams[lane])" in cuda_source
    assert "kWavgSharedBytes" in cuda_source


def test_wavg_bpref_host_gap_is_fail_closed_and_measures_from_wavg_return():
    cuda_source = CUDA_SOURCE.read_text()

    assert "RECOVAR_VDAM_WAVG_BPREF_HOST_GAP_NS" in cuda_source
    assert "wavg_bpref_host_gap_requested && !exact_wavg_predecessor_requested" in cuda_source
    assert "exact_wavg_return_time = std::chrono::steady_clock::now()" in cuda_source
    assert "exact_wavg_return_time +" in cuda_source
    assert "std::this_thread::sleep_until" in cuda_source


def test_wavg_bpref_host_gap_trace_is_targeted_and_fail_closed():
    cuda_source = CUDA_SOURCE.read_text()

    assert "RECOVAR_VDAM_WAVG_BPREF_HOST_GAP_TRACE" in cuda_source
    assert "RECOVAR_VDAM_WAVG_BPREF_HOST_GAP_TRACE_PARTICLE_ID" in cuda_source
    assert "wavg_bpref_intrinsic_gap_ns" in cuda_source
    assert "wavg_bpref_effective_gap_ns" in cuda_source
    assert 'trace << "particle\\ttrace_particle_id\\tworker_lane' in cuda_source
    assert "callbacks do not contain the requested global particle ID" in cuda_source
    assert "std::ios::app" in cuda_source
    assert "if (trace.tellp() == 0)" in cuda_source
