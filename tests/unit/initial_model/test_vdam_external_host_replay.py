from pathlib import Path

import pytest

from scripts import run_vdam_exact_native_host_replay

ROOT = Path(__file__).resolve().parents[3]
CUDA_SOURCE = ROOT / "recovar" / "cuda" / "cuda_backproject.cu"
PYTHON_WRAPPER = ROOT / "recovar" / "cuda_backproject.py"
LOCAL_ENGINE = ROOT / 'recovar' / 'em' / 'local' / 'local_em_engine.py'
REPLAY_HELPER = ROOT / 'recovar' / 'em' / 'diagnostics' / 'vdam_replay.py'
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


def test_external_host_replay_can_preserve_input_bundles_fail_closed():
    wrapper = PYTHON_WRAPPER.read_text()

    assert "RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_CAPTURE_DIR" in wrapper
    assert 'f"pid-{os.getpid()}-call-{call:04d}-input.npz"' in wrapper
    assert "refusing to overwrite VDAM host-replay capture" in wrapper
    assert "shutil.copy2(input_path, capture_path)" in wrapper


def test_external_host_replay_can_capture_quiesced_prelaunch_state():
    cuda_source = CUDA_SOURCE.read_text()
    helper = HELPER.read_text()

    assert "quiesced_prelaunch_target_particle_id" in cuda_source
    assert "cudaDeviceSynchronize()" in cuda_source
    assert "std::shared_mutex quiesced_prelaunch_launch_gate" in cuda_source
    assert "std::unique_lock<std::shared_mutex>" in cuda_source
    assert "std::shared_lock<std::shared_mutex>" in cuda_source
    assert "RECOVAR_VDAM_QUIESCED_PRELAUNCH_CAPTURE_DIR" in helper
    assert "RECOVAR_VDAM_QUIESCED_PRELAUNCH_PARTICLE_ID" in helper
    assert "recovar.vdam_quiesced_prelaunch.v1" in helper
    assert "refusing to overwrite" in helper


def test_quiesced_prelaunch_capture_requires_directory_and_particle_id(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "input.npz"
    library_path = tmp_path / "libcuda_backproject.so"
    input_path.touch()
    library_path.touch()
    monkeypatch.setenv("RECOVAR_VDAM_EXACT_NATIVE_PTX", str(tmp_path / "exact.ptx"))
    monkeypatch.setenv(
        "RECOVAR_VDAM_QUIESCED_PRELAUNCH_CAPTURE_DIR",
        str(tmp_path / "capture"),
    )

    with pytest.raises(RuntimeError, match="requires both directory and particle ID"):
        run_vdam_exact_native_host_replay.run_replay(
            input_path,
            tmp_path / "output.npz",
            library_path,
        )


def test_quiesced_prelaunch_capture_rejects_negative_particle_id(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "input.npz"
    library_path = tmp_path / "libcuda_backproject.so"
    input_path.touch()
    library_path.touch()
    monkeypatch.setenv("RECOVAR_VDAM_EXACT_NATIVE_PTX", str(tmp_path / "exact.ptx"))
    monkeypatch.setenv(
        "RECOVAR_VDAM_QUIESCED_PRELAUNCH_CAPTURE_DIR",
        str(tmp_path / "capture"),
    )
    monkeypatch.setenv("RECOVAR_VDAM_QUIESCED_PRELAUNCH_PARTICLE_ID", "-1")

    with pytest.raises(ValueError, match="must be nonnegative"):
        run_vdam_exact_native_host_replay.run_replay(
            input_path,
            tmp_path / "output.npz",
            library_path,
        )


def test_exact_wavg_predecessor_uses_relion_ptx_and_same_particle_stream():
    cuda_source = CUDA_SOURCE.read_text()

    assert "RECOVAR_VDAM_EXACT_WAVG_PREDECESSOR" in cuda_source
    assert "_Z16cuda_kernel_wavgILb1ELb1ELb0ELi256E" in cuda_source
    assert "cuModuleGetFunction(wavg)" in cuda_source
    assert "cuLaunchKernel(wavg)" in cuda_source
    assert "reinterpret_cast<CUstream>(particle_streams[lane])" in cuda_source
    assert "kWavgSharedBytes" in cuda_source


def test_runtime_bpref_launch_discriminator_is_fail_closed():
    cuda_source = CUDA_SOURCE.read_text()

    assert "RECOVAR_VDAM_RUNTIME_BPREF_WITH_EXACT_WAVG" in cuda_source
    assert "runtime_bpref_with_exact_wavg_requested &&" in cuda_source
    assert "!exact_wavg_predecessor_requested" in cuda_source
    assert "trace_runtime_gap" in cuda_source
    assert "runtime_bpref_enqueue_start" in cuda_source


def test_wavg_bpref_host_gap_is_fail_closed_and_measures_from_wavg_return():
    cuda_source = CUDA_SOURCE.read_text()

    assert "RECOVAR_VDAM_WAVG_BPREF_HOST_GAP_NS" in cuda_source
    assert "wavg_bpref_host_gap_requested && !exact_wavg_predecessor_requested" in cuda_source
    assert "exact_wavg_return_time = std::chrono::steady_clock::now()" in cuda_source
    assert "exact_wavg_return_time +" in cuda_source
    assert "std::this_thread::sleep_until" in cuda_source


def test_wavg_bpref_host_gap_trace_is_targeted_and_fail_closed():
    cuda_source = CUDA_SOURCE.read_text()
    local_engine = LOCAL_ENGINE.read_text()
    replay_helper = REPLAY_HELPER.read_text()

    assert "RECOVAR_VDAM_WAVG_BPREF_HOST_GAP_TRACE" in cuda_source
    assert "RECOVAR_VDAM_WAVG_BPREF_HOST_GAP_TRACE_PARTICLE_ID" in cuda_source
    assert "wavg_bpref_intrinsic_gap_ns" in cuda_source
    assert "wavg_bpref_effective_gap_ns" in cuda_source
    assert "wavg_host_enqueue_ns" in cuda_source
    assert "bpref_host_enqueue_ns" in cuda_source
    assert "wavg_to_bpref_return_ns" in cuda_source
    assert 'trace << "particle\\ttrace_particle_id\\tworker_lane' in cuda_source
    assert "callbacks do not contain the requested global particle ID" in cuda_source
    assert "std::ios::app" in cuda_source
    assert "if (trace.tellp() == 0)" in cuda_source
    assert 'VDAM_WAVG_BPREF_HOST_GAP_TRACE_ENV = "RECOVAR_VDAM_WAVG_BPREF_HOST_GAP_TRACE"' in replay_helper
    for diagnostic_gate in (
        "block_trace_active",
        "host_gap_trace_active",
        "host_replay_capture_active",
        "quiesced_prelaunch_capture_active",
    ):
        assert diagnostic_gate in replay_helper
    assert "candidate_trace_active = vdam_replay._relion_vdam_candidate_trace_active(" in local_engine
    assert "candidate_trace_active=candidate_trace_active," in local_engine
