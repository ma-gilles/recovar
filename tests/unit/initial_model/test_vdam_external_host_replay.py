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
