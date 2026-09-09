from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import benchmark_vdam_coarse_prehalf as benchmark


def _options(**overrides: int) -> benchmark.BenchmarkOptions:
    base = benchmark.BenchmarkOptions(
        current_size=4,
        model_max_r=1,
        rotation_count=3,
        translation_count=2,
        physical_batch_size=2,
        actual_batch_size=2,
        repetitions=3,
        top_support_size=2,
        seed=46,
    )
    return replace(base, **overrides)


def _provenance() -> dict[str, object]:
    return {
        "repository": {
            "path": "/repo",
            "head": "a" * 40,
            "tree": "b" * 40,
            "status_porcelain_v1": ["?? diagnostic.py"],
            "tracked_diff_sha256": "c" * 64,
        },
        "cuda": {
            "path": "/repo/libcuda_backproject.so",
            "loaded_path": "/repo/libcuda_backproject.so",
            "sha256": "d" * 64,
            "size_bytes": 123,
        },
        "device": {
            "backend": "gpu",
            "device": "cuda:0",
            "device_kind": "NVIDIA H100 80GB HBM3",
        },
        "runtime": {"python_executable": "/env/bin/python"},
        "script": {"path": "/repo/diagnostic.py", "sha256": "e" * 64},
    }


class _FakeOutput:
    def __init__(self, label: str, shape: tuple[int, ...], sync_log: list[str]):
        self.label = label
        self.shape = shape
        self.sync_log = sync_log
        self.synchronized = False

    def block_until_ready(self) -> _FakeOutput:
        self.synchronized = True
        self.sync_log.append(self.label)
        return self


def _comparison(left: _FakeOutput, right: _FakeOutput, **kwargs: int) -> dict[str, object]:
    assert left.synchronized and right.synchronized
    return {
        "active_batch_size": kwargs["active_batch_size"],
        "relative_l2_error": 0.0,
        "max_abs_error": 0.0,
        "max_relative_error": 0.0,
        "value_exact_equal": True,
        "argmin_equal": True,
        "fixed_top_support_size": kwargs["top_support_size"],
        "fixed_top_support_equal": True,
    }


def test_panel_warms_both_modes_then_alternates_and_synchronizes() -> None:
    options = _options()
    shape = benchmark._expected_output_shape(options)
    calls: list[tuple[str, bool]] = []
    sync_log: list[str] = []

    def invoker(api: str):
        def invoke(prehalf_weight: bool) -> _FakeOutput:
            calls.append((api, prehalf_weight))
            return _FakeOutput(f"{api}:{len(calls)}", shape, sync_log)

        return invoke

    ticks = iter(range(0, 100_000_000, 1_000_000))
    report = benchmark._run_panel(
        options,
        {name: invoker(name) for name in benchmark.API_NAMES},
        _provenance(),
        compare=_comparison,
        clock_ns=lambda: next(ticks),
        generated_at="2026-09-01T00:00:00+00:00",
    )
    expected_flags = [False, True] + [False, True] * options.repetitions
    for api in benchmark.API_NAMES:
        assert [flag for name, flag in calls if name == api] == expected_flags
        api_report = report["apis"][api]
        assert api_report["warmup_sequence"] == [False, True]
        assert api_report["timed_sequence"] == [False, True] * options.repetitions
        assert api_report["timing"]["prehalf_off"]["raw_seconds"] == [0.001] * 3
        assert api_report["timing"]["prehalf_on"]["raw_seconds"] == [0.001] * 3
        assert len(api_report["numerical"]["repeat_stability"]["prehalf_off"]) == 3
        assert len(api_report["numerical"]["repeat_stability"]["prehalf_on"]) == 3
    assert len(sync_log) == len(calls) == len(benchmark.API_NAMES) * len(expected_flags)


def test_report_schema_provenance_and_diagnostic_only_contract() -> None:
    options = _options(repetitions=2)
    shape = benchmark._expected_output_shape(options)
    ticks = iter(range(0, 100_000_000, 1_000_000))

    def invoke(prehalf_weight: bool) -> _FakeOutput:
        return _FakeOutput(str(prehalf_weight), shape, [])

    provenance = _provenance()
    report = benchmark._run_panel(
        options,
        {name: invoke for name in benchmark.API_NAMES},
        provenance,
        compare=_comparison,
        clock_ns=lambda: next(ticks),
        generated_at="2026-09-01T00:00:00+00:00",
    )
    assert report["schema"] == benchmark.SCHEMA
    assert report["classification"] == "diagnostic_only_no_decision"
    assert report["provenance"] == provenance
    assert report["configuration"]["prehalf_default"] is False
    assert report["configuration"]["canonical_reduction"] is False
    assert report["protocol"]["only_static_kernel_delta"] == "prehalf_weight=False/True"
    json.dumps(report, allow_nan=False)

    forbidden = {"pass", "passed", "accept", "accepted", "acceptance", "threshold", "tolerance"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            result = set(value)
            for item in value.values():
                result.update(keys(item))
            return result
        if isinstance(value, list):
            result: set[str] = set()
            for item in value:
                result.update(keys(item))
            return result
        return set()

    assert not (keys(report) & forbidden)


def test_default_geometry_matches_gf46_it181_output_copy() -> None:
    options = benchmark.BenchmarkOptions()
    assert benchmark._expected_output_shape(options) == (187, 36_864, 29)
    assert int(np.prod(benchmark._expected_output_shape(options), dtype=np.int64)) * 4 == 799_653_888


@pytest.mark.parametrize(
    "options",
    (
        _options(repetitions=1),
        _options(repetitions=21),
        _options(current_size=129),
        _options(model_max_r=3),
        _options(rotation_count=0),
        _options(rotation_count=36_865),
        _options(translation_count=0),
        _options(translation_count=30),
        _options(physical_batch_size=188),
        _options(actual_batch_size=3),
        _options(top_support_size=7),
        _options(seed=-1),
    ),
)
def test_malformed_or_unbounded_options_fail_closed(options: benchmark.BenchmarkOptions) -> None:
    with pytest.raises(benchmark.BenchmarkError):
        benchmark._validate_options(options)


def test_boolean_option_is_not_treated_as_an_integer() -> None:
    with pytest.raises(benchmark.BenchmarkError, match="repetitions must be an integer"):
        benchmark._validate_options(_options(repetitions=True))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("images", np.zeros((1, 12), dtype=np.complex64)),
        ("weight", np.zeros((2, 12), dtype=np.float64)),
        ("rotation_matrices", np.zeros((3, 9), dtype=np.float32)),
        ("full_to_compact", np.zeros((12,), dtype=np.int64)),
    ),
)
def test_workload_shape_and_dtype_contract_fails_closed(field: str, value: np.ndarray) -> None:
    options = _options()
    workload = benchmark._generate_host_workload(options)
    malformed = replace(workload, **{field: value})
    with pytest.raises(benchmark.BenchmarkError, match=field):
        benchmark._validate_workload(malformed, options)


def test_projector_invokers_pin_native_atomic_and_forward_static_flag() -> None:
    options = _options()
    workload = benchmark._generate_host_workload(options)
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class FakeCuda:
        @staticmethod
        def relion_coarse_diff2_projector_f32(*args: object, **kwargs: object) -> str:
            calls.append(("serial", args, kwargs))
            return "serial-output"

        @staticmethod
        def relion_coarse_diff2_projector_multistream_f32(*args: object, **kwargs: object) -> str:
            calls.append(("multistream", args, kwargs))
            return "multistream-output"

    actual = np.asarray(options.actual_batch_size, dtype=np.int32)
    invokers = benchmark._projector_invokers(FakeCuda, workload, options, actual)
    assert invokers["serial"](False) == "serial-output"
    assert invokers["serial"](True) == "serial-output"
    assert invokers["multistream"](False) == "multistream-output"
    assert invokers["multistream"](True) == "multistream-output"
    assert [row[0] for row in calls] == ["serial", "serial", "multistream", "multistream"]
    assert [row[2]["prehalf_weight"] for row in calls] == [False, True, False, True]
    for name, _, kwargs in calls:
        assert kwargs["canonical_reduction"] is False
        assert kwargs["single_lane_canonical"] is False
        assert kwargs["current_size"] == options.current_size
        if name == "multistream":
            assert kwargs["actual_batch_size"] is actual
        else:
            assert "actual_batch_size" not in kwargs


def test_wrong_projector_output_shape_fails_closed_after_synchronization() -> None:
    options = _options(repetitions=2)
    sync_log: list[str] = []

    def invoke(prehalf_weight: bool) -> _FakeOutput:
        return _FakeOutput(str(prehalf_weight), (1, 3, 2), sync_log)

    with pytest.raises(benchmark.BenchmarkError, match="output shape differs"):
        benchmark._benchmark_api("serial", invoke, options, compare=_comparison)
    assert sync_log == ["False"]


class _FakeDevice:
    device_kind = "H100"
    id = 0
    process_index = 0
    local_hardware_id = 0
    platform = "gpu"
    client = SimpleNamespace(platform_version="CUDA 12")

    def __str__(self) -> str:
        return "cuda:0"


class _FakeJax:
    __version__ = "0.test"

    def __init__(self, backend: str = "gpu", device_count: int = 1):
        self.backend = backend
        self.device_count = device_count

    def default_backend(self) -> str:
        return self.backend

    def devices(self, backend: str) -> list[_FakeDevice]:
        assert backend == "gpu"
        return [_FakeDevice() for _ in range(self.device_count)]


class _FakeCuda:
    def __init__(self, requested: bool = True):
        self.requested = requested

    def custom_cuda_requested(self) -> bool:
        return self.requested


def test_runtime_contract_resolves_one_gpu_and_pinned_cuda(tmp_path: Path) -> None:
    cuda = tmp_path / "libcuda_backproject.so"
    cuda.write_bytes(b"cuda")
    device, resolved = benchmark._runtime_contract(
        _FakeJax(),
        _FakeCuda(),
        {"RECOVAR_CUDA_LIB": str(cuda)},
    )
    assert isinstance(device, _FakeDevice)
    assert resolved == cuda.resolve()


@pytest.mark.parametrize(
    ("jax_module", "cuda_module", "environment", "message"),
    (
        (_FakeJax(backend="cpu"), _FakeCuda(), {"RECOVAR_CUDA_LIB": "/missing"}, "GPU backend"),
        (_FakeJax(), _FakeCuda(False), {"RECOVAR_CUDA_LIB": "/missing"}, "not requested"),
        (
            _FakeJax(),
            _FakeCuda(),
            {"RECOVAR_CUDA_LIB": "/missing", "RECOVAR_DISABLE_CUDA": "1"},
            "disabled",
        ),
        (_FakeJax(device_count=2), _FakeCuda(), {"RECOVAR_CUDA_LIB": "/missing"}, "does not name"),
    ),
)
def test_runtime_contract_fails_closed(
    jax_module: _FakeJax,
    cuda_module: _FakeCuda,
    environment: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(benchmark.BenchmarkError, match=message):
        benchmark._runtime_contract(jax_module, cuda_module, environment)


def test_runtime_contract_rejects_ambiguous_visible_gpu_set(tmp_path: Path) -> None:
    cuda = tmp_path / "libcuda_backproject.so"
    cuda.write_bytes(b"cuda")
    with pytest.raises(benchmark.BenchmarkError, match="exactly one visible GPU"):
        benchmark._runtime_contract(
            _FakeJax(device_count=2),
            _FakeCuda(),
            {"RECOVAR_CUDA_LIB": str(cuda)},
        )


def test_loaded_cuda_must_match_the_pinned_binary(tmp_path: Path) -> None:
    expected = tmp_path / "expected.so"
    other = tmp_path / "other.so"
    expected.write_bytes(b"expected")
    other.write_bytes(b"other")
    benchmark._verify_loaded_cuda(SimpleNamespace(_loaded_lib_path=expected), expected)
    with pytest.raises(benchmark.BenchmarkError, match="loaded CUDA library differs"):
        benchmark._verify_loaded_cuda(SimpleNamespace(_loaded_lib_path=other), expected)


def test_git_provenance_uses_resolved_head_tree_status_and_diff(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []
    diff = "diff --git a/a b/a\n+new line"

    def git_output(repo: Path, *arguments: str) -> str:
        assert repo == tmp_path.resolve()
        calls.append(arguments)
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(tmp_path.resolve())
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return "b" * 40
        if arguments[0] == "status":
            return " M recovar/cuda_backproject.py\n?? scripts/benchmark.py"
        if arguments[0] == "diff":
            return diff
        raise AssertionError(arguments)

    result = benchmark._collect_git_provenance(tmp_path, git_output=git_output)
    assert result["path"] == str(tmp_path.resolve())
    assert result["head"] == "a" * 40
    assert result["tree"] == "b" * 40
    assert result["status_porcelain_v1"] == [
        " M recovar/cuda_backproject.py",
        "?? scripts/benchmark.py",
    ]
    assert result["tracked_diff_sha256"] == hashlib.sha256(diff.encode()).hexdigest()
    assert ("rev-parse", "HEAD^{tree}") in calls


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("repository", "head", "not-a-head", "head is invalid"),
        ("repository", "tree", "bad", "tree is invalid"),
        ("cuda", "sha256", "short", "CUDA SHA256 is invalid"),
        ("device", "backend", "cpu", "not GPU-backed"),
    ),
)
def test_malformed_provenance_fails_closed(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    provenance = _provenance()
    provenance[section] = dict(provenance[section], **{field: value})
    with pytest.raises(benchmark.BenchmarkError, match=message):
        benchmark._validate_provenance(provenance)
