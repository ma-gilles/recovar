from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "qualify_vdam_gf46_certificate_state_h100.py"
RUNNER = ROOT / "scripts" / "run_vdam_gf46_certificate_state_h100.sbatch"
SPEC = importlib.util.spec_from_file_location("qualify_vdam_gf46_certificate_state_h100", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_qualification_geometry_and_operands_are_exact_gf46() -> None:
    MODULE._assert_exact_geometry()
    shape_arguments, coefficients = MODULE._shape_arguments()
    manifest = MODULE._operand_generator_manifest(coefficients)
    contract = MODULE._argument_contract(shape_arguments, manifest)
    buffers = {entry["name"]: entry for entry in manifest["buffers"]}

    assert manifest["geometry"] == {
        "batch_size": 500,
        "translation_count": 29,
        "pixel_count": 5100,
        "rotation_block": 4608,
        "total_rotations": 36864,
        "source_rotation_block": 16,
    }
    assert buffers["projected_reference"]["shape"] == [4608, 5100]
    assert buffers["image_batch.weighted_shifted"]["shape"] == [14500, 5100]
    assert buffers["state.rotation_visit_count"]["shape"] == [36864]
    assert buffers["state.raw_block_lower_max"]["shape"] == [500, 2304]
    assert all(entry["finite_by_construction"] for entry in manifest["buffers"])
    assert contract["logical_dynamic_argument_bytes"] == 1_428_827_344


@pytest.mark.unit
def test_hlo_evidence_extracts_both_nested_gemm_backend_configs() -> None:
    stablehlo = "\n".join(["%0 = stablehlo.dot_general %a, %b", "%1 = stablehlo.dot_general %c, %d"])
    pre_hlo = "\n".join(["dot.1 = f64[] dot(a, b)", "dot.2 = c128[] dot(c, d)"])
    optimized = "\n".join(
        [
            'ROOT %gemm.1 = f64[] custom-call(%a, %b), custom_call_target="__cublas$gemm", '
            'backend_config={"operation_queue_id":"0","gemm_backend_config":{"selected_algorithm":-1,'
            '"precision_config":{"operand_precision":["HIGHEST","HIGHEST"]}}}',
            '%gemm.2 = c128[] custom-call(%c, %d), custom_call_target="__cublas$gemm", '
            'backend_config={"gemm_backend_config":{"algorithm":"ALG_UNSET","epilogue":"DEFAULT"}}',
        ]
    )

    evidence = MODULE._hlo_evidence(stablehlo, pre_hlo, optimized)

    assert evidence["pass"] is True
    assert evidence["stablehlo_dot_general_count"] == 2
    assert evidence["pre_optimized_hlo_dot_count"] == 2
    assert evidence["optimized_cublas_gemm_like_count"] == 2
    first = evidence["optimized_cublas_gemm_like_calls"][0]
    assert first["backend_config"]["gemm_backend_config"]["selected_algorithm"] == -1
    assert first["selected_backend_fields"] == {
        "gemm_backend_config.precision_config": {"operand_precision": ["HIGHEST", "HIGHEST"]},
        "gemm_backend_config.precision_config.operand_precision": ["HIGHEST", "HIGHEST"],
        "gemm_backend_config.selected_algorithm": -1,
    }


@pytest.mark.unit
def test_memory_analysis_serializes_every_required_field_and_buffer_assignment(tmp_path: Path) -> None:
    class FakeStats:
        serialized_buffer_assignment_proto = b"buffer-assignment"

    stats = FakeStats()
    for index, field in enumerate(MODULE.REQUIRED_MEMORY_FIELDS):
        setattr(stats, field, index + 1)
    proto = tmp_path / "assignment.pb"

    payload = MODULE._memory_analysis_payload(stats, proto)

    assert payload["scalar_fields"] == {field: index + 1 for index, field in enumerate(MODULE.REQUIRED_MEMORY_FIELDS)}
    assert proto.read_bytes() == b"buffer-assignment"
    assert payload["serialized_buffer_assignment"]["size_bytes"] == len(b"buffer-assignment")


@pytest.mark.unit
def test_runner_seals_commit_tree_source_gpu_x64_and_exact_execution_protocol() -> None:
    source = RUNNER.read_text()

    for required in (
        '"${EXPECTED_REPO_HEAD:?pin the committed qualification head}"',
        '"${EXPECTED_REPO_TREE:?pin the committed qualification tree}"',
        '"${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the selected source-manifest SHA-256}"',
        '"${TARGET_GPU_UUID:?pin the physical H100 UUID}"',
        "status --porcelain=v1 --untracked-files=all",
        'vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}"',
        'vdam_select_target_gpu "${TARGET_GPU_UUID}" 0',
        '[[ "${gpu_name}" == *H100* ]]',
        "export JAX_ENABLE_X64=1",
        "export JAX_PLATFORMS=cuda",
        "--warmup-runs 2",
        "--timed-runs 5",
    ):
        assert required in source
    assert "--batch-size" not in source
    assert "--translation-count" not in source
    assert "--pixel-count" not in source
    assert "--rotation-block" not in source
    assert "--total-rotations" not in source
    assert "--analysis-only" not in source
    assert "sbatch " not in source


@pytest.mark.unit
def test_harness_persists_mandatory_analysis_before_unweakened_execution() -> None:
    source = SCRIPT.read_text()

    memory_write = source.index('"xla_memory_analysis.json"')
    partial_write = source.index('"qualification.partial.json"')
    execution_stage = source.index('stage = "full_geometry_execution"')
    execution_call = source.index("execution = _execution(")
    assert memory_write < partial_write < execution_stage < execution_call
    for artifact in (
        "stablehlo_pre_optimization.mlir",
        "hlo_pre_optimization.txt",
        "hlo_optimized_compiled.txt",
        "hlo_optimized_runtime.txt",
        "xla_buffer_assignment.pb",
        "hlo_gemm_evidence.json",
    ):
        assert artifact in source
    assert '"geometry_was_not_reduced": True' in source
    assert '"full_geometry_execution_failed_after_mandatory_analysis"' in source
    for option in (
        "--batch-size",
        "--translation-count",
        "--pixel-count",
        "--rotation-block",
        "--total-rotations",
        "--analysis-only",
    ):
        assert f'parser.add_argument("{option}"' not in source


@pytest.mark.unit
def test_source_manifest_bytes_use_sha256sum_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = ("one.txt", "nested/two.txt")
    (tmp_path / "nested").mkdir()
    (tmp_path / "one.txt").write_text("one\n")
    (tmp_path / "nested" / "two.txt").write_text("two\n")
    monkeypatch.setattr(MODULE, "SOURCE_FILES", files)

    encoded, entries = MODULE._source_manifest(tmp_path)

    assert encoded.decode().splitlines() == [
        f"{entries[0]['sha256']}  one.txt",
        f"{entries[1]['sha256']}  nested/two.txt",
    ]
    assert json.loads(json.dumps(entries)) == entries
