from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GPU_HELPER = REPO_ROOT / "scripts/vdam_gpu_selection.sh"


def _mock_nvidia_smi(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    executable = tmp_path / "nvidia-smi"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
mapping = json.loads(os.environ["MOCK_GPU_UUID_MAP"])
if len(args) < 2 or args[0] != "-i":
    raise SystemExit(64)
selector = args[1]
if selector == os.environ.get("MOCK_GPU_QUERY_FAILURE"):
    raise SystemExit(65)
if "--query-gpu=uuid" not in args or "--format=csv,noheader" not in args:
    raise SystemExit(66)
value = mapping.get(selector)
if value is None:
    raise SystemExit(67)
if isinstance(value, list):
    print("\\n".join(value))
else:
    print(value)
"""
    )
    executable.chmod(0o755)
    return executable


def _run_allocation_assertion(
    tmp_path: Path,
    *,
    target: str,
    allocation_spec: str = "",
    mapping: dict[str, str | list[str]],
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    _mock_nvidia_smi(tmp_path)
    env = dict(os.environ)
    for name in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES"):
        env.pop(name, None)
    env.update(environment or {})
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["MOCK_GPU_UUID_MAP"] = json.dumps(mapping)
    command = r"""
set -euo pipefail
source "$1"
vdam_assert_target_gpu_allocated "$2" "$3"
printf 'allocated=%s\n' "${VDAM_ALLOCATED_GPU_UUIDS_CSV}"
printf 'visible=%s\n' "${CUDA_VISIBLE_DEVICES-}"
"""
    return subprocess.run(
        ["bash", "-c", command, "bash", str(GPU_HELPER), target, allocation_spec],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_target_gpu_allocation_resolves_each_selector_and_preserves_initial_visibility(tmp_path):
    result = _run_allocation_assertion(
        tmp_path,
        target="GPU-second",
        allocation_spec=" gpu:3, GPU-second ",
        mapping={"3": "GPU-first", "GPU-second": "GPU-second"},
        environment={"CUDA_VISIBLE_DEVICES": "initial-visible-value"},
    )

    assert result.returncode == 0, result.stderr
    assert "allocated=GPU-first,GPU-second" in result.stdout
    assert "visible=initial-visible-value" in result.stdout


def test_target_gpu_allocation_uses_slurm_step_then_job_then_initial_visible_devices(tmp_path):
    cases = (
        (
            {
                "SLURM_STEP_GPUS": "7",
                "SLURM_JOB_GPUS": "8",
                "CUDA_VISIBLE_DEVICES": "9",
            },
            "GPU-step",
            {"7": "GPU-step", "8": "GPU-job", "9": "GPU-visible"},
        ),
        (
            {"SLURM_JOB_GPUS": "8", "CUDA_VISIBLE_DEVICES": "9"},
            "GPU-job",
            {"8": "GPU-job", "9": "GPU-visible"},
        ),
        (
            {"CUDA_VISIBLE_DEVICES": "9"},
            "GPU-visible",
            {"9": "GPU-visible"},
        ),
    )
    for index, (environment, target, mapping) in enumerate(cases):
        case_root = tmp_path / str(index)
        case_root.mkdir()
        result = _run_allocation_assertion(
            case_root,
            target=target,
            mapping=mapping,
            environment=environment,
        )

        assert result.returncode == 0, result.stderr
        assert f"allocated={target}" in result.stdout


def test_target_gpu_allocation_rejects_target_outside_resolved_allocation(tmp_path):
    result = _run_allocation_assertion(
        tmp_path,
        target="GPU-not-allocated",
        allocation_spec="3,4",
        mapping={"3": "GPU-first", "4": "GPU-second"},
    )

    assert result.returncode == 76
    assert "VDAM_TARGET_GPU_NOT_ALLOCATED" in result.stderr
    assert "resolved=GPU-first,GPU-second" in result.stderr


def test_target_gpu_allocation_rejects_empty_or_unresolvable_allocation(tmp_path):
    empty = _run_allocation_assertion(
        tmp_path / "empty",
        target="GPU-target",
        mapping={},
    )
    assert empty.returncode == 76
    assert "no allocation spec" in empty.stderr

    unresolved_root = tmp_path / "unresolved"
    unresolved_root.mkdir()
    unresolved = _run_allocation_assertion(
        unresolved_root,
        target="GPU-target",
        allocation_spec="3",
        mapping={"3": "GPU-target"},
        environment={"MOCK_GPU_QUERY_FAILURE": "3"},
    )
    assert unresolved.returncode == 76
    assert "cannot resolve allocated GPU selector 3" in unresolved.stderr


def test_target_gpu_allocation_rejects_empty_selectors(tmp_path):
    for index, allocation_spec in enumerate((",3", "3,", "3,,4")):
        result = _run_allocation_assertion(
            tmp_path / str(index),
            target="GPU-target",
            allocation_spec=allocation_spec,
            mapping={"3": "GPU-target", "4": "GPU-other"},
        )

        assert result.returncode == 76
        assert "contains an empty GPU selector" in result.stderr


def test_target_gpu_allocation_rejects_selector_with_multiple_or_malformed_uuids(tmp_path):
    multiple = _run_allocation_assertion(
        tmp_path / "multiple",
        target="GPU-first",
        allocation_spec="3",
        mapping={"3": ["GPU-first", "GPU-second"]},
    )
    assert multiple.returncode == 76
    assert "did not resolve to one UUID" in multiple.stderr

    malformed_root = tmp_path / "malformed"
    malformed_root.mkdir()
    malformed = _run_allocation_assertion(
        malformed_root,
        target="GPU-target",
        allocation_spec="3",
        mapping={"3": "not-a-gpu-uuid"},
    )
    assert malformed.returncode == 76
    assert "resolved to invalid UUID" in malformed.stderr
