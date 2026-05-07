#!/usr/bin/env python
"""Run VDAM/InitialModel merge guards and save a reproducible ledger.

This guard is intentionally EM-scoped.  It preserves the ab-initio parity
contracts from ``codex/vdam-abinitio-parity`` without running the project-wide
SPA/ET long-test suite.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch")


@dataclass(frozen=True)
class GuardCommand:
    name: str
    argv: tuple[str, ...]
    backend: str = "cpu"
    required: bool = True


def _python() -> str:
    return sys.executable


def build_guard_commands(tier: str = "cpu", *, quick: bool = False) -> list[GuardCommand]:
    """Return the command plan for the requested merge-guard tier."""
    if tier not in {"cpu", "gpu", "all"}:
        raise ValueError(f"unknown tier {tier!r}")

    commands: list[GuardCommand] = []
    if tier in {"cpu", "all"}:
        commands.extend(
            [
                GuardCommand(
                    "py_compile",
                    (
                        _python(),
                        "-m",
                        "py_compile",
                        "scripts/evaluate_ab_initio_gt.py",
                        "scripts/run_vdam_abinitio_k2_smoke.py",
                        "scripts/run_vdam_abinitio_merge_guard.py",
                        "tests/unit/initial_model/test_evaluate_ab_initio_gt.py",
                        "tests/unit/initial_model/test_gt_metrics.py",
                        "tests/unit/initial_model/test_vdam_abinitio_merge_guard.py",
                    ),
                ),
                GuardCommand(
                    "vdam_abinitio_contracts",
                    (
                        _python(),
                        "-m",
                        "pytest",
                        "-v",
                        "tests/unit/initial_model/test_vdam_abinitio_merge_guard.py",
                        "tests/unit/initial_model/test_gt_metrics.py",
                        "tests/unit/initial_model/test_evaluate_ab_initio_gt.py",
                    ),
                ),
            ]
        )
        if not quick:
            commands.append(
                GuardCommand(
                    "initial_model_unit_suite",
                    (_python(), "-m", "pytest", "-v", "tests/unit/initial_model/"),
                )
            )
        commands.append(
            GuardCommand(
                "em_fast_guard",
                ("bash", "scripts/run_em_fast_guard.sh"),
            )
        )

    if tier in {"gpu", "all"}:
        commands.append(
            GuardCommand(
                "native_initialmodel_k2_smoke_gpu",
                (_python(), "scripts/run_vdam_abinitio_k2_smoke.py"),
                backend="gpu",
            )
        )
        commands.append(
            GuardCommand(
                "em_parity_fast_gpu",
                (
                    _python(),
                    "-m",
                    "pytest",
                    "-v",
                    "-s",
                    "--run-slow",
                    "--run-integration",
                    "--run-gpu",
                    "tests/integration/test_em_parity_fast.py",
                ),
                backend="gpu",
            )
        )
        commands.append(
            GuardCommand(
                "extract_em_parity_fast_tables",
                (_python(), "scripts/extract_em_parity_tables.py", "--tier", "fast"),
                backend="gpu",
                required=False,
            )
        )

    return commands


def _run_text(argv: tuple[str, ...] | list[str], *, env: dict[str, str] | None = None) -> tuple[int, str]:
    proc = subprocess.run(argv, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    return int(proc.returncode), (proc.stdout + proc.stderr)


def _git_snapshot() -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key, argv in {
        "commit": ("git", "rev-parse", "HEAD"),
        "branch": ("git", "symbolic-ref", "--short", "HEAD"),
        "status_short": ("git", "status", "--short", "--branch"),
        "diff_stat": ("git", "diff", "--stat"),
    }.items():
        code, text = _run_text(argv)
        items[key] = text.strip() if code == 0 else f"<failed rc={code}> {text.strip()}"
    return items


def _provenance(env: dict[str, str]) -> dict[str, Any]:
    env = dict(env)
    env["JAX_PLATFORMS"] = "cpu"
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    code, text = _run_text(
        (
            _python(),
            "-c",
            (
                "import json,pathlib,recovar,jax;"
                "repo=pathlib.Path.cwd().resolve();"
                "rf=pathlib.Path(recovar.__file__).resolve();"
                "jf=pathlib.Path(jax.__file__).resolve();"
                "assert str(rf).startswith(str(repo) + '/'), rf;"
                "assert '.pixi/envs/default/' in str(jf), jf;"
                "print(json.dumps({'recovar_file':str(rf),'jax_file':str(jf)}))"
            ),
        ),
        env=env,
    )
    if code != 0:
        return {"ok": False, "output": text}
    json_line = next((line for line in reversed(text.strip().splitlines()) if line.startswith("{")), None)
    if json_line is None:
        return {"ok": False, "output": text}
    payload = json.loads(json_line)
    payload["ok"] = True
    return payload


def _gpu_snapshot() -> str:
    code, text = _run_text(
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
    )
    return text.strip() if code == 0 else f"<nvidia-smi unavailable rc={code}> {text.strip()}"


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return env


def _env_for(command: GuardCommand) -> dict[str, str]:
    env = _base_env()
    if command.backend == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env.pop("JAX_PLATFORM_NAME", None)
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
    else:
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _tail(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_guard(
    *,
    tier: str,
    quick: bool,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    commands = build_guard_commands(tier, quick=quick)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SAFE_TO_DELETE").touch()

    ledger: dict[str, Any] = {
        "schema": "vdam_abinitio_merge_guard.v1",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "repo_root": str(REPO_ROOT),
        "tier": tier,
        "quick": bool(quick),
        "dry_run": bool(dry_run),
        "git": _git_snapshot(),
        "provenance": _provenance(_base_env()),
        "gpu_snapshot_start": _gpu_snapshot(),
        "commands": [],
    }

    overall_ok = bool(ledger["provenance"].get("ok", False))
    for command in commands:
        record: dict[str, Any] = {
            **asdict(command),
            "argv": list(command.argv),
        }
        if dry_run:
            record.update({"returncode": None, "elapsed_s": 0.0, "skipped": True})
            ledger["commands"].append(record)
            continue

        env = _env_for(command)
        env["VDAM_ABINITIO_GUARD_OUTPUT_DIR"] = str(output_dir)
        log_path = output_dir / f"{command.name}.log"
        t0 = time.perf_counter()
        proc = subprocess.run(
            command.argv,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        elapsed = time.perf_counter() - t0
        output = proc.stdout or ""
        log_path.write_text(output)
        record.update(
            {
                "returncode": int(proc.returncode),
                "elapsed_s": elapsed,
                "log_path": str(log_path),
                "output_tail": _tail(output),
            }
        )
        ledger["commands"].append(record)
        if command.required and proc.returncode != 0:
            overall_ok = False
            break

    ledger["finished_at"] = datetime.now().isoformat(timespec="seconds")
    ledger["gpu_snapshot_end"] = _gpu_snapshot()
    ledger["ok"] = bool(overall_ok and all(
        (not cmd.get("required", True)) or cmd.get("returncode") in (0, None)
        for cmd in ledger["commands"]
    ))
    summary_path = output_dir / "vdam_abinitio_merge_guard_summary.json"
    summary_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    ledger["summary_path"] = str(summary_path)
    return ledger


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tier", choices=("cpu", "gpu", "all"), default="cpu")
    parser.add_argument("--quick", action="store_true", help="Skip the full tests/unit/initial_model/ CPU suite.")
    parser.add_argument("--dry-run", action="store_true", help="Write the command plan without running commands.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for logs and summary JSON. Defaults to _agent_scratch/vdam_abinitio_merge_guard_<timestamp>.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / f"vdam_abinitio_merge_guard_{timestamp}_{os.getpid()}"
    )
    ledger = run_guard(
        tier=args.tier,
        quick=bool(args.quick),
        output_dir=output_dir,
        dry_run=bool(args.dry_run),
    )
    print(f"VDAM ab-initio merge guard summary: {ledger['summary_path']}")
    print(f"ok={ledger['ok']} tier={ledger['tier']} commands={len(ledger['commands'])}")
    for command in ledger["commands"]:
        rc = command.get("returncode")
        elapsed = command.get("elapsed_s", 0.0)
        print(f"  {command['name']}: rc={rc} elapsed_s={elapsed:.2f}")
    return 0 if ledger["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
