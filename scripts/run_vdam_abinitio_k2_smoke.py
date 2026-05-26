#!/usr/bin/env python
"""GPU smoke for the native K>1 InitialModel/VDAM path.

This is intentionally narrower than the EM-long quality tests: it runs the
GUI-facing native InitialModel driver for two K=2 iterations and verifies
that both class maps and per-iteration STAR/metadata artifacts are produced.
The guard exists because K=1 EM parity tests do not exercise the sparse
K-class pass-2 path that VDAM ab-initio uses after merge work.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from recovar.data_io.starfile import read_star, write_star


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_DIR = Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_pdb_k2_5k_128")
DEFAULT_OUTPUT_ROOT = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/vdam_abinitio_k2_smoke")


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _run_text(argv: list[str], *, env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return int(proc.returncode), proc.stdout or ""


def _tail(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _provenance(env: dict[str, str]) -> dict[str, Any]:
    code, text = _run_text(
        [
            sys.executable,
            "-c",
            (
                "import json,pathlib,recovar,jax;"
                "repo=pathlib.Path.cwd().resolve();"
                "rf=pathlib.Path(recovar.__file__).resolve();"
                "jf=pathlib.Path(jax.__file__).resolve();"
                "assert str(rf).startswith(str(repo) + '/'), rf;"
                "assert '.pixi/envs/default/' in str(jf), jf;"
                "print(json.dumps({'recovar_file':str(rf),'jax_file':str(jf),'devices':[str(d) for d in jax.devices()]}))"
            ),
        ],
        env=env,
    )
    if code != 0:
        return {"ok": False, "output": text}
    json_line = next((line for line in reversed(text.splitlines()) if line.startswith("{")), None)
    if json_line is None:
        return {"ok": False, "output": text}
    payload = json.loads(json_line)
    payload["ok"] = True
    return payload


def _required_fixture_paths(fixture_dir: Path) -> list[Path]:
    return [
        fixture_dir,
        fixture_dir / "particles.star",
        fixture_dir / "particles.128.mrcs",
    ]


def _write_subset_star(fixture_dir: Path, output_dir: Path, n_particles: int) -> Path:
    """Write a deterministic particle subset STAR that still points at datadir."""

    main_star, optics_star = read_star(str(fixture_dir / "particles.star"))
    n_particles = min(int(n_particles), int(len(main_star)))
    if n_particles <= 0:
        raise ValueError("native K2 smoke subset must contain at least one particle")
    # RELION InitialModel sorts by micrograph internally; keep the source row
    # order stable here so the guard is reproducible and easy to inspect.
    subset = main_star.iloc[np.arange(n_particles, dtype=np.int64)].copy()
    subset_path = output_dir / "particles_subset.star"
    write_star(str(subset_path), subset, optics_star)
    return subset_path


def _expected_artifacts(output_prefix: Path, nr_iter: int, nr_classes: int) -> list[Path]:
    artifacts = [
        output_prefix.parent / f"{output_prefix.name}_it{nr_iter:03d}_data.star",
        output_prefix.parent / f"{output_prefix.name}_it{nr_iter:03d}_model.star",
        output_prefix.parent / f"{output_prefix.name}_it{nr_iter:03d}_recovar_meta.json",
    ]
    artifacts.extend(
        output_prefix.parent / f"{output_prefix.name}_it{nr_iter:03d}_class{class_id:03d}.mrc"
        for class_id in range(1, nr_classes + 1)
    )
    return artifacts


def run_smoke(
    *,
    fixture_dir: Path,
    output_dir: Path,
    nr_iter: int,
    n_particles: int,
    image_batch_size: int,
    rotation_block_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SAFE_TO_DELETE").touch()
    output_prefix = output_dir / "run"
    env = _base_env()
    provenance = _provenance(env)

    ledger: dict[str, Any] = {
        "schema": "vdam_abinitio_k2_smoke.v1",
        "repo_root": str(REPO_ROOT),
        "fixture_dir": str(fixture_dir),
        "output_dir": str(output_dir),
        "nr_iter": int(nr_iter),
        "nr_classes": 2,
        "n_particles": int(n_particles),
        "image_batch_size": int(image_batch_size),
        "rotation_block_size": int(rotation_block_size),
        "provenance": provenance,
        "skipped": False,
    }
    if not provenance.get("ok", False):
        ledger["ok"] = False
        return ledger

    missing = [str(path) for path in _required_fixture_paths(fixture_dir) if not path.exists()]
    if missing:
        ledger.update(
            {
                "ok": False,
                "skipped": False,
                "skip_reason": "missing fixture paths",
                "missing": missing,
            }
        )
        return ledger

    subset_star = _write_subset_star(fixture_dir, output_dir, int(n_particles))
    cmd = [
        sys.executable,
        "scripts/run_ab_initio.py",
        "--i",
        str(subset_star),
        "--datadir",
        str(fixture_dir),
        "--o",
        str(output_prefix),
        "--nr_iter",
        str(nr_iter),
        "--K",
        "2",
        "--sym",
        "C1",
        "--particle_diameter",
        "200",
        "--tau2_fudge",
        "4",
        "--random_seed",
        "0",
        "--healpix_order",
        "1",
        "--oversampling",
        "1",
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--image_batch_size",
        str(image_batch_size),
        "--rotation_block_size",
        str(rotation_block_size),
        "--padding_factor",
        "1",
        "--eager_images",
    ]
    t0 = time.perf_counter()
    returncode, output = _run_text(cmd, env=env)
    elapsed = time.perf_counter() - t0
    (output_dir / "run_ab_initio.log").write_text(output)

    expected = _expected_artifacts(output_prefix, nr_iter, 2)
    artifact_records = [
        {
            "path": str(path),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
        }
        for path in expected
    ]
    missing_outputs = [record["path"] for record in artifact_records if not record["exists"] or record["size"] <= 0]
    meta_ok = False
    meta_path = output_prefix.parent / f"{output_prefix.name}_it{nr_iter:03d}_recovar_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            class_assignments = meta.get("class_assignments")
            selected_particle_ids = meta.get("selected_particle_ids")
            meta_ok = isinstance(class_assignments, list) and isinstance(selected_particle_ids, list)
        except json.JSONDecodeError:
            meta_ok = False

    ledger.update(
        {
            "ok": bool(returncode == 0 and not missing_outputs and meta_ok),
            "cmd": cmd,
            "returncode": int(returncode),
            "elapsed_s": elapsed,
            "output_tail": _tail(output),
            "artifacts": artifact_records,
            "missing_outputs": missing_outputs,
            "meta_ok": bool(meta_ok),
            "log_path": str(output_dir / "run_ab_initio.log"),
        }
    )
    return ledger


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_guard_root = os.environ.get("VDAM_ABINITIO_GUARD_OUTPUT_DIR")
    default_output = Path(default_guard_root) / "native_initialmodel_k2_smoke" if default_guard_root else DEFAULT_OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--nr-iter", type=int, default=2)
    parser.add_argument("--n-particles", type=int, default=512)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--rotation-block-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ledger = run_smoke(
        fixture_dir=args.fixture_dir,
        output_dir=args.output_dir,
        nr_iter=int(args.nr_iter),
        n_particles=int(args.n_particles),
        image_batch_size=int(args.image_batch_size),
        rotation_block_size=int(args.rotation_block_size),
    )
    summary_path = args.output_dir / "vdam_abinitio_k2_smoke_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": ledger["ok"], "skipped": ledger.get("skipped", False), "summary_path": str(summary_path)}))
    return 0 if ledger["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
