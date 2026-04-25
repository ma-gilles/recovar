"""Quality + correctness regression test for the RELION-parity workflow.

This test runs a SHORT mid-trajectory replay (one iter) under
``parity_dump`` and compares the resulting per-iter observables against a
frozen baseline JSON at ``tests/baselines/parity/quality_baseline_5k_128.json``.
The baseline JSON is the source of truth for tolerances; if you intentionally
change the algorithm and metrics shift, update the JSON in the same commit
and document the change in the JSON's ``source_runs`` field.

Default suite (``test-parity-fast``): ~5-10 minutes total on one A100,
single GPU, **with a warm JAX compile cache**. The test sets
``JAX_COMPILATION_CACHE_DIR`` to a known location automatically. With a
COLD cache, expect 25-30 min per scenario for the first run; subsequent
runs hit the warm cache and run in ~5 min.

The pytest timeout below (``timeout=1800`` = 30 min) is large to handle
the cold-cache case. CI infrastructure should pre-warm the cache and run
this with a tighter timeout (``--timeout=600``) to catch true regressions.

Skips automatically when no GPU is available so the test is also safe
to import on login nodes.

Optional / slow scenarios (anything > 30 min on cold cache) are marked
``optional: true`` in the baseline JSON and are NOT picked up by the
default parametrize sweep — add a separate Slurm-targeted job for those.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO / "tests/baselines/parity/quality_baseline_5k_128.json"

# RELION reference dumps live under _agent_scratch/parity/relion (symlinks
# pointing at the canonical 5k/128 reference dump). The fixture path itself
# is also recorded in the baseline JSON; we use that for fallback discovery.


def _baseline_data() -> dict:
    if not BASELINE_PATH.exists():
        return {"scenarios": {}, "fixture_path": "", "relion_reference_dir": ""}
    return json.loads(BASELINE_PATH.read_text())


def _scenario_ids() -> list[str]:
    data = _baseline_data()
    return [name for name, scen in data.get("scenarios", {}).items() if not scen.get("optional", False)]


def _has_gpu() -> bool:
    """Return True if at least one JAX GPU device is available."""
    try:
        import jax

        return any(d.platform.startswith("gpu") for d in jax.devices())
    except Exception:
        return False


def _resolve_relion_dump_dir(data: dict) -> Path:
    """Find the RELION reference iter_NNN.npz dir.

    Search order:
      1. ``$RECOVAR_PARITY_RELION_REF_DIR`` env var
      2. ``_agent_scratch/parity/relion`` next to repo root
      3. ``relion_reference_dir`` from baseline JSON (a parity dump dir, not a RELION run dir)
    """
    env = os.environ.get("RECOVAR_PARITY_RELION_REF_DIR")
    if env:
        return Path(env)
    candidate = REPO / "_agent_scratch/parity/relion"
    if candidate.exists():
        return candidate
    return Path(data.get("relion_reference_dump_dir") or "")


pytestmark = [
    pytest.mark.parity,
    pytest.mark.skipif(not _has_gpu(), reason="parity test requires GPU"),
    pytest.mark.skipif(not BASELINE_PATH.exists(), reason="quality baseline JSON not present"),
]


@pytest.mark.parametrize("scenario_name", _scenario_ids())
def test_parity_quality_scenario(scenario_name: str, tmp_path: Path) -> None:
    """Run one parity scenario, dump observables, then run the checker."""

    data = _baseline_data()
    scen = data["scenarios"][scenario_name]
    cfg = scen["config"]

    fixture_path = Path(scen.get("fixture_path") or data.get("fixture_path"))
    relion_run_dir = Path(scen.get("relion_reference_dir") or data.get("relion_reference_dir"))
    if not fixture_path.exists():
        pytest.skip(f"fixture not found at {fixture_path}")
    if not relion_run_dir.exists():
        pytest.skip(f"RELION reference run dir not found at {relion_run_dir}")
    relion_dump_dir = _resolve_relion_dump_dir(data)
    relion_iter = int(cfg["init_iter"]) + int(cfg["max_iter"])
    rel_npz = relion_dump_dir / f"iter_{relion_iter:03d}.npz"
    if not rel_npz.exists():
        pytest.skip(f"RELION dump not found at {rel_npz}; regenerate with scripts/parity/dump_relion_iter.py")

    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    env = os.environ.copy()
    env["RECOVAR_PARITY_DUMP_DIR"] = str(dump_dir)
    env.setdefault("PYTHONNOUSERSITE", "1")
    # Allow the warm JAX cache to be pre-populated; tests fall back to no
    # cache when env var unset.
    if "JAX_COMPILATION_CACHE_DIR" not in env:
        # Optional: developers can pre-warm /scratch/gpfs/GILLES/mg6942/jax_cache/parity_5k_128
        guess = Path("/scratch/gpfs/GILLES/mg6942/jax_cache/parity_5k_128")
        if guess.exists():
            env["JAX_COMPILATION_CACHE_DIR"] = str(guess)
    # Pin to a single GPU. Respect what the caller already set.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    cmd = [
        sys.executable,
        str(REPO / "scripts/run_multi_iter_parity.py"),
        "--relion_dir",
        str(relion_run_dir),
        "--data_star",
        str(fixture_path / "particles.star"),
        "--iter",
        str(cfg["init_iter"]),
        "--max_iter",
        str(cfg["max_iter"]),
        "--skip_final_iteration",
        "--local_engine",
        cfg["local_engine"],
        "--output_dir",
        str(out_dir),
    ]
    if cfg.get("jax_cache_dir"):
        cmd += ["--jax_cache_dir", cfg["jax_cache_dir"]]

    # Generous timeout (30 min) tolerates a cold JAX compile cache.
    # Set ``RECOVAR_PARITY_TIMEOUT_S`` to override.
    timeout_s = int(os.environ.get("RECOVAR_PARITY_TIMEOUT_S", "1800"))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        pytest.fail(
            f"workload failed (rc={proc.returncode}):\n--- stdout tail ---\n{proc.stdout[-2000:]}"
            f"\n--- stderr tail ---\n{proc.stderr[-2000:]}"
        )

    rec_npz = dump_dir / f"iter_{relion_iter:03d}.npz"
    assert rec_npz.exists(), f"recovar parity dump not produced at {rec_npz}; tail of stdout:\n{proc.stdout[-1000:]}"

    check = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/parity/check_parity.py"),
            "--baseline",
            str(BASELINE_PATH),
            "--scenario",
            scenario_name,
            "--recovar-dump",
            str(rec_npz),
            "--relion-dump",
            str(rel_npz),
            "--exit-code-on-regression",
        ],
        capture_output=True,
        text=True,
    )
    print(check.stdout)
    if check.returncode != 0:
        pytest.fail(f"parity check FAILED for {scenario_name}\n{check.stdout}\n{check.stderr}")
