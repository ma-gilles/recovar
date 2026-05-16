"""End-to-end iter-10 InitialModel parity guard for K=1/K=2/K=4.

Pinned at commit 4a581cc5 (2026-05-09/10) after the abem-after-merge-20260508
session shipped the K=4 sigma2 underflow clamp (d9fed5b3), the K=1 200-iter
projector slab truncation (ddd24d29), and the K=1-conditional halfset-averaging
SsnrMap fix (db4f49c4). Together those land near-perfect iter-10 parity:

* K=1 5k 128² (PDB) — mean volume CC vs RELION ≥ 0.999, autosampling
  HEALPix 1→2 fires at iter-10 (matches RELION).
* K=2 5k 128² (PDB) — mean volume CC vs RELION ≥ 0.998 (best-permutation
  matched; K=2 has 2 conformations).
* K=4 5k 128² (PDB) — mean volume CC vs RELION ≥ 0.997 (best-permutation
  over the 24 K-permutations).

The upcoming merge folds in two other in-flight branches (EM, VDAM, PPCA
refinement) that touch m_step.py, gt_metrics.py, scatter, and the dense
engine. This test is the load-bearing post-merge guard: any code change
that drops iter-10 mean CC by >1 percentage point or balloons wall-time
by >50 % fails CI before reaching ``dev``.

Quality + perf baselines live in ``tests/baselines/initial_model_iter10_baseline.json``.
The test compares against the baseline if present, and writes the current
result to ``tests/baselines/initial_model_iter10_ledger.json`` for PR-time
visibility (mirroring the em_parity_quality_fast_ledger_*.json pattern).

Skips cleanly when:
* the K=N data fixture is absent (so the file is portable to dev hosts),
* RELION K=N nr_iter=10 reference dumps are absent (env-var override
  available so each lab can point at its own reference cache).
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import gpu_subprocess_env

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_AB_INITIO = REPO_ROOT / "scripts" / "run_ab_initio.py"
# Ledgers are written next to this test (NOT under tests/baselines/, which is
# protected ground-truth from the OLD recovar publication and must not be
# touched). Quality + perf baselines live INLINE in BASELINES below so they
# are tracked in git, reviewed at PR time, and never accidentally bumped by a
# bootstrap run.
LEDGER_DIR = REPO_ROOT / "tests" / "em_parity_long"
LEDGER_FILE = LEDGER_DIR / "initial_model_iter10_ledger.json"

FIXTURE_BASE = Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj")
PDB_K2_DIR = FIXTURE_BASE / "data_pdb_k2_5k_128"
PDB_K4_DIR = FIXTURE_BASE / "data_pdb_k4_5k_128"

# RELION nr_iter=10 reference dumps. By default we look at the
# `_agent_scratch/abem_long10_*` location; users can override via
# RECOVAR_INITIALMODEL_ITER10_RELION_REF=<dir> if their cache lives elsewhere.
DEFAULT_RELION_REF_BASE = Path(
    os.environ.get(
        "RECOVAR_INITIALMODEL_ITER10_RELION_REF",
        "/scratch/gpfs/GILLES/mg6942/_agent_scratch/abem_long10_20260508_143725_31556",
    )
)

# Tolerance bands. mean_cc_floor = baseline - mean_cc_drop_tol; wall_time
# is allowed wall_time_grow_factor × the baseline.
MEAN_CC_DROP_TOL = 0.01  # at most 1pp regression
WALL_TIME_GROW_FACTOR = 1.5

# Inline baselines pinned 2026-05-09/10 on abem-after-merge-20260508 at commit
# 4a581cc5. ``mean_cc`` is recovar-vs-RELION best-permutation centered
# correlation at iter-10. ``wall_time_sec`` is end-to-end run_ab_initio.py
# wall time including iter-1 JIT compile, measured on a single A100-80GB.
# Update only after deliberately validating an improvement.
BASELINES: dict[str, dict] = {
    "k1_pdb_5k_128": {
        "mean_cc": 0.9994,
        "wall_time_sec": 480.0,
        "_note": (
            "iter-10 HEALPix 1->2 autosampling fires (matches RELION). cs trajectory "
            "matches RELION at every iter except iter-7 where recovar jumps cs 52->62 "
            "vs RELION's 52->56->62 (1-iter offset, harmless)."
        ),
    },
    "k2_pdb_5k_128": {
        "mean_cc": 0.9979,
        "wall_time_sec": 700.0,
        "_note": (
            "Per-class CCs: c0=0.9975, c1=0.9983 (best-permutation). K-class path "
            "uses accum_h0.weight in M-step SsnrMap (halfset-averaging conditional "
            "applies to K=1 only)."
        ),
    },
    "k4_pdb_5k_128": {
        "mean_cc": 0.9971,
        "wall_time_sec": 950.0,
        "_note": (
            "Per-class CCs: 0.9974, 0.9993, 0.9949, 0.9968. Validates the K=4 sigma2 "
            "underflow clamp (d9fed5b3) -- completes 10 iters; pre-clamp this crashed "
            "at iter-7."
        ),
    },
}


def _require_paths(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip("Missing fixture(s) for InitialModel iter-10 parity:\n  " + "\n  ".join(missing))


def _write_ledger(payload: dict) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if LEDGER_FILE.exists():
        try:
            existing = json.loads(LEDGER_FILE.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing[payload["case"]] = payload
    LEDGER_FILE.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")


def _log_comparison(label: str, current: float, baseline: float | None, lower_is_better: bool = False) -> None:
    if baseline is None:
        line = f"  {label:<32s} current={current:>10.6f}  baseline=missing"
    else:
        delta = current - baseline
        arrow = ("↓" if delta < 0 else "↑") if lower_is_better else ("↑" if delta > 0 else "↓")
        line = f"  {label:<32s} current={current:>10.6f}  baseline={baseline:>10.6f}  Δ={delta:+.6f} {arrow}"
    logger.info(line)
    print(line, file=sys.stderr, flush=True)


def _cc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denom) if denom > 0 else 0.0


def _best_permutation_mean_cc(rec_classes: list[np.ndarray], rel_classes: list[np.ndarray]) -> float:
    K = len(rec_classes)
    assert len(rel_classes) == K
    best_mean = -np.inf
    for perm in itertools.permutations(range(K)):
        mean_cc = float(np.mean([_cc(rec_classes[i], rel_classes[perm[i]]) for i in range(K)]))
        if mean_cc > best_mean:
            best_mean = mean_cc
    return best_mean


def _run_recovar_iter10(
    *,
    case_label: str,
    K: int,
    data_dir: Path,
    output_dir: Path,
) -> tuple[float, list[Path]]:
    """Run K=K InitialModel for nr_iter=10 and return (wall_time, class MRC paths)."""
    cmd = [
        sys.executable,
        str(RUN_AB_INITIO),
        "--i",
        str(data_dir / "particles.star"),
        "--datadir",
        str(data_dir),
        "--o",
        str(output_dir / "recovar" / "run"),
        "--nr_iter",
        "10",
        "--K",
        str(K),
        "--sym",
        "C1",
        "--particle_diameter",
        "200",
        "--tau2_fudge",
        "4",
        "--j",
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
        "--bootstrap_min_particles",
        "1000",
        "--sigma2_min_particles",
        "1000",
        "--padding_factor",
        "1",
        "--eager_images",
        "--image_batch_size",
        "250",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "recovar.log"
    logger.info("[%s] cmd: %s", case_label, " ".join(cmd))
    t0 = time.time()
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log_text = log_path.read_text()[-4000:]
        pytest.fail(f"run_ab_initio.py exited {proc.returncode} for {case_label}\nlog tail:\n{log_text}")
    rec_paths = [output_dir / "recovar" / f"run_it010_class{c:03d}.mrc" for c in range(1, K + 1)]
    missing_recs = [str(p) for p in rec_paths if not p.exists()]
    if missing_recs:
        pytest.fail(f"{case_label}: expected iter-10 outputs missing: {missing_recs}")
    return elapsed, rec_paths


def _load_relion_iter10_classes(relion_dir: Path, K: int) -> list[np.ndarray]:
    """Load RELION's nr_iter=10 class volumes (require all K)."""
    import mrcfile

    paths = [relion_dir / f"run_it010_class{c:03d}.mrc" for c in range(1, K + 1)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "RELION iter-10 reference dumps not present for this K; "
            "set RECOVAR_INITIALMODEL_ITER10_RELION_REF if your cache lives elsewhere.\n  " + "\n  ".join(missing)
        )
    return [np.asarray(mrcfile.read(str(p)), dtype=np.float64) for p in paths]


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "K, data_dir, relion_subdir, baseline_key",
    [
        (1, PDB_K2_DIR, "relion_K1_10iter", "k1_pdb_5k_128"),
        (2, PDB_K2_DIR, "relion_K2_10iter", "k2_pdb_5k_128"),
        (4, PDB_K4_DIR, "relion_K4_10iter", "k4_pdb_5k_128"),
    ],
)
def test_initialmodel_iter10_parity(tmp_path, K, data_dir, relion_subdir, baseline_key):
    """End-to-end K-class InitialModel iter-10 quality + perf guard.

    Quality: best-permutation mean volume CC vs RELION nr_iter=10 reference
    must be within ``MEAN_CC_DROP_TOL`` of the baseline (or ≥ 0.99 if no
    baseline is recorded yet — first-run bootstrap).

    Perf: wall-time must be within ``WALL_TIME_GROW_FACTOR × baseline``
    (or skipped if no baseline). The wall-time check is meant to catch
    catastrophic regressions (e.g. JIT cache misses on every iter, fp32→fp64
    accidents); not micro-tuning.
    """
    _require_paths(RUN_AB_INITIO, data_dir, data_dir / "particles.star")
    relion_dir = DEFAULT_RELION_REF_BASE / relion_subdir
    rel_classes = _load_relion_iter10_classes(relion_dir, K)

    output_dir = tmp_path / f"k{K}_iter10"
    elapsed, rec_paths = _run_recovar_iter10(
        case_label=baseline_key,
        K=K,
        data_dir=data_dir,
        output_dir=output_dir,
    )

    import mrcfile

    rec_classes = [np.asarray(mrcfile.read(str(p)), dtype=np.float64) for p in rec_paths]
    if K == 1:
        mean_cc = _cc(rec_classes[0], rel_classes[0])
    else:
        mean_cc = _best_permutation_mean_cc(rec_classes, rel_classes)

    baseline = BASELINES.get(baseline_key, {})
    baseline_cc = baseline.get("mean_cc")
    baseline_wall = baseline.get("wall_time_sec")

    _log_comparison(f"{baseline_key} mean_cc", mean_cc, baseline_cc)
    _log_comparison(f"{baseline_key} wall_sec", float(elapsed), baseline_wall, lower_is_better=True)

    payload = {
        "case": baseline_key,
        "K": int(K),
        "data_dir": str(data_dir),
        "relion_ref_dir": str(relion_dir),
        "mean_cc": float(mean_cc),
        "wall_time_sec": float(elapsed),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": os.environ.get("RECOVAR_GIT_HEAD", ""),
    }
    _write_ledger(payload)

    if baseline_cc is not None:
        floor = float(baseline_cc) - MEAN_CC_DROP_TOL
        assert mean_cc >= floor, (
            f"{baseline_key}: iter-10 mean CC {mean_cc:.6f} below floor {floor:.6f} "
            f"(baseline {baseline_cc:.6f}, allowed drop {MEAN_CC_DROP_TOL:.4f})"
        )
    else:
        # Bootstrap floor when no baseline exists yet — keep this loose
        # enough that a fresh worktree can record a first baseline without
        # arguing about absolute thresholds.
        assert mean_cc >= 0.99, f"{baseline_key}: no baseline recorded and mean CC {mean_cc:.6f} below 0.99 floor"

    if baseline_wall is not None and float(baseline_wall) > 0:
        ceil = float(baseline_wall) * WALL_TIME_GROW_FACTOR
        assert float(elapsed) <= ceil, (
            f"{baseline_key}: wall-time {elapsed:.1f}s exceeded {WALL_TIME_GROW_FACTOR}× baseline "
            f"{baseline_wall:.1f}s (ceiling {ceil:.1f}s)"
        )
