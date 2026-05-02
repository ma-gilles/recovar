"""Fast-tier EM parity regression tests (~5–10 min on a single A100).

Locks down kernel-level RELION parity so future PRs cannot silently
regress the E-step / M-step / FSC / tau² / SamplingPerturbation paths.

Tests:
1. K=1 128² 5k replay (iter 3→4) against the data_noise1_5k_normalized
   fixture. Asserts ``final_half[12]_corr_vs_relion >= 0.999`` and
   ``|ΔPmax| < 1e-3`` versus RELION it004.
2. K=2 128² 5k replay (iter 0→1) against data_pdb_k2_5k_128 — runs the
   K-class parity harness and asserts mean class-map correlation and
   per-image Pmax agreement, with Hungarian class permutation.

Both tests run the worktree provenance gate first via
:func:`recovar.utils.parity_provenance.assert_parity_ancestors`. A missing
parity-fix commit raises a clear error rather than silently producing
"broken parity" — the same protection ``scripts/run_multi_iter_parity.py``
provides for command-line use.

Quality ledger artifacts are written to
``tests/baselines/em_parity_quality_fast_ledger_*.json`` for visibility in
CI logs and PR descriptions; baseline comparisons go to
``em_parity_quality_fast_baseline.json`` (auto-created on first run).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import gpu_subprocess_env

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_SCRIPT = REPO_ROOT / "scripts" / "run_multi_iter_parity.py"
KCLASS_SCRIPT = REPO_ROOT / "scripts" / "run_k_class_parity.py"
BASELINES_DIR = REPO_ROOT / "tests" / "baselines"

# Pre-existing fixtures generated under /scratch/gpfs/GILLES/mg6942/em_relion_proj/.
# Tests skip if a fixture is absent (so the file works on dev machines without
# the cluster fixtures), but in CI / Slurm the fixtures must be present.
FIXTURE_BASE = Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj")

K1_FIXTURE_DIR = FIXTURE_BASE / "data_noise1_5k_normalized"
K1_RELION_DIR = K1_FIXTURE_DIR / "relion_ref_os0"
K1_DATA_STAR = K1_FIXTURE_DIR / "particles.star"
K1_GT_VOLUME = K1_FIXTURE_DIR / "reference_gt.mrc"

K2_FIXTURE_DIR = FIXTURE_BASE / "data_pdb_k2_5k_128"
K2_RELION_DIR = K2_FIXTURE_DIR / "relion_pdb_k2_os0_ref"
K2_DATA_STAR = K2_FIXTURE_DIR / "particles.star"


def _require_fixture(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip("Missing parity fixture(s):\n  " + "\n  ".join(missing))


def _assert_parity_ancestors_or_skip() -> None:
    """Hard-fail the test (don't skip) if the parity-fix commits are missing."""
    from recovar.utils.parity_provenance import (
        ParityAncestryError,
        assert_parity_ancestors,
        print_provenance_banner,
    )

    print_provenance_banner(stream=sys.stderr)
    try:
        assert_parity_ancestors()
    except ParityAncestryError as exc:
        pytest.fail(str(exc))


def _write_quality_ledger(name: str, payload: dict) -> Path:
    """Append the test result to em_parity_quality_fast_ledger_<name>.json.

    Each test writes one ledger file so multiple parametrizations don't
    clobber each other. The companion baseline file is read-only here —
    do not auto-update.
    """
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = BASELINES_DIR / f"em_parity_quality_fast_ledger_{name}.json"
    payload = dict(payload)
    payload.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with ledger_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return ledger_path


def _log_comparison(name: str, current: float, baseline: float | None, lower_is_better: bool = False) -> None:
    """Stream a comparison line to stderr so pytest does not capture it on pass."""
    if baseline is None:
        line = f"  {name:<32s} current={current:.6f}  baseline=missing"
    else:
        delta = current - baseline
        if lower_is_better:
            arrow = "↓" if delta < 0 else "↑"
        else:
            arrow = "↑" if delta > 0 else "↓"
        line = f"  {name:<32s} current={current:.6f}  baseline={baseline:.6f}  Δ={delta:+.6f} {arrow}"
    logger.info(line)
    print(line, file=sys.stderr, flush=True)


def _read_baseline(filename: str, key: str) -> float | None:
    path = BASELINES_DIR / filename
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())[key])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_k1_replay(tmp_path):
    """K=1 128² 5k iter-3→4 replay vs RELION run_it004.

    Pass criteria (machine-precision band on H100/A100):
      * ``corr(half1, RELION it004 half1) ≥ 0.999``
      * ``corr(half2, RELION it004 half2) ≥ 0.999``
      * ``|recovar.ave_Pmax − RELION.ave_Pmax| < 1e-3``

    Walltime budget: ~3 min on a single A100 (cold compile included).
    """
    _assert_parity_ancestors_or_skip()
    _require_fixture(PARITY_SCRIPT, K1_RELION_DIR, K1_DATA_STAR, K1_GT_VOLUME)

    output_dir = tmp_path / "k1_replay"
    output_dir.mkdir()

    cmd = [
        sys.executable,
        str(PARITY_SCRIPT),
        "--relion_dir",
        str(K1_RELION_DIR),
        "--data_star",
        str(K1_DATA_STAR),
        "--iter",
        "3",
        "--max_iter",
        "1",
        "--gt_volume",
        str(K1_GT_VOLUME),
        "--output_dir",
        str(output_dir),
    ]
    logger.info("K=1 replay cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0

    if proc.returncode == 2:
        # Provenance gate failed — bubble the error to the test runner.
        pytest.fail(
            "Parity provenance gate failed (exit 2). The worktree is missing "
            "required parity-fix commits.\nstdout:\n" + proc.stdout + "\nstderr:\n" + proc.stderr
        )
    assert proc.returncode == 0, (
        f"run_multi_iter_parity.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    npz_path = output_dir / "refinement_results.npz"
    assert npz_path.exists(), f"Missing refinement_results.npz at {npz_path}"
    npz = np.load(npz_path)

    half1_corr = float(npz["final_half1_corr_vs_relion"])
    half2_corr = float(npz["final_half2_corr_vs_relion"])
    pmax_traj = np.asarray(npz["ave_Pmax_trajectory"], dtype=np.float64)
    recovar_pmax = float(pmax_traj[0])

    # RELION it004 ave_Pmax from RELION's run_it004_optimiser.star — we read it
    # back from the npz where the script logs RELION's reference value.
    # The script prints "RELION ave_Pmax=..." but doesn't always save it;
    # use the documented value 0.9735 from the verified baseline as a sanity
    # band, and rely on map correlation for the strong assertion.
    relion_pmax_reference = 0.9735  # data_noise1_5k_normalized RELION it004 ave_Pmax
    pmax_abs_diff = abs(recovar_pmax - relion_pmax_reference)

    baseline_h1 = _read_baseline("em_parity_quality_fast_baseline.json", "k1_replay_half1_corr_vs_relion")
    baseline_h2 = _read_baseline("em_parity_quality_fast_baseline.json", "k1_replay_half2_corr_vs_relion")
    baseline_pmax = _read_baseline("em_parity_quality_fast_baseline.json", "k1_replay_pmax_abs_diff")

    print(file=sys.stderr, flush=True)
    print("=== K=1 replay parity (iter 3→4 vs RELION it004) ===", file=sys.stderr, flush=True)
    _log_comparison("k1_replay_half1_corr_vs_relion", half1_corr, baseline_h1)
    _log_comparison("k1_replay_half2_corr_vs_relion", half2_corr, baseline_h2)
    _log_comparison("k1_replay_pmax_abs_diff", pmax_abs_diff, baseline_pmax, lower_is_better=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    payload = {
        "k1_replay_half1_corr_vs_relion": half1_corr,
        "k1_replay_half2_corr_vs_relion": half2_corr,
        "k1_replay_pmax_recovar": recovar_pmax,
        "k1_replay_pmax_relion_reference": relion_pmax_reference,
        "k1_replay_pmax_abs_diff": pmax_abs_diff,
        "k1_replay_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("k1_replay", payload)
    logger.info("K=1 replay ledger: %s", ledger)

    # NEVER widen tolerance to make a test pass. Fix the code instead.
    assert half1_corr >= 0.999, (
        f"K=1 replay half1 corr {half1_corr:.6f} below threshold 0.999. "
        f"Replay parity has regressed — verify the load-bearing parity commits."
    )
    assert half2_corr >= 0.999, (
        f"K=1 replay half2 corr {half2_corr:.6f} below threshold 0.999. "
        f"Replay parity has regressed — verify the load-bearing parity commits."
    )
    assert pmax_abs_diff < 1e-3, (
        f"K=1 replay |ΔPmax| {pmax_abs_diff:.6f} exceeds threshold 1e-3 vs RELION it004={relion_pmax_reference}."
    )


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_kclass_replay(tmp_path):
    """K=2 128² 5k iter-0→1 replay against the pdb_k2 RELION Class3D reference.

    Asserts joint class × pose RELION parity by:
      * Hungarian-aligning recovar's per-class maps to RELION's
      * Asserting ``mean_corr ≥ 0.95`` after Hungarian alignment
      * Asserting per-image Pmax agreement within ``|ΔPmax| < 1e-2``

    The K=2 pdb fixture only ships RELION it000 + it001, so this is the
    smallest available K-class parity check. K=4 / longer chains belong in
    the EM-long tier.
    """
    _assert_parity_ancestors_or_skip()
    _require_fixture(KCLASS_SCRIPT, K2_RELION_DIR, K2_DATA_STAR)

    output_dir = tmp_path / "kclass_replay"
    output_dir.mkdir()

    cmd = [
        sys.executable,
        str(KCLASS_SCRIPT),
        "--relion-dir",
        str(K2_RELION_DIR),
        "--data-star",
        str(K2_DATA_STAR),
        "--prev-iter",
        "0",
        "--target-iter",
        "1",
        "--output-dir",
        str(output_dir),
    ]
    logger.info("K-class replay cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0

    assert proc.returncode == 0, (
        f"run_k_class_parity.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    summary_path = output_dir / "summary.json"
    assert summary_path.exists(), f"Missing summary.json at {summary_path}"
    summary = json.loads(summary_path.read_text())

    best_perm = summary["best_permutation"]
    mean_corr = float(best_perm["mean_corr"])
    map_corrs = [float(c) for c in best_perm["map_correlations"]]
    pmax_abs_mean = float(summary["pmax"]["abs_mean"])
    pmax_abs_max = float(summary["pmax"]["abs_max"])
    class_acc = float(summary["class_assignment_accuracy_after_permutation"])

    baseline_mean = _read_baseline("em_parity_quality_fast_baseline.json", "kclass_replay_mean_corr")
    baseline_pmax = _read_baseline("em_parity_quality_fast_baseline.json", "kclass_replay_pmax_abs_mean")
    baseline_acc = _read_baseline("em_parity_quality_fast_baseline.json", "kclass_replay_class_assignment_accuracy")

    print(file=sys.stderr, flush=True)
    print("=== K=2 replay parity (iter 0→1 vs RELION it001) ===", file=sys.stderr, flush=True)
    _log_comparison("kclass_replay_mean_corr", mean_corr, baseline_mean)
    _log_comparison("kclass_replay_pmax_abs_mean", pmax_abs_mean, baseline_pmax, lower_is_better=True)
    _log_comparison("kclass_replay_class_assignment_accuracy", class_acc, baseline_acc)
    print(f"  per-class map corrs: {map_corrs}", file=sys.stderr, flush=True)
    print(f"  pmax_abs_max={pmax_abs_max:.6g}", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    payload = {
        "kclass_replay_mean_corr": mean_corr,
        "kclass_replay_per_class_map_corr": map_corrs,
        "kclass_replay_pmax_abs_mean": pmax_abs_mean,
        "kclass_replay_pmax_abs_max": pmax_abs_max,
        "kclass_replay_class_assignment_accuracy": class_acc,
        "kclass_replay_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("kclass_replay", payload)
    logger.info("K-class replay ledger: %s", ledger)

    # K=2 iter 0→1 is the first K-class iteration after class seeds are loaded;
    # it exercises the joint class × pose posterior. Expect strong but not
    # machine-precision parity because RELION's first iteration uses
    # winner-take-all in some configurations.
    assert mean_corr >= 0.95, (
        f"K-class replay mean_corr {mean_corr:.6f} below threshold 0.95. K-class kernel parity has regressed."
    )
    assert pmax_abs_mean < 1e-2, f"K-class replay |ΔPmax|.mean {pmax_abs_mean:.6g} exceeds threshold 1e-2."
    assert class_acc >= 0.95, (
        f"K-class replay class assignment accuracy {class_acc:.4f} below threshold 0.95 after Hungarian permutation."
    )
