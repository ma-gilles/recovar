"""GPU regressions for EM replay, initialization and sampling paths.

The seven cases cover K1 replay/cold start/perturbation replay, K2 replay,
and three K4 initialization/sampling combinations. K4 replay requires a
same-oracle dispatch schedule; choose the oracle for each case's grid.
The historical strict K4 case disables oversampling, unlike the available
oversampling-1 captures. Its name does not establish matched-state parity.

Quality ledgers are saved under pytest's temporary output directory; retain
them with --basetemp and inspect them with scripts/extract_em_parity_tables.py.
Correlation assertions are regression checks, not the current FSC/FSC-AUC
quality gates. Baselines are read only, and source ancestry is checked first.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import starfile
from conftest import gpu_subprocess_env

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_SCRIPT = REPO_ROOT / "scripts" / "run_multi_iter_parity.py"
KCLASS_SCRIPT = REPO_ROOT / "scripts" / "run_k_class_parity.py"
REFINE_SCRIPT = REPO_ROOT / "scripts" / "run_full_refinement.py"
BASELINES_DIR = REPO_ROOT / "tests" / "baselines"

# Pre-existing fixtures generated under /scratch/gpfs/GILLES/mg6942/em_relion_proj/.
# Tests skip if a fixture is absent (so the file works on dev machines without
# the cluster fixtures), but in CI / Slurm the fixtures must be present.
FIXTURE_BASE = Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj")

K1_FIXTURE_DIR = FIXTURE_BASE / "data_noise1_5k_normalized"
K1_RELION_DIR = K1_FIXTURE_DIR / "relion_ref_os0"
K1_DATA_STAR = K1_FIXTURE_DIR / "particles.star"
K1_GT_VOLUME = K1_FIXTURE_DIR / "reference_gt.mrc"
K1_RELION_RANDOM_SEED = 1775735620

K2_FIXTURE_DIR = FIXTURE_BASE / "data_pdb_k2_5k_128"
K2_RELION_DIR = K2_FIXTURE_DIR / "relion_pdb_k2_os0_ref"
K2_DATA_STAR = K2_FIXTURE_DIR / "particles.star"

K4_FIXTURE_DIR = FIXTURE_BASE / "data_pdb_k4_5k_128"
# The shipped K4 oracle was a single-process RELION run captured before dispatch
# logging existed, so strict K>1 replay cannot use it. Point these at a
# dispatch-capable oracle rerun (RELION MPI with RELION_DISPATCH_LOG) and its
# schema-3 schedule to run the K4 cases; unset, the tests report the fixture gap.
K4_RELION_DIR = Path(os.environ.get("EM_PARITY_FAST_K4_RELION_DIR", str(K4_FIXTURE_DIR / "relion_pdb_k4_os0_ref")))
K4_DISPATCH_SCHEDULE = os.environ.get("EM_PARITY_FAST_K4_DISPATCH_SCHEDULE") or None


def _k4_dispatch_schedule_args() -> list[str]:
    """Strict K>1 replay needs the dispatch schedule captured with the oracle."""

    if K4_DISPATCH_SCHEDULE is None:
        return []
    _require_fixture(Path(K4_DISPATCH_SCHEDULE))
    return ["--relion-dispatch-schedule", K4_DISPATCH_SCHEDULE]
K4_DATA_STAR = K4_FIXTURE_DIR / "particles.star"


def _require_fixture(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip("Missing parity fixture(s):\n  " + "\n  ".join(missing))


def _assert_parity_ancestors_or_skip() -> None:
    """Hard-fail the test (don't skip) if the parity-fix commits are missing."""
    from recovar.em.diagnostics.parity_provenance import (
        ParityAncestryError,
        assert_parity_ancestors,
        print_provenance_banner,
    )

    print_provenance_banner(stream=sys.stderr)
    try:
        assert_parity_ancestors()
    except ParityAncestryError as exc:
        pytest.fail(str(exc))


def _write_quality_ledger(name: str, payload: dict, *, output_dir: Path) -> Path:
    """Write this case's result beside its outputs, separately from baselines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / f"em_parity_quality_fast_ledger_{name}.json"
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


def _map_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


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
    """Replay K1 iteration 3→4 on the 5k/128 fixture.

    Compare both half maps and optimizer Pmax with RELION iteration 4.
    The assertions retain the correlation floor and Pmax error bound.
    """
    _assert_parity_ancestors_or_skip()
    model_path = K1_RELION_DIR / "run_it004_half1_model.star"
    _require_fixture(PARITY_SCRIPT, K1_RELION_DIR, K1_DATA_STAR, K1_GT_VOLUME, model_path)

    output_dir = tmp_path / "k1_replay"
    output_dir.mkdir(parents=True)

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
        "--skip_final_iteration",
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

    relion_pmax_reference = float(starfile.read(model_path)["model_general"]["rlnAveragePmax"])
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
    ledger = _write_quality_ledger("k1_replay", payload, output_dir=output_dir)
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
    output_dir.mkdir(parents=True)

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
        # This fixture uses RELION firstiter CC: select the global coarse winner,
        # then refine its pose within that class (acc_ml_optimiser_impl.h).
        # The replay's default broad adaptive support is a different policy.
        "--firstiter-cc-mode",
        "force",
        "--firstiter-cc-pass2-only-best-coarse",
        "--adaptive-2pass",
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
    ledger = _write_quality_ledger("kclass_replay", payload, output_dir=output_dir)
    logger.info("K-class replay ledger: %s", ledger)

    # K=2 iter 0→1 is the first K-class iteration after class seeds are loaded;
    # it exercises the joint class × pose posterior. With adaptive 2-pass
    # (--adaptive-2pass) recovar matches RELION's fine-grid pose refinement
    # for the per-class winner_take_all M-step, lifting mean_corr from
    # 0.984 (coarse-only) to ~0.99.
    assert mean_corr >= 0.99, (
        f"K-class replay mean_corr {mean_corr:.6f} below threshold 0.99. K-class kernel parity has regressed."
    )
    assert pmax_abs_mean < 1e-2, f"K-class replay |ΔPmax|.mean {pmax_abs_mean:.6g} exceeds threshold 1e-2."
    assert class_acc >= 0.95, (
        f"K-class replay class assignment accuracy {class_acc:.4f} below threshold 0.95 after Hungarian permutation."
    )


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_k1_coldstart(tmp_path):
    """Run three K1 iterations from raw 5k/128 inputs and the initial map.

    Preserve RELION half-set membership and CUDA image preprocessing, while
    letting RECOVAR initialize and update noise, tau, sigma and FSC state.
    Compare iteration-3 half maps/Pmax and check the sigma-offset update.
    """
    _assert_parity_ancestors_or_skip()
    _require_fixture(REFINE_SCRIPT, K1_FIXTURE_DIR, K1_RELION_DIR, K1_DATA_STAR)

    output_dir = tmp_path / "k1_coldstart"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(K1_FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--max_iter",
        "3",
        "--healpix_order",
        "3",
        "--offset_range",
        "3.0",
        "--offset_step",
        "1.0",
        "--adaptive_oversampling",
        "0",
        "--tau2_fudge",
        "1.0",
        "--perturb_factor",
        "0.5",  # match RELION's --perturb 0.5
        "--seed",
        str(K1_RELION_RANDOM_SEED),  # drives SamplingPerturbation like RELION's --random_seed
        "--init_resolution",
        "30.0",
        "--image_batch_size",
        "200",
        "--rotation_block_size",
        "2000",
        # Use RELION's exact half-set assignment so halfmap comparison is meaningful.
        # Without this flag recovar uses a random split via --seed, and recovar's
        # half-1 ends up containing different particles than RELION's half-1
        # (corr would be nearly anti-correlated with magnitude ≈ recovar's
        # half-2 vs RELION's half-1 — meaningless for parity).
        "--relion_half_sets",
        str(K1_FIXTURE_DIR / "particles_with_halfsets.star"),
        # The fresh K=1 defaults (source-faithful powerClass normalization, exact
        # BPref operands) score from RELION's CUDA image preprocessing, as the
        # K1 completion launcher does.
        "--image-fourier-backend",
        "relion_cuda",
    ]
    logger.info("K=1 cold-start cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    npz = np.load(output_dir / "refinement_results.npz")
    pmax_traj = np.asarray(npz["ave_Pmax_trajectory"], dtype=np.float64)
    sigma_traj = np.asarray(npz["sigma_offset_trajectory"], dtype=np.float64)
    sigma_used_traj = np.asarray(npz["sigma_offset_used_trajectory"], dtype=np.float64)
    n_iters = int(pmax_traj.size)
    assert n_iters >= 3, f"Expected ≥3 iterations, got {n_iters}"

    # Compare halfmaps against RELION it003. recovar's `write_mrc` saves
    # with a (2,1,0) transpose (cryosparc/cryoDRGN convention) and `load_mrc`
    # un-transposes round-trip; RELION's `load_relion_volume` applies
    # `-(2,1,0)` to RELION MRCs to land in recovar's frame. Mixing helpers
    # incorrectly gives corr ≈ -0.98 (raw + raw — sign flip) or 0.45 (raw
    # recovar + load_relion — half-applied transpose). Both files via the
    # right helper.
    from recovar.utils import helpers as _recovar_helpers

    def _load_recovar_real(p: Path) -> np.ndarray:
        return np.asarray(_recovar_helpers.load_mrc(str(p)), dtype=np.float64)

    def _load_relion_real(p: Path) -> np.ndarray:
        return np.asarray(_recovar_helpers.load_relion_volume(str(p)), dtype=np.float64)

    recovar_h1 = _load_recovar_real(output_dir / "final_half1.mrc")
    recovar_h2 = _load_recovar_real(output_dir / "final_half2.mrc")
    relion_h1 = _load_relion_real(K1_RELION_DIR / "run_it003_half1_class001.mrc")
    relion_h2 = _load_relion_real(K1_RELION_DIR / "run_it003_half2_class001.mrc")
    h1_corr = _map_correlation(recovar_h1, relion_h1)
    h2_corr = _map_correlation(recovar_h2, relion_h2)

    # Read the authoritative optimizer/scheduling scalar, not the arithmetic
    # mean of the per-particle data column.
    import starfile

    relion_model = starfile.read(str(K1_RELION_DIR / "run_it003_half1_model.star"))
    relion_pmax = float(relion_model["model_general"]["rlnAveragePmax"])
    pmax_diff = abs(float(pmax_traj[2]) - relion_pmax)

    payload = {
        "k1_coldstart_half1_corr_vs_relion_it003": h1_corr,
        "k1_coldstart_half2_corr_vs_relion_it003": h2_corr,
        "k1_coldstart_pmax_iter3_recovar": float(pmax_traj[2]),
        "k1_coldstart_pmax_iter3_relion": relion_pmax,
        "k1_coldstart_pmax_iter3_abs_diff": pmax_diff,
        "k1_coldstart_sigma_offset_trajectory": sigma_traj.tolist(),
        "k1_coldstart_sigma_offset_used_trajectory": sigma_used_traj.tolist(),
        "k1_coldstart_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("k1_coldstart", payload, output_dir=output_dir)
    logger.info("K=1 cold-start ledger: %s", ledger)

    print(file=sys.stderr, flush=True)
    print("=== K=1 cold-start parity (3-iter ab-initio vs RELION it003) ===", file=sys.stderr, flush=True)
    print(f"  half1_corr={h1_corr:.6f} half2_corr={h2_corr:.6f}", file=sys.stderr, flush=True)
    print(
        f"  pmax_iter3 recovar={pmax_traj[2]:.4f} relion={relion_pmax:.4f} diff={pmax_diff:.4g}",
        file=sys.stderr,
        flush=True,
    )
    print(f"  sigma_offset_trajectory={sigma_traj.tolist()}", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    # NEVER widen tolerance to make a test pass. Fix the code instead.
    # CLI-level --seed now reproduces RELION's per-iter SamplingPerturbInstance
    # stream for the fixture's --random_seed. Keep this autonomous cold-start
    # guard separate from replay because it still bootstraps model/noise/tau/
    # sigma state from raw inputs rather than injecting per-iter RELION state.
    assert h1_corr >= 0.999, f"K=1 cold-start half1 corr {h1_corr:.6f} below 0.999 vs RELION it003."
    assert h2_corr >= 0.999, f"K=1 cold-start half2 corr {h2_corr:.6f} below 0.999 vs RELION it003."
    # Pre-A.1 cold-start: iter-3 |ΔPmax| was ~22% (sigma_offset stuck at 10 Å).
    # Post-A.1: 5-12% depending on perturbation drift. Threshold 0.15 catches
    # full A.1 regression (would jump back to 22%).
    assert pmax_diff < 0.15, f"K=1 cold-start |ΔPmax| {pmax_diff:.4g} exceeds 0.15 vs RELION it003."
    # A.1 fix: iter-2 sigma_offset must drop below the 10 Å default. Pre-fix:
    # cold-start kept sigma_offset=10 Å through iter 8. Post-fix: ~2-4 Å at
    # iter 2, depending on iter-1 best-translation distribution.
    assert len(sigma_traj) >= 2 and sigma_traj[1] <= 5.0, (
        f"K=1 cold-start iter-2 sigma_offset {sigma_traj[1]:.3f} Å too large; A.1 fix may have regressed."
    )


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_k1_perturbreplay(tmp_path):
    """Run three K1 iterations with RELION perturbation/correction replay.

    The driver injects sampling perturbations, normalization and group scales;
    RECOVAR still runs E/M and updates sigma, tau and FSC. Compare iteration-3
    half maps and optimizer Pmax. This is distinct from autonomous cold start.
    """
    _assert_parity_ancestors_or_skip()
    _require_fixture(REFINE_SCRIPT, K1_FIXTURE_DIR, K1_RELION_DIR, K1_DATA_STAR)

    output_dir = tmp_path / "k1_perturbreplay"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(K1_FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--max_iter",
        "3",
        "--healpix_order",
        "3",
        "--offset_range",
        "3.0",
        "--offset_step",
        "1.0",
        "--adaptive_oversampling",
        "0",
        "--tau2_fudge",
        "1.0",
        "--perturb_factor",
        "0.5",
        "--perturb_replay_relion_dir",
        str(K1_RELION_DIR),
        "--init_resolution",
        "30.0",
        "--image_batch_size",
        "200",
        "--rotation_block_size",
        "2000",
        "--relion_half_sets",
        str(K1_FIXTURE_DIR / "particles_with_halfsets.star"),
    ]
    logger.info("K=1 perturb-replay cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    npz = np.load(output_dir / "refinement_results.npz")
    pmax_traj = np.asarray(npz["ave_Pmax_trajectory"], dtype=np.float64)

    from recovar.utils import helpers as _recovar_helpers

    rec_h1 = np.asarray(_recovar_helpers.load_mrc(str(output_dir / "final_half1.mrc")), dtype=np.float64)
    rec_h2 = np.asarray(_recovar_helpers.load_mrc(str(output_dir / "final_half2.mrc")), dtype=np.float64)
    rel_h1 = np.asarray(
        _recovar_helpers.load_relion_volume(str(K1_RELION_DIR / "run_it003_half1_class001.mrc")), dtype=np.float64
    )
    rel_h2 = np.asarray(
        _recovar_helpers.load_relion_volume(str(K1_RELION_DIR / "run_it003_half2_class001.mrc")), dtype=np.float64
    )
    h1_corr = _map_correlation(rec_h1, rel_h1)
    h2_corr = _map_correlation(rec_h2, rel_h2)

    import starfile

    relion_model = starfile.read(str(K1_RELION_DIR / "run_it003_half1_model.star"))
    relion_pmax = float(relion_model["model_general"]["rlnAveragePmax"])
    pmax_diff = abs(float(pmax_traj[2]) - relion_pmax)

    payload = {
        "k1_perturbreplay_half1_corr_vs_relion_it003": h1_corr,
        "k1_perturbreplay_half2_corr_vs_relion_it003": h2_corr,
        "k1_perturbreplay_pmax_iter3_recovar": float(pmax_traj[2]),
        "k1_perturbreplay_pmax_iter3_relion": relion_pmax,
        "k1_perturbreplay_pmax_iter3_abs_diff": pmax_diff,
        "k1_perturbreplay_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("k1_perturbreplay", payload, output_dir=output_dir)
    logger.info("K=1 perturb-replay ledger: %s", ledger)

    print(file=sys.stderr, flush=True)
    print("=== K=1 perturb-replay parity (3-iter ab-initio + RELION grid jitter sync) ===", file=sys.stderr, flush=True)
    print(f"  half1_corr={h1_corr:.6f} half2_corr={h2_corr:.6f}", file=sys.stderr, flush=True)
    print(
        f"  pmax_iter3 recovar={pmax_traj[2]:.4f} relion={relion_pmax:.4f} diff={pmax_diff:.4g}",
        file=sys.stderr,
        flush=True,
    )
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    # Tighter than coldstart since perturbation drift is eliminated. Observed
    # at HEAD: corrs ~0.992. The 0.99 floor catches any iter-1 / iter-2 path
    # regression that the looser cold-start bar would miss.
    assert h1_corr >= 0.99, f"K=1 perturb-replay half1 corr {h1_corr:.6f} below 0.99 vs RELION it003."
    assert h2_corr >= 0.99, f"K=1 perturb-replay half2 corr {h2_corr:.6f} below 0.99 vs RELION it003."
    assert pmax_diff < 0.05, f"K=1 perturb-replay |ΔPmax| {pmax_diff:.4g} exceeds 0.05 vs RELION it003."


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_kclass_coldstart(tmp_path):
    """Run K4 cold start at coarse order 2 with oversampling 1.

    Use the 5k/128 fixture, firstiter_cc and captured perturbation/correction
    replay, but derive initial noise/tau/sigma locally. Hungarian-match the four
    iteration-3 class maps. The dispatch oracle must use the same sampling grid.
    """
    _assert_parity_ancestors_or_skip()
    _require_fixture(REFINE_SCRIPT, K4_FIXTURE_DIR, K4_RELION_DIR, K4_DATA_STAR)

    output_dir = tmp_path / "kclass_coldstart"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(K4_FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--n_classes",
        "4",
        "--max_iter",
        "3",
        "--healpix_order",
        "2",
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--adaptive_oversampling",
        "1",
        "--tau2_fudge",
        "4.0",
        "--perturb_factor",
        "0.5",
        "--perturb_replay_relion_dir",
        str(K4_RELION_DIR),
        *_k4_dispatch_schedule_args(),
        "--firstiter_cc",
        "--init_resolution",
        "30.0",
        "--image_batch_size",
        "200",
        "--rotation_block_size",
        "2000",
    ]
    logger.info("K-class cold-start cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert not any(output_dir.glob("final_half*_class*.mrc")), "K>1 writer should not emit per-half class MRCs"

    # Hungarian-match recovar's per-class output to RELION's per-class it003.
    from scipy.optimize import linear_sum_assignment

    from recovar.utils import helpers as _recovar_helpers

    recov_classes = [
        np.asarray(_recovar_helpers.load_mrc(str(output_dir / f"final_class{c + 1:03d}.mrc")), dtype=np.float64)
        for c in range(4)
    ]
    relion_classes = [
        np.asarray(
            _recovar_helpers.load_relion_volume(str(K4_RELION_DIR / f"run_it003_class{c + 1:03d}.mrc")),
            dtype=np.float64,
        )
        for c in range(4)
    ]
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = _map_correlation(recov_classes[i], relion_classes[j])
    row, col = linear_sum_assignment(-M)
    matched = [float(M[i, j]) for i, j in zip(row, col)]
    mean_corr = float(np.mean(matched))
    worst_corr = float(np.min(matched))

    payload = {
        "kclass_coldstart_per_class_corrs_after_hungarian": matched,
        "kclass_coldstart_mean_corr": mean_corr,
        "kclass_coldstart_worst_class_corr": worst_corr,
        "kclass_coldstart_hungarian_assignment": [(int(i), int(j)) for i, j in zip(row, col)],
        "kclass_coldstart_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("kclass_coldstart", payload, output_dir=output_dir)
    logger.info("K-class cold-start ledger: %s", ledger)

    print(file=sys.stderr, flush=True)
    print("=== K=4 cold-start parity (3-iter ab-initio vs RELION it003) ===", file=sys.stderr, flush=True)
    print(f"  per-class corrs (after Hungarian): {matched}", file=sys.stderr, flush=True)
    print(f"  mean_corr={mean_corr:.6f}  worst_class_corr={worst_corr:.6f}", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    # NEVER widen tolerance to make a test pass. Fix the code instead.
    # Observed at 68c00297 with RELION-like sampling but no --relion_init_dir:
    # per-class [0.99757, 0.99885, 0.99924, 0.99939], mean 0.99876.
    # These floors lock in near-RELION cold-start parity without relying on
    # RELION's it000 model/noise/tau/sigma state.
    assert worst_corr >= 0.997, f"K-class cold-start worst per-class corr {worst_corr:.4f} below 0.997: {matched}"
    assert mean_corr >= 0.9985, f"K-class cold-start mean_corr {mean_corr:.4f} below 0.9985: {matched}"


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_kclass_nonadaptive_replay(tmp_path):
    """Exercise three K4 iterations at coarse order 1 without oversampling.

    Load RELION iteration-0 noise/tau/sigma and replay perturbations/corrections
    with firstiter_cc. Check all four matched maps and particle class assignments.
    RELION GPU rejects firstiter_cc without oversampling. Comparing against its
    oversampling-1 reference preserves cross-grid regression coverage, not
    strict matched-state parity.
    """
    _assert_parity_ancestors_or_skip()
    _require_fixture(REFINE_SCRIPT, K4_FIXTURE_DIR, K4_RELION_DIR, K4_DATA_STAR)
    relion_dir = K4_RELION_DIR

    output_dir = tmp_path / "kclass_strict"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(K4_FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--n_classes",
        "4",
        "--max_iter",
        "3",
        "--healpix_order",
        "1",
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--adaptive_oversampling",
        "0",
        "--perturb_factor",
        "0.5",
        "--perturb_replay_relion_dir",
        str(relion_dir),
        "--relion_init_dir",
        str(relion_dir),
        *_k4_dispatch_schedule_args(),
        "--firstiter_cc",
        "--init_resolution",
        "30.0",
        "--image_batch_size",
        "200",
        "--rotation_block_size",
        "2000",
    ]
    logger.info("K-class nonadaptive replay cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    from scipy.optimize import linear_sum_assignment

    from recovar.utils import helpers as _recovar_helpers

    recov_classes = [
        np.asarray(_recovar_helpers.load_mrc(str(output_dir / f"final_class{c + 1:03d}.mrc")), dtype=np.float64)
        for c in range(4)
    ]
    relion_classes = [
        np.asarray(
            _recovar_helpers.load_relion_volume(str(relion_dir / f"run_it003_class{c + 1:03d}.mrc")),
            dtype=np.float64,
        )
        for c in range(4)
    ]
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = _map_correlation(recov_classes[i], relion_classes[j])
    row, col = linear_sum_assignment(-M)
    matched = [float(M[i, j]) for i, j in zip(row, col)]
    mean_corr = float(np.mean(matched))
    worst_corr = float(np.min(matched))

    # iter-3 class match
    import re as _re

    import starfile as _starfile

    npz = np.load(output_dir / "refinement_results.npz", allow_pickle=True)
    rec_h1 = np.asarray(npz["class_assignments_half0"], dtype=np.int32)
    rec_h2 = np.asarray(npz["class_assignments_half1"], dtype=np.int32)
    h1_idx = np.asarray(npz["half1_indices"], dtype=np.int64)
    h2_idx = np.asarray(npz["half2_indices"], dtype=np.int64)
    rd = _starfile.read(str(relion_dir / "run_it003_data.star"))
    p = rd["particles"] if isinstance(rd, dict) and "particles" in rd else rd
    relion_class = np.asarray(p["rlnClassNumber"], dtype=np.int32) - 1
    relion_names = list(p["rlnImageName"])
    si = lambda n: int(_re.match(r"(\d+)@", n).group(1)) - 1  # noqa: E731
    rcb = {si(n): relion_class[i] for i, n in enumerate(relion_names)}
    rd2 = _starfile.read(str(K4_DATA_STAR))
    rp = rd2["particles"] if isinstance(rd2, dict) and "particles" in rd2 else rd2
    recov_names = list(rp["rlnImageName"])
    n_total = len(recov_names)
    relion_aligned = np.array([rcb.get(si(s), -1) for s in recov_names], dtype=np.int32)
    rec_class = np.full(n_total, -1, dtype=np.int32)
    rec_class[h1_idx] = rec_h1
    rec_class[h2_idx] = rec_h2
    mk = (rec_class >= 0) & (relion_aligned >= 0)
    M2 = np.zeros((4, 4), dtype=np.int64)
    for r, R in zip(rec_class[mk], relion_aligned[mk]):
        M2[r, R] += 1
    row2, col2 = linear_sum_assignment(-M2)
    iter3_match = float(M2[row2, col2].sum() / mk.sum()) if mk.sum() > 0 else 0.0

    payload = {
        "kclass_strict_per_class_corrs_after_hungarian": matched,
        "kclass_strict_mean_corr": mean_corr,
        "kclass_strict_worst_class_corr": worst_corr,
        "kclass_strict_iter3_class_match": iter3_match,
        "kclass_strict_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("kclass_strict", payload, output_dir=output_dir)
    logger.info("K-class nonadaptive replay ledger: %s", ledger)

    print(file=sys.stderr, flush=True)
    print(
        "=== K=4 nonadaptive replay (RELION os1 reference; cross-grid diagnostic) ===",
        file=sys.stderr,
        flush=True,
    )
    print(f"  per-class corrs (Hungarian): {matched}", file=sys.stderr, flush=True)
    print(f"  mean_corr={mean_corr:.6f}  worst_class_corr={worst_corr:.6f}", file=sys.stderr, flush=True)
    print(f"  iter-3 class match: {iter3_match:.4f}", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    # Strict bars — after the RELION Class3D M-step and post-reconstruction
    # mask/ini_high low-pass parity fixes, the os=0 K=4 fixture reached
    # per-class [0.9777, 0.9817, 0.9882, 0.9839], mean 0.9829. The
    # 0.982/0.975/0.84 floors lock against regressions in the source-backed
    # Class3D path without pretending that the remaining assignment gap is
    # solved.
    assert worst_corr >= 0.975, (
        f"K-class strict cold-start worst per-class corr {worst_corr:.4f} below 0.975: {matched}"
    )
    assert mean_corr >= 0.982, f"K-class strict cold-start mean_corr {mean_corr:.4f} below 0.982: {matched}"
    assert iter3_match >= 0.84, f"K-class strict cold-start iter-3 class match {iter3_match:.4f} below 0.84"


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_kclass_strict_oversample_coldstart(tmp_path):
    """Run three oracle-assisted K4 iterations at coarse order 1/oversampling 1.

    Load iteration-0 state and replay perturbations/corrections with firstiter_cc.
    All iterations use adaptive K-class execution; Hungarian-match all four final
    class maps against iteration 3 from the same dispatch-capable oracle.
    """
    _assert_parity_ancestors_or_skip()
    _require_fixture(REFINE_SCRIPT, K4_FIXTURE_DIR, K4_RELION_DIR, K4_DATA_STAR)
    relion_dir = K4_RELION_DIR

    output_dir = tmp_path / "kclass_strict_os1"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(K4_FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--n_classes",
        "4",
        "--max_iter",
        "3",
        "--healpix_order",
        "1",  # RELION coarse pass-1 order; fine pass 2 is order 2 via os=1
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--adaptive_oversampling",
        "1",  # 8× fine pose grid; matches matrix's run_k_class_parity 0.998 path
        "--perturb_factor",
        "0.5",
        "--perturb_replay_relion_dir",
        str(relion_dir),
        "--relion_init_dir",
        str(relion_dir),
        *_k4_dispatch_schedule_args(),
        "--firstiter_cc",
        "--init_resolution",
        "30.0",
        "--image_batch_size",
        "200",
        "--rotation_block_size",
        "2000",
    ]
    logger.info("K-class STRICT-PARITY oversample cold-start cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    from scipy.optimize import linear_sum_assignment

    from recovar.utils import helpers as _recovar_helpers

    recov_classes = [
        np.asarray(_recovar_helpers.load_mrc(str(output_dir / f"final_class{c + 1:03d}.mrc")), dtype=np.float64)
        for c in range(4)
    ]
    relion_classes = [
        np.asarray(
            _recovar_helpers.load_relion_volume(str(relion_dir / f"run_it003_class{c + 1:03d}.mrc")),
            dtype=np.float64,
        )
        for c in range(4)
    ]
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = _map_correlation(recov_classes[i], relion_classes[j])
    row, col = linear_sum_assignment(-M)
    matched = [float(M[i, j]) for i, j in zip(row, col)]
    mean_corr = float(np.mean(matched))
    worst_corr = float(np.min(matched))

    payload = {
        "kclass_strict_os1_per_class_corrs_after_hungarian": matched,
        "kclass_strict_os1_mean_corr": mean_corr,
        "kclass_strict_os1_worst_class_corr": worst_corr,
        "kclass_strict_os1_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("kclass_strict_os1", payload, output_dir=output_dir)
    logger.info("K-class strict-parity oversample ledger: %s", ledger)

    print(file=sys.stderr, flush=True)
    print(
        "=== K=4 STRICT-PARITY oversample=1 cold-start (relion_init + perturb_replay + firstiter_cc + adaptive engine) ===",
        file=sys.stderr,
        flush=True,
    )
    print(f"  per-class corrs (Hungarian): {matched}", file=sys.stderr, flush=True)
    print(f"  mean_corr={mean_corr:.6f}  worst_class_corr={worst_corr:.6f}", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    # Strict-os1 bars — after the RELION Class3D M-step and post-reconstruction
    # mask/ini_high low-pass parity fixes, the oversampled K=4 fixture reached
    # per-class [0.9898, 0.9950, 0.9973, 0.9987], mean 0.9952. The floors lock
    # against regressions in:
    #   * Class3D combined-accumulator single Wiener reconstruction
    #   * previous-Iref power-spectrum tau2, no gold-standard FSC
    #   * post-reconstruction solvent mask and firstiter ini_high low-pass
    #   * K-class adaptive oversampling and class rotation prior plumbing
    # Even when the os=0 strict_coldstart still passes.
    assert worst_corr >= 0.985, (
        f"K-class strict-os1 cold-start worst per-class corr {worst_corr:.4f} below 0.985: {matched}"
    )
    assert mean_corr >= 0.994, f"K-class strict-os1 cold-start mean_corr {mean_corr:.4f} below 0.994: {matched}"
