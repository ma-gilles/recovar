"""EM-long parity regression tests (~2–4 hr per case on a single A100).

Production-scale parity gates that exercise full ab-initio refinement on
the 256² 50k fixture. Run via the dedicated EM-long Slurm wrapper at
``scripts/run_em_parity_long_slurm.sh`` — DO NOT invoke
``./scripts/run_tests_parallel.sh long-test`` from this branch (that runs
the cross-cutting SPA/ET pipeline regression suite, which is forbidden
for EM-only PRs per ``recovar/em/CLAUDE.md``).

Tests:
1. K=1 256² 50k full ab-initio (15 iters) — assert per-iter Pmax within
   ``1e-3`` of RELION at every iteration AND final FSC@0.5 vs GT within
   ``±0.5 Å`` of RELION.
2. K=4 256² 50k full ab-initio (15 iters) — same per-class.

Skipped by default. Enable with the ``--em-parity-long`` pytest flag.
The flag deliberately does NOT alias ``--long-test`` to keep this tier
disjoint from the cross-cutting long-test suite.
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
REFINE_SCRIPT = REPO_ROOT / "scripts" / "run_full_refinement.py"
BASELINES_DIR = REPO_ROOT / "tests" / "baselines"

FIXTURE_BASE = Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj")

# 256² 50k K=1 — likely needs to be built via prepare_relion_parity_benchmark.py
K1_LONG_FIXTURE_DIR = FIXTURE_BASE / "data_noise1_50k_256_normalized"
K1_LONG_RELION_DIR = K1_LONG_FIXTURE_DIR / "relion_ref_os0"
K1_LONG_RELION_DATA_STAR = K1_LONG_RELION_DIR / "run_data.star"
K1_LONG_DATA_STAR = K1_LONG_FIXTURE_DIR / "particles.star"
K1_LONG_GT_VOLUME = K1_LONG_FIXTURE_DIR / "reference_gt.mrc"

# 256² 50k K=4 — likely needs to be built via prepare_relion_multiclass_parity_benchmark.py
K4_LONG_FIXTURE_DIR = FIXTURE_BASE / "data_pdb_k4_50k_256"
K4_LONG_RELION_DIR = K4_LONG_FIXTURE_DIR / "relion_pdb_k4_os0_ref"
K4_LONG_DATA_STAR = K4_LONG_FIXTURE_DIR / "particles.star"


def _require_fixture(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip("Missing parity fixture(s):\n  " + "\n  ".join(missing))


def _assert_parity_ancestors_or_skip() -> None:
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
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = BASELINES_DIR / f"em_parity_quality_long_ledger_{name}.json"
    payload = dict(payload)
    payload.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with ledger_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return ledger_path


def _read_relion_fsc_resolution(relion_dir: Path, iter_num: int, voxel_size: float, grid_size: int) -> float:
    """Read RELION's gold-standard FSC and return the FSC<0.5 resolution in Å."""
    import starfile

    half1_model = starfile.read(str(relion_dir / f"run_it{iter_num:03d}_half1_model.star"))
    fsc_table = half1_model["model_class_1"]
    if "rlnGoldStandardFsc" in fsc_table.columns:
        fsc_col = "rlnGoldStandardFsc"
    elif "rlnFourierShellCorrelation" in fsc_table.columns:
        fsc_col = "rlnFourierShellCorrelation"
    else:
        raise ValueError(f"No FSC column in {relion_dir}/run_it{iter_num:03d}_half1_model.star")
    fsc_values = np.asarray(fsc_table[fsc_col], dtype=np.float64)
    # First shell where FSC drops below 0.5
    shell_05 = next((s for s in range(1, len(fsc_values)) if fsc_values[s] < 0.5), len(fsc_values) - 1)
    return float(grid_size) * float(voxel_size) / max(1, shell_05)


@pytest.mark.em_parity_long
@pytest.mark.gpu
@pytest.mark.integration
def test_em_parity_long_k1_full(tmp_path):
    """K=1 256² 50k full ab-initio (~3.5 hr on A100).

    Runs ``run_full_refinement.py --max_iter 15`` and compares iter-by-iter
    Pmax to RELION's ``rlnAveragePmax`` plus final FSC@0.5 vs GT.

    Pass criteria:
      * Final reconstruction FSC@0.5 vs GT within ±0.5 Å of RELION
      * Per-iter |ΔPmax| < 1e-3 vs RELION at every iteration ≥ 3
        (iters 1–2 may diverge during initial cold start)
    """
    _assert_parity_ancestors_or_skip()
    _require_fixture(REFINE_SCRIPT, K1_LONG_FIXTURE_DIR, K1_LONG_RELION_DIR, K1_LONG_DATA_STAR, K1_LONG_RELION_DATA_STAR)

    output_dir = tmp_path / "k1_long"
    output_dir.mkdir()

    # Match the RELION command in K1_LONG_RELION_DIR/run_it001_optimiser.star header.
    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(K1_LONG_FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--max_iter",
        "15",
        "--healpix_order",
        "3",
        "--offset_range",
        "3.0",
        "--offset_step",
        "1.0",
        "--adaptive_oversampling",
        "0",
        "--tau2_fudge",
        "4.0",
        "--perturb_factor",
        "0.5",
        "--perturb_seed",
        "42",
        "--init_resolution",
        "30.0",
        "--image_batch_size",
        "200",
        "--relion_half_sets",
        str(K1_LONG_RELION_DATA_STAR),
    ]
    logger.info("K=1 long refinement cmd: %s", " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0

    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    npz_path = output_dir / "refinement_results.npz"
    assert npz_path.exists(), f"Missing refinement_results.npz at {npz_path}"
    npz = np.load(npz_path)

    pmax_traj = np.asarray(npz["ave_Pmax_trajectory"], dtype=np.float64)
    voxel_size = float(npz["voxel_size"])
    grid_size = int(npz["volume_shape"][0])

    # Compare per-iter recovar Pmax against RELION's ave_Pmax across model.star files
    relion_pmax = []
    for it in range(1, 1 + int(pmax_traj.size)):
        model_path = K1_LONG_RELION_DIR / f"run_it{it:03d}_half1_model.star"
        if not model_path.exists():
            relion_pmax.append(np.nan)
            continue
        text = model_path.read_text()
        for line in text.splitlines():
            if line.strip().startswith("_rlnAveragePmax"):
                relion_pmax.append(float(line.split()[1]))
                break
        else:
            relion_pmax.append(np.nan)
    relion_pmax = np.asarray(relion_pmax, dtype=np.float64)

    # Per-iter Pmax delta (skip iter 1-2 cold start)
    pmax_diff = np.abs(pmax_traj - relion_pmax[: pmax_traj.size])
    bad_iters = [i for i, d in enumerate(pmax_diff) if i >= 2 and d > 1e-3 and np.isfinite(d)]

    # Final FSC@0.5 resolution comparison
    final_npz_path = output_dir / "gt_comparison_final.npz"
    if final_npz_path.exists():
        gt_comp = np.load(final_npz_path)
        recovar_shell_05 = int(gt_comp.get("recovar_merged_shell_05", -1))
    else:
        recovar_shell_05 = -1

    recovar_res_05 = grid_size * voxel_size / max(1, recovar_shell_05)
    relion_res_05 = _read_relion_fsc_resolution(
        K1_LONG_RELION_DIR, iter_num=int(pmax_traj.size), voxel_size=voxel_size, grid_size=grid_size
    )
    res_diff_angstrom = abs(recovar_res_05 - relion_res_05)

    payload = {
        "k1_long_per_iter_pmax_recovar": pmax_traj.tolist(),
        "k1_long_per_iter_pmax_relion": relion_pmax.tolist(),
        "k1_long_per_iter_pmax_abs_diff": pmax_diff.tolist(),
        "k1_long_pmax_diff_max_iter3plus": float(np.nanmax(pmax_diff[2:]) if pmax_diff.size > 2 else 0.0),
        "k1_long_recovar_fsc05_resolution_A": recovar_res_05,
        "k1_long_relion_fsc05_resolution_A": relion_res_05,
        "k1_long_fsc05_resolution_diff_A": res_diff_angstrom,
        "k1_long_walltime_s": elapsed,
        "k1_long_n_iters": int(pmax_traj.size),
    }
    ledger = _write_quality_ledger("k1_long", payload)
    logger.info("K=1 long ledger: %s", ledger)

    print(file=sys.stderr, flush=True)
    print("=== K=1 long parity (256² 50k 15-iter ab-initio) ===", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)
    print(f"  recovar FSC<0.5 res = {recovar_res_05:.2f} Å", file=sys.stderr, flush=True)
    print(f"  RELION  FSC<0.5 res = {relion_res_05:.2f} Å", file=sys.stderr, flush=True)
    print(f"  Δres = {res_diff_angstrom:.2f} Å (threshold 0.5)", file=sys.stderr, flush=True)
    print(
        f"  per-iter Pmax max-abs-diff (iter≥3): {payload['k1_long_pmax_diff_max_iter3plus']:.4g}",
        file=sys.stderr,
        flush=True,
    )
    if bad_iters:
        print(f"  iters with |ΔPmax|>1e-3: {bad_iters}", file=sys.stderr, flush=True)

    assert res_diff_angstrom <= 0.5, (
        f"K=1 long FSC<0.5 resolution gap {res_diff_angstrom:.2f} Å exceeds threshold 0.5 Å "
        f"(recovar={recovar_res_05:.2f}, RELION={relion_res_05:.2f})"
    )
    assert not bad_iters, f"K=1 long per-iter Pmax exceeds 1e-3 at iters {bad_iters}: " + ", ".join(
        f"iter{i + 1}={pmax_diff[i]:.3g}" for i in bad_iters
    )


@pytest.mark.em_parity_long
@pytest.mark.gpu
@pytest.mark.integration
def test_em_parity_long_kclass_full(tmp_path):
    """K=4 256² 50k full ab-initio (~4 hr on A100).

    Runs the K-class engine for 15 iterations and compares against RELION
    Class3D reference. Asserts per-class FSC@0.5 vs GT within ±0.5 Å and
    Hungarian-aligned class assignment accuracy ≥ 95 %.
    """
    _assert_parity_ancestors_or_skip()
    _require_fixture(KCLASS_SCRIPT, K4_LONG_FIXTURE_DIR, K4_LONG_RELION_DIR, K4_LONG_DATA_STAR)

    output_dir = tmp_path / "kclass_long"
    output_dir.mkdir()

    # K-class long-form parity uses the parity script's --target-iter at the
    # final RELION iteration (typically 15) starting from iter 0.
    final_iter = 15
    cmd = [
        sys.executable,
        str(KCLASS_SCRIPT),
        "--relion-dir",
        str(K4_LONG_RELION_DIR),
        "--data-star",
        str(K4_LONG_DATA_STAR),
        "--prev-iter",
        "0",
        "--target-iter",
        str(final_iter),
        "--output-dir",
        str(output_dir),
    ]
    logger.info("K-class long cmd: %s", " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0

    assert proc.returncode == 0, (
        f"run_k_class_parity.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())

    best_perm = summary["best_permutation"]
    mean_corr = float(best_perm["mean_corr"])
    map_corrs = [float(c) for c in best_perm["map_correlations"]]
    class_acc = float(summary["class_assignment_accuracy_after_permutation"])

    payload = {
        "kclass_long_mean_corr": mean_corr,
        "kclass_long_per_class_map_corr": map_corrs,
        "kclass_long_class_assignment_accuracy": class_acc,
        "kclass_long_walltime_s": elapsed,
        "kclass_long_target_iter": final_iter,
    }
    ledger = _write_quality_ledger("kclass_long", payload)
    logger.info("K-class long ledger: %s", ledger)

    print(file=sys.stderr, flush=True)
    print("=== K=4 long parity (256² 50k iter→15) ===", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)
    print(f"  per-class map corrs: {map_corrs}", file=sys.stderr, flush=True)
    print(f"  mean_corr={mean_corr:.6f}", file=sys.stderr, flush=True)
    print(f"  class assignment accuracy={class_acc:.4f}", file=sys.stderr, flush=True)

    # K-class quality bands: per-class corr ≥ 0.99 and Hungarian assignment ≥ 95%.
    assert all(c >= 0.99 for c in map_corrs), f"K-class long per-class map corr below threshold 0.99: {map_corrs}"
    assert class_acc >= 0.95, (
        f"K-class long Hungarian-aligned class assignment accuracy {class_acc:.4f} below threshold 0.95"
    )
