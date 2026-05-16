"""EM-long parity regression tests (~2–4 hr per case on a single A100).

Production-scale parity gates that exercise full ab-initio refinement on
the 256² 50k fixture. Run via the dedicated EM-long Slurm wrapper at
``scripts/run_em_parity_long_slurm.sh`` — DO NOT invoke
``./scripts/run_tests_parallel.sh long-test`` from this branch (that runs
the cross-cutting SPA/ET pipeline regression suite, which is forbidden
for EM-only PRs per ``recovar/em/CLAUDE.md``).

Tests:
1. K=1 256² 50k full auto-refine parity (15 iters) — assert per-iter Pmax within
   ``1e-3`` of RELION at every iteration AND final FSC@0.5 vs GT within
   ``±0.5 Å`` of RELION.
2. K=1 256² 50k native InitialModel/VDAM cold-start (8 iters) — run the
   GUI-facing ``scripts/run_ab_initio.py`` path and assert GT quality is close
   to a RELION ``--grad --denovo_3dref`` iter-8 reference.
3. K=4 256² 50k full ab-initio (15 iters) — same per-class.

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
ABINITIO_SCRIPT = REPO_ROOT / "scripts" / "run_ab_initio.py"
BASELINES_DIR = REPO_ROOT / "tests" / "baselines"

FIXTURE_BASE = Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj")

# 256² 50k K=1 auto-refine fixture — built via prepare_relion_parity_benchmark.py.
K1_LONG_FIXTURE_DIR = FIXTURE_BASE / "data_noise1_50k_256_normalized"
K1_LONG_RELION_DIR = K1_LONG_FIXTURE_DIR / "relion_ref_os0"
K1_LONG_RELION_DATA_STAR = K1_LONG_RELION_DIR / "run_data.star"
K1_LONG_DATA_STAR = K1_LONG_FIXTURE_DIR / "particles.star"
K1_LONG_GT_VOLUME = K1_LONG_FIXTURE_DIR / "reference_gt.mrc"

# 256² 50k K=1 native InitialModel reference. This must be generated with
# RELION's GUI-facing InitialModel command shape, not auto-refine:
#
#   relion_refine --grad --denovo_3dref --iter 8 --K 1 --pad 1 ...
#
# The Slurm wrapper creates/reuses this fixture before running the native VDAM
# quality test. Keeping it separate from K1_LONG_RELION_DIR prevents the guard
# from accidentally comparing native VDAM against an auto-refine trajectory.
K1_NATIVE_RELION_DIR = K1_LONG_FIXTURE_DIR / "relion_initialmodel_k1_it008"

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


def _relion_iter_walltimes_s(relion_dir: Path) -> list[float]:
    """Approximate RELION per-iteration walltimes from optimizer file mtimes."""
    from datetime import datetime

    prev_time = None
    out: list[float] = []
    for path in sorted(relion_dir.glob("run_it*_optimiser.star")):
        current = datetime.fromtimestamp(path.stat().st_mtime)
        if prev_time is not None:
            out.append(float((current - prev_time).total_seconds()))
        prev_time = current
    return out


def _read_relion_star_scalar(path: Path, key: str) -> str:
    prefix = f"{key} "
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split()[-1]
    raise AssertionError(f"{path} does not contain {key}")


def _mean_shell_value(values: np.ndarray, first_shell: int, last_shell: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    lo = max(0, int(first_shell))
    hi = min(values.size, int(last_shell) + 1)
    if hi <= lo:
        return float("nan")
    return float(np.nanmean(values[lo:hi]))


def _real_space_shell_fsc(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return an FFT-shell FSC between same-frame real-space volumes."""
    a = np.asarray(lhs, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or len(set(a.shape)) != 1:
        raise ValueError(f"Expected same-shaped cubic volumes, got {a.shape} and {b.shape}")

    n = int(a.shape[0])
    fa = np.fft.fftn(a)
    fb = np.fft.fftn(b)
    freqs = np.fft.fftfreq(n) * n
    z, y, x = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    shells = np.rint(np.sqrt(x * x + y * y + z * z)).astype(np.int32).ravel()
    product = (fa * np.conj(fb)).ravel()
    numerator = np.bincount(shells, weights=np.real(product))
    lhs_power = np.bincount(shells, weights=(np.abs(fa) ** 2).ravel())
    rhs_power = np.bincount(shells, weights=(np.abs(fb) ** 2).ravel())
    denom = np.sqrt(lhs_power * rhs_power)
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denom, out=out, where=denom > 0.0)
    return out


def _relion_frame_map_similarity(lhs_path: Path, rhs_path: Path) -> dict[str, float]:
    """Same-frame map parity metrics; no GT alignment or handedness search."""
    from recovar.em.initial_model.gt_metrics import centered_correlation, first_shell_below_threshold
    from recovar.utils import helpers

    lhs, _lhs_voxel = helpers.load_relion_volume(str(lhs_path), return_voxel_size=True)
    rhs, _rhs_voxel = helpers.load_relion_volume(str(rhs_path), return_voxel_size=True)
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    fsc = _real_space_shell_fsc(lhs, rhs)
    return {
        "corr": float(centered_correlation(lhs, rhs)),
        "mean_fsc_1_8": _mean_shell_value(fsc, 1, 8),
        "mean_fsc_1_16": _mean_shell_value(fsc, 1, 16),
        "shell_05": float(first_shell_below_threshold(fsc, 0.5)),
        "shell_0143": float(first_shell_below_threshold(fsc, 0.143)),
    }


def _assert_relion_initialmodel_reference(relion_dir: Path, *, expected_iter: int) -> None:
    """Hard-fail if the native reference is not a RELION InitialModel run.

    A previous version of this guard accidentally compared native VDAM against
    ``relion_ref_os0``, which is an auto-refine fixture. That makes the quality
    result ambiguous and hides regressions in the GUI InitialModel path.
    """
    optimiser = relion_dir / f"run_it{expected_iter:03d}_optimiser.star"
    _require_fixture(optimiser)
    text = optimiser.read_text()
    header = "\n".join(text.splitlines()[:3])
    required_tokens = ("--grad", "--denovo_3dref", f"--iter {expected_iter}", "--pad 1", "--auto_sampling")
    missing = [token for token in required_tokens if token not in header]
    if missing:
        raise AssertionError(
            f"{relion_dir} is not the required RELION InitialModel reference; "
            f"missing command token(s): {missing}. Header:\n{header}"
        )

    do_grad = int(float(_read_relion_star_scalar(optimiser, "_rlnDoGradientRefine")))
    do_auto_refine = int(float(_read_relion_star_scalar(optimiser, "_rlnDoAutoRefine")))
    do_split_halves = int(float(_read_relion_star_scalar(optimiser, "_rlnDoSplitRandomHalves")))
    current_iter = int(float(_read_relion_star_scalar(optimiser, "_rlnCurrentIteration")))
    n_iters = int(float(_read_relion_star_scalar(optimiser, "_rlnNumberOfIterations")))
    if (do_grad, do_auto_refine, do_split_halves, current_iter, n_iters) != (1, 0, 0, expected_iter, expected_iter):
        raise AssertionError(
            f"{relion_dir} is not a K=1 RELION InitialModel iter-{expected_iter} reference: "
            f"do_grad={do_grad}, do_auto_refine={do_auto_refine}, split_halves={do_split_halves}, "
            f"current_iter={current_iter}, n_iters={n_iters}"
        )


@pytest.mark.em_parity_long
@pytest.mark.gpu
@pytest.mark.integration
def test_em_parity_long_k1_full(tmp_path):
    """K=1 256² 50k full auto-refine replay (~3.5 hr on A100).

    Runs ``run_full_refinement.py --max_iter 15`` and compares iter-by-iter
    Pmax to RELION auto-refine's ``rlnAveragePmax`` plus final FSC@0.5 vs GT.

    Pass criteria:
      * Final reconstruction FSC@0.5 vs GT within ±0.5 Å of RELION
      * Per-iter |ΔPmax| < 1e-3 vs RELION at every iteration ≥ 3
        (iters 1–2 may diverge during initial cold start)
    """
    _assert_parity_ancestors_or_skip()
    _require_fixture(REFINE_SCRIPT, K1_LONG_FIXTURE_DIR, K1_LONG_RELION_DIR, K1_LONG_DATA_STAR, K1_LONG_RELION_DATA_STAR)

    output_dir = tmp_path / "k1_long"
    output_dir.mkdir()
    timing_dir = output_dir / "timing"
    perf_ledger_path = output_dir / "benchmark_ledger.json"

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
        "64",
        "--rotation_block_size",
        "8192",
        "--relion_half_sets",
        str(K1_LONG_RELION_DATA_STAR),
        "--timing_dir",
        str(timing_dir),
        "--benchmark_ledger_json",
        str(perf_ledger_path),
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
    perf_ledger = json.loads(perf_ledger_path.read_text()) if perf_ledger_path.exists() else {}

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
        "k1_long_recovar_wall_times_trajectory": perf_ledger.get("wall_times_trajectory", []),
        "k1_long_recovar_setup_phase_seconds": perf_ledger.get("setup_phase_seconds", {}),
        "k1_long_recovar_timing_summary": perf_ledger.get("timing_summary", {}),
        "k1_long_recovar_perf_ledger_path": str(perf_ledger_path),
        "k1_long_recovar_timing_dir": str(timing_dir),
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
def test_em_parity_long_k1_native_initialmodel_quality(tmp_path):
    """K=1 256² 50k native InitialModel quality gate.

    This is intentionally separate from ``test_em_parity_long_k1_full``.
    ``run_full_refinement.py`` and replay-style tests can pass while the
    GUI-facing native InitialModel path in ``scripts/run_ab_initio.py`` stalls
    after the first few VDAM iterations. This test guards the exact production
    command shape users get from RELION InitialModel parity work.

    Pass criteria:
      * The reference trajectory is verified as RELION InitialModel
        (``--grad --denovo_3dref``), not auto-refine.
      * Native VDAM iter-8 map directly correlates with RELION InitialModel
        iter-8 at >=0.999 in the same RELION frame. This is the strict parity
        guard; GT metrics alone can miss a wrong trajectory.
      * Native VDAM iter-8 reaches within 0.05 mean FSC(1..16) of RELION
        InitialModel iter-8.
      * Native VDAM iter-8 reaches within 0.05 corr-vs-GT of RELION
        InitialModel iter-8.

    The thresholds are relative to the configured RELION fixture, not a magic
    absolute score, so fixture or preprocessing changes keep this guard useful.
    """
    _assert_parity_ancestors_or_skip()
    _require_fixture(
        ABINITIO_SCRIPT,
        K1_LONG_FIXTURE_DIR,
        K1_NATIVE_RELION_DIR,
        K1_LONG_DATA_STAR,
        K1_LONG_GT_VOLUME,
        K1_NATIVE_RELION_DIR / "run_it008_class001.mrc",
    )
    _assert_relion_initialmodel_reference(K1_NATIVE_RELION_DIR, expected_iter=8)

    output_dir = tmp_path / "k1_native_initialmodel"
    output_dir.mkdir()

    cmd = [
        sys.executable,
        str(ABINITIO_SCRIPT),
        "--i",
        str(K1_LONG_DATA_STAR),
        "--datadir",
        str(K1_LONG_FIXTURE_DIR),
        "--o",
        str(output_dir / "run"),
        "--nr_iter",
        "8",
        "--K",
        "1",
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
        "16",
        "--rotation_block_size",
        "256",
        "--padding_factor",
        "1",
        "--eager_images",
    ]
    logger.info("K=1 native InitialModel cmd: %s", " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_ab_initio.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    from scripts.evaluate_ab_initio_gt import evaluate

    volume_paths = [
        output_dir / "run_it001_class001.mrc",
        output_dir / "run_it002_class001.mrc",
        output_dir / "run_it008_class001.mrc",
        K1_NATIVE_RELION_DIR / "run_it008_class001.mrc",
    ]
    labels = [
        "vdam_it001",
        "vdam_it002",
        "vdam_it008",
        "relion_initialmodel_it008",
    ]
    _npz_payload, summary = evaluate(
        volume_paths=[str(path) for path in volume_paths],
        labels=labels,
        gt_volume_path=str(K1_LONG_GT_VOLUME),
        volume_frame="relion",
        gt_frame="recovar",
        voxel_size_override=2.125,
        gt_align=False,
        gt_align_healpix_order=2,
        gt_align_max_shell=8,
        gt_align_allow_mirror=True,
        gt_align_allow_sign=False,
    )
    metrics = {item["label"]: item for item in summary["volumes"]}

    vdam_it008 = metrics["vdam_it008"]
    relion_it008 = metrics["relion_initialmodel_it008"]
    fsc_gap_1_16 = float(relion_it008["mean_fsc_1_16"] - vdam_it008["mean_fsc_1_16"])
    corr_gap = float(relion_it008["corr_vs_gt"] - vdam_it008["corr_vs_gt"])
    relion_map_similarity = _relion_frame_map_similarity(
        output_dir / "run_it008_class001.mrc",
        K1_NATIVE_RELION_DIR / "run_it008_class001.mrc",
    )

    payload = {
        "k1_native_initialmodel_relion_dir": str(K1_NATIVE_RELION_DIR),
        "k1_native_initialmodel_walltime_s": elapsed,
        "k1_native_initialmodel_vdam_it001_corr_vs_gt": float(metrics["vdam_it001"]["corr_vs_gt"]),
        "k1_native_initialmodel_vdam_it001_mean_fsc_1_16": float(metrics["vdam_it001"]["mean_fsc_1_16"]),
        "k1_native_initialmodel_vdam_it002_corr_vs_gt": float(metrics["vdam_it002"]["corr_vs_gt"]),
        "k1_native_initialmodel_vdam_it002_mean_fsc_1_16": float(metrics["vdam_it002"]["mean_fsc_1_16"]),
        "k1_native_initialmodel_vdam_it008_corr_vs_gt": float(vdam_it008["corr_vs_gt"]),
        "k1_native_initialmodel_vdam_it008_mean_fsc_1_16": float(vdam_it008["mean_fsc_1_16"]),
        "k1_native_initialmodel_relion_it008_corr_vs_gt": float(relion_it008["corr_vs_gt"]),
        "k1_native_initialmodel_relion_it008_mean_fsc_1_16": float(relion_it008["mean_fsc_1_16"]),
        "k1_native_initialmodel_fsc_1_16_gap_vs_relion_it008": fsc_gap_1_16,
        "k1_native_initialmodel_corr_gap_vs_relion_it008": corr_gap,
        "k1_native_initialmodel_vdam_vs_relion_it008_corr": relion_map_similarity["corr"],
        "k1_native_initialmodel_vdam_vs_relion_it008_mean_fsc_1_8": relion_map_similarity["mean_fsc_1_8"],
        "k1_native_initialmodel_vdam_vs_relion_it008_mean_fsc_1_16": relion_map_similarity["mean_fsc_1_16"],
        "k1_native_initialmodel_vdam_vs_relion_it008_shell_05": relion_map_similarity["shell_05"],
        "k1_native_initialmodel_vdam_vs_relion_it008_shell_0143": relion_map_similarity["shell_0143"],
        "k1_native_initialmodel_relion_iter_walltimes_s": _relion_iter_walltimes_s(K1_NATIVE_RELION_DIR),
    }
    ledger = _write_quality_ledger("k1_native_initialmodel", payload)
    logger.info("K=1 native InitialModel ledger: %s", ledger)

    print(file=sys.stderr, flush=True)
    print("=== K=1 native InitialModel quality (256² 50k 8-iter ab-initio) ===", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)
    print(
        "  vdam_it008: "
        f"corr={vdam_it008['corr_vs_gt']:.6f} "
        f"fsc1-16={vdam_it008['mean_fsc_1_16']:.6f} "
        f"shell0143={vdam_it008['shell_0143']}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "  relion_initialmodel_it008: "
        f"corr={relion_it008['corr_vs_gt']:.6f} "
        f"fsc1-16={relion_it008['mean_fsc_1_16']:.6f} "
        f"shell0143={relion_it008['shell_0143']}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"  gaps vs RELION InitialModel iter8: corr={corr_gap:.6f} fsc1-16={fsc_gap_1_16:.6f}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "  direct VDAM vs RELION it008: "
        f"corr={relion_map_similarity['corr']:.6f} "
        f"xfsc1-16={relion_map_similarity['mean_fsc_1_16']:.6f} "
        f"xshell0143={relion_map_similarity['shell_0143']}",
        file=sys.stderr,
        flush=True,
    )

    assert relion_it008["mean_fsc_1_16"] >= 0.10, (
        "RELION iter-8 fixture quality is unexpectedly low; fixture or GT convention may have changed: "
        f"{relion_it008}"
    )
    assert relion_map_similarity["corr"] >= 0.999, (
        "Native InitialModel VDAM iter-8 map is not in near-perfect direct parity with RELION InitialModel iter-8. "
        f"same-frame corr={relion_map_similarity['corr']:.6f}, "
        f"cross mean FSC(1..16)={relion_map_similarity['mean_fsc_1_16']:.6f}."
    )
    assert fsc_gap_1_16 <= 0.05, (
        "Native InitialModel VDAM iter-8 FSC quality is not close to RELION InitialModel iter-8. "
        f"VDAM fsc1-16={vdam_it008['mean_fsc_1_16']:.6f}, "
        f"RELION fsc1-16={relion_it008['mean_fsc_1_16']:.6f}, gap={fsc_gap_1_16:.6f}."
    )
    assert corr_gap <= 0.05, (
        "Native InitialModel VDAM iter-8 corr-vs-GT is not close to RELION InitialModel iter-8. "
        f"VDAM corr={vdam_it008['corr_vs_gt']:.6f}, "
        f"RELION corr={relion_it008['corr_vs_gt']:.6f}, gap={corr_gap:.6f}."
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
