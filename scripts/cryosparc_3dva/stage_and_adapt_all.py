#!/usr/bin/env python
"""After launch_all_3dva.py — wait for var_3D jobs to complete, stage outputs,
then adapt each into the sweep result/ layout with OLS c_scale fit against GT.

Idempotent: re-running picks up any newly-completed var_3D and skips ones
already adapted.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/scratch/gpfs/GILLES/mg6942/_agent_scratch/3dva_validation/scripts")
sys.path.insert(0, "/scratch/gpfs/GILLES/mg6942")

import ppca_refit_subspace_em as pres
from adapt_3dva_to_result import adapt

CASES_JSON = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/3dva_validation/cases/cases.json")
STAGED_ROOT = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/3dva_validation/staged_outputs")
SWEEP = Path("/scratch/gpfs/GILLES/mg6942/ppca_apples_sweep_20260415_113859")
CSPARC_PROJECT_DIR = Path("/tigress/CRYOEM/singerlab/mg6942/CS-csparc-test")
SNR1_ROOT = Path("/scratch/gpfs/GILLES/mg6942/ppca_pipeline_compare_unified_20260409_091145")
HYBRID = Path("/home/mg6942/mytigress/cryobench2/benchmark_hybrid_prior")


def seed_sim_info_cache():
    pres._populate_sim_info_cache()
    for ds in ["Ribosembly", "IgG-1D", "IgG-RL", "Tomotwin-100"]:
        for n in [0.01, 0.1, 10.0]:
            p = HYBRID / f"{ds}_snr{n}" / "simulated_data" / "simulation_info.pkl"
            if p.exists():
                pres.SIM_INFO_CACHE[(ds, n, 128)] = str(p)
        p1 = SNR1_ROOT / f"{ds}_g128_n100000_snr1.0_c0p00_z10_seed42" / "simulated_data" / "simulation_info.pkl"
        if p1.exists():
            pres.SIM_INFO_CACHE[(ds, 1.0, 128)] = str(p1)


def ntag_for(snr: float) -> str:
    return "n100000" if snr == 1.0 else "n50000"


def case_name(ds: str, snr: float) -> str:
    return f"{ds}_g128_{ntag_for(snr)}_snr{snr}_c0p00_z10_seed42"


def stage_outputs(var_uid: str) -> Path:
    """Copy J<UID>_{map,component_0..9,particles}.{mrc,cs} out of cryoSPARC job dir."""
    job_dir = CSPARC_PROJECT_DIR / var_uid
    dst = STAGED_ROOT / var_uid
    dst.mkdir(parents=True, exist_ok=True)
    files = [f"{var_uid}_map.mrc", f"{var_uid}_particles.cs"]
    files += [f"{var_uid}_component_{k}.mrc" for k in range(10)]
    for fn in files:
        src = job_dir / fn
        dest = dst / fn
        if dest.exists():
            continue
        assert src.exists(), f"missing {src}"
        shutil.copy(src, dest)
    return dst


def job_status(var_uid: str) -> str:
    """Check status via job.json on filesystem (no cryosparc-tools needed)."""
    jp = CSPARC_PROJECT_DIR / var_uid / "job.json"
    if not jp.exists():
        return "missing"
    d = json.loads(jp.read_text())
    return d.get("status", "?")


def main():
    cases = json.loads(CASES_JSON.read_text())
    seed_sim_info_cache()

    results = {}
    for key, info in cases.items():
        ds = info["dataset"]
        snr = info["snr"]
        var_uid = info.get("var_uid")
        if not var_uid:
            print(f"[skip] {key}: no var_uid")
            continue
        status = job_status(var_uid)
        waited = 0
        while status in ("queued", "launched", "started", "running") and waited < 7200:
            print(f"[wait] {key} {var_uid} status={status}")
            time.sleep(30)
            waited += 30
            status = job_status(var_uid)
        if status != "completed":
            print(f"[fail] {key} {var_uid} status={status} — skipping")
            continue

        print(f"[stage] {key} {var_uid}")
        stage_dir = stage_outputs(var_uid)

        out_result = SWEEP / f"snr{snr}" / case_name(ds, snr) / "3dva" / "result"
        if (out_result / "model" / "params.pkl").exists():
            print(f"  already adapted → {out_result}")
        else:
            cov_result = SWEEP / f"snr{snr}" / case_name(ds, snr) / "cov" / "result"
            gt = pres._load_gt(ds, snr, 128)
            if gt is None:
                print(f"  [no-gt] {ds} σ²={snr} — adapting without c_scale")
                adapt(str(stage_dir), str(cov_result), str(out_result), var_uid, gt_pack=None)
            else:
                adapt(str(stage_dir), str(cov_result), str(out_result), var_uid, gt_pack=gt)

        # Score
        gt = pres._load_gt(ds, snr, 128)
        if gt is None:
            print(f"  [score-skip] no GT for {ds} σ²={snr}")
            continue
        sc = pres.score_one_result(str(out_result), gt, 10)
        if sc is None:
            print(f"  [score-fail] {key}")
            continue
        results[key] = {
            "dataset": ds,
            "noise_level": snr,
            "method": "3dva",
            "var_uid": var_uid,
            "pc_metric": float(sc.get("pc_metric", float("nan"))),
            "embed_metric": float(sc.get("embed_metric", float("nan"))),
            "cluster_metric": float(sc.get("cluster_metric", float("nan"))),
        }
        print(f"  [score] {key}: pc={sc['pc_metric']:.3f} em={sc['embed_metric']:.3f} cl={sc['cluster_metric']:.3f}")

    out_json = STAGED_ROOT / "scores_3dva.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print("\nwrote", out_json)


if __name__ == "__main__":
    main()
