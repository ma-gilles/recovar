#!/usr/bin/env python
"""Stage data + launch cryoSPARC 3DVA jobs for all 16 (dataset, SNR) cases.

Pipeline per case:
  1. Stage data under /tigress/.../data_<case>/ via symlinks: particles.128.mrcs,
     particles.star, consensus_mean.mrc (from cov baseline), mask.mrc (from cov).
  2. Create cryoSPARC jobs in project P526 / workspace W2:
       import_particles  (alignments3D_exists=True)
       import_volumes    (consensus mean — for reference only; not wired into var_3D)
       import_volumes    (mask)
       var_3D            (K=10, filter_res=12A)
  3. Queue var_3D on lane "8hrs".
  4. Persist a case -> (import_uid, mask_uid, var_uid) map in cases.json.

Skip policy: if a completed var_3D with matching (particles=<imp>, mask=<mask>) and
params (K=10, filter_res=12.0) already exists, reuse its uid. Ribosembly SNR=1.0
reuses J28.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from cryosparc.tools import CryoSPARC

STAGING_ROOT = Path("/tigress/CRYOEM/singerlab/mg6942/_agent_scratch_3dva")
SWEEP_ROOT = Path("/scratch/gpfs/GILLES/mg6942/ppca_apples_sweep_20260415_113859")
OUT_ROOT = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/3dva_validation/cases")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = ["Ribosembly", "IgG-1D", "IgG-RL", "Tomotwin-100"]
SNRS = [0.01, 0.1, 1.0, 10.0]


def ntag_for(snr: float) -> str:
    return "n100000" if snr == 1.0 else "n50000"


def case_name(ds: str, snr: float) -> str:
    return f"{ds}_g128_{ntag_for(snr)}_snr{snr}_c0p00_z10_seed42"


def staging_dir(ds: str, snr: float) -> Path:
    return STAGING_ROOT / f"data_{ds}_snr{snr}"


def source_case_dir(ds: str, snr: float) -> Path:
    return SWEEP_ROOT / f"snr{snr}" / case_name(ds, snr)


def stage_case(ds: str, snr: float) -> Path:
    """Copy (not symlink) inputs into tigress — cryoSPARC rejects symlinks to /scratch."""
    import shutil

    sdir = staging_dir(ds, snr)
    sdir.mkdir(parents=True, exist_ok=True)
    src = source_case_dir(ds, snr)
    targets = {
        "particles.128.mrcs": src / "simulated_data" / "particles.128.mrcs",
        "particles.star": src / "simulated_data" / "particles.star",
        "consensus_mean.mrc": src / "cov" / "result" / "output" / "volumes" / "mean.mrc",
        "mask.mrc": src / "cov" / "result" / "output" / "volumes" / "mask.mrc",
    }
    for name, path in targets.items():
        dest = sdir / name
        if dest.is_symlink():
            dest.unlink()
        if dest.exists() and not dest.is_symlink():
            continue
        assert path.exists(), f"missing source {path}"
        shutil.copy(path, dest)
    return sdir


def main():
    cs = CryoSPARC(
        license="f83ffad2-1d98-11ed-996c-232421014a45",
        email="mg6942@princeton.edu",
        password="changeme",
        host="della-cryoem",
        base_port=39000,
    )
    proj = cs.find_project("P526")
    ws = proj.find_workspace("W2")

    cases_json = OUT_ROOT / "cases.json"
    cases: dict = json.loads(cases_json.read_text()) if cases_json.exists() else {}

    # Seed known completed case
    cases.setdefault(
        "Ribosembly_snr1.0",
        {
            "dataset": "Ribosembly",
            "snr": 1.0,
            "import_uid": "J19",
            "mask_uid": "J26",
            "var_uid": "J28",
            "status": "reused_existing",
        },
    )

    for ds in DATASETS:
        for snr in SNRS:
            key = f"{ds}_snr{snr}"
            if key in cases and cases[key].get("var_uid"):
                print(f"[skip] {key} already has var_uid={cases[key]['var_uid']}")
                continue
            sdir = stage_case(ds, snr)
            print(f"[stage] {key} -> {sdir}")

            # Import particles with alignments
            imp = ws.create_job("import_particles")
            for k, v in {
                "particle_meta_path": str(sdir / "particles.star"),
                "particle_blob_path": str(sdir),
                "psize_A": 4.25,
                "accel_kv": 300.0,
                "cs_mm": 2.7,
                "amp_contrast": 0.07,
                "blob_exists": True,
                "ctf_exists": True,
                "alignments3D_exists": True,
                "enable_validation": True,
                "ignore_splits": True,
            }.items():
                imp.set_param(k, v)
            imp.queue(lane="8hrs")
            print(f"  import queued: {imp.uid}")

            # Import mask
            mask = ws.create_job("import_volumes")
            mask.set_param("volume_blob_path", str(sdir / "mask.mrc"))
            mask.set_param("volume_out_name", "mask")
            mask.queue(lane="8hrs")
            print(f"  mask queued: {mask.uid}")

            cases[key] = {
                "dataset": ds,
                "snr": snr,
                "import_uid": imp.uid,
                "mask_uid": mask.uid,
                "var_uid": None,
                "status": "imports_queued",
                "staging": str(sdir),
            }
            cases_json.write_text(json.dumps(cases, indent=2))

    # Wait for imports to complete, then launch var_3D for each pending case
    print("\n=== waiting for imports ===")
    pending_imports = [k for k, v in cases.items() if v.get("status") == "imports_queued"]
    for _ in range(360):  # up to 2h
        still = []
        for key in pending_imports:
            info = cases[key]
            imp_ok = proj.find_job(info["import_uid"]).doc.get("status") == "completed"
            mask_ok = proj.find_job(info["mask_uid"]).doc.get("status") == "completed"
            if not (imp_ok and mask_ok):
                still.append(key)
        if not still:
            break
        print(f"  waiting: {len(still)} still importing")
        time.sleep(20)
        pending_imports = still

    # Launch var_3D for each case whose imports completed
    for key, info in cases.items():
        if info.get("var_uid"):
            continue
        imp_j = proj.find_job(info["import_uid"])
        mask_j = proj.find_job(info["mask_uid"])
        if imp_j.doc.get("status") != "completed" or mask_j.doc.get("status") != "completed":
            print(f"[skip-var] {key} imports not ready")
            continue
        va = ws.create_job(
            "var_3D",
            connections={
                "particles": (info["import_uid"], "imported_particles"),
                "mask": (info["mask_uid"], "imported_mask_1"),
            },
        )
        va.set_param("var_K", 10)
        va.set_param("var_filter_res", 12.0)
        va.queue(lane="8hrs")
        info["var_uid"] = va.uid
        info["status"] = "var_queued"
        cases_json.write_text(json.dumps(cases, indent=2))
        print(f"[var] {key} -> {va.uid}")

    print("\nDONE — run-state in", cases_json)


if __name__ == "__main__":
    main()
