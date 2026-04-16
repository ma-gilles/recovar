"""Diagnose the remaining ~0.5 A FSC-to-GT parity gap between recovar and
RELION on the 5k normalized dataset.

Goal: attribute the gap to one (or a combination) of:
  1. Pose accuracy  -- recovar final poses vs RELION final poses
  2. Reconstruction numerics -- compare final volumes directly (FSC)
  3. Solvent flattening -- RELION --flatten_solvent / --zero_mask vs recovar
  4. GT alignment convention (real-space + RELION -transpose)

Inputs (paths are hardcoded, override via CLI flags):
  --relion_dir   RELION output dir (contains run_class001.mrc, run_it013_data.star)
  --recovar_dir  recovar run dir (contains final_merged.mrc, refinement_results.npz)
  --data_dir     Dataset dir (contains poses.pkl GT, reference_gt.mrc, etc.)

Run with:
    pixi run python scripts/diagnose_parity_gap.py
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

# Lazy jax/mrc imports so the script still starts under --help on CPU.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

VOXEL = 4.25
N = 128
PARTICLE_DIAMETER_ANG = 544.0


# ------------------------------------------------------------------ helpers
def load_mrc(path: str, relion_convention: bool = False) -> np.ndarray:
    import mrcfile

    with mrcfile.open(path, permissive=True) as m:
        vol = m.data.astype(np.float32).copy()
    if relion_convention:
        vol = -np.transpose(vol, (2, 1, 0))
    return vol


def load_relion_output_into_recovar_frame(path: str) -> np.ndarray:
    """RELION's run_*class*.mrc is stored axis-aligned with the RELION input
    frame, which is the **negative** of recovar's convention (empirical:
    pearson(recovar, -vol) = +0.946 while pearson(recovar, -transpose(vol))
    = +0.36 for the class001 final volume).  Use negation-only."""
    import mrcfile

    with mrcfile.open(path, permissive=True) as m:
        vol = m.data.astype(np.float32).copy()
    return -vol


def vol_to_ft(vol):
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu

    return np.asarray(ftu.get_dft3(jnp.asarray(vol))).reshape(-1)


def fsc_res(fsc, threshold: float) -> float:
    fsc = np.asarray(fsc)
    below = np.where(fsc < threshold)[0]
    if len(below) == 0 or below[0] == 0:
        return 999.0
    return float(N) * VOXEL / float(below[0])


def fsc_between(vol_a, vol_b):
    import jax.numpy as jnp

    from recovar.reconstruction.regularization import get_fsc_gpu

    fa, fb = vol_to_ft(vol_a), vol_to_ft(vol_b)
    return np.asarray(get_fsc_gpu(jnp.asarray(fa), jnp.asarray(fb), (N, N, N)))


# --------------------------------------------------------- RELION star reader
def parse_relion_star_particles(star_path: str):
    """Return (rot_zyz_deg, origin_angst, image_names) from a _data.star.

    RELION stores pose convention: rlnAngleRot/Tilt/Psi are the ZYZ Euler
    angles of R_inv (particle -> reference). rlnOriginXAngst/YAngst is the
    translation applied to the image to recenter.
    """
    rot, tilt, psi, ox, oy, name = [], [], [], [], [], []
    with open(star_path) as f:
        lines = f.readlines()

    # Find the data_particles loop
    i = 0
    while i < len(lines):
        if lines[i].strip() == "data_particles":
            break
        i += 1
    # skip to "loop_"
    while i < len(lines) and lines[i].strip() != "loop_":
        i += 1
    i += 1
    # parse column headers
    col_map = {}
    col_idx = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("_rln"):
            tag = line.split()[0][1:]  # strip leading _
            col_map[tag] = col_idx
            col_idx += 1
            i += 1
        else:
            break
    # parse data rows
    idx_rot = col_map["rlnAngleRot"]
    idx_tilt = col_map["rlnAngleTilt"]
    idx_psi = col_map["rlnAnglePsi"]
    idx_ox = col_map["rlnOriginXAngst"]
    idx_oy = col_map["rlnOriginYAngst"]
    idx_name = col_map["rlnImageName"]

    for row in lines[i:]:
        parts = row.split()
        if len(parts) < col_idx:
            continue
        rot.append(float(parts[idx_rot]))
        tilt.append(float(parts[idx_tilt]))
        psi.append(float(parts[idx_psi]))
        ox.append(float(parts[idx_ox]))
        oy.append(float(parts[idx_oy]))
        name.append(parts[idx_name])

    rot_zyz_deg = np.stack([np.asarray(rot), np.asarray(tilt), np.asarray(psi)], axis=1)
    origin_A = np.stack([np.asarray(ox), np.asarray(oy)], axis=1)
    return rot_zyz_deg, origin_A, name


def relion_zyz_to_matrix(rot_zyz_deg):
    """RELION convention: R = Rz(psi) Ry(tilt) Rz(rot) (particle->reference).
    Returns (N, 3, 3) matrices in the SAME convention as recovar (Euler_matrices
    of scipy ZYZ intrinsic)."""
    from scipy.spatial.transform import Rotation as R

    return R.from_euler("ZYZ", rot_zyz_deg, degrees=True).as_matrix()


def angle_between_rotations_deg(R1, R2):
    """R1, R2: (N,3,3). Returns per-particle angle diff in degrees."""
    # dR = R1 R2^T. angle = acos((tr(dR)-1)/2)
    dR = np.einsum("nij,nkj->nik", R1, R2)
    tr = np.einsum("nii->n", dR)
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


# ----------------------------------------------------- solvent/flatten check
def check_flatten_solvent_in_refine():
    """Search refine.py for any reference-masking equivalent to
    RELION's --flatten_solvent. Returns a short descriptor."""
    refine_path = REPO_ROOT / "recovar/em/dense_single_volume/refine.py"
    text = refine_path.read_text()
    hits = {}
    for key in (
        "flatten_solvent",
        "solvent_mask",
        "zero_solvent",
        "apply_solvent",
        "zero_mask",
        "soft_mask_outside_map",
    ):
        hits[key] = len(re.findall(r"\b" + key + r"\b", text))

    # Also check mask defaults used in post_process
    sph = re.findall(r"use_spherical_mask\s*=\s*(True|False)", text)
    return {"refine_hits": hits, "use_spherical_mask_settings": sph}


# ----------------------------------------------------------- main diagnostics
def diagnose(args):
    relion_dir = Path(args.relion_dir)
    recovar_dir = Path(args.recovar_dir)
    data_dir = Path(args.data_dir)

    print("=" * 78)
    print("SECTION 1. Volume-level comparison (recovar vs RELION vs GT)")
    print("=" * 78)

    gt = load_mrc(str(data_dir / "reference_gt.mrc"), relion_convention=False)
    gt_relion = load_mrc(str(data_dir / "reference_gt_relion.mrc"), relion_convention=True)

    # IMPORTANT: RELION's run_class001.mrc is stored with NEGATED values vs
    # the recovar frame (empirical pearson with recovar: +0.946 for -vol vs
    # +0.36 for -transpose(vol)). The legacy helper reference_gt_relion.mrc
    # really does need the -transpose because it was written via
    # write_relion_mrc; but RELION's own output does NOT.
    relion_final = load_relion_output_into_recovar_frame(str(relion_dir / "run_class001.mrc"))
    recovar_final = load_mrc(str(recovar_dir / "final_merged.mrc"), relion_convention=False)

    # relion_final now lives in recovar frame -> compare to recovar-frame GT.
    fsc_relion_gt = fsc_between(relion_final, gt)
    fsc_recovar_gt = fsc_between(recovar_final, gt)
    fsc_recovar_gtrelion = fsc_between(recovar_final, gt_relion)
    # recovar vs RELION directly -- need the *same* frame. recovar final is in
    # recovar (no-transpose) frame; relion_final we loaded with -transpose to
    # recovar frame, so both now live in recovar space.
    fsc_recovar_relion = fsc_between(recovar_final, relion_final)

    print(
        f"RELION  vs GT               @0.143 = {fsc_res(fsc_relion_gt, 0.143):.3f} A   @0.5 = {fsc_res(fsc_relion_gt, 0.5):.3f} A"
    )
    print(
        f"recovar vs GT               @0.143 = {fsc_res(fsc_recovar_gt, 0.143):.3f} A   @0.5 = {fsc_res(fsc_recovar_gt, 0.5):.3f} A"
    )
    print(
        f"recovar vs GT_relion        @0.143 = {fsc_res(fsc_recovar_gtrelion, 0.143):.3f} A   @0.5 = {fsc_res(fsc_recovar_gtrelion, 0.5):.3f} A"
    )
    print(
        f"recovar vs RELION (xcorr)   @0.143 = {fsc_res(fsc_recovar_relion, 0.143):.3f} A   @0.5 = {fsc_res(fsc_recovar_relion, 0.5):.3f} A"
    )
    n_show = 45
    print(f"\nFSC(recovar, RELION) shells[0:{n_show}]:")
    print(np.array2string(fsc_recovar_relion[:n_show], precision=3, suppress_small=True))
    print(f"\nmean FSC recovar<->RELION  shells 0..20 : {fsc_recovar_relion[:20].mean():.4f}")
    print(f"mean FSC recovar<->RELION  shells 20..40: {fsc_recovar_relion[20:40].mean():.4f}")

    # Also report the real-space correlation (rules out global scale/offset).
    def norm_xcorr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))

    print(f"\nreal-space Pearson correlation (recovar, RELION) = {norm_xcorr(recovar_final, relion_final):.5f}")
    print(f"real-space Pearson correlation (recovar, GT)     = {norm_xcorr(recovar_final, gt):.5f}")
    print(f"real-space Pearson correlation (RELION,  GT)        = {norm_xcorr(relion_final, gt):.5f}")

    print()
    print("=" * 78)
    print("SECTION 2. Pose comparison  --  RELION iter13 vs GT  (pose of RELION alone)")
    print("=" * 78)
    # We can't compare recovar per-particle poses: refinement_results.npz does
    # not save them (only hard_assignment indices into a not-saved local grid).
    # So report RELION-vs-GT only, as an upper bound on "how sharp is RELION".
    relion_zyz, relion_origin_A, relion_names = parse_relion_star_particles(str(relion_dir / "run_it013_data.star"))
    print(f"RELION star: {len(relion_names)} particles, rot_zyz shape={relion_zyz.shape}")

    # Load GT poses
    gt_rots, gt_trans = pickle.load(open(str(data_dir / "poses.pkl"), "rb"))
    print(f"GT poses: rotations={gt_rots.shape}, translations={gt_trans.shape}")

    relion_rots = relion_zyz_to_matrix(relion_zyz)

    # IMPORTANT:  RELION and recovar/cryodrgn use INVERSE conventions for R.
    # cryodrgn/recovar GT stores rotation so that R v_img = v_ref ; RELION
    # stores R^-1 in the star. So compare gt_rots vs relion_rots.T (or the
    # inverse). We try both orientations and report the better one.
    # NOTE: simulation_info.pkl also has raw angles -- for noise1 they are
    # drawn uniformly. What matters is CONSISTENCY, which we test directly.
    # Particle rows in run_it013_data.star are STRING-sorted by image name
    # ('1@...', '10@...', '100@...' ...), not numerical. Re-index GT into
    # that same order so we compare the same particle on both sides.
    star_idx = np.array([int(n.split("@")[0]) - 1 for n in relion_names])
    gt_rots_star = gt_rots[star_idx]
    gt_trans_star = gt_trans[star_idx]

    for label, R_try in [
        ("direct    (R_relion =  R_gt)", relion_rots),
        ("transpose (R_relion =  R_gt^T)", relion_rots.transpose(0, 2, 1)),
    ]:
        ang = angle_between_rotations_deg(R_try, gt_rots_star)
        print(f"  {label}: median={np.median(ang):.4f} deg, p95={np.percentile(ang, 95):.4f}, mean={ang.mean():.4f}")

    # RELION origin (in A). Convert to pixels.
    trans_px_relion = relion_origin_A / VOXEL
    for label, t_try in [
        ("direct  (t_rel =  t_gt)", trans_px_relion),
        ("negated (t_rel = -t_gt)", -trans_px_relion),
    ]:
        d = np.linalg.norm(t_try - gt_trans_star, axis=1)
        print(f"  trans {label}: median={np.median(d):.4f} px, p95={np.percentile(d, 95):.4f}, mean={d.mean():.4f}")

    print(
        "\nNOTE: recovar's refinement_results.npz does NOT save per-particle\n"
        "poses (only integer 'hard_assignments' indexing into a LOCAL search\n"
        "grid that is itself not saved). To compare recovar vs RELION poses\n"
        "particle-by-particle, run_full_refinement.py needs to dump\n"
        "experiment_datasets[k].rotations / .translations to the npz."
    )

    print()
    print("=" * 78)
    print("SECTION 3. Flatten-solvent equivalence audit")
    print("=" * 78)
    info = check_flatten_solvent_in_refine()
    print("Occurrences in recovar/em/dense_single_volume/refine.py:")
    for k, v in info["refine_hits"].items():
        print(f"   {k:26s}: {v}")
    print(f"use_spherical_mask settings found: {info['use_spherical_mask_settings']}")

    # Now compute what each "spherical mask" actually erases. recovar's
    # soft_mask_outside_map defaults radius=N/2=64, cosine_width=3 -> only
    # clips the 3 corner voxels. RELION --flatten_solvent zeros voxels OUTSIDE
    # particle_diameter/(2*voxel) = 544/(2*4.25) = 64 px. So at
    # particle_diameter=544A and N=128, both masks are the same radius (64).
    # The difference is that RELION does it EVERY iter on the projected
    # reference, whereas recovar applies it only to the reconstructed volume
    # via post_process_from_filter_v2.
    r_mask = PARTICLE_DIAMETER_ANG / (2.0 * VOXEL)
    print(f"\nRELION particle_diameter={PARTICLE_DIAMETER_ANG} A -> radius={r_mask:.2f} px")
    print(
        f"recovar soft_mask_outside_map default radius={N // 2} px (no cosine_width compensation for particle_diameter)"
    )

    print()
    print("=" * 78)
    print("SECTION 4. Direct volume diff (masked to particle region)")
    print("=" * 78)
    # Build a particle-diameter sphere mask centered at (N/2, N/2, N/2)
    zz, yy, xx = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    rr = np.sqrt((xx - N / 2) ** 2 + (yy - N / 2) ** 2 + (zz - N / 2) ** 2)
    sphere_mask = (rr < r_mask).astype(np.float32)

    recovar_masked = recovar_final * sphere_mask
    relion_masked = relion_final * sphere_mask
    gt_masked = gt * sphere_mask

    print(
        f"inside-particle std:   recovar={recovar_masked.std():.4e}  RELION={relion_masked.std():.4e}  GT={gt.std():.4e}"
    )
    outside = 1.0 - sphere_mask
    print(
        f"outside-particle std:  recovar={(recovar_final * outside).std():.4e}  RELION={(relion_final * outside).std():.4e}  GT={(gt * outside).std():.4e}"
    )
    print(
        f"outside-particle mean-abs:  recovar={np.abs(recovar_final * outside).mean():.4e}  RELION={np.abs(relion_final * outside).mean():.4e}  GT={np.abs(gt * outside).mean():.4e}"
    )

    # If RELION flatten_solvent is the cause, RELION's outside-particle
    # power should be ~0 and recovar's should be noticeably larger.
    # If they are similar, flatten_solvent is NOT the cause.

    print()
    print("=" * 78)
    print("SECTION 5. Summary")
    print("=" * 78)
    print(f"FSC-to-GT(recovar)            @0.143 = {fsc_res(fsc_recovar_gt, 0.143):.2f} A")
    print(f"FSC-to-GT(RELION)             @0.143 = {fsc_res(fsc_relion_gt, 0.143):.2f} A")
    print(f"FSC(recovar, RELION)          @0.143 = {fsc_res(fsc_recovar_relion, 0.143):.2f} A")
    print(f"Pearson(recovar, RELION)             = {norm_xcorr(recovar_final, relion_final):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--relion_dir", default="/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0"
    )
    ap.add_argument(
        "--recovar_dir", default="/scratch/gpfs/GILLES/mg6942/recovar_parity_experiments/runs/smoke_5k_pf2_full_6942683"
    )
    ap.add_argument("--data_dir", default="/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized")
    args = ap.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
