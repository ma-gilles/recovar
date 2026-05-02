"""Element-wise per-pose diff² compare using RELION's C++ project_volume.

This bypasses recovar's relion_project_half (which is off by N from RELION's
/N³ FFT-norm) and uses the relion_bind C++ projection at machine precision.

If diff² then matches RELION's recovered diff², we know:
  - RELION's diff² formula = our formula
  - The bug in test_estep_bpref_forward_parity is a normalization mismatch
    between recovar's run_em projection and RELION's projection.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

DEFAULT_FIXTURE_DIR = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_run_small")
DEFAULT_DUMP = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_dump_small")
DEFAULT_OUT = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/p0_diff2_diag_relion_proj")


def read3(p):
    with open(p, "rb") as f:
        nz, ny, nx = struct.unpack("qqq", f.read(24))
        size_left = (Path(p).stat().st_size - 24) // (nz * ny * nx)
        dt = np.complex128 if size_left == 16 else np.float64
        return np.fromfile(f, dtype=dt, count=nz * ny * nx).reshape(nz, ny, nx)


def read_mweight(p):
    with open(p, "rb") as f:
        dims = struct.unpack("qqqqqq", f.read(48))
        n = int(np.prod(dims))
        data = np.fromfile(f, dtype=np.float64, count=n)
    return data.reshape(dims)


def read_meta(p):
    out = {}
    for line in p.read_text().strip().split("\n"):
        k, v = line.split("=")
        out[k] = v
    return out


def euler_to_R(rot_d, tilt_d, psi_d):
    rot = np.deg2rad(rot_d)
    tilt = np.deg2rad(tilt_d)
    psi = np.deg2rad(psi_d)
    ca, sa = np.cos(rot), np.sin(rot)
    cb, sb = np.cos(tilt), np.sin(tilt)
    cg, sg = np.cos(psi), np.sin(psi)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa
    return np.array(
        [
            [cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
            [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
            [sc, ss, cb],
        ]
    )


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    bb = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom == 0:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--particle", type=int, default=0)
    parser.add_argument("--ori-size", type=int, default=64)
    parser.add_argument(
        "--project-volume-scale",
        type=float,
        default=None,
        help="Scale applied to relion_bind project_volume output before scoring; defaults to ori_size.",
    )
    return parser.parse_args()


def main():
    import mrcfile
    from recovar.relion_bind._relion_bind_core import project_volume

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"p{args.particle}"
    project_volume_scale = float(args.ori_size if args.project_volume_scale is None else args.project_volume_scale)

    Fimg = read3(args.dump_dir / f"{prefix}_Fimg.bin")[0]
    Fctf = read3(args.dump_dir / f"{prefix}_local_Fctf.bin")[0]
    Minv = read3(args.dump_dir / f"{prefix}_local_Minvsigma2.bin")[0]
    eulers = read3(args.dump_dir / f"{prefix}_oversampled_eulers.bin")
    trans = read3(args.dump_dir / f"{prefix}_oversampled_translations.bin")
    Mweight = read_mweight(args.dump_dir / f"{prefix}_exp_Mweight.bin")
    meta = read_meta(args.dump_dir / f"{prefix}_estep_meta.txt")
    min_diff2 = float(meta["exp_min_diff2"])

    N = args.ori_size
    H_w, W_w = Fimg.shape  # 28, 15
    print(f"fixture_dir={args.fixture_dir}")
    print(f"dump_dir={args.dump_dir}")
    print(f"out_dir={args.out_dir}")
    print(f"particle={args.particle}, ori_size={N}, window={Fimg.shape}, project_volume_scale={project_volume_scale}")

    # RELION ky convention
    ky = np.array([i if i < W_w else i - H_w for i in range(H_w)], dtype=np.float64)
    kx = np.arange(W_w)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")

    # Pre-build all phase shifts
    trans_flat = trans[:, :, :2].reshape(-1, 2)  # (116, 2)
    n_otra = 4
    phases = np.exp(
        -2j * np.pi * (KX[None, :, :] * trans_flat[:, 0:1, None] + KY[None, :, :] * trans_flat[:, 1:2, None]) / N
    )

    # Volume (RELION-frame, raw mrc)
    vol_real = np.asarray(
        mrcfile.open(str(args.fixture_dir / "run_it000_class001.mrc"), permissive=True).data, dtype=np.float64
    )

    # Project orient 0 sub-rot 6 to verify RELION C++ shape
    R_test = euler_to_R(*eulers[0, 0])
    test_proj = project_volume(vol_real, R_test, ori_size=N, padding_factor=1, current_size=-1, do_gridding=True)
    print(f"RELION proj test: shape={test_proj.shape}, dtype={test_proj.dtype}, max={np.abs(test_proj).max():.4e}")
    fref_path = args.dump_dir / f"{prefix}_Fref_orient0.bin"
    if fref_path.exists():
        fref_dump = read3(fref_path)[0]
        test_w = project_volume_scale * test_proj[list(range(W_w)) + list(range(N - (H_w - W_w), N))][:, :W_w]
        if test_w.shape == fref_dump.shape:
            fref_cc = _corrcoef(np.stack([test_w.real.ravel(), test_w.imag.ravel()]).ravel(),
                                np.stack([fref_dump.real.ravel(), fref_dump.imag.ravel()]).ravel())
            fref_rel = float(np.linalg.norm(test_w - fref_dump) / max(np.linalg.norm(fref_dump), 1e-30))
            print(f"Dumped Fref orient0 compare: cc={fref_cc:+.6f}, rel_l2={fref_rel:.6e}")
        else:
            print(f"Dumped Fref orient0 shape differs: projected={test_w.shape}, dumped={fref_dump.shape}")

    # Find nonzero cells
    nonzero_mask = Mweight > 0
    idxs = np.argwhere(nonzero_mask[0])
    print(f"Nonzero cells: {len(idxs)}")

    # Project each unique (orient, iover_rot)
    unique_rotations = sorted({(int(idir) * 12 + int(ipsi), int(iover_rot)) for (idir, ipsi, _, iover_rot, _) in idxs})
    print(f"Unique rotations: {len(unique_rotations)}")

    rot_to_proj = {}
    for k, (orient_idx, iover_rot) in enumerate(unique_rotations):
        eu = eulers[orient_idx, iover_rot]
        R = euler_to_R(*eu)
        # current_size=-1 gives full N output; we need to window to (28, 15)
        proj_full = project_volume(vol_real, R, ori_size=N, padding_factor=1, current_size=-1, do_gridding=True)
        # Window (full N, half) → windowed (current_size=28, half=15) using RELION's windowFourierTransform.
        # For half-spec: the windowed image has shape (current_size, current_size//2+1).
        # RELION's windowing keeps low frequencies: y rows in [0, current_size//2] from top, and [-current_size//2+1, -1] from bottom.
        # Match the FFTW-natural ky pattern:
        keep_y = list(range(W_w)) + list(range(N - (H_w - W_w), N))  # [0..14] + [50..63]
        proj_w = project_volume_scale * proj_full[keep_y][:, :W_w]
        rot_to_proj[(orient_idx, iover_rot)] = proj_w
        if k < 3 or k % 50 == 0:
            print(f"  proj {k}/{len(unique_rotations)} max={np.abs(proj_w).max():.4e}")

    # Compute diff² per cell
    recovar_diff2 = np.zeros(len(idxs))
    relion_diff2_recovered = np.zeros(len(idxs))
    for k, (idir, ipsi, itrans, iover_rot, iover_trans) in enumerate(idxs):
        orient_idx = int(idir) * 12 + int(ipsi)
        proj = rot_to_proj[(orient_idx, int(iover_rot))]
        Frefctf = Fctf * proj  # +Fctf, no negation (using RELION-native sign throughout)
        flat_t = int(itrans) * n_otra + int(iover_trans)
        Fimg_t = Fimg * phases[flat_t]
        diff = Frefctf - Fimg_t
        recovar_diff2[k] = float(np.sum((diff.real**2 + diff.imag**2) * Minv * 0.5))
        relion_diff2_recovered[k] = -np.log(Mweight[0, idir, ipsi, itrans, iover_rot, iover_trans])

    print(f"\nRecovar diff² range: [{recovar_diff2.min():.4f}, {recovar_diff2.max():.4f}]")
    print(f"RELION recovered range: [{relion_diff2_recovered.min():.4f}, {relion_diff2_recovered.max():.4f}]")

    # Within-coarse-cell relative
    cell_groups = {}
    for k, (idir, ipsi, itrans, iover_rot, iover_trans) in enumerate(idxs):
        cell_groups.setdefault((int(idir), int(ipsi), int(itrans)), []).append(k)
    diffs_recovar = []
    diffs_relion = []
    for cell, ks in cell_groups.items():
        if len(ks) < 2:
            continue
        ref_o = recovar_diff2[ks].min()
        ref_r = relion_diff2_recovered[ks].min()
        for k in ks:
            diffs_recovar.append(recovar_diff2[k] - ref_o)
            diffs_relion.append(relion_diff2_recovered[k] - ref_r)
    diffs_recovar = np.asarray(diffs_recovar)
    diffs_relion = np.asarray(diffs_relion)
    if len(diffs_recovar) > 1:
        cc = _corrcoef(diffs_recovar, diffs_relion)
        slope = float(np.dot(diffs_recovar, diffs_relion) / max(np.dot(diffs_recovar, diffs_recovar), 1e-30))
        print(f"Within-cell Δdiff² CC: {cc:+.6f}")
        print(f"Within-cell Δdiff² slope (RELION/diagnostic): {slope:.6f}")
        print(f"Within-cell range ratio: {diffs_recovar.max():.4f} vs {diffs_relion.max():.4f}")
    else:
        cc = float("nan")
        slope = float("nan")

    # Detailed for one big cell
    big_key = max(cell_groups, key=lambda k: len(cell_groups[k]))
    ks = cell_groups[big_key]
    print(f"\nDetailed for coarse {big_key} ({len(ks)} children):")
    sub_o = recovar_diff2[ks]
    sub_r = relion_diff2_recovered[ks]
    sub_idx = idxs[ks]
    o = np.argsort(sub_o)
    for i in o[:10]:
        print(
            f"  iover=({sub_idx[i, 3]},{sub_idx[i, 4]}): ours={sub_o[i]:.4f}, rel={sub_r[i]:.4f} (Δours={sub_o[i] - sub_o.min():.4f}, Δrel={sub_r[i] - sub_r.min():.4f})"
        )

    np.savez(
        args.out_dir / f"{prefix}_diff2_relion_projector_compare.npz",
        idxs=idxs,
        diagnostic_diff2=recovar_diff2,
        relion_neg_log_weight=relion_diff2_recovered,
        within_cell_diagnostic_delta=diffs_recovar,
        within_cell_relion_delta=diffs_relion,
        min_diff2_relion=min_diff2,
    )
    summary = {
        "fixture_dir": str(args.fixture_dir),
        "dump_dir": str(args.dump_dir),
        "particle": args.particle,
        "ori_size": N,
        "project_volume_scale": project_volume_scale,
        "window_shape": list(Fimg.shape),
        "nonzero_cells": int(len(idxs)),
        "unique_rotations": int(len(unique_rotations)),
        "diagnostic_diff2_min": float(recovar_diff2.min()),
        "diagnostic_diff2_max": float(recovar_diff2.max()),
        "relion_neg_log_weight_min": float(relion_diff2_recovered.min()),
        "relion_neg_log_weight_max": float(relion_diff2_recovered.max()),
        "within_cell_delta_cc": float(cc),
        "within_cell_delta_slope_relion_over_diagnostic": float(slope),
        "within_cell_diagnostic_delta_max": float(diffs_recovar.max()) if len(diffs_recovar) else None,
        "within_cell_relion_delta_max": float(diffs_relion.max()) if len(diffs_relion) else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
