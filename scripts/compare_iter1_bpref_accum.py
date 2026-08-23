#!/usr/bin/env python
"""Compare recovar's iter-1 backprojection accumulators vs RELION's downsampled-FSC ingredients.

Loads:
  * recovar's RECOVAR_BPREF_ACCUM_DUMP_DIR / recovar_bpref_accum_it001.npz
    (post-join Ft_y, Ft_ctf in centered Fourier order, full pf*N grid)
  * RELION's RECOVAR_MSTEP_DUMP_DIR / downsampled_avg_rank{01,02}_call0000.txt
    (post-getDownsampledAverage data: k, i, j, real, imag, weight)

Then runs recovar's compute_relion_fsc_from_backprojector internal downsample
on its accumulators, converts RECOVAR-native accumulator units to RELION-frame
units by default, and prints, per shell:
  * recovar's downsampled |avg|^2 sum, weight sum, FSC numerator/denom
  * RELION's same quantities from the text dump
  * The deltas

This pinpoints whether recovar's iter-1 backprojection scatter differs from
RELION's at any shell or at specific downsampled Fourier coordinates.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np


def _round_away(x):
    return np.where(x >= 0, np.floor(x + 0.5), -np.floor(-x + 0.5)).astype(np.int64)


def downsample_recovar_accumulator(Ft, Ft_ctf, volume_shape, padding_factor, r_max, accumulator_shape=None):
    """Mirror of compute_relion_fsc_from_backprojector's downsample step."""
    n = volume_shape[0]
    pf = padding_factor
    padded_shape = (
        tuple(int(v) for v in accumulator_shape)
        if accumulator_shape is not None
        else (n * pf, n * pf, n * pf)
    )
    full_size = int(np.prod(padded_shape))
    half_shape = (n * pf, n * pf, (n * pf) // 2 + 1)
    if accumulator_shape is not None:
        half_shape = (padded_shape[0], padded_shape[1], padded_shape[2] // 2 + 1)
    half_size = int(np.prod(half_shape))

    def _packed_half_to_full(arr_np):
        half_grid = arr_np.reshape(half_shape)
        n0, n1, n2 = padded_shape
        ic2 = n2 // 2
        if n2 % 2 == 0:
            packed_idx = np.concatenate([np.arange(ic2, n2, dtype=np.int64), np.asarray([0], dtype=np.int64)])
            redundant = np.arange(1, ic2, dtype=np.int64)
        else:
            packed_idx = np.arange(ic2, n2, dtype=np.int64)
            redundant = np.arange(0, ic2, dtype=np.int64)
        full_grid = np.zeros(padded_shape, dtype=half_grid.dtype)
        full_grid[:, :, packed_idx] = half_grid
        if redundant.size:
            partner_i0 = (n0 - (n0 % 2) - np.arange(n0, dtype=np.int64)) % n0
            partner_i1 = (n1 - (n1 % 2) - np.arange(n1, dtype=np.int64)) % n1
            conj_partner = np.conj(half_grid[partner_i0[:, None], partner_i1[None, :], :])
            source_cols = ic2 - redundant
            full_grid[:, :, redundant] = conj_partner[:, :, source_cols]
        return full_grid

    arr = np.asarray(Ft)
    if arr.size == full_size:
        data_padded = arr.reshape(padded_shape)
    elif arr.size == half_size:
        data_padded = _packed_half_to_full(arr)
    else:
        raise ValueError(f"shape mismatch: {arr.shape} vs full={full_size}, half={half_size}")
    arr_w = np.asarray(Ft_ctf)
    if arr_w.size == full_size:
        weight_padded = arr_w.reshape(padded_shape).real
    elif arr_w.size == half_size:
        weight_padded = _packed_half_to_full(arr_w).real
    else:
        raise ValueError("Ft_ctf shape mismatch")

    # axes = freq grid centered (range -N/2 .. N/2-1)
    def freq_axis(s):
        return np.fft.fftshift(np.fft.fftfreq(s, d=1.0)) * s  # integer indices centered

    axes = [freq_axis(s).astype(np.float64) for s in padded_shape]
    relion_z, relion_y, relion_x = np.meshgrid(axes[1], axes[2], axes[0], indexing="ij")
    dz = _round_away(relion_z / pf)
    dy = _round_away(relion_y / pf)
    dx = _round_away(relion_x / pf)
    data = np.transpose(data_padded, (1, 2, 0))
    weight = np.transpose(weight_padded, (1, 2, 0))

    half = n // 2
    max_shell = half if r_max is None else int(r_max)
    down_radius = max_shell + 1
    down_size = 2 * down_radius + 1
    down_xsize = down_size // 2 + 1

    valid = (
        (dz >= -down_radius)
        & (dz <= down_radius)
        & (dy >= -down_radius)
        & (dy <= down_radius)
        & (dx >= 0)
        & (dx < down_xsize)
    )
    labels = ((dz[valid] + down_radius) * down_size + (dy[valid] + down_radius)) * down_xsize + dx[valid]
    labels = labels.reshape(-1)
    minlength = down_size * down_size * down_xsize

    weight_flat = weight[valid].reshape(-1)
    data_flat = data[valid].reshape(-1)
    sum_weight = np.bincount(labels, weights=weight_flat, minlength=minlength)
    sum_real = np.bincount(labels, weights=data_flat.real, minlength=minlength)
    sum_imag = np.bincount(labels, weights=data_flat.imag, minlength=minlength)
    avg = (sum_real + 1j * sum_imag).reshape((down_size, down_size, down_xsize))
    sum_weight_3d = sum_weight.reshape((down_size, down_size, down_xsize))
    nz = sum_weight_3d > 0
    avg_div = np.where(nz, avg / np.where(nz, sum_weight_3d, 1.0), 0.0)
    return avg_div, sum_weight_3d, down_radius, down_size, down_xsize, max_shell


def shell_bin(avg, weight, down_radius, down_size, down_xsize, max_shell):
    z_axis = np.arange(-down_radius, down_radius + 1, dtype=np.float64)
    y_axis = np.arange(-down_radius, down_radius + 1, dtype=np.float64)
    x_axis = np.arange(0, down_xsize, dtype=np.float64)
    rz, ry, rx = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    radius = np.sqrt(rz * rz + ry * ry + rx * rx)
    shell = _round_away(radius)
    shell_count = max_shell + 1 + 1  # one extra for boundary
    shell_valid = radius <= float(max_shell)
    sl = shell[shell_valid].reshape(-1)
    av = avg[shell_valid].reshape(-1)
    we = weight[shell_valid].reshape(-1)
    p = np.bincount(sl, weights=np.abs(av) ** 2, minlength=shell_count)
    s = np.bincount(sl, weights=we, minlength=shell_count)
    return p, s


def load_relion_dump(path):
    """Parse RELION's downsampled_avg_rank?_call?.txt."""
    with open(path) as f:
        for line in f:
            if line.startswith("# rank"):
                tokens = line.split()
                ori_size = int(tokens[tokens.index("ori_size") + 1])
                r_max = int(tokens[tokens.index("r_max") + 1])
                pf = float(tokens[tokens.index("padding_factor") + 1])
            elif line.startswith("# xsize"):
                tokens = line.split()
                xsize = int(tokens[tokens.index("xsize") + 1])
                ysize = int(tokens[tokens.index("ysize") + 1])
                zsize = int(tokens[tokens.index("zsize") + 1])
                xinit = int(tokens[tokens.index("xinit") + 1])
                yinit = int(tokens[tokens.index("yinit") + 1])
                zinit = int(tokens[tokens.index("zinit") + 1])
                break
    data = np.loadtxt(path, comments="#")
    k = data[:, 0].astype(np.int64)
    i = data[:, 1].astype(np.int64)
    j = data[:, 2].astype(np.int64)
    real = data[:, 3]
    imag = data[:, 4]
    weight = data[:, 5]
    return dict(ori_size=ori_size, r_max=r_max, pf=pf, k=k, i=i, j=j, real=real, imag=imag, weight=weight)


def shell_bin_relion(dump):
    radius = np.sqrt(dump["k"] ** 2 + dump["i"] ** 2 + dump["j"] ** 2)
    shell = _round_away(radius)
    shell_valid = radius <= float(dump["r_max"])
    sl = shell[shell_valid]
    pwr = np.abs(dump["real"][shell_valid] + 1j * dump["imag"][shell_valid]) ** 2
    we = dump["weight"][shell_valid]
    n_shells = dump["r_max"] + 2
    p = np.bincount(sl, weights=pwr, minlength=n_shells)
    s = np.bincount(sl, weights=we, minlength=n_shells)
    return p, s


def _apply_recovar_frame(avg, weight, *, grid_size, recovar_frame):
    """Convert downsampled RECOVAR accumulators into the requested numeric frame."""

    if recovar_frame == "native":
        return avg, weight
    if recovar_frame != "relion":
        raise ValueError(f"unsupported recovar_frame={recovar_frame!r}")
    n2 = float(grid_size) ** 2
    n4 = float(grid_size) ** 4
    return avg / n2, weight * n4


def _relative_error(actual, expected):
    return np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-30)


def _summarize_errors(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return "n=0"
    return (
        f"n={values.size} median={np.median(values):.4e} mean={np.mean(values):.4e} "
        f"p95={np.percentile(values, 95):.4e} p99={np.percentile(values, 99):.4e} "
        f"max={np.max(values):.4e}"
    )


def coordinate_stats(avg, weight, down_radius, relion_dump):
    """Compare downsampled RECOVAR and RELION values at dumped RELION coordinates.

    RELION dumps coordinates as ``k, i, j``. The downsampled RECOVAR helper above
    stores those axes as ``i, k, j`` after converting from RECOVAR's centered
    Fourier array order.
    """

    k = relion_dump["k"].astype(np.int64)
    i = relion_dump["i"].astype(np.int64)
    j = relion_dump["j"].astype(np.int64)
    z_idx = i + int(down_radius)
    y_idx = k + int(down_radius)
    x_idx = j
    valid = (
        (z_idx >= 0)
        & (z_idx < avg.shape[0])
        & (y_idx >= 0)
        & (y_idx < avg.shape[1])
        & (x_idx >= 0)
        & (x_idx < avg.shape[2])
    )
    if not np.all(valid):
        z_idx = z_idx[valid]
        y_idx = y_idx[valid]
        x_idx = x_idx[valid]
    rec_avg = avg[z_idx, y_idx, x_idx]
    rec_weight = weight[z_idx, y_idx, x_idx]
    rel_avg = relion_dump["real"][valid] + 1j * relion_dump["imag"][valid]
    rel_weight = relion_dump["weight"][valid]
    avg_plus = _relative_error(rec_avg, rel_avg)
    avg_minus = _relative_error(-rec_avg, rel_avg)
    sign = -1 if np.median(avg_minus) < np.median(avg_plus) else 1
    avg_err = avg_minus if sign == -1 else avg_plus
    weight_err = _relative_error(rec_weight, rel_weight)
    return {
        "valid": int(np.count_nonzero(valid)),
        "total": int(valid.size),
        "avg_sign": sign,
        "avg_err": avg_err,
        "weight_err": weight_err,
    }


def scan_coordinate_mappings(avg, weight, down_radius, relion_dump, *, min_valid_fraction=0.95, top_n=8):
    """Rank axis/sign mappings for RELION ``k,i,j`` dump coordinates.

    This is a diagnostic guard against mistaking coordinate-frame drift for a
    real BPref accumulator mismatch.  The expected RECOVAR mapping is
    ``perm=(1, 0, 2), signs=(1, 1, 1)`` with an optional complex avg sign flip.
    """

    coords = np.stack(
        [
            relion_dump["k"].astype(np.int64),
            relion_dump["i"].astype(np.int64),
            relion_dump["j"].astype(np.int64),
        ],
        axis=1,
    )
    rel_avg_all = relion_dump["real"] + 1j * relion_dump["imag"]
    rel_weight_all = relion_dump["weight"]
    results = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            mapped = coords[:, perm] * np.asarray(signs, dtype=np.int64)
            z_idx = mapped[:, 0] + int(down_radius)
            y_idx = mapped[:, 1] + int(down_radius)
            x_idx = mapped[:, 2]
            valid = (
                (z_idx >= 0)
                & (z_idx < avg.shape[0])
                & (y_idx >= 0)
                & (y_idx < avg.shape[1])
                & (x_idx >= 0)
                & (x_idx < avg.shape[2])
            )
            valid_count = int(np.count_nonzero(valid))
            total = int(valid.size)
            if total and valid_count / total < float(min_valid_fraction):
                continue
            rec_avg = avg[z_idx[valid], y_idx[valid], x_idx[valid]]
            rec_weight = weight[z_idx[valid], y_idx[valid], x_idx[valid]]
            rel_avg = rel_avg_all[valid]
            rel_weight = rel_weight_all[valid]
            avg_plus = _relative_error(rec_avg, rel_avg)
            avg_minus = _relative_error(-rec_avg, rel_avg)
            avg_sign = -1 if np.median(avg_minus) < np.median(avg_plus) else 1
            avg_err = avg_minus if avg_sign == -1 else avg_plus
            weight_err = _relative_error(rec_weight, rel_weight)
            results.append(
                {
                    "perm": tuple(int(x) for x in perm),
                    "signs": tuple(int(x) for x in signs),
                    "avg_sign": int(avg_sign),
                    "valid": valid_count,
                    "total": total,
                    "avg_median": float(np.median(avg_err)),
                    "avg_mean": float(np.mean(avg_err)),
                    "weight_median": float(np.median(weight_err)),
                    "weight_mean": float(np.mean(weight_err)),
                }
            )
    results.sort(key=lambda item: (item["avg_median"], item["avg_mean"], item["weight_median"]))
    return results[: int(top_n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovar-npz", required=True, help="RECOVAR_BPREF_ACCUM_DUMP_DIR/recovar_bpref_accum_it001.npz")
    ap.add_argument("--relion-dir", required=True, help="RECOVAR_MSTEP_DUMP_DIR with downsampled_avg_rank?_call?.txt")
    ap.add_argument(
        "--relion-call",
        type=int,
        default=None,
        help="RELION call number to compare against (0 for iter-1, 1 for iter-2, ...). "
        "If unset, auto-detect from recovar npz r_max.",
    )
    ap.add_argument(
        "--recovar-frame",
        choices=("relion", "native"),
        default="relion",
        help="Numeric frame for printed RECOVAR values. 'relion' converts avg by 1/N^2 and weight by N^4.",
    )
    ap.add_argument(
        "--skip-coordinate-stats",
        action="store_true",
        help="Only print shell-binned summaries.",
    )
    ap.add_argument(
        "--scan-coordinate-mapping",
        action="store_true",
        help="Rank axis/sign coordinate mappings for RELION k,i,j dump rows.",
    )
    args = ap.parse_args()

    npz = np.load(args.recovar_npz)
    pf = int(npz["padding_factor"])
    cs = int(npz["current_size"])
    n = int(npz["grid_size"])
    vs = tuple(int(x) for x in npz["volume_shape"])
    accumulator_shape = (
        tuple(int(x) for x in npz["mstep_accumulator_shape"])
        if "mstep_accumulator_shape" in npz
        else None
    )
    r_max = cs // 2

    print(
        f"recovar accumulators: cs={cs}, padding_factor={pf}, grid_size={n}, "
        f"r_max={r_max}, recovar_frame={args.recovar_frame}, "
        f"accumulator_shape={accumulator_shape if accumulator_shape is not None else 'legacy_even'}"
    )

    # Auto-detect RELION call number by matching r_max if not provided.
    if args.relion_call is None:
        import re as _re

        relion_dir = Path(args.relion_dir)
        candidates = sorted(relion_dir.glob("downsampled_avg_rank01_call*.txt"))
        relion_call = 0
        for c in candidates:
            with open(c) as f:
                line = f.readline()
            m = _re.search(r"r_max\s+(\d+)", line)
            if m and int(m.group(1)) == r_max:
                m2 = _re.search(r"call\s+(\d+)", line)
                if m2:
                    relion_call = int(m2.group(1))
                    break
        print(f"Auto-detected RELION call={relion_call} (matched r_max={r_max})")
    else:
        relion_call = args.relion_call

    for k_half in (0, 1):
        Ft = npz[f"Ft_y_{k_half}"]
        W = npz[f"Ft_ctf_{k_half}"]
        avg, weight, dr, ds, dx, mr = downsample_recovar_accumulator(
            Ft,
            W,
            vs,
            pf,
            r_max,
            accumulator_shape=accumulator_shape,
        )
        avg, weight = _apply_recovar_frame(avg, weight, grid_size=n, recovar_frame=args.recovar_frame)
        p_rec, s_rec = shell_bin(avg, weight, dr, ds, dx, mr)

        rel_path = Path(args.relion_dir) / f"downsampled_avg_rank{k_half + 1:02d}_call{relion_call:04d}.txt"
        rel = load_relion_dump(str(rel_path))
        p_rel, s_rel = shell_bin_relion(rel)
        print(f"\n=== half {k_half + 1} ===")
        print(f"  recovar nshells={len(p_rec)}, RELION nshells={len(p_rel)}")
        n_show = min(len(p_rec), len(p_rel), 35)
        print("  shell  recovar |avg|^2_sum   RELION |avg|^2_sum   ratio_p   recovar w_sum   RELION w_sum   ratio_w")
        for sh in range(n_show):
            rp = p_rec[sh]
            rrp = p_rel[sh]
            sw = s_rec[sh]
            rsw = s_rel[sh]
            ratio_p = rp / max(rrp, 1e-30)
            ratio_w = sw / max(rsw, 1e-30)
            print(f"  {sh:5d}  {rp:.4e}  {rrp:.4e}  {ratio_p:7.4f}  {sw:.4e}  {rsw:.4e}  {ratio_w:7.4f}")
        if not args.skip_coordinate_stats:
            stats = coordinate_stats(avg, weight, dr, rel)
            print(
                "  coordinate stats: "
                f"valid={stats['valid']}/{stats['total']} avg_sign={stats['avg_sign']}"
            )
            print(f"    avg relerr:    {_summarize_errors(stats['avg_err'])}")
            print(f"    weight relerr: {_summarize_errors(stats['weight_err'])}")
            if args.scan_coordinate_mapping:
                print("  coordinate mapping scan:")
                for row in scan_coordinate_mappings(avg, weight, dr, rel):
                    print(
                        "    "
                        f"perm={row['perm']} signs={row['signs']} avg_sign={row['avg_sign']} "
                        f"valid={row['valid']}/{row['total']} "
                        f"avg_median={row['avg_median']:.4e} avg_mean={row['avg_mean']:.4e} "
                        f"weight_median={row['weight_median']:.4e}"
                    )


if __name__ == "__main__":
    main()
