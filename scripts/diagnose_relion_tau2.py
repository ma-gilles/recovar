#!/usr/bin/env python
"""Compare RELION tau2 shells against recovar refinement outputs.

Primary use case:
  1. Load RELION ``run_itNNN_half*_model.star`` tau2/FSC/SSNR.
  2. Convert RELION tau2 to recovar units via ``N^4``.
  3. Compare shell-by-shell against recovar's saved tau2 arrays.
  4. Optionally recompute an oracle-FSC tau2 using recovar's saved sigma2.
"""

import argparse
from pathlib import Path

import numpy as np
import starfile


def _load_relion_half_model(relion_dir: Path, relion_iter: int, half: int):
    path = relion_dir / f"run_it{relion_iter:03d}_half{half}_model.star"
    model = starfile.read(str(path))
    general = model["model_general"]
    class_df = model["model_class_1"]
    return general, class_df


def _get_npz_array(npz, prefix: str, iteration: int):
    key = f"{prefix}_{iteration:03d}"
    if key not in npz.files:
        return None
    return np.asarray(npz[key], dtype=np.float64)


def _format_float(val):
    if val is None or not np.isfinite(val):
        return "      —"
    if abs(val) >= 1e6 or (0 < abs(val) < 1e-3):
        return f"{val:9.3e}"
    return f"{val:9.4f}"


def _truncate_pair(a, b):
    if a is None or b is None:
        return a, b
    n = min(len(a), len(b))
    return a[:n], b[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relion_dir", required=True)
    parser.add_argument("--relion_iter", type=int, default=4)
    parser.add_argument("--half", type=int, default=1, choices=[1, 2])
    parser.add_argument("--recovar_npz", required=True)
    parser.add_argument("--recovar_iter", type=int, default=0)
    parser.add_argument("--shells", type=int, default=20)
    parser.add_argument(
        "--oracle_relion_fsc",
        action="store_true",
        help="Recompute tau2 using RELION FSC with recovar's saved sigma2 shells.",
    )
    args = parser.parse_args()

    relion_dir = Path(args.relion_dir)
    npz = np.load(args.recovar_npz, allow_pickle=False)

    general, class_df = _load_relion_half_model(relion_dir, args.relion_iter, args.half)
    n = int(general["rlnOriginalImageSize"])
    n4 = n**4
    tau2_fudge = float(general.get("rlnTau2FudgeFactor", 1.0))

    relion_tau2 = np.asarray(class_df["rlnReferenceTau2"], dtype=np.float64) * n4
    relion_fsc = np.asarray(class_df["rlnGoldStandardFsc"], dtype=np.float64)
    relion_ssnr = np.asarray(class_df["rlnSsnrMap"], dtype=np.float64)
    relion_sigma2 = np.where(relion_ssnr > 0, relion_tau2 / relion_ssnr, np.nan)
    relion_data_star = relion_dir / f"run_it{args.relion_iter:03d}_data.star"
    relion_half_count = None
    if relion_data_star.exists():
        relion_data = starfile.read(str(relion_data_star))
        relion_df = relion_data["particles"] if isinstance(relion_data, dict) else relion_data
        if "rlnRandomSubset" in relion_df.columns:
            relion_half_count = int(np.sum(np.asarray(relion_df["rlnRandomSubset"]) == args.half))

    recovar_tau2 = _get_npz_array(npz, "tau2_radial_iter", args.recovar_iter)
    recovar_fsc = _get_npz_array(npz, "tau2_fsc_used_iter", args.recovar_iter)
    if recovar_fsc is None:
        recovar_fsc = _get_npz_array(npz, "fsc_iter", args.recovar_iter)
    recovar_ssnr = _get_npz_array(npz, "tau2_ssnr_iter", args.recovar_iter)
    if recovar_ssnr is None:
        recovar_ssnr = _get_npz_array(npz, "data_vs_prior_iter", args.recovar_iter)
    recovar_sigma2 = _get_npz_array(npz, "tau2_sigma2_iter", args.recovar_iter)
    recovar_tau2, recovar_ssnr = _truncate_pair(recovar_tau2, recovar_ssnr)
    if recovar_sigma2 is None and recovar_tau2 is not None and recovar_ssnr is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            recovar_sigma2 = np.where(recovar_ssnr > 0, recovar_tau2 / recovar_ssnr, np.nan)

    recovar_avg_weight = _get_npz_array(npz, "tau2_avg_weight_iter", args.recovar_iter)
    recovar_half_count_key = f"n_half{args.half}_particles"
    recovar_half_count = int(npz[recovar_half_count_key]) if recovar_half_count_key in npz.files else None
    oracle_tau2 = None
    if args.oracle_relion_fsc and recovar_sigma2 is not None:
        clipped_fsc = np.clip(relion_fsc, 1e-3, 1.0 - 1e-3)
        oracle_ssnr = clipped_fsc / (1.0 - clipped_fsc) * tau2_fudge
        oracle_tau2 = oracle_ssnr[: len(recovar_sigma2)] * recovar_sigma2

    if recovar_tau2 is None:
        raise SystemExit(f"Missing tau2_radial_iter_{args.recovar_iter:03d} in {args.recovar_npz}")

    n_shells = min(
        args.shells,
        len(relion_tau2),
        len(recovar_tau2),
        len(relion_fsc),
        len(recovar_fsc) if recovar_fsc is not None else len(relion_fsc),
        len(relion_ssnr),
        len(recovar_ssnr) if recovar_ssnr is not None else len(relion_ssnr),
    )

    print(
        f"RELION iter {args.relion_iter} half {args.half} vs recovar iter {args.recovar_iter}: "
        f"N={n}, N^4={n4}, tau2_fudge={tau2_fudge:.3f}"
    )
    if relion_half_count is not None or recovar_half_count is not None:
        particle_scale = (
            float(relion_half_count) / float(recovar_half_count)
            if relion_half_count is not None and recovar_half_count not in (None, 0)
            else np.nan
        )
        print(
            f"half-set particles: RELION={relion_half_count if relion_half_count is not None else '—'} "
            f"recovar={recovar_half_count if recovar_half_count is not None else '—'} "
            f"(RELION/recovar scale hint={particle_scale:.4f})"
        )
    print(
        "shell   rel_tau2    rec_tau2   ratio   rel_fsc   rec_fsc  rel_ssnr  rec_ssnr  rel_sigma2 rec_sigma2 oracle_tau2"
    )
    print("-" * 118)
    for shell in range(n_shells):
        rec_tau2 = recovar_tau2[shell]
        rel_tau2 = relion_tau2[shell]
        ratio = rec_tau2 / rel_tau2 if abs(rel_tau2) > 1e-30 else np.nan
        rec_f = recovar_fsc[shell] if recovar_fsc is not None and shell < len(recovar_fsc) else np.nan
        rec_s = recovar_ssnr[shell] if recovar_ssnr is not None and shell < len(recovar_ssnr) else np.nan
        rec_sig = recovar_sigma2[shell] if recovar_sigma2 is not None and shell < len(recovar_sigma2) else np.nan
        oracle = oracle_tau2[shell] if oracle_tau2 is not None and shell < len(oracle_tau2) else np.nan
        print(
            f"{shell:5d}  {_format_float(rel_tau2)} {_format_float(rec_tau2)} "
            f"{_format_float(ratio)} {_format_float(relion_fsc[shell])} {_format_float(rec_f)} "
            f"{_format_float(relion_ssnr[shell])} {_format_float(rec_s)} "
            f"{_format_float(relion_sigma2[shell])} {_format_float(rec_sig)} {_format_float(oracle)}"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        tau2_ratio = recovar_tau2[:n_shells] / relion_tau2[:n_shells]
        sigma2_ratio = (
            recovar_sigma2[:n_shells] / relion_sigma2[:n_shells]
            if recovar_sigma2 is not None
            else np.full(n_shells, np.nan)
        )
        ssnr_ratio = (
            recovar_ssnr[:n_shells] / relion_ssnr[:n_shells]
            if recovar_ssnr is not None
            else np.full(n_shells, np.nan)
        )

    print("-" * 118)
    print(
        "median ratios: "
        f"tau2={np.nanmedian(tau2_ratio):.4f} "
        f"sigma2={np.nanmedian(sigma2_ratio):.4f} "
        f"ssnr={np.nanmedian(ssnr_ratio):.4f}"
    )
    if relion_half_count is not None and recovar_half_count not in (None, 0):
        particle_scale = float(relion_half_count) / float(recovar_half_count)
        print(
            f"particle-count hint: expected sigma2/tau2 scale from subset size alone = {particle_scale:.4f}"
        )
    if recovar_avg_weight is not None:
        print(
            "avg_weight shells[0:5]:",
            np.array2string(recovar_avg_weight[: min(5, len(recovar_avg_weight))], precision=6),
        )


if __name__ == "__main__":
    main()
