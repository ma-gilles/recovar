#!/usr/bin/env python
"""Postprocess saved multi-iter parity volumes against GT and RELION.

This script reads the intermediate per-iteration volumes saved by
``scripts/run_multi_iter_parity.py`` and computes, for every available
iteration:

- recovar regularized half/merged FSC and correlation vs GT
- recovar unregularized half/merged FSC and correlation vs GT
- RELION half/merged FSC and correlation vs GT
- recovar-vs-RELION real-space correlations for the regularized maps

Results are written back into the recovar output directory as
``gt_comparison_iterNNN.npz`` files so the diff tooling can surface them.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovar_dir", required=True, help="run_multi_iter_parity output directory")
    parser.add_argument("--relion_dir", required=True, help="RELION run directory with run_itNNN_* outputs")
    parser.add_argument(
        "--relion_start_iter",
        type=int,
        required=True,
        help="RELION iteration used as the initial state for recovar (e.g. 3)",
    )
    parser.add_argument(
        "--gt_volume",
        type=str,
        default=None,
        help="Optional recovar-frame GT MRC. Defaults to <relion_dir>/../reference_gt.mrc if present.",
    )
    parser.add_argument("--max_iter", type=int, default=None, help="Optional cap on recovar iteration count")
    parser.add_argument(
        "--gt_align",
        action="store_true",
        help="Also compute alignment-aware GT metrics for merged maps.",
    )
    parser.add_argument(
        "--gt_align_healpix_order",
        type=int,
        default=1,
        help="RELION/RECOVAR rotation-grid order used for GT alignment.",
    )
    parser.add_argument(
        "--gt_align_max_shell",
        type=int,
        default=8,
        help="Maximum Fourier shell used to score coarse GT alignment.",
    )
    parser.add_argument(
        "--gt_align_no_mirror",
        action="store_true",
        help="Do not test the x-axis mirror handedness ambiguity during GT alignment.",
    )
    parser.add_argument(
        "--gt_align_allow_sign",
        action="store_true",
        help="Allow a global sign flip during GT alignment. Off by default.",
    )
    parser.add_argument(
        "--gt_align_all_series",
        action="store_true",
        help="Compute aligned GT metrics for half maps and unregularized maps too; default is merged maps only.",
    )
    args = parser.parse_args()

    import jax.numpy as jnp
    import mrcfile

    from recovar.core import fourier_transform_utils as ftu
    from recovar.em.initial_model.gt_metrics import (
        align_volume_to_reference,
        relion_alignment_rotations,
    )
    from recovar.reconstruction import regularization
    from recovar.utils import helpers

    recovar_dir = Path(args.recovar_dir)
    relion_dir = Path(args.relion_dir)
    intermediates_dir = recovar_dir / "intermediates"
    if not intermediates_dir.exists():
        raise FileNotFoundError(f"Missing intermediates directory: {intermediates_dir}")

    gt_path = Path(args.gt_volume) if args.gt_volume else relion_dir.parent / "reference_gt.mrc"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT volume: {gt_path}")

    results_npz = recovar_dir / "refinement_results.npz"
    if results_npz.exists():
        results = np.load(results_npz, allow_pickle=False)
        volume_shape = tuple(int(x) for x in results["volume_shape"])
        if len(volume_shape) != 3 or len(set(volume_shape)) != 1:
            raise ValueError(f"Expected cubic 3D volume shape, got {volume_shape}")
        n = int(volume_shape[0])
        pixel_size = float(results["voxel_size"])
    else:
        probe_path = next(intermediates_dir.glob("it*_half1_reg.mrc"), None)
        if probe_path is None:
            raise FileNotFoundError(f"No saved recovar half maps found in {intermediates_dir}")
        with mrcfile.open(str(probe_path), permissive=True) as mrc:
            if mrc.data.ndim != 3 or len(set(mrc.data.shape)) != 1:
                raise ValueError(f"Expected cubic 3D MRC data, got {mrc.data.shape} from {probe_path}")
            n = int(mrc.data.shape[0])
            volume_shape = (n, n, n)
            pixel_size = float(mrc.voxel_size.x)

    gt_real = helpers.load_mrc(str(gt_path))
    gt_ft = np.asarray(ftu.get_dft3(jnp.asarray(gt_real))).reshape(-1)
    align_rotations = None
    if args.gt_align:
        align_rotations = relion_alignment_rotations(args.gt_align_healpix_order)
        print(
            "GT alignment enabled: "
            f"healpix_order={args.gt_align_healpix_order}, rotations={align_rotations.shape[0]}, "
            f"score_shell<={args.gt_align_max_shell}, mirror={not args.gt_align_no_mirror}, "
            f"allow_sign={args.gt_align_allow_sign}, "
            f"series={'all' if args.gt_align_all_series else 'merged-only'}"
        )

    def _first_shell_below_threshold(fsc_values, threshold):
        below = np.where(np.asarray(fsc_values, dtype=np.float64) < float(threshold))[0]
        return int(below[0]) if below.size else -1

    def _shell_to_resolution_angstrom(shell_idx):
        if shell_idx is None or int(shell_idx) <= 0:
            return np.nan
        return float(n * pixel_size) / float(shell_idx)

    def _real_to_ft(real_vol):
        return np.asarray(ftu.get_dft3(jnp.asarray(real_vol))).reshape(-1)

    def _fsc_vs_gt(vol_ft):
        return np.asarray(
            regularization.get_fsc_gpu(
                jnp.asarray(vol_ft),
                jnp.asarray(gt_ft),
                volume_shape,
            ),
            dtype=np.float64,
        )

    def _corr(a, b):
        return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])

    def _should_align(prefix):
        return bool(args.gt_align and (args.gt_align_all_series or prefix.endswith("_merged")))

    def _add_metrics(store, prefix, real_vol, vol_ft):
        fsc_vs_gt = _fsc_vs_gt(vol_ft)
        shell_05 = _first_shell_below_threshold(fsc_vs_gt, 0.5)
        shell_0143 = _first_shell_below_threshold(fsc_vs_gt, 0.143)
        store[f"{prefix}_corr_vs_gt"] = np.float64(_corr(real_vol, gt_real))
        store[f"{prefix}_fsc_vs_gt"] = fsc_vs_gt
        store[f"{prefix}_shell_05"] = np.int32(shell_05)
        store[f"{prefix}_shell_0143"] = np.int32(shell_0143)
        if _should_align(prefix):
            assert align_rotations is not None
            alignment = align_volume_to_reference(
                real_vol,
                gt_real,
                align_rotations,
                score_max_shell=int(args.gt_align_max_shell),
                allow_mirror=not bool(args.gt_align_no_mirror),
                allow_sign=bool(args.gt_align_allow_sign),
            )
            aligned_ft = _real_to_ft(alignment.aligned_volume)
            aligned_fsc_vs_gt = _fsc_vs_gt(aligned_ft)
            aligned_shell_05 = _first_shell_below_threshold(aligned_fsc_vs_gt, 0.5)
            aligned_shell_0143 = _first_shell_below_threshold(aligned_fsc_vs_gt, 0.143)
            store[f"{prefix}_aligned_corr_vs_gt"] = np.float64(alignment.corr)
            store[f"{prefix}_aligned_score_vs_gt"] = np.float64(alignment.score)
            store[f"{prefix}_aligned_fsc_vs_gt"] = aligned_fsc_vs_gt
            store[f"{prefix}_aligned_shell_05"] = np.int32(aligned_shell_05)
            store[f"{prefix}_aligned_shell_0143"] = np.int32(aligned_shell_0143)
            store[f"{prefix}_gt_align_rotation_index"] = np.int32(alignment.rotation_index)
            store[f"{prefix}_gt_align_rotation_matrix"] = alignment.rotation_matrix
            store[f"{prefix}_gt_align_mirror_x"] = np.bool_(alignment.mirror_x)
            store[f"{prefix}_gt_align_sign"] = np.int32(alignment.sign)
            store[f"{prefix}_gt_align_score_max_shell"] = np.int32(args.gt_align_max_shell)
            store[f"{prefix}_gt_align_healpix_order"] = np.int32(args.gt_align_healpix_order)

    iter_pattern = re.compile(r"it(\d+)_half1_reg\.mrc$")
    iter_indices = []
    for path in intermediates_dir.glob("it*_half1_reg.mrc"):
        match = iter_pattern.search(path.name)
        if match:
            iter_indices.append(int(match.group(1)))
    iter_indices = sorted(set(iter_indices))
    if args.max_iter is not None:
        iter_indices = [it for it in iter_indices if it < int(args.max_iter)]

    if not iter_indices:
        print(f"No saved intermediate regularized volumes found in {intermediates_dir}")
        return 0

    if args.gt_align:
        print(
            f"{'iter':>4s} {'REL':>5s} {'rec_reg_corr':>12s} {'rec_aln':>10s} "
            f"{'rel_corr':>10s} {'rel_aln':>10s} {'rec-rel':>10s} {'0.143_Ra':>8s} {'0.143_La':>8s}"
        )
        print("-" * 92)
    else:
        print(
            f"{'iter':>4s} {'REL':>5s} {'rec_reg_corr':>12s} {'rel_corr':>10s} "
            f"{'rec_unreg':>10s} {'rec-rel':>10s} {'0.143_R':>8s} {'0.143_L':>8s}"
        )
        print("-" * 76)

    for rec_it in iter_indices:
        tag = f"{rec_it:03d}"
        relion_it = int(args.relion_start_iter) + rec_it + 1
        out = {
            "recovar_iter_index": np.int32(rec_it),
            "relion_iteration": np.int32(relion_it),
            "volume_shape": np.asarray(volume_shape, dtype=np.int32),
            "voxel_size": np.float64(pixel_size),
        }

        rec_reg_half1_real = helpers.load_mrc(str(intermediates_dir / f"it{tag}_half1_reg.mrc"))
        rec_reg_half2_real = helpers.load_mrc(str(intermediates_dir / f"it{tag}_half2_reg.mrc"))
        rec_reg_half1_ft = _real_to_ft(rec_reg_half1_real)
        rec_reg_half2_ft = _real_to_ft(rec_reg_half2_real)
        rec_reg_merged_real = 0.5 * (rec_reg_half1_real + rec_reg_half2_real)
        rec_reg_merged_ft = 0.5 * (rec_reg_half1_ft + rec_reg_half2_ft)

        _add_metrics(out, "recovar_reg_half1", rec_reg_half1_real, rec_reg_half1_ft)
        _add_metrics(out, "recovar_reg_half2", rec_reg_half2_real, rec_reg_half2_ft)
        _add_metrics(out, "recovar_reg_merged", rec_reg_merged_real, rec_reg_merged_ft)

        rec_unreg_half1_path = intermediates_dir / f"it{tag}_half1_unreg.mrc"
        rec_unreg_half2_path = intermediates_dir / f"it{tag}_half2_unreg.mrc"
        if rec_unreg_half1_path.exists() and rec_unreg_half2_path.exists():
            rec_unreg_half1_real = helpers.load_mrc(str(rec_unreg_half1_path))
            rec_unreg_half2_real = helpers.load_mrc(str(rec_unreg_half2_path))
            rec_unreg_half1_ft = _real_to_ft(rec_unreg_half1_real)
            rec_unreg_half2_ft = _real_to_ft(rec_unreg_half2_real)
            rec_unreg_merged_real = 0.5 * (rec_unreg_half1_real + rec_unreg_half2_real)
            rec_unreg_merged_ft = 0.5 * (rec_unreg_half1_ft + rec_unreg_half2_ft)
            _add_metrics(out, "recovar_unreg_half1", rec_unreg_half1_real, rec_unreg_half1_ft)
            _add_metrics(out, "recovar_unreg_half2", rec_unreg_half2_real, rec_unreg_half2_ft)
            _add_metrics(out, "recovar_unreg_merged", rec_unreg_merged_real, rec_unreg_merged_ft)

        rel_h1_path = relion_dir / f"run_it{relion_it:03d}_half1_class001.mrc"
        rel_h2_path = relion_dir / f"run_it{relion_it:03d}_half2_class001.mrc"
        if rel_h1_path.exists() and rel_h2_path.exists():
            rel_half1_real = helpers.load_relion_volume(str(rel_h1_path))
            rel_half2_real = helpers.load_relion_volume(str(rel_h2_path))
            rel_half1_ft = _real_to_ft(rel_half1_real)
            rel_half2_ft = _real_to_ft(rel_half2_real)
            rel_merged_real = 0.5 * (rel_half1_real + rel_half2_real)
            rel_merged_ft = 0.5 * (rel_half1_ft + rel_half2_ft)
            _add_metrics(out, "relion_half1", rel_half1_real, rel_half1_ft)
            _add_metrics(out, "relion_half2", rel_half2_real, rel_half2_ft)
            _add_metrics(out, "relion_merged", rel_merged_real, rel_merged_ft)

            out["recovar_reg_half1_corr_vs_relion"] = np.float64(_corr(rec_reg_half1_real, rel_half1_real))
            out["recovar_reg_half2_corr_vs_relion"] = np.float64(_corr(rec_reg_half2_real, rel_half2_real))
            out["recovar_reg_merged_corr_vs_relion"] = np.float64(_corr(rec_reg_merged_real, rel_merged_real))

        out_path = recovar_dir / f"gt_comparison_iter{tag}.npz"
        np.savez(out_path, **out)

        rec_corr = float(out["recovar_reg_merged_corr_vs_gt"])
        rel_corr = float(out["relion_merged_corr_vs_gt"]) if "relion_merged_corr_vs_gt" in out else np.nan
        rec_unreg = float(out["recovar_unreg_merged_corr_vs_gt"]) if "recovar_unreg_merged_corr_vs_gt" in out else np.nan
        rec_rel = float(out["recovar_reg_merged_corr_vs_relion"]) if "recovar_reg_merged_corr_vs_relion" in out else np.nan
        rec_0143 = int(out["recovar_reg_merged_shell_0143"])
        rel_0143 = int(out["relion_merged_shell_0143"]) if "relion_merged_shell_0143" in out else -1
        if args.gt_align:
            rec_aligned_corr = (
                float(out["recovar_reg_merged_aligned_corr_vs_gt"])
                if "recovar_reg_merged_aligned_corr_vs_gt" in out
                else np.nan
            )
            rel_aligned_corr = (
                float(out["relion_merged_aligned_corr_vs_gt"])
                if "relion_merged_aligned_corr_vs_gt" in out
                else np.nan
            )
            rec_aligned_0143 = (
                int(out["recovar_reg_merged_aligned_shell_0143"])
                if "recovar_reg_merged_aligned_shell_0143" in out
                else -1
            )
            rel_aligned_0143 = (
                int(out["relion_merged_aligned_shell_0143"])
                if "relion_merged_aligned_shell_0143" in out
                else -1
            )
            print(
                f"{rec_it + 1:4d} {relion_it:5d} {rec_corr:12.6f} {rec_aligned_corr:10.6f} "
                f"{rel_corr:10.6f} {rel_aligned_corr:10.6f} {rec_rel:10.6f} "
                f"{rec_aligned_0143:8d} {rel_aligned_0143:8d}"
            )
        else:
            print(
                f"{rec_it + 1:4d} {relion_it:5d} {rec_corr:12.6f} {rel_corr:10.6f} "
                f"{rec_unreg:10.6f} {rec_rel:10.6f} {rec_0143:8d} {rel_0143:8d}"
            )

    print(f"\nSaved per-iteration GT comparison files in {recovar_dir}")
    print(f"GT volume: {gt_path}")
    print("Resolution shells are reported as first shell where FSC drops below threshold.")
    print(
        f"Example: shell 41 corresponds to {_shell_to_resolution_angstrom(41):.2f} A "
        f"for N={n}, pixel_size={pixel_size:.3f}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
