#!/usr/bin/env python
"""Parse a RELION operand dump directory into one compressed NPZ."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


REAL_2D_FILES = {"Fctf", "Minvsigma2", "Mctf", "pdf_direction", "sigma2_noise"}
COMPLEX_2D_FILES = {
    "Fimg_store",
    "Fimg_unweighted",
    "Fimg_shifted_t0",
    "Fref",
    "Fref_orient0",
    "Frefctf",
    "Frefctf_orient0",
}
FLAT_COMPLEX_FILES = {
    "Fimg",
}
FLAT_REAL_FILES = {
    "diff2_weights",
    "exp_Mweight_diff2",
    "exp_Mweight_posterior",
    "candidate_weight_normalized",
    "candidate_weight_cumulative_fraction",
    "candidate_orientation_log_prior",
    "candidate_offset_log_prior",
    "candidate_combined_log_prior",
    "candidate_translation_x",
    "candidate_translation_y",
    "coarse_candidate_weight_normalized",
    "coarse_log_weight_preexp",
    "coarse_raw_diff2",
    "exp_Mweight_raw_preprior",
    "firstiter_cc_exp_Mweight_raw_preonehot",
    "Fimg_corrected_imag",
    "Fimg_corrected_real",
    "corr_img",
    "eulers_matrices",
    "trans_xyz_phases",
    "translations_x",
    "translations_y",
    "directions_prior",
    "fine_eulers",
    "fine_psis",
    "fine_ref_imag",
    "fine_ref_real",
    "fine_rots",
    "fine_shifted_imag",
    "fine_shifted_real",
    "fine_tilts",
    "psi_prior",
    "pdf_offset",
    "pdf_orientation",
    "ppref_imag",
    "ppref_real",
    "sorted_weights",
    "cc_component_weight",
    "cc_component_norm",
}
FLAT_INT_FILES = {
    "pointer_dir_nonzeroprior",
    "pointer_psi_nonzeroprior",
    "acc_rot_id",
    "acc_rot_idx",
    "acc_trans_idx",
    "acc_ihidden_overs",
    "firstiter_cc_raw_ihidden_overs",
    "firstiter_cc_raw_rot_id",
    "firstiter_cc_raw_rot_idx",
    "firstiter_cc_raw_trans_idx",
    "firstiter_cc_weight_dims",
    "candidate_in_denominator_set",
    "candidate_in_fine_threshold_set",
    "candidate_in_reconstruction_set",
    "candidate_class_idx",
    "candidate_sorted_rank",
    "candidate_coarse_trans_idx",
    "coarse_candidate_class_idx",
    "coarse_candidate_in_threshold_set",
    "coarse_candidate_rot_idx",
    "coarse_candidate_trans_idx",
    "fine_class_entries",
    "fine_class_idx",
    "fine_iorientclasses",
    "fine_iover_rots",
    "ppref_dims",
}


def _parse_dimensions(path):
    dims = {}
    if not path.exists():
        return dims
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        dims[f"header_{key.strip()}"] = np.array(float(value) if "." in value else int(value))
    return dims


def _read_real_2d(path):
    raw = path.read_bytes()
    ydim = np.frombuffer(raw[:4], dtype=np.int32)[0]
    xdim = np.frombuffer(raw[4:8], dtype=np.int32)[0]
    data = np.frombuffer(raw[8:], dtype=np.float64).copy()
    return data.reshape(ydim, xdim)


def _read_complex_2d(path):
    raw = path.read_bytes()
    ydim = np.frombuffer(raw[:4], dtype=np.int32)[0]
    xdim = np.frombuffer(raw[4:8], dtype=np.int32)[0]
    data = np.frombuffer(raw[8:], dtype=np.complex128).copy()
    return data.reshape(ydim, xdim)


def _read_flat_real(path):
    raw = path.read_bytes()
    ndim = np.frombuffer(raw[:4], dtype=np.int32)[0]
    return np.frombuffer(raw[4:], dtype=np.float64, count=ndim).copy()


def _read_flat_split_complex(path):
    raw = path.read_bytes()
    ndim = np.frombuffer(raw[:4], dtype=np.int32)[0]
    data = np.frombuffer(raw[4:], dtype=np.float64, count=ndim).copy()
    if data.size % 2:
        raise ValueError(f"Split-complex RELION dump {path} has odd float count {data.size}")
    n_complex = data.size // 2
    return data[:n_complex] + 1j * data[n_complex:]


def _read_flat_int(path):
    raw = path.read_bytes()
    ndim = np.frombuffer(raw[:4], dtype=np.int32)[0]
    return np.frombuffer(raw[4:], dtype=np.int32, count=ndim).copy()


def _read_scalar(path):
    size = path.stat().st_size
    if size == 4:
        return np.array(np.fromfile(path, dtype=np.int32, count=1)[0])
    return np.array(np.fromfile(path, dtype=np.float64, count=1)[0])


def _layout_name(name):
    while True:
        stripped = re.sub(
            r"^(?:(?:pass|over|img|part|class)\d+|store_candidate\d+|storeWavg)_",
            "",
            name,
            count=1,
        )
        if stripped == name:
            return name
        name = stripped


def parse_dump_dir(dump_dir):
    dump_dir = Path(dump_dir)
    payload = _parse_dimensions(dump_dir / "dimensions.txt")
    for bin_path in sorted(dump_dir.glob("*.bin")):
        name = bin_path.stem
        layout_name = _layout_name(name)
        if layout_name in REAL_2D_FILES:
            payload[name] = _read_real_2d(bin_path)
        elif layout_name in COMPLEX_2D_FILES:
            payload[name] = _read_complex_2d(bin_path)
        elif layout_name in FLAT_COMPLEX_FILES:
            payload[name] = _read_flat_split_complex(bin_path)
        elif layout_name in FLAT_REAL_FILES:
            payload[name] = _read_flat_real(bin_path)
        elif layout_name in FLAT_INT_FILES:
            payload[name] = _read_flat_int(bin_path)
        else:
            payload[name] = _read_scalar(bin_path)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", help="Directory containing RELION .bin dumps")
    parser.add_argument("--output", default=None, help="Optional output NPZ path")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    out_path = Path(args.output) if args.output is not None else dump_dir / "relion_operands.npz"
    payload = parse_dump_dir(dump_dir)
    np.savez_compressed(out_path, **payload)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
