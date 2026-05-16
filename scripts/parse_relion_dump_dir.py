#!/usr/bin/env python
"""Parse a RELION operand dump directory into one compressed NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REAL_2D_FILES = {"Fctf", "Minvsigma2", "pdf_direction", "sigma2_noise"}
COMPLEX_2D_FILES = {"Fimg_unweighted", "Fimg_shifted_t0", "Fref_orient0", "Frefctf_orient0"}
FLAT_REAL_FILES = {
    "exp_Mweight_diff2",
    "exp_Mweight_posterior",
    "candidate_weight_normalized",
    "candidate_weight_cumulative_fraction",
    "candidate_orientation_log_prior",
    "candidate_offset_log_prior",
    "candidate_combined_log_prior",
    "candidate_translation_x",
    "candidate_translation_y",
    "translations_x",
    "translations_y",
    "directions_prior",
    "psi_prior",
    "pdf_offset",
    "pdf_orientation",
}
FLAT_INT_FILES = {
    "pointer_dir_nonzeroprior",
    "pointer_psi_nonzeroprior",
    "acc_rot_id",
    "acc_rot_idx",
    "acc_trans_idx",
    "acc_ihidden_overs",
    "candidate_in_denominator_set",
    "candidate_in_fine_threshold_set",
    "candidate_in_reconstruction_set",
    "candidate_sorted_rank",
    "candidate_coarse_trans_idx",
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


def _read_flat_int(path):
    raw = path.read_bytes()
    ndim = np.frombuffer(raw[:4], dtype=np.int32)[0]
    return np.frombuffer(raw[4:], dtype=np.int32, count=ndim).copy()


def _read_scalar(path):
    return np.array(np.fromfile(path, dtype=np.float64, count=1)[0])


def parse_dump_dir(dump_dir):
    dump_dir = Path(dump_dir)
    payload = _parse_dimensions(dump_dir / "dimensions.txt")
    for bin_path in sorted(dump_dir.glob("*.bin")):
        name = bin_path.stem
        if name in REAL_2D_FILES:
            payload[name] = _read_real_2d(bin_path)
        elif name in COMPLEX_2D_FILES:
            payload[name] = _read_complex_2d(bin_path)
        elif name in FLAT_REAL_FILES:
            payload[name] = _read_flat_real(bin_path)
        elif name in FLAT_INT_FILES:
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
