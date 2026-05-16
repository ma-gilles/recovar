"""Build init NPZ from RELION K=Q ab-initio output.

μ = anchor (most-populated class) OR weighted mean.
W = SVD basis of (K vols − μ), padded to q with zeros.
All in RELION's frame (NO GT alignment / cheat).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--relion-dir", required=True, type=Path)
    p.add_argument("--iter", type=int, default=200)
    p.add_argument("--q", type=int, default=4)
    p.add_argument("--out-npz", required=True, type=Path)
    p.add_argument(
        "--workdir", type=str, default="/scratch/gpfs/GILLES/mg6942/recovar_wt_ppca_postmerge_followup_20260510_110827"
    )
    p.add_argument("--anchor", choices=("most_populated", "weighted_mean"), default="most_populated")
    return p.parse_args()


def _parse_class_counts(star: Path, q: int):
    import numpy as np

    counts = {}
    in_loop = False
    col = -1
    cols = []
    with open(star) as fh:
        for line in fh:
            s = line.strip()
            if s == "loop_":
                in_loop = True
                cols = []
                col = -1
                continue
            if in_loop and s.startswith("_rln"):
                cols.append(s.split()[0])
                if s.split()[0] == "_rlnClassNumber":
                    col = len(cols) - 1
                continue
            if in_loop and s and not s.startswith("_") and col >= 0:
                f = s.split()
                if len(f) > col:
                    c = int(f[col])
                    counts[c] = counts.get(c, 0) + 1
    return np.array([counts.get(k + 1, 0) for k in range(q)], dtype=np.float64)


def main():
    args = _parse_args()
    sys.path.insert(0, args.workdir)
    import numpy as np

    from recovar.utils import helpers as _helpers

    vol_paths = sorted(args.relion_dir.glob(f"run_it{args.iter:03d}_class*.mrc"))
    if len(vol_paths) != args.q:
        vol_paths = sorted(args.relion_dir.glob(f"run_it{args.iter}_class*.mrc"))
    if len(vol_paths) != args.q:
        raise SystemExit(f"expected {args.q} class vols at iter {args.iter}, got {len(vol_paths)}")
    vols = np.stack([_helpers.load_relion_volume(str(p)) for p in vol_paths]).astype(np.float32)
    counts = _parse_class_counts(args.relion_dir / f"run_it{args.iter:03d}_data.star", args.q)
    weights = counts / max(counts.sum(), 1.0)
    print(f"RELION K={args.q} counts: {counts.astype(int).tolist()}")

    if args.anchor == "most_populated":
        anchor_idx = int(np.argmax(counts))
        mu_init = vols[anchor_idx].copy()
        other = np.stack([vols[k] for k in range(args.q) if k != anchor_idx], axis=0)
        ow = np.array([weights[k] for k in range(args.q) if k != anchor_idx], dtype=np.float64)
        print(f"anchor class {anchor_idx + 1} ({int(counts[anchor_idx])} particles)")
    else:
        mu_init = (weights[:, None, None, None] * vols).sum(axis=0).astype(np.float32)
        other, ow = vols, weights

    deltas = (other - mu_init[None, ...]).reshape(other.shape[0], -1).astype(np.float64)
    deltas_w = deltas * np.sqrt(np.maximum(ow, 0.0))[:, None]
    _, S, Vt = np.linalg.svd(deltas_w, full_matrices=False)
    W_init = np.zeros((args.q,) + vols.shape[1:], dtype=np.float32)
    for k in range(min(args.q, Vt.shape[0])):
        W_init[k] = Vt[k].reshape(vols.shape[1:]).astype(np.float32) * float(S[k])

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        mu=mu_init,
        W=W_init,
        weights=weights.astype(np.float64),
        singular_values=np.concatenate([S, np.zeros(max(args.q - S.shape[0], 0))]),
        anchor_mode=args.anchor,
        anchor_class=int(np.argmax(counts) + 1),
    )
    print(
        f"wrote {args.out_npz}  μ RMS={np.sqrt(np.mean(mu_init**2)):.3e}  W RMS={np.sqrt(np.mean(W_init**2)):.3e}  σ={S}"
    )


if __name__ == "__main__":
    main()
