"""Phase B-deep probe: drive RELION's exact posterior table through our
adjoint primitives, then compare to RELION's BPref dump.

If CC → +1.0: posterior parity is the gap. Port adaptive 2-pass.
If CC stays low: adjoint / image-preprocessing parity is the gap. Drill there.

Usage:
  CUDA_VISIBLE_DEVICES=<gpu_idx> RECOVAR_DISABLE_CUDA=1 \
    pixi run python scripts/probe_estep_relion_posterior.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path("/scratch/gpfs/GILLES/mg6942/tmp/relion_initialmodel_64_20260420_121428_8956_run")
PARTICLES_STAR = Path(
    "/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.tmp/"
    "slurm_7178672/pytest-of-mg6942/pytest-0/test_pipeline_spa_gpu0/"
    "gpu_spa/test_dataset/particles.star"
)
DUMP_DIR = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_dump_small")
RELION_BPREF = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_debug_dump")


def _read_bin(p: Path) -> np.ndarray:
    with open(p, "rb") as f:
        nz, ny, nx = struct.unpack("qqq", f.read(24))
        pos = f.tell()
        f.seek(0, 2)
        rem = f.tell() - pos
        f.seek(pos)
        bp = rem // (nz * ny * nx)
        dt = np.complex128 if bp == 16 else np.float64
        return np.fromfile(f, dtype=dt, count=nz * ny * nx).reshape(nz, ny, nx)


def _cc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel() - a.mean()
    bf = b.ravel() - b.mean()
    return float(np.real(np.vdot(af, bf)) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))


def _euler_to_matrix(rot_deg: float, tilt_deg: float, psi_deg: float) -> np.ndarray:
    rot = np.deg2rad(rot_deg)
    tilt = np.deg2rad(tilt_deg)
    psi = np.deg2rad(psi_deg)
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


def _read_relion_posterior(part_id: int) -> np.ndarray:
    """Return posterior table reshaped to (n_rot=4608, n_trans=116)."""
    with open(DUMP_DIR / f"p{part_id}_exp_Mweight.bin", "rb") as f:
        dims = struct.unpack("qqqqqq", f.read(48))
        arr = np.fromfile(f, dtype=np.float64, count=int(np.prod(dims))).reshape(dims)
    # arr: (cls=1, dir=48, psi=12, trans=29, orot=8, otra=4)
    inner = arr[0]  # drop class axis → (48, 12, 29, 8, 4)
    # Rotation index = dir*psi*orot, translation = trans*otra
    rot_post = np.transpose(inner, (0, 1, 3, 2, 4)).reshape(48 * 12 * 8, 29 * 4)
    return rot_post


def main() -> None:
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.em.initial_model.gpu_pipeline import _split_halfset_particle_ids

    # Load RELION-exact rotations + translations + per-particle posterior
    with open(DUMP_DIR / "p0_oversampled_eulers.bin", "rb") as f:
        h = struct.unpack("qqq", f.read(24))
        eulers = np.fromfile(f, dtype=np.float64, count=h[0] * h[1] * h[2]).reshape(-1, 3)
    with open(DUMP_DIR / "p0_oversampled_translations.bin", "rb") as f:
        h = struct.unpack("qqq", f.read(24))
        trans_3d = np.fromfile(f, dtype=np.float64, count=h[0] * h[1] * h[2]).reshape(-1, 3)
    rotations = np.array([_euler_to_matrix(*e) for e in eulers]).astype(np.float64)
    translations = trans_3d[:, :2].astype(np.float64)
    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    print(f"RELION-exact grid: {n_rot} rotations × {n_trans} translations")

    # Load dataset and pick the FIRST particle of halfset 0 (corresponds to
    # RELION's part_id=0).
    ds = load_dataset(str(PARTICLES_STAR), lazy=False)
    h0_ids, _ = _split_halfset_particle_ids(ds.n_images)
    p0_relion_post = _read_relion_posterior(0)
    print(
        f"RELION posterior shape: {p0_relion_post.shape}, sum={p0_relion_post.sum():.3e}, max={p0_relion_post.max():.3e}, "
        f"nonzero={int((p0_relion_post > 0).sum())}"
    )

    # For our adjoint, we need to combine multiple particles. The dump only has
    # 3 particles. Process those 3 only and compare to RELION BPref (which is
    # the SUM over ALL 250 h0 particles). So this is a partial probe — direct
    # CC won't reach +1.0, but the *direction* of the bp_data should match.
    #
    # Actually: the cleanest test is to compute a synthetic bp_data by combining
    # OUR adjoint of OUR images with RELION's posterior, then compare *normalised
    # angular content* (don't expect amplitude to match).

    print("\n(Probe scaffolded — full implementation needs run_em internals exposed)")
    print("Direction: confirm that with RELION-exact posterior, our adjoint reproduces RELION's bp_data direction.")


if __name__ == "__main__":
    main()
