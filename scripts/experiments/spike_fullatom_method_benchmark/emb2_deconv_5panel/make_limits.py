"""Build FULL-volume noiseless limits (emb2 noise1, state 50), consistent with the saved
deconv_noiseless_bound (which is a DEVIATION volume). Standard limit = ordinary Epanechnikov
product kernel (non-deconvolution), best candidate by deviation masked-FSC; deconv limit full =
saved deviation bound + GT mean. Both saved as full volumes so they render like the others."""

import os
import sys

import mrcfile
import numpy as np

WT = "/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_kernel_dev2_rebase"
sys.path.insert(0, WT)
import recovar.core.fourier_transform_utils as ftu
from recovar.heterogeneity import highdim_deconv as H
from recovar.output import plot_utils
from recovar.reconstruction import regularization

ROOT = "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_gt_embeddings_noise1_dataset_20260601"
EMB = os.path.join(ROOT, "emb2_zdim1_index_white", "controlled_pipeline", "model", "zdim_1")
RUN = "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise1_b80_true_oracle_sweep_20260527/n01000000/runs/n01000000_seed0000"
BROAD = "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
DL = os.path.join(ROOT, "download_emb2_noise1")
OUT = "/scratch/gpfs/CRYOEM/gilleslab/tmp/emb2_noise1_movingmask_5panel_20260607"
GRID, VOX, K, TARGET = 256, 1.25, 100, 50
VS = (GRID, GRID, GRID)


def load(p):
    with mrcfile.open(p, permissive=True) as m:
        return np.asarray(m.data, np.float32).reshape(-1)


z = np.load(os.path.join(EMB, "latent_coords_noreg.npy")).astype(np.float64)
prec = np.load(os.path.join(EMB, "latent_precision_noreg.npy")).astype(np.float64)
target = np.loadtxt(os.path.join(ROOT, "target_emb2_zdim1_state50.txt")).reshape(-1).astype(np.float64)
sa = np.load(os.path.join(RUN, "03_dataset", "state_assignment.npy")).astype(int).reshape(-1)

print("loading 100 GT vols...", flush=True)
Vfull = np.empty((K, GRID**3), np.float32)
for k in range(K):
    Vfull[k] = load(os.path.join(RUN, "04_ground_truth", f"gt_vol{k:04d}.mrc"))
Vmean = Vfull.mean(axis=0)
Vdev = Vfull - Vmean[None, :]
mask = load(BROAD)
fg = np.asarray(ftu.get_dft3((Vdev[TARGET] * mask).reshape(VS))).reshape(-1)


def res_of(dev_vec):
    c = np.asarray(regularization.get_fsc(np.asarray(ftu.get_dft3((dev_vec * mask).reshape(VS))).reshape(-1), fg, VS))
    return 1.0 / float(plot_utils.fsc_score(c, GRID, VOX, threshold=0.5))


def save_full(path, dev_vec):
    full = (dev_vec + Vmean).astype(np.float32)
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(full.reshape(GRID, GRID, GRID))
        m.voxel_size = VOX


# ---- deconv limit full = saved deviation bound + mean ----
import glob

dec_dev = load(sorted(glob.glob(os.path.join(DL, "deconv_noiseless_bound_cand*A.mrc")))[0])
save_full(os.path.join(OUT, "deconv_limit_full.mrc"), dec_dev)
print(f"deconv limit (full) res={res_of(dec_dev):.2f}A -> deconv_limit_full.mrc", flush=True)

# ---- standard limit: ordinary Epanechnikov product kernel (non-deconvolution) ----
W, info, diag = H.weights_ordinary_product_epan(z, target, prec)
NC = W.shape[0]
best = (-1, None, None)
for c in range(NC):
    if not info[c].get("accepted", True):
        continue
    Wk = np.bincount(sa, weights=np.asarray(W[c], np.float64), minlength=K)
    s = Wk.sum()
    if abs(s) < 1e-12:
        continue
    alpha = (Wk / s).astype(np.float32)
    dev = alpha @ Vdev
    r05 = res_of(dev)
    if 1.0 / r05 > best[0]:
        best = (1.0 / r05, c, dev)
print(f"standard limit best cand {best[1]}: res={1.0 / best[0]:.2f}A", flush=True)
save_full(os.path.join(OUT, "standard_limit_full.mrc"), best[2])
print("saved standard_limit_full.mrc", flush=True)

# sanity: scales of all 5 full volumes (within broad mask)
mb = mask > 0.5
for name, p in [
    ("deconv_limit", os.path.join(OUT, "deconv_limit_full.mrc")),
    ("deconv_1M", os.path.join(DL, "best_deconvolved_eigenratio_3.99A.mrc")),
    ("standard_limit", os.path.join(OUT, "standard_limit_full.mrc")),
    ("standard_1M", os.path.join(DL, "best_convolved_standard_4.09A.mrc")),
    ("GT", os.path.join(DL, "GT_state50.mrc")),
]:
    v = load(p)
    print(
        f"  {name:14s} max={v.max():.4f} p99.9={np.percentile(v, 99.99):.4f} mean_in_broad={v[mb].mean():.4f}",
        flush=True,
    )
print("DONE", flush=True)
