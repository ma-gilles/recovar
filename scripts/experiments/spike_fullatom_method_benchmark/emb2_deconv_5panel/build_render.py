"""Split each of the 5 emb2 state-50 volumes into moving (inside, colored) + context (outside, grey),
and emit a ChimeraX script that renders each in the reference moving-mask view (single shared contour)."""

import os

import mrcfile
import numpy as np

OUT = "/scratch/gpfs/CRYOEM/gilleslab/tmp/emb2_noise1_movingmask_5panel_20260607"
DL = "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_gt_embeddings_noise1_dataset_20260601/download_emb2_noise1"
MASK = "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_nonuniform_B70_noise1_b80_300k_20260604/corrected_nonuniform_v13_solid_mask_scoring_shellfix_20260605/masks/state0050_tracked_atoms_soft.mrc"
CAM = "-0.461300085,-0.130549751,0.877587026,327.227726,0.878505065,0.0712772908,0.472385852,222.126509,-0.124221883,0.988876285,0.0818083987,224.957178"
LEVEL = 0.026  # single shared contour (all 5 on same scale)
os.makedirs(OUT + "/split", exist_ok=True)
os.makedirs(OUT + "/png", exist_ok=True)

# columns in the user's order: deconv-limit, deconv-1M, standard-limit, standard-1M, GT
COLS = [
    ("deconv_limit", f"{OUT}/deconv_limit_welltuned.mrc", "#b30000", "deconvolved — limit (∞ img)\n3.52 Å"),
    ("deconv_1M", f"{DL}/best_deconvolved_eigenratio_3.99A.mrc", "#ff5b5b", "deconvolved — 1M images\n3.99 Å"),
    ("standard_limit", f"{OUT}/standard_limit_welltuned.mrc", "#08306b", "standard — limit (∞ img)\n3.61 Å"),
    ("standard_1M", f"{DL}/best_convolved_standard_4.09A.mrc", "#4d94ff", "standard — 1M images\n4.09 Å"),
    ("GT", f"{DL}/GT_state50.mrc", "#c8a020", "ground truth\nstate 50"),
]


def load(p):
    with mrcfile.open(p, permissive=True) as m:
        return np.asarray(m.data, np.float32), float(m.voxel_size.x)


def save(p, v, vx):
    with mrcfile.new(p, overwrite=True) as m:
        m.set_data(v.astype(np.float32))
        m.voxel_size = vx


mk, _ = load(MASK)
mk = np.clip(mk, 0, 1)
for name, path, color, label in COLS:
    v, vx = load(path)
    save(f"{OUT}/split/{name}_inside.mrc", v * mk, vx)
    save(f"{OUT}/split/{name}_outside.mrc", v * (1 - mk), vx)

lines = []
for name, path, color, label in COLS:
    lines += [
        "run(session,'close')",
        f"run(session,'open \"{OUT}/split/{name}_outside.mrc\"')",
        f"run(session,'open \"{OUT}/split/{name}_inside.mrc\"')",
        "run(session,'set bgColor white'); run(session,'lighting soft'); run(session,'graphics silhouettes true')",
        f"run(session,'volume #1 style surface'); run(session,'volume #1 level {LEVEL}'); run(session,'color #1 #9a9a9a'); run(session,'transparency #1 58 surfaces')",
        f"run(session,'volume #2 style surface'); run(session,'volume #2 level {LEVEL}'); run(session,'color #2 {color}')",
        f"run(session,'view matrix camera {CAM}')",
        f"run(session,'save \"{OUT}/png/{name}.png\" width 1100 height 1100 supersample 3 transparentBackground true')",
        f"print('RENDERED {name}')",
    ]
open(f"{OUT}/render_all.py", "w").write("from chimerax.core.commands import run\n" + "\n".join(lines) + "\n")
print("split + render_all.py written for", len(COLS), "columns", flush=True)
