# Spike state 50, noise=1, 1M images — 5-way moving-mask comparison

Renders 5 reconstructions of state 50 in the reference "moving-mask highlight" view
(conformational/moving region colored; static core grey & transparent), same camera as
nonuniform_noise1_300k_tracked_atoms_v13 GIF.

## Columns (user order)
1. deconvolved — LIMIT (∞ images), 3.52 Å   -> deconv_limit_full.mrc        (computed)
2. deconvolved — 1M images,        3.99 Å   -> download_emb2_noise1/best_deconvolved_eigenratio_3.99A.mrc
3. standard   — LIMIT (∞ images), 3.61 Å   -> standard_limit_full.mrc       (computed)
4. standard   — 1M images,        4.09 Å   -> download_emb2_noise1/best_convolved_standard_4.09A.mrc
5. ground truth state 50                    -> download_emb2_noise1/GT_state50.mrc

## Outputs
- state50_5panel_movingmask.png        the 5-panel comparison figure
- png/<col>.png                        individual transparent-bg renders (1100x1100, ss3)
- deconv_limit_full.mrc, standard_limit_full.mrc   the computed noiseless limits (FULL volumes)
- split/<col>_{inside,outside}.mrc     moving/context split fed to ChimeraX

## How the limits were computed (make_limits.py)
Oracle reconstructions sum_k alpha_k V_gt(k) from the 100 noise1 GT volumes (oracle sweep run),
emb2 zdim1 index-white embedding, target state 50:
- deconv limit  = best per_image_eigen_ratio candidate (weights_per_image_eigen_ratio), aggregated
  per state. The saved download_emb2_noise1 bound is a DEVIATION volume; here +GT mean -> full.
- standard limit = best ordinary Epanechnikov product-kernel candidate (weights_ordinary_product_epan),
  same aggregation, +GT mean -> full.
Best candidate chosen by deviation masked-FSC@0.5 vs GT state50 (broad_mask).

## Render (build_render.py + render_all.py, ChimeraX 1.9 clean env)
Per volume: inside=vol*movingmask (colored), outside=vol*(1-mask) (#9a9a9a, 58% opaque);
single shared contour 0.026; camera = reference moving_view matrix; supersample 3.
Moving mask: .../corrected_nonuniform_v13_solid_mask_scoring_shellfix_20260605/masks/state0050_tracked_atoms_soft.mrc
