# RECOVAR / RELION VDAM InitialModel parity scorecard

> **Legacy, non-authoritative short-prefix suite.** This fixed12 v1 board covers only iterations 0, 1, 2, 4, and 8; it does not report the frozen v3 20-case, iteration 0--200 score. See `docs/math/vdam_relion_parity_dashboard.md` for authoritative status.

**Fixed-suite score: 12 / 12 passing (12 / 12 evaluated).**

Suite: `vdam-k1-gui-grid0-fixed12` (version 1; denominator frozen at 12).
Frozen case-definition SHA-256: `1a37a1b360b022d60eefdd0481eb0784d4a0e98a4d92066199625ceaf6d11dd1`.
Source fixture manifest SHA-256: `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`.

A pass requires every fixed checkpoint to preserve the schedule and artifact topology, cross-engine FSC-AUC >= `0.999`, and RECOVAR-minus-RELION GT FSC-AUC >= `-0.002` on the same physical GPU.
Map correlation is not computed or gated. Historical correlation-only runs are non-scoring.

| Case | Reused fixed EM fixture | Result | Checkpoints | Evidence |
|---|---|---:|---|---|
| `vdam-01` small_baseline | `k1-11` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `fbccbcbf23e3e0266aac04812bf77ea18b3ba40be495b6ca24051481b16f1d22` |
| `vdam-02` small_very_high_noise | `k1-12` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `6aa5f5496e82d9081b9a9645caf760ccf4c2db4a2ac6bec0e41a04ce8a4e33a0` |
| `vdam-03` small_anisotropic | `k1-13` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `4d501c845f44d7f10e727eefbd19a456e34191c400013dc4438b4209fa9ffe16` |
| `vdam-04` small_noctf | `k1-14` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `7b9ee22176cbe71779f7e5343c3b11843b46336c72a049724269c3c7ec56a4a3` |
| `vdam-05` small_contrast_noise_scale | `k1-18` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `5fb662ed072dd9b21f72ce0911e7a35ccd622a26a8064f550df9ee1f56f9686f` |
| `vdam-06` small_image_offset | `k1-19` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `33333dc7ed29125b9e4c9c08a1df34b46e0f60800f581d1dba9e982e2e301f86` |
| `vdam-07` small_severe_outliers | `k1-22` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `1fa5d1ee9d0af5ca9e8e594f3a15cf048ba3fc56fcb62856d79651a0bf4f123e` |
| `vdam-08` tiny_baseline | `k1-25` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `8a3d17131449baf23ab20c275a5552b68ed4c860d7306ebc37ca773bc7a251b1` |
| `vdam-09` small_high_res_radial | `k1-20` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `69661d3c4d60c8b23ebb73716dd781d4b7f44c96eb48c09bd035f9c2fc2cacfe` |
| `vdam-10` mid_baseline | `k1-31` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `b8d8e3fccb24a5e8d7489f4efee6cbb4a03c801c08637ce8537329d5dc76ba7b` |
| `vdam-11` production_baseline | `k1-01` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `df1801f8ea4fe4d77f48b3d56d14a95b482439dd3c9934869ae78f09ca22ca18` |
| `vdam-12` production_near_nyquist | `k1-09` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `46d4335ad91dc29d4af11cc80ba5528dd8ec652add3680ef3b067ebcb4bb438d` |

## Fixed checkpoints

The v1 trajectory checkpoints are iterations `0`, `1`, `2`, `4`, and `8`. Iteration 0 covers bootstrap/reference initialization; later checkpoints cover the complete VDAM schedule, E-step state, pseudo-halfset M-step, and written maps.

Regenerate and validate this page with:

```bash
pixi run python scripts/summarize_vdam_relion_parity_scorecard.py --check
```
