# RECOVAR / RELION VDAM InitialModel parity scorecard

**Fixed-suite score: 0 / 12 passing (0 / 12 evaluated).**

Suite: `vdam-k1-gui-grid0-fixed12` (version 1; denominator frozen at 12).
Frozen case-definition SHA-256: `1a37a1b360b022d60eefdd0481eb0784d4a0e98a4d92066199625ceaf6d11dd1`.
Source fixture manifest SHA-256: `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`.

A pass requires every fixed checkpoint to preserve the schedule and artifact topology, cross-engine FSC-AUC >= `0.999`, and RECOVAR-minus-RELION GT FSC-AUC >= `-0.002` on the same physical GPU.
Map correlation is not computed or gated. Historical correlation-only runs are non-scoring.

| Case | Reused fixed EM fixture | Result | Checkpoints | Evidence |
|---|---|---:|---|---|
| `vdam-01` small_baseline | `k1-11` | — | not run | `—` |
| `vdam-02` small_very_high_noise | `k1-12` | — | not run | `—` |
| `vdam-03` small_anisotropic | `k1-13` | — | not run | `—` |
| `vdam-04` small_noctf | `k1-14` | — | not run | `—` |
| `vdam-05` small_contrast_noise_scale | `k1-18` | — | not run | `—` |
| `vdam-06` small_image_offset | `k1-19` | — | not run | `—` |
| `vdam-07` small_severe_outliers | `k1-22` | — | not run | `—` |
| `vdam-08` tiny_baseline | `k1-25` | — | not run | `—` |
| `vdam-09` small_high_res_radial | `k1-20` | — | not run | `—` |
| `vdam-10` mid_baseline | `k1-31` | — | not run | `—` |
| `vdam-11` production_baseline | `k1-01` | — | not run | `—` |
| `vdam-12` production_near_nyquist | `k1-09` | — | not run | `—` |

## Fixed checkpoints

The v1 trajectory checkpoints are iterations `0`, `1`, `2`, `4`, and `8`. Iteration 0 covers bootstrap/reference initialization; later checkpoints cover the complete VDAM schedule, E-step state, pseudo-halfset M-step, and written maps.

Regenerate and validate this page with:

```bash
pixi run python scripts/summarize_vdam_relion_parity_scorecard.py --check
```
