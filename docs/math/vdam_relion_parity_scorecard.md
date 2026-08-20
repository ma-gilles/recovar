# RECOVAR / RELION VDAM InitialModel parity scorecard

**Fixed-suite score: 6 / 12 passing (12 / 12 evaluated).**

Suite: `vdam-k1-gui-grid0-fixed12` (version 1; denominator frozen at 12).
Frozen case-definition SHA-256: `1a37a1b360b022d60eefdd0481eb0784d4a0e98a4d92066199625ceaf6d11dd1`.
Source fixture manifest SHA-256: `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`.

A pass requires every fixed checkpoint to preserve the schedule and artifact topology, cross-engine FSC-AUC >= `0.999`, and RECOVAR-minus-RELION GT FSC-AUC >= `-0.002` on the same physical GPU.
Map correlation is not computed or gated. Historical correlation-only runs are non-scoring.

| Case | Reused fixed EM fixture | Result | Checkpoints | Evidence |
|---|---|---:|---|---|
| `vdam-01` small_baseline | `k1-11` | FAIL | 0:pass, 1:pass, 2:pass, 4:fail, 8:fail | `95605b72a2ab13c63af52d2ed52dfbad40fedcead954469d04dbbd784267700d` |
| `vdam-02` small_very_high_noise | `k1-12` | FAIL | 0:pass, 1:pass, 2:pass, 4:pass, 8:fail | `4ff0ca29029fbffcd5d93649abf860aa435fbc9c9cde32273b00231866145bda` |
| `vdam-03` small_anisotropic | `k1-13` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `b5428af040140d54baec548548839c55e407a12c8553fd9d7ade3d72f7a9561e` |
| `vdam-04` small_noctf | `k1-14` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `8247af430cfcafff6862cfa9203627d2de2640570178fcaf442acb571718d5c5` |
| `vdam-05` small_contrast_noise_scale | `k1-18` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `05462686b83ceaa0dca0dcc0e84129ba32c1ee3fadc1212c96d8e62430b8628d` |
| `vdam-06` small_image_offset | `k1-19` | FAIL | 0:pass, 1:pass, 2:pass, 4:fail, 8:fail | `fc78050fd2ccdcc6a77db9a36ce689a8cbc04cf17a3c18164dda406ba9c43fd8` |
| `vdam-07` small_severe_outliers | `k1-22` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `e425a83b0fcb3b42d830f116f1e295d887b55f7fb7fc29318b2fa8d5e3ab7383` |
| `vdam-08` tiny_baseline | `k1-25` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `b2d674406b8c81eefff32318b321f0950b09cc8dfc80b5bbe83674cd97bff27c` |
| `vdam-09` small_high_res_radial | `k1-20` | FAIL | 0:pass, 1:fail, 2:fail, 4:fail, 8:fail | `b457b94fc2ddd4d29e6be25b3320277cd0b69d8118ea4c4eb87d77617b28acf6` |
| `vdam-10` mid_baseline | `k1-31` | FAIL | 0:pass, 1:pass, 2:pass, 4:pass, 8:fail | `89748dbb669cc1ce5597cf9b45db5b6229d26777628e81a774898e42be57805b` |
| `vdam-11` production_baseline | `k1-01` | FAIL | 0:pass, 1:pass, 2:pass, 4:pass, 8:fail | `ef3266923a7b512bb94365b74623293023ae6f064de50dfe044d9243d24282c9` |
| `vdam-12` production_near_nyquist | `k1-09` | PASS | 0:pass, 1:pass, 2:pass, 4:pass, 8:pass | `6450095ddafee18a67a63ebe7aec64c4815779608a19121d730ce63cca6860f2` |

## Fixed checkpoints

The v1 trajectory checkpoints are iterations `0`, `1`, `2`, `4`, and `8`. Iteration 0 covers bootstrap/reference initialization; later checkpoints cover the complete VDAM schedule, E-step state, pseudo-halfset M-step, and written maps.

Regenerate and validate this page with:

```bash
pixi run python scripts/summarize_vdam_relion_parity_scorecard.py --check
```
