# RECOVAR / RELION EM Parity Scorecard

**K=1 fixed-suite score: 25 / 34 passing (34 / 34 evaluated; 31 / 34 intermediate-topology passes).**

Suite: `k1-gui-grid0-local-highshell-full34` (version 1; denominator frozen at 34).
Frozen case-definition SHA-256: `9e3f2cb7192eb2cbf8a50181cf47de8562adfb98734bab05a736fb7d4d404fc1`.

A checked box means the complete autonomous FSC/FSC-AUC trajectory contract passed. Unchecked cases remain in the denominator. New diagnostics do not enter this suite; changing the case set or scientific definitions requires a new suite version.

The artifact-pinned fixture manifest is checked into the repository and binds all 34 cases (470,170,958,467 bytes) to exact file sizes and SHA-256 digests. Manifest SHA-256: `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`. Regenerated inputs are non-scoring replicates.

Acceptance uses shellwise FSC and normalized FSC-AUC, exact schedule/topology, convergence/finalization semantics, same-physical-GPU RELION/RECOVAR pairs, grid correction unset/off, and no forced K-class-like finalization. Correlation is not computed or gated.

Evidence snapshot: `em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v6`, generated `2026-07-24T01:04:11.826284+00:00`, JSON SHA-256 `32c6512a8507f7b17a59d0be527fa5c9609067e0d8f598a2d108bed9a3fc8a56`.
Progress: +5 passing cases since the first frozen snapshot; +2 since the previous snapshot.

| Done | Case | Fixture | Trajectory | Topology | Final cross-engine FSC-AUC | Final GT delta | Jobs |
|---|---|---|---|---|---:|---:|---|
| [x] | `k1-01` | `baseline_100k_g256_white_noise1_bf80` | pass | pass | 0.998379294 | +0.008163340 | science 11384176; trajectory 11384362; intermediate 11384363 |
| [x] | `k1-02` | `more_images_200k_g256_white_noise1_bf80` | pass | pass | 0.998574606 | +0.005625197 | science 11501888; trajectory 11501907; intermediate 11501907 |
| [ ] | `k1-03` | `more_images_300k_g256_white_noise1_bf80` | fail | fail | 0.998786703 | +0.005373891 | science 11384178; trajectory 11384366; intermediate 11384367 |
| [ ] | `k1-04` | `high_noise_100k_g256_white_noise3_bf80` | fail | pass | 0.991556309 | +0.003869282 | science 11384179; trajectory 11384368; intermediate 11384369 |
| [ ] | `k1-05` | `very_high_noise_100k_g256_white_noise10_bf80` | fail | pass | 0.985743479 | +0.000544950 | science 11384180; trajectory 11384370; intermediate 11384371 |
| [x] | `k1-06` | `noctf_control_100k_g256_white_noise3_bf80` | pass | pass | 0.997522945 | +0.005563842 | science 11384181; trajectory 11384372; intermediate 11384373 |
| [ ] | `k1-07` | `anisotropic_100k_g256_white_noise1_bf80` | fail | fail | 0.843316945 | +0.006670338 | science 11384182; trajectory 11384374; intermediate 11384375 |
| [x] | `k1-08` | `anisotropic_high_noise_100k_g256_white_noise3_bf80` | pass | pass | 0.996260789 | +0.001007928 | science 11384183; trajectory 11384376; intermediate 11384377 |
| [x] | `k1-09` | `high_res_near_nyquist_100k_g384_white_noise1_bf0` | pass | pass | 0.995510893 | +0.003664545 | science 11432807; trajectory 11454201; intermediate 11432810 |
| [ ] | `k1-10` | `high_res_anisotropic_100k_g384_radial_noise3_bf0` | fail | pass | 0.983006504 | +0.000128347 | science 11421265; trajectory 11454202; intermediate 11421267 |
| [x] | `k1-11` | `small_baseline_3k_g128_white_noise1_bf80` | pass | pass | 0.998515876 | +0.019981607 | science 11384186; trajectory 11384382; intermediate 11384383 |
| [x] | `k1-12` | `small_very_high_noise_3k_g128_white_noise10_bf80` | pass | pass | 0.998135578 | +0.002422120 | science 11384187; trajectory 11384384; intermediate 11384385 |
| [x] | `k1-13` | `small_anisotropic_3k_g128_white_noise3_bf80` | pass | pass | 0.997569995 | +0.011223852 | science 11385531; trajectory 11385557; intermediate 11385558 |
| [x] | `k1-14` | `small_noctf_3k_g128_white_noise3_bf80` | pass | pass | 0.997775943 | +0.017579713 | science 11385532; trajectory 11385559; intermediate 11385560 |
| [x] | `k1-15` | `small_outliers_3k_g128_pct20_noise1_bf80` | pass | pass | 0.998431014 | +0.019058454 | science 11385533; trajectory 11385561; intermediate 11385562 |
| [x] | `k1-16` | `small_anisotropic_outliers_3k_g128_pct25_noise3_bf80` | pass | pass | 0.996556471 | +0.008243245 | science 11385534; trajectory 11385563; intermediate 11385564 |
| [x] | `k1-17` | `small_extra_particles_3k_g128_noise1_bf80` | pass | pass | 0.998794039 | +0.016078006 | science 11385535; trajectory 11385565; intermediate 11385566 |
| [x] | `k1-18` | `small_contrast_noise_scale_3k_g128_noise1_bf80` | pass | pass | 0.998712222 | +0.014739591 | science 11385536; trajectory 11385567; intermediate 11385568 |
| [x] | `k1-19` | `small_image_offset_3k_g128_noise1_bf80` | pass | pass | 0.998259358 | +0.020106905 | science 11385537; trajectory 11385569; intermediate 11385570 |
| [x] | `k1-20` | `small_high_res_radial_3k_g256_noise3_bf0` | pass | pass | 0.998129368 | +0.001149427 | science 11498687; trajectory 11498738; intermediate 11498738 |
| [x] | `k1-21` | `small_kent_angles_3k_g128_white_noise3_bf80` | pass | pass | 0.998345537 | +0.010110173 | science 11385539; trajectory 11385573; intermediate 11385574 |
| [ ] | `k1-22` | `small_severe_outliers_3k_g128_radial_noise5_bf80` | fail | fail | 0.825938890 | -0.000351848 | science 11385540; trajectory 11385575; intermediate 11385576 |
| [x] | `k1-23` | `small_noctf_radial_3k_g128_noise3_bf80` | pass | pass | 0.998342408 | +0.012298496 | science 11501524; trajectory 11501622; intermediate 11501622 |
| [ ] | `k1-24` | `small_kent_outliers_3k_g128_pct20_noise3_bf80` | fail | pass | 0.994805104 | +0.008173298 | science 11385542; trajectory 11385579; intermediate 11385580 |
| [x] | `k1-25` | `tiny_baseline_1k_g128_white_noise3_bf80` | pass | pass | 0.998192576 | +0.009181804 | science 11385543; trajectory 11385581; intermediate 11385582 |
| [ ] | `k1-26` | `tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80` | fail | pass | 0.954913646 | +0.010098947 | science 11385544; trajectory 11385583; intermediate 11385585 |
| [x] | `k1-27` | `small_extreme_outliers_3k_g128_pct70_noise1_bf80` | pass | pass | 0.998332271 | +0.010086417 | science 11385545; trajectory 11385587; intermediate 11385588 |
| [x] | `k1-28` | `small_kent_extra_offset_3k_g128_noise3_bf80` | pass | pass | 0.998534963 | +0.016603039 | science 11384203; trajectory 11384427; intermediate 11384428 |
| [x] | `k1-29` | `small_low_noise_3k_g128_white_noise0p2_bf80` | pass | pass | 0.998867525 | +0.014987020 | science 11384204; trajectory 11384429; intermediate 11384430 |
| [x] | `k1-30` | `small_low_noise_kent_3k_g128_white_noise0p2_bf80` | pass | pass | 0.998823366 | +0.013967656 | science 11384205; trajectory 11384433; intermediate 11384434 |
| [x] | `k1-31` | `mid_10k_g128_white_noise1_bf80` | pass | pass | 0.998725941 | +0.016924536 | science 11384206; trajectory 11384436; intermediate 11384437 |
| [ ] | `k1-32` | `mid_10k_kent_g128_radial_noise3_bf80` | fail | pass | 0.974500501 | +0.004132488 | science 11384207; trajectory 11384438; intermediate 11384439 |
| [x] | `k1-33` | `max_images_400k_g128_white_noise1_bf80` | pass | pass | 0.999734254 | +0.000244294 | science 11508260; trajectory 11508286; intermediate 11508286 |
| [x] | `k1-34` | `max_images_400k_g128_radial_noise3_nonuniform_bf80` | pass | pass | 0.995757412 | +0.002869240 | science 11384210; trajectory 11384443; intermediate 11384444 |

## Progress history

| Snapshot | Date (UTC) | Commit boundary | Passed | Δ passed | Failed | Not evaluated/error |
|---|---|---|---:|---:|---:|---:|
| `strict-k1-v1-old-head-20260721` | 2026-07-21T04:33:00.281935+00:00 | `ac5177d2b0cd` | 20 | — | 12 | 2 |
| `strict-k1-v3-20260721` | 2026-07-21T10:35:40.626248+00:00 | `ac5177d2b0cd`, `9d1722781e1d` | 21 | +1 | 13 | 0 |
| `strict-k1-v4-20260722` | 2026-07-22T15:57:09.593124+00:00 | `ac5177d2b0cd`, `9d1722781e1d`, `6ddd094011db` | 22 | +1 | 12 | 0 |
| `strict-k1-v5-20260722` | 2026-07-22T19:00:51.329249+00:00 | `ac5177d2b0cd`, `9d1722781e1d`, `6ddd094011db`, `ab52b1ff4038` | 23 | +1 | 11 | 0 |
| `strict-k1-v6-20260724` | 2026-07-24T01:04:11.826284+00:00 | `ac5177d2b0cd`, `9d1722781e1d`, `6ddd094011db`, `ab52b1ff4038`, `84143872a517`, `a2be302cdc08` | 25 | +2 | 9 | 0 |

## Post-snapshot fixed-fixture intervention diagnostics

These rows use frozen case bytes but do not rewrite the immutable snapshot
above.  A failing intervention remains unchecked and does not change the
25/34 score.

| Done | Case | Commit/intervention | Trajectory | Topology | Final cross-engine FSC-AUC | Final GT delta | Jobs |
|---|---|---|---|---|---:|---:|---|
| [ ] | `k1-24` | `b826bc52`; direct-real initial projector + bounded firstiter top-2 tree rescore | fail | pass | 0.994801463 | +0.008173125 | setup 11562037; science 11562038; summary 11562039; strict audit 11562082 |

The case-24 intervention is effectively exact for the first three numbered
maps (merged cross-engine FSC-AUC `0.999999999973`,
`0.999999999903`, `0.999999999901`).  The first material map drift is
iteration 4.  Particle-state audit localizes its seed to one 2.125 Angstrom
translation decision at iteration 3; patched RELION operand replay `11562574`
is pending.  The fixed denominator and score remain unchanged.

## Non-scoring regenerated-data diagnostics

These runs exercise the same parameter definitions with newly generated particle bytes. They are useful robustness evidence but never change the fixed-suite score.

| Case | Trajectory | Topology | Final cross-engine FSC-AUC | Final GT delta | Jobs |
|---|---|---|---:|---:|---|
| `k1-20` | fail | pass | 0.994327835 | +0.001101584 | science 11497499; audit 11498100 |
| `k1-22` | fail | fail | 0.825960160 | -0.000306153 | science 11497555; audit 11497575 |
| `k1-23` | pass | pass | 0.997483478 | +0.012306248 | science 11497556; audit 11497576 |
| `k1-24` | fail | pass | 0.989305857 | +0.008729054 | science 11497557; audit 11497577 |

Generate this PR-ready table with:

```bash
pixi run python scripts/summarize_em_relion_parity_scorecard.py
```

Launch a scoring rerun with `--scorecard`. This fail-closed mode requires the
checked-in fixture manifest/root pair and forces autonomous RELION pairing,
per-iteration RECOVAR maps, grid correction off, and valid convergence-only
finalization. For example:

```bash
EM_K1_MATRIX_FIXTURE_MANIFEST="$PWD/docs/math/em_relion_parity_fixture_manifest_v2.json" \
EM_K1_MATRIX_FIXTURE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex \
EM_K1_MATRIX_CASES=2,3 \
./scripts/run_em_k1_robustness_matrix_slurm.sh --scorecard
```
