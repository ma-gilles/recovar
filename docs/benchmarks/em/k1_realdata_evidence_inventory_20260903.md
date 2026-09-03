# PR #158 K=1 real-data evidence audit

Captured at `2026-09-03T12:08:55Z`. This was a read-only audit of existing
evidence. No live job, source checkout, Git branch, or GitHub object was
mutated.

## Bottom line

PR #158 has **zero completed scoring cases out of one**. The sole scoring case,
EMPIAR-10202 set 6 under I1 symmetry, remains pending because no natural-final
RECOVAR half-map pair exists at the frozen scorecard commit
`6e414838463e30e977114f3f5ffe93a2e232766c`.

EMPIAR-10073 and EMPIAR-10345 are sealed, natural-final calibration passes.
EMPIAR-10097 is a sealed natural-final calibration with good within-engine
half-map agreement, but it is cross-engine unqualified: its raw resolved-band
merged/half-1/half-2 AUCs are `0.935736/0.885803/0.888938`, below the frozen
`0.95/0.90/0.90` gates. A proper-SO(3)+translation diagnostic also fails.

EMPIAR-10076 has no reportable two-engine full-native result. Job `13093156`
is hung/final-incomplete, not merely slow: it remained RUNNING beyond
`5-22:42`, the scientific log had not advanced since
`2026-08-29T08:57:31-04:00`, GPU utilization was `0%`, and main PID `2020342`
was sleeping in a futex/JAX PyArray-readiness path with RSS `151974696 KiB`.
It was observed only and was not cancelled. The older 10k/128 pilot is
explicitly invalidated because RECOVAR received an unconverted RELION-native
initial reference, the exact negative of the intended canonical reference.

## Inventory

All FSC values below use the scorecard's first three-consecutive-shell crossing
below `1/7` and the shared resolved band, except the explicitly invalidated
10076 pilot, whose archived values use its older first-shell policy.

| Dataset / lineage | Endpoint and role | Result | Direct unmasked half-FSC | Corrected masked half-FSC |
| --- | --- | --- | --- | --- |
| 10073 native | Natural final; calibration | PASS, not scoring | crossings `81/81` (`6.568/6.568 A`); AUC `0.717354/0.718615`; delta `0.001260`; RMSE `0.002646`; cross-engine merged/h1/h2 `0.969136/0.940802/0.941530` | crossings `128/130` (`4.156/4.092 A`); AUC `0.718514/0.720115`; delta `0.001600`; RMSE `0.004180` |
| 10345 native | Natural final; calibration | PASS, not scoring | crossings `49/49` (`8.235/8.235 A`); AUC `0.672507/0.672673`; delta `0.000165`; RMSE `0.003242`; cross-engine `0.975610/0.965867/0.965089` | crossings `77/77` (`5.240/5.240 A`); AUC `0.701143/0.700830`; delta `0.000313`; RMSE `0.004790` |
| 10097 native | Natural final; calibration | Within-half PASS; cross-engine FAIL/unqualified | crossings `44/45` (`7.622/7.452 A`); AUC `0.691593/0.690092`; delta `0.001501`; RMSE `0.004926`; cross-engine `0.935736/0.885803/0.888938` | crossings `58/59` (`5.782/5.684 A`); AUC `0.735700/0.733587`; delta `0.002113`; RMSE `0.008370`; supporting only |
| 10076 native | RELION natural final; RECOVAR hung at iter 2 | FINAL-INCOMPLETE / UNSEALED | Missing RECOVAR final maps and comparison | Missing |
| 10076 10k/128 pilot | Natural endpoints, wrong initial frame | INVALIDATED | Archived crossings `29/35` (`14.229/11.790 A`); not rescored under frozen policy | Missing / not applicable |
| 10202 final | RELION natural final; RECOVAR missing; sole scoring case | PENDING | No final pair; RELION-only unmasked estimate `2.511554 A` | RELION-only support `2.122559 A`; cannot score |
| 10202 iteration 11 | Matched fixed checkpoint | SEALED DIAGNOSTIC ONLY | crossings `250/250` (`2.521600 A`); AUC `0.873278/0.873530`; delta `0.000252`; RMSE `0.004634`; cross-engine `0.991225/0.987207/0.986915` | crossings `248/248` (`2.541935 A`); AUC `0.878330/0.879358`; delta `0.001028`; RMSE `0.011651` |
| 10202 corrected iteration 16 | Matched fixed checkpoint; source still nonfinal | SEALED DIAGNOSTIC ONLY; commit mismatch | crossings `281/281` (`2.243416 A`); AUC `0.846775/0.847366`; delta `0.000590`; RMSE `0.005195`; cross-engine `0.985819/0.979634/0.980308` | crossings `280/277` (`2.251/2.276 A`); AUC `0.858614/0.859459`; delta `0.000846`; RMSE `0.005381` |

In paired values, RECOVAR is listed before RELION. In cross-engine triplets,
the order is merged, half 1, half 2. Masked results are supporting-only.

## Runtime and HBM snapshot

| Lineage | RECOVAR | RELION | Analysis |
| --- | --- | --- | --- |
| 10073 | `92798 s`, `44987 MiB` | InitialModel `6277 s`, `80329 MiB`; refine `12012 s`, `80823 MiB` | Full Slurm job `13093155`, `1-06:53:13`; masked job `13273806` |
| 10345 | `90123 s`, `35265 MiB` | InitialModel `2419 s`, `80327 MiB`; refine `5011 s`, `79969 MiB` | Full job `13093157`, `1-03:06:52`; masked job `13273806` |
| 10097 | `235120 s`, `34659 MiB` | InitialModel `3826 s`, `80325 MiB`; refine `8416 s`, `79811 MiB` | Full job `13124450`, `2-20:43:39`; masked `13274377`; SO(3) `13276576` |
| 10076 native | Hung job `13093156`, `>=5-22:42`, sampled `17058 MiB`, no final | InitialModel `3568 s`, `80327 MiB`; refine `76460 s`, `80311 MiB` | None |
| 10076 invalid pilot | `23431 s`, `16963 MiB` | InitialModel `231 s`, `80313 MiB`; refine `507 s`, `79599 MiB` | Job `13068730`, `06:30:52`, terminal failure after science stages |
| 10202 final | Exact-subject attempt `13356985`: `08:43:50`, `76329 MiB`, OOM in iter 12 | Job `13217551`: `14:10:07`, two GPUs, `78392 MiB` per GPU; postprocess `13254149`: `00:03:06` CPU | Final pair missing |
| 10202 iter 11 | Source job `13339556`: `07:51:32`, `75849 MiB`, OOM in iter 12 after checkpoint | Same full RELION run | Full-FFT scorer `13373359`: `00:02:04` CPU |
| 10202 corrected iter 16 | Live job `13376414` snapshot: `06:18:13`, `68509 MiB`, not converged | Same full RELION run | Direct `13381233`: `00:01:38` CPU; masked `13381344`: `00:05:31` CPU |

## Reproduction and authority

The authoritative scorecard replay completed with exit `0`:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_origin_docs_8cbebdecc_20260902
/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_origin_docs_8cbebdecc_20260902/.pixi/envs/default/bin/python -B scripts/summarize_em_k1_realdata_science_equivalence.py --verify-calibrations --verify-masked-support --verify-target-partial --check-markdown --output-json /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_realdata_scorecard_audit_20260903T114754Z/authoritative_replay_report.json
```

The exact native-run submissions were:

```bash
sbatch --export=ALL,DATASET_ID=10073 /home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution/scripts/run_dataset_native.sbatch
sbatch --export=ALL,DATASET_ID=10076 /home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution/scripts/run_dataset_native.sbatch
sbatch --export=ALL,DATASET_ID=10345 /home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution/scripts/run_dataset_native.sbatch
sbatch --parsable --export=NONE /home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution_replacement_10097_20260828T211155EDT/scripts/run_dataset_native_10097.sbatch
```

Exact per-engine command artifacts, all map paths and SHA-256 values, job
resources, and checkpoint reproduction scripts are in:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_realdata_scorecard_audit_20260903T114754Z/inventory.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_realdata_scorecard_audit_20260903T114754Z/inventory.csv`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_realdata_scorecard_audit_20260903T114754Z/authoritative_replay_report.json`

The authoritative scorecard source is
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_origin_docs_8cbebdecc_20260902/docs/math/em_k1_realdata_science_equivalence_scorecard_v1.json`.

The original masked-analysis submission argv for 10073, 10345, and 10097 was
not recorded. The launchers and drivers are hash-pinned, but this audit does
not invent an original argv.
