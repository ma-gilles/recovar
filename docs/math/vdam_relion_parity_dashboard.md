<!-- frozen-vdam-parity-scorecard-v3 -->
### Frozen VDAM / InitialModel RELION parity scorecard

This PR carries a fixed-denominator K=1 full-trajectory suite modeled on the
EM PR scorecard. The v3 denominator is **20 cases**, every case spans numbered
iterations **0--200**, and adding or changing cases requires a new suite
version. The checked definition SHA-256 is
`9842b2c9cb7646d75127541801ef5982ed19e4a80485f9ce586ceabdb3ed0091`.

| Fixed K=1 v3 suite | Passed | Evaluated | Denominator | Live science |
|---|---:|---:|---:|---:|
| All quality/state/schedule gates | **2** | 10 | 20 | 17 complete, 3 running, 0 queued |
| Comparable same-H100 runtime | **0** | 10 | 20 | measured range: **5.89--11.58x RELION** |

Progress against the unchanged strict denominator is **0 -> 2 accepted
trajectories**. Earlier expansion v2 remains a separate regression track at
**6/15 accepted** and cannot change the v3 score.

> **Status: draft, not merge-ready.** K=1 correctness is the active gate.
> Runtime, K>1, real-data, and final CLI/GUI qualification follow only after
> the K=1 0--200 suite has no unexplained failures.

Last scientific update: **2026-08-26 18:35 ET**

Tracking branch: `codex/vdam-relion-parity-20260820`

Base: PR #158 (shared supplied-map EM machinery)

Policy: focused VDAM tests and frozen trajectories only; **no generic RECOVAR
full/long suite** is being run for this campaign.

## At a glance

| Gate | Current result | Required to close |
|---|---|---|
| Frozen K=1 v3 quality | **2/20 pass**, 10/20 evaluated | 20/20 accepted with no unexplained case |
| Frozen K=1 v3 production | **17/20 complete**, 3 running, 0 queued | 20/20 science artifacts sealed |
| Earlier 0--200 expansion | **6/15 accepted**, 9 classified failures; GF42 remains outside the sealed count | Every failure repaired and requalified |
| First causal production fix | **151x lower iteration-1 map error**, focused guards 12/12 | Complete GF38 0--200 native-envelope audit |
| Checkpoint-state fix | GF47 iteration-2 pose match **93.8% -> 100%** in sealed replay; focused guards 2/2 | Complete immutable composed 0--200 trajectory and audit |
| Controller-precision fix | GF47 iteration-10 calibrated controller envelope passes all four native repeats; focused guards 6/6 | Complete immutable composed 0--200 trajectory and audit |
| Runtime | **0/10 comparable**; current range **5.89--11.58x RELION** | Comparable same-H100 wall time |
| K>1 / real data | Existing short K=2/K=4 panels pass; final campaign deliberately deferred | Requalify after K=1 closes |
| CLI / GUI | Unified backend/default contract exists | Final defaults and important controls requalified |

### Latest change

GF47's apparent first particle-state split at iteration 2 is now localized to
checkpoint serialization, not the E-step. RECOVAR and RELION select the exact
same 200 particle identities, and native changes exactly those identities.
RECOVAR nevertheless rewrote 186 particles visited only by iteration 1 back to
their input-STAR poses because its `visited` mask was reset to the current
subset. Local unpushed commit `9685e9317` preserves cumulative visitation.
Replaying the sealed iteration-1/2 metadata changes pose match from **93.8% to
100%** (maximum angular difference `9.58e-6` degrees) while translation match
remains 100%. Focused guards pass 2/2. Exact-H100 replacement trajectory array
`13011552`, task 1, wrote all 201 checkpoints, but it is diagnostic-only: the
runner correctly failed its final provenance check because the worktree
advanced from `9685e9317` to `0a001923e` during execution. The RECOVAR process
itself exited 0. No score is promoted from that mixed-lifetime artifact.

GF47 also had a distinct pre-branch controller miss at iteration 10. RELION's
GPU search grid is float32, but persistent particle origins and hidden-variable
monitoring use CPU `RFLOAT` double precision. RECOVAR incorrectly reused the
float32 search translations for both roles. Local unpushed commit `0a001923e`
keeps the GPU grid float32 while preserving float64 metadata translations.
Sealed replay moves the iteration-10 range from `14.077556407` to
`14.077556605`, which serializes to RELION's `14.077557` and falls inside the
unchanged `5.1e-7` gate. Focused guards pass 6/6. Exact-H100 task
`13011940_1` completed on the target GPU; its calibrated iteration-10 mode
matches all four native repeats across every active controller check. The
report SHA-256 is `fea5a1824e0f29a4f516098ba0d58ab7715440a15add2d614afda2c9342fb5bc`.
The composed head is now isolated in an immutable worktree, and full 0--200
array `13013006` is running against the original four-repeat native panel.

The first exact production defect remains repaired locally at unpushed commit
`6387ff7c9`: the oversampling-zero big-JIT preserves RELION's retained coarse
posterior mass. Exact-H100 iteration-1 evidence improves posterior relative-L2
**258x**, noise relative-L2 **343x**, and map relative-L2 **151x**. Full 0--200
science task `13008435` completed all 201 checkpoints naturally in 2,152
seconds on the reference H100. Focused native-envelope audit `13010186` is
pending scheduler priority; the trajectory is not counted as a pass until that
audit is terminal.

### What is still failing

| Failure class | Current evidence | Next closure gate |
|---|---|---|
| Map/particle parity | GF43, GF46, GF48, GF50--GF52 fail; GF49 passes particle state but fails map late; GF47 composed repair is requalifying | classify earliest boundary and repair without changing gates |
| Controller topology | GF47 iteration-10 precision repair is requalifying; GF48 still diverges before an admissible native branch | reproduce RELION schedule decisions exactly |
| Runtime | every audited v3 case is 5.89--11.58x slower | profile only after a repaired trajectory passes end to end |
| Coverage | 10/20 v3 cases not yet audited | finish science and dependent focused audits |

No tolerance, baseline, or denominator has been widened to obtain a pass.
Native-repeat envelopes are used only when RELION itself branches; a candidate
must still remain inside the measured native quality/state envelope and obey
the pre-divergence controller topology.

## Current 20-case K=1 matrix

Every row is K=1 with GUI defaults, iterations **0--200**, four native RELION
repeats on the same physical H100, and an independent RECOVAR perturbation
seed. The audit checks maps, GT quality, particle state, schedule topology,
artifact topology, and wall time.

| Case | Seed | Distribution / stress | Science | Map | Particle | Schedule | Runtime | Overall |
|---|---:|---|---|---|---|---|---:|---|
| GF43 | 29 | baseline, uniform, white noise | complete | fail @73 | fail @91 | pre-split pass | 7.35x | **FAIL; seeded re-audit pending** |
| GF44 | 29 | anisotropic, outliers, high noise | complete | pass | pass | pre-split pass | 8.03x | **ACCEPTED quality; runtime open** |
| GF45 | 29 | Kent, outliers, high noise | seeded run complete | pass | pass | pre-split pass | 9.33x | **ACCEPTED quality; runtime open** |
| GF46 | 29 | anisotropic, severe outliers, radial/high noise | complete | fail @20 | fail @4 | pre-split pass | 9.40x | **FAIL** |
| GF47 | 29 | extreme outliers, uniform, white noise | complete | pass | pass | fail @10 | 6.42x | **FAIL: controller/runtime** |
| GF48 | 29 | very-high noise, uniform, white noise | complete | fail @45 | fail @30 | fail @10 | 6.84x | **FAIL** |
| GF49 | 29 | low noise, uniform | complete | fail @115 | pass | pre-split pass | 11.07x | **FAIL: map/runtime** |
| GF50 | 29 | low noise, Kent | complete | fail @41 | fail @40 | pre-split pass | 11.58x | **FAIL** |
| GF51 | 29 | no CTF, radial noise | complete | fail @74 | fail @39 | pre-split pass | 5.89x | **FAIL** |
| GF52 | 29 | Kent, junk particles, translations | complete | fail @40 | fail @40 | fail @40 | 8.20x | **FAIL** |
| GF53 | 29 | high resolution, radial noise | complete | pending | pending | pending | pending | audit queued |
| GF54 | 29 | midscale, Kent, radial noise | complete | pending | pending | pending | pending | audit queued |
| GF55 | 101 | anisotropic, outliers, high noise | complete | pending | pending | pending | pending | audit queued |
| GF56 | 101 | Kent, outliers, high noise | complete | pending | pending | pending | pending | audit queued |
| GF57 | 101 | anisotropic, severe outliers, radial/high noise | complete | pending | pending | pending | pending | audit queued |
| GF58 | 101 | extreme outliers, uniform, white noise | complete | pending | pending | pending | pending | audit queued |
| GF59 | 101 | very-high noise, uniform, white noise | complete | pending | pending | pending | pending | audit queued |
| GF60 | 101 | low noise, uniform | running | -- | -- | -- | -- | running |
| GF61 | 101 | low noise, Kent | running | -- | -- | -- | -- | running |
| GF62 | 101 | Kent, junk particles, translations | running | -- | -- | -- | -- | running |

The earlier expansion's accepted rows are GF27 (70% outliers), GF29
(low-noise uniform poses after input-orientation seeding), GF35
(`tau2_fudge=2`), GF36 (very-high noise with tau2 variants), GF37 (Healpix 2),
and GF40 (translation range/step 8/2). Its failures are retained as regression
targets; they are not removed from the program when a newer seed matrix is
added.

## Repaired first K=1 boundary: GF38 oversampling-zero posterior mass

The first map drift is no longer an undifferentiated reconstruction failure.
The causal ladder is now:

| Boundary | Observation | Classification |
|---|---|---|
| Iteration-0 maps and iteration-1 incoming moments | bitwise exact over 1,064,960 complex values | input state is not the cause |
| Raw BPref | data rel-L2 0.00263/0.00276; weight rel-L2 0.00162/0.00145 | first nonexact M-step artifact |
| Particle-0 hypotheses | identical 17 retained hypotheses | support/geometry are not the cause |
| Posterior normalization | RECOVAR mass 1.0000000137 vs RELION 0.9986371638 | coarse denominator differs by about 0.136% |
| Native `corr_img` factorial | coarse-score RMS 0.00449659 -> 0.0000362248 (**124x reduction**) | dominant error is the incoming radial score weight |
| Translation/accumulation-expression A/B | no improvement | coarse translation and written reduction expression rejected |
| Shell analysis | within-shell ratios constant to ~4e-8; shell means 0.99991797--1.00019730 | radial inverse-noise construction/state is the current locus |
| Fresh iteration-1 capture | native-vs-candidate score RMS 0.0000327728; native `corr_img` does not improve it | initial score weights are already correct to the residual floor |
| First noise update | iteration-0 noise rel-L2 9.19e-8; iteration-1 rel-L2 1.592e-4 | mismatch is introduced by the update, not initialization |
| High-shell denominator | native inferred sum-weight 199.6766 vs candidate metadata 200.0 | candidate renormalizes retained posterior mass to one |
| Production boundary | fallback local E-step accepts coarse Pmax; big-JIT never receives it | selected os0 hypotheses are silently renormalized only in production |
| Bounded correction | target posterior rel-L2 1.883e-3 -> 7.309e-6; noise rel-L2 1.592e-4 -> 4.645e-7 | cause and first update are repaired |
| Iteration-1 map | scaled rel-L2 2.759e-4 -> 1.822e-6 (**151x**) | promoted full-trajectory discriminator is justified |

The fresh full-schedule discriminator closes the initialization question.
Iteration-1 scoring is already at the same roughly `3.3e-5` residual floor
whether candidate or native `corr_img` is replayed; the 124x native-weight
rescue appears only after later state updates.  The production big-JIT omitted
the coarse Pmax normalization that its fallback path already honored, so the
100 selected os0 hypotheses were renormalized to unit mass.

Local correction `6387ff7c9` threads that normalization through the big-JIT.
Exact-GPU task `13007504_1` changes aggregate `noise_sumw` from `200.0` to
`199.676544`, within `6.1e-5` of the independently inferred native value.
The target posterior mass is `0.99811662` versus native `0.99811831`, with
identical support and argmax.  Iteration-1 noise relative-L2 improves 343x and
the map improves 151x. This implementation remains isolated and unpushed.
Arrays `13007637` and `13008037` exited 75 before science because Slurm
assigned non-reference GPUs. Task `13008278_1` reached the reference GPU but
was canceled before science when a stale-timestamp check attempted to rebuild
the shared qualified CUDA binary; the shared digest remained unchanged. Safe
replacement science task `13008435` used a private, read-only, digest-pinned
CUDA copy and completed all numbered iterations 0--200 with exit 0 in 2,152
seconds. Dependent audit `13010186` is pending against the existing four-repeat
native envelope.

Key sealed evidence:

- M-step boundary: `mstep_boundary.json`, SHA-256 `96302c189a16b463768efedd3cdd4ec35941265b09e1fea787a0bb39af120052`
- Particle posterior: `storewavg_particle0_posterior.json`, SHA-256 `2020b2260deeb25485c46d8c72e5855880820d1032962d195a8111c37966539d`
- Native-correlation factorial: `coarse_projector_boundary.json`, SHA-256 `2ccace9f0bc98b9acd5a5f075fd4d3195931569fce0f9ebcbbe88880aa8640af`
- Fresh iteration-1 score/noise discriminator: `coarse_projector_boundary.json`, SHA-256 `24b2a0e393ebf707cb538afcef6e3ecdc0f09ebbb4506a97ed029055a91ec863`
- Corrected posterior discriminator: `analysis_fresh_fused_posterior.json`, SHA-256 `6b005700b90ea2fe0cc802b6e45e332882118df782580fd2c711105c465455e3`

## Runtime

| Case | RELION median | RECOVAR | Ratio |
|---|---:|---:|---:|
| GF51 no-CTF radial | 281.2 s | 1655.6 s | **5.89x** |
| GF47 extreme outliers | 299.1 s | 1920.9 s | **6.42x** |
| GF48 very-high noise | 286.8 s | 1962.6 s | **6.84x** |
| GF38 oversampling zero, repaired; audit pending | 308.4 s | 2152.0 s | **6.98x** |
| GF43 baseline | 297.3 s | 2186.4 s | **7.35x** |
| GF44 anisotropic/outliers | 286.2 s | 2298.7 s | **8.03x** |
| GF52 Kent/junk/translations | 323.3 s | 2651.3 s | **8.20x** |
| GF45 Kent/outliers, seeded repair | 297.4 s | 2775.8 s | **9.33x** |
| GF46 severe radial/outliers | 466.8 s | 4388.6 s | **9.40x** |
| GF49 low-noise uniform | 406.6 s | 4501.6 s | **11.07x** |
| GF50 low-noise Kent | 436.3 s | 5053.6 s | **11.58x** |

Correctness remains the first gate. Runtime work will start from a sealed
accepted K=1 trajectory so performance changes cannot hide scientific drift.

## Live work and next gates

| Priority | Work | Slurm / state | Exit condition |
|---:|---|---|---|
| 1 | Requalify composed GF47 state/controller repair | immutable head `0a001923e`; exact-H100 full array `13013006` running | complete 0--200 map/state/schedule/native-envelope acceptance |
| 2 | Localize GF46 iteration-4 particle boundary | same-H100 native/RECOVAR capture `13012587` terminal; analysis pending | identify first score/support/posterior divergence for particle 286 |
| 3 | Promote repaired GF38 posterior mass through 0--200 | science `13008435` complete; focused audit `13010186` running | complete map/state/schedule/native-envelope acceptance |
| 4 | Seeded GF29 / GF43 / GF45 calibrated audits | GF29 and GF45 pass; GF43 `13002877` running | authoritative map/state/schedule results |
| 5 | GF49--GF62 trajectory matrix | 17/20 science complete; GF60--GF62 running in `12996103`; GF53--GF58 auditing in `12999424` | every row terminal and sealed |
| 6 | GF41 authoritative re-audit | `12999430` terminal: map pass, particle fail | retain as classified repair target |
| 7 | Local corrections | `6387ff7c9` posterior mass; `9685e9317` cumulative checkpoint state; `0a001923e` metadata precision | full trajectories pass before implementation is published |
| 8 | Runtime, K>1, real-data, CLI/GUI finalization | deferred behind K=1 | comparable runtime and zero suite failures |

Implementation and diagnostic experiments remain in isolated local worktrees.
Only tracking/evidence commits are pushed to this PR while the science change
is unsettled; implementation changes are not pushed without approval.

## Validation policy

- Focused unit/source guards for the VDAM code being changed.
- Same-GPU native-repeat panels and all 201 numbered trajectory checkpoints.
- FSC/FSC-AUC map gates; no correlation fallback.
- Particle-state, controller/schedule, topology, provenance, and runtime gates.
- Exact source, CUDA, RELION binding, executable, fixture, GPU UUID, command,
  job ID, report path, and SHA-256 recorded for every promoted result.
- No generic RECOVAR full or long suite for this campaign, by explicit project
  direction.

Detailed chronological evidence remains in
`docs/math/em_parity_program.md`; this dashboard is the concise PR scorecard.
