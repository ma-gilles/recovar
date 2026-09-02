# EMPIAR-10076 K=4 shared-200 causal replay

This diagnostic freezes the first K=4 real-data checkpoint at which RELION
and RECOVAR visited exactly the same particles. It replays iteration 1 from
the same four iteration-0 maps, tau/noise/controller state, particle bytes,
poses, CTFs, half labels, and random perturbation. It then joins every
class/rotation/translation candidate before reconstruction and separately
gates the reconstructed class maps with shellwise FSC/FSC-AUC.

This is a causal, fixed-state diagnostic. It is not a gold-standard half-map
refinement, a final-resolution result, or evidence that four biological
classes are stable. Its purpose is to determine whether the real K=4 gap is
already present in candidate scores/posteriors/support, or first appears in
the reconstruction/update boundary.

## Current-source capped replay

Job `13338468` repeated the complete shared-200 discriminator at source
`6ac57e1c7965541b4a49f485ccf511c689879cb0`, after matching RELION's active
gradient-controller pose cap. RELION serializes
`_rlnMaximumSignificantPoses=-1`, but its 3-D gradient path actually limits
the active set to `100 * K`, or 400 poses for this K=4 case. The RECOVAR replay
now records all three values: saved argument `-1`, active cap `400`, and source
`relion_gradient_runtime_default`.

That controller correction closes essentially the entire measured E-step gap:

| Metric | Uncapped replay | Capped replay |
| --- | ---: | ---: |
| Exact candidate records | 771/800 (0.96375) | 798/800 (0.9975) |
| Centered raw-score relative L2 | not used for acceptance | 4.7393e-6 |
| Centered score-with-prior relative L2 | not used for acceptance | 5.2029e-6 |
| Joint posterior relative L2 | 8.3e-2 | 3.2830e-5 |
| Significant-support Jaccard | nonexact | 0.999967922 |
| Joint winner agreement | 1.0 | 1.0 |
| Hard-class assignment agreement | 1.0 | 1.0 |
| Pmax RMSE / maximum error | not retained here | 1.0269e-5 / 3.3110e-5 |

Only two of 800 particle/class records differ, each by one marginal boundary
candidate. The strict fixed gate remains red because it deliberately requires
800/800 candidate identity, support Jaccard 1.0, and posterior row-sum error at
most `1e-7`; the observed values are 798/800, `0.999967922`, and
`7.2458e-6`. These are now the only admissible failures. They must remain
visible until a matched full VDAM trajectory shows whether this boundary noise
has any material consequence.

The measured class-map FSC-AUC values
`[0.683914, 0.929546, 0.740678, 0.788427]` are retained as diagnostic
telemetry, but they are not a parity gate for this harness. The native arm is
explicitly `_rlnDoGradientRefine=1` and executes RELION InitialModel
gradient/VDAM, while `scripts.run_k_class_parity` produces a direct
ordinary-EM reconstruction. Comparing those maps cannot establish or refute
update-rule parity. Native control repeatability, passive-capture inertness,
and native frozen-target replay remain admissible matched-rule checks and have
minimum FSC-AUC `0.999999999286`, `0.999999999282`, and `0.999510174`,
respectively.

The RECOVAR replay took 98.70 s, including 79.61 s for its eight-bucket fine
E+M pass. It staged 0.012 GiB of raw diff2 values and reduced active M-step
rows to 2,888/124,800 (2.3%). The enclosing job took 8 minutes 14 seconds,
with peak batch RSS 8,263,708 KiB. ReqTRES and AllocTRES were identical:
`billing=15,cpu=8,gres/gpu=1,mem=192G,node=1`; no exclusive allocation was
used. The job's exit `1:0` is the expected fail-closed audit result, not a
crash.

The update-rule-aware v7 report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_fine46_maxsig400_6ac57e1c7_20260902/analysis/causal_replay_report_v7_update_rule_audit.json`
(SHA-256
`22cac3ba5c752e436a9019d383a2e3cebae7831ecb4497c7597df8acb4d75f54`).
Its post-hoc manifest is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_fine46_maxsig400_6ac57e1c7_20260902/launch_manifest_v7_posthoc.json`
(SHA-256
`cff7bb3fd2bb8404de7e75c354fe781d2f3ac85883664ccb625249f05b61ace6`).
The science arrays are unchanged from job `13338468`; auditor commit
`cc898f30e103eac17ff68e6b422c6203e6dd9946` only classifies the mixed-rule
map comparison correctly.

Reproduce the update-rule-aware report from the sealed products, choosing a
new output filename because the auditor refuses to overwrite evidence:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_harness_integrate_20260901
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
pixi run python scripts/audit_em_real_k4_shared200_causal_replay.py \
  --manifest /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_fine46_maxsig400_6ac57e1c7_20260902/launch_manifest_v7_posthoc.json \
  --output /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_fine46_maxsig400_6ac57e1c7_20260902/analysis/causal_replay_report_v7_reproduced.json
```

The command intentionally returns 1 while the three strict E-step gates above
remain nonexact.

The failure reported below is retained as historical localization evidence at
source `24317e40c`; it is not the current-source coarse result. Commit
`5f74755c2` repaired RELION's rounded projection-shell handling, and clean
current-source replay job `13337519` now matches all 1,336/1,336 selected
coarse candidates over the 16 frozen probes with projected-reference and Euler
errors exactly zero. Its 11 replay state arrays are bitwise repeatable, while
the eight atomic reconstruction maps have FSC-AUC at least
`0.999999997335`. See
`real_k4_native_coarse_operand_boundary_20260902.md` for the sealed reports and
claim boundary. Fine-score/posterior parity and a complete independent-half
trajectory remain open.

## Completed diagnostic

Job `13322235` ran the six native RELION control/capture arms and the RECOVAR
shared-200 replay from immutable RECOVAR source
`24317e40cc4b0b19406f75c917857c03869a9372`. ReqTRES and AllocTRES were both
`billing=15,cpu=8,gres/gpu=1,mem=192G,node=1`; the elapsed time was 469 s on
`della-h19g1`. The outer job exited `1:0` because its wrapper still required an
optional contribution dump after every scientific arm had completed. The v6
post-hoc auditor removes that stale assertion and consumes the complete sealed
capture set: 800 native fine-score records, 800 native BPref factor records,
800 RECOVAR pass-2 records, and six native data STAR files.

The strict parity gate fails before reconstruction:

| Metric | Result |
| --- | ---: |
| Hard-class assignment agreement | 0.995 (199/200) |
| Minimum class fraction | 0.070 |
| Candidate tuple-set exact fraction | 0.5725 |
| Centered raw / combined score relative L2 | 0.018354 / 0.018491 |
| Joint posterior relative L2 | 0.250740 |
| Posterior row-sum maximum error | 8.812e-6 |
| Significant-support Jaccard | 0.882688 |
| Joint winner agreement | 0.890 |
| Pmax RMSE / maximum error | 0.051736 / 0.458229 |

There are 249 native empty-support records, including six for which RECOVAR
retains nonempty support. Across all 800 particle/class records, 395 have
different rotation counts; the exact union contains 768 native-only and 4,216
RECOVAR-only rotation rows. Every selected coarse parent expands completely to
eight rotation children times four translation children in both engines. After
converting RELION's direction-major coarse rotation IDs into RECOVAR's
psi-major IDs and validating that permutation against every exactly shared
fine rotation matrix, the upstream boundary is:

| Coarse support | Exact records | Aggregate Jaccard |
| --- | ---: | ---: |
| Joint rotation/translation parent pairs | 0.5725 | 0.847195 |
| Rotation parents | 0.8025 | 0.891799 |
| Translation parents | 0.74375 | 0.949900 |

The fine candidate intersection and union are exactly 32 times the joint
coarse-parent intersection and union (504,576 and 595,584 fine tuples versus
15,768 and 18,612 coarse pairs). Thus, the first measured discrepancy is the
coarse significance selection itself; fine expansion preserves that decision
exactly, and the failure is not M-step-only. Most parents still overlap, with
the remaining difference concentrated in marginal parent choices and their
rotation/translation coupling rather than a wholesale orientation mismatch.

The correctly framed RECOVAR-versus-RELION class-map FSC-AUC values are
`[0.604141, 0.906911, 0.693999, 0.748018]`. Native repeatability and passive
capture inertness are both at least `0.99999999928`, and native replay versus
the frozen target is at least `0.999510`. Thus the maps are positively related
and the capture is inert. The cross-engine value is diagnostic-only: this
harness compares RELION gradient/VDAM maps with RECOVAR direct ordinary-EM
maps, so its nominal 0.999 threshold is not an admissible parity gate.
The earlier negative-map interpretation came from loading maps written with
`write_relion_mrc` through the RECOVAR-frame loader a second time; it is not
scientific evidence.

The authoritative report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_causal_v4_24317e40c_20260901/analysis/causal_replay_report_v6.json`
(SHA-256
`2e7e5bb8b2d84b2392b205af6304e73a5c0c52860fe2932ef643477d4cdc4328`).
It was produced by auditor commit
`0ec8b85c5d59f22d5c6c3d1c90de9168845eae77`. The launch manifest is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_causal_v4_24317e40c_20260901/launch_manifest.json`
(SHA-256
`d30c978f933f76af430d2df9da74e571c6633965917e3b5e43795021de2f929e`).
The retained v4 report at `analysis/causal_replay_report.json` has SHA-256
`6c7ac1d0e3229e8257429d02ebec5c1fbdca2d3c2ade4a80754dcc3669d19edc`
and is explicitly superseded because of the map-frame loader error.
The v5 report remains valid for its published score, posterior, support, and
map metrics, but v6 supersedes it as the complete causal report because v5 did
not collapse fine candidates onto their global coarse parents. A pre-commit
v6 replay and the committed replay were byte-identical.

Reproduce that post-hoc report from the sealed GPU products with a fresh
output filename:

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
PIXI_PY="$(pixi run which python)"
"${PIXI_PY}" scripts/audit_em_real_k4_shared200_causal_replay.py \
  --manifest /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_causal_v4_24317e40c_20260901/launch_manifest.json \
  --output /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_causal_v4_24317e40c_20260901/analysis/causal_replay_report_v6_replay.json
```

The command intentionally exits 1 while the strict scientific gate fails;
the complete report is still written before that exit.

## Coarse parent-count counterfactual

The causal replay identifies different coarse parent sets, but it does not by
itself distinguish a different significance cutoff/count from a different
ordering of the coarse scores. Job `13330316` replayed the same frozen
iteration-1 boundary and passively retained RECOVAR's already-computed coarse
score cache for 12 mismatch probes and four exact controls. The capture is at
coarse current size 20; the unchanged fine pass remains at current size 56.
The job requested and received exactly one H100, eight CPUs, and 192 GB,
without exclusivity. It completed in 6 minutes 11 seconds with a peak batch
RSS of 8,051,016 KiB.

For each particle, the counterfactual keeps every RECOVAR score and its stable
class/rotation/translation order fixed, changing only the selected count to
RELION's observed coarse-parent count. No global or per-class counterfactual
landed on an equal-score boundary tie.

| Support metric | Observed RECOVAR | RELION-count counterfactual |
| --- | ---: | ---: |
| Exact particle records, all 16 | 4/16 | 5/16 |
| Exact mismatch probes, 12 only | 0/12 | 1/12 |
| Aggregate particle Jaccard, all 16 | 0.558287 | 0.951790 |
| Aggregate particle Jaccard, mismatch probes | 0.546957 | 0.949618 |
| Exact particle/class records | 29/64 | 50/64 |
| Aggregate particle/class Jaccard | 0.558287 | 0.958944 |

Thus the parent-count difference explains most, but not all, of the support
gap. The remaining disagreement is not a stable-sort artifact: 11 of 12
mismatch particles remain nonexact without a boundary tie. Further causal
work must compare the native and RECOVAR coarse score components/order; merely
copying RELION's significant-parent count is not an acceptable fix.

That next comparison is now complete. The native score boundary and exact
likelihood/prior swap result are documented in
`real_k4_native_coarse_score_boundary_20260902.md`. They localize the first
material support difference to the raw likelihood/`diff2` surface; the priors
are not the cause.

The authoritative report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_significance_probe_r5_24317e40c_20260901/analysis/coarse_score_support.json`
(SHA-256
`d137d59506a859e4b2ddec4c7afd13366645b131174c5f0ea24e58fa617879b2`).
The exact Slurm launcher is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_significance_probe_r5_24317e40c_20260901/jobs/run_significance_probe.sbatch`
(SHA-256
`c70b446601b142ee36dfc09c57d4b0587240a13e1008406961396b4b1c87147c`).
The analyzer joins three deliberately distinct identities: native particle
ID, one-based stack image ID from the causal report, and zero-based row in the
reordered 200-row STAR. The frozen STAR is an explicit analyzer input; integer
equality across those domains is never assumed.

Reproduce the report from the sealed captures with a fresh output filename:

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
pixi run python -m scripts.analyze_em_real_k4_coarse_score_support \
  --significance-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_significance_probe_r5_24317e40c_20260901/significance \
  --causal-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_causal_v4_24317e40c_20260901 \
  --causal-report /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_causal_v4_24317e40c_20260901/analysis/causal_replay_report_v6.json \
  --data-star /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_causal_v4_24317e40c_20260901/inputs/particles_shared200.star \
  --expected-indices 7,9,29,42,53,68,71,82,84,89,91,102,111,166,192,194 \
  --exact-control-indices 82,84,91,102 \
  --output-json /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_significance_probe_r5_24317e40c_20260901/analysis/coarse_score_support_replay.json
```

Probe jobs `13329542`, `13329814`, and `13329956` are harness-only failures:
they used an incompatible current-size or iteration predicate and wrote no
usable target capture. Job `13330159` completed and established that the
diagnostic IDs are reduced-dataset row indices, but its integer targets had
been mistaken for causal/native particle IDs; its science products are
excluded from this result. The successful r5 join resolves causal IDs through
the STAR's `rlnImageName` stack identity before reading any dump.

## Rejected native-BPref reconstruction counterfactual

The coarse/support boundary above leaves open whether RECOVAR's ordinary
batched reconstruction order amplifies otherwise fixed E-step differences.
Job `13327702` therefore replayed the same 200-particle K=4 E-step and staged
the exact sparse posterior slots, raw BPref operands, CTF sign, class order,
and original particle order into an opt-in one-class/one-particle-at-a-time
native reconstruction arm. The job requested and received exactly one H100,
eight CPUs, and 192 GB without exclusivity. Its qualified source commit was
`55c2098654180435a54c74f769086b3554630af0` in the immutable worktree
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_native_bpref_gate_55c209865_20260901`.

The corrected counterfactual is numerically the same reconstruction as the
standard RECOVAR path:

| Metric | Result |
| --- | ---: |
| Minimum native replay vs standard FSC-AUC | 0.999999997101 |
| Maximum native replay vs standard relative L2 | 9.813e-7 |
| Minimum native replay vs RELION FSC-AUC | 0.604141192254 |
| Minimum standard vs RELION FSC-AUC | 0.604141364863 |

Per-class native-versus-standard relative L2 values are
`[9.813e-7, 2.072e-7, 5.984e-7, 4.668e-7]`. The experimental reconstruction
arm therefore reproduces RECOVAR but does not close any material RELION gap.
It is rejected as a production change, and its opt-in implementation is
removed from the integration branch. This negative result shifts further
work upstream to coarse significance scoring/selection.

The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_bpref_gate_439a1049c_20260901/attempt3/analysis/native_bpref_gate_report.json`
(SHA-256
`942d70a68ee82999372713b150883ea310fde7961ddd99c2195dae40c3ee42ef`).
The exact command is sealed at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_bpref_gate_439a1049c_20260901/attempt3/provenance/command_13327702.sh`,
and the complete Slurm launcher is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_bpref_gate_439a1049c_20260901/scripts/run_native_bpref_gate_attempt3.sbatch`.
The first corrected-sign replay report, source commit, maps, allocation audit,
and command remain immutable under the attempt-3 root. Earlier attempts are
not scientific evidence: attempt 1 was cancelled during an unnecessary CUDA
rebuild, and attempt 2 exposed the now-fixed CTF-sign discriminator.

## Frozen case

- Dataset: EMPIAR-10076 frozen 10,000-particle fixture.
- Replay subset: the exact 200 iteration-1 particles assigned by both engines,
  with the deposited frozen split of 93 particles in half 1 and 107 in half 2.
- Controller: K=4, C1, iteration 1, current size 56, HEALPix order 1, seed 0,
  perturbation `-0.07990610599517822`, 50-image batches, and 5,000-rotation
  blocks.
- Native controls: two uninstrumented-output repeats plus one passive capture
  repeat for each class.
- RECOVAR: one all-class replay on a newly materialized deterministic 200-row
  STAR. The reduced STAR is essential: the earlier nominal shared-200 attempt
  still reconstructed all 10,000 rows because its particle list only limited
  diagnostics.
- Hardware request: one H100, eight CPUs, 192 GB, two hours, without
  `--exclusive`.

The launcher pins the fixture STAR/index/34.6 GB stack hashes, shared-target
artifact, frozen pair report, passive RELION capture binary and clean source
tree, clean RELION projector source tree, all four iteration-0 and iteration-1
maps, controller state, and the integrated clean RECOVAR source commit/tree.
The scientific artifacts are fixed independently of the RECOVAR source
commit: after this harness is integrated, the manifest binds whichever clean
qualified staging commit actually runs it.

The native capture default is the clean RELION source commit
`6697bf85a98297153cd57e4485c63c4381548a1c`, tree
`3b940fe717ded8e109364ace1b746ab0164a0874`, with non-MPI executable
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_empty_support_capture_20260901/build/bin/relion_refine`
(SHA-256
`2e109e842c93e34410be219db6ab0e978d4d26e52da0964fea0133d0878f0e84`).
Preflight also requires and hashes
`src/acc/empty_support_capture_contract.h`; a source tree with only the older
stop-after-live-iteration hook is rejected.

The 93/107 split is not an assumed balance. The launcher takes the ordered
`rlnImageName` list from the sealed shared-set JSON, joins those identities
exactly into the sealed fixture STAR, and counts `rlnRandomSubset`. The target
manifest retains both an SHA-256 of the ordered identity list and an SHA-256
of ordered `identity<TAB>half` rows, plus the fixture-STAR and shared-set
digests. Manifest revalidation rereads the materialized STAR and replays these
order and half-assignment digests.

## Dry run and review

From a clean integrated checkout, create a fresh disposable launch bundle:

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
PIXI_PY="$(pixi run which python)"
RUN_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_causal_$(git rev-parse --short=10 HEAD)_20260901
"${PIXI_PY}" -m scripts.launch_em_real_k4_shared200_causal_replay_slurm \
  --output-root "${RUN_ROOT}" \
  --pixi-python "${PIXI_PY}"
```

The default is dry-run only. It writes `SAFE_TO_DELETE`, a deterministic
`inputs/particles_shared200.star`, `inputs/frozen_targets.json`, the sealed
v4 `launch_manifest.json`, and `scripts/run_shared200_causal_replay.sbatch`. It
prints the only admissible submission command, including the manifest digest.
The v4 validator rejects manifests sealed against the superseded capture
binary or source tree. Do not submit if manifest validation or any focused
test fails.

To revalidate a sealed bundle without launching:

```bash
"${PIXI_PY}" -m scripts.launch_em_real_k4_shared200_causal_replay_slurm \
  --validate-manifest "${RUN_ROOT}/launch_manifest.json"
```

## Fixed acceptance policy

The audit requires exactly 800 native fine-score files, 800 geometry-only
BPref factor files, and 800 RECOVAR pass-2 captures. It joins them by immutable
stack, class, exact float32 rotation matrix, and translation identity. The
fixed gates are:

- candidate tuple-set exact fraction 1.0;
- centered raw and combined score relative L2 at most `5e-5`;
- joint posterior relative L2 at most `1e-4` and row-sum error at most `1e-7`;
- significant-support Jaccard 1.0;
- global class/pose winner agreement at least 0.995;
- Pmax RMSE at most `1e-4` and maximum error at most `1e-3`;
- mapped hard-class agreement at least 0.995 and every class at least 1%;
- every matched RECOVAR/RELION class-map FSC-AUC at least 0.999;
- native repeat and passive-capture inertness FSC-AUC at least 0.999999.
- native replay versus the frozen target maps FSC-AUC at least 0.999 and
  native hard-class assignment accuracy at least 0.995; and
- exact native hard-class assignments between the uninstrumented replay and
  every repeat/capture arm.

The coarse-parent exact fractions and Jaccards are diagnostic localizers, not
additional acceptance gates: the existing exact fine candidate-tuple gate
already requires the same topology. The auditor nevertheless fails closed if
either engine does not emit a complete 8-by-4 expansion, if RECOVAR's local
parent map is inconsistent with its global oversampled IDs, or if the native
rotation-order permutation disagrees with any exactly shared fine child.

A particle/class with no RELION sparse support still contributes one file of
each native capture type. Fine-score v1 marks it with header flag word 32 bit
0, sparse-weight word 33 equal to zero, zero candidate/footer counts, and no
candidate payload. BPref v2 marks it with header flag word 54 bit 0, retains
the exact rotation geometry when RELION constructed it (`header[20] ==
header[46]`), retains the full translation grid, and has zero candidate,
hypothesis, pixel, summary, and term counts. The two native sentinels must
agree. A native-empty/RECOVAR-empty record receives empty-set candidate
equality and support Jaccard 1.0. A native-empty/RECOVAR-nonempty record is
retained as asymmetric topology and contributes to the ordinary candidate,
posterior, and support errors rather than being discarded or rejected before
measurement. Missing RECOVAR reconstruction enrichment is permitted only for
a truly empty candidate/posterior record; partial enrichment, an unflagged
zero-rotation record, unknown flag, inconsistent geometry, or mismatched
native sentinels fails closed.

Scores use centered, scale-sensitive errors solely to remove a class-table
additive offset. Posterior and Pmax comparisons are not centered or fitted.
Maps use signed shellwise non-DC FSC and normalized FSC-AUC. Correlation is
never computed. Missing, duplicate, non-finite, malformed, wrong-identity, or
wrong-shape evidence fails before the report is written.

The final machine-readable result is
`analysis/causal_replay_report.json`; `provenance/science_outputs_<job>.sha256`
hashes all native, RECOVAR, and audit products. A failed gate remains a
diagnostic result and is not admitted to the benchmark registry. A passing
result localizes this single iteration but still does not replace the planned
multi-iteration, multi-dataset K=4 half-map refinement evidence.

## Focused checks

```bash
pixi run ruff check \
  scripts/analyze_em_real_k4_coarse_score_support.py \
  scripts/launch_em_real_k4_shared200_causal_replay_slurm.py \
  scripts/audit_em_real_k4_shared200_causal_replay.py \
  scripts/validate_relion_bpref_factor_capture.py \
  scripts/validate_relion_fine_score_capture.py \
  tests/unit/test_launch_em_real_k4_shared200_causal_replay_slurm.py \
  tests/unit/test_analyze_em_real_k4_coarse_score_support.py \
  tests/unit/test_audit_em_real_k4_shared200_causal_replay.py \
  tests/unit/test_validate_relion_bpref_factor_capture.py \
  tests/unit/test_validate_relion_fine_score_capture.py
pixi run pytest -q \
  tests/unit/test_analyze_em_real_k4_coarse_score_support.py \
  tests/unit/test_launch_em_real_k4_shared200_causal_replay_slurm.py \
  tests/unit/test_audit_em_real_k4_shared200_causal_replay.py \
  tests/unit/test_validate_relion_bpref_factor_capture.py \
  tests/unit/test_validate_relion_fine_score_capture.py
```
