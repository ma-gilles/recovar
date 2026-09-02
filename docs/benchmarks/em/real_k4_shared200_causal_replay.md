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
and the capture is inert, but the cross-engine map gate of 0.999 still fails.
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
  scripts/launch_em_real_k4_shared200_causal_replay_slurm.py \
  scripts/audit_em_real_k4_shared200_causal_replay.py \
  scripts/validate_relion_bpref_factor_capture.py \
  scripts/validate_relion_fine_score_capture.py \
  tests/unit/test_launch_em_real_k4_shared200_causal_replay_slurm.py \
  tests/unit/test_audit_em_real_k4_shared200_causal_replay.py \
  tests/unit/test_validate_relion_bpref_factor_capture.py \
  tests/unit/test_validate_relion_fine_score_capture.py
pixi run pytest -q \
  tests/unit/test_launch_em_real_k4_shared200_causal_replay_slurm.py \
  tests/unit/test_audit_em_real_k4_shared200_causal_replay.py \
  tests/unit/test_validate_relion_bpref_factor_capture.py \
  tests/unit/test_validate_relion_fine_score_capture.py
```
