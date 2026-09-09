# Current EM development scope

This is the current workboard, not a run history. Read the relevant linked
evidence for the active change. The [September 9 archive](em_cleanup_history_20260909.md)
preserves the previous status verbatim, including failures and superseded next
actions. Earlier history remains in the
[September 8 archive](em_cleanup_history_20260908.md).

## Milestone and invariants

Complete cleanup and professional development setup across RECOVAR, **EM first;
GUI and the new HIA engine are excluded**. Remove demonstrated dead/duplicate
code, clarify ownership/APIs, maintain consistent agent/contributor guidance, and
establish reproducible synthetic, real-data and K-class accuracy/performance
evidence for the selected source. See the [cleanup plan](cleanup_plan.md),
[codebase map](codebase.md) and [benchmark contract](benchmarks.md).

EM APIs may change with callers/tests/docs migrated together. Preserve non-EM
APIs, saved formats, scientific defaults, numerical behavior, reduction order,
JIT boundaries and memory lifetime during structural cleanup. Keep proposed
numerical/runtime repairs separate until authorized.

**Production EM remains float32. Double is diagnostic only.** A smaller double
gap does not prove roundoff. Preserve deliberate higher-precision metadata,
host calculations and necessary non-EM mathematics. Do not widen tolerances,
change baselines, dismiss discrete/convergence mismatches without evidence, or
use map correlation in place of FSC/FSC-AUC. Follow the
[EM operating contract](../../recovar/em/AGENTS.md).

## Source and ownership

| Item | Identity or rule |
| --- | --- |
| Implementation | `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_structural_cleanup_20260907/`, branch `codex/integrate-pr180` |
| Latest production cleanup | `a392cd145`: K1 raw-operand schema shared by selected/full dumps; sparse scorer 17,780 lines, half scorer 1,358, controller 6,123 |
| Authorized performance integration | `5a39eab29` merges compact-CTF `b1d57608d`; host gather before stacking, full-grid default preserved |
| Latest runner guard | `0954fdfd0`: concrete import provenance includes the extracted half-scoring and policy owners |
| Pinned PR158 control | `44d770de3f9336ab2f3f6a34203394bae8d1aeed`; preserve unchanged |
| Incorporated history | PR180 `1e2f229b3`; cleanup anchor `0b52c995a`; VDAM merge `c38fc62f0`; separate global-window correction `0219caa35` |
| Publication | [Draft PR179](https://github.com/ma-gilles/recovar/pull/179), stacked on [PR158](https://github.com/ma-gilles/recovar/pull/158); em_clean is sole integrator/publisher |
| Live coordination | [Shared board](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/README.md); exact scopes/handoffs supersede historical grants |

Read actual HEAD, branch, dirty fingerprint and untracked manifest before tests
or claims. This table is a checkpoint record, not a moving HEAD pointer. WIP
draft publication with incomplete checks is user-authorized; merging and
scientific gates are not waived. The combined branch includes behavioral
integration changes as well as structural cleanup.

em_clean owns shared docs/status and publication. EM remains paused with frozen
source/jobs untouched. User-authorized compact-CTF `b1d57608d` is incorporated by
merge `5a39eab29`, preserving the separate performance commit. All six transferred
files match the frozen candidate byte-for-byte. Combined-source CPU checks pass
241 cases (23 GPU deselected) plus the 38-case guard. Reviewed H100 13638270
compares four synthetic configurations, six pairs each, with all five operands
and coarse scores byte-exact. This is **not trajectory/runtime qualification**.
See `handoffs/em_clean_compact_ctf_integration_20260909.json`; the private
candidate and its binaries remain frozen.

VDAM's separate two-path full-float32-product candidate `907b02ce` is now frozen
and reviewed: exactly two `HIGHEST` contraction keywords differ in production;
all 32 recorded H100 cases pass (13638400), after two expected-red cases
(13638324). Native-cutoff candidate replay 13638431 is diagnostic only. See
`handoffs/em_clean_vdam_full_float32_review_20260909.json`. Prepared-state jobs
13638710/13638814 on that private source now preserve complete incoming/candidate
inputs and particle decisions/Pmax. Noise/BPref arrays vary even within a policy;
the first panel's one-ULP support-sum policy association did not reproduce in the
second. Only the rectangle-power helper executes in this replay; the atomic
fallback has focused-test coverage only. These are not full-state equivalence,
trajectory or runtime acceptance.

VDAM's private composition `fe8472947` carries the same `907b02ce` patch atop
compact-CTF merge `5a39eab29`; only two documentation files separate that parent
from the assigned `cbff0b092` base. The candidate applies cleanly to the current
primary, but is **not adopted**. H100 13639506 completed four natural 200-iteration
old/new/new/old trajectories (0:0, 1,819 s). Whole-process old times are
449.269/454.861 s and new times 456.089/457.051 s: **1.009965×** by mean,
a small-fixture diagnostic only. The completed 804-file/201-checkpoint metadata
ledger is schema-valid but fails numeric comparison at 199 iterations (2–200).
Old-policy repeats first differ in pose at 32 and resolution shell at 45;
new-policy repeats first differ in pose at 52 and keep the same resolution shell.
Their winning margins and convergence implications remain unresolved.

The completed producer FSC review reports old-repeat minimum cross-FSC-AUC
0.969668 at iteration 155, versus 0.998219 for new repeats at 86. The two paired
old/new minima are 0.969722 and 0.998219. These are cross-map diagnostics;
GT registration and historical native source-to-binary closure are missing.
The candidate cross-native minima span 0.998196–0.998767 at iteration 86,
versus a native-repeat minimum of 0.9997155. The second old-policy arm is closer
to both native endpoints than either candidate arm: no uniform candidate win.
The producer reports exact independent agreement for all 15 terminal AUCs and
shell curves; em_clean's review here is the pinned report, not a fresh curve
recomputation. See `handoffs/vdam_full201_fsc_terminal_20260909.json` and
`vdam_f32_full200_20260909/fsc_trajectory_figure/fsc_trajectories.png` in scratch.
em_clean verified the review's seven artifact hashes, without independently
recomputing its arrays. No full-state, trajectory or runtime acceptance follows.
See `vdam_f32_full200_20260909/analysis_integrated_v2_review/result_summary.json`
under the scratch artifact root; review receipt is
`dense_firstiter_arguments_20260909/full200_peer_report_review.json` under the
source-review root below.

Prepared E/M job 13639984 failed in diagnostic callable serialization before
science E/M completion; the failed evidence remains. VDAM's artifact-only v2
repair completed as H100 13640121 (0:0, 145 s) on frozen fe847. Report review
verified five artifact hashes and four completion receipts, without independently
recomputing the numeric array comparisons. It reports exact incoming/compact-CTF
operands and local decisions, but coarse scores vary up to 1.220703125e-4,
including same-policy repeats. Accumulators and six post-state arrays vary;
crossed maxima are not uniformly bounded by the two same-policy comparisons.
Single-boundary cross-map FSC near one does not establish trajectory quality.

Follow-up raw FFI job **13640613** completed on frozen fe847. Independent CPU
recomputation of all 16 saved `(200,576,29)` float32 outputs verifies variation
in all 15 comparisons with repeat 0: maximum 1.220703125e-4, p95 3.0517578125e-5.
All 200 pre-prior raw winners remain exact across 16 runs; the minimum represented
score margin is 0.0013885498046875. Thirteen manifest pins, 17 native source
inputs in both checkouts, the library hash and terminal receipt were checked.
The frozen harness compiles the raw scorer once and uses distinct retained
output buffers; it reports exact negation/max controls on one fixed output.

The captured route is the shared-pretranslated direct float32 FFI, whose CUDA
source merges lane sums with atomics. Recompilation and the later Wavg product
intervention are not necessary for this observed variation. Actual atomic order
was not recorded, and these are raw pre-prior margins, not posterior margins or
evidence for later trajectory flips. No arithmetic change or quality acceptance
follows. The [durable audit archive](evidence/vdam-coarse-repeat-20260909/README.md) preserves
metrics, hashes, the exact CPU audit script and its reproduction command. Original
outputs are under `vdam_coarse_atomic_repeat_20260909` in scratch.

Paired-history H100 job **13641091** completed two fresh old-policy prefixes on
frozen fe847, through M31 and ordinary E32 plus three prepared repeats, stopping
before M32. Both choose rotation111738 for particle1367; signed competing margins
(best163798 minus best111738) are −0.0009765625 and −0.000244140625. Full target
candidate geometry/order agrees (3,360 cells, 256 finite). Target debug payloads
are exact within each history, and all 200 published decisions/Pmax agree across
ordinary/clean/debug calls. Whole E outputs still vary in 17 accumulator/noise
leaves. Incoming model, momentum, noise and priors differ before target preparation;
the cross-history score gap0.00115966796875 is not a same-input arithmetic bound.
Neither fresh history reproduces the original F200 pose flip, whose saved inputs
are unavailable. Reconstructed raw scores are not an independent arithmetic trace.

The bounded producer RNG audit found no scientific consumer of the differing
Python/legacy NumPy globals under these options; target preparation/E32 preserves
them within each history. Native RNG state and independent generators are outside
those snapshots. No global-seeding patch or automatic tie classification follows.
em_clean verified 11 report/manifest hashes, two terminal child receipts and
summary margin arithmetic; it did not independently recompute snapshot arrays or
repeat the full RNG reachability audit. See board handoff
`vdam_paired_history_terminal_20260909.json` and review/reproduction script under
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/paired_history_review_20260909/`.
That prefix audit is now complete: all 64 metadata files for checkpoints 0–31
have matching schemas. Initial stored map arrays are exact; small accumulator,
noise and offset-sum differences appear at iteration 1, followed by support mass
at 2. Significant counts first differ at 18 for particle 1290 (36 versus 35);
no saved pose change occurs through 31. Full incoming E1 operands were not saved,
so initial-map equality does not establish identical E1 inputs or identify the
first divergent operation. The support-count tie remains unclassified.

The producer's momentum audit reports matching update formulas and no near-zero
square-root denominator (minimum 0.98634); the ten largest Fourier-cell differences
account for 98.68% of squared second-moment difference. Neither observation proves
the cause or its effect on poses. em_clean checked both pinned reports and the
prefix CPU completion receipt, without recomputing arrays or independently auditing
the full momentum formula. See `handoffs/vdam_prefix_first_divergence_20260909.json`
and `pass2_raw_schema_20260909/prefix_review.json` under the source-review root.
VDAM's next artifact-only CPU pilot includes translation in GT registration,
fitting one transform on native_1 and applying it unchanged to all six final maps,
with synthetic recovery and null controls. Strict accumulator/tie and quality/speed
gates remain open; no production or native-source change is assigned.

The reviewed runs use **float32 scoring with an existing double-precision
M-step**. Effective precision is stage-specific:

| Stage in the reviewed composition | Effective execution | Evidence/scope |
| --- | --- | --- |
| Captured coarse scoring/projector operands | float32/complex64 | Captured-path evidence, not a claim about every intermediate or route |
| Candidate Wavg products | Two `HIGHEST` contraction keywords on float32 products | Same output dtypes; no M-step precision change |
| JAX VDAM M-step | float64/complex128 numerical computation in both arms | Explicit device casts and host call in [relion_vdam_mstep.py](../../recovar/em/dense_single_volume/helpers/relion_vdam_mstep.py); actual M executed in 13640121 |

The M helper is byte-identical in shared source and frozen fe847 (SHA-256
`0ba75b68202380372e0ffbf777e23bf5b1a82145437723a519429b7507524f2c`).
This is more than metadata precision and is not complete float32 M qualification.
No existing arithmetic is changed or newly accepted by this reporting correction;
the intended production-float32 goal remains. See board handoff
`vdam_effective_precision_boundary_20260909.json` and review receipt
`effective_precision_reporting_20260909/review.json` under the source-review root.
Warm prepared E is 268.953 → 284.035 ms,
**1.056075× (+5.61%)** in this small preloaded-data panel; no full-runtime
acceptance. Only rectangle power executes; full local score surfaces are absent.
VDAM continues the GT-registration pilot described above, keeping both
Wavg helpers fixed. Preserve fe847 and its original evidence; no shared
precision adoption, new arithmetic change or duplicate GPU job. See
`handoffs/em_clean_prepared_em_composition_review_20260909.json` and the producer
`vdam_wavg_composition_v2_20260909/{RESULTS.md,result_summary.json}` under the
scratch artifact root. These runs precede newer structural cleanup and do not
qualify the current primary.
Raw-prefetch source is unassigned. The RELION header lock is
released: five diagnostic insertions remain, and shared benchmark binaries plus
private native captures/builds stay frozen. No shared native writer/build is
assigned. Local GPU 0 is always reserved; check GPUs 1–3 immediately before use
and restrict by idle-device UUID, or use Slurm visibility. Avoid duplicate jobs.

## Engineering work and recent evidence

The controller now delegates dense/local half scoring to `half_scoring.py`;
`scoring_policy.py` owns shared defaults and diagnostic selectors. Scheduling,
state transitions, reconstruction and offloading remain in the controller.
All 17 moved functions and 17 constant expressions are exact under AST review.
Tests migrate direct calls and monkeypatches to their real owners; shared sizing
patches still reach both controller and scorer. No numeric assertion, default,
reduction or stored result class changes. The helper import guard now includes
the policy owner and rejects accidental half-scoring imports.

Controller size falls **7,814 → 6,123 lines (−1,691)**. This is responsibility
separation, not net source deletion: the three owner modules together add 73
lines for explicit imports, module documentation and spacing. The dense routine still needs internal readability work. The subsequent
local cleanup gives 19 identical operand/options keywords one definition across parent,
denominator and final scoring; pass-specific controls stay at each call. It
removes another 31 production lines (half-scoring module 1,453 → 1,422), with no
new layer or numerical expression. Expanded call arguments match the original
module AST after normalizing keyword order; the callee has explicit parameters.
Simple names/constants now bind once per half, and no input arrays are copied.

Support-count reporting now belongs to the existing `local_debug.py` owner.
Its masked/unmasked count calculation is shared by the two log paths. The scorer
loses 52 lines (1,422 → 1,370); the two modules together have no net line change.
Inlining both logging helpers and the shared count helper reproduces every
original scorer statement except imports, including exact logger arguments.
Only temporary host integer count arrays become helper-local; layouts, support
selection, engine calls and scoring buffers remain unchanged.

The K1 and K-class first-iteration dispatches now share 19 identical input
bindings. Mean shape, accumulator options, logging label and rotation-ID
handling stay explicit at the two call sites. The module loses 12 production
lines (1,370 → 1,358); expanding the dictionary reproduces the original whole
module AST after keyword-order normalization. The callee has explicit parameters.
No arithmetic, casts or array buffers change; one host dictionary binds names
and constants when first-iteration CC is selected.

BPref membership capture now shares the existing `helpers/bpref_diagnostics.py`
owner with its numbered-half context. Three helpers, three environment constants
and the single process counter move together; three production calls and tests
use that owner directly. Function bodies and all retained scorer statements are
AST-identical after namespace mapping. Counter sequencing, filter short-circuit
order, NPZ schema, casts and error behavior remain unchanged. Sparse scoring
loses 176 lines (19,460 → 19,284); the two modules together add one line.

| Checkpoint | Executed evidence | Limit |
| --- | --- | --- |
| K1 raw schema `a392cd145` | Same 79 CPU cases before/after (7.75/7.68 s), guard 38/38 (43.83 s), zero skips/Ruff findings | Includes four new selected/full × source-float32/float64 round trips that pass before the change; expanding both constructor calls reproduces the original module AST. No GPU/trajectory/runtime claim |
| Pass-2 diagnostic owner `f401cbb09` | Identical 76-case CPU inventory before/after (6.84/6.76 s), extended import/CPU guard 38/38 (45.37 s), no skips | Five function/three constant ASTs exact; all retained scorer statements and 260 test assertions preserved under owner mapping. No new Ruff findings; no GPU/trajectory/runtime claim |
| BPref membership owner `2b6cf430c` | Original 32/32 plus 11 new cases pass before source changes; candidate 43/43 (4.06 s), guard 38/38 (44.08 s), no skips | Every original case retained; 3 function/4 state ASTs exact. No new Ruff findings (existing sparse import I001 remains). Ownership only, no GPU/trajectory/runtime claim |
| First-iteration arguments `3970ea71d` | Control 75/75; four K1/K4 × batch-update variants pass before source changes; candidate 78/78 (4.72 s), guard 38/38 (44.78 s), no skips/Ruff findings | 74 original IDs retained; one case expanded to four. Initial new K4 fixture incorrectly expected the K1 cap700, then corrected to existing K4 cap368 and rerun before source edits. No GPU/trajectory/runtime claim |
| Support diagnostics `5b42961e7` | Control 32/32; candidate 39/39 (3.51 s), CPU guard 38/38 (44.88 s), no skips or Ruff findings | One static guard replaced by five exact-log cases, three denominator cases added; other 31 identities unchanged. Reporting ownership only, no GPU/trajectory/runtime claim |
| Local-scoring arguments `19073ab50` | Control 42/42; candidate 47/47 (9.95 s), including six denominator/spectrum variants and existing exact-K4 dispatch; CPU guard 38/38 (44.30 s); no skips or Ruff findings | 41 original case IDs unchanged, one original case expanded to six; two source guards resolve the common kwargs; no GPU/trajectory or timing-gain claim |
| Runner provenance `0954fdfd0` | Four missing/foreign scorer-source regressions fail before the fix; all 8 CPU unit/CLI-entry cases pass after (8.09 s), no skips | Two checked module names added; no scientific operation changed or GPU job launched |
| Half-scoring ownership `650d71a43` | Same 189-case CPU inventory before/after; final unused-import follow-up 6/6; extended CPU guard 38/38; exact function/constant and test-assertion audits | Intermediate stale owner guard failed once and was migrated; logger namespaces follow owners; no GPU/trajectory claim |
| Compact-CTF integration `5a39eab29` | 241 CPU passed/23 GPU deselected (31.13 s), 38-case guard; all peer cases retained and six transferred files exact; peer H100 13638270 byte-exact operands/scores | Full-image powerClass and source precision preserved; allocation microbenchmark only, no end-to-end or trajectory claim |
| Reference-replay ownership `b3b51f0c8` | Same 39-case control/candidate inventory, including tiny K4 map loading and state-swap order; 38-case CPU guard. Exact function/retained-controller AST; no new Ruff findings | Controller −95 lines by relocation; logger namespace follows replay owner; no GPU or trajectory claim |
| InitialModel test repair `44d414a6b` | Fourier-window control 49 passed/3 failed, then 52 passed (8.73 s); coarse-audit control 8 passed/4 failed, then 12 passed (2.75 s). Identical inventories; four in-memory native-contract mutations rejected; no new Ruff findings | CPU/test-source contracts, not compiled CUDA or trajectory qualification |
| Replay-order ownership `4da8954ae` | Same 24-case control/candidate inventory, 493 deselections; 38-case CPU guard. Separate parser-caller repair `1895bc50e`: 28 passed | AST-exact relocation; controller −73 lines, not net deletion |
| Diagnostic selectors `acbf6545e` | Same 93-case control/final inventory; 38-case guard; AST-exact relocation | Controller −126 lines; diagnostic logger namespace follows owner |
| Unused helpers `b52a78705` | Eight functions removed, 202 production lines; same 309 passed/1 opt-in CUDA skip/10 GPU deselections before/after | Skip unqualified; combined branch remains larger than PR158/PR180 |

Static review covered 1,404 tracked non-GUI Python files, 3,229 explicit EM
import names and 2,213 accesses through imported EM modules. Both stale callers
were confirmed by failing tests before migration. This does not prove dynamic
imports/registration or every runtime path. A tracked-reference scan found no
further unreferenced, undecorated top-level private EM function candidates;
do not remove code merely to reduce line counts.

Pass-2 score, K-class, norm-residual and chunked scale-AA dump writers now live
in `helpers/pass2_diagnostics.py`, with their target-row selector and three
environment constants. Sparse scoring calls this owner at the same boundaries;
scheduling, operand materialization, JIT kernels and production reductions stay
in the scorer. Five function bodies and three constant definitions are AST-exact;
all NPZ schemas, casts, diagnostic reductions and errors are preserved. The
scorer loses 1,504 lines (19,284 → 17,780); combined modules add 20 lines of
imports/documentation, so this is an ownership improvement, not net deletion.

K1 selected/full captures now share `_k1_raw_operand_fields`, which owns their
common ten-field raw schema. Field order, row indexing, casts and the two names
for the same raw-score array are preserved; inlining both calls reproduces the
original module AST. Four round-trip cases cover source float32/float64 arrays,
the second batch row, padded rotations, nonconsecutive row selection, exact
dtypes, shapes and bytes. This removes duplicate schema definitions but adds six
production lines for the helper boundary/documentation (1,524 → 1,530).
Audit and exact test commands are in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/pass2_raw_schema_20260909/validation.json`;
CPU logs/XML use `pass2_raw_schema_{control,after,final,guard}_20260909` under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/`.
Only the unit marker was added after the first before/after panel; test bodies
are unchanged, and the final panel/guard use identical source manifests.

Pass-2 source/caller audit and exact reproduction commands:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/pass2_diagnostics_owner_20260909/validation.json`.
CPU logs/XML are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/pass2_diagnostics_owner_{control,after,guard}_20260909/`.
All panels preserve their source/native hashes. The private907 patch still
apply-checks without applying; VDAM's frozen diagnostics remain untouched.

Membership-owner source/caller audit, exact commands and test inventories:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/bpref_membership_owner_20260909/validation.json`.
CPU logs/XML use `bpref_membership_owner_{control,expanded_control,after,guard}_20260909`
under the CPU run root below. Source/native hashes stay fixed during every panel.
Explicit module reloads now follow the diagnostic state owner; frozen sources
and existing output formats are untouched.

The first-iteration audit
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/dense_firstiter_arguments_20260909/validation.json`
records exact commands, test identities, failure history and fingerprints; logs/XML use
`dense_firstiter_arguments_{control,expanded_control,expanded_control_v2,after,guard}_20260909`.
The prior support-reporting audit is
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/local_support_diagnostics_20260909/validation.json`;
CPU logs/XML use `local_support_diagnostics_{control,after,guard}_20260909` under
the CPU run root. Masked/unmasked/empty support and explicit empty/repeated-ID
selections have exact log assertions. Final guard uses the final source.

Local-scoring AST/case audits, test commands and source fingerprints are in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/local_scoring_arguments_20260909/validation.json`;
CPU logs/XML use `local_scoring_arguments_{control,after,guard}_20260909` under
the CPU run root. Only the new test resolver's whitespace changed after checks;
its complete module AST stayed identical. No new GPU jobs were launched.

Runner provenance rejects a foreign or unlocated `half_scoring`/`scoring_policy`
module even when the controller comes from the expected checkout. Evidence and
exact CPU commands are in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/runner_scoring_provenance_20260909/validation.json`;
logs/XML use `runner_scoring_provenance_{red,green}_20260909` under the CPU root.
The original four failures remain recorded. Existing assertions are preserved;
Ruff has the same two inherited findings under identical configuration.

Half-scoring commands, source fingerprints, the original owner-guard failure,
case inventories and AST audits are in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/half_scoring_owner_20260909/validation.json`.
CPU logs/XML use `half_scoring_{control,extra_control,after,final,import_cleanup,guard}_20260909`
under the CPU run root below. No new GPU job was launched.

Compact-CTF integration commands, fingerprints and case inventories are in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/compact_ctf_integration_20260909/validation.json`.
Its CPU logs use `compact_ctf_combined{,_guard}_20260909` under the same CPU run
root below. No duplicate GPU job was launched.

Reference-replay commands, fingerprints and AST/case inventories are in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/reference_replay_owner_20260909/validation.json`.
Control/after/guard logs use `reference_replay_{control,after,guard}_20260909`
under the CPU run root below. The previous test repair’s commands, fingerprints,
original failures and mutation/AST audits are in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/em_import_audit_20260909/validation.json`.
CPU logs/XML are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/{stable_window_import,coarse_audit_import}_{control,after}_20260909/`.
Use fresh output roots when reproducing recorded commands. Earlier evidence is
in the [dated archive](em_cleanup_history_20260909.md); the original full-code
review and decisions remain under `hia_source_review_20260906/`.

Next, continue controller/state/kernel ownership and duplicate-code review with
bounded changes and proportional tests. Parsers with different blank/unknown-
token semantics are not interchangeable. Keep production source frozen during
its tests/jobs. Group changes into checkpoints for broader qualification; do
not run completion workloads after each helper edit.

## Unresolved validation gates

**The selected source is not scientifically or performance qualified.** Historical
checks qualify only their recorded source. This consolidation waives no failure,
skip, baseline or tolerance.

| Evidence or gap | Current interpretation |
| --- | --- |
| Frozen combined `d21f52d72` shared CPU | 194 passed, 3 GPU deselected; selected contracts, not complete shared workflows |
| H100 native 13634222 | 222 passed, 1 skipped; inherited rectangular CUDA/JAX ordering case remains unqualified |
| H100 API 13634313 | 592 passed, 12 failed; nine original failures have targeted follow-ups, not a fresh full-panel pass |
| Normalization 13636581 | Dtype migration CPU 8/8; GPU 4 passed/4 failed bytewise spectrum cases. Strict failures remain |
| Normalization replay 13636814 | Exact first-bucket inputs; ordinary results vary in 11/12 same-setting and 14/16 crossed pairs, max norm gap 0.00018310546875. Captured deferred equality does not erase uninstrumented failures; diagnostic precision lane |
| PR180 CPU 13623235 | 6,452 passed/340 skipped/25 failed; targeted repairs/environment reruns do not create a fresh suite pass. [Reconciliation](evidence/pr180-unit-checkpoint-20260908/README.md) preserves CPU-only failure and inventory mapping |
| K1 matched-noise replay | Six Pmax rows still fail; candidate margins and generating oracle identity are missing. [Paired failures](evidence/pr180-k1-noise-state-20260908/README.md), [targeted captures](evidence/pr180-k1-targeted-capture-20260908/README.md). Maps cannot override particle failures |
| Synthetic/real repeatability and global window | Current float32 K1 support/Pmax/pose/convergence and real10076 repeatability remain unresolved; full FSC acceptance of `0219caa35` is pending |
| Exact K4 and shared workflows | No selected-source ≥100k/256 K1 + exactly-K4 same-GPU completion pair; K2/K15 proxies do not count. Shared SPA/ET, outlier/downstream and speed/memory coverage remain outstanding |

Completion requires FSC curves/FSC-AUC against GT and RELION, tie-aware discrete
comparisons, exact convergence/finalization, Hungarian matching and per-class K4
results, identical fixtures/seeds/masks/initial maps and matched GPU models.
Close synthetic K1 trajectory parity, then a characterized real-particle check,
then exact K4. Preserve the reviewed final-grid-correction default during cleanup;
its difference from strict parity requires separate scientific work. Quantitative
gates remain in the [program archive](../math/em_parity_program.md);
[parity notes](../math/relion_parity_agent_notes.md) and
[completion records](../math/em_parity_best_metrics.md) retain detailed evidence.

## Frozen jobs and representative performance

Slurm was checked on September 9 UTC during the precision review: K4 pair
**13560202** was running on `della-l08g4`, audit **13560356** pending. K1 real-data
window **13610518** was running on `della-h20g1`, with **13610539** and
**13629099** pending. The recorded poll is in
`hia_source_review_20260906/precision_prepared_review_20260909/jobs.txt`. Poll these exact jobs before using this status as current.
Preserve their frozen sources/outputs; they do not automatically qualify later
commits. No new jobs were launched for this audit.

Both individual timing pairs on frozen `8ab1a44be` are now terminal. Each pair
used the same physical A100 for RECOVAR and RELION, in opposite execution orders:

| Pair | RECOVAR whole process (s) | RELION whole process (s) | Ratio |
| --- | ---: | ---: | ---: |
| Forward | 13,125.865 | 7,966.705 | 1.647590× |
| Reverse | 14,399.809 | 7,070.191 | 2.036693× |

The geometric mean is **1.831839×**, with changing shared-host contention;
it does not isolate an intrinsic speed factor. All four individual completion
markers, return codes, matching per-pair GPU UUIDs and 16 receipt hashes were
checked. The original forward five-arm controller lacks a final summary despite
its two successful individual arms. These runs predate compact CTF and the
precision proposal; no current-source speed or quality acceptance follows.
Native CLI-only timing is unavailable, so both ratio terms use whole-process
time. Exact commands, timestamps and limitations are in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_speed_monitor_20260909/refresh_20260909T043656Z/report.json`;
board review: `handoffs/em_clean_vdam_prepared_ack_and_paired_timing_20260909.json`.
Large inputs and binaries were not rehashed for this timing review.

The separate instrumented profile took
14,764.576 s. Backend compilation was **195.662 s / 1.33%** of profiled exclusive
time (excluding tracing/lowering). Prefetch queue waiting was 5,777.491 s; CTF
host `np.stack` was 3,709.627 s exclusive versus 69.060 s in the native CTF
binding. The helper's 4,557.538 s cumulative overlaps its children. These costs
are not guaranteed removable time; queue waiting is not proven disk I/O.
The profile does not isolate removable time or qualify current-source speed/quality.

The frozen 21-checkpoint cross-engine diagnostic reports final FSC-AUC
**0.9924569** and minimum **0.9900306 at iteration 70**. Iteration 40 is the first
saved checkpoint below 0.999, not the first divergence or an executed acceptance
gate. Only iterations 0,10,…200 are available. Initial cross-FSC 1 does not imply
bit-identical initial maps. Raw GT metrics are invalid for quality because
registration is unverified; the later shared rotation/mirror check omitted
translation search and did not establish adequate registration. Exact curves,
commands and producer provenance are in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_speed_quality_inventory_20260909/refresh_030533/`;
see board acknowledgment `em_clean_vdam_cross_fsc_ack_20260909.json` for limits.
The 8.68 → 1.74 ms compact-CTF CPU allocation measurement is a microbenchmark,
not an end-to-end runtime prediction.

Native build **13636510** completed; bounded capture **13636553** obtained exact
inputs/residual rows, then deliberately stopped. Wavg replay must use masked
`Fimgs` planes 0/1, not unmasked planes 2/3. Later private arithmetic diagnostics,
including the full-float32 cutoff-product proposal, remain outside trajectory
acceptance and structural cleanup.

Report and hashes:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_native_residual_audit_20260909T024139Z/report.md`,
plus `handoffs/em_clean_vdam_terminal_100k_ack_20260909.json` on the board.
The new engine remains a later milestone.
