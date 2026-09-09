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

**Do not switch production EM to double to close parity gaps.** Double replay
is diagnostic; a smaller gap does not prove roundoff. The existing VDAM numerical
M-step already uses float64/complex128 and must be reported separately from its
float32/complex64 E-step. Preserve that boundary until a separate float32 M-step
change is qualified, alongside deliberate higher-precision metadata, host
calculations and necessary non-EM mathematics. Do not widen tolerances,
change baselines, dismiss discrete/convergence mismatches without evidence, or
use map correlation in place of FSC/FSC-AUC. Follow the
[EM operating contract](../../recovar/em/AGENTS.md).

## Source and ownership

| Item | Identity or rule |
| --- | --- |
| Implementation | `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_structural_cleanup_20260907/`, branch `codex/integrate-pr180` |
| Latest production cleanup | `19544d3b6`: normalization inputs are prepared separately from local scoring; local engine 8,879 lines, sparse scorer 17,383, half scorer 1,358, controller 6,123 |
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

VDAM precision907 remains private. The [dated numerical review](vdam_precision_review_20260909.md)
preserves fixed-input E/M, raw-FFI repeats, paired histories and registered-GT
results, including failures and limits. The two float32 `HIGHEST` product changes
pass focused tests; both reviewed policies retain an existing float64/complex128
numerical M-step. No all-float32 M qualification or silent precision change follows.

The completed 3k/128 panel covers six histories and 201 checkpoints using one
frozen native1-final GT transform. em_clean independently recomputed all 2,412
saved-curve AUCs and eight RECOVAR/native summaries. Old1 first misses the unchanged
−0.002 GT condition at 155 (minimum −0.004248842); new1, new2 and old2 satisfy it
throughout against both natives. This is not a uniform policy win. Held-out shells
are descriptive, with no separate threshold. Registration/native provenance limits,
199 metadata comparison failures, repeat pose/support changes and cross-engine
FSC gaps remain. The detailed record names exactly which arrays, summaries and
hashes were independently checked. No quality acceptance or source adoption.

The full small-fixture old/new runtime ratio is 1.009965×; prepared E is 1.056075×.
Neither measures current representative runtime. Preserve frozen fe847, its native
libraries and all failed/superseded diagnostics. VDAM's separate private assignment
was opt-in rigid reporting only, integrated separately below; future numerical admission requires review
of the remaining state/tie and source/native gates. No duplicate peer GPU experiment
is assigned.

The later private full RELION executable `6c54d2ac…` has recorded source/build
identity for new diagnostics (CPU build 13636510); em_clean verified 54 linked
material hashes across its inventory and the reporting proposal. Historical
`2d070d64…` closure remains missing: the later build cannot qualify that binary.
Read the exact audit scope in `handoffs/em_clean_reporting_oracle_review_20260909.json`.
Peer replacement job 13642331 completed 0:0, four full 200-iteration arms on one H100;
Slurm terminal state and the saved-curve analysis are independently checked.
The registered-GT condition fails once at iteration 155; cross-engine FSC gaps remain.
Whole-child RECOVAR/native time is 1.468982× on this 3k/128 fixture. See the
[dated review](vdam_precision_review_20260909.md#source-closed-four-arm-panel).
Failed launcher 13642161 is preserved. No duplicate job or rebuild was launched.
The new run uses frozen fe847, including the existing double numerical M-step;
it does not qualify the current primary or all-float32 execution.

The saved-spectrum resolution follow-up verifies the active rule over iterations
139–156: candidate 2 SSNR at shell 27 falls below 1 at 145 and154, producing size 72
instead of 74 at 146 and155. Integrator review independently checks 166 named hashes,
72 serialized scans and 68 next-size links. This explains those branches from the
incoming spectra; it does not establish the spectra's upstream cause or a causal
link to the GT loss. The eight counter differences from a replay omitting sampling
resets remain recorded; full convergence/finalization parity is not established.
See the [resolution review](vdam_precision_review_20260909.md#saved-spectrum-resolution-boundary).

Raw-prefetch source is unassigned. The RELION header lock is
released: five diagnostic insertions remain, and shared benchmark binaries plus
private native captures/builds stay frozen. No shared native writer/build is
assigned. Local GPU 0 is always reserved; check GPUs 1–3 immediately before use
and restrict by idle-device UUID, or use Slurm visibility. Avoid duplicate jobs.

## Engineering work and recent evidence

Normalization-input cleanup `19544d3b6` gives local scoring a separate preparation
stage in `helpers/normalization_inputs.py`. Its named result contains log-Z,
log-evidence, Pmax and reconstruction-threshold arrays. Local, sparse and K-class
callers share eight formerly repeated optional-vector conversions while retaining
their own domain/exclusivity rules. Existing F64 conversion, exact image-axis shape,
error order/messages and array storage/lifetimes are preserved. This is not a new
double-precision execution policy or a change to numerical normalization.

Sealed-preimage comparison matches **14,641 local input combinations and 312
individual conversions**, including dtype/shape/stride/bytes and exceptions. The
retained three caller-module ASTs are exact after expanding the changed blocks.
All **10 original end-to-end CPU cases** pass before/after, plus **21 new boundary
cases** (31 total, zero skips); the extended CPU/import guard passes **38/38**.
Focused/guard source manifests are identical and both preserve native hashes.
Existing tests and thresholds are unchanged; no new Ruff findings.

The local module loses 34 lines (8,913→8,879), including a 35-line reduction inside
the execution function. Total production source grows 26 lines for the explicit
owner, named result and shared validation contract; this is a clarity change,
not net deletion. Commands and evidence:
`hia_source_review_20260906/normalization_input_owner_20260909/{scope,audit,validation}.json`;
logs/XML: `pr180_integration_20260908/normalization_inputs_{control,after,guard}_20260909/`.
No GPU, trajectory or representative runtime qualification follows.

Scale-group cleanup `2cc66f316` replaces four validation/inference copies in
local EM, sparse K1, fused sparse K-class and K-class subset routing with
`helpers/scale_groups.py`. Explicit counts remain lower bounds, absent IDs
retain routing information without allocating engine scale statistics, and empty
IDs retain one group. Flattening, int64 conversion, validation order, exception
types/messages and the original caller allocation shapes/dtypes are preserved.
No arithmetic, reduction, JIT boundary or saved format changes.

The retained three module ASTs agree after expanding the changed blocks. A
sealed-preimage differential audit matches **768 engine-input cases and 195
router cases**, including invalid counts/IDs and conflicting errors. The same
seven end-to-end CPU cases pass before/after: local K-class details, default/split
local scoring, sparse chunking and fused K-class scoring with explicit group
counts. Eighteen new boundary cases also pass (**25 total**, zero skips), followed
by the extended **38-case CPU/import guard**, zero skips. Each run preserves its
source/native manifests. Only an import blank line changed between the focused
panel and guard; its AST is exact. Existing tests and thresholds are unchanged.

This removes **22 production lines net** across the four owner/caller modules;
the new tests add coverage. It does not establish GPU, trajectory, exact-K4
completion or runtime acceptance. Commands, sealed preimages and comparisons:
`hia_source_review_20260906/scale_group_owner_20260909/{scope,audit,validation}.json`.
Logs and XML: `pr180_integration_20260908/scale_groups_{control,after,guard}_20260909/`.
No duplicate peer job, native build, numerical907 or private F32 M adoption.

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

The current GPU panel exposed one missed owner migration in the parent-support
source guard. Test-only `2228f25a2` reads `half_scoring.py` instead of the controller;
all four assertions and the rest of the test module AST are unchanged. CPU control
reproduces one missing-substring failure with six passing execution cases; the same
seven cases pass after the correction, with no skips. Four in-memory flag mutations
are still rejected. Logs/XML: `parent_probe_owner_{red,green}_20260909` under the
CPU root; review: `pr179_current_api_20260909/parent_probe_owner_review.json`.
The frozen GPU panel retains the original failure; its source is not modified.

H100 API job **13641893** completed on `della-h20g3` at frozen `1b046647f`:
**727 passed, 6 failed, zero skipped/errors** in 709.89 pytest seconds
(727 s Slurm elapsed, exit1:0). Its exact 733-case inventory comprises 595 unchanged
historical IDs, nine historical cases mapped to 38 reviewed replacements, and
100 additional cases. All nine previously repaired non-normalization failures pass
in this broader cohort. One failure is the separately repaired source-owner guard;
the other five preserve strict byte-equality failures: two raw normalization carries
and three other result leaves. Failure count changes across different inventories
or repeats are not proof of numerical improvement.

All 317 package versions match the historical panel. The run verifies unchanged
2,270 tracked files, baselines, private native libraries and their source inputs.
The frozen panel remains failed; the targeted CPU repair does not relabel it green.
There was no duplicate native/trajectory run or precision adoption. Reproduction,
complete failure text and mappings are in `pr179_current_api_20260909` under the
source-review root: `inputs.json`, `api_only.sbatch`, `terminal_review.json` and
`parent_probe_owner_review.json`. Logs/XML are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr179_current_api_1b046647f_20260909/api/`.
The isolated source tree and private libraries remain frozen.
K-class diagnostic ownership cleanup `6ae78f029` is complete. Effective raw-operand
capture now lives beside pass-2 serialization, and fused capture-row materialization
beside BPref signatures. The sparse scorer loses 201 lines (17,780 → 17,579);
the three production modules together have zero net line change. Both moved
function ASTs and all retained module ASTs are exact after owner substitution,
including conversion/blocking order and call-site gates. All 312 assertions in
the two migrated test modules are retained, with no forwarding wrappers.
The same 97 CPU cases pass before/after with zero skips; the 38-case CPU guard
also passes. After/guard source manifests match. Ruff retains one inherited
sparse-module finding and adds none. No new GPU job or arithmetic change.
Commands, source fingerprints, case inventories, AST and lint audits are in
`hia_source_review_20260906/kclass_capture_owner_20260909/validation.json`.
Logs/XML are under the CPU root in
`kclass_capture_owner_{control,after,guard}_20260909/`.

VDAM reporting commit `383772fa0` is preserved by separate merge `49efca34e`.
All four incoming files are byte-identical to the frozen peer; the legacy
`gt_metrics.py` and its seven-field result remain unchanged. New
`gt_registration.py` provides the CPU rigid fitter and immutable, serializable
transform; `evaluate_ab_initio_gt.py` explicitly selects independent fits,
fit-once/apply-many, or saved-transform application. Defaults/output schemas and
E/M behavior are preserved. [Usage and geometry](gt_reporting.md) describe the
fixed frame, shape/voxel/GT checks, controls and limits. The new fitter uses
float64 on CPU; this is reporting, not a change to production EM precision.

Current-source validation: all 12 pre-existing cases pass before integration;
all 78 combined helper/CLI cases pass afterward (56.27 s), with zero skips.
Four direct opt-out comparisons to the pinned legacy evaluator preserve exact
JSON and every raw 19/aligned 31 NPZ field's dtype, shape and bytes. The final
CPU/import guard passes 38/38 (42.20 s); all three candidate manifests match.
Ruff passes all four paths; source/native hashes stay unchanged during checks.
Producer tuple/list and naive-NaN comparison failures remain preserved separately.
Commands, provenance and inventories:
`hia_source_review_20260906/rigid_reporting_integration_20260909/validation.json`;
logs/XML use `rigid_reporting_{control,combined,legacy_payloads,guard}_20260909/`
under the CPU root. No new GPU job, quality acceptance or precision907 adoption.
The four source paths are released after integration; shared docs/publication
remain em_clean-owned, and the original peer checkout stays frozen.

Source-STAR CTF evaluation and its single process cache now live in
`helpers/relion_ctf.py` (`60d6a5596`), called directly by coarse, local and sparse
scoring. Four function bodies and the cache definition are AST-exact; all retained
caller statements and 1,661 test assertions are preserved after owner mapping.
Cache key/lifetime, native calls, compact pixel order/duplicates, casts and device
placement are unchanged. Sparse scoring loses 155 lines (17,579 → 17,424); the
four production modules together add 24 lines for the explicit owner boundary.
There are no forwarding aliases, new tests, or new Ruff findings.

The control, first candidate and final CPU panels each execute the same 181 cases:
**179 passed, two failed**, with 23 GPU cases deselected and no skips. Both failures
are the unchanged, unmarked local-operand cases that call a CUDA-only translation
routine on CPU; their exception messages match exactly. They passed in the earlier
H100 panel at `1b046647f`, which does not qualify this new source. Tests and their
GPU requirements were not weakened. The final extended import/CPU guard passes
38/38 (44.49 s). Final/guard manifests match, and all panels preserve source/native
hashes. No new GPU job or trajectory/runtime claim. Exact commands, fingerprints,
case inventories and AST/lint audits:
`hia_source_review_20260906/source_ctf_owner_20260909/validation.json`.
Logs/XML use `source_ctf_owner_{control,after,final,guard}_20260909/` under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/`.

Sealed VDAM worker/chronology replay now belongs to `helpers/vdam_replay.py`
(`dd28135ba`). All21 function bodies, ten constants and four LRU decorators are
AST-exact; retained engine statements and 123 test assertions match after owner
mapping. Kernel execution and block-map publication stay in the local engine.
It loses 854 lines (10,228→9,374); combined production grows 18 lines. No forwarding
aliases, numerical changes, new tests or new Ruff findings.
The expanded 116-case CPU panel matches exactly before/after: 111 passed, five
existing source-guard failures, zero skips. The CPU/import guard passes 38/38.
Original control had six failures; separate `1d2e24174` repairs the stale inline
trace-selector assertion to check its bound value and forwarding. That repaired
case passes before source moves and rejects both removal mutations. The remaining
native first-atomic, first-state/M-step runner, tau2-owner and prior-owner source
guards were preserved for the separate review below; this cleanup did not change them.
Exact commands, failure identities and audits:
`hia_source_review_20260906/vdam_replay_owner_20260909/validation.json`.
CPU logs use `vdam_replay_owner_{control,expanded_control,after,guard}_20260909/`
and `vdam_trace_guard_repair_20260909/`. No new GPU job.

Test-only follow-up `598a3fa0e` repairs four of those five guards. Native-input
chronology now checks its default, override and export; tau2 checks the selected
refresh callback and its invocation; the posterior guard checks the K/oversampling
policy call. Git history `51fa361c6`/`4c06babd9` shows GPU-release assertions were
added to the wrong runner test: all nine checks now reside with the full-schedule
runner, while first-state selection checks its shared helper. No runner or kernel
changed. All 178 assertions remain, and 17 in-memory missing-wiring mutations fail.

The final CPU panel retains all 116 original case IDs and adds eight unchanged
functional controls: **123 passed, one failed, zero skips** in 4.66 s. The native
trace-placement guard remains failed and unchanged. Its declaration token is gone,
and current CUDA computes interpolation coordinates before the actual trace timer
in the scatter macro (`ecab47c05`); fixing that guard requires deciding which trace
boundary must be preserved. It is not a numerical-noise acceptance. All production
source and the CPU reference binding remained byte-identical. Exact commands,
case identities, failures and manifests:
`hia_source_review_20260906/vdam_source_guard_review_20260909/validation.json`;
CPU logs: `pr180_integration_20260908/vdam_stale_guard_{control,after,final}_20260909/`.
No GPU run or native rebuild.

Follow-up `6ccc3a3fd` tightens the remaining trace guard without changing CUDA.
The prior matcher selected zero initialization, so merely removing the old `int`
token falsely passed current code. It now identifies the actual clock call and
keeps the required pre-interpolation ordering. Eight source-only controls prove
both false-pass cases, historical pre-warp success, and rejection of missing or
late clock samples. Read-only historical RELION source `c77028723` confirms its
clock at `BP.cuh:620`, before interpolation at622; `64db9df71` deliberately matched
that boundary before `ecab47c05` moved it into the scatter macro.

Affected replay/chronology CPU tests: **36 passed, one expected ordering failure,
zero skips** (2.67 s). This replaces a missing-substring error with the demonstrated
contract failure; it does not waive it. First-atomic rank/latency comparisons need
the instrumentation boundary reconciled separately. Block-start replay uses the
separate block-start field. No actual atomic chronology, binary equivalence or
scientific effect was established. Production/native bytes are unchanged.
Commands, source snapshots and controls:
`hia_source_review_20260906/native_trace_guard_review_20260909/validation.json`;
logs: `pr180_integration_20260908/native_trace_guard_20260909/`.

Independent saved-array review of source-closed 13642331 reproduces all 2,814
full-shell AUCs, 1,608 held-out AUCs, 201 streamed records and threshold summaries
exactly. Candidate 2 minus native 2 is −0.002423452 at 155, below the unchanged
−0.002 condition, although both candidate final GT values exceed both natives.
This verifies the measurement, **not quality acceptance**. No MRC/FFT replay,
GT refit, state or score-margin audit was performed by this review. State/resolution
and cross-engine FSC failures remain; see the dated review for precise scope.
Receipt: `handoffs/em_clean_source_closed_saved_curve_review_20260909.json`. Preserve failed
launcher 13642161: its native 1 completed 201 maps before the parser failed, and
RECOVAR never started. The v2 launcher repair and terminal four-arm run are separate
records; no original-attempt reuse or retrospective qualification.

Fixed-capacity host binding cleanup `a7b15a197` moves five call-selection,
operand-validation and padding-validation functions into `fixed_capacity_local.py`,
beside the sealed types they validate. The M-step rotation accessor moves to
`local_layout.py`. The engine still selects, fetches and validates at the same
positions before JIT execution. Three call-0 wrappers with only test callers are
removed; tests call the general owner with an explicit zero call index where needed.
No dataclass, numerical operation, validation message or saved format changes.

All six moved function ASTs and the metadata constant are exact. Retained engine,
binding, layout and test ASTs agree after import/caller mapping and wrapper expansion;
all128 binding-test assertions remain. The same **200 CPU cases pass before/after**
(8.03/8.50 s), including zero/nonzero calls, operand mutation, fetch order, metadata,
poison exclusion and padding. The extended CPU/import guard passes **38/38** (43.29 s)
with identical source/native manifests and no skips. No new Ruff findings; the
engine's inherited import-order finding remains. No GPU or trajectory/runtime claim.

Local engine:9,374→8,928 lines (**−446**); combined production files **−46 lines**.
`fixed_capacity_local.py` is1,018 lines and `local_layout.py` is1,951. This reduces
controller clutter and removes obsolete APIs; the large execution function still
needs stage-level cleanup. Commands, immutable controls, AST/caller audit and logs:
`hia_source_review_20260906/fixed_capacity_binding_owner_20260909/validation.json`;
`pr180_integration_20260908/fixed_capacity_owner_{control,after,guard}_20260909/`.

Continue controller/state/kernel ownership and duplicate-code review with bounded
changes and proportional tests. Parsers with different blank/unknown-
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
| Current H100 API 13641893 |727 passed/6 failed/0 skipped at `1b046647f`; one stale guard repaired separately, five strict byte-equality failures remain |
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
