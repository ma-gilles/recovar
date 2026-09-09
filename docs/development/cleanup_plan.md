# Cleanup plan before new-engine development

The milestone covers RECOVAR except the GUI/frontend, with EM first. The
[codebase map](codebase.md) identifies the current owners; the
[checkpoint record](em_status.md) identifies the source actually tested.
This plan describes remaining engineering work, not acceptance of an untested
refactor or permission to change the scientific contract.

## Establish reliable comparisons

| Work | Current evidence | Next bounded step |
| --- | --- | --- |
| PR180 K1 integration | Per-half noise-state correction reduces large Pmax gaps from 292 to 6. A six-particle replay exactly preserves those Pmax values and saved poses. The score dump changes Pmax by at most `3.05e-5`; float64 normalization of row 901's captured scores changes Pmax by only `5.01e-10`. | Compare matched score operands, priors and candidate geometry. The [targeted evidence](evidence/pr180-k1-targeted-capture-20260908/README.md) preserves the dump's execution change; final normalization does not close its gap. RELION live candidate scores and the historical generating build remain unavailable. |
| Synthetic K1 repeatability | One of three no-capture candidate pairs fails support/Pmax checks; all final-map gates pass. One candidate repeat differs from its two repeats. | Use the first recorded boundary to select a fixed-state diagnostic with the same candidates, priors, noise and map inputs. Capture the competing probabilities for image 685 before classifying the support change. |
| Real K1 repeatability | Two unchanged PR158 runs fail the direct-map gate from iteration 8. All captured first-iteration half-1 operands match; three identical-input production accumulation trials differ, including the warm pair. | Preserve the fixed-input evidence and investigate its relationship to the first iteration-2 support/Pmax differences before attributing trajectory changes. Keep autonomous convergence and fixed-state arithmetic as separate results. |
| K-class execution and comparisons | PR180 fixes the PR158 undefined-variable path; its focused comparison passes on the merged source. Older repaired-control K2 passes; K8/K16 retain historical failures, and K16 has a class/pose flip without score margins. The historical exact-K4 pair started September 8 in Slurm13560202; its audit remains pending. | Qualify the merged PR180 source for K2/K4/K8/K16 after the K1 checks. Preserve the prior failures and distinguish PR180's numerical changes from structural equivalence to PR158. |
| Shared SPA/ET metrics | Both existing tests pass 16 canonical required metric keys. Ten historical aliases explain the first external inventory failure. | Complete the held strict-inventory proposal without dropping required metrics, widening tolerances or writing baselines. |
| Performance measurements | Repeated K1 timings exist; sampled RSS warnings and quality failures remain. Shared stage timings are historical single-run comparisons. | Compare accepted workloads on the same physical GPU with repeated orders and independent caches. Retain sampled process-tree RSS and OS process high-water RSS as separate measurements. |

The current captures locate divergence but do not contain every candidate score
or accumulator. A tiny first-iteration shell-statistic difference is not proof
of the later particle-level failure's cause. Do not replace a failed trajectory
with a map-only pass or label it rounding noise without fixed-state evidence.

The prepared validation patches, GPU-test markers and unused shared-`/tmp`
removal remain separate from behavior-preserving source changes. Their concrete
patches and decision records are in the Della review directory linked from the
checkpoint record. No established baseline may be rewritten implicitly.

## Make existing execution easier to read

Prioritize ownership boundaries that can be checked independently. The existing
controller still combines dispatch, half/class state, replay, iteration history,
reconstruction and finalization. Moving code is useful when it establishes one
owner and removes dependencies back into the controller; a forwarding wrapper
or a new class that merely stores every local variable does not solve this.

| Area | Existing owner | Refactor boundary and evidence needed |
| --- | --- | --- |
| Replay interventions | `helpers/state_swap_probe.py`, `helpers/state_swap_runtime.py`, `frozen_boundary.py` | Variant definitions, snapshot copying, restoration and integrity checks are separated from controller scheduling. Preserve shallow/deep-copy semantics and exact mutation order in subsequent changes. |
| Half/class outputs | `score_outputs.py`, `relion_replay._RelionHalfInputState`, `helpers.types` and local-search dispatch | Dense/local scoring and class-result scatter use named results. The local kernel returns `LocalEMResult`; the positional packer and both decoders are removed, with all sixteen return-flag combinations and existing caller cases preserved. Continue reviewing state transitions and finalization boundaries, preserving optional captures, class mass, array ownership and update order. |
| Sparse scoring | `helpers/sparse_pass2_bucketed.py`, `helpers/bpref_diagnostics.py`, `helpers/pass2_diagnostics.py` | BPref capture context, membership selectors/counter/rotation-mass writers, and contribution validation have one diagnostic owner with direct callers. Continue separating candidate planning, kernel calls and sufficient-statistic reduction one boundary at a time. Pass-2 score/noise writers and target-row selection now have their own diagnostic module. Preserve packed shapes, reduction order, casts and ownership; scheduling and operand materialization remain explicit in the sparse scorer. |
| Iteration configuration | Grouped refinement options and `iteration_loop.refine_single_volume` | PR180 replaces the individual option keywords; callers and tests are migrated in the integration. Continue separating initialization and per-iteration state from immutable configuration. Preserve option validation, defaults and scientific behavior in further cleanup. |
| Follower dispatch and history | `relion_worker_scale.py` and `helpers/iteration_history.py` | Setup, dispatch, input remapping, follower correction updates and replay-completion validation share the scale owner. The controller supplies precision, numbered iteration, applied history and logger. Existing tests and exact comparisons cover state mutations, telemetry and all three completion return paths. Continue normalization and finalization orchestration cleanup while keeping scheduling decisions in the controller. |
| Shared pipeline orchestration | `commands/pipeline.py` and domain modules | Keep loading, covariance/PPCA, embedding and output ownership explicit. Preserve non-EM public APIs and serialized results. Validate the affected shared workflows at a source checkpoint. |
| Historical experiment scripts | `scripts/`, shared JSON/hash and scorecard helpers | Retain independent numerical references. Remove or consolidate only after reviewing imports, CLI entry points, notebooks and serialized names; preserve recorded reproduction commands. |

A read-only trace of the existing two-iteration CPU test at `dbada1eb5`
(Slurm13588500) finds live accumulator references after the controller's cleanup
boundary. `PerHalfOutputs` still holds both halves; the last-half result and
payload aliases survive into the next iteration's first scoring call. The test
passes and source hashes remain unchanged. Tracing may affect reclamation;
these named-reference observations do not measure GPU memory or prove an OOM.
The trace, source identity and eight distinct binding snapshots are preserved
in the review directory under `structural_cleanup/result_lifetime_probe/`.
Changing these lifetimes requires separate measurement and validation.

Do not combine these into a single executor rewrite. Each change should state
its input/output ownership, preserve the relevant executable behavior, and
carry focused caller checks. Move to larger scientific workloads after those
checks pass and when a checkpoint needs qualification.

## Keep development checks proportional

A small helper or import change gets affected tests, import/CLI checks when
relevant, and static or exact old/new comparisons for a pure refactor. Several
related changes form a frozen checkpoint for broader CPU and applicable GPU
checks. Synthetic/real/K-class pairs qualify scientific checkpoints, rather
than every edit. Re-auditing saved outputs should not rerun their GPU workload.

For a merge-ready shared checkpoint, follow the full shared-suite and rebase
requirements in [CONTRIBUTING](../../CONTRIBUTING.md). The user authorized
PR179 draft checkpoints stacked on PR158 with incomplete validation; keep their
failed and missing checks explicit. A rebase creates a new candidate. Keep
source snapshots for queued jobs immutable and continue independent work in the
implementation checkout. Never claim a later commit passed an earlier commit's
end-to-end test.

## Conditions for this milestone to close

- Applicable agent guides agree on scope, environment, ownership, validation
  and delivery; historical run instructions are clearly separated.
- Removed code has a documented caller review; consolidated code has canonical
  owners and migrated consumers. New numerical behavior is reviewed separately.
- Selected synthetic, real and K-class workloads have complete source/fixture
  identities, preserved failures and reproducible accuracy/performance reports.
- Shared SPA/ET, outlier and downstream qualification covers the actual selected
  source, and all applicable merge checks pass before the draft is accepted.
- Remaining numerical and architectural issues have concrete owners, evidence
  and next checks. The new engine remains a subsequent milestone.
