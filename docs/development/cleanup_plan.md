# Cleanup plan before new-engine development

The milestone covers RECOVAR except the GUI/frontend, with EM first. The
[codebase map](codebase.md) identifies the current owners; the
[checkpoint record](em_status.md) identifies the source actually tested.
This plan describes remaining engineering work, not acceptance of an untested
refactor or permission to change the scientific contract.

## Establish reliable comparisons

The table below routes remaining work; quantitative results and reviewed limits
live in [current EM status](em_status.md#unresolved-validation-gates).
Historical plans are not job state or current-source qualification.

| Work | Established evidence | Next decision and owner |
| --- | --- | --- |
| K1 score/state parity | PR180's six remaining Pmax gaps lack complete matched candidate/oracle evidence. Frozen4f9 synthetic full200 meets map conditions but has strict state differences. | EM/VDAM own their private first-divergence investigations. Review matched inputs, competing scores and margins before integrating numerical repairs; do not duplicate their captures. |
| Real K1 qualification | Frozen5ca9 real10076 prefix20 passes map conditions, with state differences. Its reviewed full200 final cross-engine AUCs fail .999; native-repeat variation is not a waiver. | Review source/fixture/build closure and first divergence. New completed runs require their own result admission; earlier prefix results do not qualify full trajectories or moving source. |
| Exact K4 | Reviewed saved20-iteration synthetic and real comparisons contain failures; a synthetic repeat passes, while the closest real repeat still fails at20. Earlier job13560356 failed at10/class2. | Admit raw maps and complete source/build identities before acceptance. VDAM owns the separate full200 K4 experiment; completion still requires exactlyK4 at100k/256 with per-class Hungarian matching. |
| Robustness and repeats | Saved GT screening includes failures in cases13/22/32. Per-map alignment differs from the prespecified shared-transform gate. Same-state saved E-step summaries are not wholly exact. | Review incoming state, candidate margins and the prescribed GT transform. Deterministic-reduction experiments are separate numerical/runtime candidates, not structural changes or automatic proof of parity. |
| Shared SPA/ET and downstream | Historical required-metric inventories and failures are retained; the selected moving source has no complete shared qualification. | Freeze a source checkpoint and run applicable shared checks before merge acceptance. Never drop required metrics or rewrite baselines to pass. |
| Performance | Historical ratios describe their frozen sources, fixtures and hardware; no moving-source100k/256 K1+K4 completion is accepted. | Admit completed timing pairs with matched source/input/hardware and quality evidence. Consult the ownership board before launching anything; pending/terminal jobs alone establish no scientific result. |

Live assignments and jobs belong to the
[coordination board](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/README.md),
not this plan. Check scheduler state when a decision depends on it; a status
file's empty top-level job list can conflict with pending entries elsewhere.
Keep private numerical proposals out of the structural series until reviewed
and authorized. Detailed historical plans remain in Git history and the
[refresh receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/benchmark_plan_refresh_20260910/result.json).

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
| Sparse scoring | `helpers/sparse_pass2_bucketed.py`, `helpers/bpref_diagnostics.py`, `helpers/pass2_diagnostics.py`, `helpers/norm_scale_diagnostics.py` | BPref capture context, membership selectors/counter/rotation-mass writers, and contribution validation have one diagnostic owner with direct callers. Continue separating candidate planning, kernel calls and sufficient-statistic reduction one boundary at a time. Pass-2 score/raw-operand writers and target-row selection have one owner; normalization/group-scale writers have a separate owner, with direct callers. Preserve packed shapes, reduction order, casts and ownership; scheduling and operand materialization remain explicit in the sparse scorer. |
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
