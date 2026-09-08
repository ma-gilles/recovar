# Cleanup plan before new-engine development

The milestone covers RECOVAR except the GUI/frontend, with EM first. The
[codebase map](codebase.md) identifies the current owners; the
[checkpoint record](em_status.md) identifies the source actually tested.
This plan describes remaining engineering work, not acceptance of an untested
refactor or permission to change the scientific contract.

## Establish reliable comparisons

| Work | Current evidence | Next bounded step |
| --- | --- | --- |
| PR180 K1 integration | Two float32 iteration-3-to-4 replays isolate a half-2 noise-state mismatch: large Pmax gaps fall from 292 to 6 and the 7.5-degree pose difference disappears. Half-1 Pmax is exactly unchanged. Both particle gates still fail. | Capture matched state and candidate scores for the six remaining rows; preserve both failed audits. The versioned [comparison](evidence/pr180-k1-noise-state-20260908/README.md) records the explicit noise policy and unknown generating oracle build. |
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
| Half/class outputs | `score_outputs.py`, `relion_replay._RelionHalfInputState` and local-search dispatch | Dense/local scoring and class-result scatter now receive the existing output container. Exact comparisons cover unequal half sizes and K1/K2/K4/K8/K16 without changing list identities. Dense scoring and the local-search wrapper now return named result fields to the controller. The local kernel still has a tuple contract decoded once by its wrapper; review that remaining boundary next, preserving optional captures, class mass and update order. |
| Sparse scoring | `helpers/sparse_pass2_bucketed.py` | Separate candidate planning, kernel calls, sufficient-statistic reduction and capture/report handling one boundary at a time. Preserve packed shapes, reduction order, casts and ownership. |
| Iteration configuration | Grouped refinement options and `iteration_loop.refine_single_volume` | PR180 replaces the individual option keywords; callers and tests are migrated in the integration. Continue separating initialization and per-iteration state from immutable configuration. Preserve option validation, defaults and scientific behavior in further cleanup. |
| Follower dispatch and history | `relion_worker_scale.py` and `helpers/iteration_history.py` | Setup, dispatch, input remapping and follower scale/image-correction updates share the scale owner. The controller supplies precision, numbered iteration and logger. Existing tests and exact comparisons cover replay replacement, correction updates, mutation order and telemetry. Review remaining controller-side finalization and normalization orchestration while keeping scheduling decisions in the controller. |
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

Before publication, follow the full shared-suite and rebase requirements in
[CONTRIBUTING](../../CONTRIBUTING.md). A rebase creates a new candidate. Keep
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
  source, and all publication checks pass before a push or PR.
- Remaining numerical and architectural issues have concrete owners, evidence
  and next checks. The new engine remains a subsequent milestone.
