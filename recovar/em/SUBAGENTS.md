# EM Parity Subagent Protocol

Use this protocol only when the user has allowed subagents or parallel agent
work. Parallelism is for independent evidence gathering, not for multiplying
edits to the same hot path.

## Ownership Model

The primary agent is the integrator. It owns:

- the active hypothesis and acceptance gate;
- edits to `docs/math/em_parity_program.md`;
- final source integration, validation choice, and user-facing claims;
- allocation of Slurm jobs and coordination of the shared RELION build.

Prefer read-only subagents. A subagent may edit only when assigned an explicit,
disjoint file set. Never let two agents modify the same file, rebuild the
shared RELION tree, mutate a fixture, or update the program board concurrently.

## Good Parallel Tasks

- inspect a specific RELION source behavior and return exact file/line evidence;
- analyze an existing dump or completed run without changing it;
- inspect the corresponding RECOVAR call chain and propose the first-divergence
  boundary;
- design a focused test in a uniquely assigned new test file;
- summarize performance traces from already completed logs;
- audit benchmark provenance, command parity, or metric completeness.

Do not delegate the final diagnosis, merge decision, broad unspecific code
review, or instruction-file interpretation. Do not have several agents run the
same expensive benchmark “for confidence.”

## Task Packet

Every subagent receives:

1. one falsifiable question;
2. absolute worktree and artifact paths;
3. immutable commit plus dirty fingerprint;
4. allowed files and whether the task is read-only;
5. commands or data already tried, to avoid duplicate work;
6. expected evidence and a stop condition;
7. a time/GPU budget and whether Slurm submission is authorized.

Example:

```text
Question: Does RELION firstiter_cc route pass 2 only through coarse winners?
Scope: read-only RELION source and existing iter-1 dumps.
Do not edit or submit jobs.
Return: exact source locations, dump evidence, confidence, and the cheapest
RECOVAR regression that would distinguish winner-only from full support.
Stop after the behavior is proven or two independent evidence paths disagree.
```

## Recommended Three-Way Split

For a difficult divergence, use at most three parallel evidence roles while
the primary integrates:

- **oracle investigator:** RELION source, STAR state, and dump semantics;
- **RECOVAR investigator:** matching call chain, state boundary, and candidate
  replay;
- **evidence auditor:** existing run metrics, provenance, and cheapest
  discriminating experiment.

Only one role should write code. If code changes become necessary, stop the
other writers, assign exact files, then integrate serially.

## Slurm And Shared Resources

- The primary maintains a job registry with job ID, purpose, dependency,
  expected output, and owner.
- Slurm jobs may be submitted broadly and allowed to queue. Locally, use at
  most three GPUs total and only after `nvidia-smi` confirms that each selected
  GPU is idle and not in use by someone else.
- Reuse immutable fixtures and completed RELION outputs. Each writer gets a
  distinct CRYOEM `em_work/codex/<dated-run-name>/` root and runtime directory.
- Every disposable root gets `SAFE_TO_DELETE`; never delete another agent’s
  root during the program.
- The patched RELION checkout/build is a mutex. The primary explicitly grants
  its write/rebuild lock to one agent and records what changed.

## Handoff Contract

A subagent returns:

- answer to the assigned question and confidence level;
- exact commands and immutable provenance;
- quantitative evidence, not “looks correct”;
- absolute paths to logs/artifacts and any Slurm IDs;
- files changed, if authorized, with tests run;
- negative findings and remaining ambiguity;
- recommended next experiment, ranked by information gained per GPU-hour.

The primary verifies material claims before integration. Subagent conclusions
are evidence, not accepted program state, until entered in
`docs/math/em_parity_program.md`.

## Conflict And Stop Rules

Stop parallel work and serialize when:

- two agents need the same source file or shared RELION resource;
- fixture, branch, command, or hardware provenance differs;
- a result crosses the `1e-3` numeric escalation threshold or flips winners;
- quality and strict-oracle conclusions disagree;
- a job would expand beyond the agreed GPU/time budget.

Do not hide disagreement by averaging metrics. Preserve both results, identify
the differing condition, and design one discriminating experiment.
