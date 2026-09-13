# Long-term agent workflow

Keep the full cleanup milestone and scientific gates. Reduce repeated context,
bookkeeping and publication overhead, not validation fidelity.

## Model and work package

Model selection and delegation are opt-in session choices. The selected RECOVAR
workstream has an [Astra-led delegation policy](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/delegated/POLICY.md)
and [activation/recovery instructions](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/delegated/README.md).
Scientific requirements apply equally to every model. Earlier global model
settings are historical; this profile does not change them.

Start a fresh thread for a cohesive package at a stable boundary. Its handoff
should fit roughly one page: objective, checkout/HEAD/dirty identity, constraints,
owned files, relevant evidence/failures, exact live job IDs, next action and
acceptance criteria. It must retain the larger objective. Use current status
and code as authority; read linked historical evidence only when needed.
Do not fork the entire old conversation to emulate a fresh start.

A package delivers a meaningful ownership/readability improvement with migrated
callers and appropriate checks. Avoid both per-deletion commits with elaborate
receipts and giant unreviewable batches. Run focused checks while implementing;
perform one cohesive integration review and PR update after the package. Keep
numerical proposals separate. An untested WIP publication never implies acceptance.

## Current memory and ownership

The [EM status page](em_status.md) links the shared coordination board. The board
contains only current ownership, source, active jobs and actionable handoffs.
Each agent writes only its own status and handoffs; the integrator owns shared
status/docs and publication. Replace historical payloads with archive links.
Target <8 KB for a current agent status and <6 KB for the board README. These
are context targets, not reasons to discard a scientific failure or active lock.

One compact receipt per package links commands, source/native/input inventories,
test results and open gates. Keep large JSON/XML and full logs on disk. Do not
load them wholesale: scripts extract failures, counts and changed hashes.
Exchange decisions, ownership changes, terminal results and blockers, not
continuous narrative histories. No extra agents are implied by this workflow.

## Command and job bookkeeping

`scripts/em_work_package.py` uses the standard library and launches no model.
Its `run` command records tracked/untracked file hashes and source identity before
and after the exact command, a log, elapsed time, return code and selected
environment. A changed source inventory makes the wrapper fail. It does not
replace native/input pins, import checks, scientific review or existing test
runners. Use existing runners for those checks, wrapped once per package.
Outputs must be in a new directory outside the checkout.

```bash
# CPU example; select the checkout's pixi Python and verify imports for real tests.
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 \
  .pixi/envs/default/bin/python scripts/em_work_package.py \
  --output /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/PACKAGE_checks \
  run -- .pixi/envs/default/bin/python scripts/check_agent_guides.py

# Watch existing exact IDs; this does not submit, restart, cancel or qualify them.
.pixi/envs/default/bin/python scripts/em_work_package.py \
  --output /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/PACKAGE_watch \
  watch JOB_ID ANOTHER_JOB_ID
```

Run a watcher under the site's persistent process mechanism. It writes current
`jobs.json` and append-only transition/error `events.jsonl`; polling defaults to
five minutes. Missing jobs and scheduler errors stay unknown until accounting
establishes terminal state and exit code. The watcher does not wake an agent or
send notifications. Read its summary at the next decision boundary; do not keep
an expensive model turn alive just to poll it. Use Slurm `afterok` dependencies
for dependent execution and `afterany` for terminal summaries where the existing
launcher supports them. Never restart a job because observation failed.

## Weekly efficiency review

Keep one row per accepted package in the board's `efficiency.csv`: week, package,
model, accepted changes, resolved hypotheses, rework hours, regressions, evidence,
usage source and usage reading. Counts are review outcomes, not lines deleted or
commits made. Leave unknown usage empty; historical thread counters are not
billing or allowance consumption. Do not reconstruct spending from this thread.

At the first work session each week, compare the previous week's accepted work,
rework and regressions with the account dashboard or CLI `/status`, retaining
its measurement window and units. This is a weekly review procedure, not an
installed account-monitoring daemon. The agent cannot read dashboard allowance
through local conversation tokens. See [official usage guidance](https://learn.chatgpt.com/docs/pricing).
Escalate only packages needing deeper review; adjust the model/work-package size
from measured outcomes rather than assuming lower per-response usage is better.

No unnecessary MCP server is configured in this session's user config. Do not
remove account connectors or alter other sessions' settings speculatively.
