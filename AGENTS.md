# RECOVAR Development Guide

Apply all repo instructions in [CLAUDE.md](/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/CLAUDE.md). This file exists to make the validation gate explicit for agents that prioritize `AGENTS.md`.

## Validation Completion Rule

- For any code-changing task, targeted tests are only for debugging. They do not replace the repo-required validation suite.
- In `recovar`, validation is not complete until the required GPU/integration coverage has passed. By default, run `./scripts/run_tests_parallel.sh long-test` and require all groups to pass before calling a branch push or PR ready.
- The user may explicitly override the default push gate. If the user explicitly asks for a push before `long-test` passes, you may push the branch, but you must clearly report that validation is incomplete and list exactly what has and has not been run.
- If you submit any Slurm job for testing, benchmarking, or validation, you must wait for the job to reach a terminal state before considering the task complete.
- After the job finishes, read the summary log and any relevant failing job logs.
- Report the real outcome to the user, including Slurm job IDs, final states, exact commands run, and log paths.
- Do not stop at "submitted" or "pending" unless the user explicitly asked you to only submit the job without waiting.
- Do not say validation is complete, do not say the task is done, and do not mark a PR ready while required Slurm jobs are still running.
