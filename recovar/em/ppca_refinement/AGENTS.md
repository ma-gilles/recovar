# Pose-Marginalized PPCA Refinement — Agent Bootstrap

This subtree's durable agent instructions live in
`recovar/em/ppca_refinement/CLAUDE.md`. Read that file in full before making
code, test, benchmark, or parity changes anywhere related to the
`recovar ppca-refine` project.

`CLAUDE.md` is canonical for both Claude and Codex. Keep this file as a
short loader only; put new workflow rules, math conventions, naming, prior
conventions, and milestones in `CLAUDE.md` to avoid instruction drift.

The full mathematical specification (model, per-pose math, augmented
M-step, prior conventions, milestones, non-negotiables) lives at
`docs/math/ppca_refine_plan_2026_05_01.md`. Read it once for reference;
day-to-day work follows `CLAUDE.md`.

The parent `recovar/em/CLAUDE.md` and `recovar/em/AGENTS.md` apply to all
EM code. If they conflict with this subtree's guide during day-to-day
ppca-refinement work, follow the narrower guide here. Root-level release
and PR requirements still apply before pushing or opening a PR.
