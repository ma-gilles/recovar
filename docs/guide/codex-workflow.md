# Codex Workflow Template

Use this as the template for long multi-day editing/debugging runs where context will be reloaded often. The canonical repo-root resume point is `STATE.md`.

## Persistent State

Keep a tiny note with only the facts needed to resume. Match `STATE.md`:

```md
# State
Goal:
Branch:
Worktree:
Current focus:
Last good step:
Last failing command:
Failure mode:
Next action:
Open questions:
```

Update it after each meaningful step. Keep it short enough to read in one screen.

## Reload Prompt

When resuming a session, provide only:

```md
Resume from `docs/guide/codex-workflow.md` and the current git diff.
Use the smallest possible set of file reads.
State the next concrete action first.
Do not restate repo-wide instructions unless they changed.
```

## Check-in Format

Use this after each step:

```md
- Done:
- Verified:
- Failed:
- Next:
```

## What Not to Paste

- Full logs unless they contain the exact failing line.
- Whole files when a diff or function excerpt is enough.
- Repeated copies of repo instructions.
- Broad design summaries when a precise next action is available.

## Decision Rule

If a change can be validated with one targeted test, do that first. Only widen the scope when the local fix is credible.
