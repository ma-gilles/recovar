#!/bin/bash
# Block direct pushes to dev or main. Must use feature branches.
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# relax (github.com/ma-gilles/relax) uses main as its integration branch, so pushes to
# relax main are allowed. The target repository is the `git -C <dir>` argument of the
# push if present, otherwise the tool call's working directory.
push_repo_dir() {
    local dir
    dir=$(echo "$COMMAND" | grep -oE 'git +-C +[^ ;&|]+ +push' | head -1 | awk '{print $3}')
    if [ -z "$dir" ]; then
        dir=$(echo "$INPUT" | jq -r '.cwd // empty')
    fi
    echo "$dir"
}
if echo "$COMMAND" | grep -qE 'git\b.*\bpush\b'; then
    REPO_DIR=$(push_repo_dir)
    if [ -n "$REPO_DIR" ] && git -C "$REPO_DIR" remote get-url origin 2>/dev/null | grep -qE 'ma-gilles/relax(\.git)?$'; then
        if echo "$COMMAND" | grep -qE -- '--force|-f\b|\+[^ ]*:'; then
            echo "Blocked: never force-push relax branches." >&2
            exit 2
        fi
        exit 0
    fi
fi

# Match: git push [anything] [remote] dev, git push [remote] main, etc.
# Also catch: git push origin HEAD:dev, git push recovar ...:dev
if echo "$COMMAND" | grep -qE 'git push\b.*\b(main|dev)\b' && \
   ! echo "$COMMAND" | grep -qE 'git push\b.*--delete'; then
    # Allow pushing feature branches that happen to contain "dev" in the name
    # Block only when the target IS dev or main
    if echo "$COMMAND" | grep -qE 'git push\s+\S+\s+(main|dev)\s*$' || \
       echo "$COMMAND" | grep -qE 'git push\s+\S+\s+\S+:(refs/heads/)?(main|dev)\s*$' || \
       echo "$COMMAND" | grep -qE 'git push\s+\S+\s+\S+:(main|dev)\b.*--force'; then
        echo "Blocked: Do not push directly to dev or main. Push a feature branch and create a PR." >&2
        exit 2
    fi
fi
exit 0
