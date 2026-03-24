#!/bin/bash
# Block direct pushes to dev or main. Must use feature branches.
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

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
