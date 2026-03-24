#!/bin/bash
# Block edits to tests/baselines/ files.
# Baselines are ground truth from OLD recovar — never modify unless explicitly asked.
INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ "$FILE" == *"tests/baselines/"* ]]; then
    echo "Blocked: tests/baselines/ files are sacred ground truth from the OLD published code. Do NOT modify them. If the user explicitly asked to regenerate a baseline, tell them to approve when re-prompted." >&2
    exit 2
fi
exit 0
