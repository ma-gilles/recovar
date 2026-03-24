#!/bin/bash
# Auto-format Python files after Claude edits them.
# Only formats the file that was just edited — never touches other files.
INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ "$FILE" == *.py ]] && [ -f "$FILE" ]; then
    ruff format --quiet "$FILE" 2>/dev/null || true
    ruff check --fix --quiet "$FILE" 2>/dev/null || true
fi
exit 0
