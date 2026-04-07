#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/check_changed_python.sh [--base REV] [--head REV] [--format] [--lint] [--typecheck]

Runs Ruff format check, Ruff lint, and/or mypy on Python files changed between two revisions.

Defaults:
  base: merge-base(HEAD, origin/dev) when origin/dev exists, else HEAD~1
  head: HEAD
  checks: format + lint + typecheck

Examples:
  bash scripts/check_changed_python.sh
  bash scripts/check_changed_python.sh --base origin/dev --lint --typecheck
  bash scripts/check_changed_python.sh --base "$BASE" --format
EOF
}

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

base=""
head="HEAD"
declare -a checks=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      base="${2:?missing value for --base}"
      shift 2
      ;;
    --head)
      head="${2:?missing value for --head}"
      shift 2
      ;;
    --format)
      checks+=("format")
      shift
      ;;
    --lint)
      checks+=("lint")
      shift
      ;;
    --typecheck)
      checks+=("typecheck")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#checks[@]} -eq 0 ]]; then
  checks=("format" "lint" "typecheck")
fi

if [[ -z "$base" ]]; then
  if git show-ref --verify --quiet refs/remotes/origin/dev; then
    base="$(git merge-base "$head" origin/dev)"
  elif git rev-parse --verify -q "${head}~1" >/dev/null; then
    base="${head}~1"
  else
    echo "Unable to determine a base revision automatically" >&2
    exit 2
  fi
fi

mapfile -d '' files < <(git diff -z --name-only --diff-filter=ACMR "$base" "$head" -- '*.py')

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No changed Python files between $base and $head"
  exit 0
fi

echo "Changed Python files between $base and $head:"
printf '  %s\n' "${files[@]}"

for check in "${checks[@]}"; do
  case "$check" in
    format)
      echo
      echo "==> Ruff format check"
      python -m ruff format --check --diff "${files[@]}"
      ;;
    lint)
      echo
      echo "==> Ruff lint"
      python -m ruff check "${files[@]}"
      ;;
    typecheck)
      echo
      echo "==> Mypy"
      python -m mypy --config-file pyproject.toml "${files[@]}"
      ;;
    *)
      echo "Unknown check: $check" >&2
      exit 2
      ;;
  esac
done
