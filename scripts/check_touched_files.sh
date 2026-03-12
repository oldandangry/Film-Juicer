#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/style_common.sh"

usage() {
    cat <<'EOF'
Usage:
  scripts/check_touched_files.sh [file ...]

Checks:
  - clang-format dry-run using the repo .clang-format
  - trailing whitespace
  - merge conflict markers
  - banned JUICER_TESTS references

Notes:
  - Pass explicit file paths when possible.
  - With no file arguments, the script falls back to modified/untracked Git paths in this repo.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

style_require_cmd rg

REPO_ROOT="$(style_repo_root)"
mapfile -t FILES < <(style_collect_candidate_files "$REPO_ROOT" "$@")

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No eligible C/C++/CUDA files found."
    exit 0
fi

"$SCRIPT_DIR/format_touched_files.sh" --check "${FILES[@]}"

cd "$REPO_ROOT"

FAIL=0

if rg -n -H "[[:blank:]]$" -- "${FILES[@]}"; then
    echo "ERROR: Trailing whitespace found." >&2
    FAIL=1
fi

if rg -n -H "^(<<<<<<<|=======|>>>>>>>)" -- "${FILES[@]}"; then
    echo "ERROR: Merge conflict markers found." >&2
    FAIL=1
fi

if rg -n -H "JUICER_TESTS" -- "${FILES[@]}"; then
    echo "ERROR: JUICER_TESTS is deprecated and must not be introduced." >&2
    FAIL=1
fi

if [[ "$FAIL" -ne 0 ]]; then
    exit 1
fi

echo "All touched-file checks passed for ${#FILES[@]} file(s)."
