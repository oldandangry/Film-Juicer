#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Shared touched-file selection so format/check stay aligned.
source "$SCRIPT_DIR/lib/style_common.sh"

usage() {
    cat <<'EOF'
Usage:
  scripts/format_touched_files.sh [--check] [file ...]

Options:
  --check     Verify formatting without modifying files.
  -h, --help  Show this help.

Notes:
  - Pass explicit file paths when possible.
  - With no file arguments, the script falls back to modified/untracked Git paths in this repo.
  - Only C/C++/CUDA source files are considered.
EOF
}

CHECK_ONLY=0
declare -a INPUT_PATHS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            CHECK_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                INPUT_PATHS+=("$1")
                shift
            done
            ;;
        *)
            INPUT_PATHS+=("$1")
            shift
            ;;
    esac
done

CLANG_FORMAT_BIN="$(style_clang_format_bin)"
style_require_cmd "$CLANG_FORMAT_BIN"
style_verify_clang_format "$CLANG_FORMAT_BIN"

REPO_ROOT="$(style_repo_root)"
mapfile -t FILES < <(style_collect_candidate_files "$REPO_ROOT" "${INPUT_PATHS[@]}")

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No eligible C/C++/CUDA files found."
    exit 0
fi

cd "$REPO_ROOT"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "Checking formatting for ${#FILES[@]} file(s)..."
    "$CLANG_FORMAT_BIN" --dry-run --Werror --style=file "${FILES[@]}"
else
    echo "Formatting ${#FILES[@]} file(s)..."
    "$CLANG_FORMAT_BIN" -i --style=file "${FILES[@]}"
fi

printf '  %s\n' "${FILES[@]}"
