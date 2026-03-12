#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/style_common.sh"

usage() {
    cat <<'EOF'
Usage:
  scripts/hard_gate_git_changed_files.sh [file ...]

Hard gates:
  - clang-format on changed Git lines for tracked files
  - clang-format on whole files for untracked source files
  - no trailing whitespace
  - no merge conflict markers
  - no JUICER_TESTS references
  - no std::endl in touched source files
  - no using namespace directives in touched headers

Notes:
  - This script is intended for a repo whose Git history is current.
  - With no file arguments, the script uses modified and untracked Git paths.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

style_require_cmd clang-format
style_require_cmd clang-format-diff
style_require_cmd rg

REPO_ROOT="$(style_repo_root)"
mapfile -t FILES < <(style_collect_candidate_files "$REPO_ROOT" "$@")

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No eligible C/C++/CUDA files found."
    exit 0
fi

cd "$REPO_ROOT"

HAS_HEAD=0
if git rev-parse --verify HEAD >/dev/null 2>&1; then
    HAS_HEAD=1
fi

declare -a TRACKED_CHANGED_FILES=()
declare -a WHOLE_FILE_FILES=()

for file in "${FILES[@]}"; do
    if [[ "$HAS_HEAD" -eq 1 ]] && git ls-files --error-unmatch -- "$file" >/dev/null 2>&1; then
        if ! git diff --quiet HEAD -- "$file"; then
            TRACKED_CHANGED_FILES+=("$file")
        fi
    else
        WHOLE_FILE_FILES+=("$file")
    fi
done

FAIL=0
EXTENSIONS="c,cc,cpp,cxx,h,hh,hpp,hxx,inl,ixx,cu,cuh"
TRACKED_DIFF_IREGEX='.*\.(c|cc|cpp|cxx|h|hh|hpp|hxx|inl|ixx|cu|cuh)$'

if [[ ${#TRACKED_CHANGED_FILES[@]} -gt 0 ]]; then
    TRACKED_FORMAT_DIFF="$(
        git diff -U0 --no-color HEAD -- "${TRACKED_CHANGED_FILES[@]}" |
        clang-format-diff -p1 -style=file -iregex "$TRACKED_DIFF_IREGEX"
    )"
    if [[ -n "$TRACKED_FORMAT_DIFF" ]]; then
        printf '%s\n' "$TRACKED_FORMAT_DIFF"
        echo "ERROR: Tracked changed lines are not clang-formatted." >&2
        FAIL=1
    fi
fi

if [[ ${#WHOLE_FILE_FILES[@]} -gt 0 ]]; then
    if ! clang-format --dry-run --Werror --style=file "${WHOLE_FILE_FILES[@]}"; then
        echo "ERROR: Untracked source files are not clang-formatted." >&2
        FAIL=1
    fi
fi

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

if rg -n -H "\bstd::endl\b" -- "${FILES[@]}"; then
    echo "ERROR: std::endl is banned in touched files; use '\\n' instead." >&2
    FAIL=1
fi

declare -a HEADER_FILES=()
for file in "${FILES[@]}"; do
    case "$file" in
        *.h|*.hh|*.hpp|*.hxx|*.inl|*.ixx)
            HEADER_FILES+=("$file")
            ;;
    esac
done

if [[ ${#HEADER_FILES[@]} -gt 0 ]]; then
    if rg -n -H "^[[:space:]]*using namespace\b" -- "${HEADER_FILES[@]}"; then
        echo "ERROR: using namespace directives are banned in touched headers." >&2
        FAIL=1
    fi
fi

if [[ "$FAIL" -ne 0 ]]; then
    exit 1
fi

echo "All Git-backed hard-gate checks passed for ${#FILES[@]} file(s)."
