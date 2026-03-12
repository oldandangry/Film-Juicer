#!/usr/bin/env bash
set -euo pipefail

style_repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$script_dir/../.." && pwd
}

style_die() {
    echo "ERROR: $*" >&2
    exit 1
}

style_require_cmd() {
    local name="$1"
    command -v "$name" >/dev/null 2>&1 || style_die "Required command not found: $name"
}

style_is_supported_source() {
    local path="$1"
    case "$path" in
        *.c|*.cc|*.cpp|*.cxx|*.h|*.hh|*.hpp|*.hxx|*.inl|*.ixx|*.cu|*.cuh)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

style_is_ignored_path() {
    local path="$1"
    case "$path" in
        external/*|installer/*|juicer/*|x64/*|Resources/*|docs/*)
            return 0
            ;;
        src/GeneratedColorSpaces.cpp|src/GeneratedColorSpaces.h)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

style_normalize_relpath() {
    local root="$1"
    local path="${2//\\//}"

    while [[ "$path" == ./* ]]; do
        path="${path#./}"
    done

    if [[ "$path" == "$root/"* ]]; then
        path="${path#"$root"/}"
    fi

    printf '%s\n' "$path"
}

style_collect_candidate_files() {
    local root="$1"
    shift

    local -a raw_paths=()
    if [[ $# -gt 0 ]]; then
        raw_paths=("$@")
    elif git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        local line
        if git -C "$root" rev-parse --verify HEAD >/dev/null 2>&1; then
            while IFS= read -r line; do
                [[ -n "$line" ]] && raw_paths+=("$line")
            done < <(git -C "$root" diff --name-only --diff-filter=ACMR HEAD --)
        else
            while IFS= read -r line; do
                [[ -n "$line" ]] && raw_paths+=("$line")
            done < <(git -C "$root" ls-files --modified)
        fi

        while IFS= read -r line; do
            [[ -n "$line" ]] && raw_paths+=("$line")
        done < <(git -C "$root" ls-files --others --exclude-standard)
    else
        style_die "No file paths were provided and this directory is not a Git work tree"
    fi

    local -A seen=()
    local -a filtered=()
    local path rel
    for path in "${raw_paths[@]}"; do
        rel="$(style_normalize_relpath "$root" "$path")"
        [[ -n "$rel" ]] || continue
        [[ -f "$root/$rel" ]] || continue
        style_is_supported_source "$rel" || continue
        style_is_ignored_path "$rel" && continue
        [[ -n "${seen[$rel]:-}" ]] && continue
        seen["$rel"]=1
        filtered+=("$rel")
    done

    if [[ ${#filtered[@]} -eq 0 ]]; then
        return 0
    fi

    printf '%s\n' "${filtered[@]}"
}
