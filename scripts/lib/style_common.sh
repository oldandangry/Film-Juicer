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

style_clang_format_bin() {
    if [[ -n "${JUICER_CLANG_FORMAT:-}" ]]; then
        printf '%s\n' "$JUICER_CLANG_FORMAT"
        return 0
    fi

    local candidate
    for candidate in \
        "/mnt/c/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin/clang-format.exe" \
        "/c/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin/clang-format.exe"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    style_die "VS 18 clang-format was not found at a known path; set JUICER_CLANG_FORMAT to the tested executable"
}

style_verify_clang_format() {
    local binary="$1"
    local version
    if ! version="$("$binary" --version 2>&1)"; then
        style_die "Unable to query clang-format version: $binary"
    fi
    if [[ ! "$version" =~ clang-format[[:space:]]+version[[:space:]]+22\. ]]; then
        style_die "Film-Juicer requires VS 18 clang-format major version 22; selected '$binary' reported '$version'"
    fi
    printf 'Using clang-format: %s (%s)\n' "$binary" "$version"
}

STYLE_POLICY_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/source_file_policy.json"
declare -a STYLE_SUPPORTED_EXTENSIONS=()
declare -a STYLE_HEADER_LIKE_EXTENSIONS=()
declare -a STYLE_TRANSLATION_UNIT_EXTENSIONS=()
declare -a STYLE_IGNORED_PREFIXES=()
declare -a STYLE_GENERATED_PATHS=()

style_load_source_policy() {
    [[ -f "$STYLE_POLICY_PATH" ]] || style_die "Source file policy not found: $STYLE_POLICY_PATH"
    style_require_cmd python3

    local category value
    while IFS=$'\t' read -r category value; do
        case "$category" in
            supported)
                STYLE_SUPPORTED_EXTENSIONS+=("$value")
                ;;
            header)
                STYLE_HEADER_LIKE_EXTENSIONS+=("$value")
                ;;
            translation)
                STYLE_TRANSLATION_UNIT_EXTENSIONS+=("$value")
                ;;
            ignored)
                STYLE_IGNORED_PREFIXES+=("$value")
                ;;
            generated)
                STYLE_GENERATED_PATHS+=("$value")
                ;;
            *)
                style_die "Unknown source policy category: $category"
                ;;
        esac
    done < <(python3 - "$STYLE_POLICY_PATH" <<'PY'
import json
import pathlib
import sys

policy = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for category, key in (
    ("supported", "supportedExtensions"),
    ("header", "headerLikeExtensions"),
    ("translation", "translationUnitExtensions"),
    ("ignored", "ignoredPrefixes"),
    ("generated", "generatedPaths"),
):
    for value in policy[key]:
        print(f"{category}\t{value}")
PY
    )

    [[ ${#STYLE_SUPPORTED_EXTENSIONS[@]} -gt 0 ]] || style_die "Source file policy has no supported extensions"
    [[ ${#STYLE_HEADER_LIKE_EXTENSIONS[@]} -gt 0 ]] || style_die "Source file policy has no header-like extensions"
    [[ ${#STYLE_TRANSLATION_UNIT_EXTENSIONS[@]} -gt 0 ]] || style_die "Source file policy has no translation-unit extensions"
}

style_extension_in() {
    local path="${1,,}"
    shift
    local extension
    for extension in "$@"; do
        [[ "$path" == *"${extension,,}" ]] && return 0
    done
    return 1
}

style_is_supported_source() {
    style_extension_in "$1" "${STYLE_SUPPORTED_EXTENSIONS[@]}"
}

style_is_header_like() {
    style_extension_in "$1" "${STYLE_HEADER_LIKE_EXTENSIONS[@]}"
}

style_is_translation_unit() {
    style_extension_in "$1" "${STYLE_TRANSLATION_UNIT_EXTENSIONS[@]}"
}

style_is_ignored_path() {
    local path="$1"
    local prefix generated
    for prefix in "${STYLE_IGNORED_PREFIXES[@]}"; do
        [[ "$path" == "$prefix"* ]] && return 0
    done
    for generated in "${STYLE_GENERATED_PATHS[@]}"; do
        [[ "$path" == "$generated" ]] && return 0
    done
    return 1
}

style_supported_extension_regex() {
    local -a names=()
    local extension
    for extension in "${STYLE_SUPPORTED_EXTENSIONS[@]}"; do
        names+=("${extension#.}")
    done
    local joined
    joined="$(IFS='|'; printf '%s' "${names[*]}")"
    printf '.*\\.(%s)$\n' "$joined"
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

style_load_source_policy
