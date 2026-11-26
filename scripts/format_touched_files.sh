#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${JUICER_PYTHON:-python3}"
if [[ "${1:-}" == "--check" ]]; then
    shift
    exec "$python_bin" "$script_dir/check.py" format --base HEAD "$@"
fi
exec "$python_bin" "$script_dir/check.py" format --base HEAD --fix "$@"
