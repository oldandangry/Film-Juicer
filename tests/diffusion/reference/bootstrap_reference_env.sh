#!/usr/bin/env bash
set -euo pipefail

DIFFUSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DIFFUSION_VENV="$DIFFUSION_ROOT/.tmp/diffusion/reference-venv"
DIFFUSION_EVIDENCE="$DIFFUSION_ROOT/.tmp/diffusion/evidence/reference"
DIFFUSION_LOCK="$DIFFUSION_ROOT/tests/diffusion/reference/requirements.lock"
DIFFUSION_UV_CACHE="$DIFFUSION_ROOT/.tmp/diffusion/uv-cache"
DIFFUSION_UV_PYTHON="$DIFFUSION_ROOT/.tmp/diffusion/uv-python"
DIFFUSION_MPL_CONFIG="$DIFFUSION_ROOT/.tmp/diffusion/matplotlib"

export UV_CACHE_DIR="$DIFFUSION_UV_CACHE"
export UV_PYTHON_INSTALL_DIR="$DIFFUSION_UV_PYTHON"
export MPLCONFIGDIR="$DIFFUSION_MPL_CONFIG"

command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }
mkdir -p \
    "$DIFFUSION_EVIDENCE" \
    "$DIFFUSION_MPL_CONFIG" \
    "$DIFFUSION_UV_CACHE" \
    "$DIFFUSION_UV_PYTHON"
cd "$DIFFUSION_ROOT"

uv python install 3.13 --no-bin
uv venv --clear --managed-python --python 3.13 "$DIFFUSION_VENV"
uv pip sync --python "$DIFFUSION_VENV/bin/python" "$DIFFUSION_LOCK"
uv pip install --python "$DIFFUSION_VENV/bin/python" --no-deps -e "$DIFFUSION_ROOT/external/spektrafilm"

"$DIFFUSION_VENV/bin/python" -c '
import pathlib
import sys
import spektrafilm.model.diffusion as diffusion_module
from spektrafilm.model.diffusion import diffusion_filter_psf, apply_diffusion_filter_um
root = pathlib.Path.cwd().resolve()
module_path = pathlib.Path(diffusion_module.__file__).resolve()
expected = (root / "external/spektrafilm").resolve()
if expected not in module_path.parents:
    raise SystemExit(f"wrong spektrafilm source: {module_path}")
if sys.version_info[:2] != (3, 13):
    raise SystemExit(f"wrong Python: {sys.version}")
print(f"reference_import=PASS python={sys.version.split()[0]} source={module_path}")
'
uv pip freeze --python "$DIFFUSION_VENV/bin/python" > "$DIFFUSION_EVIDENCE/python-freeze.txt"
uv --version > "$DIFFUSION_EVIDENCE/uv-version.txt"
