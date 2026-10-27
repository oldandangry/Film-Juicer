#!/usr/bin/env python3
"""Check maintained Python, shell and workflow source without changing it."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


class ToolingError(RuntimeError):
    pass


def source_files(root: Path) -> dict[str, list[str]]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root, check=True, capture_output=True,
    )
    files: dict[str, list[str]] = {"python": [], "shell": [], "workflow": []}
    for name in sorted({os.fsdecode(p) for p in result.stdout.split(b"\0") if p}):
        path = Path(name)
        parts = path.parts
        if any(p in {"fixtures", "generated", "vendor", "vendored", "third_party"} for p in parts):
            continue
        kind = None
        if parts[0] in {"scripts", "tests"}:
            kind = {".py": "python", ".sh": "shell"}.get(path.suffix)
        elif len(parts) == 3 and parts[:2] == (".github", "workflows") and path.suffix in {".yml", ".yaml"}:
            kind = "workflow"
        if kind is None:
            continue
        if not (root / path).exists() and not (root / path).is_symlink():
            continue  # Deleted tracked files are outside the current source membership.
        resolved = (root / path).resolve(strict=True)
        if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
            raise ToolingError(f"lint input is outside the repository or is not a file: {name}")
        files[kind].append(name)
    return files


def require_tool(name: str, value: str, version_arg: str, pattern: str) -> Path:
    executable = shutil.which(value)
    if executable is None:
        raise ToolingError(f"required tool is missing: {name} ({value})")
    path = Path(executable).resolve(strict=True)
    result = subprocess.run([str(path), version_arg], check=True, capture_output=True, text=True, timeout=None)
    text = result.stdout + result.stderr
    if re.search(pattern, text, re.MULTILINE) is None:
        raise ToolingError(f"wrong {name} version: {text.strip()}")
    print(f"{name}: {text.splitlines()[0]}", flush=True)
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_identities(paths: list[Path]) -> dict[Path, tuple[str, str]]:
    identities = {}
    for path in paths:
        checksum = digest(path)
        identities[path] = (str(path.resolve(strict=True)), checksum)
    return identities


def check_python(root: Path, files: list[str], ruff: Path, config: Path) -> None:
    if files:
        subprocess.run([str(ruff), "check", "--no-cache", "--config", str(config), "--", *files],
                       cwd=root, check=True, timeout=None)


def check(root: Path, ruff: str, shellcheck: str, actionlint: str) -> dict[str, int]:
    root = root.resolve(strict=True)
    tools = {
        "ruff": require_tool("Ruff", ruff, "--version", r"^ruff 0\.15\.22$"),
        "shellcheck": require_tool("ShellCheck", shellcheck, "--version", r"^version: 0\.11\.0$"),
        "actionlint": require_tool("actionlint", actionlint, "-version", r"^1\.7\.12$"),
    }
    files = source_files(root)
    config = root / "ruff.toml"
    inputs = [root / n for names in files.values() for n in names]
    inputs += [config, Path(__file__), *tools.values()]
    before = file_identities(inputs)
    check_python(root, files["python"], tools["ruff"], config)
    if files["shell"]:
        subprocess.run([str(tools["shellcheck"]), "--external-sources", "--source-path=SCRIPTDIR", "--", *files["shell"]],
                       cwd=root, check=True, timeout=None)
    if files["workflow"]:
        subprocess.run([str(tools["actionlint"]), "-shellcheck", str(tools["shellcheck"]), "-pyflakes", "", "--", *files["workflow"]],
                       cwd=root, check=True, timeout=None)
    after = file_identities(inputs)
    if before != after or files != source_files(root):
        raise ToolingError("tooling inputs changed during checks")
    return {kind: len(names) for kind, names in files.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--ruff", default="ruff")
    parser.add_argument("--shellcheck", default="shellcheck")
    parser.add_argument("--actionlint", default="actionlint")
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        counts = check(args.repo_root, args.ruff, args.shellcheck, args.actionlint)
    except (ToolingError, OSError, subprocess.SubprocessError) as error:
        print(f"Tooling checks failed: {error}")
        return 1
    print(f"Tooling checks passed: {counts}; elapsed={time.perf_counter() - start:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
