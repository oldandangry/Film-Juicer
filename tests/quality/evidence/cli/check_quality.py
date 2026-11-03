"""Finite quality checks for the synthetic C11 CLI admission recipe only."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid


SOURCE_FILES = (
    ".gitignore", "CMakeLists.txt", "CMakePresets.json", "input.txt",
    "scripts/check-quality.py", "scripts/qualification.py", "src/evidence_child.c",
    "tests/quality/test_miniature.py",
)
POLICY = {
    "ruff": ["check", "--isolated", "--no-cache", "--target-version", "py313", "--select", "E4,E7,E9,F"],
    "format": "{BasedOnStyle: Microsoft, IndentWidth: 4, ColumnLimit: 0, PointerAlignment: Left, BreakBeforeBraces: Attach, SortIncludes: Never}",
    "tidy": "{Checks: '-*,clang-analyzer-*', WarningsAsErrors: '*', InheritParentConfig: false}",
    "checks": ["diff", "syntax", "ruff", "format", "tidy"],
    "executable_roles": {"posix": "execute-mask-and-effective-id-launch-access",
                         "windows": "ordinary-x-ok-without-acl-proof"},
}


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    lexical = Path(path).absolute()
    resolved = lexical.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("synthetic quality input is not a file: " + str(lexical))
    return {"resolved": str(resolved), "sha256": checksum(resolved)}


def executable_identity(path):
    binding = identity(path)
    path = Path(path)
    if os.name == "posix":
        if os.access not in os.supports_effective_ids:
            raise ValueError("synthetic executable identity needs effective-id launch access support")
        execute_mask = path.stat().st_mode & 0o111
        launch_access = os.access(path, os.X_OK, effective_ids=True)
        if not execute_mask or not launch_access:
            raise ValueError("synthetic executable has no effective-id launch access: " + str(path))
    else:
        execute_mask = None
        launch_access = os.access(path, os.X_OK)
        if not launch_access:
            raise ValueError("synthetic executable is unavailable: " + str(path))
    return {**binding, "execute_mask": execute_mask, "launch_access": launch_access}


def source(root):
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    ).decode().split("\0")
    if set(names) - {""} != set(SOURCE_FILES):
        raise ValueError("synthetic quality source membership changed")
    return {name: checksum(root / name) for name in SOURCE_FILES}


def tool(name):
    suffix = ".exe" if os.name == "nt" else ""
    llvm = (Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin")
            if os.name == "nt" else Path("/usr/lib/llvm-22/bin"))
    choices = ([sys.executable] if name == "python" else
               [str(Path(sys.executable).parent / ("ruff" + suffix)), "ruff"] if name == "ruff" else
               [name] if name == "git" else
               [os.environ.get("JUICER_" + name.upper().replace("-", "_"), ""),
                name + "-22", str(llvm / (name + suffix)), name])
    for choice in choices:
        if not choice:
            continue
        selected = shutil.which(choice)
        if selected:
            path = Path(selected).absolute()
            version = subprocess.check_output([str(path), "--version"], text=True, stderr=subprocess.STDOUT).strip()
            if name == "ruff" and version != "ruff 0.15.22":
                raise ValueError("synthetic quality requires Ruff 0.15.22")
            if name in {"clang-format", "clang-tidy"} and not re.search(r"(?:LLVM|clang-format|clang-tidy) version 22\.", version):
                raise ValueError("synthetic quality requires LLVM 22: " + version)
            if name == "python" and not version.startswith("Python 3.13."):
                raise ValueError("synthetic quality requires Python 3.13")
            return {"path": str(path), **executable_identity(path), "version": version}
    raise ValueError("synthetic quality tool unavailable: " + name)


def quality_commands(root, build, base, tools):
    selected = {name: value["path"] for name, value in tools.items()}
    python_files = [name for name in SOURCE_FILES if name.endswith(".py")]
    return {
        "diff": [selected["git"], "diff", "--check", base, "--"],
        "syntax": [selected["python"], "-B", "-c",
            "import ast,pathlib,sys; [ast.parse(pathlib.Path(p).read_bytes(),filename=p) for p in sys.argv[1:]]",
            *python_files],
        "ruff": [selected["ruff"], *POLICY["ruff"], "--", *python_files],
        "format": [selected["clang-format"], "--dry-run", "--Werror", "--style=" + POLICY["format"],
                   "src/evidence_child.c"],
        "tidy": [selected["clang-tidy"], "--config=" + POLICY["tidy"], "-p", str(build), "src/evidence_child.c"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=("linux-debug", "windows-clang-debug"))
    parser.add_argument("--qualification", action="store_true", required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    build = root / "out/build" / args.preset
    destination = root / "out/validation" / args.preset / "synthetic-quality" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    report_path = destination / "quality.json"
    report = {
        "protocol": "synthetic-cli-quality-v1", "assurance": "synthetic-c11-cli-quality",
        "status": "running", "root": str(root), "preset": args.preset, "base": None,
        "candidate": {}, "inputs": {}, "tools": {}, "policy": POLICY, "checks": [],
        "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
    }
    print("Synthetic quality report:", report_path, flush=True)

    def run(name, command):
        stdout = destination / (name + ".stdout")
        stderr = destination / (name + ".stderr")
        with stdout.open("wb") as out, stderr.open("wb") as err:
            completed = subprocess.run(command, cwd=root, stdout=out, stderr=err, check=False)
        report["checks"].append({
            "name": name, "command": list(command), "cwd": str(root), "returncode": completed.returncode,
            "stdout": {"path": str(stdout), "sha256": checksum(stdout)},
            "stderr": {"path": str(stderr), "sha256": checksum(stderr)},
        })
        if completed.returncode:
            raise ValueError("synthetic quality check failed: " + name)

    try:
        report["candidate"] = source(root)
        report["base"] = subprocess.check_output(
            ["git", "rev-parse", "--verify", args.base + "^{commit}"], cwd=root, text=True
        ).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", report["base"]):
            raise ValueError("synthetic quality base is not an immutable Git commit")
        entry_path = build / "compile_commands.json"
        entries = json.loads(entry_path.read_text())
        if len(entries) != 1 or Path(entries[0]["file"]).resolve() != root / "src/evidence_child.c":
            raise ValueError("synthetic quality requires its one actual C11 compilation entry")
        paths = [root / name for name in SOURCE_FILES] + [entry_path]
        report["inputs"] = {str(path): identity(path) for path in paths}
        report["tools"] = {name: tool(name) for name in ("python", "git", "ruff", "clang-format", "clang-tidy")}
        python_files = [name for name in SOURCE_FILES if name.endswith(".py")]
        for name in python_files:
            ast.parse((root / name).read_bytes(), filename=name)
        for name, command in quality_commands(root, build, report["base"], report["tools"]).items():
            run(name, command)
        if source(root) != report["candidate"] or {str(path): identity(path) for path in paths} != report["inputs"]:
            raise ValueError("synthetic quality inputs changed during checks")
        for name, binding in report["tools"].items():
            if executable_identity(binding["path"]) != {
                    key: binding[key] for key in ("resolved", "sha256", "execute_mask", "launch_access")}:
                raise ValueError("synthetic quality tool changed during checks: " + name)
        report["status"] = "passed"
        return 0
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print("Synthetic quality seal:", checksum(report_path), flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print("Synthetic quality failed:", error, file=sys.stderr)
        raise SystemExit(1) from error
