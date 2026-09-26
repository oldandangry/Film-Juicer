#!/usr/bin/env python3
"""Run Film-Juicer's tracked source hygiene, formatting, and lint checks."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


RUST_POLICY_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "clippy.toml",
    "rust-toolchain.toml",
    "rustfmt.toml",
}
ROOT_OWNED_FILES = {
    ".clang-format",
    ".clang-tidy",
    ".gitignore",
    "CMakeLists.txt",
    "CMakePresets.json",
    "CONTRIBUTING.md",
}
RUST_BUILD_FILES = {
    "CMakeLists.txt",
    "cmake/JuicerRust.cmake",
}
NATIVE_POLICY_FILES = {
    ".clang-format",
    ".clang-tidy",
    "scripts/check-quality.py",
    "scripts/source_file_policy.json",
}
QUALITY_RUNNER_PREFIXES = ("tests/quality/",)
PRESET_TARGETS = {
    "linux-debug": "x86_64-unknown-linux-gnu",
    "linux-release": "x86_64-unknown-linux-gnu",
    "windows-clang-debug": "x86_64-pc-windows-msvc",
    "windows-clang-release": "x86_64-pc-windows-msvc",
}
RUST_VERSION = "1.98.1"
REPRESENTATIVE_NATIVE_FILES = (
    "src/main.cpp",
    "tests/hash/hash_contract_test.cpp",
)


class QualityError(RuntimeError):
    """A required quality check could not complete successfully."""


@dataclass(frozen=True)
class SourcePolicy:
    native_extensions: frozenset[str]
    header_extensions: frozenset[str]
    translation_extensions: frozenset[str]
    owned_prefixes: tuple[str, ...]
    excluded_prefixes: tuple[str, ...]
    generated_paths: frozenset[str]


@dataclass(frozen=True)
class CompilationEntry:
    path: str
    command: str


class Runner:
    def __init__(self, root: Path, log_dir: Path) -> None:
        self.root = root
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0

    def run(
        self,
        command: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        label: str,
    ) -> str:
        self.index += 1
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
        log_path = self.log_dir / f"{self.index:02d}-{safe_label}.log"
        print("+", subprocess.list2cmdline(command))
        completed = subprocess.run(
            command,
            cwd=self.root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode != 0:
            raise QualityError(
                f"{label} failed with exit code {completed.returncode}; see {log_path}"
            )
        return completed.stdout.strip()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--base", help="accepted parent or merge-base")
    selection.add_argument("--files", nargs="+", help="explicit changed paths")
    selection.add_argument(
        "--all-owned", action="store_true", help="audit every tracked owned source"
    )
    parser.add_argument("--preset", required=True, choices=sorted(PRESET_TARGETS))
    parser.add_argument("--cargo", help="Cargo executable override")
    parser.add_argument("--rustc", help="rustc executable override")
    parser.add_argument("--clang-format", dest="clang_format")
    parser.add_argument("--clang-tidy", dest="clang_tidy")
    return parser.parse_args()


def normalize_path(root: Path, value: str) -> str:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(root)
        except ValueError as exc:
            raise QualityError(f"path is outside the repository: {value}") from exc
    normalized = PurePosixPath(path.as_posix())
    if normalized.is_absolute() or ".." in normalized.parts:
        raise QualityError(f"path is outside the repository: {value}")
    return normalized.as_posix().removeprefix("./")


def parse_name_status(payload: bytes) -> tuple[list[str], list[str]]:
    tokens = payload.split(b"\0")
    if tokens and not tokens[-1]:
        tokens.pop()
    selected: list[str] = []
    excluded: list[str] = []
    index = 0
    while index < len(tokens):
        status = tokens[index].decode("utf-8", errors="strict")
        index += 1
        if not status:
            raise QualityError("git produced an empty name-status record")
        kind = status[0]
        if kind in {"R", "C"}:
            if index + 1 >= len(tokens):
                raise QualityError("git produced a truncated rename/copy record")
            old_path = tokens[index].decode("utf-8", errors="surrogateescape")
            new_path = tokens[index + 1].decode("utf-8", errors="surrogateescape")
            index += 2
            excluded.append(f"renamed source: {old_path}")
            selected.append(new_path)
        else:
            if index >= len(tokens):
                raise QualityError("git produced a truncated name-status record")
            path = tokens[index].decode("utf-8", errors="surrogateescape")
            index += 1
            if kind == "D":
                excluded.append(f"deleted: {path}")
            else:
                selected.append(path)
    return selected, excluded


def run_git(root: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise QualityError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def load_policy(root: Path) -> SourcePolicy:
    path = root / "scripts/source_file_policy.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityError(f"invalid source policy: {path}: {exc}") from exc
    if raw.get("schemaVersion") != 1:
        raise QualityError(f"unsupported source policy schema: {path}")
    required = (
        "supportedExtensions",
        "headerLikeExtensions",
        "translationUnitExtensions",
        "qualityOwnedPrefixes",
        "qualityExcludedPrefixes",
        "generatedPaths",
    )
    if any(not isinstance(raw.get(key), list) for key in required):
        raise QualityError(f"source policy is missing a required list: {path}")
    return SourcePolicy(
        native_extensions=frozenset(raw["supportedExtensions"]),
        header_extensions=frozenset(raw["headerLikeExtensions"]),
        translation_extensions=frozenset(raw["translationUnitExtensions"]),
        owned_prefixes=tuple(raw["qualityOwnedPrefixes"]),
        excluded_prefixes=tuple(raw["qualityExcludedPrefixes"]),
        generated_paths=frozenset(raw["generatedPaths"]),
    )


def is_owned(path: str, policy: SourcePolicy) -> bool:
    if path in RUST_POLICY_FILES or path in ROOT_OWNED_FILES:
        return True
    if path.startswith(".github/workflows/"):
        return True
    return path.startswith(policy.owned_prefixes)


def exclusion_reason(path: str, policy: SourcePolicy) -> str | None:
    if path in policy.generated_paths:
        return "generated source"
    for prefix in policy.excluded_prefixes:
        if path.startswith(prefix):
            return f"excluded prefix {prefix}"
    if not is_owned(path, policy):
        return "outside owned quality scope"
    return None


def is_native(path: str, policy: SourcePolicy) -> bool:
    suffix = PurePosixPath(path).suffix.lower()
    return (
        suffix in policy.native_extensions
        and path.startswith(("src/", "native/", "tests/"))
        and "/fixtures/" not in path
    )


def is_rust(path: str) -> bool:
    return path.endswith(".rs") or path in RUST_POLICY_FILES or path.endswith("/Cargo.toml")


def check_source_hygiene(
    root: Path, selected: Sequence[str], policy: SourcePolicy, log_path: Path
) -> None:
    # Preserve the existing source-text gates, including matches in comments.
    rules = (
        (re.compile(r"[ \t]+$"), "trailing whitespace"),
        (re.compile(r"^(<<<<<<<|=======|>>>>>>>|\|\|\|\|\|\|\|)"), "merge conflict marker"),
        (re.compile(r"\bJUICER_TESTS\b"), "retired JUICER_TESTS flag; use BUILD_TESTING"),
        (
            re.compile(r"\bJUICER_BUILD_VALIDATION\b"),
            "retired JUICER_BUILD_VALIDATION flag; use BUILD_TESTING",
        ),
        (re.compile(r"\bstd::endl\b"), r"std::endl is prohibited; use '\n'"),
    )
    header_rule = (
        re.compile(r"^\s*using\s+namespace\b"),
        "using namespace directives are prohibited in headers",
    )
    diagnostics: list[str] = []
    checked = 0
    for path in selected:
        if exclusion_reason(path, policy) is not None or not is_native(path, policy):
            continue
        checked += 1
        try:
            source = (root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            diagnostics.append(f"{path}: cannot read source for hygiene checks: {exc}")
            continue
        path_rules = rules
        if PurePosixPath(path).suffix.lower() in policy.header_extensions:
            path_rules = (*rules, header_rule)
        for line_number, line in enumerate(source.splitlines(), start=1):
            for pattern, message in path_rules:
                match = pattern.search(line)
                if match is not None:
                    diagnostics.append(
                        f"{path}:{line_number}:{match.start() + 1}: {message}"
                    )

    report = "\n".join(diagnostics) if diagnostics else (
        f"Source hygiene checks passed for {checked} native source file(s)."
    )
    log_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    if diagnostics:
        raise QualityError(f"source hygiene checks failed; see {log_path}")


def select_paths(
    root: Path, arguments: argparse.Namespace, policy: SourcePolicy
) -> tuple[list[str], list[str]]:
    excluded: list[str] = []
    if arguments.base:
        payload = run_git(
            root,
            ["diff", "--name-status", "-z", "--find-renames", arguments.base, "--"],
        )
        raw_paths, excluded = parse_name_status(payload)
        untracked = run_git(root, ["ls-files", "--others", "--exclude-standard", "-z"])
        raw_paths.extend(
            token.decode("utf-8", errors="surrogateescape")
            for token in untracked.split(b"\0")
            if token
        )
    elif arguments.files:
        raw_paths = list(arguments.files)
    else:
        tracked = run_git(root, ["ls-files", "-z"])
        raw_paths = [
            token.decode("utf-8", errors="surrogateescape")
            for token in tracked.split(b"\0")
            if token
        ]

    selected: set[str] = set()
    for raw_path in raw_paths:
        path = normalize_path(root, raw_path)
        reason = exclusion_reason(path, policy)
        if reason is not None:
            excluded.append(f"{path}: {reason}")
            continue
        if not (root / path).is_file():
            excluded.append(f"{path}: not a present file")
            continue
        selected.add(path)
    if not selected:
        raise QualityError("the required owned-file selection is empty")
    return sorted(selected), sorted(set(excluded))


def resolve_tool(
    requested: str | None, environment_name: str, candidates: Sequence[str]
) -> str:
    environment_value = os.environ.get(environment_name)
    if requested:
        choices = [requested]
    elif environment_value:
        choices = [environment_value]
    else:
        choices = list(candidates)
    for choice in choices:
        if not choice:
            continue
        resolved = shutil.which(choice)
        if resolved:
            return resolved
        path = Path(choice)
        if path.is_file():
            return str(path)
    raise QualityError(f"required tool not found: {environment_name}")


def cargo_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
        environment.pop(name, None)
    return environment


def check_workspace_contract(root: Path) -> None:
    for manifest in (
        root / "rust/film-juicer-core/Cargo.toml",
        root / "rust/film-juicer-plugin/Cargo.toml",
    ):
        try:
            parsed = tomllib.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise QualityError(f"invalid member manifest: {manifest}: {exc}") from exc
        if parsed.get("lints", {}).get("workspace") is not True:
            raise QualityError(f"member manifest does not inherit workspace lints: {manifest}")
    core = (root / "rust/film-juicer-core/src/lib.rs").read_text(encoding="utf-8")
    if "#![forbid(unsafe_code)]" not in core:
        raise QualityError("film-juicer-core must retain #![forbid(unsafe_code)]")


def check_rust(
    root: Path,
    runner: Runner,
    arguments: argparse.Namespace,
    target: str,
) -> None:
    cargo = resolve_tool(arguments.cargo, "JUICER_CARGO", ("cargo",))
    cargo_dir = str(Path(cargo).resolve().parent)
    rustc_candidates = (str(Path(cargo_dir) / ("rustc.exe" if os.name == "nt" else "rustc")), "rustc")
    rustc = resolve_tool(arguments.rustc, "JUICER_RUSTC", rustc_candidates)
    environment = cargo_environment()
    environment["CLIPPY_CONF_DIR"] = str(root)

    cargo_version = runner.run([cargo, "--version"], env=environment, label="cargo-version")
    rustc_version = runner.run(
        [rustc, "--version", "--verbose"], env=environment, label="rustc-version"
    )
    rustfmt_version = runner.run(
        [cargo, "fmt", "--version"], env=environment, label="rustfmt-version"
    )
    clippy_version = runner.run(
        [cargo, "clippy", "--version"], env=environment, label="clippy-version"
    )
    if not cargo_version.startswith(f"cargo {RUST_VERSION} "):
        raise QualityError(f"Cargo {RUST_VERSION} is required; got {cargo_version}")
    if not rustc_version.startswith(f"rustc {RUST_VERSION} "):
        raise QualityError(f"rustc {RUST_VERSION} is required; got {rustc_version.splitlines()[0]}")
    if "rustfmt 1.9.0-stable" not in rustfmt_version:
        raise QualityError(f"qualified rustfmt 1.9.0-stable is required; got {rustfmt_version}")
    if not clippy_version.startswith("clippy 0.1.98 "):
        raise QualityError(f"qualified Clippy 0.1.98 is required; got {clippy_version}")

    check_workspace_contract(root)
    target_dir = root / "out/build" / arguments.preset / "cargo"
    common = [
        cargo,
        "clippy",
        "--locked",
        "--target",
        target,
        "--target-dir",
        str(target_dir),
    ]
    runner.run([cargo, "fmt", "--all", "--", "--check"], env=environment, label="rustfmt")
    for package in ("film-juicer-core", "film-juicer-plugin"):
        runner.run(
            [*common, "-p", package, "--all-targets", "--", "-D", "warnings"],
            env=environment,
            label=f"clippy-{package}-dev",
        )
        runner.run(
            [*common, "-p", package, "--all-targets", "--release", "--", "-D", "warnings"],
            env=environment,
            label=f"clippy-{package}-release",
        )

    runner.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests/quality",
         "-p", "test_rust_naming.py", "-v"],
        env={
            **environment,
            "JUICER_CARGO": cargo,
            "JUICER_RUST_TARGET": target,
            "JUICER_TEST_ARTIFACT_DIR": str(runner.log_dir / "rust-naming"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        label="rust-naming-enforcement",
    )


def compilation_entries(root: Path, preset: str) -> list[CompilationEntry]:
    path = root / "out/build" / preset / "compile_commands.json"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityError(f"missing or invalid compilation database: {path}: {exc}") from exc
    entries: list[CompilationEntry] = []
    for raw in parsed:
        file_path = Path(raw.get("file", ""))
        if not file_path.is_absolute():
            file_path = Path(raw.get("directory", "")) / file_path
        try:
            relative = file_path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        command = raw.get("command") or " ".join(raw.get("arguments", []))
        if command:
            entries.append(CompilationEntry(relative, command))
    if not entries:
        raise QualityError(f"compilation database contains no owned entries: {path}")
    return entries


INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def includes_header(root: Path, translation_unit: str, header: str) -> bool:
    pending = [translation_unit]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if current == header:
            return True
        current_path = root / current
        try:
            text = current_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for include in INCLUDE_PATTERN.findall(text):
            candidates = (
                current_path.parent / include,
                root / include,
                root / "src" / include,
                root / "native" / include,
                root / "tests" / include,
            )
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                try:
                    relative = candidate.resolve().relative_to(root).as_posix()
                except ValueError:
                    continue
                pending.append(relative)
                break
    return False


def tidy_translation_units(
    root: Path,
    selected_native: Sequence[str],
    entries: Sequence[CompilationEntry],
    policy: SourcePolicy,
) -> list[str]:
    host_entries = [entry for entry in entries if not entry.path.endswith(".cu")]
    selected: set[str] = set()
    cuda_paths: list[str] = []
    for path in selected_native:
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in {".cu", ".cuh"}:
            cuda_paths.append(path)
            continue
        if suffix in policy.translation_extensions:
            matches = [entry for entry in host_entries if entry.path == path]
            production_matches = [
                entry for entry in matches if "JUICER_ADMISSION_TEST_HOOK" not in entry.command
                and "JUICER_CONTEXT_DRAIN_TEST_HOOK" not in entry.command
            ]
            if production_matches:
                matches = production_matches
            if not matches:
                raise QualityError(f"selected translation unit is absent from the compilation database: {path}")
            selected.add(path)
        elif suffix in policy.header_extensions:
            owners = [
                entry.path for entry in host_entries if includes_header(root, entry.path, path)
            ]
            if not owners:
                raise QualityError(f"no consuming translation unit found for selected header: {path}")
            selected.update(owners)
    if cuda_paths:
        joined = ", ".join(cuda_paths)
        raise QualityError(
            "tracked CUDA clang-tidy argument translation is not yet qualified; "
            f"selected CUDA source is incomplete evidence: {joined}"
        )
    return sorted(selected)


def check_native(
    root: Path,
    runner: Runner,
    arguments: argparse.Namespace,
    selected_native: Sequence[str],
    policy: SourcePolicy,
) -> None:
    format_candidates = (
        arguments.clang_format,
        os.environ.get("JUICER_CLANG_FORMAT"),
        "clang-format-22",
        "clang-format",
        r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin\clang-format.exe",
    )
    tidy_candidates = (
        arguments.clang_tidy,
        os.environ.get("JUICER_CLANG_TIDY"),
        "clang-tidy-22",
        "clang-tidy",
        r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin\clang-tidy.exe",
    )
    clang_format = resolve_tool(format_candidates[0], "JUICER_CLANG_FORMAT", format_candidates[2:])
    clang_tidy = resolve_tool(tidy_candidates[0], "JUICER_CLANG_TIDY", tidy_candidates[2:])
    format_version = runner.run([clang_format, "--version"], label="clang-format-version")
    tidy_version = runner.run([clang_tidy, "--version"], label="clang-tidy-version")
    if not re.search(r"clang-format version 22\.", format_version):
        raise QualityError(f"qualified clang-format major 22 is required; got {format_version}")
    if not re.search(r"LLVM version 22\.|clang-tidy version 22\.", tidy_version):
        raise QualityError(f"qualified clang-tidy major 22 is required; got {tidy_version}")

    runner.run(
        [clang_format, "--dry-run", "--Werror", "--style=file", *selected_native],
        label="clang-format",
    )
    runner.run([clang_tidy, "--verify-config"], label="clang-tidy-config")
    entries = compilation_entries(root, arguments.preset)
    translation_units = tidy_translation_units(root, selected_native, entries, policy)
    if not translation_units:
        raise QualityError("native selection has no translation unit for clang-tidy")
    build_dir = root / "out/build" / arguments.preset
    for index, translation_unit in enumerate(translation_units, start=1):
        runner.run(
            [
                clang_tidy,
                "-p",
                str(build_dir),
                "--warnings-as-errors=*",
                translation_unit,
            ],
            label=f"clang-tidy-{index}-{PurePosixPath(translation_unit).name}",
        )


def main() -> int:
    arguments = parse_arguments()
    root = Path(__file__).resolve().parent.parent
    policy = load_policy(root)
    selected, excluded = select_paths(root, arguments, policy)
    target = PRESET_TARGETS[arguments.preset]
    log_dir = root / "out/validation" / arguments.preset / "quality"
    runner = Runner(root, log_dir)

    print("Selected files:")
    for path in selected:
        print(f"  {path}")
    print("Excluded categories:")
    if excluded:
        for reason in excluded:
            print(f"  {reason}")
    else:
        print("  none")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Rust target: {target}")

    check_source_hygiene(root, selected, policy, log_dir / "source-hygiene.log")

    if arguments.base:
        diff_command = ["git", "diff", "--check", arguments.base, "--"]
    elif arguments.files:
        diff_command = ["git", "diff", "--check", "--", *selected]
    else:
        diff_command = ["git", "diff", "--check"]
    runner.run(diff_command, label="git-diff-check")

    runner_changed = (
        any(path.startswith(QUALITY_RUNNER_PREFIXES) for path in selected)
        or any(path in NATIVE_POLICY_FILES for path in selected)
        or any(path.startswith(".github/workflows/") for path in selected)
    )
    rust_required = (
        runner_changed
        or any(path in RUST_BUILD_FILES for path in selected)
        or any(is_rust(path) for path in selected)
    )
    native_required = runner_changed or any(
        path in {".clang-format", ".clang-tidy"} or is_native(path, policy)
        for path in selected
    )

    if rust_required:
        check_rust(root, runner, arguments, target)
    if native_required:
        native_files = [path for path in selected if is_native(path, policy)]
        if runner_changed or any(path in {".clang-format", ".clang-tidy"} for path in selected):
            native_files.extend(
                path for path in REPRESENTATIVE_NATIVE_FILES if (root / path).is_file()
            )
        native_files = sorted(set(native_files))
        if not native_files:
            raise QualityError("native policy changed without a representative native selection")
        check_native(root, runner, arguments, native_files, policy)

    if not rust_required and not native_required:
        print("Documentation/configuration-only selection: diff check and review apply.")
    print(f"Quality checks passed; logs: {log_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
