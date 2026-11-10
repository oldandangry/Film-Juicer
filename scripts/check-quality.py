#!/usr/bin/env python3
"""Run Film-Juicer's tracked source hygiene, formatting, and lint checks."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
# Direct package dependencies, including inactive target-specific declarations.
# A new dependency or build hook is an architectural change, not just a lock update.
RUST_DEPENDENCIES = {
    "film-juicer-core": frozenset({"libm", "serde", "serde_json"}),
    "film-juicer-plugin": frozenset({"film-juicer-core"}),
}
REPRESENTATIVE_NATIVE_FILES = (
    "native/ofx_exports.cpp",
    "src/Cuda/Film/JuicerCudaFilmPipeline.cu",
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
    verifier_independent_prefixes: tuple[str, ...]
    verifier_independent_extensions: frozenset[str]
    verifier_required_paths: frozenset[str]


@dataclass(frozen=True)
class CompilationEntry:
    path: str
    command: str
    directory: Path = Path(".")
    arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisCommand:
    command: Sequence[str]
    directory: Path
    label: str
    source: str = ""


class Runner:
    def __init__(self, root: Path, log_dir: Path, jobs: int = 1, native_run=None,
                 name_prefix: str = "") -> None:
        if jobs < 1:
            raise QualityError("analysis jobs must be at least 1")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from execution import NativeRun

        self.root = root
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.native_run = native_run or NativeRun(root, log_dir)
        self.jobs = jobs
        self.name_prefix = name_prefix
        self.output_lock = threading.Lock()
        self.names: set[str] = set()
        self.expected_checks: list[str] = []
        self.completed_checks: list[str] = []
        self.expected_commands: list[str] = []
        self.python_cases: list[str] = []
        self.consumed_inputs: set[str] = set()
        self.command_output_read_bytes = 0

    def complete(self, name: str) -> None:
        if name not in self.expected_checks:
            raise QualityError(f"completed an undeclared quality check: {name}")
        if name in self.completed_checks:
            raise QualityError(f"completed a quality check more than once: {name}")
        self.completed_checks.append(name)

    def consume(self, paths: Sequence[Path | str]) -> None:
        for path in paths:
            candidate = Path(path)
            if not candidate.is_file():
                continue
            try:
                name = candidate.resolve().relative_to(self.root.resolve()).as_posix()
            except ValueError:
                continue
            self.consumed_inputs.add(name)

    @staticmethod
    def _name(label: str) -> str:
        return re.sub(r"[^a-z0-9-]+", "-", label.lower()).strip("-") or "check"

    def run(
        self,
        command: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        label: str,
        display_output: bool = True,
        stdout_only: bool = False,
    ) -> str:
        with self.output_lock:
            safe_label = self._name(label)
            name = safe_label
            suffix = 2
            while name in self.names:
                name = f"{safe_label}-{suffix}"
                suffix += 1
            self.names.add(name)
            command_name = self.name_prefix + name
            self.expected_commands.append(command_name)
            print("+", subprocess.list2cmdline(command))
        try:
            output_path = self.native_run.command(
                command_name, command, cwd=cwd or self.root, env=env, timeout=None,
                stdout_only=stdout_only,
            )
        except subprocess.CalledProcessError as exc:
            raise QualityError(f"{label} failed with exit code {exc.returncode}; see {self.log_dir / (command_name + '.log')}") from exc
        except Exception as exc:
            raise QualityError(f"{label} could not start command: {exc}; see {self.log_dir / (command_name + '.log')}") from exc
        if stdout_only:
            stdout_path = output_path
            stderr_path = self.log_dir / f"{command_name}.stderr.log"
            stdout_bytes = stdout_path.read_bytes()
            stderr_bytes = stderr_path.read_bytes()
            self.command_output_read_bytes += len(stdout_bytes) + len(stderr_bytes)
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            diagnostics = stderr_bytes.decode("utf-8", errors="replace")
            output = stdout + diagnostics
        else:
            output_bytes = output_path.read_bytes()
            self.command_output_read_bytes += len(output_bytes)
            stdout = output_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n")
            output = stdout
        if output and display_output:
            with self.output_lock:
                print(output, end="" if str(output).endswith("\n") else "\n")
        return stdout.strip()


    def run_analysis(self, commands: Sequence[AnalysisCommand]) -> None:
        failures: list[str] = []
        remaining = iter(commands)
        with ThreadPoolExecutor(max_workers=self.jobs) as executor:
            pending = {}
            while True:
                while not failures and len(pending) < self.jobs:
                    item = next(remaining, None)
                    if item is None:
                        break
                    future = executor.submit(self.run, item.command, cwd=item.directory, label=item.label)
                    pending[future] = item
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    item = pending.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(f"{item.label}: {exc}")
        if failures:
            raise QualityError("analysis failed:\n" + "\n".join(failures))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--base", help="accepted parent or merge-base")
    selection.add_argument("--files", nargs="+", help="explicit changed paths")
    selection.add_argument(
        "--all-owned", action="store_true", help="audit every tracked owned source"
    )
    parser.add_argument("--qualification", action="store_true", help="include representative native/CUDA and complete Rust gates")
    parser.add_argument("--compiler-consumers", nargs="+", help="actual compiler dependency consumers supplied by development verification")
    parser.add_argument("--rust-packages", nargs="+", choices=("film-juicer-core", "film-juicer-plugin"))
    parser.add_argument("--development", action="store_true", help="development checks only; never publishes qualification probes")
    parser.add_argument("--preset", required=True, choices=sorted(PRESET_TARGETS))
    parser.add_argument("--verifier-scope-output", type=Path,
                        help="report-only verifier classification; no build or analysis")
    parser.add_argument("--candidate", default="HEAD", help="commit classified by --verifier-scope-output")
    parser.add_argument("--cargo", help="Cargo executable override")
    parser.add_argument("--rustc", help="rustc executable override")
    parser.add_argument("--clang-format", dest="clang_format")
    parser.add_argument("--clang-tidy", dest="clang_tidy")
    parser.add_argument("--jobs", type=int, default=min(os.process_cpu_count() or 1, 4),
                        help="maximum concurrent native/CUDA analysis commands (default: up to 4 available CPUs)")
    arguments = parser.parse_args()
    if arguments.rust_packages and not arguments.development:
        parser.error("named Rust checks require development verification")
    if arguments.compiler_consumers and not arguments.development:
        parser.error("compiler consumers require development verification")
    if getattr(arguments, "development", False) and arguments.qualification:
        parser.error("development and qualification scopes are distinct")
    if arguments.jobs < 1:
        parser.error("--jobs must be at least 1")
    return arguments


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
        "verifierIndependentPrefixes",
        "verifierIndependentExtensions",
        "verifierRequiredPaths",
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
        verifier_independent_prefixes=tuple(raw["verifierIndependentPrefixes"]),
        verifier_independent_extensions=frozenset(raw["verifierIndependentExtensions"]),
        verifier_required_paths=frozenset(raw["verifierRequiredPaths"]),
    )


def verifier_scope(root: Path, base: str | None, candidate: str, policy: SourcePolicy) -> dict:
    """Classify cumulative committed scope. Uncertainty always requires integration."""
    head = run_git(root, ["rev-parse", "--verify", candidate + "^{commit}"]).decode().strip()
    report = {"candidate": head, "mode": "report-only", "complete": True,
              "required": True, "base": None, "reasons": []}
    try:
        if not base:
            raise QualityError("missing base; complete integration required")
        ancestor = run_git(root, ["merge-base", base, head]).decode().strip()
        report["base"] = ancestor
        payload = run_git(root, ["diff", "--name-status", "-z", "--find-renames", ancestor, head, "--"])
        paths, uncertain = parse_name_status(payload)
        # Renames/deletions and every status other than modification stay broad.
        statuses = run_git(root, ["diff", "--name-only", "-z", "--diff-filter=ACDRTUXB", ancestor, head, "--"])
        reasons = list(uncertain)
        reasons.extend("non-modification: " + path.decode("utf-8", errors="strict")
                       for path in statuses.split(b"\0") if path)
        for name in paths:
            path = normalize_path(root, name)
            independent = (path.startswith(policy.verifier_independent_prefixes)
                           and PurePosixPath(path).suffix in policy.verifier_independent_extensions
                           and path not in policy.generated_paths | policy.verifier_required_paths)
            if not independent:
                reasons.append("verifier dependency or unknown input: " + path)
        report.update(required=bool(reasons), reasons=sorted(set(reasons)))
    except (QualityError, UnicodeError) as error:
        report["reasons"] = [str(error)]
    return report


def is_owned(path: str, policy: SourcePolicy) -> bool:
    if path in RUST_POLICY_FILES or path in ROOT_OWNED_FILES:
        return True
    if path.startswith(".github/workflows/"):
        return True
    return path.startswith(policy.owned_prefixes)


def exclusion_reason(path: str, policy: SourcePolicy) -> str | None:
    for prefix in policy.excluded_prefixes:
        if path.startswith(prefix):
            return f"excluded prefix {prefix}"
    if not is_owned(path, policy):
        return "outside owned quality scope"
    # Generated Rust still needs workspace formatting and lint checks.
    if path in policy.generated_paths and not path.endswith(".rs"):
        return "generated source"
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
            if arguments.base:
                raise QualityError(f"discovered owned file is not present: {path}")
            excluded.append(f"{path}: not a present file")
            continue
        selected.add(path)
    if not selected and not arguments.base:
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
    for name in ("RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CLIPPY_CONF_DIR"):
        environment.pop(name, None)
    return environment


def check_probe_configuration(root: Path, workspace: Path) -> None:
    """Disposable probe sources must inherit the repository Cargo configuration."""
    def configurations(directory):
        result = []
        for parent in (directory, *directory.parents):
            for name in ("config", "config.toml"):
                path = parent / ".cargo" / name
                if path.exists() or path.is_symlink():
                    if not path.is_file():
                        raise QualityError("Cargo configuration is not a file: " + str(path))
                    result.append(path.resolve())
        return result
    if configurations(workspace) != configurations(root):
        raise QualityError("probe output directory has different Cargo configuration")


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


def check_dependency_contract(root: Path, metadata: dict) -> None:
    packages = {package["id"]: package for package in metadata["packages"]}
    resolved = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    members = [packages[identity] for identity in metadata["workspace_members"]]
    if {member["name"] for member in members} != set(RUST_DEPENDENCIES) or len(members) != 2:
        raise QualityError("Rust workspace must contain only the core and plugin crates")
    for member in members:
        name = member["name"]
        manifest = root / "rust" / name / "Cargo.toml"
        if Path(member["manifest_path"]).resolve() != manifest.resolve():
            raise QualityError(f"unexpected workspace member location: {name}")
        if member.get("links") or any(
            "custom-build" in target["kind"] for target in member["targets"]
        ):
            raise QualityError(f"{name}: native link declarations/build scripts belong to CMake")
        dependencies = member["dependencies"]
        if {dependency["name"] for dependency in dependencies} != RUST_DEPENDENCIES[name]:
            raise QualityError(f"{name}: direct dependencies differ from the architectural contract")
        for dependency in dependencies:
            identity = dependency["name"]
            if dependency["kind"] is not None:
                raise QualityError(f"{name}: unapproved {dependency['kind']} dependency: {identity}")
            if identity == "film-juicer-core":
                expected = root / "rust/film-juicer-core"
                if not dependency.get("path") or Path(dependency["path"]).resolve() != expected.resolve():
                    raise QualityError("plugin must depend on the workspace core by path")
            elif dependency.get("path") or dependency.get("source") != "registry+https://github.com/rust-lang/crates.io-index":
                raise QualityError(f"{name}: {identity} must retain its crates.io source")
        # Inspect actual direct resolutions too: a workspace patch can replace
        # a crates.io declaration with a local/native implementation.
        for edge in resolved[member["id"]]["deps"]:
            dependency = packages[edge["pkg"]]
            if dependency["name"] not in RUST_DEPENDENCIES[name]:
                raise QualityError(f"{name}: unexpected resolved dependency: {dependency['name']}")
            if dependency["name"] == "film-juicer-core":
                if Path(dependency["manifest_path"]).resolve() != (root / "rust/film-juicer-core/Cargo.toml").resolve():
                    raise QualityError("plugin resolves a different core package")
            elif dependency.get("source") != "registry+https://github.com/rust-lang/crates.io-index":
                raise QualityError(f"{name}: dependency source override: {dependency['name']}")


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

    runner.consume([root / "Cargo.toml", root / "Cargo.lock", root / "rust-toolchain.toml",
                    root / "rustfmt.toml", root / "clippy.toml", root / "rust/film-juicer-core/Cargo.toml",
                    root / "rust/film-juicer-plugin/Cargo.toml", root / "rust/film-juicer-core/src/lib.rs"])
    check_workspace_contract(root)
    metadata = runner.run(
        [cargo, "metadata", "--locked", "--offline", "--format-version", "1"],
        env=environment,
        label="rust-dependency-contract",
        display_output=False,
        stdout_only=True,
    )
    packages = json.loads(metadata)
    check_dependency_contract(root, packages)
    for package in packages["packages"]:
        if package.get("source") is None:
            directory = Path(package["manifest_path"]).parent
            try:
                directory.relative_to(root / "rust")
            except ValueError:
                continue
            runner.consume([directory / "Cargo.toml", *directory.rglob("*.rs")])
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
    profile = "release" if arguments.preset.endswith("-release") else "dev"
    profile_args = ["--release"] if profile == "release" else []
    selected_packages = getattr(arguments, "rust_packages", None) or ["film-juicer-core", "film-juicer-plugin"]
    development = getattr(arguments, "development", False)
    format_packages = [item for package in selected_packages for item in ("--package", package)] if development else ["--all"]
    runner.run([cargo, "fmt", *format_packages, "--", "--check"], env=environment, label="rustfmt")
    for package in selected_packages:
        features = ["--features", "test-support"] if package == "film-juicer-plugin" else []
        runner.run(
            [*common, "-p", package, "--all-targets", *profile_args, "--", "-D", "warnings"],
            env=environment,
            label=f"clippy-{package}-{profile}",
        )
        if features:
            runner.run(
                [*common, "-p", package, "--all-targets", *features, *profile_args,
                 "--", "-D", "warnings"],
                env=environment,
                label=f"clippy-{package}-test-support-{profile}",
            )

def split_compiler_command(command: str) -> tuple[str, ...]:
    if os.name != "nt":
        try:
            return tuple(shlex.split(command))
        except ValueError as exc:
            raise QualityError(f"invalid compiler command: {exc}") from exc
    # CMake uses Windows quoting, including attached -I"paths with spaces".
    # POSIX shlex would also consume the backslashes in Windows paths.
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    split = shell32.CommandLineToArgvW
    split.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    split.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    memory = split(command, ctypes.byref(count))
    if not memory:
        raise QualityError("cannot parse Windows compiler command")
    try:
        return tuple(memory[index] for index in range(count.value))
    finally:
        kernel32.LocalFree(memory)


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
        command = raw.get("command") or subprocess.list2cmdline(raw.get("arguments", []))
        if command:
            arguments = tuple(raw["arguments"]) if "arguments" in raw else split_compiler_command(command)
            entries.append(CompilationEntry(relative, command, Path(raw["directory"]), arguments))
    if not entries:
        raise QualityError(f"compilation database contains no owned entries: {path}")
    return entries


INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*(["<])([^">]+)[">]', re.MULTILINE)


@dataclass(frozen=True)
class IncludeSearch:
    quoted: tuple[Path, ...]
    ordinary: tuple[Path, ...]
    forced: tuple[str, ...]


def include_search(root: Path, entry: CompilationEntry, *, resolve=None) -> IncludeSearch:
    directory = entry.directory if entry.directory.is_absolute() else root / entry.directory
    arguments = list(entry.arguments or split_compiler_command(entry.command))
    quoted, ordinary, system, after, forced = [], [], [], [], []
    options = (("--include-path", ordinary), ("--system-include", system),
               ("--pre-include", forced), ("-isystem", system), ("-iquote", quoted),
               ("-idirafter", after), ("-include", forced), ("/external:I", system),
               ("/FI", forced), ("/I", ordinary), ("-I", ordinary))
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument.startswith("@"):
            raise QualityError(f"expand compiler response file before header selection: {argument}")
        for option, destination in options:
            if argument == option:
                index += 1
                if index == len(arguments):
                    raise QualityError(f"missing compiler include argument: {option}")
                destination.append(arguments[index])
                break
            if argument.startswith(option) and (option in {"-I", "/I", "/FI", "/external:I"}
                                                or argument.startswith(option + "=")):
                destination.extend(argument[len(option):].lstrip("=").split(",")
                                   if option.startswith("--") else [argument[len(option):].lstrip("=")])
                break
        index += 1
    ordinary.extend(value for value in os.environ.get("CPATH", "").split(os.pathsep) if value)
    system.extend(value for value in os.environ.get("INCLUDE" if os.name == "nt" else "CPLUS_INCLUDE_PATH", "").split(os.pathsep) if value)

    def paths(values):
        return tuple(resolve(directory / value) if resolve else (directory / value).resolve() for value in values)

    return IncludeSearch(paths(quoted), paths([*ordinary, *system, *after]), tuple(forced))


def included_paths(root: Path, entry: CompilationEntry,
                   includes: dict[str, tuple[tuple[str, str], ...]],
                   edges: dict[tuple[str, IncludeSearch], tuple[str, ...]], *,
                   search=None, resolve=None, is_file=None) -> set[str]:
    """Share parsed text; resolve edges separately for each ordered search context."""
    search = include_search(root, entry) if search is None else search
    pending = [entry.path]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        key = (current, search)
        if key in edges:
            pending.extend(edges[key])
            continue
        current_path = root / current
        children: list[str] = []
        if current not in includes:
            try:
                includes[current] = tuple(INCLUDE_PATTERN.findall(current_path.read_text(encoding="utf-8")))
            except (OSError, UnicodeDecodeError) as exc:
                raise QualityError(f"cannot read include input {current}: {exc}") from exc
        directives = includes[current]
        if current == entry.path:
            directives += tuple(('"', name) for name in search.forced)
        for delimiter, include in directives:
            directories = (*((current_path.parent, *search.quoted) if delimiter == '"' else ()), *search.ordinary)
            candidates = (directory / include for directory in directories)
            for candidate in candidates:
                if not (is_file(candidate) if is_file else candidate.is_file()):
                    continue
                try:
                    relative = (resolve(candidate) if resolve else candidate.resolve()).relative_to(root).as_posix()
                except ValueError:
                    break  # The first match shadows later project paths too.
                children.append(relative)
                break
        edges[key] = tuple(children)
        pending.extend(children)
    return visited


def validate_translation_units(selected_native: Sequence[str], entries: Sequence[CompilationEntry], policy: SourcePolicy) -> None:
    available = {entry.path for entry in entries}
    missing = [path for path in selected_native
               if PurePosixPath(path).suffix.lower() in policy.translation_extensions and path not in available]
    if missing:
        raise QualityError("selected translation unit is absent from the compilation database: " + ", ".join(missing))


def tidy_translation_units(
    root: Path,
    selected_native: Sequence[str],
    entries: Sequence[CompilationEntry],
    policy: SourcePolicy,
) -> list[str]:
    root = root.resolve()
    validate_translation_units(selected_native, entries, policy)
    selected: set[str] = set()
    headers = {path for path in selected_native if PurePosixPath(path).suffix.lower() in policy.header_extensions}
    owners: dict[str, set[str]] = {header: set() for header in headers}
    includes: dict[str, tuple[tuple[str, str], ...]] = {}
    edges: dict[tuple[str, IncludeSearch], tuple[str, ...]] = {}
    if headers:
        for entry in entries:
            for header in headers & included_paths(root, entry, includes, edges):
                owners[header].add(entry.path)
    for path in selected_native:
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in policy.translation_extensions:
            selected.add(path)
        elif suffix in policy.header_extensions:
            if not owners[path]:
                raise QualityError(f"no consuming translation unit found for selected header: {path}")
            selected.update(owners[path])
    return sorted(selected)


def cuda_frontend_arguments(root: Path, entry: CompilationEntry) -> tuple[Path, Path, list[str]]:
    """Translate the supported CMake/NVCC dialect; never discard unknown options."""
    arguments = entry.arguments or split_compiler_command(entry.command)
    if not arguments or Path(arguments[0]).name.lower() not in {"nvcc", "nvcc.exe"}:
        raise QualityError(f"CUDA entry must invoke NVCC directly: {entry.path}")
    nvcc = (entry.directory / arguments[0]).resolve()
    frontend: list[str] = []
    host: Path | None = None
    architectures: set[str] = set()
    language = False
    source = False
    exception_mode = False
    runtime = False
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        index += 1

        def operand() -> str:
            nonlocal index
            if index == len(arguments):
                raise QualityError(f"missing operand for {argument}: {entry.path}")
            value = arguments[index]
            index += 1
            return value

        if argument.startswith(("-D", "-U", "-I")):
            value = argument[2:] or operand()
            frontend.extend([argument[:2], value])
        elif argument == "-isystem":
            frontend.extend([argument, operand()])
        elif argument.startswith("-isystem="):
            frontend.extend(["-isystem", argument.partition("=")[2]])
        elif argument in {"-std=c++20", "--std=c++20"}:
            frontend.append("-std=c++20")
            language = True
        elif argument.startswith(("-ccbin=", "--compiler-bindir=")):
            host = (entry.directory / argument.partition("=")[2]).resolve()
        elif argument.startswith("--generate-code="):
            architectures.add(argument.partition("=")[2])
        elif argument.startswith("-Xcompiler="):
            # NVCC accepts comma-separated options and CMake's quoted groups.
            for flag in argument.partition("=")[2].replace(",", " ").split():
                if os.name == "nt":
                    option = flag.removeprefix("/").removeprefix("-")
                    if option == "EHsc":
                        exception_mode = True
                    elif option in {"MD", "MDd"}:
                        frontend.append("-fms-runtime-lib=" + ("dll_dbg" if option == "MDd" else "dll"))
                        runtime = True
                    elif option in {"O2", "Od"}:
                        frontend.append("-O2" if option == "O2" else "-O0")
                    elif option in {"Ob0", "Ob2", "Oi", "Gy", "Gw", "GF", "RTC1", "Z7", "FS"} or option.startswith("Fd"):
                        # Inlining, object layout, runtime instrumentation and PDB output only.
                        continue
                    else:
                        raise QualityError(f"unsupported CUDA host option {flag}: {entry.path}")
                elif flag in {"-fPIC", "-fPIE", "-fvisibility=hidden"}:
                    frontend.append(flag)
                else:
                    raise QualityError(f"unsupported CUDA host option {flag}: {entry.path}")
        elif argument == "-x":
            if operand() != "cu":
                raise QualityError(f"CUDA entry must use -x cu: {entry.path}")
        elif argument == "-c":
            source_path = (entry.directory / operand()).resolve()
            if source_path == (root / entry.path).resolve():
                source = True
            else:
                raise QualityError(f"CUDA source operand differs from database file: {entry.path}")
        elif argument == "-o":
            operand()
        elif argument in {"-g", "-O0", "-O1", "-O2", "-O3"}:
            frontend.append(argument)
        elif argument in {"-forward-unknown-to-host-compiler", "--cudart=hybrid", "-Xptxas=-O3"}:
            # Driver dispatch, runtime linking and assembler optimization do not configure the AST.
            continue
        else:
            raise QualityError(f"unsupported NVCC option {argument}: {entry.path}")
    if not language or not source or host is None:
        raise QualityError(f"CUDA entry requires C++20, a source operand and an explicit host compiler: {entry.path}")
    if architectures != {"arch=compute_75,code=[sm_75]", "arch=compute_75,code=[compute_75]"}:
        raise QualityError(f"CUDA entry requires the qualified sm_75/compute_75 architecture pair: {entry.path}")
    if os.name == "nt":
        if host.name.lower() != "cl.exe" or not exception_mode or not runtime:
            raise QualityError(f"CUDA Windows entry requires MSVC, /EHsc and /MD or /MDd: {entry.path}")
        frontend.extend(["-fms-extensions", "-fms-compatibility", "-fdelayed-template-parsing"])
    elif not re.fullmatch(r"(?:[\w-]+-)?g\+\+-13", host.name):
        raise QualityError(f"CUDA Linux entry requires the GCC 13 host compiler: {entry.path}")
    return nvcc, host, ["-xcuda", f"--cuda-path={nvcc.parent.parent}",
                        "--cuda-gpu-arch=sm_75", "-fexceptions", *frontend]


def prepare_cuda_overlay(root: Path, log_dir: Path) -> Path:
    """Keep Clang 22/CUDA 13.2 parsing adaptations out of product include paths."""
    overlay = log_dir / "cuda-include"
    (overlay / "openrand").mkdir(parents=True, exist_ok=True)
    (overlay / "texture_fetch_functions.h").write_text(
        "// CUDA 13 removed this header; Clang 22's runtime wrapper still includes it.\n"
        "#pragma once\n", encoding="utf-8",
    )
    vendor = root / "third_party/openrand/util.h"
    try:
        text = vendor.read_text(encoding="utf-8")
    except OSError as exc:
        raise QualityError(f"cannot read CUDA analysis input: {vendor}: {exc}") from exc
    condition = "#ifdef __CUDA_ARCH__\n"
    if text.count(condition) != 1:
        raise QualityError("OpenRAND CUDA attribute guard changed; review the Clang analysis adaptation")
    # Clang parses both host/device callables in the same pass. Retain the vendor's
    # entire implementation and license; only make its attributes visible in both passes.
    (overlay / "openrand/util.h").write_text(
        text.replace(condition, "#if defined(__CUDACC__)\n"), encoding="utf-8",
    )
    return overlay


def prepare_cuda_analysis(
    root: Path, runner: Runner, clang_tidy: str, entries: Sequence[CompilationEntry],
) -> list[AnalysisCommand]:
    overlay = prepare_cuda_overlay(root, runner.log_dir)
    runner.consume([root / "third_party/openrand/util.h"])
    qualified: dict[tuple[Path, Path], list[str]] = {}
    commands: list[AnalysisCommand] = []
    for index, entry in enumerate(entries, start=1):
        nvcc, host, frontend = cuda_frontend_arguments(root, entry)
        key = (nvcc, host)
        if key not in qualified:
            version = runner.run([str(nvcc), "--version"], label="cuda-nvcc-version")
            if not re.search(r"release 13\.2,", version):
                raise QualityError(f"CUDA analysis requires Toolkit 13.2: {nvcc}")
            toolkit = nvcc.parent.parent
            if not (toolkit / "include/curand_mtgp32_kernel.h").is_file():
                raise QualityError(f"Clang CUDA analysis requires the cuRAND development headers in {toolkit}")
            if os.name == "nt":
                active_tools = os.environ.get("VCToolsInstallDir", "")
                active_host = Path(active_tools) / "bin/Hostx64/x64/cl.exe"
                if not active_tools or active_host.resolve() != host:
                    raise QualityError("run CUDA quality in the VS developer environment used by the CMake preset")
                qualified[key] = []
            else:
                version = runner.run([str(host), "-dumpfullversion"], label="cuda-host-version")
                if not version.startswith("13."):
                    raise QualityError(f"CUDA analysis requires GCC 13: {host}")
                library = runner.run([str(host), "-print-file-name=libstdc++.so"], label="cuda-host-library")
                if not Path(library).is_file():
                    raise QualityError(f"cannot locate GCC 13's libstdc++: {library}")
                qualified[key] = [f"--gcc-install-dir={Path(library).parent}"]
        # _NV_RSQRT_SPECIFIER must exist before Clang's forced runtime wrapper
        # includes CUDA's math declarations. No diagnostics are disabled.
        frontend = [*qualified[key], "-D_NV_RSQRT_SPECIFIER=", "-I", str(overlay), *frontend]
        command = [clang_tidy, "--warnings-as-errors=*", f"--config-file={root / '.clang-tidy'}",
                   str(root / entry.path), "--", *frontend]
        commands.append(AnalysisCommand(
            command, entry.directory, f"clang-tidy-cuda-{index}-{Path(entry.path).name}",
            entry.path,
        ))
    return commands


def prepare_native_analysis(
    root: Path,
    runner: Runner,
    arguments: argparse.Namespace,
    selected_native: Sequence[str],
    policy: SourcePolicy,
) -> list[AnalysisCommand]:
    entries = compilation_entries(root, arguments.preset)
    runner.consume([root / "out/build" / arguments.preset / "compile_commands.json",
                    root / ".clang-format", root / ".clang-tidy"])
    validate_translation_units(selected_native, entries, policy)
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
    consumers = getattr(arguments, "compiler_consumers", None) or []
    validate_translation_units(consumers, entries, policy)
    translation_units = sorted(set(tidy_translation_units(root, selected_native, entries, policy)) | set(consumers))
    if not translation_units:
        raise QualityError("native selection has no translation unit for clang-tidy")
    includes, edges = {}, {}
    consumed = set()
    for entry in entries:
        if entry.path in translation_units:
            consumed.update(included_paths(root.resolve(), entry, includes, edges))
    runner.consume([root / path for path in sorted(consumed)])
    build_dir = root / "out/build" / arguments.preset
    commands: list[AnalysisCommand] = []
    for index, translation_unit in enumerate(translation_units, start=1):
        if PurePosixPath(translation_unit).suffix.lower() == ".cu":
            continue
        commands.append(AnalysisCommand(
            [
                clang_tidy,
                "-p",
                str(build_dir),
                "--warnings-as-errors=*",
                translation_unit,
            ],
            root,
            f"clang-tidy-{index}-{PurePosixPath(translation_unit).name}",
            translation_unit,
        ))
    cuda_entries = list(dict.fromkeys(
        entry for entry in entries
        if entry.path in translation_units and PurePosixPath(entry.path).suffix.lower() == ".cu"
    ))
    if cuda_entries:
        commands.extend(prepare_cuda_analysis(root, runner, clang_tidy, cuda_entries))
    return commands


def check_build_path(root: Path, preset: str) -> None:
    cache = root / "out/build" / preset / "CMakeCache.txt"
    if not cache.is_file():
        return
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
            configured = Path(line.split("=", 1)[1])
            if str(configured) != str(root):
                raise QualityError(f"use the configured checkout path {configured}; current path {root} changes build arguments")
            return


def quality_requirements(arguments: argparse.Namespace, selected: Sequence[str], policy: SourcePolicy):
    runner_changed = (
        getattr(arguments, "qualification", False)
        or any(path.startswith(QUALITY_RUNNER_PREFIXES) for path in selected)
        or any(path in NATIVE_POLICY_FILES for path in selected)
        or any(path.startswith(".github/workflows/") for path in selected)
    )
    rust_required = (
        runner_changed
        or any(path in RUST_BUILD_FILES for path in selected)
        or any(is_rust(path) for path in selected)
    )
    native_required = bool(getattr(arguments, "compiler_consumers", None)) or runner_changed or any(
        path in {".clang-format", ".clang-tidy"} or is_native(path, policy)
        for path in selected
    )
    expected = ["source-hygiene", "git-diff-check", "native-boundary-enforcement"]
    if rust_required:
        expected.append("rust-quality")
    if native_required:
        expected.append("native-quality")
    if not selected:
        expected.append("empty-selection")
    return expected, rust_required, native_required


def complete_check(runner: Runner, name: str) -> None:
    runner.complete(name)


def run_checks(root: Path, arguments: argparse.Namespace, policy: SourcePolicy,
               selected: Sequence[str], excluded: Sequence[str], runner: Runner) -> int:
    target = PRESET_TARGETS[arguments.preset]
    log_dir = runner.log_dir
    expected_checks, rust_required, native_required = quality_requirements(arguments, selected, policy)
    runner_changed = (
        getattr(arguments, "qualification", False)
        or any(path.startswith(QUALITY_RUNNER_PREFIXES) for path in selected)
        or any(path in NATIVE_POLICY_FILES for path in selected)
        or any(path.startswith(".github/workflows/") for path in selected)
    )
    runner.expected_checks = expected_checks
    runner.python_cases = ["quality-boundary"]

    print("Selected files:")
    if not selected:
        print("  none (valid base diff)")
    for path in selected:
        detail = " (generated Rust; checked with workspace)" if path in policy.generated_paths else ""
        print(f"  {path}{detail}")
    print("Excluded categories:")
    if excluded:
        for reason in excluded:
            print(f"  {reason}")
    else:
        print("  none")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Rust target: {target}")

    runner.consume([root / "scripts/check-quality.py", root / "scripts/source_file_policy.json",
                    root / "scripts/verify.py", root / "scripts/qualification.py",
                    root / "scripts/execution.py", root / "scripts/process_tree.py",
                    root / "scripts/build_dependencies.py",
                    root / ".clang-format", root / ".clang-tidy", root / "tests/ffi/test_native_boundary.py",
                    *(root / path for path in selected)])
    check_source_hygiene(root, selected, policy,
                         log_dir / (getattr(runner, "name_prefix", "") + "source-hygiene.log"))
    complete_check(runner, "source-hygiene")

    if arguments.base:
        diff_command = ["git", "diff", "--check", arguments.base, "--"]
    elif arguments.files:
        diff_command = ["git", "diff", "--check", "--", *selected]
    else:
        diff_command = ["git", "diff", "--check"]
    runner.run(diff_command, label="git-diff-check")
    complete_check(runner, "git-diff-check")

    # These inexpensive guards inspect their complete boundary, even for a
    # documentation-only selection or a newly introduced native helper/header.
    runner.run(
        [sys.executable, "-B", "scripts/verify.py", "python-tests", "--preset", arguments.preset,
         "--key", "quality-boundary", "--", sys.executable, "-m", "unittest", "discover", "-s", "tests/ffi",
         "-p", "test_native_boundary.py", "-v"],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "FJ_HOST_BUNDLE": str(log_dir),
             "JUICER_PRODUCT_BUILD": str(root / "out/build" / arguments.preset)},
        label="native-boundary-enforcement",
    )
    complete_check(runner, "native-boundary-enforcement")

    native_commands: list[AnalysisCommand] = []
    if native_required:
        native_files = [path for path in selected if is_native(path, policy)]
        if runner_changed or any(path in {".clang-format", ".clang-tidy"} for path in selected):
            native_files.extend(
                path for path in REPRESENTATIVE_NATIVE_FILES if (root / path).is_file()
            )
        native_files = sorted(set(native_files))
        if not native_files:
            raise QualityError("native policy changed without a representative native selection")
        native_commands = prepare_native_analysis(root, runner, arguments, native_files, policy)
        # Edited producers fail before the broader header-consumer analysis.
        runner.run_analysis([item for item in native_commands if item.source in native_files])

    if rust_required:
        check_rust(root, runner, arguments, target)
        complete_check(runner, "rust-quality")
    if native_required:
        runner.run_analysis([item for item in native_commands if item.source not in native_files])
        complete_check(runner, "native-quality")

    if not selected:
        print("No owned files selected: diff/boundary checks and result validation still apply.")
        complete_check(runner, "empty-selection")
    elif not rust_required and not native_required:
        print("Documentation/configuration-only selection: diff/boundary checks and review apply.")
    return 0


def create_run_directory(root: Path, preset: str) -> Path:
    runs = root / "out/validation" / preset / "results"
    runs.mkdir(parents=True, exist_ok=True)
    directory = runs / f"quality-{uuid.uuid4().hex}"
    directory.mkdir()
    return directory


def capture_candidate(root: Path):
    from verify import capture_native_source

    return capture_native_source(root, "scripts/check-quality.py", domain_tests=False)


def finish_quality_result(native_run, result, candidate, costs):
    from qualification import validate_quality_result
    from verify import finish_native_run

    return finish_native_run(native_run, result, candidate, costs, validate_quality_result)


def main() -> int:
    sys.dont_write_bytecode = True
    arguments = parse_arguments()
    root = Path(__file__).resolve().parent.parent
    if arguments.verifier_scope_output:
        report = verifier_scope(root, arguments.base, arguments.candidate, load_policy(root))
        arguments.verifier_scope_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.verifier_scope_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print("Integration required by scope." if report["required"] else "Integration not required by scope.")
        print("Report-only rollout: the complete tooling selection still runs.")
        return 0
    sys.path.insert(0, str(root / "scripts"))
    from execution import OutputLease
    with OutputLease(root, root / "out/build" / arguments.preset):
        return quality_main(root, arguments)


def quality_main(root, arguments):
    check_build_path(root, arguments.preset)
    policy = load_policy(root)
    selected, excluded = select_paths(root, arguments, policy)
    bundle = os.environ.get("FJ_HOST_BUNDLE")
    in_bundle = bundle is not None
    from execution import expire_diagnostics
    expiry = None if in_bundle else expire_diagnostics(root, arguments.preset)
    log_dir = Path(bundle) if in_bundle else create_run_directory(root, arguments.preset)
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "scripts"))
    from execution import NativeRun, write_json
    native_run = NativeRun(root, log_dir)
    runner = Runner(root, log_dir, jobs=arguments.jobs, native_run=native_run,
                    name_prefix="quality-" if in_bundle else "")
    runner.consume([root / "out/build" / arguments.preset / "CMakeCache.txt",
                    root / ".cargo/config", root / ".cargo/config.toml"])
    expected_checks, _, _ = quality_requirements(arguments, selected, policy)
    runner.expected_checks = expected_checks
    runner.python_cases = ["quality-boundary"]
    candidate = None
    candidate_patch = b""
    source_costs = {"source_capture_seconds": 0.0, "git_output_bytes": 0, "source_read_bytes": 0}
    failure = None
    if not in_bundle:
        try:
            candidate, candidate_patch, measured = capture_candidate(root)
            source_costs.update(measured)
        except Exception as exc:
            failure = exc
    print(f"Quality run: {log_dir}", flush=True)
    start = time.perf_counter()
    candidate_unchanged = None
    if failure is None:
        try:
            result = run_checks(root, arguments, policy, selected, excluded, runner)
            if result != 0:
                raise QualityError(f"quality checks returned {result}")
        except Exception as exc:
            failure = exc
    if candidate is not None:
        try:
            after, after_patch, measured = capture_candidate(root)
            for name, value in measured.items():
                source_costs[name] = source_costs.get(name, 0) + value
            candidate_unchanged = after == candidate and after_patch == candidate_patch
        except Exception as exc:
            candidate_unchanged = False
            if failure is None:
                failure = exc
        if not candidate_unchanged and failure is None:
            failure = QualityError("quality source candidate changed while checks ran")
    status = "failed" if failure else "passed"
    if status == "passed":
        print(f"Quality checks passed; logs: {log_dir}")
    commands = native_run.commands
    consumed_inputs = getattr(runner, "consumed_inputs", set())
    if not isinstance(consumed_inputs, set):
        consumed_inputs = set()
    completed_checks = getattr(runner, "completed_checks", [])
    if not isinstance(completed_checks, list):
        completed_checks = []
    expected_commands = getattr(runner, "expected_commands", [])
    if not isinstance(expected_commands, list):
        expected_commands = []
    python_cases = getattr(runner, "python_cases", ["quality-boundary"])
    if not isinstance(python_cases, list):
        python_cases = ["quality-boundary"]
    command_output_read_bytes = getattr(runner, "command_output_read_bytes", 0)
    if not isinstance(command_output_read_bytes, int):
        command_output_read_bytes = 0
    costs = {
        **source_costs,
        "seconds": time.perf_counter() - start,
        "command_output_bytes": sum(command.get("output_bytes", 0) + command.get("stderr_bytes", 0)
                                     for command in commands),
        "command_output_read_bytes": command_output_read_bytes,
        "bookkeeping_read_bytes": 0,
        "report_read_bytes": 0,
    }
    if expiry is not None:
        costs["expiry"] = expiry
        costs["report_read_bytes"] += expiry["read_bytes"]
    checks = {
        "schema": "native-quality-checks-v1",
        "status": status,
        "preset": arguments.preset,
        "selected": selected,
        "excluded": excluded,
        "mode": "development" if getattr(arguments, "development", False) else "qualification",
        "expected_checks": expected_checks,
        "completed_checks": completed_checks,
        "expected_commands": expected_commands,
        "commands": commands,
        "python_cases": python_cases,
        "consumed_inputs": sorted(consumed_inputs),
        "costs": costs,
    }
    if failure is not None:
        checks["error"] = f"{type(failure).__name__}: {failure}"
    write_json(log_dir / "quality.checks.json", checks)
    if in_bundle:
        if failure is not None:
            raise failure
        return 0

    if candidate is None:
        raise failure or QualityError("could not capture the standalone quality source candidate")
    (log_dir / "candidate.patch").write_bytes(candidate_patch)
    candidate_record = {**candidate, "preset": arguments.preset}
    result_payload = {
        "schema": "native-quality-v1",
        "status": status,
        "preset": arguments.preset,
        "candidate_head": candidate["head"],
        "candidate_unchanged": candidate_unchanged,
        "shipping_isolation": False,
        "commands": commands,
        "selected": selected,
        "excluded": excluded,
        "scope": "quality",
        "mode": "development" if getattr(arguments, "development", False) else "qualification",
        "members": {},
        "costs": costs,
    }
    # An unproved descendant keeps the owner blocked and cannot publish a pass.
    if all(command.get("drained", False) for command in commands):
        finished = finish_quality_result(native_run, result_payload, candidate_record, costs)
        if finished != 0 and failure is None:
            raise QualityError(result_payload.get("error", "quality result validation failed"))
    if failure is not None:
        raise failure
    return 0



if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
