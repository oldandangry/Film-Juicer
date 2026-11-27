#!/usr/bin/env python3
"""Run independent source, compiler and package checks."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIRS = {"native", "src", "tests"}
TRANSLATIONS = {".c", ".cc", ".cpp", ".cxx", ".cu"}
HEADERS = {".h", ".hh", ".hpp", ".hxx", ".cuh", ".inc", ".inl", ".ixx"}
REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
DEPENDENCIES = {
    "film-juicer-core": {"libm", "serde", "serde_json", "triomphe"},
    "film-juicer-plugin": {"film-juicer-core"},
}
RUST_NOTICE_RELEASES = {
    "itoa": "1.0.18",
    "libm": "0.2.15",
    "memchr": "2.8.3",
    "proc-macro2": "1.0.107",
    "quote": "1.0.47",
    "serde": "1.0.229",
    "serde_core": "1.0.229",
    "serde_derive": "1.0.229",
    "serde_json": "1.0.151",
    "syn": "3.0.6",
    "triomphe": "0.1.17",
    "unicode-ident": "1.0.26",
    "zmij": "1.0.23",
}


def run(command, *, cwd=ROOT, capture=False):
    return subprocess.run(
        [str(argument) for argument in command],
        cwd=cwd,
        text=True,
        check=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def pinned(executable, version_argument, pattern):
    version = run([executable, version_argument], capture=True)
    if not re.search(pattern, version, re.MULTILINE):
        raise ValueError(f"unexpected {executable} version: {version.strip()}")
    return executable


def git_paths(*arguments):
    return {Path(path) for path in run(["git", *arguments, "-z"], capture=True).split("\0") if path}


def source_paths(base=None):
    if base is None:
        paths = git_paths("ls-files", "--cached", "--others", "--exclude-standard")
    else:
        paths = git_paths("diff", "--name-only", "--diff-filter=ACMRT", base)
        paths |= git_paths("ls-files", "--others", "--exclude-standard")
    return sorted(path for path in paths if (ROOT / path).is_file())


def native_paths(paths):
    return [
        path
        for path in paths
        if path.parts[0] in NATIVE_DIRS
        and path.suffix in TRANSLATIONS | HEADERS
        and not set(path.parts) & {"fixtures", "generated", "vendor", "vendored"}
    ]


def check_format(args):
    paths = [Path(p) for p in args.files] if args.files else source_paths(args.base)
    if args.files:
        paths = [path.resolve().relative_to(ROOT) for path in paths]
    paths = native_paths(paths)
    if paths:
        formatter = pinned(args.clang_format, "--version", r"clang-format version 22\.")
        flags = ["-i"] if args.fix else ["--dry-run", "--Werror"]
        # Limit command length on Windows.
        for start in range(0, len(paths), 100):
            run([formatter, "--style=file", *flags, *paths[start : start + 100]])


def check_hygiene(args):
    run(["git", "diff", "--check", args.base or "HEAD"])
    failed = []
    for path in native_paths(source_paths(args.base)):
        for line_number, line in enumerate((ROOT / path).read_text(encoding="utf-8").splitlines(), 1):
            if (
                line.rstrip(" \t") != line
                or re.search(r"^(<<<<<<< |=======$|>>>>>>> |\|\|\|\|\|\|\| )", line)
                or re.search(r"\b(?:std::endl|JUICER_TESTS|JUICER_BUILD_VALIDATION)\b", line)
                or (path.suffix in HEADERS and re.match(r"^\s*using\s+namespace\b", line))
            ):
                failed.append(f"{path}:{line_number}: {line}")
    if failed:
        raise ValueError("source hygiene violations:\n" + "\n".join(failed))
    shipping_json = [
        path
        for path in native_paths(source_paths())
        if path.parts[0] in {"src", "native"} and "nlohmann/json.hpp" in (ROOT / path).read_text(encoding="utf-8")
    ]
    if shipping_json:
        raise ValueError(f"shipping sources include the test-only nlohmann header: {shipping_json}")


def check_tooling(args):
    ruff = pinned(args.ruff, "--version", r"^ruff 0\.15\.22$")
    run([ruff, "check", "--no-cache"])
    run([ruff, "format", "--check", "--no-cache"])
    shellcheck = pinned(args.shellcheck, "--version", r"^version: 0\.11\.0$")
    actionlint = pinned(args.actionlint, "-version", r"^1\.7\.12$")
    scripts = [path for path in source_paths() if path.parts[0] in {"scripts", "tests"} and path.suffix == ".sh"]
    run([shellcheck, "--external-sources", "--source-path=SCRIPTDIR", *scripts])
    run([actionlint, "-shellcheck", shellcheck, "-pyflakes", ""])


def rust_notices():
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    external = [package for package in lock["package"] if "source" in package]
    releases = {package["name"]: package["version"] for package in external}
    if len(external) != len(releases) or releases != RUST_NOTICE_RELEASES:
        raise ValueError(f"Cargo.lock notice allowlist differs: {releases}")
    notice_dir = ROOT / "third_party/rust"
    provenance = (notice_dir / "PROVENANCE.md").read_text(encoding="utf-8")
    hashes = dict(re.findall(r"\|\s*\[[^]]+\]\(([^)]+)\)\s*\|\s*\x60([a-f0-9]{64})\x60", provenance))
    for package in external:
        name, version = package["name"], package["version"]
        if package["source"] != REGISTRY or package["checksum"] not in provenance:
            raise ValueError(f"invalid notice provenance for {name} {version}")
        if not any(path.startswith(f"{name}-{version}/") for path in hashes):
            raise ValueError(f"missing notice for {name} {version}")
    actual = {path.relative_to(notice_dir).as_posix() for path in notice_dir.rglob("*") if path.is_file()}
    if actual != set(hashes) | {"PROVENANCE.md"}:
        raise ValueError("Rust notice files differ from recorded provenance")
    for path, expected in hashes.items():
        if digest(notice_dir / path) != expected:
            raise ValueError(f"modified Rust notice: {path}")
    return actual


def check_rust(args):
    cargo = args.cargo
    rust_notices()
    toolchain = tomllib.loads((ROOT / "rust-toolchain.toml").read_text(encoding="utf-8"))["toolchain"]["channel"]
    pinned(cargo, "--version", rf"^cargo {re.escape(toolchain)} ")
    run([cargo, "fmt", "--all", "--", "--check"])
    metadata = json.loads(run([cargo, "metadata", "--locked", "--format-version", "1"], capture=True))
    packages = {package["id"]: package for package in metadata["packages"]}
    resolved = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    members = [packages[member] for member in metadata["workspace_members"]]
    if {package["name"] for package in members} != DEPENDENCIES.keys() or len(members) != len(DEPENDENCIES):
        raise ValueError("unexpected Cargo workspace membership")
    for member in members:
        name = member["name"]
        manifest = ROOT / "rust" / name / "Cargo.toml"
        if Path(member["manifest_path"]).resolve() != manifest or member["links"]:
            raise ValueError(f"unexpected layout or native linkage for {name}")
        if any("custom-build" in target["kind"] for target in member["targets"]):
            raise ValueError(f"{name} must leave native build ownership with CMake")
        if not tomllib.loads(manifest.read_text(encoding="utf-8")).get("lints", {}).get("workspace"):
            raise ValueError(f"{name} must inherit workspace lints")
        if {dependency["name"] for dependency in member["dependencies"]} != DEPENDENCIES[name]:
            raise ValueError(f"unexpected direct dependencies for {name}")
        for dependency in member["dependencies"]:
            source = None if dependency["name"] == "film-juicer-core" else REGISTRY
            if dependency["kind"] is not None or dependency["source"] != source:
                raise ValueError(f"unexpected dependency kind or source for {name}: {dependency['name']}")
        for edge in resolved[member["id"]]["deps"]:
            dependency = packages[edge["pkg"]]
            if dependency["name"] == "film-juicer-core":
                if Path(dependency["manifest_path"]).resolve() != ROOT / "rust/film-juicer-core/Cargo.toml":
                    raise ValueError("plugin resolves a non-workspace core")
            elif dependency["name"] not in DEPENDENCIES[name] or dependency["source"] != REGISTRY:
                raise ValueError(f"unexpected resolved dependency for {name}: {dependency['name']}")
    if "#![forbid(unsafe_code)]" not in (ROOT / "rust/film-juicer-core/src/lib.rs").read_text(encoding="utf-8"):
        raise ValueError("film-juicer-core must forbid unsafe code")
    command = [cargo, "clippy", "--workspace", "--locked", "--all-targets"]
    if args.release:
        command += ["--release"]
    run([*command, "--", "-D", "warnings"])


@dataclass(frozen=True)
class CompilationEntry:
    path: str
    directory: Path
    arguments: tuple[str, ...]


def split_command(command):
    if os.name != "nt":
        return tuple(shlex.split(command))
    # Windows CMake uses native quoting, including attached -I"paths with spaces".
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
        raise ValueError("cannot parse Windows compiler command")
    try:
        return tuple(memory[index] for index in range(count.value))
    finally:
        kernel32.LocalFree(memory)


def cuda_frontend_arguments(root: Path, entry: CompilationEntry) -> tuple[Path, Path, list[str]]:
    """Translate the supported CMake/NVCC dialect; never discard unknown options."""
    arguments = entry.arguments
    if not arguments or Path(arguments[0]).name.lower() not in {"nvcc", "nvcc.exe"}:
        raise ValueError(f"CUDA entry must invoke NVCC directly: {entry.path}")
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

        def operand(option: str = argument) -> str:
            nonlocal index
            if index == len(arguments):
                raise ValueError(f"missing operand for {option}: {entry.path}")
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
                    elif option in {"Ob0", "Ob2", "Oi", "Gy", "Gw", "GF", "RTC1", "Z7", "FS"} or option.startswith(
                        "Fd"
                    ):
                        # Inlining, object layout, runtime instrumentation and PDB output only.
                        continue
                    else:
                        raise ValueError(f"unsupported CUDA host option {flag}: {entry.path}")
                elif flag in {"-fPIC", "-fPIE", "-fvisibility=hidden"}:
                    frontend.append(flag)
                else:
                    raise ValueError(f"unsupported CUDA host option {flag}: {entry.path}")
        elif argument == "-x":
            if operand() != "cu":
                raise ValueError(f"CUDA entry must use -x cu: {entry.path}")
        elif argument == "-c":
            source_path = (entry.directory / operand()).resolve()
            if source_path == (root / entry.path).resolve():
                source = True
            else:
                raise ValueError(f"CUDA source operand differs from database file: {entry.path}")
        elif argument == "-o":
            operand()
        elif argument in {"-g", "-O0", "-O1", "-O2", "-O3"}:
            frontend.append(argument)
        elif argument in {"-forward-unknown-to-host-compiler", "--cudart=hybrid", "-Xptxas=-O3", "-lineinfo"}:
            # Driver dispatch, runtime linking, assembler optimization and line tables do not configure the AST.
            continue
        else:
            raise ValueError(f"unsupported NVCC option {argument}: {entry.path}")
    if not language or not source or host is None:
        raise ValueError(f"CUDA entry requires C++20, a source operand and an explicit host compiler: {entry.path}")
    if architectures != {"arch=compute_75,code=[sm_75]", "arch=compute_75,code=[compute_75]"}:
        raise ValueError(f"CUDA entry requires the qualified sm_75/compute_75 architecture pair: {entry.path}")
    if os.name == "nt":
        if host.name.lower() != "cl.exe" or not exception_mode or not runtime:
            raise ValueError(f"CUDA Windows entry requires MSVC, /EHsc and /MD or /MDd: {entry.path}")
        frontend.extend(["-fms-extensions", "-fms-compatibility", "-fdelayed-template-parsing"])
    elif not re.fullmatch(r"(?:[\w-]+-)?g\+\+-13", host.name):
        raise ValueError(f"CUDA Linux entry requires the GCC 13 host compiler: {entry.path}")
    return (
        nvcc,
        host,
        ["-xcuda", f"--cuda-path={nvcc.parent.parent}", "--cuda-gpu-arch=sm_75", "-fexceptions", *frontend],
    )


def prepare_cuda_overlay(root: Path, log_dir: Path) -> Path:
    """Keep Clang 22/CUDA 13.2 parsing adaptations out of product include paths."""
    overlay = log_dir / "cuda-include"
    (overlay / "openrand").mkdir(parents=True, exist_ok=True)
    (overlay / "texture_fetch_functions.h").write_text(
        "// CUDA 13 removed this header; Clang 22's runtime wrapper still includes it.\n#pragma once\n",
        encoding="utf-8",
    )
    vendor = root / "third_party/openrand/util.h"
    try:
        text = vendor.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read CUDA analysis input: {vendor}: {exc}") from exc
    condition = "#ifdef __CUDA_ARCH__\n"
    if text.count(condition) != 1:
        raise ValueError("OpenRAND CUDA attribute guard changed; review the Clang analysis adaptation")
    # Clang parses both host/device callables in the same pass. Retain the vendor's
    # entire implementation and license; only make its attributes visible in both passes.
    (overlay / "openrand/util.h").write_text(
        text.replace(condition, "#if defined(__CUDACC__)\n"),
        encoding="utf-8",
    )
    return overlay


def compilation_entries(build):
    raw_entries = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
    entries = []
    for raw in raw_entries:
        directory = Path(raw["directory"])
        path = (directory / raw["file"]).resolve()
        if not path.is_relative_to(ROOT) or not native_paths([path.relative_to(ROOT)]):
            continue
        arguments = tuple(raw["arguments"]) if "arguments" in raw else split_command(raw["command"])
        entries.append(CompilationEntry(path.relative_to(ROOT).as_posix(), directory, arguments))
    if not entries:
        raise ValueError(f"no Film-Juicer translation units in {build}")
    return entries


def check_tidy(args):
    build = args.build_dir.resolve()
    tidy = pinned(args.clang_tidy, "--version", r"(?:LLVM|clang-tidy) version 22\.")
    run([tidy, "--verify-config"])
    entries = compilation_entries(build)
    selected = {Path(path).resolve().relative_to(ROOT).as_posix() for path in args.files}
    missing = selected - {entry.path for entry in entries}
    if missing:
        raise ValueError(f"files absent from the configured compile database: {sorted(missing)}")
    entries = [entry for entry in entries if not selected or entry.path in selected]
    # clang-tidy reads every native compile variant for a named C or C++ file.
    native = sorted({entry.path for entry in entries if Path(entry.path).suffix != ".cu"})
    with TemporaryDirectory(prefix="juicer-clang-cuda-") as temporary:
        overlay = prepare_cuda_overlay(ROOT, Path(temporary))
        commands = [([tidy, "-p", build, "--warnings-as-errors=*", ROOT / path], ROOT) for path in native]
        for entry in dict.fromkeys(entry for entry in entries if Path(entry.path).suffix == ".cu"):
            _, host, arguments = cuda_frontend_arguments(ROOT, entry)
            if os.name != "nt":
                library = Path(run([host, "-print-file-name=libstdc++.so"], capture=True).strip())
                if not library.is_file():
                    raise ValueError(f"GCC runtime for Clang CUDA analysis is unavailable: {library}")
                arguments.insert(0, f"--gcc-install-dir={library.parent}")
            flags = ["-D_NV_RSQRT_SPECIFIER=", "-I", str(overlay), *arguments]
            commands.append(
                (
                    [
                        tidy,
                        "--warnings-as-errors=*",
                        f"--config-file={ROOT / '.clang-tidy'}",
                        ROOT / entry.path,
                        "--",
                        *flags,
                    ],
                    entry.directory,
                )
            )
        print(f"Analyzing {len(commands)} compile invocations", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as workers:

            def analyze(entry):
                command, directory = entry
                return subprocess.run([str(argument) for argument in command], cwd=directory, check=False).returncode

            failures = sum(code != 0 for code in workers.map(analyze, commands))
        if failures:
            raise ValueError(f"clang-tidy failed for {failures} of {len(commands)} compile invocations")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_bundle(args):
    stage = args.stage.resolve()
    platform = "Win64" if os.name == "nt" else "Linux-x86-64"
    bundle = stage / "juicer.ofx.bundle"
    contents = bundle / "Contents"
    module = contents / platform / "juicer.ofx"
    expected_exports = {"OfxGetNumberOfPlugins", "OfxGetPlugin"}
    if os.name == "nt":
        dumpbin = args.dumpbin
        raw_exports = run([dumpbin, "/nologo", "/exports", module], capture=True)
        exports = re.findall(
            r"^[ \t]+\d+[ \t]+[A-F0-9]+[ \t]+[A-F0-9]+[ \t]+(\w+)(?:[ \t]+=[ \t]+\w+)?[ \t]*$",
            raw_exports,
            re.MULTILINE,
        )
        raw_dependencies = run([dumpbin, "/nologo", "/dependents", module], capture=True)
        dependencies = set(re.findall(r"^\s+([\w.-]+\.dll)\s*$", raw_dependencies.lower(), re.MULTILINE))
        files = {path.name.lower() for path in module.parent.iterdir() if path.is_file()}
        required = {"cudart64_13.dll", "cufft64_12.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"}
        platform_libraries = {"kernel32.dll", "ntdll.dll", "bcryptprimitives.dll", "ucrtbase.dll"}
        unresolved = {name for name in dependencies - files - platform_libraries if not name.startswith("api-ms-win-")}
        if unresolved or not required <= files or "cufft64_12.dll" not in dependencies:
            raise ValueError(f"incorrect Win64 dependencies or runtime DLLs: {dependencies}; staged: {files}")
        for library in files - {module.name}:
            if library not in required and not library.startswith(("msvcp140_", "concrt140")):
                raise ValueError(f"unexpected Win64 runtime DLL: {library}")
        runtime_notice = "msvc-runtime/PROVENANCE.md"
    else:
        raw_exports = run(["nm", "-D", "--defined-only", "--format=posix", module], capture=True)
        exports = [line.split()[0] for line in raw_exports.splitlines()]
        raw_dependencies = run(["readelf", "-d", module], capture=True)
        dependencies = set(re.findall(r"\(NEEDED\).*\[([^]]+)\]", raw_dependencies))
        allowed = {"libgcc_s.so.1", "libm.so.6", "libc.so.6", "ld-linux-x86-64.so.2"}
        if not {"libm.so.6", "libc.so.6"} <= dependencies <= allowed:
            raise ValueError(f"incorrect Linux runtime dependencies: {dependencies}")
        if {path.name for path in module.parent.iterdir()} != {module.name}:
            raise ValueError("Linux module directory contains unexpected runtime files")
        runtime_notice = "gcc-runtime/PROVENANCE.md"
    if len(exports) != len(expected_exports) or set(exports) != expected_exports:
        raise ValueError(f"unexpected OFX exports: {exports}")

    resource_config = (ROOT / "cmake/JuicerResources.cmake").read_text(encoding="utf-8")
    resources = set(resource_config.split("set(JUICER_RUNTIME_RESOURCES", 1)[1].split(")", 1)[0].split())
    resource_root = contents / "Resources"
    staged_resources = {p.relative_to(resource_root).as_posix() for p in resource_root.rglob("*") if p.is_file()}
    if staged_resources != resources:
        raise ValueError(f"unexpected bundle resource membership: {resources ^ staged_resources}")
    for resource in resources:
        if digest(resource_root / resource) != digest(ROOT / "Resources" / resource):
            raise ValueError(f"bundle resource bytes differ: {resource}")

    legal = contents / "Legal"
    notices = {p.relative_to(legal).as_posix() for p in legal.rglob("*") if p.is_file()}
    required_notices = {
        "LICENSE",
        "CUDA/CUDA-LICENSE.txt",
        "CUDA/cuFFT-LICENSE.txt",
        "fotufilm/LICENSE.Fotufilm",
        "fotufilm/NOTICE.md",
        "fotufilm/ENGINE-NOTICE.md",
        "ofxs/OpenFX-LICENSE.txt",
        "openrand/LICENSE.MIT",
        "spektrafilm/SPEKTRAFILM_LICENSE.txt",
        runtime_notice,
        *(f"rust/{name}" for name in rust_notices()),
    }
    if not required_notices <= notices or any(name.startswith("nlohmann/") for name in notices):
        raise ValueError(f"unexpected license inventory: {sorted(notices)}")
    for name in notices:
        if re.search(rb"(?:[A-Za-z]:[/\\]Users[/\\]|/(?:home|mnt/[a-z]/Users)/)", (legal / name).read_bytes()):
            raise ValueError(f"personal path in legal notice: {name}")
        if name.startswith(("rust/", "openrand/")) and digest(legal / name) != digest(ROOT / "third_party" / name):
            raise ValueError(f"bundled notice differs from source: {name}")

    archive = stage / f"juicer-{platform}.tar.gz"
    files = {p.relative_to(stage).as_posix(): digest(p) for p in bundle.rglob("*") if p.is_file()}
    with tarfile.open(archive, mode="r:gz") as packaged:
        for member in packaged:
            if member.isdir():
                continue
            expected = files.pop(member.name, None)
            if not member.isfile() or expected is None:
                raise ValueError(f"unexpected archive member: {member.name}")
            with packaged.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"archive bytes differ from stage: {member.name}")
    if files:
        raise ValueError(f"archive misses staged files: {sorted(files)}")
    print(f"{module.name}: {len(exports)} exports, {len(dependencies)} dependencies, {len(resources)} resources")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)

    formatting = commands.add_parser("format", help="run pinned clang-format on native source")
    formatting.add_argument("--base")
    formatting.add_argument("--fix", action="store_true")
    formatting.add_argument("--clang-format", default=os.environ.get("JUICER_CLANG_FORMAT", "clang-format"))
    formatting.add_argument("files", nargs="*")
    formatting.set_defaults(check=check_format)

    hygiene = commands.add_parser("hygiene", help="check whitespace and native source hygiene")
    hygiene.add_argument("--base")
    hygiene.set_defaults(check=check_hygiene)

    rust = commands.add_parser("rust", help="check the locked workspace and all Rust targets")
    rust.add_argument("--cargo", default="cargo")
    rust.add_argument("--release", action="store_true")
    rust.set_defaults(check=check_rust)

    tidy = commands.add_parser("tidy", help="analyze the configured native compile database")
    tidy.add_argument("--build-dir", required=True, type=Path)
    tidy.add_argument("--clang-tidy", default=os.environ.get("JUICER_CLANG_TIDY", "clang-tidy"))
    tidy.add_argument("--jobs", type=int, default=2)
    tidy.add_argument("files", nargs="*", help="optional translation unit selection")
    tidy.set_defaults(check=check_tidy)

    tooling = commands.add_parser("tooling", help="run Ruff, ShellCheck, and actionlint")
    tooling.add_argument("--ruff", default="ruff")
    tooling.add_argument("--shellcheck", default="shellcheck")
    tooling.add_argument("--actionlint", default="actionlint")
    tooling.set_defaults(check=check_tooling)

    bundle = commands.add_parser("bundle", help="inspect a completed native stage and archive")
    bundle.add_argument("--stage", required=True, type=Path)
    bundle.add_argument("--dumpbin", default="dumpbin")
    bundle.set_defaults(check=check_bundle)

    args = parser.parse_args()
    try:
        args.check(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
