#!/usr/bin/env python3
"""Scoped local development and explicit native checkpoint verification."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Collection can run with only this entry point and the pure reader present.
if __name__ == "__main__" and sys.argv[1:2] == ["collect-qualification"]:
    from qualification import collection_main
    raise SystemExit(collection_main(sys.argv[2:]))

from execution import OutputLease, UnresolvedScope, write_json, windows_vctip
from build_dependencies import header_consumers
from qualification import collect_bundles

OPTICS_FUNCTION = "optics::tests::matches_frozen_native_words_identities_and_failures"
ROOT = Path(__file__).resolve().parent.parent
SUPPORTED = {"src/ScanRoute.cpp", "src/ScanRoute.h", "rust/film-juicer-core/src/route.rs", "rust/film-juicer-core/src/color/output.rs"}
OWNERS = ("tests/route/development.json", "tests/ffi/output-color-development.json")


def rust_body_edit(root, name, revision):
    """Prove a narrow existing expression edit; uncertainty broadens package checks."""
    if name not in {"rust/film-juicer-core/src/route.rs", "rust/film-juicer-core/src/color/output.rs"}:
        return False
    try:
        original = subprocess.check_output(["git", "show", revision + ":" + name], cwd=root, stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        return False
    current = (root / name).read_text(encoding="utf-8")
    if name.endswith("/color/output.rs"):
        def normalize(text):
            start = text.find("const SOURCE: Literal = Literal {")
            end = text.find("\nfn subhashes(", start)
            if start < 0 or end < 0:
                return None
            literals = re.sub(r"(?<![A-Za-z0-9_])-?[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?", "FLOAT", text[start:end])
            return text[:start] + literals + text[end:]
        baseline = normalize(original)
        return baseline is not None and baseline == normalize(current)
    # These four existing match arms are inside resolve, never declarations.
    pattern = r"(\(CapturePolarity::(?:Negative|Positive), RouteSelection::(?:Direct|Print)\) => )ScanRoute::(?:NegativeDirect|NegativePrint|PositiveDirect|PositivePrint)(,)"
    before, count = re.subn(pattern, r"\1ROUTE\2", original)
    after, actual = re.subn(pattern, r"\1ROUTE\2", current)
    return count == actual == 4 and before == after


def select_consumers(root, build, selected, *, require_compiled=False, compilation=None):
    unknown = set(selected) - SUPPORTED
    if unknown:
        raise UnresolvedScope("unresolved development scope: " + ", ".join(sorted(unknown)))
    if not selected:
        raise UnresolvedScope("empty development selection")
    if any(not (root / name).is_file() for name in selected):
        raise UnresolvedScope("unresolved deleted or renamed development input")
    targets, tests, packages = set(), set(), set()
    compiler = {name for name in selected if name.endswith(".cpp")}
    body_only = {}
    for owner_path in OWNERS:
        owner = json.loads((root / owner_path).read_text(encoding="utf-8"))
        owned = set(owner.get("sources", [owner.get("source")])) & set(selected)
        if not owned:
            continue
        targets.add(owner["target"])
        tests.add(owner["test"])
        if any(name.endswith(".rs") for name in owned):
            packages.add("film-juicer-core")
            tests.add("Quality.RustBoundaries")
            body_only.update({name: rust_body_edit(root, name, owner["rust_interface_revision"]) for name in owned if name.endswith(".rs")})
    if "src/ScanRoute.h" in selected:
        objects, sources = header_consumers(root, build, ["src/ScanRoute.h"], require_compiled=require_compiled, compilation=compilation)
        targets.update(objects)
        compiler.update(sources)
    if any(not proven for proven in body_only.values()):
        packages.add("film-juicer-plugin")
    if not targets:
        raise UnresolvedScope("no explicit development owner")
    return {"files": sorted(selected), "compiler_consumers": sorted(compiler),
            "targets": sorted(targets), "tests": sorted(tests), "rust_packages": sorted(packages)}


def quality_command(preset, selection):
    command = [sys.executable, "-B", "scripts/check-quality.py", "--preset", preset,
               "--development", "--files", *selection["files"]]
    if selection["compiler_consumers"]:
        command += ["--compiler-consumers", *selection["compiler_consumers"]]
    if selection["rust_packages"]:
        command += ["--rust-packages", *selection["rust_packages"]]
    return command


def rust_tests(args):
    """Run one Rust test package from CMake-prepared nextest binaries."""
    from qualification import framework_terminals

    package = getattr(args, "package", "core")
    package_name = {"core": "film-juicer-core", "bridge": "film-juicer-plugin"}.get(package)
    if package_name is None:
        raise ValueError("Rust test package must be core or bridge")
    registration = "Rust.Core" if package == "core" else "Rust.Bridge"
    if os.environ.get("FJ_RUST_FILTER") or os.environ.get("FJ_OPTICS_BUNDLE"):
        raise ValueError("Rust function filtering requires an explicit supported selection; the pilot environment is retired")
    bundle_value = os.environ.get("FJ_HOST_BUNDLE")
    if bundle_value is None:
        from execution import NativeRun

        # Direct CTest remains supervised when the shared native runner is absent.
        directory = ROOT / "out/validation" / args.preset / "rust-tests" / (registration + "-" + uuid.uuid4().hex)
        with OutputLease(ROOT, ROOT / "out/build" / args.preset):
            directory.mkdir(parents=True)
            run = NativeRun(ROOT, directory)
            environment = dict(os.environ, FJ_HOST_BUNDLE=str(directory))
            try:
                command = [sys.executable, "-B", str(Path(__file__).absolute()), *sys.argv[1:]]
                log = run.command("rust-tests-" + package, command, env=environment)
                print(log.read_text(), end="")
            except BaseException:
                print("Rust test diagnostics retained:", directory, file=sys.stderr)
                raise
            else:
                shutil.rmtree(directory)  # NativeRun proved descendant drainage.
        return 0

    bundle = Path(bundle_value)
    discovery_path = bundle / f"{registration}.discovery.json"
    junit_path = bundle / f"{registration}.xml"
    execution_path = bundle / f"{registration}.execution.json"
    events_path = bundle / "rust.jsonl" if package == "core" else None
    config_path = bundle / f"{registration}.nextest.toml"

    # Cargo metadata supplies the test-target contract independently of the
    # nextest discovery it will later validate. This catches a missing lib or
    # integration-test binary without maintaining a second list of test names.
    cargo_bytes = args.cargo_metadata.read_bytes()
    binaries_bytes = args.binaries_metadata.read_bytes()
    bookkeeping_reads = len(cargo_bytes) + len(binaries_bytes)
    cargo_metadata = json.loads(cargo_bytes)
    binaries_metadata = json.loads(binaries_bytes)
    packages = {row["id"]: row for row in cargo_metadata["packages"]}
    selected_package = next((row for row in packages.values() if row["name"] == package_name), None)
    if selected_package is None:
        raise ValueError("Cargo metadata omits Rust test package: " + package_name)
    expected_binaries = {
        (selected_package["id"], target["name"])
        for target in selected_package["targets"]
        if target.get("test") and any(kind in target["kind"] for kind in ("test", "lib", "staticlib"))
    }
    actual_binaries = {
        (binary["package-id"], binary["binary-name"])
        for binary in binaries_metadata["rust-binaries"].values()
        if binary.get("package-id") == selected_package["id"]
    }
    missing_binaries = expected_binaries - actual_binaries
    unexpected_binaries = actual_binaries - expected_binaries
    if not expected_binaries or missing_binaries or unexpected_binaries:
        raise ValueError("Rust test binary contract differs from Cargo targets: "
                         + repr({"missing": sorted(missing_binaries), "unexpected": sorted(unexpected_binaries)}))

    source_root = Path(selected_package["manifest_path"]).parent
    doctest_sources = []
    scanned_source_count = 0
    scanned_source_bytes = 0
    for source_path in source_root.rglob("*.rs"):
        scanned_source_count += 1
        source_bytes = source_path.read_bytes()
        scanned_source_bytes += len(source_bytes)
        if "```" in source_bytes.decode("utf-8"):
            doctest_sources.append(str(source_path.relative_to(ROOT)))
    if doctest_sources:
        raise ValueError("Rust doctest source requires an explicit Cargo doc-test path: " + repr(doctest_sources))

    # Keep the source/build-independent reader's binary-membership evidence in
    # the same result bundle. Both Rust registrations share these exact inputs.
    for destination_name, payload in (
        ("rust-cargo.json", cargo_bytes),
        ("rust-binaries.json", binaries_bytes),
    ):
        destination = bundle / destination_name
        if destination.exists():
            previous = destination.read_bytes()
            bookkeeping_reads += len(previous)
            if previous != payload:
                raise ValueError("Rust metadata changed within one host result bundle: " + destination_name)
        else:
            destination.write_bytes(payload)

    config_bytes = args.config_file.read_bytes()
    bookkeeping_reads += len(config_bytes)
    config_path.write_text(
        config_bytes.decode("utf-8")
        + "\npath = " + json.dumps(str(junit_path)) + "\n"
    )
    common = [str(args.nextest), "nextest"]
    metadata = [
        "--config-file", str(config_path), "--user-config-file", "none", "--profile", "default",
        "--binaries-metadata", str(args.binaries_metadata),
        "--cargo-metadata", str(args.cargo_metadata),
    ]
    filter_expr = "package(=" + package_name + ")"
    metadata.extend(["-E", filter_expr])
    environment = dict(os.environ)
    environment.pop("FJ_HOST_BUNDLE", None)
    environment.pop("FJ_OPTICS_CASES", None)
    environment.pop("FJ_RUST_ROWS", None)
    environment["FJ_RUST_ROWS"] = str(bundle / (registration + ".rows.jsonl"))
    if events_path is not None:
        environment["FJ_OPTICS_CASES"] = str(events_path)
    commands = [
        [*common, "list", *metadata, "--message-format", "json"],
        [*common, "run", *metadata, "--test-threads", "1", "--retries", "0",
         "--no-fail-fast", "--no-tests", "fail"],
    ]
    records = []
    discovery_bytes = 0
    try:
        started = time.perf_counter()
        with discovery_path.open("xb") as output:
            completed = subprocess.run(commands[0], env=environment, stdout=output)
        records.append({"command": commands[0], "returncode": completed.returncode,
                        "seconds": time.perf_counter() - started})
        completed.check_returncode()
        discovery = discovery_path.read_bytes()
        discovery_bytes = len(discovery)
        discovery_document = json.loads(discovery)
        discovered_binaries = set(discovery_document["rust-suites"])
        binary_packages = {
            binary["binary-id"]: binary["package-id"]
            for binary in binaries_metadata["rust-binaries"].values()
        }
        metadata_binary_ids = set(binary_packages)
        if not metadata_binary_ids or discovered_binaries != metadata_binary_ids:
            raise ValueError("nextest discovery does not match prepared workspace binaries")
        expected_tests = set()
        filtered_tests = set()
        for binary_id, suite in discovery_document["rust-suites"].items():
            for test_name, case in suite["testcases"].items():
                key = (binary_id, test_name)
                match = case["filter-match"]["status"]
                if match == "matches":
                    if binary_packages.get(binary_id) != selected_package["id"]:
                        raise ValueError("nextest selected a test from the other Rust package: " + test_name)
                    if case["ignored"]:
                        raise ValueError("unexpected ignored Rust test: " + test_name)
                    expected_tests.add(key)
                else:
                    filtered_tests.add(key)
        if not expected_tests:
            raise ValueError("Rust nextest selection is empty")
        started = time.perf_counter()
        completed = subprocess.run(commands[1], env=environment)
        records.append({"command": commands[1], "returncode": completed.returncode,
                        "seconds": time.perf_counter() - started})
        if completed.returncode == 0:
            framework_terminals(
                junit_path.read_bytes(), expected_tests, nextest=True, filtered=filtered_tests
            )
            from qualification import rust_case_events
            contract_path = bundle / "host-contracts.json"
            contract_bytes = (contract_path if contract_path.exists() else ROOT / "tests/host-contracts.json").read_bytes()
            row_bytes = (bundle / (registration + ".rows.jsonl")).read_bytes()
            bookkeeping_reads += len(contract_bytes) + len(row_bytes)
            contract = json.loads(contract_bytes)
            rust_case_events(row_bytes, contract, expected_tests)
        return completed.returncode
    finally:
        write_json(execution_path, {
            "registration": registration,
            "package": package_name,
            "cargo": str(args.cargo),
            "features": "test-support",
            "required_binaries": sorted(expected_binaries),
            "discovered_binaries": sorted(discovered_binaries) if "discovered_binaries" in locals() else [],
            "doctests": {"status": "none", "source_files_scanned": scanned_source_count,
                         "source_read_bytes": scanned_source_bytes, "code_fence_sources": doctest_sources},
            "commands": records,
            "discovery_read_bytes": discovery_bytes,
            "bookkeeping_read_bytes": bookkeeping_reads + scanned_source_bytes + discovery_bytes
                + (junit_path.stat().st_size if junit_path.exists() else 0),
        })


def native_case_key(command):
    import hashlib
    return "host-" + hashlib.sha256(json.dumps(command, ensure_ascii=False).encode()).hexdigest()[:16]


def gtest_environment(environment):
    """The registered command owns selection, repetition and sharding."""
    return {k: v for k, v in environment.items() if not k.startswith("GTEST_") and k != "TESTBRIDGE_TEST_ONLY"}


def native_case(args):
    """CTest launcher; a contained child never builds or admits old evidence."""
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("native case command required")
    # GoogleTest's CMake post-build discovery is already inside the build owner.
    if "--gtest_list_tests" in command:
        return subprocess.run(command, env=gtest_environment(os.environ)).returncode
    if "FJ_HOST_BUNDLE" not in os.environ:
        from execution import NativeRun
        directory = ROOT / "out/validation" / args.preset / "native-host" / uuid.uuid4().hex
        with ExitStack() as ownership:
            ownership.enter_context(OutputLease(ROOT, ROOT / "out/build" / args.preset))
            if args.runtime == "gpu":
                ownership.enter_context(OutputLease(ROOT))
            directory.mkdir(parents=True)
            run = NativeRun(ROOT, directory)
            environment = dict(os.environ, FJ_HOST_BUNDLE=str(directory))
            try:
                log = run.command("native-case", [sys.executable, "-B", str(Path(__file__).absolute()),
                                  "native-case", "--preset", args.preset, "--protocol", args.protocol,
                                  "--runtime", args.runtime,
                                  "--", *command], env=environment)
                print(log.read_text(), end="")
            except BaseException:
                print("Native case diagnostics retained:", directory, file=sys.stderr)
                raise
            else:
                shutil.rmtree(directory)
        return 0
    bundle = Path(os.environ["FJ_HOST_BUNDLE"])
    key = native_case_key(command)
    environment = gtest_environment(os.environ)
    for name in ("FJ_HOST_CASES", "FJ_OPTICS_CASES"):
        environment.pop(name, None)
    records = []
    bookkeeping_reads = 0
    outcomes_valid = False
    contract_name = args.runtime + "-contracts.json"
    contract_path = bundle / contract_name
    contract_bytes = (contract_path if contract_path.exists() else ROOT / "tests" / contract_name).read_bytes()
    bookkeeping_reads += len(contract_bytes)
    contract = json.loads(contract_bytes)
    def invoke(arguments, *, output=None, env=environment):
        started = time.perf_counter()
        completed = subprocess.run(arguments, env=env, stdout=output)
        records.append({"command": arguments, "returncode": completed.returncode,
                        "seconds": time.perf_counter() - started})
        completed.check_returncode()
        return completed
    try:
        # Discover inside this already-contained CTest family, just as nextest
        # does for Rust. No extra supervisor is needed for each fixture listing.
        if args.protocol == "gtest":
            discovery = bundle / (key + ".discovery.json")
            unfiltered = [argument for argument in command if not argument.startswith("--gtest_filter=")]
            invoke([*unfiltered, "--gtest_list_tests"], env=dict(environment, GTEST_OUTPUT="json:" + str(bundle / (key + ".target.json"))))
            invoke([*command, "--gtest_list_tests"], env=dict(environment, GTEST_OUTPUT="json:" + str(discovery)))
        rows = args.protocol == "custom" or os.environ.get("FJ_HOST_ROWS") == "1"
        if rows:
            with (bundle / (key + ".list.jsonl")).open("xb") as output:
                invoke([*command, "--list-cases"], output=output)
        environment["FJ_HOST_CASES"] = str(bundle / (key + ".jsonl"))
        if args.protocol == "gtest":
            environment["GTEST_OUTPUT"] = "xml:" + str(bundle / (key + ".xml"))
        invoke(command)
        from qualification import case_events, framework_terminals, gtest_discovery, native_cohort_count, optics_keys
        def read(name):
            nonlocal bookkeeping_reads
            data = (bundle / (key + name)).read_bytes()
            bookkeeping_reads += len(data)
            return data
        if args.protocol == "gtest":
            excluded = {tuple(n.split(".", 1)) for n in contract.get("excluded_gtest", {})}
            discovered = gtest_discovery(read(".discovery.json"), excluded=excluded)
            selected = discovered - excluded
            if not selected:
                raise ValueError("empty native framework selection")
            if not selected <= gtest_discovery(read(".target.json"), excluded=excluded):
                raise ValueError("GoogleTest selection is outside its owning target")
            framework_terminals(read(".xml"), selected, filtered=discovered & excluded)
            cohorts = {s + "." + n: contract["gtest_cohorts"].get(s + "." + n, 0) for s, n in selected}
        else:
            name = os.environ.get("FJ_HOST_TEST")
            if not name:
                raise ValueError("custom host registration identity missing")
            cohorts = native_cohort_count(contract, name, args.preset)
        expected = set()
        if rows:
            expected = optics_keys([json.loads(line) for line in read(".list.jsonl").splitlines()])
            if not expected:
                raise ValueError("empty native host case listing")
        case_events(read(".jsonl"), expected, cohorts=cohorts)
        outcomes_valid = True
        return 0
    finally:
        write_json(bundle / (key + ".execution.json"), {"command": command,
                   "returncode": records[-1]["returncode"] if records else None,
                   "commands": records, "outcomes_valid": outcomes_valid, "bookkeeping_read_bytes": bookkeeping_reads})


def tool_versions(args):
    """Read independent versions inside one caller-owned process tree."""
    from concurrent.futures import ThreadPoolExecutor
    names = [name for name, _ in args.tool]
    if len(names) != len(set(names)) or set(names) - {"cmake", "ctest", "cargo", "rustc", "nextest", "cxx", "cuda"}:
        raise ValueError("invalid version probe selection")
    def probe(tool):
        name, executable = tool
        command = [executable, "-Vv" if name == "rustc" else "--version"]
        started = time.perf_counter()
        record = {"name": name, "command": command, "returncode": None}
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            record.update(returncode=completed.returncode, output=completed.stdout.strip())
        except OSError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - started
        return record
    # These are read-only probes. The enclosing NativeRun supervisor contains
    # all children until every probe finishes, including failures.
    with ThreadPoolExecutor(max_workers=min(5, len(names))) as workers:
        records = list(workers.map(probe, args.tool))
    print(json.dumps(records))
    return 0 if all(r["returncode"] == 0 for r in records) else 1


def capture_native_source(root, literal_name, *, gpu_witness=False, domain_tests=False):
    """Describe this working candidate using Git's native source identities.

    HEAD plus a binary working-tree patch reconstruct tracked inputs; new inputs
    retain their bytes. Runtime fixture bytes are compared separately by the run.
    No checkout copies, historical admission or installed-tool reads are involved.
    """
    import base64
    import hashlib
    import stat
    started = time.perf_counter()
    costs = {"git_output_bytes": 0, "source_read_bytes": 0}
    def git(*arguments):
        data = subprocess.check_output(["git", "-c", "core.fsmonitor=false", "-c", "core.checkStat=minimal", "-c", "core.trustctime=false", *arguments], cwd=root)
        costs["git_output_bytes"] += len(data)
        return data
    filemode = git("config", "--type=bool", "--default=true", "--get", "core.filemode").strip() == b"true"
    explicit = {"CMakeLists.txt", "CMakePresets.json", ".clang-tidy", ".clang-format", ".gitattributes", ".cargo/config",
                "tests/CMakeLists.txt", "tests/ToolingTests.cmake", "tests/ffi/prepared_sanitizers.cmake",
                "tests/host/CMakeLists.txt", "tests/host/CMakePresets.json",
                "tests/route/development.json", "tests/ffi/output-color-development.json",
                "tests/film/optics-fixtures/manifest.json", literal_name, "scripts/source_file_policy.json"}
    roots = ("src", "native", "rust", "scripts", "cmake", "third_party", ".config", ".cargo", ".github/workflows")
    if gpu_witness or domain_tests:
        roots += ("Resources",)
    if domain_tests:
        roots += ("tests",)
    policy = (root / "scripts/source_file_policy.json").read_bytes()
    costs["source_read_bytes"] += len(policy)
    code = set(json.loads(policy)["supportedExtensions"]) | {".rs", ".py", ".cmake", ".toml", ".lock", ".yml", ".yaml"}
    paths = [":(literal)" + name for name in [*roots, *sorted(explicit), "Cargo.toml", "Cargo.lock",
             "rust-toolchain.toml", "rustfmt.toml", "clippy.toml", "ruff.toml"]]
    # All public test code stays conservative: the selected executable links
    # support sources in other domains. Git avoids reopening their unchanged bytes.
    paths += [":(glob)tests/**/" + "*" + extension for extension in sorted(code)]
    paths += [":(glob)tests/**/CMakeLists.txt"]
    # Include directory-link entries in discovery even when their names have no
    # source suffix. No scientific payload bytes are opened by this Git listing.
    discovery = [":(literal)" + name for name in [*roots, "tests", *sorted(explicit), "Cargo.toml", "Cargo.lock",
                 "rust-toolchain.toml", "rustfmt.toml", "clippy.toml", "ruff.toml"]]
    entries = git("ls-files", "-v", "--stage", "--cached", "--others", "--exclude-standard", "-z", "--", *discovery)
    checked_dirs = set()
    def reject_links(name):
        path = root / name
        leaf = None
        while path != root:
            if path in checked_dirs:
                break
            try:
                info = path.lstat()
            except FileNotFoundError:
                path = path.parent
                continue
            if path == root / name:
                leaf = info
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("selected worktree source must be a regular file: " + str(name))
            if stat.S_ISDIR(info.st_mode):
                checked_dirs.add(path)
            path = path.parent
        return leaf
    inputs, new_files = {}, {}
    for entry in entries.decode().split("\0"):
        if not entry:
            continue
        tag, entry = entry[:1], entry[2:]
        if tag == "?":
            name, identity = entry, None
            info = reject_links(name)
        else:
            metadata, name = entry.split("\t", 1)
            mode, blob, stage = metadata.split()
            identity = {"git_blob": blob, "mode": mode}
        if (name not in explicit and Path(name).suffix not in code and Path(name).name != "CMakeLists.txt"
                and not ((gpu_witness or domain_tests) and name.startswith("Resources/"))
                and not (domain_tests and name.startswith("tests/"))):
            continue
        if tag != "?":
            reject_links(Path(name).parent)
        if tag != "?" and (tag != "H" or stage != "0" or mode not in {"100644", "100755"}):
            raise ValueError("selected source must have ordinary, merged Git tracking: " + name)
        if identity is None:
            path = root / name
            if info is None or not stat.S_ISREG(info.st_mode):
                raise ValueError("untracked source must be a regular file: " + name)
            data = path.read_bytes()
            costs["source_read_bytes"] += len(data)
            identity = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                        "mode": "100755" if filemode and info.st_mode & 0o111 else "100644"}
            new_files[name] = base64.b64encode(data).decode("ascii")
        inputs[name] = identity
    if not inputs:
        raise ValueError("empty spatial-optics source scope")
    # Obtain worktree modes and the reconstructible patch in one Git scan.
    # --raw --patch -z separates its records from the patch with an empty NUL record.
    diff = git("diff", "--raw", "--patch", "--binary", "--no-abbrev", "--no-renames",
               "--no-ext-diff", "--no-textconv", "-z", "HEAD", "--", *paths)
    patch = b""
    if diff:
        raw, patch = diff.split(b"\0\0", 1)
        changed = raw.split(b"\0")
        for position in range(0, len(changed), 2):
            mode = changed[position].split()[1]
            name = changed[position + 1].decode()
            if mode not in {b"000000", b"100644", b"100755"}:
                raise ValueError("selected worktree source must be a regular file: " + name)
            if mode == b"000000":
                reject_links(name)  # Git may report files under a linked directory as deleted.
    source = {"head": git("rev-parse", "HEAD").decode().strip(), "inputs": inputs,
              "new_files_base64": new_files,
              "git_filemode": filemode,
              "git_version": git("version").decode().strip(),
              "source_identity": "HEAD + index + scoped binary patch; Git minimal stat uses whole-second mtime/size and racy-entry content checks; new files retain bytes"}
    costs["source_capture_seconds"] = time.perf_counter() - started
    return source, patch, costs


def gpu_lock_probe(*, wait_for_owner=False):
    """An independent native contender must not borrow the owner's GPU token."""
    if "JUICER_OUTPUT_LEASES" in os.environ:
        raise ValueError("GPU contender must be independent of delegated leases")
    if wait_for_owner:
        deadline = time.monotonic() + 30
        while not (ROOT / "out/verification-locks/gpu/owner.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("no GPU owner appeared for the native contender")
            time.sleep(.02)
    try:
        with OutputLease(ROOT, timeout=0.1):
            raise ValueError("GPU contender unexpectedly acquired the physical-device lease")
    except TimeoutError:
        print(json.dumps({"physical_gpu": "excluded"}))
    return 0


def cache_values(build):
    path = build / "CMakeCache.txt"
    return {line.split(":", 1)[0]: line.partition("=")[2] for line in path.read_text().splitlines()
            if ":" in line and "=" in line and not line.startswith(("#", "//"))} if path.exists() else {}


def finish_native_run(run, result, candidate, costs, validate):
    """Publish one flat bundle after all commands and source checks complete."""
    import gzip
    import hashlib
    publication_started = time.perf_counter()
    if any(not command["drained"] for command in run.commands):
        write_json(run.directory / "incomplete.json", result)
        raise RuntimeError("native owner remains blocked; descendants unproved")
    data = (json.dumps(candidate, separators=(",", ":")) + "\n").encode()
    (run.directory / "candidate.json").write_bytes(data)
    costs["candidate_write_bytes"] = len(data)
    payloads = {}
    costs["worker_read_bytes"] = 0
    for path in sorted(run.directory.iterdir()):
        if path.suffix not in {".json", ".jsonl", ".xml", ".toml", ".patch"} or path.name.endswith((".process.json", ".gate.json")):
            continue
        if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("linked native result member")
        data = path.read_bytes()
        costs["report_read_bytes"] += len(data)
        payloads[path.name] = data
        result["members"][path.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if path.name.endswith(".execution.json"):
            costs["worker_read_bytes"] += json.loads(data).get("bookkeeping_read_bytes", 0)
    if result["status"] == "passed":
        started = time.perf_counter()
        try:
            validate(result, payloads)
        except (ValueError, KeyError, TypeError, OSError) as error:
            result.update(status="failed", error=str(error))
        costs["publication_validation_seconds"] = time.perf_counter() - started
    with gzip.open(run.directory / "diagnostics.gz", "wb") as output:
        for path in sorted(run.directory.iterdir()):
            if path.name.endswith((".log", ".process.json", ".gate.json")):
                data = path.read_bytes()
                costs["report_read_bytes"] += len(data)
                output.write(("\n--- " + path.name + " ---\n").encode())
                output.write(data)
                path.unlink()
    costs["mandatory_retained_bytes"] = sum(item["bytes"] for item in result["members"].values())
    costs["compressed_diagnostic_bytes"] = (run.directory / "diagnostics.gz").stat().st_size
    result["diagnostics"] = {"path": "diagnostics.gz", "optional": True}
    costs["publication_seconds"] = time.perf_counter() - publication_started
    if "elapsed_seconds" in costs:
        costs["elapsed_seconds"] += costs["publication_seconds"]
    run.publish(result)
    print(run.directory / "result.json")
    return 0 if result["status"] == "passed" else 1


def configure_command(cmake, preset, *, cuda_free=False, testing="ON"):
    command = [cmake, "--preset", preset + ("-host-tests" if cuda_free else "")]
    if cuda_free:
        command += ["-S", str(ROOT / "tests/host")]
    return [*command, "-DBUILD_TESTING=" + testing, "-DPython3_EXECUTABLE=" + sys.executable]


def check_owner(args):
    """Require the preset owner already held by this configuration's caller."""
    lease = OutputLease(ROOT, ROOT / "out/build" / args.preset, timeout=0)
    with lease:
        if lease.role not in {"delegated", "nested"}:
            raise ValueError("shared Cargo configuration requires verify.py prepare/development ownership")
    return 0


def native_tests(args):
    """Fresh CMake-selected native/Rust/Python execution under one build owner."""
    import xml.etree.ElementTree as ET
    from execution import NativeRun, name_component
    from qualification import membership, native_registration, python_registration, validate_tests_result
    if args.preset.startswith("windows") != (os.name == "nt"):
        raise ValueError("tests must execute on their native platform")
    started = time.perf_counter()
    cuda_free = getattr(args, "cuda_free", False)
    preset_build = ROOT / "out/build" / args.preset
    build = ((preset_build / "host-tests" if cuda_free else args.build_dir) or preset_build).absolute()
    directory = ROOT / "out/validation" / args.preset / "results" / name_component(args.run_id or uuid.uuid4().hex)
    costs = {"source_read_bytes": 0, "git_output_bytes": 0, "source_capture_seconds": 0.0, "report_read_bytes": 0}
    def capture():
        source, patch, measured = capture_native_source(ROOT, "tests/python-contracts.json", domain_tests=True)
        for key, value in measured.items():
            costs[key] += value
        return source, patch
    with OutputLease(ROOT, preset_build):
        directory.mkdir(parents=True, exist_ok=False)
        run = NativeRun(ROOT, directory)
        initial, patch = capture()
        candidate = {**initial, "preset": args.preset, "configuration": {},
                     "features": {},
                     "environment": {k: os.environ[k] for k in ("CARGO_HOME", "RUSTUP_HOME", "CUDACXX", "CUDAToolkit_ROOT", "CMAKE_BUILD_PARALLEL_LEVEL") if k in os.environ}}
        (directory / "candidate.patch").write_bytes(patch)
        contracts = {}
        for runtime in ("host", "gpu", "python"):
            data = (ROOT / "tests" / (runtime + "-contracts.json")).read_bytes()
            costs["source_read_bytes"] += len(data)
            contracts[runtime] = json.loads(data)
            (directory / (runtime + "-contracts.json")).write_bytes(data)
            (directory / (runtime + "-targets.json")).write_text("{}\n")
        (directory / "registration.json").write_text('{"tests":[]}\n')
        (directory / "target-discovery.json").write_text("{}\n")
        result = {"schema": "native-tests-v1", "status": "failed", "preset": args.preset,
                  "candidate_head": initial["head"], "shipping_isolation": False,
                  "scope": "checkpoint" if args.mode == "qualification" and not args.label and not args.tests else "focused",
                  "groups": args.selection or [], "runtime": args.runtime, "selection": [], "case_files": {},
                  "commands": run.commands, "members": {}, "optics_rows": [], "costs": costs, "build_kind": "product", "reason": args.reason,
                  "quality_files": sorted(args.files or [])}
        try:
            cmake = shutil.which("cmake")
            if not cmake:
                raise ValueError("qualified native CMake required")
            ctest = str(Path(cmake).with_name("ctest" + (".exe" if os.name == "nt" else "")))
            cache = cache_values(build)
            expected_home = ROOT / "tests/host" if cuda_free else ROOT
            home = Path(cache.get("CMAKE_HOME_DIRECTORY", expected_home))
            standalone = home.samefile(ROOT / "tests/tooling")
            if cache and not (home.samefile(expected_home) or standalone and not cuda_free):
                raise ValueError("build cache belongs to another source tree")
            if args.build_dir and not standalone and build != ROOT / "out/build" / args.preset:
                raise ValueError("explicit build directory must be the configured native tooling entry")
            result["build_kind"] = "cuda-free" if cuda_free else "tooling" if standalone else "product"
            configure = configure_command(cmake, args.preset, cuda_free=cuda_free)
            cargo = str(args.cargo or cache.get("JUICER_CARGO_EXECUTABLE") or shutil.which("cargo") or "")
            nextest = str(args.nextest or cache.get("JUICER_NEXTEST_EXECUTABLE") or shutil.which("cargo-nextest") or "")
            if not standalone:
                if not cargo or not nextest:
                    raise ValueError("qualified Rust/Cargo and cargo-nextest required")
                configure += ["-DJUICER_CARGO_EXECUTABLE=" + cargo, "-DJUICER_NEXTEST_EXECUTABLE=" + nextest]
                desired = {"BUILD_TESTING": "ON", "CMAKE_BUILD_TYPE": "Debug" if args.preset.endswith("debug") else "Release",
                           "JUICER_CARGO_EXECUTABLE": cargo, "JUICER_NEXTEST_EXECUTABLE": nextest, "Python3_EXECUTABLE": sys.executable}
                if any(os.path.normcase(os.path.normpath(cache.get(k, ""))) != os.path.normcase(os.path.normpath(v)) for k, v in desired.items()):
                    run.command("configure", configure, vctip_executable=windows_vctip(os.environ))
            build_command = [cmake, "--build", str(build)]
            run.command("configuration", [*build_command, "--target", "help"], vctip_executable=windows_vctip(os.environ))
            cache = cache_values(build)
            candidate["configuration"] = {k: cache.get(k) for k in ("BUILD_TESTING", "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER", "CMAKE_HOME_DIRECTORY")}
            if not standalone and not cuda_free:
                artifact_data = (build / "tests/rust-artifacts.json").read_bytes()
                costs["report_read_bytes"] += len(artifact_data)
                (directory / "rust-artifacts.json").write_bytes(artifact_data)
                candidate["features"]["rust_artifacts"] = "rust-artifacts.json"
                result["rust_artifacts"] = True
            data = run.command("registration", [ctest, "--test-dir", str(build), "--show-only=json-v1"], stdout_only=True).read_bytes()
            costs["report_read_bytes"] += len(data)
            (directory / "registration.json").write_bytes(data)
            inventory = json.loads(data)
            target_data = {}
            for runtime in ("host", "gpu", "python"):
                path = build / "tests" / (runtime + "-targets.json")
                data = path.read_bytes() if runtime == "python" or not standalone else b"{}"
                costs["report_read_bytes"] += len(data)
                (directory / (runtime + "-targets.json")).write_bytes(data)
                target_data[runtime] = json.loads(data)
            native = {} if standalone else {runtime: native_registration(contracts[runtime], target_data[runtime], inventory, runtime=runtime, cuda_free=cuda_free) for runtime in ("host", "gpu")}
            python = python_registration(contracts["python"], inventory, args.preset, standalone=standalone, cuda_free=cuda_free)
            entries = {**python, **{name: entry for domain in native.values() for name, entry in domain.items()}}
            available = set(entries) | (set() if standalone else {"Rust.Core", "Rust.Bridge"})
            if {test["name"] for test in inventory["tests"]} != available:
                raise ValueError("missing or unowned public CTest registration")
            prerequisites = {test["name"]: {p["name"]: p["value"] for p in test["properties"]} for test in inventory["tests"]}
            source_selection = None
            if args.files:
                source_selection = select_consumers(ROOT, build, args.files, require_compiled=False)
                selected = set(source_selection["tests"])
                if source_selection["rust_packages"]:
                    selected.add("Rust.Core")
                    if "film-juicer-plugin" in source_selection["rust_packages"]:
                        selected.add("Rust.Bridge")
            elif args.tests:
                selected = set(args.tests)
            elif args.targets:
                selected = {name for name, entry in entries.items() if entry.get("target") in args.targets}
            else:
                groups = membership(inventory, args.selection, args.runtime)
                selected = {name for names in groups.values() for name in names}
            if not selected <= available:
                raise ValueError("unknown explicit test selection")
            if args.tests and any(args.runtime != "all" and args.runtime not in prerequisites[name]["LABELS"] for name in selected):
                raise ValueError("explicit test selection disagrees with runtime")
            if args.label:
                selected = {name for name in selected if any(re.search(args.label, label) for label in prerequisites[name]["LABELS"])}
            selected = {name for name in selected if args.runtime == "all" or args.runtime in prerequisites.get(name, {}).get("LABELS", [])}
            if not selected or not selected <= available:
                raise ValueError("empty or unknown explicit test selection")
            result["selection"] = sorted(selected)
            result["case_files"] = {name: {k: entries[name][k] for k in ("key", "protocol", "rows")} for name in selected if name in entries}
            targets = set(args.targets or ())
            for name in selected:
                if name in native.get("host", {}) or name in native.get("gpu", {}):
                    targets.add(entries[name]["target"])
                if name in python:
                    targets.update(target_data["python"][name])
            if selected & {"Rust.Core", "Rust.Bridge"}:
                targets.add("juicer_rust_tests")
            if source_selection:
                targets.update(source_selection["targets"])
            if targets:
                run.command("build", [*build_command, "--target", *sorted(targets), "--", "-k0"], vctip_executable=windows_vctip(os.environ))
            versions = [sys.executable, "-B", str(Path(__file__).absolute()), "tool-versions", "--tool", "cmake", cmake, "--tool", "ctest", ctest]
            if not standalone:
                for name, executable in (("cargo", cargo), ("rustc", str(Path(cargo).with_name("rustc.exe" if os.name == "nt" else "rustc"))), ("nextest", nextest)):
                    versions += ["--tool", name, executable]
                for name, key in (("cxx", "CMAKE_CXX_COMPILER"), ("cuda", "CMAKE_CUDA_COMPILER")):
                    if name != "cuda" or not cuda_free:
                        versions += ["--tool", name, cache[key]]
            version_data = run.command("versions", versions, stdout_only=True).read_bytes()
            costs["report_read_bytes"] += len(version_data)
            candidate["version_commands"] = json.loads(version_data)
            candidate["versions"] = {item["name"]: item["output"] for item in candidate["version_commands"]}
            if not standalone and (not candidate["versions"]["cargo"].startswith("cargo 1.98.1 ") or not candidate["versions"]["rustc"].startswith("rustc 1.98.1 ") or not candidate["versions"]["nextest"].startswith("cargo-nextest 0.9.143 ")):
                raise ValueError("qualified Rust/Cargo 1.98.1 and nextest 0.9.143 required")
            if "Rust.Core" in selected:
                manifest = json.loads((ROOT / "tests/film/optics-fixtures/manifest.json").read_bytes())
                literal = ROOT / "tests/film/optics-fixtures" / manifest["presets"][args.preset]["literal"]
                result["optics_rows"] = [row["id"] for row in json.loads(literal.read_bytes())["cases"]]
            environment = dict(os.environ, FJ_HOST_BUNDLE=str(directory))
            for name in ("FJ_RUST_FILTER", "FJ_OPTICS_BUNDLE", "FJ_HOST_ROWS"):
                environment.pop(name, None)
            if source_selection:
                run.command("quality", quality_command(args.preset, source_selection), env=environment)
            combined = ET.Element("testsuites")
            for runtime in ("host", "gpu"):
                names = sorted(name for name in selected if runtime in prerequisites[name]["LABELS"])
                if not names:
                    continue
                with ExitStack() as ownership:
                    if runtime == "gpu":
                        ownership.enter_context(OutputLease(ROOT))
                        contender = dict(os.environ)
                        contender.pop("JUICER_OUTPUT_LEASES", None)
                        data = run.command("gpu-exclusion", [sys.executable, "-B", str(Path(__file__).absolute()), "gpu-lock-probe"], env=contender).read_bytes()
                        if json.loads(data) != {"physical_gpu": "excluded"}:
                            raise ValueError("physical GPU contender was admitted")
                        (directory / "gpu-exclusion.json").write_bytes(data)
                    env = dict(environment)
                    env["JUICER_OUTPUT_LEASES"] = os.environ["JUICER_OUTPUT_LEASES"]
                    junit = directory / ("ctest-" + runtime + ".xml")
                    timeout = 60 + sum(prerequisites[name]["TIMEOUT"] for name in names)
                    run.command("ctest-" + runtime, [ctest, "--test-dir", str(build), "--no-tests=error", "--output-on-failure",
                        "--output-junit", str(junit), "-j", "1", "-R", "^(" + "|".join(map(re.escape, names)) + ")$"], env=env, timeout=timeout)
                    tree = ET.fromstring(junit.read_bytes())
                    combined.extend(list(tree) if tree.tag == "testsuites" else [tree])
                    junit.unlink()
                    if runtime == "gpu":
                        result["physical_gpu"] = {"contender_excluded": True, "drained_while_locked": True}
            (directory / "ctest.xml").write_bytes(ET.tostring(combined, encoding="utf-8"))
            discovery = {}
            for name in selected:
                if name in entries and entries[name]["protocol"] == "gtest":
                    path = directory / (entries[name]["key"] + ".target.json")
                    data = path.read_bytes()
                    costs["report_read_bytes"] += len(data)
                    document = json.loads(data)
                    target = entries[name]["target"]
                    if target in discovery and discovery[target] != document:
                        raise ValueError("native target discovery changed")
                    discovery[target] = document
                    path.unlink()
            (directory / "target-discovery.json").write_text(json.dumps(discovery, separators=(",", ":")))
            result["status"] = "passed"
        except BaseException as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                final, final_patch = capture()
                result["candidate_unchanged"] = initial == final and patch == final_patch
                if not result["candidate_unchanged"]:
                    result.update(status="failed", error="candidate changed during native execution")
            except BaseException as error:
                result.update(status="failed", error=str(error))
            costs["elapsed_seconds"] = time.perf_counter() - started
            costs["measurement"] = "explicit source/Git and result bookkeeping; compiler/test/resource/OS I/O and transient rewrites unmeasured"
            return finish_native_run(run, result, candidate, costs, validate_tests_result)


def development(args):
    return native_tests(args)


def fresh_prepare(args):
    """Configure/build explicit targets without creating preparation admission."""
    from execution import NativeRun, name_component
    from qualification import validate_build_result
    if args.preset.startswith("windows") != (os.name == "nt"):
        raise ValueError("build preparation requires its native platform")
    started = time.perf_counter()
    cuda_free = getattr(args, "cuda_free", False)
    preset_build = ROOT / "out/build" / args.preset
    build = preset_build / "host-tests" if cuda_free else preset_build
    directory = ROOT / "out/validation" / args.preset / "results" / name_component(args.run_id or "build-" + uuid.uuid4().hex)
    with OutputLease(ROOT, preset_build):
        directory.mkdir(parents=True, exist_ok=False)
        run = NativeRun(ROOT, directory)
        source, patch, costs = capture_native_source(ROOT, "scripts/verify.py")
        costs["report_read_bytes"] = 0
        candidate = {**source, "preset": args.preset}
        result = {"schema": "native-build-v1", "status": "failed", "preset": args.preset, "scope": "build",
                  "candidate_head": source["head"], "shipping_isolation": False, "commands": run.commands, "members": {}, "costs": costs}
        (directory / "candidate.patch").write_bytes(patch)
        try:
            cmake = shutil.which("cmake")
            if not cmake:
                raise ValueError("qualified native CMake required")
            cache = cache_values(build)
            home = ROOT / "tests/host" if cuda_free else ROOT
            if cache and not Path(cache["CMAKE_HOME_DIRECTORY"]).samefile(home):
                raise ValueError("build cache belongs to another source tree")
            run.command("configure", configure_command(cmake, args.preset, cuda_free=cuda_free, testing=args.testing), vctip_executable=windows_vctip(os.environ))
            targets = args.targets or select_consumers(ROOT, build, args.files, require_compiled=False)["targets"]
            if not targets:
                raise ValueError("explicit preparation selected no build targets")
            run.command("build", [cmake, "--build", str(build), "--target", *targets, "--", "-k0"], vctip_executable=windows_vctip(os.environ))
            cache = cache_values(build)
            candidate["configuration"] = {key: cache.get(key) for key in
                ("BUILD_TESTING", "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER")}
            version_data = run.command("versions", [sys.executable, "-B", str(Path(__file__).absolute()),
                "tool-versions", "--tool", "cmake", cmake], stdout_only=True).read_bytes()
            costs["report_read_bytes"] += len(version_data)
            candidate["version_commands"] = json.loads(version_data)
            result.update(status="passed", targets=targets)
        except BaseException as error:
            result["error"] = str(error)
        final, final_patch, measured = capture_native_source(ROOT, "scripts/verify.py")
        for key, value in measured.items():
            costs[key] += value
        result["candidate_unchanged"] = source == final and patch == final_patch
        costs["elapsed_seconds"] = time.perf_counter() - started
        costs["measurement"] = "explicit source/Git and result bookkeeping; compiler/test/resource/OS I/O and transient rewrites unmeasured"
        return finish_native_run(run, result, candidate, costs, validate_build_result)


def python_tests(args):
    """Run an existing unittest entry point with independent function/row outcomes."""
    import contextlib
    import runpy
    import tempfile
    import unittest
    from execution import NativeRun
    from qualification import validate_python_result

    started = time.perf_counter()
    command = list(args.command)
    if command[:1] == ["--"]:
        command.pop(0)
    if not command or Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError("Python tests require the configured native interpreter")
    key = args.key or native_case_key(command)
    from execution import name_component
    name_component(key)
    parent = os.environ.get("FJ_HOST_BUNDLE")
    if not parent:
        build = ROOT / "out/build" / args.preset
        with OutputLease(ROOT, build):
            scratch = ROOT / "out/validation" / args.preset / "python-controls"
            scratch.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="case-", dir=scratch))
            run = NativeRun(ROOT, directory)
            environment = dict(os.environ, FJ_HOST_BUNDLE=str(directory))
            try:
                run.command("python-suite", [sys.executable, "-B", str(Path(__file__).absolute()),
                            "python-tests", "--preset", args.preset, "--key", key, "--", *command],
                            env=environment, timeout=1200)
            except BaseException:
                print("Python controls retained:", directory, file=sys.stderr)
                raise
            else:
                shutil.rmtree(directory)
                return 0

    directory = Path(parent)
    discovery_path = directory / (key + ".discovery.json")
    functions_path = directory / (key + ".functions.jsonl")
    rows_path = directory / (key + ".rows.jsonl")
    execution_path = directory / (key + ".execution.json")
    original_main, original_subtest = unittest.main, unittest.TestCase.subTest
    terminal, declared_rows, active_tests = {}, set(), set()
    main_started = False
    selected = None
    functions = rows = None
    outcome = {"command": command, "returncode": 1, "outcomes_valid": False}

    def emit(stream, record):
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def cases(suite):
        for item in suite:
            if isinstance(item, unittest.TestSuite):
                yield from cases(item)
            else:
                yield item

    class Result(unittest.TextTestResult):
        def startTest(self, test):
            super().startTest(test)
            active_tests.add(id(test))
            emit(functions, {"id": test.id(), "event": "start"})

        def addSuccess(self, test):
            super().addSuccess(test)
            terminal[test.id()] = {"outcome": "passed"}

        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            if isinstance(test, unittest.case._SubTest):
                emit(rows, {"caller": test.test_case.id() + "/subtests", "id": test.params["case"],
                            "event": "terminal", "outcome": "skipped"})
            else:
                terminal[test.id()] = {"outcome": "skipped", "reason": reason}

        def addSubTest(self, test, subtest, error):
            super().addSubTest(test, subtest, error)
            emit(rows, {"caller": test.id() + "/subtests", "id": subtest.params["case"],
                        "event": "terminal", "outcome": "passed" if error is None else "failed"})

        def stopTest(self, test):
            emit(functions, {"id": test.id(), "event": "terminal", **terminal.pop(test.id(), {"outcome": "failed"})})
            active_tests.discard(id(test))
            super().stopTest(test)

    class Runner(unittest.TextTestRunner):
        resultclass = Result

        def run(self, suite):
            nonlocal selected, functions, rows
            selected = []
            for test in cases(suite):
                method = getattr(test, test._testMethodName)
                skipped = bool(getattr(type(test), "__unittest_skip__", False) or getattr(method, "__unittest_skip__", False))
                reason = getattr(type(test), "__unittest_skip_why__", "") if getattr(type(test), "__unittest_skip__", False) else getattr(method, "__unittest_skip_why__", "")
                selected.append({"id": test.id(), "skip": skipped, "skip_reason": str(reason),
                                 "rows": bool(getattr(method, "required_subtests", False))})
            if not selected or len({test["id"] for test in selected}) != len(selected):
                raise ValueError("empty or duplicate Python function discovery")
            discovery_path.write_text(json.dumps({"functions": selected}, separators=(",", ":")))
            with functions_path.open("x", encoding="utf-8") as function_stream, rows_path.open("x", encoding="utf-8") as row_stream:
                functions, rows = function_stream, row_stream
                result = super().run(suite)
                outcome["returncode"] = 0 if result.wasSuccessful() else 1
                return result

    @contextlib.contextmanager
    def observed_subtest(test, msg=unittest.case._subtest_msg_sentinel, **parameters):
        if id(test) not in active_tests:
            with original_subtest(test, msg, **parameters):
                yield
            return
        method = getattr(test, test._testMethodName)
        case_id = parameters.get("case")
        if not getattr(method, "required_subtests", False) or not isinstance(case_id, str) or not case_id:
            raise ValueError("Python subtest needs declared owning membership and a stable case ID: " + test.id())
        if test.id() not in declared_rows:
            intended = getattr(test, "subtest_cases", ())
            if (not intended or len(intended) != len(set(intended))
                    or any(not isinstance(name, str) or not name for name in intended)):
                raise ValueError("invalid Python owning subtest table: " + test.id())
            emit(rows, {"caller": test.id() + "/subtests", "event": "membership", "ids": list(intended)})
            declared_rows.add(test.id())
        emit(rows, {"caller": test.id() + "/subtests", "id": case_id, "event": "start"})
        with original_subtest(test, msg, **parameters):
            yield

    def observed_main(*values, **options):
        nonlocal main_started
        if main_started:
            return original_main(*values, **options)
        main_started = True
        options["testRunner"] = Runner
        return original_main(*values, **options)

    try:
        arguments = command[1:]
        while arguments and arguments[0] in {"-B", "-u"}:
            arguments.pop(0)
        if "-f" in arguments or "--failfast" in arguments:
            raise ValueError("Python outcome collection requires no fail-fast")
        unittest.main = observed_main
        unittest.TestCase.subTest = observed_subtest
        previous = sys.argv
        if arguments[:2] == ["-m", "unittest"]:
            # Match Python's -m entry semantics; executing this verifier sets
            # sys.path[0] to scripts rather than the caller's working directory.
            sys.path.insert(0, str(Path.cwd()))
            sys.argv = ["unittest", *arguments[2:]]
            observed_main(module=None)
        else:
            script = Path(arguments[0]).resolve(strict=True)
            sys.path.insert(0, str(script.parent))
            sys.argv = [str(script), *arguments[1:]]
            runpy.run_path(str(script), run_name="__main__")
    except SystemExit as error:
        outcome["returncode"] = int(error.code or 0)
    finally:
        unittest.main = original_main
        unittest.TestCase.subTest = original_subtest
        if "previous" in locals():
            sys.argv = previous
        if outcome["returncode"] == 0:
            try:
                discovery_data, function_data, row_data = discovery_path.read_bytes(), functions_path.read_bytes(), rows_path.read_bytes()
                outcome["bookkeeping_read_bytes"] = len(discovery_data) + len(function_data) + len(row_data)
                outcome["counts"] = validate_python_result(json.loads(discovery_data), function_data, row_data)
                outcome["outcomes_valid"] = True
            except (ValueError, KeyError, OSError, TypeError) as error:
                outcome.update(returncode=1, error=str(error))
                print("Python outcomes invalid:", error, file=sys.stderr)
        outcome["seconds"] = time.perf_counter() - started
        execution_path.write_text(json.dumps(outcome, separators=(",", ":")))
    return outcome["returncode"]


def main():
    from qualification import PRESETS, GROUPS
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("development", "qualification"):
        command = modes.add_parser(mode, help="fresh native execution; no historical admission")
        command.add_argument("--preset", required=True, choices=PRESETS)
        command.add_argument("--runtime", choices=("host", "gpu", "all"), default="host")
        command.add_argument("--build-dir", type=Path)
        command.add_argument("--cargo", type=Path)
        command.add_argument("--nextest", type=Path)
        command.add_argument("--run-id")
        command.add_argument("--label")
        if mode == "development":
            command.add_argument("--cuda-free", action="store_true",
                                 help="select the shared CUDA-independent test configuration")
            selection = command.add_mutually_exclusive_group(required=True)
            selection.add_argument("--files", nargs="+")
            selection.add_argument("--tests", nargs="+")
            selection.add_argument("--targets", nargs="+")
            command.set_defaults(selection=None, reason="focused development")
        else:
            command.add_argument("--reason", required=True)
            command.add_argument("--selection", required=True, nargs="+", choices=sorted(GROUPS))
            command.set_defaults(files=None, tests=None, targets=None)
    prepare = modes.add_parser("prepare", help="direct native configure/build; no reusable preparation receipt")
    prepare.add_argument("--preset", required=True, choices=PRESETS)
    selected = prepare.add_mutually_exclusive_group(required=True)
    selected.add_argument("--targets", nargs="+")
    selected.add_argument("--files", nargs="+")
    prepare.add_argument("--testing", choices=("ON", "OFF"), default="ON")
    prepare.add_argument("--run-id")
    prepare.add_argument("--cuda-free", action="store_true",
                         help="prepare selected CUDA-independent test targets")
    owner = modes.add_parser("check-owner", help=argparse.SUPPRESS)
    owner.add_argument("--preset", required=True, choices=PRESETS)
    collection = modes.add_parser("collect-qualification", help="pure offline reading of named bundles")
    collection.add_argument("--bundles", type=Path, nargs="+", required=True)
    collection.add_argument("--output", type=Path, required=True)
    collection.add_argument("--selection", nargs="+", choices=sorted(GROUPS))
    collection.add_argument("--runtime", choices=("host", "gpu", "all"), default="host")
    collection.add_argument("--presets", nargs="+", choices=PRESETS)
    native = modes.add_parser("native-case", help=argparse.SUPPRESS)
    native.add_argument("--preset", required=True)
    native.add_argument("--protocol", choices=("gtest", "custom"), required=True)
    native.add_argument("--runtime", choices=("host", "gpu"), default="host")
    native.add_argument("command", nargs=argparse.REMAINDER)
    python = modes.add_parser("python-tests", help=argparse.SUPPRESS)
    python.add_argument("--preset", required=True)
    python.add_argument("--key")
    python.add_argument("--build-dir", type=Path)
    python.add_argument("command", nargs=argparse.REMAINDER)
    rust = modes.add_parser("rust-tests", help=argparse.SUPPRESS)
    rust.add_argument("--package", choices=("core", "bridge"), required=True)
    rust.add_argument("--preset", required=True, choices=PRESETS)
    for flag in ("cargo", "nextest", "binaries-metadata", "cargo-metadata", "config-file"):
        rust.add_argument("--" + flag, type=Path, required=True)
    versions = modes.add_parser("tool-versions", help=argparse.SUPPRESS)
    versions.add_argument("--tool", nargs=2, action="append", required=True)
    probe = modes.add_parser("gpu-lock-probe", help=argparse.SUPPRESS)
    probe.add_argument("--wait-for-owner", action="store_true")
    args = parser.parse_args()
    if getattr(args, "cuda_free", False):
        if getattr(args, "files", None) or getattr(args, "build_dir", None):
            parser.error("--cuda-free requires explicit --tests or --targets and its standard configuration directory")
        if getattr(args, "testing", "ON") != "ON":
            parser.error("--cuda-free requires BUILD_TESTING=ON")
    dispatch = {"prepare": fresh_prepare, "development": development, "qualification": native_tests,
                "collect-qualification": collect_bundles, "native-case": native_case,
                "python-tests": python_tests, "rust-tests": rust_tests, "tool-versions": tool_versions,
                "check-owner": check_owner}
    if args.mode == "gpu-lock-probe":
        return gpu_lock_probe(wait_for_owner=args.wait_for_owner)
    return dispatch[args.mode](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        print("Verification failed:", error, file=sys.stderr)
        raise SystemExit(1)
