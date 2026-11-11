"""Check staged resources, native cuFFT closures and Rust artifact links."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

from production_contract import (
    PRODUCTION_FORBIDDEN_STRINGS,
    PRODUCTION_FORBIDDEN_SYMBOLS,
    PRODUCTION_MODULE_EXPORTS,
    production_module_export_names,
)


EXPECTED_RUST_CONSUMER_VARIANTS = {
    "juicer": "production",
    "JuicerProductionPreparedRenderProbe": "none",
    "JuicerFilmRecipeTests": "test-support",
    "JuicerOfxContextTests": "production",
    "JuicerOfxInstanceTests": "production",
    "JuicerOfxActionTests": "production",
    "JuicerOfxRenderTests": "production",
    "JuicerOfxRenderControlTests": "test-support",
    "JuicerOfxRenderGpuTests": "production",
    "JuicerOfxRenderTraceTests": "test-support",
    "JuicerOfxCaptureTests": "test-support",
    "JuicerOfxInstanceControlTests": "test-support",
    "JuicerOfxContextControlTests": "test-support",
    "JuicerOfxModuleTests": "production",
    "JuicerOfxModuleControlTests": "test-support",
}
RUST_VARIANTS = {
    "production": ("production", ()),
    "test-support": ("test-support", ("test-support",)),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resource_paths(stage: Path) -> set[str]:
    return {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}


def resource_inputs(manifest: dict) -> tuple[Path, Path, tuple[str, ...]]:
    source = Path(manifest["resource_source"])
    stage = Path(manifest["resource_stage"])
    for path in (source, stage):
        if not path.is_absolute() or not path.is_dir():
            raise ValueError(f"resource directory is missing or not absolute: {path}")
    resources = manifest["resources"]
    if not isinstance(resources, list) or not resources or any(not isinstance(name, str) for name in resources):
        raise ValueError("resources must be a nonempty list of relative paths")
    if len(set(resources)) != len(resources):
        raise ValueError("resource membership contains duplicate paths")
    for name in resources:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            raise ValueError(f"resource path is not a canonical relative path: {name}")
    return source, stage, tuple(resources)


def inspect(tool: str, *arguments: str) -> str:
    result = subprocess.run(
        [tool, *arguments], capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}, check=False
    )
    if result.returncode:
        raise ValueError(f"{tool} failed for {arguments[-1]} (exit={result.returncode}): {result.stderr.strip()}")
    return result.stdout


def cufft_tools() -> dict[str, str]:
    tools = {}
    for name in ("nm", "readelf", "ldd"):
        tool = shutil.which(name)
        if not tool:
            raise ValueError(f"required Linux artifact inspection tool is missing: {name}")
        tools[name] = tool
    return tools


def check_cufft_provider(manifest: dict, tools: dict[str, str]):
    selected = Path(manifest["cufft"])
    if not selected.is_absolute() or not selected.is_file():
        raise ValueError(f"selected cuFFT provider is missing or not absolute: {selected}")
    provider = selected.resolve(strict=True)
    provider_hash = digest(provider)
    dynamic = inspect(tools["readelf"], "--dynamic", "--wide", str(provider))
    sonames = re.findall(r"\(SONAME\).*\[([^\]]+)\]", dynamic)
    if len(sonames) != 1 or not sonames[0].startswith("libcufft.so."):
        raise ValueError(f"selected cuFFT provider has no unique cuFFT SONAME: {provider}")
    soname = sonames[0]
    library = ctypes.CDLL(str(provider))
    get_version = library.cufftGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    status = get_version(ctypes.byref(version))
    if status != 0 or version.value <= 0:
        raise ValueError(f"cuFFT version query failed for {provider}: status={status} version={version.value}")
    return provider, provider_hash, soname, version.value, provider.stat()


def cufft_binary_paths(manifest: dict) -> tuple[tuple[str, str], ...]:
    binaries = []
    for group in ("static", "shared", "without_cufft"):
        names = manifest[group]
        if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
            raise ValueError(f"{group} must contain a nonempty list of binary paths")
        if len(set(names)) != len(names):
            raise ValueError(f"{group} contains duplicate binary paths")
        binaries.extend((group, name) for name in names)
    return tuple(binaries)


def check_cufft_binary(
    group: str, name: str, soname: str, provider: Path, tools: dict[str, str]
) -> tuple[Path, os.stat_result]:
    binary = Path(name)
    if not binary.is_absolute() or not binary.is_file():
        raise ValueError(f"{group} binary is missing or not absolute: {binary}")
    before = binary.stat()
    symbols = inspect(tools["nm"], "--defined-only", str(binary))
    defines_cufft = any(
        line.split() and line.split()[-1].split("@", 1)[0] == "cufftCreate" for line in symbols.splitlines()
    )
    dynamic = inspect(tools["readelf"], "--dynamic", "--wide", str(binary))
    needed = re.findall(r"\(NEEDED\).*\[([^\]]+)\]", dynamic)
    cufft_needed = [dependency for dependency in needed if dependency.startswith("libcufft")]
    if group == "static":
        if not defines_cufft or cufft_needed:
            raise ValueError(
                f"production static cuFFT closure differs: {binary}; defines cufftCreate={defines_cufft}, NEEDED={cufft_needed}"
            )
    elif group == "without_cufft":
        if defines_cufft or cufft_needed:
            raise ValueError(
                f"unneeded cuFFT closure is present: {binary}; defines cufftCreate={defines_cufft}, NEEDED={cufft_needed}"
            )
    else:
        if defines_cufft or cufft_needed != [soname]:
            raise ValueError(
                f"shared cuFFT closure differs: {binary}; defines cufftCreate={defines_cufft}, NEEDED={cufft_needed}, expected={soname}"
            )
        # Only the selected cuFFT provider is required here. GPU driver
        # dependencies of other objects may be absent in a host lane.
        result = subprocess.run(
            [tools["ldd"], str(binary)], capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}, check=False
        )
        resolved = re.findall(
            r"^\s*" + re.escape(soname) + r"\s+=>\s+(.+?)\s+\(0x[0-9a-fA-F]+\)\s*$", result.stdout, re.MULTILINE
        )
        if len(resolved) != 1 or Path(resolved[0]).resolve(strict=True) != provider:
            raise ValueError(
                f"cuFFT provider resolution differs for {binary}: expected={provider}; ldd exit={result.returncode}\n{result.stdout}{result.stderr}"
            )
    return binary, before


def cmake_cache_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith("//") or line.startswith("#") or ":" not in line or "=" not in line:
            continue
        key, value = line.split(":", 1)[0], line.split("=", 1)[1]
        values[key] = value
    return values


def target_link_commands(ninja: str, build: Path, consumers: dict[str, Path]) -> dict[str, str]:
    result = subprocess.run(
        [ninja, "-C", str(build), "-t", "commands", *consumers], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ValueError(f"Ninja command inspection failed for Rust consumers: {result.stderr.strip()}")

    commands = result.stdout.splitlines()
    link_scripts = []
    for command in commands:
        if "cmake_link_script" not in command:
            continue
        match = re.search(r"cmake_link_script\s+(.+?)\s+--verbose(?:=\d+)?(?:\s|$)", command)
        if match:
            script = Path(match.group(1).strip('"'))
            if not script.is_absolute():
                script = build / script
            if script.is_file():
                link_scripts.extend(line for line in script.read_text().splitlines() if line.strip())
    if link_scripts:
        commands.extend(link_scripts)

    commands_by_output: dict[str, set[str]] = {}
    for command in commands:
        normalized = command.replace("\\", "/")
        if not re.search(r"(?:clang(?:-cl|\+\+)?|g\+\+|c\+\+|ld(?:\.lld)?|link(?:\.exe)?)", normalized, re.I):
            continue
        response_files = re.findall(r"(?:^|[\s,])@\s*(?:\"([^\"]+)\"|([^\s,]+))", normalized)
        expanded = [normalized]
        for quoted, plain in response_files:
            path = Path(quoted or plain)
            if not path.is_absolute():
                path = build / path
            if path.is_file():
                expanded.append(path.read_text(errors="replace").replace("\\", "/"))
        full_command = " ".join(expanded)
        output_arguments = re.findall(
            r"(?:^|\s)-o(?:\s+|=)\s*(?:\"([^\"]+)\"|([^\s]+))"
            r"|(?:^|\s)/out:\s*(?:\"([^\"]+)\"|([^\s]+))",
            full_command,
            re.IGNORECASE,
        )
        outputs = {normalize_link_output(next(value for value in row if value)) for row in output_arguments}
        for output in outputs:
            commands_by_output.setdefault(output, set()).add(full_command)

    result = {}
    for target, binary in consumers.items():
        expected_outputs = {
            normalize_link_output(str(binary)),
            normalize_link_output(binary.relative_to(build).as_posix()),
        }
        selected = set().union(*(commands_by_output.get(output, set()) for output in expected_outputs))
        if len(selected) != 1:
            raise ValueError(f"expected one final host link command for {target} -> {binary}, found {len(selected)}")
        result[target] = next(iter(selected))
    return result


def normalize_link_output(path: str) -> str:
    normalized = path.replace("\\", "/")
    while "/./" in normalized:
        normalized = normalized.replace("/./", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.casefold() if os.name == "nt" else normalized


def archive_occurrences(command: str, archive: Path, build: Path) -> int:
    normalized = command.replace("\\", "/")
    full_path = str(archive).replace("\\", "/")
    relative_path = archive.relative_to(build).as_posix()
    spellings = sorted({full_path, relative_path}, key=len, reverse=True)
    pattern = r"(?<![A-Za-z0-9_./:\\-])(?:" + "|".join(re.escape(path) for path in spellings) + r")(?![A-Za-z0-9_.-])"
    return len(re.findall(pattern, normalized, re.IGNORECASE if os.name == "nt" else 0))


def defined_symbols(nm: str, archive: Path) -> set[str]:
    output = inspect(nm, "--defined-only", "--demangle", str(archive))
    symbols = set()
    for line in output.splitlines():
        match = re.match(r"^.*?\b[0-9A-Fa-f]+\s+[A-Za-z?]\s+(.+)$", line)
        if match:
            symbols.add(match.group(1).strip())
    return symbols


OPTIONS = None


class BuildArtifactContractTests(unittest.TestCase):
    def test_staged_resources_match_manifest(self):
        manifest_bytes = OPTIONS.manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        manifest_resources = manifest.get("resources", ())
        resource_names = (
            tuple(name for name in manifest_resources if isinstance(name, str))
            if isinstance(manifest_resources, list)
            else ()
        )
        self.subtest_cases = (
            "resource-manifest",
            "resource-membership-initial",
            *(f"resource:{index}:{name}" for index, name in enumerate(resource_names)),
            *(f"resource-final:{index}:{name}" for index, name in enumerate(resource_names)),
            "resource-membership-final",
            "resource-manifest-final",
        )
        source, stage, resources = resource_inputs(manifest)

        with self.subTest(case="resource-manifest"):
            self.assertIsInstance(manifest, dict)
            self.assertIn(manifest.get("platform"), {"Linux", "Windows"})
            self.assertEqual(
                set(manifest),
                {"resource_source", "resource_stage", "resources", "platform"}
                | {"rust_artifacts"}
                | ({"cufft", "static", "shared", "without_cufft"} if manifest["platform"] == "Linux" else set()),
            )

        expected_resources = set(resources)
        with self.subTest(case="resource-membership-initial"):
            actual_resources = resource_paths(stage)
            self.assertEqual(
                actual_resources,
                expected_resources,
                f"missing={sorted(expected_resources - actual_resources)} "
                f"extra={sorted(actual_resources - expected_resources)}",
            )

        identities: dict[str, tuple[Path, str]] = {}
        staged_bytes = 0
        for index, name in enumerate(resources):
            with self.subTest(case=f"resource:{index}:{name}"):
                original = source / name
                staged = stage / name
                original_hash = digest(original)
                staged_hash = digest(staged)
                self.assertEqual(
                    staged_hash, original_hash, f"staged resource differs from source: {staged} versus {original}"
                )
                identities[name + ":source"] = (original, original_hash)
                identities[name + ":stage"] = (staged, staged_hash)
                staged_bytes += staged.stat().st_size

        for index, name in enumerate(resources):
            with self.subTest(case=f"resource-final:{index}:{name}"):
                for suffix in ("source", "stage"):
                    path, expected_hash = identities[name + ":" + suffix]
                    self.assertEqual(
                        digest(path), expected_hash, f"resource changed during artifact validation: {path}"
                    )

        with self.subTest(case="resource-membership-final"):
            self.assertEqual(resource_paths(stage), expected_resources)
        with self.subTest(case="resource-manifest-final"):
            self.assertEqual(OPTIONS.manifest.read_bytes(), manifest_bytes)
        print(f"PASS staged resource contract: resources={len(resources)} staged_bytes={staged_bytes}")

    @unittest.skipIf(os.name == "nt", "native cuFFT closures are inspected on Linux")
    def test_selected_native_cufft_closures(self):
        manifest_bytes = OPTIONS.manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        self.assertEqual(manifest.get("platform"), "Linux")
        declared_binaries = tuple(
            (group, name)
            for group in ("static", "shared", "without_cufft")
            for name in (manifest.get(group, ()) if isinstance(manifest.get(group), list) else ())
            if isinstance(name, str)
        )
        case_names = tuple(f"cufft:{group}:{index}" for index, (group, _) in enumerate(declared_binaries))
        final_names = tuple(f"final:{group}:{index}" for index, (group, _) in enumerate(declared_binaries))
        self.subtest_cases = (
            "cufft-provider",
            *case_names,
            "cufft-provider-final",
            *final_names,
            "cufft-manifest-final",
        )
        binary_cases = cufft_binary_paths(manifest)

        tools = None
        provider = None
        provider_hash = None
        provider_soname = None
        provider_version = None
        provider_before = None
        provider_error = None
        with self.subTest(case="cufft-provider"):
            try:
                tools = cufft_tools()
                (provider, provider_hash, provider_soname, provider_version, provider_before) = check_cufft_provider(
                    manifest, tools
                )
            except Exception as error:
                provider_error = error
                raise

        binaries: dict[tuple[str, int], tuple[Path, os.stat_result]] = {}
        counts = {"static": 0, "shared": 0, "without_cufft": 0}
        for index, (group, name) in enumerate(binary_cases):
            with self.subTest(case=f"cufft:{group}:{index}"):
                if provider_error is not None:
                    self.fail(f"cannot inspect binary without a valid selected provider: {provider_error}")
                binary, before = check_cufft_binary(group, name, provider_soname, provider, tools)
                binaries[(group, index)] = (binary, before)
                counts[group] += 1

        with self.subTest(case="cufft-provider-final"):
            if provider_error is not None:
                self.fail(f"selected provider validation failed: {provider_error}")
            self.assertEqual(digest(provider), provider_hash)
            self.assertEqual(Path(manifest["cufft"]).resolve(strict=True), provider)
            after = provider.stat()
            self.assertEqual(
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                (
                    provider_before.st_dev,
                    provider_before.st_ino,
                    provider_before.st_size,
                    provider_before.st_mtime_ns,
                    provider_before.st_ctime_ns,
                ),
            )

        for index, (group, _) in enumerate(binary_cases):
            with self.subTest(case=f"final:{group}:{index}"):
                if (group, index) not in binaries:
                    self.fail("binary closure was not inspected")
                binary, before = binaries[(group, index)]
                after = binary.stat()
                self.assertEqual(
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns),
                )

        with self.subTest(case="cufft-manifest-final"):
            self.assertEqual(OPTIONS.manifest.read_bytes(), manifest_bytes)
        print(f"cuFFT provider={provider} sha256={provider_hash} version={provider_version}")
        print(
            f"cuFFT closures: static={counts['static']} shared={counts['shared']} without_cufft={counts['without_cufft']}"
        )
        print(f"PASS native cuFFT contract: binaries={len(binary_cases)}")

    def test_rust_artifacts_are_separate_at_final_links(self):
        manifest_bytes = OPTIONS.manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        rust_manifest_path = Path(manifest["rust_artifacts"])
        rust_manifest_bytes = rust_manifest_path.read_bytes()
        rust_manifest = json.loads(rust_manifest_bytes)
        consumers = rust_manifest.get("consumers", {})
        consumer_names = tuple(sorted(consumers)) if isinstance(consumers, dict) else ()
        self.subtest_cases = (
            "rust-artifact-manifest",
            "rust-variant:production",
            "rust-variant:test-support",
            "rust-native-archive",
            *(f"rust-consumer-role:{target}" for target in EXPECTED_RUST_CONSUMER_VARIANTS),
            "rust-consumer-binary:juicer",
            "rust-consumer-binary:JuicerProductionPreparedRenderProbe",
            "rust-final-link-discovery",
            *(f"rust-final-link:{target}" for target in consumer_names),
            "rust-production-symbols:archive",
            "rust-production-symbols:module",
            "rust-production-symbols:prepared-probe",
            "rust-production-exports:module",
            "rust-neutral-native-archive",
            *(
                f"rust-artifact-final:{name}"
                for name in dict.fromkeys(
                    (
                        "production.archive",
                        "production.feature_input",
                        "production.stamp",
                        "test-support.archive",
                        "test-support.feature_input",
                        "test-support.stamp",
                        "native_archive",
                        *consumer_names,
                    )
                )
            ),
            "rust-artifact-manifest-final",
        )

        with self.subTest(case="rust-artifact-manifest"):
            self.assertIsInstance(rust_manifest, dict)
            self.assertEqual(set(rust_manifest), {"schema", "build", "variants", "native_archive", "consumers"})
            self.assertEqual(rust_manifest.get("schema"), "rust-artifacts-v1")
            self.assertEqual(set(rust_manifest.get("variants", {})), set(RUST_VARIANTS))
            self.assertIsInstance(consumers, dict)
            for target, variant in EXPECTED_RUST_CONSUMER_VARIANTS.items():
                self.assertIn(target, consumers)
                self.assertEqual(consumers[target].get("variant"), variant)
            for target, variant in {
                "JuicerProductionOfxProbe": "none",
                "JuicerCudaInspectTests": "test-support",
                "JuicerCudaInspectGpuTests": "test-support",
                "JuicerCudaOwnerUnloadTests": "test-support",
            }.items():
                self.assertIn(target, consumers)
                self.assertEqual(consumers[target].get("variant"), variant)

        build = next((parent for parent in rust_manifest_path.parents if (parent / "CMakeCache.txt").is_file()), None)
        self.assertIsNotNone(build, f"Rust artifact manifest is outside a configured CMake tree: {rust_manifest_path}")
        self.assertEqual(Path(rust_manifest["build"]), build)
        cache = cmake_cache_values(build / "CMakeCache.txt")
        ninja = cache.get("CMAKE_MAKE_PROGRAM")
        if ninja and not Path(ninja).is_file():
            ninja = shutil.which(ninja)
        self.assertIsNotNone(ninja, "configured CMake cache has no native Ninja executable")
        if not ninja:
            self.fail("configured CMake cache has no native Ninja executable")
        self.assertTrue(Path(ninja).is_file(), f"configured Ninja executable is missing: {ninja}")
        nm = cache.get("CMAKE_NM") or shutil.which("llvm-nm") or shutil.which("nm")
        self.assertIsNotNone(nm, "configured native symbol inspector is unavailable")
        if not nm:
            self.fail("configured native symbol inspector is unavailable")

        variants = rust_manifest["variants"]
        artifact_paths: dict[str, dict[str, Path]] = {}
        for key, (directory_name, expected_features) in RUST_VARIANTS.items():
            with self.subTest(case=f"rust-variant:{key}"):
                row = variants[key]
                self.assertEqual(set(row), {"archive", "features", "feature_input", "stamp", "target_dir"})
                self.assertEqual(row["features"], list(expected_features))
                paths = {}
                for field in ("archive", "feature_input", "stamp", "target_dir"):
                    path = Path(row[field])
                    self.assertTrue(path.is_absolute(), f"{key}.{field} must be absolute: {path}")
                    paths[field] = path
                self.assertEqual(paths["target_dir"], build / "cargo" / directory_name)
                self.assertEqual(paths["archive"].parent.parent.parent, paths["target_dir"])
                self.assertEqual(paths["feature_input"], paths["target_dir"] / "features.txt")
                self.assertEqual(paths["stamp"], paths["target_dir"] / "build.stamp")
                for field in ("archive", "feature_input", "stamp"):
                    self.assertTrue(paths[field].is_file(), f"missing {key} {field}: {paths[field]}")
                feature_lines = dict(
                    line.split("=", 1) for line in paths["feature_input"].read_text().splitlines() if "=" in line
                )
                self.assertEqual(feature_lines.get("features"), "none" if not expected_features else "test-support")
                self.assertIn("target", feature_lines)
                self.assertIn("profile", feature_lines)
                self.assertIn("cargo", feature_lines)
                artifact_paths[key] = paths

        native_archive = Path(rust_manifest["native_archive"])
        with self.subTest(case="rust-native-archive"):
            self.assertTrue(native_archive.is_absolute())
            self.assertTrue(native_archive.is_file(), f"native archive is missing: {native_archive}")

        consumers = rust_manifest["consumers"]
        frozen_paths = {
            f"{key}.{field}": paths[field]
            for key, paths in artifact_paths.items()
            for field in ("archive", "feature_input", "stamp")
        }
        frozen_paths["native_archive"] = native_archive
        try:
            link_commands = target_link_commands(
                ninja,
                build,
                {target: Path(row["binary"]) for target, row in consumers.items()},
            )
            link_command_error = None
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            link_commands = {}
            link_command_error = error
        with self.subTest(case="rust-final-link-discovery"):
            if link_command_error is not None:
                self.fail(f"could not inspect final Rust consumer links: {link_command_error}")
        for target in consumer_names:
            row = consumers[target]
            expected_case = f"rust-final-link:{target}"
            with self.subTest(case=expected_case):
                self.assertEqual(set(row), {"binary", "variant"})
                variant = row["variant"]
                self.assertIn(variant, {"production", "test-support", "none"})
                binary = Path(row["binary"])
                self.assertTrue(binary.is_absolute())
                self.assertTrue(binary.is_file(), f"final consumer output is missing: {binary}")
                frozen_paths[target] = binary
                if target not in link_commands:
                    self.fail(f"no inspected final link command for {target}")
                command = link_commands[target]
                production_count = archive_occurrences(
                    command,
                    artifact_paths["production"]["archive"],
                    build,
                )
                test_count = archive_occurrences(
                    command,
                    artifact_paths["test-support"]["archive"],
                    build,
                )
                if variant == "production":
                    self.assertEqual((production_count, test_count), (1, 0), command)
                elif variant == "test-support":
                    self.assertEqual((production_count, test_count), (0, 1), command)
                else:
                    self.assertEqual((production_count, test_count), (0, 0), command)

        for target, expected_variant in EXPECTED_RUST_CONSUMER_VARIANTS.items():
            with self.subTest(case=f"rust-consumer-role:{target}"):
                self.assertEqual(consumers[target]["variant"], expected_variant)

        module = build / "juicer.ofx"
        prepared_probe = consumers["JuicerProductionPreparedRenderProbe"]["binary"]
        with self.subTest(case="rust-consumer-binary:juicer"):
            self.assertEqual(Path(consumers["juicer"]["binary"]), module)
        with self.subTest(case="rust-consumer-binary:JuicerProductionPreparedRenderProbe"):
            self.assertEqual(
                Path(prepared_probe),
                build / "tests" / "bin" / ("JuicerProductionPreparedRenderProbe" + (".exe" if os.name == "nt" else "")),
            )
        production_paths = (
            ("archive", artifact_paths["production"]["archive"]),
            ("module", module),
            ("prepared-probe", Path(prepared_probe)),
        )
        frozen_identities = {}
        for name, path in frozen_paths.items():
            before = path.stat()
            frozen_identities[name] = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
        for case, path in production_paths:
            with self.subTest(case=f"rust-production-symbols:{case}"):
                self.assertTrue(path.is_file(), f"production artifact is missing: {path}")
                symbols = inspect(nm, "--defined-only", "--demangle", str(path))
                match = PRODUCTION_FORBIDDEN_SYMBOLS.search(symbols)
                self.assertIsNone(match, f"forbidden production symbol in {path}: {match.group(0) if match else ''}")
                retired_legacy = r"\bfj_legacy_[A-Za-z0-9_]+\b"
                self.assertNotRegex(symbols, retired_legacy)
                self.assertRegex("fj_legacy_retired", retired_legacy)
                data = path.read_bytes()
                forbidden_string = next((value for value in PRODUCTION_FORBIDDEN_STRINGS if value in data), None)
                self.assertIsNone(forbidden_string, f"forbidden production string in {path}: {forbidden_string!r}")

        with self.subTest(case="rust-production-exports:module"):
            if os.name == "nt":
                readobj = Path(nm).with_name("llvm-readobj.exe")
                self.assertTrue(readobj.is_file(), f"configured LLVM export inspector is missing: {readobj}")
                exports = inspect(str(readobj), "--coff-exports", str(module))
            else:
                exports = inspect(nm, "--dynamic", "--defined-only", str(module))
            export_names = production_module_export_names(exports, os.name == "nt")
            self.assertEqual(sorted(export_names), sorted(PRODUCTION_MODULE_EXPORTS), export_names)

        with self.subTest(case="rust-neutral-native-archive"):
            production_exports = defined_symbols(nm, artifact_paths["production"]["archive"])
            support_exports = defined_symbols(nm, artifact_paths["test-support"]["archive"])
            native_exports = defined_symbols(nm, native_archive)
            legacy_boundary = {symbol for symbol in production_exports if symbol.startswith("fj_legacy_")}
            self.assertFalse(
                legacy_boundary, f"production archive defines retired legacy boundaries: {sorted(legacy_boundary)}"
            )
            production_boundary = {
                "fj_ofx_plugin",
                "fj_ofx_plugin_count",
                "fj_diffusion_components_prepare",
                "fj_dir_boundary_describe",
                "fj_dir_boundary_acquire",
                "fj_dir_boundary_view",
                "fj_dir_boundary_release",
            }
            actual_boundary = {symbol for symbol in production_exports if symbol.startswith("fj_")}
            self.assertEqual(
                actual_boundary,
                production_boundary,
                f"production boundary mismatch: missing={sorted(production_boundary - actual_boundary)} "
                f"unexpected={sorted(actual_boundary - production_boundary)}",
            )
            support_boundary = {symbol for symbol in support_exports if symbol.startswith("fj_test_")}
            self.assertTrue(support_boundary, "test-support archive has no defined fj_test_ boundary symbols")
            embedded_production = native_exports & production_boundary
            embedded_support = native_exports & support_boundary
            self.assertFalse(
                embedded_production or embedded_support,
                "neutral native archive defines Rust FFI boundary symbols: "
                f"production={sorted(embedded_production)[:12]} test-support={sorted(embedded_support)[:12]}",
            )

        with self.subTest(case="rust-artifact-manifest-final"):
            self.assertEqual(OPTIONS.manifest.read_bytes(), manifest_bytes)
            self.assertEqual(rust_manifest_path.read_bytes(), rust_manifest_bytes)

        for name, path in frozen_paths.items():
            with self.subTest(case=f"rust-artifact-final:{name}"):
                self.assertTrue(path.is_file(), f"artifact disappeared during inspection: {path}")
                after = path.stat()
                self.assertEqual(
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    frozen_identities[name],
                    f"artifact changed during Rust artifact inspection: {path}",
                )


BuildArtifactContractTests.test_staged_resources_match_manifest.required_subtests = True
BuildArtifactContractTests.test_selected_native_cufft_closures.required_subtests = True
BuildArtifactContractTests.test_rust_artifacts_are_separate_at_final_links.required_subtests = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    global OPTIONS
    OPTIONS = parser.parse_args()
    return unittest.main(argv=[sys.argv[0]], exit=False).result.wasSuccessful()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
