"""Check staged runtime resources and the selected native cuFFT link closures."""

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
    result = subprocess.run([tool, *arguments], capture_output=True, text=True,
                            env={**os.environ, "LC_ALL": "C"}, check=False)
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


def check_cufft_binary(group: str, name: str, soname: str, provider: Path,
                       tools: dict[str, str]) -> tuple[Path, os.stat_result]:
    binary = Path(name)
    if not binary.is_absolute() or not binary.is_file():
        raise ValueError(f"{group} binary is missing or not absolute: {binary}")
    before = binary.stat()
    symbols = inspect(tools["nm"], "--defined-only", str(binary))
    defines_cufft = any(line.split() and line.split()[-1].split("@", 1)[0] == "cufftCreate"
                        for line in symbols.splitlines())
    dynamic = inspect(tools["readelf"], "--dynamic", "--wide", str(binary))
    needed = re.findall(r"\(NEEDED\).*\[([^\]]+)\]", dynamic)
    cufft_needed = [dependency for dependency in needed if dependency.startswith("libcufft")]
    if group == "static":
        if not defines_cufft or cufft_needed:
            raise ValueError(f"production static cuFFT closure differs: {binary}; defines cufftCreate={defines_cufft}, NEEDED={cufft_needed}")
    elif group == "without_cufft":
        if defines_cufft or cufft_needed:
            raise ValueError(f"unneeded cuFFT closure is present: {binary}; defines cufftCreate={defines_cufft}, NEEDED={cufft_needed}")
    else:
        if defines_cufft or cufft_needed != [soname]:
            raise ValueError(f"shared cuFFT closure differs: {binary}; defines cufftCreate={defines_cufft}, NEEDED={cufft_needed}, expected={soname}")
        # Only the selected cuFFT provider is required here. GPU driver
        # dependencies of other objects may be absent in a host lane.
        result = subprocess.run([tools["ldd"], str(binary)], capture_output=True, text=True,
                                env={**os.environ, "LC_ALL": "C"}, check=False)
        resolved = re.findall(r"^\s*" + re.escape(soname) + r"\s+=>\s+(.+?)\s+\(0x[0-9a-fA-F]+\)\s*$",
                              result.stdout, re.MULTILINE)
        if len(resolved) != 1 or Path(resolved[0]).resolve(strict=True) != provider:
            raise ValueError(f"cuFFT provider resolution differs for {binary}: expected={provider}; ldd exit={result.returncode}\n{result.stdout}{result.stderr}")
    return binary, before


OPTIONS = None


class BuildArtifactContractTests(unittest.TestCase):
    def test_staged_resources_match_manifest(self):
        manifest_bytes = OPTIONS.manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        manifest_resources = manifest.get("resources", ())
        resource_names = (tuple(name for name in manifest_resources if isinstance(name, str))
                          if isinstance(manifest_resources, list) else ())
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
                | ({"cufft", "static", "shared", "without_cufft"}
                   if manifest["platform"] == "Linux" else set()),
            )

        expected_resources = set(resources)
        with self.subTest(case="resource-membership-initial"):
            actual_resources = resource_paths(stage)
            self.assertEqual(
                actual_resources, expected_resources,
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
                self.assertEqual(staged_hash, original_hash,
                                 f"staged resource differs from source: {staged} versus {original}")
                identities[name + ":source"] = (original, original_hash)
                identities[name + ":stage"] = (staged, staged_hash)
                staged_bytes += staged.stat().st_size

        for index, name in enumerate(resources):
            with self.subTest(case=f"resource-final:{index}:{name}"):
                for suffix in ("source", "stage"):
                    path, expected_hash = identities[name + ":" + suffix]
                    self.assertEqual(digest(path), expected_hash,
                                     f"resource changed during artifact validation: {path}")

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
                (provider, provider_hash, provider_soname,
                 provider_version, provider_before) = check_cufft_provider(manifest, tools)
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
                (provider_before.st_dev, provider_before.st_ino, provider_before.st_size,
                 provider_before.st_mtime_ns, provider_before.st_ctime_ns),
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
        print(f"cuFFT closures: static={counts['static']} shared={counts['shared']} without_cufft={counts['without_cufft']}")
        print(f"PASS native cuFFT contract: binaries={len(binary_cases)}")


BuildArtifactContractTests.test_staged_resources_match_manifest.required_subtests = True
BuildArtifactContractTests.test_selected_native_cufft_closures.required_subtests = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    global OPTIONS
    OPTIONS = parser.parse_args()
    return unittest.main(argv=[sys.argv[0]], exit=False).result.wasSuccessful()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
