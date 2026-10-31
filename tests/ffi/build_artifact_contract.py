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


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resource_paths(stage: Path) -> set[str]:
    return {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}


def check_resources(manifest: dict) -> tuple[dict[Path, str], int]:
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
    expected = set(resources)
    actual = resource_paths(stage)
    if actual != expected:
        raise ValueError(f"staged resource membership differs at {stage}: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    identities = {}
    staged_bytes = 0
    for name in resources:
        original = source / name
        staged = stage / name
        original_hash = digest(original)
        staged_hash = digest(staged)
        if staged_hash != original_hash:
            raise ValueError(f"staged resource differs from source: {staged} versus {original}")
        identities[original] = original_hash
        identities[staged] = staged_hash
        staged_bytes += staged.stat().st_size
    return identities, staged_bytes


def inspect(tool: str, *arguments: str) -> str:
    result = subprocess.run([tool, *arguments], capture_output=True, text=True,
                            env={**os.environ, "LC_ALL": "C"}, check=False)
    if result.returncode:
        raise ValueError(f"{tool} failed for {arguments[-1]} (exit={result.returncode}): {result.stderr.strip()}")
    return result.stdout


def check_cufft(manifest: dict) -> tuple[Path, str, dict[Path, os.stat_result]]:
    tools = {}
    for name in ("nm", "readelf", "ldd"):
        tool = shutil.which(name)
        if not tool:
            raise ValueError(f"required Linux artifact inspection tool is missing: {name}")
        tools[name] = tool
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
    binaries = {}
    counts = {}
    for group in ("static", "shared", "without_cufft"):
        names = manifest[group]
        if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
            raise ValueError(f"{group} must contain a nonempty list of binary paths")
        if len(set(names)) != len(names):
            raise ValueError(f"{group} contains duplicate binary paths")
        counts[group] = len(names)
        for name in names:
            binary = Path(name)
            if not binary.is_absolute() or not binary.is_file():
                raise ValueError(f"{group} binary is missing or not absolute: {binary}")
            binaries[binary] = binary.stat()
            symbols = inspect(tools["nm"], "--defined-only", str(binary))
            defines_cufft = any(line.split() and line.split()[-1].split("@", 1)[0] == "cufftCreate"
                               for line in symbols.splitlines())
            dynamic = inspect(tools["readelf"], "--dynamic", "--wide", str(binary))
            needed = re.findall(r"\(NEEDED\).*\[([^\]]+)\]", dynamic)
            cufft_needed = [name for name in needed if name.startswith("libcufft")]
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
    library = ctypes.CDLL(str(provider))
    get_version = library.cufftGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    status = get_version(ctypes.byref(version))
    if status != 0 or version.value <= 0:
        raise ValueError(f"cuFFT version query failed for {provider}: status={status} version={version.value}")
    print(f"cuFFT provider={provider} sha256={provider_hash} version={version.value}")
    print(f"cuFFT closures: static={counts['static']} shared={counts['shared']} without_cufft={counts['without_cufft']}")
    return provider, provider_hash, binaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    manifest_bytes = arguments.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError(f"artifact manifest is not an object: {arguments.manifest}")
    platform = manifest.get("platform")
    if platform not in {"Linux", "Windows"}:
        raise ValueError(f"unsupported artifact platform: {platform}")
    keys = {"resource_source", "resource_stage", "resources", "platform"}
    if platform == "Linux":
        keys.update({"cufft", "static", "shared", "without_cufft"})
    if set(manifest) != keys:
        raise ValueError(f"artifact manifest fields differ: {arguments.manifest}")
    resources, staged_bytes = check_resources(manifest)
    if manifest["platform"] == "Linux":
        provider, provider_hash, binaries = check_cufft(manifest)
    for path, expected in resources.items():
        if digest(path) != expected:
            raise ValueError(f"resource changed during inspection: {path}")
    if resource_paths(Path(manifest["resource_stage"])) != set(manifest["resources"]):
        raise ValueError(f"staged resource membership changed during inspection: {manifest['resource_stage']}")
    if arguments.manifest.read_bytes() != manifest_bytes:
        raise ValueError(f"artifact manifest changed during inspection: {arguments.manifest}")
    if manifest["platform"] == "Linux":
        if digest(provider) != provider_hash or Path(manifest["cufft"]).resolve(strict=True) != provider:
            raise ValueError(f"selected cuFFT provider changed during inspection: {provider}")
        for binary, before in binaries.items():
            after = binary.stat()
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                    before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns):
                raise ValueError(f"binary changed during inspection: {binary}")
    print(f"PASS build artifact contract: resources={len(manifest['resources'])} staged_bytes={staged_bytes}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as error:
        print(f"FAIL build artifact contract: {error}", file=sys.stderr)
        raise SystemExit(1) from error
