#!/usr/bin/env python3
"""Author frozen Film v1 banks with the product profile evaluator and pinned Fotufilm."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOTUFILM = "87a3f46af093a9e6bf76acf21aca83d3b9e2c600"
ADAPTERS = ROOT / "tests/grain"
CALIBRATION_FIELDS = {
    "strength",
    "size_mm",
    "layer_weights",
    "base_density_offset",
    "fog_density",
    "read_density",
    "radius_span",
    "sublayer_shares",
    "coating_density_scale",
    "provenance",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(args):
    version = subprocess.check_output([args.swift, "--version"], text=True)
    if not version.startswith("Swift version 6.1.2 (") or "x86_64-unknown-linux-gnu" not in version:
        raise ValueError("expected Swift 6.1.2 for x86_64 Linux")
    revision = subprocess.check_output(["git", "-C", args.fotufilm, "rev-parse", "HEAD"], text=True).strip()
    if revision != FOTUFILM:
        raise ValueError(f"Fotufilm checkout must be at {FOTUFILM}")
    output = args.output.absolute()
    output.mkdir()
    # Export the verified Git tree. Local changes and previous builds cannot
    # affect the pinned source, and build products stay in the explicit output.
    archive = output / "source.tar"
    subprocess.run(["git", "-C", args.fotufilm, "archive", "-o", archive, revision], check=True)
    shutil.unpack_archive(archive, output)
    archive.unlink()

    main = output / "Sources/graincapture"
    main.mkdir()
    reference = ADAPTERS / "capture_film_reference.swift"
    probe = ADAPTERS / "capture_film_probe.swift"
    shutil.copyfile(reference, main / "main.swift")
    shutil.copyfile(probe, output / "Sources/FotufilmCore/CaptureFilmProbe.swift")
    package = output / "Package.swift"
    original = package.read_text(encoding="utf-8")
    marker = "    targets: [\n"
    if original.count(marker) != 1:
        raise ValueError("Fotufilm Package.swift targets marker has changed")
    target = '        .executableTarget(name: "graincapture", dependencies: ["FotufilmCore"]),\n'
    package.write_text(original.replace(marker, marker + target), encoding="utf-8")
    subprocess.run(
        [args.swift, "build", "-c", "release", "--product", "graincapture", "-j", str(args.jobs)],
        cwd=output,
        env={**os.environ, "FOTUFILM_DISABLE_HALIDE": "1"},
        check=True,
    )
    write_json(
        output / "capture.json",
        {
            "fotufilm_revision": revision,
            "swift_version": version.splitlines()[0],
            "capture_sha256": digest(output / ".build/release/graincapture"),
            "reference_adapter_sha256": digest(reference),
            "probe_adapter_sha256": digest(probe),
            "coating_seed": "0xA11C40575EED0001",
            "measurement_seed": "0x5A3F1E1D",
            "halide_enabled": False,
        },
    )


def response(resources, stock):
    return json.loads(
        subprocess.check_output(
            [
                "cargo",
                "run",
                "--quiet",
                "--locked",
                "--release",
                "-p",
                "film-juicer-core",
                "--example",
                "export_grain_response",
                "--",
                resources.resolve(),
                stock,
            ],
            cwd=ROOT,
        )
    )


def read_calibration(path):
    values = json.loads(path.read_bytes())
    if not isinstance(values, dict):
        raise ValueError("expected a calibration JSON object")
    missing = CALIBRATION_FIELDS - values.keys()
    if missing:
        raise ValueError(f"missing calibration inputs: {', '.join(sorted(missing))}")
    extra = values.keys() - CALIBRATION_FIELDS
    if extra:
        raise ValueError(f"unknown calibration inputs: {', '.join(sorted(extra))}")

    def number(value, minimum):
        return type(value) in (int, float) and math.isfinite(value) and value >= minimum

    for key in ("strength", "size_mm", "radius_span", "coating_density_scale"):
        if not number(values[key], 0) or values[key] == 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("fog_density", "base_density_offset"):
        if not number(values[key], 0):
            raise ValueError(f"{key} must be finite and nonnegative")
    for key, minimum, length in (("layer_weights", 0, 3), ("sublayer_shares", 0, 4)):
        vector = values[key]
        if not isinstance(vector, list) or len(vector) != length:
            raise ValueError(f"{key} must have {length} entries")
        if any(not number(value, minimum) or value == 0 for value in vector):
            raise ValueError(f"{key} must have finite positive entries")
    if not math.isclose(sum(values["sublayer_shares"]), 1.0, abs_tol=1e-6):
        raise ValueError("sublayer_shares must sum to 1")
    if values["read_density"] not in ("net", "gross"):
        raise ValueError("read_density must be net or gross")
    provenance = values["provenance"]
    if (
        not isinstance(provenance, dict)
        or provenance.get("basis") not in ("measured", "fitted", "authored", "trial")
        or not isinstance(provenance.get("source"), str)
        or not provenance["source"].strip()
    ):
        raise ValueError("provenance requires basis (measured, fitted, authored, or trial) and source")
    return values


def generate(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    output = args.output.absolute()
    calibration = read_calibration(args.calibration)
    evaluated = response(args.resources, args.stock)
    capture = evaluated["capture"]
    profile_hash = digest(Path(capture["source"]))
    tool = args.tool.resolve()
    executable = tool / ".build/release/graincapture"
    producer = json.loads((tool / "capture.json").read_bytes())
    if producer["fotufilm_revision"] != FOTUFILM or producer["capture_sha256"] != digest(executable):
        raise ValueError("capture tool differs from its pinned preparation")

    # Build and qualify output privately. mkdir arbitrates simultaneous requests
    # at publication, so a late occupant can never be overwritten.
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        scratch = Path(scratch)
        response_path = scratch / "response.json"
        write_json(response_path, {"data": evaluated["data"]})
        request = scratch / "request.json"
        write_json(request, {**evaluated, "calibration": calibration})
        bank = scratch / "bank"
        bank.mkdir()
        capture_environment = os.environ.copy()
        capture_environment.pop("FROZEN_FILM_BANK", None)
        subprocess.run([executable, request, bank, *args.footprint], env=capture_environment, check=True)
        manifest = {
            "version": 1,
            "stock": capture["stock"],
            "records": "records.f32",
            "sums": "sums.f32",
            "records_sha256": digest(bank / "records.f32"),
            "sums_sha256": digest(bank / "sums.f32"),
            "provenance": {
                "polarity": capture["polarity"],
                "profile_sha256": profile_hash,
                "response_sha256": digest(response_path),
                "calibration": calibration,
                "generator": producer,
            },
        }
        write_json(bank / "manifest.json", manifest)
        shutil.copyfile(response_path, bank / response_path.name)
        output.mkdir()
        for product in bank.iterdir():
            product.replace(output / product.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare", help="build the pinned Swift capture tool")
    setup.add_argument("--fotufilm", type=Path, required=True)
    setup.add_argument("--swift", type=lambda path: Path(path).absolute(), required=True)
    setup.add_argument("--output", type=Path, required=True)
    setup.add_argument("--jobs", type=int, default=4)
    setup.set_defaults(run=prepare)
    bank = commands.add_parser("generate", help="generate a capture stock bank")
    bank.add_argument("--tool", type=Path, required=True)
    bank.add_argument("--resources", type=Path, default=ROOT / "Resources")
    bank.add_argument("--stock", required=True)
    bank.add_argument("--calibration", type=Path, required=True)
    bank.add_argument("--output", type=Path, required=True)
    bank.add_argument("--footprint", action="append", default=[], help="add a reference footprint in texels")
    bank.set_defaults(run=generate)
    catalog = commands.add_parser("list", help="list the capture catalog")
    catalog.add_argument("--resources", type=Path, default=ROOT / "Resources")
    catalog.set_defaults(run=lambda args: print(json.dumps(response(args.resources, "--list"), indent=2)))
    args = parser.parse_args()
    try:
        args.run(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Film v1 bank: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
