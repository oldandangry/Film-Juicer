#!/usr/bin/env python3
"""Run an explicit Film reference capture, comparison, or measurement."""

import argparse
import importlib
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
execution = importlib.import_module("execution")


def summarize(log):
    observations = []
    for line in log.read_text(encoding="utf-8").splitlines():
        cells = line.split(",")
        if cells[0] in {"warmup", "sample"}:
            kind, block, index, variant, *timings = cells
            values = [float(value) for value in timings]
            if len(values) != 4 or any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("invalid grain timing row")
            observations.append(((kind, int(block), int(index), variant), values))
    orders = (("current", "film"), ("film", "current"), ("current", "film"))
    expected = [("warmup", -1, 0, arm) for arm in orders[0]] + [
        ("warmup" if index < 10 else "sample", block, index, arm)
        for block, order in enumerate(orders)
        for arm in order
        for index in range(40)
    ]
    if [key for key, _ in observations] != expected:
        raise ValueError("incomplete or out-of-order grain timing rows")
    metrics = ("device_ms", "host_submit_ms", "host_total_ms", "common_reset_ms")
    summary = {}
    for arm in orders[0]:
        samples = [values for (kind, _, _, variant), values in observations if kind == "sample" and variant == arm]
        summary[arm] = {"samples": len(samples)}
        for metric, values in zip(metrics, zip(*samples), strict=True):
            ordered = sorted(values)
            summary[arm][metric] = {
                "median": statistics.median(ordered),
                "p90": ordered[math.ceil(0.9 * len(ordered)) - 1],
            }
    summary["film_over_current_device"] = {
        key: summary["film"]["device_ms"][key] / summary["current"]["device_ms"][key] for key in ("median", "p90")
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("response", "capture", "reference", "benchmark"))
    parser.add_argument("--preset", choices=("linux-release", "windows-clang-release"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--input", type=Path, help="developed CMY file, new for capture and frozen for benchmark")
    args = parser.parse_args()
    if args.preset.startswith("windows") != (os.name == "nt"):
        parser.error("use the preset's native platform")
    if bool(args.bank) != (args.mode in {"reference", "benchmark"}):
        parser.error("reference and benchmark require --bank")
    if bool(args.input) != (args.mode in {"capture", "benchmark"}):
        parser.error("capture and benchmark require --input")
    run_id = execution.name_component(args.run_id)
    build = ROOT / "out/build" / args.preset
    resources = build / "tests/Resources"
    binary = build / "tests/bin" / ("JuicerGrainPrototype.exe" if os.name == "nt" else "JuicerGrainPrototype")
    directory = ROOT / "out/validation" / args.preset / "results" / run_id
    inputs = [binary, resources / "profiles/kodak_portra_400.json"]
    if args.bank:
        inputs.extend(
            args.bank.resolve() / name
            for name in (
                "sums.f32",
                "records.f32",
                "footprint-1.f32",
                "footprint-9.375.f32",
                "footprint-32.f32",
                "reference.tsv",
            )
        )
    if args.mode == "benchmark":
        inputs.append(args.input.resolve())

    command = [binary, args.mode]
    if args.bank:
        command.append(args.bank.resolve())
    if args.mode != "reference":
        command.append(resources)
    if args.input:
        command.append(args.input.resolve())

    with execution.OutputLease(ROOT, build):
        directory.mkdir(parents=True)
        run = execution.NativeRun(ROOT, directory)
        result = {
            "schema": "grain-prototype-v1",
            "mode": args.mode,
            "preset": args.preset,
            "status": "failed",
            "commands": run.commands,
        }
        try:
            result["head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            result["input_hashes"] = {str(path): execution.digest(path) for path in inputs}
            environment = {key: value for key, value in os.environ.items() if key.upper() != "JUICER_DIAGNOSTICS"}
            if args.mode == "response":
                run.command(args.mode, command, env=environment, stdout_only=True)
            else:
                smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
                with execution.OutputLease(ROOT):
                    run.command("device-before", [smi])
                    try:
                        log = run.command(args.mode, command, env=environment)
                    finally:
                        run.command("device-after", [smi])
                if args.mode == "capture":
                    result["developed_input_sha256"] = execution.digest(args.input)
                elif args.mode == "benchmark":
                    result["summary"] = summarize(log)
            if result["input_hashes"] != {str(path): execution.digest(path) for path in inputs}:
                raise ValueError("grain prototype input changed during execution")
            result["status"] = "passed"
        finally:
            run.publish(result)
    print(directory / "result.json")


if __name__ == "__main__":
    main()
