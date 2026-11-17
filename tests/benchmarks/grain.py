#!/usr/bin/env python3
"""Run an explicit Film reference capture, comparison, or measurement."""

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
execution = importlib.import_module("execution")


def quad_copy_report(path, bank):
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "quad_count": 3342336,
        "checked_words": 13369344,
        "bit_mismatches": 0,
        "device_quad_bytes": 53477376,
    }
    if any(report.get(key) != value for key, value in expected.items()) or Path(report["bank"]).resolve() != bank:
        raise ValueError("quad copy report has mismatches or an unexpected representation")
    return report


def research(args, parser):
    from grain_campaign import calibrate, footprint_input, freeze, read_json, write_json

    if args.experiment is None:
        parser.error(f"{args.mode} requires --experiment")
    experiment = args.experiment.resolve()
    if args.mode == "freeze":
        if args.input:
            from grain_stimuli import freeze_quality

            freeze_quality(ROOT, read_json(experiment), write_json, args.input.resolve())
            print(args.input.resolve())
            return
        freeze(ROOT, experiment.parent, experiment)
        print(experiment)
        return
    if args.preset is None or args.run_id is None:
        parser.error(f"{args.mode} requires --preset and --run-id")
    if args.preset.startswith("windows") != (os.name == "nt"):
        parser.error("use the preset's native platform")
    if args.mode in {"research", "quality"} and args.input is None:
        parser.error(f"{args.mode} requires --input with frozen capture requests")
    run_id = execution.name_component(args.run_id)
    build = ROOT / "out/build" / args.preset
    resources = build / "tests/Resources"
    binary = build / "tests/bin" / ("JuicerGrainPrototype.exe" if os.name == "nt" else "JuicerGrainPrototype")
    directory = ROOT / "out/validation" / args.preset / "results" / run_id
    output = experiment.parent / run_id
    environment = {key: value for key, value in os.environ.items() if key.upper() != "JUICER_DIAGNOSTICS"}
    with execution.OutputLease(ROOT, build):
        directory.mkdir(parents=True)
        output.mkdir()
        run = execution.NativeRun(ROOT, directory)
        result = {
            "schema": "grain-g2-research-v1",
            "preset": args.preset,
            "mode": args.mode,
            "commands": run.commands,
            "status": "failed",
            "captures": [],
        }
        try:
            script = Path(__file__).resolve()
            paths = [
                experiment,
                binary,
                script,
                script.with_name("grain_campaign.py"),
                script.with_name("grain_analysis.py"),
            ]
            if args.mode == "calibrate" and "chroma_fraction" in read_json(experiment):
                paths.append(script.with_name("grain_chroma.py"))
            if args.input is not None:
                paths.append(args.input.resolve())
            inputs = {path: execution.digest(path) for path in paths}
            result["input_hashes"] = {str(path): digest for path, digest in inputs.items()}
            result["head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

            def render(name, request):
                path = output / name
                path.mkdir()
                inputs = [resources / "profiles" / f"{request['stock']}.json"]
                request = request.copy()
                if request["engine"] == "film":
                    bank, tables = ROOT / request["bank"], ROOT / request["tables"]
                    inputs.extend(
                        (
                            bank / "sums.f32",
                            bank / "records.f32",
                            bank / "footprint-1.f32",
                            tables / "records.f32",
                            footprint_input(ROOT, request),
                        )
                    )
                    request.update(bank=str(bank), tables=str(tables))
                request_path = path / "request.json"
                write_json(request_path, request)
                inputs.append(request_path)
                hashes = {str(file): execution.digest(file) for file in inputs}
                capture = {"path": str(path), "inputs": hashes, "status": "failed"}
                result["captures"].append(capture)
                with execution.OutputLease(ROOT):
                    run.command(name, [binary, "research", resources, request_path], env=environment)
                if hashes != {str(file): execution.digest(file) for file in inputs}:
                    raise ValueError("research capture input changed")
                manifest = read_json(path / "manifest.json")
                expected = 4 * 3 * request["crop_width"] * request["crop_height"] * len(request["requests"])
                frames = path / "frames.f32"
                if (
                    frames.stat().st_size != expected
                    or len(manifest["resolved"]) != len(request["requests"])
                    or any(manifest.get(key) != value for key, value in request.items())
                ):
                    raise ValueError("incomplete research capture")
                capture.update(
                    status="passed", manifest=execution.digest(path / "manifest.json"), frames=execution.digest(frames)
                )
                return path

            smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
            run.command("device-before", [smi])
            try:
                if args.mode == "calibrate":
                    calibrate(read_json(experiment), render, output, args.rows)
                elif args.mode == "quality":
                    requests = read_json(args.input)
                    if args.rows:
                        requests = {name: requests[name] for name in args.rows}
                    if not requests:
                        raise ValueError("missing quality capture requests")
                    failures = []
                    for name, request in requests.items():
                        print(f"Capturing {name}", flush=True)
                        try:
                            render(execution.name_component(name), request)
                        except (OSError, ValueError, subprocess.CalledProcessError) as error:
                            failures.append({"name": name, "error": str(error)})
                    if failures:
                        result["capture_failures"] = failures
                        raise ValueError("incomplete quality captures")
                else:
                    render("capture", read_json(args.input))
            finally:
                run.command("device-after", [smi])
            if inputs != {path: execution.digest(path) for path in inputs}:
                raise ValueError("research run inputs changed")
            result["status"] = "passed"
        finally:
            run.publish(result)
    print(directory / "result.json")


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
    parser.add_argument(
        "mode",
        choices=(
            "response",
            "capture",
            "reference",
            "reference-g2",
            "reference-scalar",
            "reference-g2-scalar",
            "reference-reuse",
            "reference-g2-reuse",
            "reference-paired",
            "reference-g2-paired",
            "reference-quad",
            "reference-g2-quad",
            "verify-quads",
            "benchmark",
            "freeze",
            "calibrate",
            "research",
            "quality",
            "g3-inputs",
            "g3-check",
            "g3",
            "g3-report",
            "g3-profile",
            "g3-screen",
            "g3-verify-opt",
            "g3-opt-report",
        ),
    )
    parser.add_argument("--preset", choices=("linux-release", "windows-clang-release"))
    parser.add_argument("--run-id")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--stock", help="stock resource ID for response; defaults to kodak_portra_400")
    parser.add_argument("--input", type=Path, help="developed CMY file or G2 capture request JSON")
    parser.add_argument("--tables", type=Path, help="footprint-table directory for reference-g2")
    parser.add_argument("--probes", type=Path, help="independent probe TSV for reference-g2")
    parser.add_argument("--experiment", type=Path, help="frozen G2 experiment JSON in the shared output directory")
    parser.add_argument("--rows", nargs="+", help="optional calibration or quality capture IDs")
    parser.add_argument("--current-receipt", type=Path, help="retained G2 Current capture result")
    parser.add_argument("--film-receipt", type=Path, help="retained G2 reduced-chroma capture result")
    parser.add_argument("--check-result", type=Path, help="completed G3 stage check for the same binary and manifest")
    parser.add_argument("--results", nargs="+", type=Path, help="completed G3 stage results for offline reporting")
    parser.add_argument("--route-results", nargs="+", type=Path, help="completed Current route results")
    parser.add_argument("--output", type=Path, help="new G3 offline report JSON")
    parser.add_argument(
        "--film-variant", choices=("reference", "scalar", "reuse", "paired", "quad"), default="reference"
    )
    parser.add_argument("--film-block", choices=("32x4", "16x8", "32x8"), default="32x4")
    parser.add_argument("--baseline-variant", choices=("reference", "scalar"), default="reference")
    parser.add_argument("--baseline-block", choices=("32x4", "16x8", "32x8"), default="32x4")
    parser.add_argument("--profile-set", choices=("standard", "sat"), default="standard")
    parser.add_argument("--profiler", type=Path, help="native Nsight Compute CLI for explicit profiling")
    parser.add_argument(
        "--profile-result", type=Path, help="closed failed profile receipt whose captured kernels must not repeat"
    )
    parser.add_argument("--agreement-result", type=Path, help="completed reference/optimized full-frame agreement")
    parser.add_argument(
        "--profile-reader", type=Path, help="Nsight Compute extras/python directory for offline reports"
    )
    args = parser.parse_args()
    if args.mode == "g3-opt-report":
        import grain_optimization

        if not (args.input and args.results and args.output):
            parser.error("g3-opt-report requires --input, --results and --output")
        grain_optimization.offline_report(ROOT, args)
        return
    if args.mode in {"g3-profile", "g3-screen", "g3-verify-opt"}:
        import grain_optimization

        grain_optimization.run_native(ROOT, args, parser)
        return
    if args.mode in {"g3-inputs", "g3-check", "g3", "g3-report"}:
        import grain_g3

        if args.mode == "g3-report":
            if args.input is None or args.output is None:
                parser.error("g3-report requires --input and --output")
            grain_g3.offline_report(args.input, args.results or [], args.route_results or [], args.output, root=ROOT)
        else:
            grain_g3.run_native(ROOT, args, parser)
        return
    if args.mode in {"freeze", "calibrate", "research", "quality"}:
        research(args, parser)
        return
    if args.preset is None or args.run_id is None:
        parser.error(f"{args.mode} requires --preset and --run-id")
    if args.preset.startswith("windows") != (os.name == "nt"):
        parser.error("use the preset's native platform")
    reference_modes = {
        "reference",
        "reference-scalar",
        "reference-reuse",
        "reference-g2",
        "reference-g2-scalar",
        "reference-g2-reuse",
        "reference-paired",
        "reference-g2-paired",
        "reference-quad",
        "reference-g2-quad",
    }
    g2_reference_modes = {
        "reference-g2",
        "reference-g2-scalar",
        "reference-g2-reuse",
        "reference-g2-paired",
        "reference-g2-quad",
    }
    if bool(args.bank) != (args.mode in reference_modes | {"benchmark", "verify-quads"}):
        parser.error("reference, reference-g2, verify-quads and benchmark require --bank")
    if args.mode in g2_reference_modes and not (args.tables and args.probes):
        parser.error("reference-g2 requires --tables and --probes")
    if bool(args.input) != (args.mode in {"capture", "benchmark"}):
        parser.error("capture and benchmark require --input")
    if args.stock is not None and args.mode != "response":
        parser.error("--stock is only supported for response")
    stock = args.stock or "kodak_portra_400"
    if re.fullmatch(r"[a-z0-9_]+", stock) is None:
        parser.error("invalid stock resource ID")
    run_id = execution.name_component(args.run_id)
    build = ROOT / "out/build" / args.preset
    resources = build / "tests/Resources"
    binary = build / "tests/bin" / ("JuicerGrainPrototype.exe" if os.name == "nt" else "JuicerGrainPrototype")
    directory = ROOT / "out/validation" / args.preset / "results" / run_id
    inputs = [Path(__file__), binary, resources / "profiles" / f"{stock}.json"]
    if args.mode in g2_reference_modes:
        inputs.extend(args.bank.resolve() / name for name in ("sums.f32", "records.f32", "footprint-1.f32"))
        inputs.extend(args.tables.resolve() / f"footprint-{pitch}.f32" for pitch in (0.25, 0.78125, 3.17, 37.5))
        inputs.extend((args.tables.resolve() / "records.f32", args.probes.resolve()))
    elif args.mode == "verify-quads":
        inputs.extend(args.bank.resolve() / name for name in ("sums.f32", "records.f32", "footprint-1.f32"))
    elif args.bank:
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
    if args.mode in g2_reference_modes:
        command.extend((args.tables.resolve(), args.probes.resolve()))
    elif args.mode == "verify-quads":
        command.append(directory / "quad-copy.json")
    elif args.mode not in reference_modes:
        command.append(resources)
    if args.input:
        command.append(args.input.resolve())
    if args.mode == "response":
        command.append(stock)

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
                elif args.mode == "verify-quads":
                    raw = directory / "quad-copy.json"
                    result.update(
                        quad_copy=quad_copy_report(raw, args.bank.resolve()), report_sha256=execution.digest(raw)
                    )
            if result["input_hashes"] != {str(path): execution.digest(path) for path in inputs}:
                raise ValueError("grain prototype input changed during execution")
            result["status"] = "passed"
        finally:
            run.publish(result)
    print(directory / "result.json")


if __name__ == "__main__":
    main()
