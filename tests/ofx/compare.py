#!/usr/bin/env python3
"""Compare supplied production OFX modules in the native host without installing."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
execution = importlib.import_module("execution")
verify = importlib.import_module("verify")

SIDES = ("reference", "candidate")
OBSERVER = ROOT / "tests/benchmarks/observe_seed.py"
CASES = tuple(
    f"{polarity}-{route}-ae-{ae}"
    for polarity in ("negative", "positive")
    for route in ("direct", "print")
    for ae in ("off", "on")
)
SHAPE = (1080, 1920, 4)
ADVANCED_DEFAULTS = {
    "HalationScatterCoreSigmaUm": [2.2, 2.0, 1.6],
    "HalationScatterTailScaleUm": [9.3, 9.7, 9.1],
    "HalationScatterTailWeight": [0.78, 0.65, 0.67],
    "HalationReflectionSource": [0],
    "HalationStrengthRgb": [0.05, 0.015, 0.0],
    "HalationFirstBounceSigmaUm": [65.0, 65.0, 65.0],
    "HalationBounceCount": [3],
    "HalationBounceDecay": [0.5],
    "HalationRenormalize": [1],
    "HalationHighlightBoostEv": [0.0],
    "HalationHighlightProtectionEv": [4.0],
    "HalationHighlightBoostRange": [0.3],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory(path):
    return {
        item.relative_to(path).as_posix(): execution.digest(item) for item in sorted(path.rglob("*")) if item.is_file()
    }


def validate_capture(report, side):
    require(report.get("schema") == "ofx-halation-capture-v1", "capture schema")
    expected_ids = [f"{case}/{phase}" for case in CASES for phase in ("cold", "warm")]
    rows = report["rows"]
    require([row["case_id"] for row in rows] == expected_ids, "capture membership or ordering")
    for row in rows:
        for name in (
            "finite_output",
            "alpha_preserved",
            "padding_preserved",
            "source_preserved",
            "stream_completed",
            "readback_complete",
            "context_preserved",
            "context_restored",
            "cold_warm_identical",
        ):
            require(row.get(name) is True, f"{row['case_id']}: {name}")
        require(row["ofx_status"] == 0 and row["messages"] == [], "OFX render failure")
        require(row["input_shape"] == list(SHAPE) == row["output_shape"], "capture geometry")
        require(row["samples_compared"] == int(np.prod(SHAPE)), "capture sample count")
        require(row["source_image_releases"] == row["output_image_releases"] == 1, "image release count")
    life = report["lifecycle"]
    require(life["effects_created"] == life["effects_destroyed"] == len(CASES), "instance lifetime")
    require(life["destroy_statuses"] == [0] * len(CASES), "instance destruction")
    require(life["module_load_status"] == life["module_unload_status"] == 0, "module lifetime")
    require(life["context_restored"] is True, "context restoration")
    require(set(report["parameters"]) == set(CASES), "parameter membership")
    for case in CASES:
        parameters = report["parameters"][case]
        expected = {
            "HalationActive": [1],
            "HalationScatterAmount": [1.0],
            "HalationScatterSpatialScale": [1.0],
            "HalationAmount": [1.0],
            "HalationSpatialScale": [1.0],
            "SpectralUpsampling": [0],
            "InputColorSpace": [1],
            "CameraAutoExposure": [int(case.endswith("-on"))],
            "CameraFilmFormatMm": [35.0],
            "FilmProfileKey": ["kodak_portra_400" if case.startswith("negative") else "fujifilm_provia_100f"],
            "ScanRoute": ["NegativePrintScan" if "-print-" in case else "NegativeDirectScan"],
            "GrainActive": [0],
            "CameraDiffusionEnabled": [0],
            "PrintDiffusionEnabled": [0],
        }
        if side == "candidate":
            expected.update(ADVANCED_DEFAULTS)
        else:
            require(not (parameters.keys() & ADVANCED_DEFAULTS.keys()), "reference has advanced controls")
        for name, value in expected.items():
            require(parameters.get(name) == value, f"{case}: unexpected {name}")


def compare_pixels(reference, candidate, shape, *, preserve_alpha=True):
    samples = int(np.prod(shape))
    require(reference.stat().st_size == candidate.stat().st_size == samples * 4, "FP32 file size")
    left = np.fromfile(reference, dtype="<f4").reshape(shape)
    right = np.fromfile(candidate, dtype="<f4").reshape(shape)
    require(np.isfinite(left).all() and np.isfinite(right).all(), "nonfinite captured pixels")
    changed = left.view("<u4") != right.view("<u4")
    if shape[2] == 4 and preserve_alpha:
        require(not changed[..., 3].any(), "alpha differs between modules")
    delta = right[..., :3].astype(np.float64) - left[..., :3].astype(np.float64)
    return {
        "bit_identical": not bool(changed.any()),
        "pixels": shape[0] * shape[1],
        "changed_pixels": int(changed.any(axis=2).sum()),
        "changed_rgb_samples": int(changed[..., :3].sum()),
        "changed_alpha_samples": int(changed[..., 3].sum()) if shape[2] == 4 else 0,
        "max_abs_rgb": float(np.abs(delta).max()),
        "rms_rgb": float(np.sqrt(np.mean(delta * delta))),
        "reference_sha256": execution.digest(reference),
        "candidate_sha256": execution.digest(candidate),
    }


def compare_captures(directory):
    reports = {side: json.loads((directory / side / "capture.json").read_text()) for side in SIDES}
    for side, report in reports.items():
        validate_capture(report, side)
    reference, candidate = (reports[side] for side in SIDES)
    require(reference["geometry"] == candidate["geometry"], "different host geometry")
    require(reference["device"] == candidate["device"], "different CUDA device/driver")
    for phase in ("authored_parameters", "parameters"):
        require(set(reference[phase]) == set(candidate[phase]) == set(CASES), "parameter phase membership")
        for case in CASES:
            common = {key: value for key, value in candidate[phase][case].items() if key not in ADVANCED_DEFAULTS}
            require(reference[phase][case] == common, f"{case}: common {phase} differ")
            for name, value in ADVANCED_DEFAULTS.items():
                require(candidate[phase][case].get(name) == value, f"{case}: changed default {name}")
    source = compare_pixels(directory / "reference/input.f32", directory / "candidate/input.f32", SHAPE)
    require(source["bit_identical"], "different source pixels")
    rows = {
        case: compare_pixels(directory / "reference" / f"{case}.f32", directory / "candidate" / f"{case}.f32", SHAPE)
        for case in CASES
    }
    return {
        "status": "passed" if all(row["bit_identical"] for row in rows.values()) else "failed",
        "input": source,
        "rows": rows,
    }


def compare_frame(directory, frame):
    capture = json.loads((frame / "capture.json").read_text())
    require(capture["schema"] == "juicer-ofx-capture-v1", "frame capture schema")
    require(capture["complete"] is True and capture["status"] == 0, "incomplete host capture")
    bounds = capture["output"]["bounds"]
    require(capture["action"]["render_window"] == bounds, "comparison requires a full output frame")
    components = {"OfxImageComponentRGB": 3, "OfxImageComponentRGBA": 4}[capture["output"]["components"]]
    shape = (bounds[3] - bounds[1], bounds[2] - bounds[0], components)
    reports = {side: json.loads((directory / side / "replay.json").read_text()) for side in SIDES}
    for side, report in reports.items():
        require(report["schema"] == "juicer-ofx-replay-v1" and report["context_restored"], "replay completion")
        require([row["pass"] for row in report["rows"]] == ["cold", "warm"], "replay membership")
        require(all(row["status"] == 0 for row in report["rows"]), "replay render status")
    rows = {
        "between_binaries": compare_pixels(
            directory / "reference/cold.f32", directory / "candidate/cold.f32", shape, preserve_alpha=False
        ),
        "host_to_rollback": compare_pixels(
            frame / capture["output_file"], directory / "reference/cold.f32", shape, preserve_alpha=False
        ),
        "host_to_candidate": compare_pixels(
            frame / capture["output_file"], directory / "candidate/cold.f32", shape, preserve_alpha=False
        ),
    }
    for side in SIDES:
        rows[f"{side}_cold_warm"] = compare_pixels(directory / side / "cold.f32", directory / side / "warm.f32", shape)
    # This is a diagnostic comparison: a measured difference is a completed
    # result, not a renderer execution failure or an implementation parity pass.
    snapshot_differences = []
    for name, observed in capture.get("latest_reads", {}).items():
        current = capture["parameters"].get(name)
        if current is None or any(
            current.get(key) != observed.get(key) for key in ("type", "status", "values", "f64_bits")
        ):
            snapshot_differences.append(name)
    return {
        "status": "passed",
        "measurement_only": True,
        "shape": list(shape),
        "rows": rows,
        "snapshot_vs_latest_read_differences": snapshot_differences,
        "parameters": reports,
    }


def scanner_frame_input(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest["schema"] == "juicer-native-scanner-frame-v1", "native scanner frame schema")
    name = manifest["source_file"]
    require(Path(name).name == name, "native scanner source must accompany the manifest")
    source = path.parent / name
    require(execution.digest(source) == manifest["source_sha256"], "native scanner frame source identity")
    return {"path": str(path), "sha256": execution.digest(path), "source_sha256": manifest["source_sha256"]}


def benchmark_reader():
    sys.path.insert(0, str(OBSERVER.parent))
    try:
        return importlib.import_module("ofx")
    finally:
        sys.path.pop(0)


def verify_photo_readbacks(frames, host_rows, route_tag, *, cubic):
    bits = [struct.unpack("<Q", struct.pack("<d", row["time"]))[0] for row in host_rows]
    require(len(frames) == len(bits) == 3, "scanner photo readback membership")
    cold, warm, reporter = frames
    require(
        all(
            row["index"] == index
            and row["time_bits"] == bits[index]
            and row["route"] == route_tag
            and row["instance"] == cold["instance"] != 0
            and row["thread"] == cold["thread"]
            and row["outcome"] == [0, 0, 0, 0]
            for index, row in enumerate(frames)
        ),
        "scanner photo draw identity differs",
    )
    poll = benchmark_reader().spectral_replay_poll
    return {
        "cold": poll(cold, warm, frozen_photo=True, legacy_producer=cubic),
        "warm": poll(warm, reporter, legacy_producer=cubic),
    }


def compare_scanner_frame(directory, frame, *, strict_guards=True, scanner_xyz=False):
    manifest = json.loads(frame.read_text(encoding="utf-8"))
    image = manifest["output"]
    x1, y1, x2, y2 = image["bounds"]
    require(manifest["action"]["render_window"] == image["bounds"], "native scanner comparison requires a full frame")
    components = {"OfxImageComponentRGB": 3, "OfxImageComponentRGBA": 4}.get(image["components"])
    require(components is not None, "native scanner frame requires RGB or RGBA")
    shape = (y2 - y1, x2 - x1, components)
    observe_print = manifest.get("observe_print", False)
    observe_current = manifest.get("variant") == "ordinary" and "grain_replay" in manifest
    frozen = manifest["action"]["time"]
    reports = {side: json.loads((directory / side / "replay.json").read_text()) for side in SIDES}
    scanner_xyz = scanner_xyz or any("scanner_evidence" in row for report in reports.values() for row in report["rows"])
    require(
        not scanner_xyz or observe_print and manifest.get("print_glare") == "constant", "unexpected Print XYZ capture"
    )
    for side, report in reports.items():
        require(report["context_restored"] and report["schema"] == "juicer-ofx-replay-v1", "native replay completion")
        require(report["cubic_grid"] == (17 if side == "reference" else None), "native scanner method identity")
        require(report.get("grain_replay") == manifest.get("grain_replay"), "native replay tokens differ")
        phases = ["cold", "warm", "reporter"] if observe_print or observe_current else ["cold", "warm"]
        require([row["pass"] for row in report["rows"]] == phases, "native replay membership")
        require(
            all(row["status"] == 0 and row["source_preserved"] and row["stream_completed"] for row in report["rows"]),
            "native scanner replay terminal status",
        )
        if strict_guards or observe_print or observe_current:
            require(
                all(
                    row["context_preserved"]
                    and row["destination_guard_checked"]
                    and row["source_releases"] == row["output_releases"] == 1
                    and row["time"] == frozen + int(row["pass"] == "reporter")
                    and row["output_file"] == (None if row["pass"] == "reporter" else f"{row['pass']}.f32")
                    for row in report["rows"]
                ),
                "native scanner guards or frozen draw identity differ",
            )
        require(
            compare_pixels(directory / side / "cold.f32", directory / side / "warm.f32", shape)["bit_identical"],
            f"{side} native replay output changed on warm render",
        )
    require(reports["reference"]["parameters"] == reports["candidate"]["parameters"], "native scanner controls differ")
    files = {side: directory / side / "cold.f32" for side in SIDES}
    pixels = {side: np.memmap(path, dtype="<f4", mode="r", shape=shape) for side, path in files.items()}
    if components == 4:
        require(
            all(manifest["source"][field] == image[field] for field in ("bounds", "components")),
            "native scanner alpha check requires matching source geometry",
        )
        source = np.memmap(frame.parent / manifest["source_file"], dtype="<u4", mode="r", shape=shape)
        require(
            all(np.array_equal(raster.view("<u4")[..., 3], source[..., 3]) for raster in pixels.values()),
            "native scanner replay changed source alpha",
        )
    channels = []
    for channel in range(3):
        cubic, spectral = (pixels[side][..., channel] for side in SIDES)
        delta = np.abs(spectral.astype(np.float64) - cubic)
        flat = int(delta.argmax())
        y, x = divmod(flat, shape[1])
        rank = (999 * delta.size + 999) // 1000 - 1
        channels.append(
            {
                "channel": channel,
                "mae": float(delta.mean()),
                "p999": float(np.partition(delta.ravel(), rank)[rank]),
                "max": float(delta[y, x]),
                "worst_xy": [x + x1, y + y1],
                "cubic": float(cubic[y, x]),
                "spectral": float(spectral[y, x]),
            }
        )
    comparison = {
        "status": "passed",
        "measurement_only": True,
        "shape": list(shape),
        "channels": channels,
        "parameters": reports,
    }
    if observe_print:
        glare = manifest.get("print_glare", "off")
        route = manifest["parameters"]["ScanRoute"]["values"][0]
        route_tag = {"NegativePrintScan": 1, "PositivePrintScan": 3}[route]
        observed = {}
        terminal = {}
        stages, xyz_rasters = {}, {}
        shared_inputs = None
        pixel_count = shape[0] * shape[1]
        for side in SIDES:
            path = directory / side
            report = json.loads((path / "print-scanner.json").read_text(encoding="utf-8"))
            frames = report["frames"]
            require(report["status"] == "passed" and report["exit"] == 0, "Print scanner observation did not complete")
            terminal[side] = verify_photo_readbacks(frames, reports[side]["rows"], route_tag, cubic=side == "reference")
            compared = frames[:2]
            xyz_hash = None
            for row, host in zip(compared, reports[side]["rows"][:2], strict=True):
                item = row["density"]
                require(
                    (item["width"], item["height"]) == (shape[1], shape[0])
                    and (path / item["file"]).stat().st_size == pixel_count * 12
                    and execution.digest(path / item["file"]) == item["sha256"],
                    "Print scanner paper C/M/Y readback differs",
                )
                if scanner_xyz:
                    capture = host.get("scanner_evidence")
                    stage_inputs = benchmark_reader().print_stage_inputs(
                        capture, row, host["pass"], pixel_count, cubic=side == "reference"
                    )
                    if shared_inputs is None:
                        shared_inputs = stage_inputs
                    require(stage_inputs == shared_inputs, "matched-field Print correction or color inputs differ")
                    xyz = path / capture["xyz"]
                    require(xyz.is_file() and xyz.stat().st_size == pixel_count * 24, "pre-glare XYZ raster size")
                    digest = execution.digest(xyz)
                    if host["pass"] == "cold":
                        xyz_hash = digest
                        stages[side] = {**capture, "sha256": digest}
                        xyz_rasters[side] = np.memmap(xyz, dtype="<f8", mode="r", shape=(3, *shape[:2]))
                    else:
                        require(digest == xyz_hash, "pre-glare XYZ changed on the warm draw")
                    name = capture["glare_file"]
                    row["glare"] = {"file": name, "sha256": execution.digest(path / name)}
                    glare_present = capture["glare"]
                else:
                    glare_present = row["scanner_glare_pointer"] != 0
                if glare == "off":
                    require("glare" not in row and not glare_present, "glare off consumed a field")
                    continue
                field = row["glare"]
                raster = path / field["file"]
                require(
                    glare_present
                    and raster.stat().st_size == pixel_count * 4
                    and execution.digest(raster) == field["sha256"],
                    "Print scanner glare field readback differs",
                )
                if glare == "constant":
                    require(
                        np.all(np.memmap(raster, dtype="<f4", mode="r") == 1.0),
                        "expected exact uniform one-percent glare",
                    )
            observed[side] = compared
            for name in ("density", "glare"):
                if name in compared[0]:
                    require(compared[0][name]["sha256"] == compared[1][name]["sha256"], f"warm Print {name} changed")
        comparison["print_scanner"] = {
            **benchmark_reader().print_input_verdict(observed, glare),
            "frames": observed,
            "readback": terminal,
        }
        if scanner_xyz:
            summary = []
            for cubic, spectral in zip(xyz_rasters["reference"], xyz_rasters["candidate"], strict=True):
                delta = np.abs(spectral - cubic)
                require(np.isfinite(delta).all(), "nonfinite pre-glare XYZ delta")
                y, x = np.unravel_index(delta.argmax(), delta.shape)
                summary.append(
                    {"mae": float(delta.mean()), "max": float(delta[y, x]), "worst_xy": [int(x) + x1, int(y) + y1]}
                )
            comparison["print_scanner"].update(
                claim="matched-field", pre_glare_xyz={"draws": stages, "channels": summary}
            )
    if observe_current:
        route = manifest["parameters"]["ScanRoute"]["values"][0]
        require(route == "NegativeDirectScan", "seeded Ordinary photo requires Negative Direct")
        terminal, payloads = {}, []
        for side in SIDES:
            report = json.loads((directory / side / "current-scanner.json").read_text(encoding="utf-8"))
            require(report["status"] == "passed" and report["exit"] == 0, "Current grain observation did not complete")
            frames = report["frames"]
            terminal[side] = verify_photo_readbacks(frames, reports[side]["rows"], 0, cubic=side == "reference")
            for row in frames[:2]:
                require(row["tokens"] == manifest["grain_replay"], "live Current replay tokens differ")
                require(len(row["current_grain"]) == 4, "missing live Current grain words")
                payloads.append(row["current_grain"])
        require(all(payload == payloads[0] for payload in payloads), "live Current grain payload differs")
        comparison["current_scanner"] = {
            "tokens": manifest["grain_replay"],
            "payload_words": payloads[0],
            "readback": terminal,
        }
    return comparison


def photo_observer_command(command, *, declaration, module, output, frame, environment, mode):
    debugger_environment = benchmark_reader().scanner_debugger_environment
    debugger = Path(declaration["debugger"]).resolve(strict=True)
    require(execution.digest(debugger) == declaration["debugger_sha256"], "scanner photo debugger identity")
    require(execution.digest(module) == declaration["module_sha256"], "scanner photo module identity")
    symbols = Path(declaration["symbols"]).resolve(strict=True) if os.name == "nt" else module
    require(execution.digest(symbols) == declaration["symbols_sha256"], "scanner photo symbol identity")
    manifest = json.loads(frame.read_text(encoding="utf-8"))
    route = manifest["parameters"]["ScanRoute"]["values"][0]
    routes = ("NegativePrintScan", "PositivePrintScan") if mode == "print" else ("NegativeDirectScan",)
    require(route in routes, "scanner photo observer route does not match the fixture")
    x0, y0, x1, y1 = manifest["output"]["bounds"]
    settings = output.parent / f"{output.name}-{mode}-scanner-observer.json"
    execution.write_json(
        settings,
        {
            "mode": f"{mode}-photo",
            "expected_route": route,
            "module": str(module),
            "symbols": str(symbols) if os.name == "nt" else None,
            "uuid": declaration["uuid"],
            "points": declaration["points"],
            "layout": declaration["layout"],
            "spectral_payload": declaration["method"] == "spectral",
            "density_geometry": [x1 - x0, y1 - y0],
            "print_glare": manifest.get("print_glare", "off"),
            "trace": str(output.parent / f"{output.name}-{mode}-scanner.jsonl"),
            "receipt": str(output / f"{mode}-scanner.json"),
        },
    )
    environment = {**environment, "FJ_SCANNER_OBSERVER": str(settings)}
    debugger_inputs = declaration
    if os.name == "nt":
        debugger_vars = declaration.get("debugger_environment", {})
        debugger_inputs = {
            **declaration,
            "debugger_environment": {
                **debugger_vars,
                "PATH": str(module.parent) + os.pathsep + debugger_vars.get("PATH", environment.get("PATH", "")),
            },
        }
    extra = [] if os.name == "nt" else ["-o", "settings set target.disable-aslr false"]
    launch = [
        str(debugger),
        "--batch",
        *extra,
        "-o",
        f'command script import "{OBSERVER.as_posix()}"',
        "-o",
        "script observe_seed.install(lldb.debugger)",
        "-o",
        "run",
        "-o",
        "script observe_seed.finish(lldb.debugger)",
        "--",
        *command,
    ]
    return debugger_environment(launch, environment, debugger_inputs)


def run(args):
    preset = "windows-clang-release" if os.name == "nt" else "linux-release"
    build = ROOT / "out/build" / preset
    directory = ROOT / "out/validation" / preset / "results" / execution.name_component(args.run_id)
    with execution.OutputLease(ROOT, build), execution.OutputLease(ROOT):
        cache = verify.cache_values(build)
        require(cache["CMAKE_BUILD_TYPE"] == "Release", "probe requires Release")
        require(args.scanner_frame or cache["BUILD_TESTING"] == "OFF", "legacy probe requires testing OFF")
        probe = (
            build
            / "tests/bin"
            / ("JuicerHalationComparisonProbe.exe" if os.name == "nt" else "JuicerHalationComparisonProbe")
        )
        require(probe.is_file(), "prepare JuicerHalationComparisonProbe first")
        modules = {side: Path(getattr(args, side)).resolve(strict=True) for side in SIDES}
        before = {}
        for side, module in modules.items():
            require(execution.digest(module) == getattr(args, f"{side}_sha256").lower(), f"{side} module identity")
            before[side] = inventory(module.parent.parent)
            require(any(name.startswith("Resources/") for name in before[side]), f"{side} resources missing")
        require(
            {k: v for k, v in before["reference"].items() if k.startswith("Resources/")}
            == {k: v for k, v in before["candidate"].items() if k.startswith("Resources/")},
            "bundled resources differ",
        )
        directory.mkdir(parents=True, exist_ok=False)
        owner = execution.NativeRun(ROOT, directory)
        result = {
            "schema": "ofx-halation-comparison-v1",
            "status": "failed",
            "preset": preset,
            "commands": owner.commands,
        }
        source, patch, _ = verify.capture_native_source(ROOT, "tests/ofx/compare.py", domain_tests=True)
        execution.write_json(directory / "harness-source.json", source)
        (directory / "harness-source.patch").write_bytes(patch)
        inputs = {
            "probe": {"path": str(probe), "sha256": execution.digest(probe)},
            "reader_sha256": execution.digest(Path(__file__)),
            "bundles": before,
            "modules": {
                side: {"path": str(module), "sha256": execution.digest(module)} for side, module in modules.items()
            },
            "build_testing": cache["BUILD_TESTING"],
            "build_type": cache["CMAKE_BUILD_TYPE"],
        }
        frame = args.frame.resolve(strict=True) if args.frame else None
        scanner_frame = args.scanner_frame.resolve(strict=True) if args.scanner_frame else None
        if frame:
            inputs["frame"] = {"path": str(frame), "members": inventory(frame)}
        if scanner_frame:
            inputs["scanner_frame"] = scanner_frame_input(scanner_frame)
        photo_arg = args.print_scanner_observation or args.current_scanner_observation
        photo = photo_arg.resolve(strict=True) if photo_arg else None
        mode = "current" if args.current_scanner_observation else "print"
        if scanner_frame:
            spec = json.loads(scanner_frame.read_text(encoding="utf-8"))
            require(
                bool(spec.get("observe_print")) == bool(args.print_scanner_observation),
                "Print observation and frame manifest must agree",
            )
            current = spec.get("variant") == "ordinary" and "grain_replay" in spec
            require(
                current == bool(args.current_scanner_observation), "Current observation and frame manifest must agree"
            )
        require(photo is None or scanner_frame is not None, "scanner observation requires a native frame")
        if photo:
            inputs["scanner_observation"] = {"path": str(photo), "sha256": execution.digest(photo), "mode": mode}
            inputs["observer_sha256"] = execution.digest(OBSERVER)
            photo_bindings = json.loads(photo.read_text(encoding="utf-8"))
        overrides = args.parameter_overrides.resolve(strict=True) if args.parameter_overrides else None
        require(overrides is None or frame is not None, "parameter overrides require a frame capture")
        if overrides:
            inputs["parameter_overrides"] = {"path": str(overrides), "sha256": execution.digest(overrides)}
        try:
            # Refresh only this standalone consumer under the same owner as its
            # source capture and execution. Supplied product bundles are inputs.
            owner.command(
                "build-probe",
                [
                    cache["CMAKE_COMMAND"],
                    "--build",
                    str(build),
                    "--target",
                    "JuicerHalationComparisonProbe",
                    "--",
                    "-k0",
                ],
                vctip_executable=execution.windows_vctip(os.environ),
            )
            inputs["probe"]["sha256"] = execution.digest(probe)
            artifacts = json.loads((build / "tests/rust-artifacts.json").read_text())
            require(artifacts["consumers"]["JuicerHalationComparisonProbe"]["variant"] == "none", "probe links Rust")
            inputs["rust_variant"] = "none"
            execution.write_json(directory / "inputs.json", inputs)
            for side in SIDES:
                (directory / side).mkdir()
                environment = dict(os.environ)
                if os.name == "nt":
                    # Load the supplied bundle's CUDA runtime, not another
                    # installed toolkit that happens to precede it on PATH.
                    environment["PATH"] = str(modules[side].parent) + os.pathsep + environment.get("PATH", "")
                environment["FJ_HOST_CASES"] = str(directory / f"{side}.cases.jsonl")
                command = [str(probe), "--halation-capture", str(modules[side]), str(directory / side)]
                if frame:
                    command = [str(probe), "--frame-replay", str(modules[side]), str(frame), str(directory / side)]
                if scanner_frame:
                    command = [
                        str(probe),
                        "--native-scanner-frame",
                        str(modules[side]),
                        str(scanner_frame),
                        str(directory / side),
                    ]
                if overrides:
                    command.extend(["--parameters", str(overrides)])
                if photo:
                    command, environment = photo_observer_command(
                        command,
                        declaration=photo_bindings[side],
                        module=modules[side],
                        output=directory / side,
                        frame=scanner_frame,
                        environment=environment,
                        mode=mode,
                    )
                owner.command(side, command, env=environment)
            if scanner_frame:
                comparison = compare_scanner_frame(directory, scanner_frame)
            else:
                comparison = compare_frame(directory, frame) if frame else compare_captures(directory)
            execution.write_json(directory / "comparison.json", comparison)
            require(comparison["status"] == "passed", "module output differs; see comparison.json")
            result["status"] = "passed"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            after, after_patch, _ = verify.capture_native_source(ROOT, "tests/ofx/compare.py", domain_tests=True)
            result["harness_unchanged"] = (
                after == source and after_patch == patch and execution.digest(probe) == inputs["probe"]["sha256"]
            )
            result["bundles_unchanged"] = all(
                inventory(module.parent.parent) == before[side] for side, module in modules.items()
            )
            result["frame_unchanged"] = frame is None or inventory(frame) == inputs["frame"]["members"]
            result["scanner_frame_unchanged"] = (
                scanner_frame is None or scanner_frame_input(scanner_frame) == inputs["scanner_frame"]
            )
            result["overrides_unchanged"] = (
                overrides is None or execution.digest(overrides) == inputs["parameter_overrides"]["sha256"]
            )
            if photo and (
                execution.digest(photo) != inputs["scanner_observation"]["sha256"]
                or execution.digest(OBSERVER) != inputs["observer_sha256"]
            ):
                result.update(status="failed", error="scanner observation bindings changed")
            if not all(
                result[key]
                for key in (
                    "harness_unchanged",
                    "bundles_unchanged",
                    "frame_unchanged",
                    "scanner_frame_unchanged",
                    "overrides_unchanged",
                )
            ):
                result.update(status="failed", error="consumed inputs changed")
            result["members"] = inventory(directory)
            owner.publish(result)
        print(
            json.dumps(
                {"status": result["status"], "result": str(directory / "result.json"), "error": result.get("error")}
            )
        )
        return 0 if result["status"] == "passed" else 1


def collect(directory):
    result = json.loads((directory / "result.json").read_text())
    require(result["schema"] == "ofx-halation-comparison-v1", "comparison result schema")
    require(result["harness_unchanged"] is True and result["bundles_unchanged"] is True, "consumed inputs changed")
    require([command["name"] for command in result["commands"]] == ["build-probe", *SIDES], "command membership")
    require(
        all(
            command["returncode"] == 0 and command["drained"] is True and command["process_status"] == "completed"
            for command in result["commands"]
        ),
        "incomplete commands",
    )
    actual = inventory(directory)
    actual.pop("result.json")
    require(actual == result["members"], "retained member identity")
    inputs = json.loads((directory / "inputs.json").read_text())
    path_type = PureWindowsPath if result["preset"].startswith("windows-") else PurePosixPath
    producer_root = path_type(result["commands"][0]["directory"])

    def retained_path(recorded):
        path = Path(recorded)
        original = path_type(recorded)
        if original.is_absolute() and original.is_relative_to(producer_root):
            relative = original.relative_to(producer_root)
            require(".." not in relative.parts, "unconfined retained input")
            return ROOT.joinpath(*relative.parts)
        return path

    if "parameter_overrides" in inputs:
        override = inputs["parameter_overrides"]
        require(
            execution.digest(retained_path(override["path"])) == override["sha256"] and result["overrides_unchanged"],
            "override identity",
        )
    for key in ("print_scanner_observation", "scanner_observation"):
        if key in inputs:
            bound = inputs[key]
            require(
                execution.digest(retained_path(bound["path"])) == bound["sha256"],
                "scanner observation bindings changed",
            )
    if "frame" in inputs:
        frame = retained_path(inputs["frame"]["path"])
        require(inventory(frame) == inputs["frame"]["members"] and result["frame_unchanged"], "frame identity")
        comparison = compare_frame(directory, frame)
    elif "scanner_frame" in inputs:
        original = inputs["scanner_frame"]
        scanner_frame = retained_path(original["path"])
        require(
            {**scanner_frame_input(scanner_frame), "path": original["path"]} == original
            and result["scanner_frame_unchanged"],
            "native scanner frame identity",
        )
        comparison = compare_scanner_frame(
            directory,
            scanner_frame,
            strict_guards="reader_sha256" in inputs,
            scanner_xyz=inputs.get("scanner_xyz", False),
        )
    else:
        comparison = compare_captures(directory)
    retained = json.loads((directory / "comparison.json").read_text())
    print_scanner = retained.get("print_scanner", {})
    # Early input-only receipts used the matched-field label before live XYZ was captured.
    if print_scanner.get("claim") == "matched-field" and "pre_glare_xyz" not in print_scanner:
        print_scanner["claim"] = "matched-inputs"
    require(comparison == retained, "comparison changed")
    require(result["status"] == comparison["status"], "result status differs")
    print(json.dumps(comparison, indent=2))
    return 0 if comparison["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    capture = modes.add_parser("run", help="GPU: compare two supplied module identities without installing")
    capture.add_argument("--run-id", required=True)
    frames = capture.add_mutually_exclusive_group()
    frames.add_argument("--frame", type=Path, help="replay a retained actual OFX frame capture")
    frames.add_argument("--scanner-frame", type=Path, help="compare a native OFX frame with controlled grain and glare")
    capture.add_argument(
        "--parameter-overrides", type=Path, help="explicit counterfactual values; capture remains immutable"
    )
    photo = capture.add_mutually_exclusive_group()
    photo.add_argument("--print-scanner-observation", type=Path, help="matching symbols for materialized Print input")
    photo.add_argument("--current-scanner-observation", type=Path, help="matching symbols for seeded Ordinary Current")
    for side in SIDES:
        capture.add_argument(f"--{side}", required=True)
        capture.add_argument(f"--{side}-sha256", required=True)
    offline = modes.add_parser("collect", help="offline: verify retained members and recompute the comparison")
    offline.add_argument("directory", type=Path)
    args = parser.parse_args()
    return run(args) if args.mode == "run" else collect(args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
