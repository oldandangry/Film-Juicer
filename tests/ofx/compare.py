#!/usr/bin/env python3
"""Compare supplied production OFX modules with active legacy halation defaults."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
execution = importlib.import_module("execution")
verify = importlib.import_module("verify")

SIDES = ("reference", "candidate")
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


def run(args):
    preset = "windows-clang-release" if os.name == "nt" else "linux-release"
    build = ROOT / "out/build" / preset
    directory = ROOT / "out/validation" / preset / "results" / execution.name_component(args.run_id)
    with execution.OutputLease(ROOT, build), execution.OutputLease(ROOT):
        cache = verify.cache_values(build)
        require(
            cache["BUILD_TESTING"] == "OFF" and cache["CMAKE_BUILD_TYPE"] == "Release", "probe requires Release OFF"
        )
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
            "bundles": before,
            "modules": {
                side: {"path": str(module), "sha256": execution.digest(module)} for side, module in modules.items()
            },
            "build_testing": cache["BUILD_TESTING"],
            "build_type": cache["CMAKE_BUILD_TYPE"],
        }
        frame = args.frame.resolve(strict=True) if args.frame else None
        if frame:
            inputs["frame"] = {"path": str(frame), "members": inventory(frame)}
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
                if overrides:
                    command.extend(["--parameters", str(overrides)])
                owner.command(side, command, env=environment)
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
            result["overrides_unchanged"] = (
                overrides is None or execution.digest(overrides) == inputs["parameter_overrides"]["sha256"]
            )
            if not all(
                result[key]
                for key in ("harness_unchanged", "bundles_unchanged", "frame_unchanged", "overrides_unchanged")
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
    if "parameter_overrides" in inputs:
        override = inputs["parameter_overrides"]
        require(
            execution.digest(Path(override["path"])) == override["sha256"] and result["overrides_unchanged"],
            "override identity",
        )
    if "frame" in inputs:
        frame = Path(inputs["frame"]["path"])
        require(inventory(frame) == inputs["frame"]["members"] and result["frame_unchanged"], "frame identity")
        comparison = compare_frame(directory, frame)
    else:
        comparison = compare_captures(directory)
    require(comparison == json.loads((directory / "comparison.json").read_text()), "comparison changed")
    require(result["status"] == comparison["status"], "result status differs")
    print(json.dumps(comparison, indent=2))
    return 0 if comparison["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    capture = modes.add_parser("run", help="GPU: compare two supplied module identities without installing")
    capture.add_argument("--run-id", required=True)
    capture.add_argument("--frame", type=Path, help="replay a retained actual OFX frame capture")
    capture.add_argument(
        "--parameter-overrides", type=Path, help="explicit counterfactual values; capture remains immutable"
    )
    for side in SIDES:
        capture.add_argument(f"--{side}", required=True)
        capture.add_argument(f"--{side}-sha256", required=True)
    offline = modes.add_parser("collect", help="offline: verify retained members and recompute the comparison")
    offline.add_argument("directory", type=Path)
    args = parser.parse_args()
    return run(args) if args.mode == "run" else collect(args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
