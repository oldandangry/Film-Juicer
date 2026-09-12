#!/usr/bin/env python3
"""Stage, launch, and compare the focused gamma validation executable."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np


KNOWN_GROUPS = {
    "default-paper",
    "default-routes",
    "sampling-dir",
    "print-backend",
    "state",
    "routes",
    "lifetime",
    "performance",
    "all",
}

BASELINE_RUNS = (
    ("default-paper-1", "default-paper"),
    ("default-paper-2", "default-paper"),
    ("default-routes", "default-routes"),
    ("sampling-dir", "sampling-dir"),
    ("print-backend", "print-backend"),
    ("state", "state"),
    ("routes", "routes"),
    ("lifetime", "lifetime"),
    ("performance", "performance"),
)

VALIDATION_SOURCE_FILES = (
    "GammaValidation.cpp",
    "GammaValidation.cu",
    "GammaValidationCuda.h",
    "GammaValidation.vcxproj",
    "README.md",
    "generate_reference.py",
    "reference_issues.json",
    "run_validation.py",
    "test_generate_reference.py",
    "test_run_validation.py",
)

LEGACY_DIRECT_FUSED_CASE_IDS = {
    "negative-direct",
    "positive-direct",
    "dir-off-direct",
    "spatial-dir-ramp-edge-direct",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def validate_runtime_resource_inventory(inventory: dict, label: str) -> None:
    if not isinstance(inventory, dict) or not inventory:
        raise RuntimeError(f"{label}: runtime resource inventory is empty or malformed")
    for name, identity in inventory.items():
        if not isinstance(name, str) or not name or not isinstance(identity, dict):
            raise RuntimeError(f"{label}: malformed runtime resource entry")
        sha256 = identity.get("sha256")
        byte_count = identity.get("bytes")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or type(byte_count) is not int
            or byte_count < 0
        ):
            raise RuntimeError(f"{label}: malformed runtime resource identity: {name}")


def runtime_resource_inventory_differences(expected: dict, actual: dict) -> list[str]:
    differences = []
    for name in sorted(set(expected) - set(actual)):
        differences.append(f"missing {name}")
    for name in sorted(set(actual) - set(expected)):
        differences.append(f"extra {name}")
    for name in sorted(set(expected) & set(actual)):
        if expected[name] != actual[name]:
            differences.append(
                f"changed {name} expected={expected[name]!r} actual={actual[name]!r}"
            )
    return differences


def verify_runtime_resource_inventory(expected: dict, actual: dict, label: str) -> None:
    validate_runtime_resource_inventory(expected, f"{label} expected")
    validate_runtime_resource_inventory(actual, f"{label} actual")
    differences = runtime_resource_inventory_differences(expected, actual)
    if differences:
        raise RuntimeError(
            f"{label}: runtime resource inventory mismatch: "
            + "; ".join(differences)
        )


def choose_runtime_resource_inventory(sources: list[dict], label: str) -> dict:
    if not sources:
        raise RuntimeError(f"{label}: no verified frozen runtime resource source")
    selected = sources[0].get("inventory")
    validate_runtime_resource_inventory(selected, f"{label} {sources[0].get('path')}")
    for source in sources[1:]:
        inventory = source.get("inventory")
        validate_runtime_resource_inventory(
            inventory,
            f"{label} {source.get('path')}",
        )
        differences = runtime_resource_inventory_differences(selected, inventory)
        if differences:
            raise RuntimeError(
                f"{label}: frozen runtime resource reports disagree: "
                f"{sources[0].get('path')} versus {source.get('path')}: "
                + "; ".join(differences)
            )
    return selected


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_fresh_output_root(output_root: Path, fixture_root: Path, label: str) -> None:
    if output_root.exists():
        raise RuntimeError(f"{label} output already exists: {output_root}")
    if is_within(output_root, fixture_root):
        raise RuntimeError(f"{label} output must not be inside the fixture root")


def windows_path(path: Path) -> str:
    if os.name == "nt":
        return str(path.resolve())
    return subprocess.check_output(
        ["wslpath", "-w", str(path.resolve())],
        text=True,
    ).strip()


def load_reference_manifest(fixture_root: Path) -> dict:
    manifest_path = fixture_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing reference manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", manifest.get("case", {}).get("files", {}))
    for name, identity in files.items():
        path = fixture_root / "reference" / name
        if not path.is_file() or sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"reference fixture checksum mismatch: {path}")
    return manifest


def vector_metrics(actual, expected) -> dict:
    actual_values = np.asarray(actual, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    if actual_values.shape != expected_values.shape:
        raise RuntimeError(
            f"probe shape mismatch: {actual_values.shape} versus {expected_values.shape}"
        )
    if not np.isfinite(actual_values).all() or not np.isfinite(expected_values).all():
        raise RuntimeError("probe comparison contains NaN or infinity")
    delta = actual_values - expected_values
    absolute = np.abs(delta)
    worst_flat = int(np.argmax(absolute))
    worst = tuple(int(value) for value in np.unravel_index(worst_flat, absolute.shape))
    return {
        "maximum_absolute_error": float(absolute.reshape(-1)[worst_flat]),
        "rmse": float(math.sqrt(float(np.mean(delta * delta)))),
        "worst_location": worst,
        "actual_at_worst": float(actual_values.reshape(-1)[worst_flat]),
        "expected_at_worst": float(expected_values.reshape(-1)[worst_flat]),
    }


def component_metrics(actual, expected, labels: tuple[str, ...]) -> dict:
    actual_values = np.asarray(actual, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    if actual_values.shape != expected_values.shape:
        raise RuntimeError(
            f"component shape mismatch: {actual_values.shape} versus "
            f"{expected_values.shape}"
        )
    if actual_values.ndim < 1 or actual_values.shape[-1] != len(labels):
        raise RuntimeError(
            f"component data must end in {len(labels)} channels: "
            f"{actual_values.shape}"
        )
    return {
        label: vector_metrics(actual_values[..., index], expected_values[..., index])
        for index, label in enumerate(labels)
    }


def probe_metrics(capture: dict) -> dict:
    if capture["case"] == "sampling-dir":
        return {
            "curve_cases": {
                row["id"]: vector_metrics(
                    row["actual_float32"], row["expected_float64"]
                )
                for row in capture["curve_cases"]
            },
            "dir_cases": {
                row["id"]: {
                    "corrected_log_exposure_bgr": component_metrics(
                        row["actual_corrected_log_exposure_bgr_float32"],
                        row["expected_corrected_log_exposure_bgr_float64"],
                        ("B", "G", "R"),
                    ),
                    "final_density_bgr": component_metrics(
                        row["actual_final_density_bgr_float32"],
                        row["expected_final_density_bgr_float64"],
                        ("B", "G", "R"),
                    ),
                }
                for row in capture["dir_cases"]
            },
        }
    return {
        "cdf_cases": {
            row["id"]: {
                "float64": vector_metrics(
                    row["actual_float64"], row["expected_float64"]
                ),
                "float32_conversion": vector_metrics(
                    row["actual_float32"], row["expected_float32"]
                ),
            }
            for row in capture["cdf_cases"]
        },
        "sampler_cases": {
            row["id"]: vector_metrics(
                row["actual_float32"], row["expected_float64"]
            )
            for row in capture["sampler_cases"]
        },
    }


def metric_rows(actual: np.ndarray, expected: np.ndarray, labels: tuple[str, ...]) -> dict:
    if actual.shape != expected.shape:
        raise RuntimeError(f"shape mismatch: {actual.shape} versus {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError("comparison contains NaN or infinity")
    rows = {}
    for channel, label in enumerate(labels):
        delta = np.asarray(actual[..., channel], dtype=np.float64) - expected[..., channel]
        absolute = np.abs(delta)
        worst_flat = int(np.argmax(absolute))
        worst = tuple(int(value) for value in np.unravel_index(worst_flat, absolute.shape))
        rows[label] = {
            "maximum_absolute_error": float(absolute[worst]),
            "rmse": float(math.sqrt(float(np.mean(delta * delta)))),
            "worst_location_yx": worst,
            "actual_at_worst": float(actual[worst + (channel,)]),
            "expected_at_worst": float(expected[worst + (channel,)]),
        }
    return rows


def write_output_comparison_images(
    output_root: Path,
    identifier: str,
    actual: np.ndarray,
    expected: np.ndarray,
) -> dict:
    actual_values = np.asarray(actual, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    if (
        actual_values.shape != expected_values.shape
        or actual_values.ndim != 3
        or actual_values.shape[-1] != 3
    ):
        raise RuntimeError(
            f"output image shape mismatch: {actual_values.shape} versus "
            f"{expected_values.shape}"
        )
    if not np.isfinite(actual_values).all() or not np.isfinite(expected_values).all():
        raise RuntimeError("output image comparison contains NaN or infinity")

    image_root = output_root / "images" / identifier
    image_root.mkdir(parents=True)
    difference_scale = 16.0
    images = {
        "actual": actual_values,
        "expected": expected_values,
        "absolute_difference_x16":
            np.abs(actual_values - expected_values) * difference_scale,
    }
    identities = {}
    height, width, _ = actual_values.shape
    for name, values in images.items():
        path = image_root / f"{name}.ppm"
        pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
        path.write_bytes(
            f"P6\n{width} {height}\n255\n".encode("ascii") + pixels.tobytes()
        )
        identities[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return {
        "encoding": "binary PPM P6; clipped [0,1] RGB quantized to uint8",
        "shape": [height, width, 3],
        "difference_scale": difference_scale,
        "files": identities,
    }


def stage_runtime(
    executable: Path,
    resources: Path,
    runtime_root: Path,
    *,
    expected_resources: dict | None = None,
    runtime_label: str = "runtime",
) -> tuple[Path, Path, dict, dict]:
    if runtime_root.exists():
        raise RuntimeError(f"runtime destination already exists: {runtime_root}")
    source_pre_stage_sha256 = sha256_file(executable)
    binary_root = runtime_root / "bin"
    staged_resources = runtime_root / "Resources"
    binary_root.mkdir(parents=True)
    launched_executable = binary_root / executable.name
    shutil.copy2(executable, launched_executable)
    for dll in executable.parent.glob("*.dll"):
        shutil.copy2(dll, binary_root / dll.name)
    shutil.copytree(resources, staged_resources)
    resource_identity = tree_manifest(staged_resources)
    if expected_resources is not None:
        verify_runtime_resource_inventory(
            expected_resources,
            resource_identity,
            runtime_label,
        )
    source_pre_launch_sha256 = sha256_file(executable)
    staged_pre_launch_sha256 = sha256_file(launched_executable)
    if (
        source_pre_launch_sha256 != source_pre_stage_sha256
        or staged_pre_launch_sha256 != source_pre_stage_sha256
    ):
        raise RuntimeError(
            f"{runtime_label}: candidate executable changed during staging "
            f"source_pre_stage={source_pre_stage_sha256} "
            f"source_pre_launch={source_pre_launch_sha256} "
            f"staged_pre_launch={staged_pre_launch_sha256}"
        )
    executable_identity = {
        "source_path": str(executable),
        "source_pre_stage_sha256": source_pre_stage_sha256,
        "source_pre_launch_sha256": source_pre_launch_sha256,
        "staged_path": str(launched_executable),
        "staged_pre_launch_sha256": staged_pre_launch_sha256,
    }
    return (
        launched_executable,
        staged_resources,
        resource_identity,
        executable_identity,
    )


def verify_runtime_executable_after_run(
    source_executable: Path,
    launched_executable: Path,
    identity: dict,
    runtime_label: str,
) -> dict:
    source_post_run_sha256 = sha256_file(source_executable)
    staged_post_run_sha256 = sha256_file(launched_executable)
    expected = identity.get("source_pre_stage_sha256")
    if (
        not isinstance(expected, str)
        or source_post_run_sha256 != expected
        or staged_post_run_sha256 != expected
    ):
        raise RuntimeError(
            f"{runtime_label}: candidate executable identity changed "
            f"expected={expected} source_post_run={source_post_run_sha256} "
            f"staged_post_run={staged_post_run_sha256}"
        )
    return {
        **identity,
        "source_post_run_sha256": source_post_run_sha256,
        "staged_post_run_sha256": staged_post_run_sha256,
    }


def run_all_baseline(arguments: argparse.Namespace) -> int:
    fixture_root = arguments.fixture_root.resolve()
    output_root = arguments.output_root.resolve()
    validate_fresh_output_root(output_root, fixture_root, "baseline")
    manifest_before = sha256_file(fixture_root / "manifest.json")
    runs = {}
    for run_name, case_group in BASELINE_RUNS:
        child_arguments = argparse.Namespace(**vars(arguments))
        child_arguments.case_group = case_group
        child_arguments.output_root = output_root / run_name
        child_arguments.quiet = True
        status = run_baseline(child_arguments)
        if status != 0:
            return status
        report_path = child_arguments.output_root / "report.json"
        runs[run_name] = {
            "case_group": case_group,
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
        }

    first_capture = tree_manifest(output_root / "default-paper-1" / "capture")
    second_capture = tree_manifest(output_root / "default-paper-2" / "capture")
    first_numeric = {
        name: identity
        for name, identity in first_capture.items()
        if name != "capture.json"
    }
    second_numeric = {
        name: identity
        for name, identity in second_capture.items()
        if name != "capture.json"
    }
    if first_numeric != second_numeric:
        raise RuntimeError("repeated default-paper numerical captures are not identical")

    report = {
        "operation": getattr(arguments, "report_operation", "baseline"),
        "case_group": "all",
        "reference_manifest_sha256": manifest_before,
        "source_executable": str(arguments.exe.resolve()),
        "source_executable_sha256": sha256_file(arguments.exe.resolve()),
        "runs": runs,
        "default_paper_numeric_reproducible": True,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if sha256_file(fixture_root / "manifest.json") != manifest_before:
        raise RuntimeError("frozen reference manifest changed during all capture")
    if not getattr(arguments, "quiet", False):
        print(json.dumps({"report": str(report_path), "status": "captured"}))
    return 0


def run_baseline(arguments: argparse.Namespace) -> int:
    if arguments.case_group == "all":
        return run_all_baseline(arguments)
    fixture_root = arguments.fixture_root.resolve()
    output_root = arguments.output_root.resolve()
    validate_fresh_output_root(output_root, fixture_root, "baseline")
    manifest_before = sha256_file(fixture_root / "manifest.json")
    manifest = load_reference_manifest(fixture_root)
    case = manifest["case"]
    if arguments.case_group not in {
        "default-paper",
        "default-routes",
        "sampling-dir",
        "print-backend",
        "state",
        "routes",
        "lifetime",
        "performance",
    }:
        raise RuntimeError(f"baseline group is not implemented: {arguments.case_group}")

    expected_runtime = getattr(arguments, "expected_runtime_resources", None)
    runtime_label = getattr(
        arguments,
        "runtime_label",
        getattr(arguments, "report_operation", "baseline"),
    )
    launched_exe, staged_resources, staged_before, executable_identity = stage_runtime(
        arguments.exe.resolve(),
        arguments.resource_root.resolve(),
        output_root / "runtime",
        expected_resources=expected_runtime,
        runtime_label=runtime_label,
    )
    capture_root = output_root / "capture"
    capture_root.mkdir(parents=True)
    case_root = fixture_root / "reference" / case["id"]
    if arguments.case_group == "performance":
        input_path = fixture_root / "reference" / "performance" / "input_rgba_interleaved.f32"
        run_width = int(manifest["performance"]["width"])
        run_height = int(manifest["performance"]["height"])
    elif arguments.case_group in {
        "default-paper",
        "default-routes",
        "routes",
        "lifetime",
    }:
        input_path = case_root / "input_rgba_interleaved.f32"
        run_width = int(case["width"])
        run_height = int(case["height"])
    else:
        input_path = (
            fixture_root
            / "reference"
            / "probes"
            / f"{arguments.case_group.replace('-', '_')}.json"
        )
        run_width = int(case["width"])
        run_height = int(case["height"])
    command = [
        str(launched_exe),
        "--case-group",
        arguments.case_group,
        "--resource-root",
        windows_path(staged_resources),
        "--input",
        windows_path(input_path),
        "--output-dir",
        windows_path(capture_root),
        "--width",
        str(run_width),
        "--height",
        str(run_height),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    (output_root / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_root / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    executable_identity = verify_runtime_executable_after_run(
        arguments.exe.resolve(),
        launched_exe,
        executable_identity,
        runtime_label,
    )
    staged_after = tree_manifest(staged_resources)
    if staged_after != staged_before:
        raise RuntimeError("staged runtime resources changed during execution")
    if completed.returncode != 0:
        raise RuntimeError(
            f"GammaValidation failed with {completed.returncode}: {completed.stderr.strip()}"
        )

    capture = json.loads((capture_root / "capture.json").read_text(encoding="utf-8"))
    common_report = {
        "operation": getattr(arguments, "report_operation", "baseline"),
        "case_group": arguments.case_group,
        "reference_manifest_sha256": manifest_before,
        "source_executable": str(arguments.exe.resolve()),
        "source_executable_sha256": executable_identity[
            "source_pre_stage_sha256"
        ],
        "launched_executable": str(launched_exe),
        "launched_executable_sha256": executable_identity[
            "staged_post_run_sha256"
        ],
        "executable_provenance": executable_identity,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "staged_resource_root": str(staged_resources),
        "staged_resources": staged_after,
        "capture_files": tree_manifest(capture_root),
        "capture": capture,
        "command": command,
        "return_code": completed.returncode,
    }
    expected_runtime_provenance = getattr(
        arguments,
        "expected_runtime_resource_provenance",
        None,
    )
    if expected_runtime is not None:
        common_report["expected_staged_resources"] = {
            "inventory": expected_runtime,
            **(expected_runtime_provenance or {}),
        }
    if arguments.case_group in {"default-routes", "routes"}:
        route_metrics = {}
        route_images = {}
        height = int(case["height"])
        width = int(case["width"])
        route_cases = (
            manifest["default_routes"]
            if arguments.case_group == "default-routes"
            else manifest["routes_g09"]
        )
        reference_subdir = (
            "routes" if arguments.case_group == "default-routes" else "routes-g09"
        )
        for route in route_cases:
            identifier = route["id"]
            actual_root = capture_root / identifier
            expected_root = fixture_root / "reference" / reference_subdir / identifier
            film = np.moveaxis(
                np.fromfile(
                    actual_root / "film_density_cmy_planar.f32", dtype="<f4"
                ).reshape(3, height, width),
                0,
                -1,
            )
            linear = np.moveaxis(
                np.fromfile(
                    actual_root / "scanner_linear_rgb_planar.f32", dtype="<f4"
                ).reshape(3, height, width),
                0,
                -1,
            )
            output = np.fromfile(
                actual_root / "output_rgba_interleaved.f32", dtype="<f4"
            ).reshape(height, width, 4)
            expected_output = np.fromfile(
                expected_root / "output_rgb_clipped_interleaved.f64",
                dtype="<f8",
            ).reshape(height, width, 3)
            metrics = {
                "film_density": metric_rows(
                    film,
                    np.fromfile(
                        expected_root / "film_density_cmy_interleaved.f64",
                        dtype="<f8",
                    ).reshape(height, width, 3),
                    ("C", "M", "Y"),
                ),
                "scanner_linear_rgb": metric_rows(
                    linear,
                    np.fromfile(
                        expected_root / "scanner_linear_rgb_interleaved.f64",
                        dtype="<f8",
                    ).reshape(height, width, 3),
                    ("R", "G", "B"),
                ),
                "output_rgb": metric_rows(
                    output[..., :3],
                    expected_output,
                    ("R", "G", "B"),
                ),
            }
            if not route["scan_film"]:
                print_density = np.moveaxis(
                    np.fromfile(
                        actual_root / "print_density_cmy_planar.f32", dtype="<f4"
                    ).reshape(3, height, width),
                    0,
                    -1,
                )
                metrics["print_density"] = metric_rows(
                    print_density,
                    np.fromfile(
                        expected_root / "print_density_cmy_interleaved.f64",
                        dtype="<f8",
                    ).reshape(height, width, 3),
                    ("C", "M", "Y"),
                )
            route_metrics[identifier] = metrics
            route_images[identifier] = write_output_comparison_images(
                output_root,
                identifier,
                output[..., :3],
                expected_output,
            )
        common_report["metrics"] = route_metrics
        common_report["comparison_images"] = route_images
        report_path = output_root / "report.json"
        report_path.write_text(
            json.dumps(common_report, indent=2) + "\n", encoding="utf-8"
        )
        if sha256_file(fixture_root / "manifest.json") != manifest_before:
            raise RuntimeError("frozen reference manifest changed during route capture")
        if not getattr(arguments, "quiet", False):
            print(json.dumps({"report": str(report_path), "status": "captured"}))
        return 0
    if arguments.case_group in {"state", "lifetime"}:
        common_report["metrics"] = {
            "contract_assertions": capture,
        }
        report_path = output_root / "report.json"
        report_path.write_text(
            json.dumps(common_report, indent=2) + "\n", encoding="utf-8"
        )
        if sha256_file(fixture_root / "manifest.json") != manifest_before:
            raise RuntimeError("frozen reference manifest changed during contract capture")
        if not getattr(arguments, "quiet", False):
            print(json.dumps({"report": str(report_path), "status": "captured"}))
        return 0
    if arguments.case_group == "performance":
        samples = capture["samples"]
        preparation = [float(row["prepare_elapsed_ms"]) for row in samples]
        gpu = [float(row["fused_gpu_elapsed_ms"]) for row in samples]

        def timing_summary(values: list[float]) -> dict:
            return {
                "raw_ms": values,
                "median_ms": statistics.median(values),
                "minimum_ms": min(values),
                "maximum_ms": max(values),
                "spread_ms": max(values) - min(values),
            }

        common_report["metrics"] = {
            "cold_prepare_ms": float(capture["cold"]["prepare_elapsed_ms"]),
            "cold_fused_gpu_ms": float(capture["cold"]["fused_gpu_elapsed_ms"]),
            "warm_preparation": timing_summary(preparation),
            "fused_gpu_execution": timing_summary(gpu),
        }
        gamma_edit_samples = capture.get("gamma_edit_samples", {})
        if gamma_edit_samples:
            common_report["metrics"]["gamma_edit_preparation"] = {
                transition: {
                    "recipe_build": timing_summary(
                        [float(row["recipe_build_elapsed_ms"]) for row in rows]
                    ),
                    "prepare": timing_summary(
                        [float(row["prepare_elapsed_ms"]) for row in rows]
                    ),
                }
                for transition, rows in gamma_edit_samples.items()
            }
        report_path = output_root / "report.json"
        report_path.write_text(
            json.dumps(common_report, indent=2) + "\n", encoding="utf-8"
        )
        if sha256_file(fixture_root / "manifest.json") != manifest_before:
            raise RuntimeError("frozen reference manifest changed during timing capture")
        if not getattr(arguments, "quiet", False):
            print(json.dumps({"report": str(report_path), "status": "captured"}))
        return 0
    if arguments.case_group != "default-paper":
        common_report["metrics"] = probe_metrics(capture)
        report_path = output_root / "report.json"
        report_path.write_text(
            json.dumps(common_report, indent=2) + "\n", encoding="utf-8"
        )
        if sha256_file(fixture_root / "manifest.json") != manifest_before:
            raise RuntimeError("frozen reference manifest changed during probe capture")
        if not getattr(arguments, "quiet", False):
            print(json.dumps({"report": str(report_path), "status": "captured"}))
        return 0

    height = int(case["height"])
    width = int(case["width"])
    pixel_count = height * width
    film_planar = np.fromfile(
        capture_root / "film_density_cmy_planar.f32",
        dtype="<f4",
    ).reshape(3, height, width)
    print_planar = np.fromfile(
        capture_root / "print_density_cmy_planar.f32",
        dtype="<f4",
    ).reshape(3, height, width)
    linear_planar = np.fromfile(
        capture_root / "scanner_linear_rgb_planar.f32",
        dtype="<f4",
    ).reshape(3, height, width)
    output = np.fromfile(
        capture_root / "output_rgba_interleaved.f32",
        dtype="<f4",
    ).reshape(height, width, 4)
    if film_planar.size != pixel_count * 3 or print_planar.size != pixel_count * 3:
        raise RuntimeError("capture has an unexpected element count")
    film = np.moveaxis(film_planar, 0, -1)
    print_density = np.moveaxis(print_planar, 0, -1)
    linear = np.moveaxis(linear_planar, 0, -1)
    expected_film = np.fromfile(
        case_root / "film_density_cmy_interleaved.f64",
        dtype="<f8",
    ).reshape(height, width, 3)
    expected_print = np.fromfile(
        case_root / "print_density_cmy_interleaved.f64",
        dtype="<f8",
    ).reshape(height, width, 3)
    expected_linear = np.fromfile(
        case_root / "scanner_linear_rgb_interleaved.f64",
        dtype="<f8",
    ).reshape(height, width, 3)
    expected_output = np.fromfile(
        case_root / "output_rgb_clipped_interleaved.f64",
        dtype="<f8",
    ).reshape(height, width, 3)
    report = common_report
    report["metrics"] = {
            "film_density": metric_rows(film, expected_film, ("C", "M", "Y")),
            "print_density": metric_rows(print_density, expected_print, ("C", "M", "Y")),
            "scanner_linear_rgb": metric_rows(linear, expected_linear, ("R", "G", "B")),
            "output_rgb": metric_rows(output[..., :3], expected_output, ("R", "G", "B")),
    }
    report["comparison_images"] = write_output_comparison_images(
        output_root,
        "default-paper",
        output[..., :3],
        expected_output,
    )
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if sha256_file(fixture_root / "manifest.json") != manifest_before:
        raise RuntimeError("frozen reference manifest changed during baseline capture")
    if not getattr(arguments, "quiet", False):
        print(json.dumps({"report": str(report_path), "status": "captured"}))
    return 0


def run_route_probe(arguments: argparse.Namespace) -> int:
    fixture_root = arguments.fixture_root.resolve()
    output_root = arguments.output_root.resolve()
    candidate = bool(getattr(arguments, "candidate", False))
    operation = "candidate-route-probe" if candidate else "route-probe"
    validate_fresh_output_root(output_root, fixture_root, operation)
    baseline, baseline_sha256 = verify_frozen_baseline(fixture_root)
    route_probe_record = None
    route_probe_sha256 = None
    if candidate:
        route_probe_record, route_probe_sha256 = verify_frozen_route_probe(
            fixture_root
        )
    reference_manifest_sha256 = sha256_file(fixture_root / "manifest.json")
    manifest = load_reference_manifest(fixture_root)
    route_spec = manifest.get("route_probe", {})
    if route_spec.get("status") != (
        "inputs-and-upstream-frozen; limits-pending-phase-3a"
    ):
        raise RuntimeError("route-probe inputs are not in the frozen pending state")
    expected_runtime_provenance = None
    if candidate:
        expected_runtime_provenance = resolve_candidate_runtime_resources(
            fixture_root,
            baseline,
            arguments.exe.resolve().parent.name,
            "candidate-route-probe",
            route_probe_record,
        )

    print_backend_path = fixture_root / "reference" / "probes" / "print_backend.json"
    table_index = {}
    if not candidate:
        print_backend = load_json(print_backend_path)
        table_index = {
            (row["stock"], float(row["gamma"])): row
            for row in print_backend["sampler_cases"]
        }
    case = manifest["case"]
    chart_path = (
        fixture_root
        / "reference"
        / case["id"]
        / "input_rgba_interleaved.f32"
    )
    route_input = {
        "schema_version": 1,
        "chart_input": windows_path(chart_path),
        "reference_manifest_sha256": reference_manifest_sha256,
        "print_backend_sha256": sha256_file(print_backend_path),
        "cases": [],
    }
    for route in route_spec["cases"]:
        route_case = {
            "id": route["id"],
            "film_profile": route["film_profile"],
            "print_profile": route["print_profile"],
            "film_gamma": float(route["settings"]["film_gamma"]),
            "print_gamma": float(route["settings"]["print_gamma"]),
        }
        if not candidate:
            key = (
                route["print_profile"],
                float(route["settings"]["print_gamma"]),
            )
            table = table_index.get(key)
            if table is None:
                raise RuntimeError(
                    "missing frozen print table for route-probe "
                    f"{route['id']}: {key[0]} gamma {key[1]:.17g}"
                )
            route_case.update(
                {
                    "injected_print_table_id": table["id"],
                    "axis_float32": table["axis_float32"],
                    "density_cmy_float32": table["density_cmy_float32"],
                }
            )
        route_input["cases"].append(route_case)

    output_root.mkdir(parents=True)
    input_path = output_root / f"{operation}-input.json"
    input_path.write_text(json.dumps(route_input, indent=2) + "\n", encoding="utf-8")
    launched_exe, staged_resources, staged_before, executable_identity = stage_runtime(
        arguments.exe.resolve(),
        arguments.resource_root.resolve(),
        output_root / "runtime",
        expected_resources=(
            expected_runtime_provenance["inventory"]
            if expected_runtime_provenance is not None
            else None
        ),
        runtime_label=operation,
    )
    capture_root = output_root / "capture"
    capture_root.mkdir()
    command = [
        str(launched_exe),
        "--case-group",
        operation,
        "--resource-root",
        windows_path(staged_resources),
        "--input",
        windows_path(input_path),
        "--output-dir",
        windows_path(capture_root),
        "--width",
        str(int(case["width"])),
        "--height",
        str(int(case["height"])),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    (output_root / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_root / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    executable_identity = verify_runtime_executable_after_run(
        arguments.exe.resolve(),
        launched_exe,
        executable_identity,
        operation,
    )
    staged_after = tree_manifest(staged_resources)
    if staged_after != staged_before:
        raise RuntimeError(f"staged runtime resources changed during {operation}")
    if completed.returncode != 0:
        raise RuntimeError(
            f"GammaValidation failed with {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )

    capture = load_json(capture_root / "capture.json")
    if capture.get("case") != operation:
        raise RuntimeError(f"{operation} executable returned the wrong case")
    height = int(case["height"])
    width = int(case["width"])
    route_metrics = {}
    route_images = {}
    for route in route_spec["cases"]:
        identifier = route["id"]
        actual_root = capture_root / identifier
        expected_root = fixture_root / "reference" / "route-probe" / identifier
        film = np.moveaxis(
            np.fromfile(
                actual_root / "film_density_cmy_planar.f32", dtype="<f4"
            ).reshape(3, height, width),
            0,
            -1,
        )
        print_density = np.moveaxis(
            np.fromfile(
                actual_root / "print_density_cmy_planar.f32", dtype="<f4"
            ).reshape(3, height, width),
            0,
            -1,
        )
        linear = np.moveaxis(
            np.fromfile(
                actual_root / "scanner_linear_rgb_planar.f32", dtype="<f4"
            ).reshape(3, height, width),
            0,
            -1,
        )
        output = np.fromfile(
            actual_root / "output_rgba_interleaved.f32", dtype="<f4"
        ).reshape(height, width, 4)
        expected_output = np.fromfile(
            expected_root / "output_rgb_clipped_interleaved.f64",
            dtype="<f8",
        ).reshape(height, width, 3)
        route_metrics[identifier] = {
            "film_density": metric_rows(
                film,
                np.fromfile(
                    expected_root / "film_density_cmy_interleaved.f64",
                    dtype="<f8",
                ).reshape(height, width, 3),
                ("C", "M", "Y"),
            ),
            "print_density": metric_rows(
                print_density,
                np.fromfile(
                    expected_root / "print_density_cmy_interleaved.f64",
                    dtype="<f8",
                ).reshape(height, width, 3),
                ("C", "M", "Y"),
            ),
            "scanner_linear_rgb": metric_rows(
                linear,
                np.fromfile(
                    expected_root / "scanner_linear_rgb_interleaved.f64",
                    dtype="<f8",
                ).reshape(height, width, 3),
                ("R", "G", "B"),
            ),
            "output_rgb": metric_rows(
                output[..., :3],
                expected_output,
                ("R", "G", "B"),
            ),
        }
        route_images[identifier] = write_output_comparison_images(
            output_root,
            identifier,
            output[..., :3],
            expected_output,
        )

    repository_root = Path(__file__).resolve().parents[2]
    film_wiring_paths = (
        "src/JuicerState.h",
        "src/JuicerState.cpp",
        "src/RenderRecipe.h",
        "src/RenderRecipe.cpp",
    )
    downstream_owner_paths = (
        "src/Cuda/JuicerCudaFilmPayloads.cpp",
        "src/Cuda/JuicerCudaResources.cpp",
        "src/Cuda/Film/JuicerCudaFilmPipeline.cu",
        "src/Cuda/Scan/JuicerCudaScanPipeline.cu",
    )
    production_patch = subprocess.check_output(
        ["git", "diff", "--binary", "--", *film_wiring_paths],
        cwd=repository_root,
    )
    input_sources = {
        "chart": {"path": str(chart_path), "sha256": sha256_file(chart_path)},
    }
    if candidate:
        input_sources["frozen_route_probe"] = {
            "path": str(fixture_root / "route-probe.json"),
            "sha256": route_probe_sha256,
        }
    else:
        input_sources["print_backend"] = {
            "path": str(print_backend_path),
            "sha256": sha256_file(print_backend_path),
        }
    report = {
        "operation": operation,
        "case_group": operation,
        "reference_manifest_sha256": reference_manifest_sha256,
        "baseline_manifest_sha256": baseline_sha256,
        "source_executable": str(arguments.exe.resolve()),
        "source_executable_sha256": executable_identity[
            "source_pre_stage_sha256"
        ],
        "launched_executable": str(launched_exe),
        "launched_executable_sha256": executable_identity[
            "staged_post_run_sha256"
        ],
        "executable_provenance": executable_identity,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_sources": input_sources,
        "staged_resource_root": str(staged_resources),
        "staged_resources": staged_after,
        "capture_files": tree_manifest(capture_root),
        "capture": capture,
        "metrics": route_metrics,
        "comparison_images": route_images,
        "command": command,
        "return_code": completed.returncode,
        "film_wiring": {
            "revision": git_output(repository_root, "rev-parse", "HEAD"),
            "patch_sha256": hashlib.sha256(production_patch).hexdigest(),
            "files": {
                name: sha256_file(repository_root / name)
                for name in film_wiring_paths
            },
        },
        "unchanged_downstream_owners": {
            name: sha256_file(repository_root / name)
            for name in downstream_owner_paths
        },
        "validation_sources": {
            name: sha256_file(Path(__file__).resolve().parent / name)
            for name in VALIDATION_SOURCE_FILES
        },
    }
    if expected_runtime_provenance is not None:
        report["expected_staged_resources"] = expected_runtime_provenance
    if candidate:
        report["route_probe_manifest_sha256"] = route_probe_sha256
        failures = []
        compare_metric_tree(
            route_metrics,
            route_probe_record["shared_limits"],
            "candidate-route-probe",
            failures,
        )
        required_cases = set(route_probe_record["required_cases"])
        captured_cases = {route.get("case") for route in capture.get("routes", [])}
        if captured_cases != required_cases:
            failures.append(
                "candidate-route-probe: captured cases do not match the frozen gate"
            )
        for route in capture.get("routes", []):
            settings = route.get("settings", {})
            if route.get("reference_print_curves_injected") is not False:
                failures.append(
                    f"candidate-route-probe.{route.get('case')}: reference table was injected"
                )
            if route.get("print_gamma_recipe_factor") != settings.get("print_gamma"):
                failures.append(
                    f"candidate-route-probe.{route.get('case')}: print gamma did not reach the recipe"
                )
            if not route.get("print_develop_recipe_hash"):
                failures.append(
                    f"candidate-route-probe.{route.get('case')}: missing print development identity"
                )
        if capture.get("candidate_print_curves_are_recipe_owned") is not True:
            failures.append(
                "candidate-route-probe: recipe-owned curve contract was not reported"
            )
        fused_report, fused_failures = compare_fused_equivalence(
            "candidate-route-probe",
            arguments.exe.resolve().parent.name,
            capture,
            fixture_root,
            baseline,
            route_probe_record,
        )
        report["fused_equivalence"] = fused_report
        failures.extend(fused_failures)
        report["acceptance"] = {
            "status": "passed" if not failures else "failed",
            "failures": failures,
        }
    else:
        report["injection_contract"] = route_spec["injection"]
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _, baseline_after = verify_frozen_baseline(fixture_root)
    if baseline_after != baseline_sha256:
        raise RuntimeError(f"frozen baseline changed during {operation}")
    if sha256_file(fixture_root / "manifest.json") != reference_manifest_sha256:
        raise RuntimeError(f"frozen reference manifest changed during {operation}")
    if candidate:
        _, route_probe_after = verify_frozen_route_probe(fixture_root)
        if route_probe_after != route_probe_sha256:
            raise RuntimeError(
                "frozen route-probe record changed during candidate comparison"
            )
        print(json.dumps({"report": str(report_path), **report["acceptance"]}))
        return 0 if not report["acceptance"]["failures"] else 1
    print(json.dumps({"report": str(report_path), "status": "captured"}))
    return 0


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing JSON record: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifested_tree(root: Path, expected: dict, label: str) -> None:
    actual = tree_manifest(root)
    if actual != expected:
        raise RuntimeError(f"{label} checksum manifest does not match {root}")


def metric_ceilings(value):
    if isinstance(value, dict):
        if "maximum_absolute_error" in value and "rmse" in value:
            return {
                "maximum_absolute_error": float(value["maximum_absolute_error"]),
                "rmse": float(value["rmse"]),
            }
        return {key: metric_ceilings(child) for key, child in value.items()}
    raise RuntimeError("metric tree contains a non-object node")


def sampling_dir_limits(report: dict) -> dict:
    metrics = report["metrics"]
    curve_metrics = metrics["curve_cases"]
    measured_sampler_ids = (
        "endpoint-repeats",
        "single-sample",
        "nonuniform-factor-sweep",
    )
    sampler_bound = max(
        float(curve_metrics[identifier]["maximum_absolute_error"])
        for identifier in measured_sampler_ids
    )
    curve_limits = metric_ceilings(curve_metrics)
    curve_limits["internal-repeat-rightmost"] = {
        "maximum_absolute_error": sampler_bound,
        "rmse": sampler_bound,
    }

    operation_count = 16
    epsilon = float(np.finfo(np.float32).eps)
    gamma_n = operation_count * epsilon / (1.0 - operation_count * epsilon)
    fixture_magnitude_envelope = 64.0
    arithmetic_bound = gamma_n * fixture_magnitude_envelope
    dir_limits = {}
    for row in report["capture"]["dir_cases"]:
        dir_limits[row["id"]] = {
            boundary: {
                channel: {
                    "maximum_absolute_error": arithmetic_bound,
                    "rmse": arithmetic_bound,
                }
                for channel in ("B", "G", "R")
            }
            for boundary in (
                "corrected_log_exposure_bgr",
                "final_density_bgr",
            )
        }
    return {
        "curve_cases": curve_limits,
        "dir_cases": dir_limits,
        "derivation": {
            "right_biased_knot_limit": (
                "largest measured Float32 sampler error among the unchanged "
                "endpoint, single-sample, and nonuniform-factor probes"
            ),
            "right_biased_knot_maximum_absolute_error": sampler_bound,
            "dir_model": "gamma_n forward-error ceiling for Float32 operations",
            "dir_operation_count": operation_count,
            "float32_epsilon": epsilon,
            "fixture_magnitude_envelope": fixture_magnitude_envelope,
            "dir_absolute_ceiling": arithmetic_bound,
        },
    }


def limits_for_reports(reports: dict[str, dict]) -> dict:
    default_paper_metrics = reports["default-paper-1"]["metrics"]
    if reports["default-paper-2"]["metrics"] != default_paper_metrics:
        raise RuntimeError("repeated default-paper metrics differ during promotion")
    return {
        "default-paper": metric_ceilings(default_paper_metrics),
        "default-routes": metric_ceilings(reports["default-routes"]["metrics"]),
        "sampling-dir": sampling_dir_limits(reports["sampling-dir"]),
        "print-backend": metric_ceilings(reports["print-backend"]["metrics"]),
        "routes": metric_ceilings(reports["routes"]["metrics"]),
    }


def read_all_baseline_reports(
    output_root: Path,
    expected_configuration: str,
    reference_manifest_sha256: str,
) -> tuple[dict, dict[str, dict]]:
    top_report_path = output_root / "report.json"
    top_report = load_json(top_report_path)
    if top_report.get("operation") != "baseline" or top_report.get("case_group") != "all":
        raise RuntimeError(f"not an all-baseline record: {top_report_path}")
    if top_report.get("reference_manifest_sha256") != reference_manifest_sha256:
        raise RuntimeError(f"reference identity mismatch in {top_report_path}")
    source_executable = Path(top_report["source_executable"])
    if source_executable.parent.name != expected_configuration:
        raise RuntimeError(
            f"expected {expected_configuration} executable, got {source_executable}"
        )
    reports = {}
    for run_name, case_group in BASELINE_RUNS:
        report_path = output_root / run_name / "report.json"
        report = load_json(report_path)
        top_identity = top_report.get("runs", {}).get(run_name)
        if not top_identity or top_identity.get("case_group") != case_group:
            raise RuntimeError(f"missing all-baseline run identity: {run_name}")
        if sha256_file(report_path) != top_identity.get("report_sha256"):
            raise RuntimeError(f"all-baseline report checksum mismatch: {report_path}")
        if report.get("operation") != "baseline" or report.get("case_group") != case_group:
            raise RuntimeError(f"unexpected baseline report contents: {report_path}")
        if report.get("reference_manifest_sha256") != reference_manifest_sha256:
            raise RuntimeError(f"reference identity mismatch in {report_path}")
        if report.get("source_executable_sha256") != top_report.get(
            "source_executable_sha256"
        ):
            raise RuntimeError(f"executable identity mismatch in {report_path}")
        verify_manifested_tree(
            output_root / run_name / "capture",
            report["capture_files"],
            f"{expected_configuration} {run_name} capture",
        )
        reports[run_name] = report
    return top_report, reports


def git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository_root, text=True
    ).strip()


def promote_baseline(arguments: argparse.Namespace) -> int:
    fixture_root = arguments.fixture_root.resolve()
    baseline_root = fixture_root / "baseline"
    baseline_record_path = fixture_root / "baseline.json"
    promoting_root = fixture_root / "baseline.promoting"
    promoting_record_path = fixture_root / "baseline.promoting.json"
    for path in (
        baseline_root,
        baseline_record_path,
        promoting_root,
        promoting_record_path,
    ):
        if path.exists():
            raise RuntimeError(f"frozen baseline destination already exists: {path}")

    reference_manifest_path = fixture_root / "manifest.json"
    reference_manifest_sha256 = sha256_file(reference_manifest_path)
    load_reference_manifest(fixture_root)
    environment = load_json(arguments.environment_record.resolve())
    required_environment_fields = {
        "msbuild",
        "clang_cl",
        "msvc",
        "nvcc",
        "cuda",
        "gpu",
        "driver",
        "reference_environment_creation",
        "capture_sources",
    }
    missing_environment = sorted(required_environment_fields - environment.keys())
    if missing_environment:
        raise RuntimeError(
            "environment record is missing: " + ", ".join(missing_environment)
        )

    baseline_inputs = {
        "Release-Clang": arguments.clang_output_root.resolve(),
        "Release": arguments.release_output_root.resolve(),
    }
    top_reports = {}
    reports_by_configuration = {}
    for configuration, output_root in baseline_inputs.items():
        top_report, reports = read_all_baseline_reports(
            output_root,
            configuration,
            reference_manifest_sha256,
        )
        top_reports[configuration] = top_report
        reports_by_configuration[configuration] = reports

    for configuration, output_root in baseline_inputs.items():
        for run_name, case_group in BASELINE_RUNS:
            destination = promoting_root / configuration / run_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            capture_source = output_root / run_name / "capture"
            capture_destination = destination / "capture"
            if case_group in {
                "sampling-dir",
                "print-backend",
                "state",
                "lifetime",
                "performance",
            }:
                capture_destination.mkdir(parents=True)
                shutil.copy2(
                    capture_source / "capture.json",
                    capture_destination / "capture.json",
                )
            else:
                shutil.copytree(capture_source, capture_destination)
            promoted_report = dict(reports_by_configuration[configuration][run_name])
            promoted_report["capture"] = {
                "path": "capture/capture.json",
                "sha256": sha256_file(capture_destination / "capture.json"),
            }
            promoted_report_path = destination / "report.json"
            promoted_report_path.write_text(
                json.dumps(promoted_report, indent=2) + "\n", encoding="utf-8"
            )
        promoted_top_report = dict(top_reports[configuration])
        promoted_top_report["runs"] = {
            run_name: {
                "case_group": case_group,
                "report": f"{run_name}/report.json",
                "report_sha256": sha256_file(
                    promoting_root / configuration / run_name / "report.json"
                ),
            }
            for run_name, case_group in BASELINE_RUNS
        }
        (promoting_root / configuration / "report.json").write_text(
            json.dumps(promoted_top_report, indent=2) + "\n", encoding="utf-8"
        )

    clang_numeric = {
        name: identity
        for name, identity in tree_manifest(
            promoting_root / "Release-Clang" / "default-paper-1" / "capture"
        ).items()
        if name != "capture.json"
    }
    release_numeric = {
        name: identity
        for name, identity in tree_manifest(
            promoting_root / "Release" / "default-paper-1" / "capture"
        ).items()
        if name != "capture.json"
    }

    repository_root = Path(__file__).resolve().parents[2]
    production_paths = ("src", "Resources", "juicer.vcxproj", "juicer.sln")
    production_status = git_output(
        repository_root,
        "status",
        "--short",
        "--",
        *production_paths,
    )
    if production_status:
        raise RuntimeError(
            "production sources changed before the phase-1 baseline was frozen"
        )
    production_patch = subprocess.check_output(
        ["git", "diff", "--binary", "--", *production_paths],
        cwd=repository_root,
    )
    source_root = Path(__file__).resolve().parent
    source_identity = {
        name: {
            "sha256": sha256_file(source_root / name),
            "bytes": (source_root / name).stat().st_size,
        }
        for name in VALIDATION_SOURCE_FILES
    }
    for name, expected_sha256 in environment["capture_sources"].items():
        if name == "run_validation.py":
            continue
        path = source_root / name
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise RuntimeError(f"capture source identity no longer matches: {path}")

    promoting_root.replace(baseline_root)
    immutable_files = {
        name: identity
        for name, identity in tree_manifest(fixture_root).items()
        if name != "baseline.json"
    }
    record = {
        "schema_version": 1,
        "status": "phase-1-frozen",
        "reference_manifest": {
            "path": "manifest.json",
            "sha256": reference_manifest_sha256,
        },
        "film_juicer": {
            "revision": git_output(repository_root, "rev-parse", "HEAD"),
            "production_patch_sha256": hashlib.sha256(production_patch).hexdigest(),
            "production_status": "clean for frozen production paths",
        },
        "validation_sources": {
            "capture": environment["capture_sources"],
            "promotion": source_identity,
        },
        "environment": environment,
        "toolchains": {
            configuration: {
                "source_executable": top_reports[configuration]["source_executable"],
                "source_executable_sha256": top_reports[configuration][
                    "source_executable_sha256"
                ],
                "all_report": f"baseline/{configuration}/report.json",
                "all_report_sha256": sha256_file(
                    baseline_root / configuration / "report.json"
                ),
            }
            for configuration in baseline_inputs
        },
        "default_paper": {
            "repeat_count_per_toolchain": 2,
            "numerically_identical_within_each_toolchain": True,
            "numerically_identical_across_toolchains": clang_numeric
            == release_numeric,
        },
        "limits": {
            configuration: limits_for_reports(
                reports_by_configuration[configuration]
            )
            for configuration in baseline_inputs
        },
        "performance_baseline": {
            configuration: reports_by_configuration[configuration]["performance"][
                "metrics"
            ]
            for configuration in baseline_inputs
        },
        "required_runs": [name for name, _ in BASELINE_RUNS],
        "immutable_files": immutable_files,
    }
    promoting_record_path.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    promoting_record_path.replace(baseline_record_path)
    print(
        json.dumps(
            {
                "baseline": str(baseline_record_path),
                "sha256": sha256_file(baseline_record_path),
                "status": "phase-1-frozen",
            }
        )
    )
    return 0


def verify_frozen_baseline(fixture_root: Path) -> tuple[dict, str]:
    record_path = fixture_root / "baseline.json"
    record_sha256 = sha256_file(record_path)
    record = load_json(record_path)
    if record.get("status") != "phase-1-frozen":
        raise RuntimeError("phase-1 baseline is not frozen")
    reference = record["reference_manifest"]
    if sha256_file(fixture_root / reference["path"]) != reference["sha256"]:
        raise RuntimeError("frozen reference manifest checksum mismatch")
    for name, identity in record["immutable_files"].items():
        path = fixture_root / name
        if not path.is_file() or sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"frozen fixture checksum mismatch: {path}")
        if path.stat().st_size != identity["bytes"]:
            raise RuntimeError(f"frozen fixture size mismatch: {path}")
    return record, record_sha256


def combined_metric_ceilings(*values):
    if not values:
        raise RuntimeError("cannot combine an empty metric set")
    if all(
        isinstance(value, dict)
        and "maximum_absolute_error" in value
        and "rmse" in value
        for value in values
    ):
        return {
            "maximum_absolute_error": max(
                float(value["maximum_absolute_error"]) for value in values
            ),
            "rmse": max(float(value["rmse"]) for value in values),
        }
    keys = set(values[0])
    if any(not isinstance(value, dict) or set(value) != keys for value in values):
        raise RuntimeError("route-probe metric trees do not match")
    return {
        key: combined_metric_ceilings(*(value[key] for value in values))
        for key in sorted(keys)
    }


def read_route_probe_report(
    output_root: Path,
    expected_configuration: str,
    reference_manifest_sha256: str,
    baseline_sha256: str,
) -> dict:
    report_path = output_root / "report.json"
    report = load_json(report_path)
    if report.get("operation") != "route-probe" or report.get("case_group") != (
        "route-probe"
    ):
        raise RuntimeError(f"not a route-probe record: {report_path}")
    if report.get("reference_manifest_sha256") != reference_manifest_sha256:
        raise RuntimeError(f"reference identity mismatch in {report_path}")
    if report.get("baseline_manifest_sha256") != baseline_sha256:
        raise RuntimeError(f"baseline identity mismatch in {report_path}")
    source_executable = Path(report["source_executable"])
    if source_executable.parent.name != expected_configuration:
        raise RuntimeError(
            f"expected {expected_configuration} executable, got {source_executable}"
        )
    if report.get("source_executable_sha256") != report.get(
        "launched_executable_sha256"
    ):
        raise RuntimeError(f"staged executable identity mismatch in {report_path}")
    verify_manifested_tree(
        output_root / "capture",
        report["capture_files"],
        f"{expected_configuration} route-probe capture",
    )
    capture = report.get("capture", {})
    if (
        capture.get("case") != "route-probe"
        or capture.get("film_gamma_changes_balance_and_midgray") is not True
        or capture.get("injected_print_curves_are_validation_owned") is not True
    ):
        raise RuntimeError(
            f"route-probe semantic contract is incomplete in {report_path}"
        )
    wiring = capture.get("film_gamma_wiring", {})
    for name in (
        "supported_range_endpoints_retained",
        "out_of_range_values_rejected",
        "adjacent_float32_hashes_distinct",
    ):
        if wiring.get(name) is not True:
            raise RuntimeError(f"route-probe wiring check failed: {name}")
    for route in capture.get("routes", []):
        film_gamma = float(route["settings"]["film_gamma"])
        if (
            route.get("reference_print_curves_injected") is not True
            or int(route.get("injected_print_sample_count", 0)) != 256
            or route.get("film_gamma_recipe_rgb") != [film_gamma] * 3
            or route.get("film_gamma_payload_bgr") != [film_gamma] * 3
            or route.get("injected_print_sampling_gamma_cmy") != [1.0] * 3
            or route.get("print_balance_film_develop_recipe_hash")
            != route.get("film_develop_recipe_hash")
        ):
            raise RuntimeError(
                f"route-probe payload contract failed for {route.get('case')}"
            )
    return report


def promote_route_probe(arguments: argparse.Namespace) -> int:
    fixture_root = arguments.fixture_root.resolve()
    probe_root = fixture_root / "route-probe"
    record_path = fixture_root / "route-probe.json"
    promoting_root = fixture_root / "route-probe.promoting"
    promoting_record_path = fixture_root / "route-probe.promoting.json"
    for path in (probe_root, record_path, promoting_root, promoting_record_path):
        if path.exists():
            raise RuntimeError(f"frozen route-probe destination already exists: {path}")

    baseline, baseline_sha256 = verify_frozen_baseline(fixture_root)
    reference_manifest_sha256 = sha256_file(fixture_root / "manifest.json")
    if baseline["reference_manifest"]["sha256"] != reference_manifest_sha256:
        raise RuntimeError("phase-1 baseline and reference manifest disagree")
    inputs = {
        "Release-Clang": arguments.clang_output_root.resolve(),
        "Release": arguments.release_output_root.resolve(),
    }
    reports = {
        configuration: read_route_probe_report(
            output_root,
            configuration,
            reference_manifest_sha256,
            baseline_sha256,
        )
        for configuration, output_root in inputs.items()
    }
    clang_report = reports["Release-Clang"]
    release_report = reports["Release"]
    for field in (
        "input_sha256",
        "input_sources",
        "film_wiring",
        "unchanged_downstream_owners",
        "injection_contract",
    ):
        if clang_report[field] != release_report[field]:
            raise RuntimeError(f"route-probe toolchains disagree on {field}")
    if clang_report["metrics"] != release_report["metrics"]:
        raise RuntimeError("route-probe toolchains produced different metrics")

    for configuration, output_root in inputs.items():
        destination = promoting_root / configuration
        destination.mkdir(parents=True)
        shutil.copytree(output_root / "capture", destination / "capture")
        shutil.copy2(output_root / "route-probe-input.json", destination / "input.json")
        shutil.copy2(output_root / "report.json", destination / "report.json")

    clang_numeric = {
        name: identity
        for name, identity in tree_manifest(
            promoting_root / "Release-Clang" / "capture"
        ).items()
        if Path(name).name != "capture.json"
    }
    release_numeric = {
        name: identity
        for name, identity in tree_manifest(
            promoting_root / "Release" / "capture"
        ).items()
        if Path(name).name != "capture.json"
    }
    if clang_numeric != release_numeric:
        raise RuntimeError(
            "route-probe numerical captures differ across optimized toolchains"
        )

    promoting_root.replace(probe_root)
    source_root = Path(__file__).resolve().parent
    promotion_sources = {
        name: {
            "sha256": sha256_file(source_root / name),
            "bytes": (source_root / name).stat().st_size,
        }
        for name in VALIDATION_SOURCE_FILES
    }
    shared_limits = combined_metric_ceilings(
        clang_report["metrics"], release_report["metrics"]
    )
    record = {
        "schema_version": 1,
        "status": "phase-3a-route-probe-frozen",
        "reference_manifest": {
            "path": "manifest.json",
            "sha256": reference_manifest_sha256,
        },
        "phase_1_baseline": {
            "path": "baseline.json",
            "sha256": baseline_sha256,
        },
        "film_juicer": clang_report["film_wiring"],
        "injection_contract": clang_report["injection_contract"],
        "input_sources": clang_report["input_sources"],
        "unchanged_downstream_owners": clang_report[
            "unchanged_downstream_owners"
        ],
        "validation_sources": {
            "capture": clang_report["validation_sources"],
            "promotion": promotion_sources,
        },
        "toolchains": {
            configuration: {
                "source_executable": reports[configuration]["source_executable"],
                "source_executable_sha256": reports[configuration][
                    "source_executable_sha256"
                ],
                "report": f"route-probe/{configuration}/report.json",
                "report_sha256": sha256_file(
                    probe_root / configuration / "report.json"
                ),
            }
            for configuration in inputs
        },
        "limits": {
            configuration: metric_ceilings(reports[configuration]["metrics"])
            for configuration in inputs
        },
        "shared_limits": shared_limits,
        "limit_derivation": (
            "Per-case, per-boundary, per-channel maximum and RMSE measured from "
            "the pre-candidate independent-table route; shared ceilings are the "
            "larger optimized-toolchain values. Candidate print derivation did "
            "not exist when these limits were captured."
        ),
        "numerically_identical_across_toolchains": True,
        "required_cases": [
            route["id"]
            for route in load_reference_manifest(fixture_root)["route_probe"]["cases"]
        ],
        "immutable_files": tree_manifest(probe_root),
    }
    promoting_record_path.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    promoting_record_path.replace(record_path)
    print(
        json.dumps(
            {
                "route_probe": str(record_path),
                "sha256": sha256_file(record_path),
                "status": "phase-3a-route-probe-frozen",
            }
        )
    )
    return 0


def verify_frozen_route_probe(fixture_root: Path) -> tuple[dict, str]:
    record_path = fixture_root / "route-probe.json"
    if not record_path.is_file():
        raise RuntimeError("missing frozen phase-3A route-probe record")
    record_sha256 = sha256_file(record_path)
    record = load_json(record_path)
    if record.get("status") != "phase-3a-route-probe-frozen":
        raise RuntimeError("phase-3A route-probe is not frozen")
    if sha256_file(fixture_root / record["reference_manifest"]["path"]) != (
        record["reference_manifest"]["sha256"]
    ):
        raise RuntimeError("frozen route-probe reference identity mismatch")
    if sha256_file(fixture_root / record["phase_1_baseline"]["path"]) != (
        record["phase_1_baseline"]["sha256"]
    ):
        raise RuntimeError("frozen route-probe baseline identity mismatch")
    probe_root = fixture_root / "route-probe"
    verify_manifested_tree(
        probe_root,
        record["immutable_files"],
        "phase-3A route-probe",
    )
    return record, record_sha256


def resolved_fixture_child(
    base: Path,
    relative: str,
    fixture_root: Path,
    label: str,
) -> Path:
    path = (base / relative).resolve()
    if not is_within(path, fixture_root):
        raise RuntimeError(f"{label} escapes the fixture root: {path}")
    return path


def read_verified_baseline_report(
    fixture_root: Path,
    baseline: dict,
    configuration: str,
    run_name: str,
    expected_case_group: str,
) -> tuple[dict, dict]:
    toolchain = baseline.get("toolchains", {}).get(configuration)
    if not isinstance(toolchain, dict):
        raise RuntimeError(
            f"frozen baseline has no toolchain for configuration {configuration}"
        )
    top_path = resolved_fixture_child(
        fixture_root,
        toolchain.get("all_report", ""),
        fixture_root,
        f"{configuration} all-baseline report",
    )
    if not top_path.is_file() or sha256_file(top_path) != toolchain.get(
        "all_report_sha256"
    ):
        raise RuntimeError(f"frozen all-baseline report checksum mismatch: {top_path}")
    top = load_json(top_path)
    if top.get("case_group") != "all" or top.get("operation") != "baseline":
        raise RuntimeError(f"unexpected frozen all-baseline report: {top_path}")
    if Path(top.get("source_executable", "")).parent.name != configuration:
        raise RuntimeError(
            f"frozen all-baseline configuration mismatch: {top_path}"
        )
    run_identity = top.get("runs", {}).get(run_name)
    if not isinstance(run_identity, dict) or run_identity.get(
        "case_group"
    ) != expected_case_group:
        raise RuntimeError(
            f"missing frozen {configuration} {run_name} run identity"
        )
    report_path = resolved_fixture_child(
        top_path.parent,
        run_identity.get("report", ""),
        fixture_root,
        f"{configuration} {run_name} report",
    )
    if not report_path.is_file() or sha256_file(report_path) != run_identity.get(
        "report_sha256"
    ):
        raise RuntimeError(f"frozen baseline report checksum mismatch: {report_path}")
    report = load_json(report_path)
    if report.get("case_group") != expected_case_group or report.get(
        "operation"
    ) != "baseline":
        raise RuntimeError(f"unexpected frozen baseline report: {report_path}")
    if Path(report.get("source_executable", "")).parent.name != configuration:
        raise RuntimeError(f"frozen baseline configuration mismatch: {report_path}")
    if report.get("source_executable_sha256") != report.get(
        "launched_executable_sha256"
    ):
        raise RuntimeError(f"frozen baseline executable mismatch: {report_path}")
    validate_runtime_resource_inventory(
        report.get("staged_resources"),
        f"frozen baseline report {report_path}",
    )
    return report, {
        "path": str(report_path),
        "sha256": run_identity["report_sha256"],
    }


def read_verified_baseline_capture(
    fixture_root: Path,
    baseline: dict,
    configuration: str,
    run_name: str,
    expected_case_group: str,
) -> tuple[dict, dict]:
    report, report_source = read_verified_baseline_report(
        fixture_root,
        baseline,
        configuration,
        run_name,
        expected_case_group,
    )
    report_path = Path(report_source["path"])
    capture_reference = report.get("capture")
    if not isinstance(capture_reference, dict) or not isinstance(
        capture_reference.get("path"), str
    ) or not isinstance(capture_reference.get("sha256"), str):
        raise RuntimeError(
            f"frozen baseline report has no verified capture reference: {report_path}"
        )
    capture_path = resolved_fixture_child(
        report_path.parent,
        capture_reference["path"],
        fixture_root,
        f"{configuration} {run_name} capture",
    )
    if not capture_path.is_file() or sha256_file(capture_path) != capture_reference[
        "sha256"
    ]:
        raise RuntimeError(f"frozen baseline capture checksum mismatch: {capture_path}")
    return load_json(capture_path), {
        "path": str(capture_path),
        "sha256": capture_reference["sha256"],
    }


def read_verified_route_probe_report(
    fixture_root: Path,
    route_probe_record: dict,
    configuration: str,
) -> tuple[dict, dict]:
    toolchain = route_probe_record.get("toolchains", {}).get(configuration)
    if not isinstance(toolchain, dict):
        raise RuntimeError(
            f"frozen route-probe has no toolchain for configuration {configuration}"
        )
    report_path = resolved_fixture_child(
        fixture_root,
        toolchain.get("report", ""),
        fixture_root,
        f"{configuration} route-probe report",
    )
    if not report_path.is_file() or sha256_file(report_path) != toolchain.get(
        "report_sha256"
    ):
        raise RuntimeError(f"frozen route-probe report checksum mismatch: {report_path}")
    report = load_json(report_path)
    if report.get("operation") != "route-probe" or report.get(
        "case_group"
    ) != "route-probe":
        raise RuntimeError(f"unexpected frozen route-probe report: {report_path}")
    if Path(report.get("source_executable", "")).parent.name != configuration:
        raise RuntimeError(f"frozen route-probe configuration mismatch: {report_path}")
    if report.get("source_executable_sha256") != report.get(
        "launched_executable_sha256"
    ):
        raise RuntimeError(f"frozen route-probe executable mismatch: {report_path}")
    validate_runtime_resource_inventory(
        report.get("staged_resources"),
        f"frozen route-probe report {report_path}",
    )
    return report, {
        "path": str(report_path),
        "sha256": toolchain["report_sha256"],
    }


def read_verified_route_probe_capture(
    fixture_root: Path,
    route_probe_record: dict,
    configuration: str,
) -> tuple[dict, dict]:
    report, report_source = read_verified_route_probe_report(
        fixture_root,
        route_probe_record,
        configuration,
    )
    report_path = Path(report_source["path"])
    capture = report.get("capture")
    capture_files = report.get("capture_files")
    if not isinstance(capture, dict) or not isinstance(capture_files, dict):
        raise RuntimeError(f"frozen route-probe capture is malformed: {report_path}")
    capture_root = report_path.parent / "capture"
    verify_manifested_tree(
        capture_root,
        capture_files,
        f"{configuration} frozen route-probe capture",
    )
    capture_identity = capture_files.get("capture.json")
    capture_path = capture_root / "capture.json"
    if not isinstance(capture_identity, dict) or load_json(capture_path) != capture:
        raise RuntimeError(
            f"frozen route-probe embedded capture mismatch: {capture_path}"
        )
    return capture, {
        "path": str(capture_path.resolve()),
        "sha256": capture_identity.get("sha256"),
    }


def resolve_candidate_runtime_resources(
    fixture_root: Path,
    baseline: dict,
    configuration: str,
    case_group: str,
    route_probe_record: dict | None = None,
) -> dict:
    baseline_configurations = baseline.get("toolchains", {})
    if configuration in baseline_configurations:
        configurations = (configuration,)
    elif case_group in {"state", "lifetime"}:
        configurations = tuple(sorted(baseline_configurations))
    else:
        raise RuntimeError(
            f"{case_group}: no frozen runtime resources for configuration "
            f"{configuration}"
        )
    if not configurations:
        raise RuntimeError("frozen baseline has no runtime-resource toolchain")

    sources = []
    authoritative_paths = set()
    if case_group == "candidate-route-probe":
        if route_probe_record is None or configuration not in baseline_configurations:
            raise RuntimeError(
                "candidate-route-probe requires a matching frozen optimized toolchain"
            )
        route_report, route_source = read_verified_route_probe_report(
            fixture_root,
            route_probe_record,
            configuration,
        )
        source = {
            **route_source,
            "configuration": configuration,
            "case_group": "route-probe",
            "role": "authoritative",
            "inventory": route_report["staged_resources"],
        }
        sources.append(source)
        authoritative_paths.add(source["path"])

    for frozen_configuration in configurations:
        for run_name, frozen_case_group in BASELINE_RUNS:
            report, report_source = read_verified_baseline_report(
                fixture_root,
                baseline,
                frozen_configuration,
                run_name,
                frozen_case_group,
            )
            authoritative = False
            if case_group == "default-paper":
                authoritative = frozen_case_group == "default-paper"
            elif case_group in {"state", "lifetime"}:
                authoritative = run_name == "default-paper-1"
            elif case_group != "candidate-route-probe":
                authoritative = (
                    frozen_case_group == case_group
                    and frozen_configuration == configuration
                )
            source = {
                **report_source,
                "configuration": frozen_configuration,
                "case_group": frozen_case_group,
                "run_name": run_name,
                "role": "authoritative" if authoritative else "agreement-check",
                "inventory": report["staged_resources"],
            }
            sources.append(source)
            if authoritative:
                authoritative_paths.add(source["path"])

    if not authoritative_paths:
        raise RuntimeError(
            f"{case_group}: no authoritative frozen runtime-resource report"
        )
    inventory = choose_runtime_resource_inventory(
        sources,
        f"{case_group} {configuration}",
    )
    public_sources = [
        {name: value for name, value in source.items() if name != "inventory"}
        for source in sources
    ]
    return {
        "inventory": inventory,
        "authoritative_sources": [
            source for source in public_sources if source["role"] == "authoritative"
        ],
        "agreement_sources": [
            source for source in public_sources if source["role"] == "agreement-check"
        ],
    }


def fused_observation(record: dict, label: str, historical: bool) -> tuple[str, bool, float | None]:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label}: fused capture is not an object")
    case_id = record.get("case")
    if not isinstance(case_id, str) or not case_id:
        raise RuntimeError(f"{label}: missing case")
    if "fused_path_applicable" in record:
        applicable = record["fused_path_applicable"]
        if type(applicable) is not bool:
            raise RuntimeError(f"{label}: fused_path_applicable is not Boolean")
    elif historical and case_id in LEGACY_DIRECT_FUSED_CASE_IDS:
        applicable = True
    else:
        raise RuntimeError(f"{label}: missing fused_path_applicable")
    difference = record.get("fused_maximum_absolute_difference")
    if not applicable:
        if not historical and difference is not None:
            raise RuntimeError(
                f"{label}: non-applicable fused difference must be null"
            )
        if historical and difference is not None:
            try:
                historical_difference = float(difference)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"{label}: malformed historical non-applicable difference"
                ) from error
            if not math.isfinite(historical_difference):
                raise RuntimeError(
                    f"{label}: historical non-applicable difference is nonfinite"
                )
        return case_id, False, None
    if difference is None:
        raise RuntimeError(f"{label}: missing fused maximum difference")
    try:
        measured = float(difference)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label}: malformed fused maximum difference") from error
    if not math.isfinite(measured) or measured < 0.0:
        raise RuntimeError(
            f"{label}: fused maximum difference must be finite and nonnegative"
        )
    return case_id, True, measured


def compare_fused_case(
    candidate: dict,
    frozen: dict,
    expected_case: str,
    frozen_sources: list[dict],
    label: str,
) -> tuple[dict, list[str]]:
    result = {
        "case": expected_case,
        "frozen_sources": frozen_sources,
    }
    failures = []
    try:
        candidate_case, candidate_applicable, candidate_difference = fused_observation(
            candidate, f"{label}.candidate", False
        )
        frozen_case, frozen_applicable, frozen_difference = fused_observation(
            frozen, f"{label}.frozen", True
        )
        if candidate_case != expected_case:
            failures.append(
                f"{label}: candidate case {candidate_case!r}, expected {expected_case!r}"
            )
        if frozen_case != expected_case:
            failures.append(
                f"{label}: frozen case {frozen_case!r}, expected {expected_case!r}"
            )
        if candidate_applicable != frozen_applicable:
            failures.append(
                f"{label}: fused applicability changed from {frozen_applicable} "
                f"to {candidate_applicable}"
            )
        result["candidate_applicable"] = candidate_applicable
        result["frozen_applicable"] = frozen_applicable
        if candidate_applicable and frozen_applicable:
            result["candidate_difference"] = candidate_difference
            result["frozen_bound"] = frozen_difference
            if candidate_difference > frozen_difference:
                failures.append(
                    f"{label}: fused difference {candidate_difference:.17g}, "
                    f"frozen bound {frozen_difference:.17g}"
                )
            result["status"] = "passed" if not failures else "failed"
        elif not candidate_applicable and not frozen_applicable:
            result["candidate_difference"] = None
            result["frozen_bound"] = None
            result["status"] = "not-applicable" if not failures else "failed"
        else:
            result["status"] = "failed"
    except RuntimeError as error:
        failures.append(str(error))
        result["status"] = "failed"
    result["failures"] = failures
    return result, failures


def indexed_route_captures(capture: dict, expected_outer_case: str, label: str) -> dict:
    if capture.get("case") != expected_outer_case:
        raise RuntimeError(
            f"{label}: outer case {capture.get('case')!r}, expected {expected_outer_case!r}"
        )
    routes = capture.get("routes")
    if not isinstance(routes, list) or not routes:
        raise RuntimeError(f"{label}: missing route captures")
    indexed = {}
    for route in routes:
        case_id = route.get("case") if isinstance(route, dict) else None
        if not isinstance(case_id, str) or not case_id or case_id in indexed:
            raise RuntimeError(f"{label}: malformed or duplicate route case")
        indexed[case_id] = route
    return indexed


def compare_fused_equivalence(
    case_group: str,
    configuration: str,
    capture: dict,
    fixture_root: Path,
    baseline: dict,
    route_probe_record: dict | None = None,
) -> tuple[dict, list[str]]:
    report = {
        "configuration": configuration,
        "case_group": case_group,
        "comparisons": [],
        "unbounded_candidate_only": [],
    }
    failures = []
    if case_group == "default-paper":
        first, first_source = read_verified_baseline_capture(
            fixture_root, baseline, configuration, "default-paper-1", case_group
        )
        second, second_source = read_verified_baseline_capture(
            fixture_root, baseline, configuration, "default-paper-2", case_group
        )
        first_observation = fused_observation(
            first, "default-paper first frozen repetition", True
        )
        second_observation = fused_observation(
            second, "default-paper second frozen repetition", True
        )
        if first_observation != second_observation:
            raise RuntimeError(
                "default-paper frozen split/fused repetitions are inconsistent"
            )
        comparison, case_failures = compare_fused_case(
            capture,
            first,
            "default-paper",
            [first_source, second_source],
            "default-paper",
        )
        report["comparisons"].append(comparison)
        failures.extend(case_failures)
    elif case_group in {"default-routes", "routes"}:
        frozen, source = read_verified_baseline_capture(
            fixture_root, baseline, configuration, case_group, case_group
        )
        candidate_routes = indexed_route_captures(
            capture, case_group, f"{case_group}.candidate"
        )
        frozen_routes = indexed_route_captures(
            frozen, case_group, f"{case_group}.frozen"
        )
        if set(candidate_routes) != set(frozen_routes):
            raise RuntimeError(
                f"{case_group}: candidate route cases do not match frozen evidence"
            )
        for case_id in frozen_routes:
            comparison, case_failures = compare_fused_case(
                candidate_routes[case_id],
                frozen_routes[case_id],
                case_id,
                [source],
                f"{case_group}.{case_id}",
            )
            report["comparisons"].append(comparison)
            failures.extend(case_failures)
    elif case_group == "candidate-route-probe":
        if route_probe_record is None:
            raise RuntimeError("candidate route-probe comparison has no frozen record")
        frozen, source = read_verified_route_probe_capture(
            fixture_root, route_probe_record, configuration
        )
        candidate_routes = indexed_route_captures(
            capture, "candidate-route-probe", "candidate-route-probe.candidate"
        )
        frozen_routes = indexed_route_captures(
            frozen, "route-probe", "candidate-route-probe.frozen"
        )
        if set(candidate_routes) != set(frozen_routes):
            raise RuntimeError(
                "candidate-route-probe: candidate cases do not match frozen evidence"
            )
        for case_id in frozen_routes:
            comparison, case_failures = compare_fused_case(
                candidate_routes[case_id],
                frozen_routes[case_id],
                case_id,
                [source],
                f"candidate-route-probe.{case_id}",
            )
            report["comparisons"].append(comparison)
            failures.extend(case_failures)
    elif case_group == "performance":
        frozen, source = read_verified_baseline_capture(
            fixture_root, baseline, configuration, "performance", case_group
        )
        if capture.get("case") != "performance" or frozen.get("case") != "performance":
            raise RuntimeError("performance capture case mismatch")
        paired = [("cold", capture.get("cold"), frozen.get("cold"))]
        for name in ("warmups", "samples"):
            candidate_rows = capture.get(name)
            frozen_rows = frozen.get(name)
            if not isinstance(candidate_rows, list) or not isinstance(
                frozen_rows, list
            ) or len(candidate_rows) != len(frozen_rows):
                raise RuntimeError(f"performance {name} do not match frozen evidence")
            paired.extend(
                (f"{name}[{index}]", candidate_row, frozen_row)
                for index, (candidate_row, frozen_row) in enumerate(
                    zip(candidate_rows, frozen_rows)
                )
            )
        for selector, candidate_row, frozen_row in paired:
            if not isinstance(candidate_row, dict) or not isinstance(frozen_row, dict):
                raise RuntimeError(f"performance {selector} capture is malformed")
            expected_case = frozen_row.get("case")
            comparison, case_failures = compare_fused_case(
                candidate_row,
                frozen_row,
                expected_case,
                [{**source, "selector": selector}],
                f"performance.{selector}",
            )
            report["comparisons"].append(comparison)
            failures.extend(case_failures)
        gamma_edit_samples = capture.get("gamma_edit_samples", {})
        if gamma_edit_samples:
            if not isinstance(gamma_edit_samples, dict):
                raise RuntimeError("performance gamma-edit samples are malformed")
            for transition, rows in gamma_edit_samples.items():
                if not isinstance(rows, list):
                    raise RuntimeError(
                        f"performance gamma-edit transition {transition} is malformed"
                    )
                for index, row in enumerate(rows):
                    case_id, applicable, difference = fused_observation(
                        row,
                        f"performance.gamma_edit_samples.{transition}[{index}]",
                        False,
                    )
                    report["unbounded_candidate_only"].append(
                        {
                            "transition": transition,
                            "index": index,
                            "case": case_id,
                            "applicable": applicable,
                            "difference": difference,
                            "status": "finite-unbounded",
                        }
                    )
    else:
        report["status"] = "not-applicable"
        return report, failures
    report["status"] = "passed" if not failures else "failed"
    return report, failures


def compare_metric_tree(actual, limits, path: str, failures: list[str]) -> None:
    if not isinstance(actual, dict) or not isinstance(limits, dict):
        failures.append(f"{path}: malformed metric tree")
        return
    if "maximum_absolute_error" in limits and "rmse" in limits:
        for metric_name in ("maximum_absolute_error", "rmse"):
            if metric_name not in actual:
                failures.append(f"{path}: missing {metric_name}")
                continue
            measured = float(actual[metric_name])
            ceiling = float(limits[metric_name])
            if not math.isfinite(measured) or measured > ceiling:
                failures.append(
                    f"{path}.{metric_name}: measured {measured:.17g}, "
                    f"ceiling {ceiling:.17g}"
                )
        return
    for name, child_limits in limits.items():
        if name == "derivation":
            continue
        if name not in actual:
            failures.append(f"{path}: missing {name}")
            continue
        compare_metric_tree(actual[name], child_limits, f"{path}.{name}", failures)


def exact_settings_failures(
    settings: object,
    required: dict,
    path: str,
) -> list[str]:
    if not isinstance(settings, dict):
        return [f"{path}: missing effective settings"]
    failures = []
    for name, expected in required.items():
        if settings.get(name) != expected:
            failures.append(
                f"{path}.{name}: measured {settings.get(name)!r}, "
                f"expected {expected!r}"
            )
    return failures


def load_historical_reference_issues(
    fixture_root: Path,
    reference_manifest_sha256: str,
) -> tuple[dict, str]:
    issue_path = Path(__file__).resolve().with_name("reference_issues.json")
    record = load_json(issue_path)
    record_sha256 = sha256_file(issue_path)
    if record.get("schema_version") != 1:
        raise RuntimeError("historical reference issue record has an unsupported schema")
    manifest_binding = record.get("reference_manifest", {})
    if manifest_binding != {
        "path": "fixtures/manifest.json",
        "sha256": reference_manifest_sha256,
    }:
        raise RuntimeError(
            "historical reference issue record does not bind the active manifest"
        )
    if sha256_file(fixture_root / "manifest.json") != reference_manifest_sha256:
        raise RuntimeError("active reference manifest changed while reading issues")

    expected_boundaries = {
        "enlarger-diffusion-black-pro-mist-half": {
            "film_density",
            "print_density",
            "scanner_linear_rgb",
            "output_rgb",
        },
        "spatial-dir-ramp-edge-direct": {
            "film_density",
            "scanner_linear_rgb",
            "output_rgb",
        },
    }
    suffixes = {
        "film_density": "film_density_cmy_interleaved.f64",
        "print_density": "print_density_cmy_interleaved.f64",
        "scanner_linear_rgb": "scanner_linear_rgb_interleaved.f64",
        "output_rgb": "output_rgb_clipped_interleaved.f64",
    }
    issues = record.get("issues")
    if not isinstance(issues, list) or {
        issue.get("case_id") for issue in issues if isinstance(issue, dict)
    } != set(expected_boundaries):
        raise RuntimeError("historical reference issue record has the wrong case inventory")
    for issue in issues:
        case_id = issue["case_id"]
        if issue.get("acceptance_status") != "configuration-invalid":
            raise RuntimeError(
                f"historical reference issue {case_id} has the wrong status"
            )
        affected = issue.get("affected_comparisons")
        if not isinstance(affected, list) or set(affected) != expected_boundaries[case_id]:
            raise RuntimeError(
                f"historical reference issue {case_id} has the wrong boundary inventory"
            )
        evidence = issue.get("capture_evidence", {})
        for boundary in affected:
            fixture_path = (
                fixture_root
                / "reference"
                / "routes-g09"
                / case_id
                / suffixes[boundary]
            )
            if evidence.get(boundary) != sha256_file(fixture_path):
                raise RuntimeError(
                    f"historical reference issue {case_id} no longer matches "
                    f"the frozen {boundary} capture"
                )
    return record, record_sha256


def historical_reference_validity(
    case_group: str,
    fixture_root: Path,
    reference_manifest_sha256: str,
    metrics: dict,
) -> tuple[dict, list[str]]:
    if case_group != "routes":
        return {"status": "not-applicable", "invalid_cases": []}, []
    record, record_sha256 = load_historical_reference_issues(
        fixture_root,
        reference_manifest_sha256,
    )
    failures = []
    invalid_cases = []
    for issue in record["issues"]:
        case_id = issue["case_id"]
        case_metrics = metrics.get(case_id)
        if not isinstance(case_metrics, dict):
            raise RuntimeError(
                f"historical configuration-invalid case is missing metrics: {case_id}"
            )
        affected = issue["affected_comparisons"]
        missing_boundaries = [
            boundary for boundary in affected if boundary not in case_metrics
        ]
        if missing_boundaries:
            raise RuntimeError(
                f"historical configuration-invalid case {case_id} is missing "
                f"boundaries: {missing_boundaries}"
            )
        invalid_cases.append(
            {
                "case_id": case_id,
                "status": "configuration-invalid",
                "reason": issue["reason"],
                "affected_comparisons": affected,
                "intended_effective_settings": issue[
                    "intended_effective_settings"
                ],
                "historical_effective_settings": issue[
                    "historical_effective_settings"
                ],
            }
        )
        failures.append(
            f"routes.{case_id}: configuration-invalid frozen reference; "
            "numerical metrics are diagnostic only"
        )
    return (
        {
            "status": "failed",
            "record": str(
                Path(__file__).resolve().with_name("reference_issues.json")
            ),
            "record_sha256": record_sha256,
            "reference_manifest_sha256": reference_manifest_sha256,
            "invalid_cases": invalid_cases,
        },
        failures,
    )


def contract_failures(case_group: str, capture: dict) -> list[str]:
    failures = []
    if case_group == "state":
        required = {
            "direct_unused_print_edit_reused_build": True,
            "invalid_blocks_old_publication": True,
            "retained_admitted_direct_state_survived": True,
            "uninitialized_status": 0,
        }
        for name, expected in required.items():
            if capture.get(name) != expected:
                failures.append(
                    f"state.{name}: measured {capture.get(name)!r}, expected {expected!r}"
                )
        gamma_assignment = capture.get("gamma_snapshot_assignment", {})
        for name in (
            "authored_film_double_validated_before_float32",
            "authored_range_endpoints_retained",
            "equal_retained_film_identity_reused",
            "distinct_retained_film_identity_changed",
            "near_one_print_identity_distinct",
            "direct_unused_invalid_print_retained",
            "route_switch_rejects_invalid_print",
        ):
            if gamma_assignment.get(name) is not True:
                failures.append(
                    f"state.gamma_snapshot_assignment.{name}: expected true"
                )
    elif case_group == "lifetime":
        abort = capture.get("abort_and_retire", {})
        for name in ("aborted_frame_inactive", "idle_context_retired"):
            if abort.get(name) is not True:
                failures.append(f"lifetime.abort_and_retire.{name}: expected true")
        if "recovered_print" not in capture or "direct_switch" not in capture:
            failures.append("lifetime: missing route-switch recovery captures")
        transitions = capture.get("print_table_transitions", {})
        for name in (
            "queued_a_finished_before_sync",
            "queued_a_old_table_sample_verified",
            "queued_b_new_table_sample_verified",
            "equal_size_storage_reused",
            "round_trip_1_to_1_25_to_1_verified",
            "different_size_storage_replaced",
            "abort_recovery_verified",
            "stock_balance_invariant",
            "stock_preflash_invariant",
        ):
            if transitions.get(name) is not True:
                failures.append(f"lifetime.print_table_transitions.{name}: expected true")
        preflash_cache = capture.get("preflash_host_cache", {})
        for name in (
            "host_spectrum_valid_after_every_preparation",
            "host_spectrum_exactly_stock_anchored",
            "preflash_hashes_raw_and_storage_invariant",
            "creative_table_identity_transition_verified",
            "device_spectrum_matches_host_after_every_preparation",
            "local_owner_drained",
            "ledger_zero_after_destroy",
        ):
            if preflash_cache.get(name) is not True:
                failures.append(
                    f"lifetime.preflash_host_cache.{name}: expected true"
                )
        for route_name in ("initial_print", "recovered_print"):
            if capture.get(route_name, {}).get("reference_print_curves_injected") is not False:
                failures.append(f"lifetime.{route_name}: reference table was injected")
    elif case_group == "default-paper":
        failures.extend(
            exact_settings_failures(
                capture.get("settings"),
                {
                    "dir_active": True,
                    "dir_diffusion_size_um": 0.0,
                    "spatial_dir_active": False,
                    "scatter_active": False,
                    "halation_active": False,
                    "camera_diffusion_active": False,
                    "enlarger_diffusion_active": False,
                    "scanner_black_correction": False,
                    "scanner_white_correction": False,
                    "scanner_lens_blur_sigma_px": 0.0,
                    "scanner_unsharp_sigma_px": 0.0,
                    "scanner_unsharp_amount": 0.0,
                },
                "default-paper.settings",
            )
        )
    elif case_group == "routes":
        route_index = {
            route.get("case"): route
            for route in capture.get("routes", [])
            if isinstance(route, dict)
        }
        correction = route_index.get("scanner-black-white-correction-on", {})
        diffusion = route_index.get("enlarger-diffusion-black-pro-mist-half", {})
        spatial_dir = route_index.get("spatial-dir-ramp-edge-direct", {})
        common_inactive = {
            "dir_active": True,
            "dir_diffusion_size_um": 0.0,
            "spatial_dir_active": False,
            "scatter_active": False,
            "halation_active": False,
            "camera_diffusion_active": False,
            "enlarger_diffusion_active": False,
            "scanner_lens_blur_sigma_px": 0.0,
            "scanner_unsharp_sigma_px": 0.0,
            "scanner_unsharp_amount": 0.0,
        }
        failures.extend(
            exact_settings_failures(
                correction.get("settings"),
                {
                    **common_inactive,
                    "scanner_black_correction": True,
                    "scanner_white_correction": True,
                },
                "routes.scanner-black-white-correction-on.settings",
            )
        )
        failures.extend(
            exact_settings_failures(
                diffusion.get("settings"),
                {
                    **common_inactive,
                    "enlarger_diffusion_active": True,
                    "enlarger_diffusion_family": "black_pro_mist",
                    "enlarger_diffusion_scatter_fraction": 0.35 * 0.75,
                    "enlarger_diffusion_spatial_scale": 1.0,
                    "scanner_black_correction": False,
                    "scanner_white_correction": False,
                },
                "routes.enlarger-diffusion-black-pro-mist-half.settings",
            )
        )
        failures.extend(
            exact_settings_failures(
                spatial_dir.get("settings"),
                {
                    **common_inactive,
                    "dir_diffusion_size_um": 2187.5,
                    "spatial_dir_active": True,
                    "scanner_black_correction": False,
                    "scanner_white_correction": False,
                },
                "routes.spatial-dir-ramp-edge-direct.settings",
            )
        )
        required = {
            "enlarger_diffusion_active": True,
            "print_log_exposure_boundary": "print_development_input",
            "print_log_exposure_source": "post_diffusion_planes",
            "print_development_observer_exact": True,
            "print_development_observer_padded_stride_verified": True,
            "fused_path_applicable": False,
            "fused_maximum_absolute_difference": None,
        }
        for name, expected in required.items():
            if diffusion.get(name) != expected:
                failures.append(
                    "routes.enlarger-diffusion-black-pro-mist-half."
                    f"{name}: measured {diffusion.get(name)!r}, expected {expected!r}"
                )
    return failures


def run_compare(arguments: argparse.Namespace) -> int:
    fixture_root = arguments.fixture_root.resolve()
    output_root = arguments.output_root.resolve()
    validate_fresh_output_root(output_root, fixture_root, "candidate")
    baseline, baseline_sha256 = verify_frozen_baseline(fixture_root)
    if arguments.case_group == "all":
        _, route_probe_sha256 = verify_frozen_route_probe(fixture_root)
        failures = []
        runs = {}
        for run_name, case_group in BASELINE_RUNS:
            child_arguments = argparse.Namespace(**vars(arguments))
            child_arguments.case_group = case_group
            child_arguments.output_root = output_root / run_name
            child_arguments.quiet = True
            status = run_compare(child_arguments)
            runs[run_name] = {
                "case_group": case_group,
                "status": status,
                "report": str(child_arguments.output_root / "report.json"),
            }
            if status != 0:
                failures.append(run_name)
        candidate_route_arguments = argparse.Namespace(**vars(arguments))
        candidate_route_arguments.output_root = output_root / "candidate-route-probe"
        candidate_route_arguments.candidate = True
        candidate_route_status = run_route_probe(candidate_route_arguments)
        runs["candidate-route-probe"] = {
            "case_group": "candidate-route-probe",
            "status": candidate_route_status,
            "report": str(candidate_route_arguments.output_root / "report.json"),
        }
        if candidate_route_status != 0:
            failures.append("candidate-route-probe")
        report = {
            "operation": "compare",
            "case_group": "all",
            "baseline_manifest_sha256": baseline_sha256,
            "route_probe_manifest_sha256": route_probe_sha256,
            "runs": runs,
            "acceptance": {
                "status": "passed" if not failures else "failed",
                "failed_runs": failures,
            },
        }
        report_path = output_root / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        verify_frozen_baseline(fixture_root)
        verify_frozen_route_probe(fixture_root)
        if not getattr(arguments, "quiet", False):
            print(json.dumps({"report": str(report_path), **report["acceptance"]}))
        return 0 if not failures else 1

    configuration = arguments.exe.resolve().parent.name
    configuration_limits = baseline["limits"].get(configuration)
    if configuration_limits is None and arguments.case_group not in {
        "state",
        "lifetime",
    }:
        raise RuntimeError(
            f"candidate executable configuration has no frozen limits: {configuration}"
        )
    runtime_provenance = resolve_candidate_runtime_resources(
        fixture_root,
        baseline,
        configuration,
        arguments.case_group,
    )
    capture_arguments = argparse.Namespace(**vars(arguments))
    capture_arguments.report_operation = "compare"
    capture_arguments.quiet = True
    capture_arguments.expected_runtime_resources = runtime_provenance["inventory"]
    capture_arguments.expected_runtime_resource_provenance = {
        "authoritative_sources": runtime_provenance["authoritative_sources"],
        "agreement_sources": runtime_provenance["agreement_sources"],
    }
    capture_arguments.runtime_label = f"compare {arguments.case_group}"
    capture_status = run_baseline(capture_arguments)
    if capture_status != 0:
        return capture_status
    report_path = output_root / "report.json"
    report = load_json(report_path)
    failures = []
    if configuration_limits is not None and arguments.case_group in configuration_limits:
        compare_metric_tree(
            report["metrics"],
            configuration_limits[arguments.case_group],
            arguments.case_group,
            failures,
        )
    elif arguments.case_group in {"state", "lifetime"}:
        pass
    elif arguments.case_group != "performance":
        failures.append(f"no frozen acceptance rule for {arguments.case_group}")
    failures.extend(contract_failures(arguments.case_group, report["capture"]))
    reference_validity, reference_failures = historical_reference_validity(
        arguments.case_group,
        fixture_root,
        sha256_file(fixture_root / "manifest.json"),
        report["metrics"],
    )
    report["historical_reference_validity"] = reference_validity
    failures.extend(reference_failures)
    fused_report, fused_failures = compare_fused_equivalence(
        arguments.case_group,
        configuration,
        report["capture"],
        fixture_root,
        baseline,
    )
    report["fused_equivalence"] = fused_report
    failures.extend(fused_failures)
    report["baseline_manifest_sha256"] = baseline_sha256
    report["acceptance"] = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _, baseline_after = verify_frozen_baseline(fixture_root)
    if baseline_after != baseline_sha256:
        raise RuntimeError("frozen baseline record changed during candidate comparison")
    if not getattr(arguments, "quiet", False):
        print(json.dumps({"report": str(report_path), **report["acceptance"]}))
    return 0 if not failures else 1


def run_reference(arguments: argparse.Namespace) -> int:
    generator = Path(__file__).with_name("generate_reference.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(generator),
            "--reference-root",
            str(arguments.reference_root.resolve()),
            "--fixture-root",
            str(arguments.fixture_root.resolve()),
        ]
    )
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    reference = subparsers.add_parser("reference")
    reference.add_argument("--reference-root", type=Path, required=True)
    reference.add_argument("--fixture-root", type=Path, required=True)
    reference.set_defaults(handler=run_reference)

    for operation in ("baseline", "compare"):
        command = subparsers.add_parser(operation)
        command.add_argument("--exe", type=Path, required=True)
        command.add_argument("--resource-root", type=Path, required=True)
        command.add_argument("--fixture-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--case-group", choices=sorted(KNOWN_GROUPS), required=True)
        command.set_defaults(
            handler=run_baseline if operation == "baseline" else run_compare
        )

    promote = subparsers.add_parser("promote-baseline")
    promote.add_argument("--fixture-root", type=Path, required=True)
    promote.add_argument("--clang-output-root", type=Path, required=True)
    promote.add_argument("--release-output-root", type=Path, required=True)
    promote.add_argument("--environment-record", type=Path, required=True)
    promote.set_defaults(handler=promote_baseline)

    promote_probe = subparsers.add_parser("promote-route-probe")
    promote_probe.add_argument("--fixture-root", type=Path, required=True)
    promote_probe.add_argument("--clang-output-root", type=Path, required=True)
    promote_probe.add_argument("--release-output-root", type=Path, required=True)
    promote_probe.set_defaults(handler=promote_route_probe)

    route_probe = subparsers.add_parser("route-probe")
    route_probe.add_argument("--exe", type=Path, required=True)
    route_probe.add_argument("--resource-root", type=Path, required=True)
    route_probe.add_argument("--fixture-root", type=Path, required=True)
    route_probe.add_argument("--output-root", type=Path, required=True)
    route_probe.set_defaults(handler=run_route_probe, candidate=False)
    candidate_route_probe = subparsers.add_parser("candidate-route-probe")
    candidate_route_probe.add_argument("--exe", type=Path, required=True)
    candidate_route_probe.add_argument("--resource-root", type=Path, required=True)
    candidate_route_probe.add_argument("--fixture-root", type=Path, required=True)
    candidate_route_probe.add_argument("--output-root", type=Path, required=True)
    candidate_route_probe.set_defaults(handler=run_route_probe, candidate=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        return int(arguments.handler(arguments))
    except Exception as error:
        print(f"fatal: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
