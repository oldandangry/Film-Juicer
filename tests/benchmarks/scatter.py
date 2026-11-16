#!/usr/bin/env python3
"""Collect retained matched scatter precision characterization."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
execution = importlib.import_module("execution")
VARIANTS = ("existing", "fp32_direct", "fp32_factored")
HIGHLIGHT_VARIANTS = ("production", "fp32_direct", "fp32_stable")
HIGHLIGHT_TARGET = "JuicerHighlightBoostPrecisionTests"
HIGHLIGHT_FIXTURE_ROOT = ROOT / "tests/scatter_halation/parity-fixtures"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory(path):
    return {
        item.relative_to(path).as_posix(): execution.digest(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and item != path / "result.json"
    }


def regular_inventory(path):
    """Hash a fixture tree while rejecting links and non-regular entries."""
    result = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("linked fixture member: " + str(item))
        if item.is_file():
            result[item.relative_to(path).as_posix()] = execution.digest(item)
        elif not item.is_dir():
            raise ValueError("unsupported fixture member: " + str(item))
    return result


def source_identity(source, patch):
    projection = {key: source.get(key) for key in ("head", "inputs", "new_files_base64")}
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded + b"\0" + patch).hexdigest()


def native_source_snapshot(*, domain_tests, literal_name="tests/benchmarks/scatter.py"):
    verify = importlib.import_module("verify")
    source, patch, _ = verify.capture_native_source(ROOT, literal_name, domain_tests=domain_tests)
    return source, patch


def read_build_binding(path, preset, current_source):
    """Require the completed prepare bundle for this target and source snapshot."""
    path = path.resolve(strict=True)
    if path.name != "result.json":
        raise ValueError("prepare receipt must name its result.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema") != "native-build-v1" or result.get("status") != "passed":
        raise ValueError("build result is not a completed successful verify prepare")
    if result.get("preset") != preset or HIGHLIGHT_TARGET not in result.get("targets", []):
        raise ValueError("build result preset or target does not match this run")
    if not result.get("commands") or any(
        command.get("returncode") != 0
        or command.get("drained") is not True
        or command.get("process_status") != "completed"
        for command in result["commands"]
    ):
        raise ValueError("build result has failed or undrained commands")
    qualification = importlib.import_module("qualification")
    summary = qualification.read_native_result(path.parent)
    if summary.get("status") != "passed" or summary.get("preset") != preset:
        raise ValueError("build result does not validate as a completed prepare bundle")
    for name, expected in result.get("members", {}).items():
        member = path.parent / name
        if not member.is_file() or member.is_symlink() or execution.digest(member) != expected.get("sha256"):
            raise ValueError("build result member identity mismatch: " + name)
    candidate_path = path.parent / "candidate.json"
    patch_path = path.parent / "candidate.patch"
    if not candidate_path.is_file() or not patch_path.is_file():
        raise ValueError("build result source evidence is missing")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("preset") != preset:
        raise ValueError("build candidate preset mismatch")
    config = candidate.get("configuration", {})
    expected_type = "Debug" if preset.endswith("-debug") else "Release"
    if config.get("BUILD_TESTING") != "ON" or config.get("CMAKE_BUILD_TYPE") != expected_type:
        raise ValueError("probe build type or testing configuration does not match preset")

    # `verify prepare` captures public code, including test code. Match that view;
    # the broader domain snapshot additionally retains non-code fixture identities.
    verify = importlib.import_module("verify")
    build_view, build_patch, _ = verify.capture_native_source(ROOT, "scripts/verify.py", domain_tests=False)
    if source_identity(candidate, patch_path.read_bytes()) != source_identity(build_view, build_patch):
        raise ValueError("prepared build source does not match the current product candidate")
    if candidate.get("head") != current_source.get("head"):
        raise ValueError("prepared build and benchmark source have different HEAD revisions")
    return result, candidate, patch_path.read_bytes()


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _accuracy_case_ids(supplement):
    manifest = json.loads((HIGHLIGHT_FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    fixture_ids = {row["id"] for row in manifest["rows"] if "parameters" in row and "boost_ev" in row["parameters"]}
    supplement_rows = supplement.get("rows")
    if not isinstance(supplement_rows, list) or not supplement_rows:
        raise ValueError("highlight supplement must contain explicit rows")
    supplement_ids = {row.get("id") for row in supplement_rows}
    if None in supplement_ids or len(supplement_ids) != len(supplement_rows):
        raise ValueError("highlight supplement has missing or duplicate row IDs")
    if fixture_ids & supplement_ids:
        raise ValueError("highlight supplement duplicates a frozen fixture ID")
    return manifest, sorted(fixture_ids | supplement_ids)


def validate_highlight_accuracy(report, case_ids, *, production=False):
    schema = "highlight-production-accuracy.v1" if production else "highlight-precision-accuracy.v1"
    require(report.get("schema") == schema, "highlight accuracy schema")
    variants = ("production",) if production else HIGHLIGHT_VARIANTS
    rows = report.get("rows")
    expected = {(case, variant, padding) for case in case_ids for variant in variants for padding in (0, 7)}
    keys = [(row.get("id"), row.get("variant"), row.get("padding")) for row in rows or []]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "highlight accuracy membership")
    failure_counts = {variant: 0 for variant in variants}
    for row in rows:
        require(type(row.get("pass")) is bool, "highlight accuracy pass flag")
        for field in ("max_absolute", "max_relative", "rms"):
            require(_finite_number(row.get(field)) and row[field] >= 0, "invalid highlight accuracy metric: " + field)
        require(type(row.get("max_ulp")) is int and row["max_ulp"] >= 0, "invalid highlight ULP metric")
        require(
            type(row.get("tolerance_failures")) is int and row["tolerance_failures"] >= 0,
            "invalid highlight tolerance failure count",
        )
        require(type(row.get("samples")) is int and row["samples"] >= 0, "invalid highlight sample count")
        if production:
            require(
                type(row.get("finite_samples")) is int and 0 <= row["finite_samples"] <= row["samples"],
                "invalid finite sample count",
            )
            require(type(row.get("nonfinite_count")) is int and row["nonfinite_count"] >= 0, "invalid nonfinite count")
            require(row.get("metrics_scope") == "finite_samples_only", "invalid metric scope")
            require(row.get("tolerance_pass") is (row["tolerance_failures"] == 0), "tolerance counter mismatch")
            require(isinstance(row.get("first_failing_sample"), (dict, type(None))), "invalid failing sample")
            if row["nonfinite_count"]:
                require(
                    not row.get("pass") and isinstance(row.get("first_failing_sample"), dict),
                    "unreported nonfinite sample",
                )
        for flag in (
            "status_pass",
            "maximum_pass",
            "repeat_pass",
            "layout_pass",
            "exact_pass",
            "finite_pass",
            "monotonic_pass",
            "tolerance_pass",
        ):
            require(type(row.get(flag)) is bool, "missing highlight failure counter: " + flag)
        require(
            row["pass"]
            is all(
                row[flag]
                for flag in (
                    "status_pass",
                    "maximum_pass",
                    "repeat_pass",
                    "layout_pass",
                    "exact_pass",
                    "finite_pass",
                    "monotonic_pass",
                    "tolerance_pass",
                )
            ),
            "highlight row pass disagrees with failure counters",
        )
        require(isinstance(row.get("worst_sample"), (str, dict, type(None))), "invalid worst sample identity")
        if not row["pass"]:
            failure_counts[row["variant"]] += 1
    variant_pass = {variant: failures == 0 for variant, failures in failure_counts.items()}
    production_pass = variant_pass["production"]
    require(report.get("production_pass") is production_pass, "production accuracy summary mismatch")
    if production:
        require(production_pass, "production accuracy failed")
        return {"production_pass": production_pass, "failure_counts": failure_counts, "variant_pass": variant_pass}
    experimental_pass = all(variant_pass[name] for name in HIGHLIGHT_VARIANTS[1:])
    require(report.get("experimental_pass") is experimental_pass, "experimental accuracy summary mismatch")
    if not production_pass:
        raise ValueError("baseline production accuracy failed")
    return {
        "production_pass": production_pass,
        "experimental_pass": experimental_pass,
        "failure_counts": failure_counts,
        "variant_pass": variant_pass,
    }


def validate_highlight_timing(report, width, height, density):
    require(report.get("schema") == "highlight-precision-timing.v1", "highlight timing schema")
    require(
        (report.get("width"), report.get("height"), report.get("density")) == (width, height, density),
        "highlight timing input mismatch",
    )
    require(report.get("warmups") == 10 and report.get("samples") == 30, "highlight timing counts")
    require(
        isinstance(report.get("source_algorithm"), str) and report["source_algorithm"],
        "highlight timing source algorithm",
    )
    descriptor = report.get("descriptor")
    require(
        isinstance(descriptor, dict) and set(descriptor) == {"recipe_hash", "protection", "sharpness", "peak_gain"},
        "highlight timing descriptor",
    )
    require(
        type(descriptor["recipe_hash"]) is int and descriptor["recipe_hash"] != 0, "invalid highlight timing recipe"
    )
    require(
        all(
            _finite_number(descriptor[key]) and descriptor[key] > 0 for key in ("protection", "sharpness", "peak_gain")
        ),
        "invalid highlight timing controls",
    )
    pairs = report.get("pairs")
    require(isinstance(pairs, list) and len(pairs) == 3, "highlight timing pair count")
    summary = []
    variant_pass = {variant: True for variant in HIGHLIGHT_VARIANTS}
    for pair_index, pair in enumerate(pairs):
        order = "reverse" if pair_index == 1 else "forward"
        require(pair.get("pair") == pair_index and pair.get("order") == order, "highlight timing pair/order membership")
        variants = pair.get("variants")
        require(
            isinstance(variants, list)
            and [row.get("variant") for row in variants]
            == (list(HIGHLIGHT_VARIANTS) if order == "forward" else list(reversed(HIGHLIGHT_VARIANTS))),
            "highlight timing variant order",
        )
        for row in variants:
            require(
                type(row.get("finite_pass")) is bool and type(row.get("baseline_pass")) is bool,
                "invalid highlight timing correctness flags",
            )
            require(
                type(row.get("baseline_failures")) is int and row["baseline_failures"] >= 0,
                "invalid highlight timing failure count",
            )
            require(
                row["baseline_pass"] is (row["baseline_failures"] == 0),
                "highlight timing baseline flag disagrees with failure count",
            )
            variant_pass[row["variant"]] &= row["finite_pass"] and row["baseline_pass"]
            require(
                _finite_number(row.get("max_absolute")) and row["max_absolute"] >= 0, "invalid highlight timing error"
            )
            first = row.get("first_sample")
            require(
                isinstance(first, dict)
                and all(
                    _finite_number(first.get(key)) and first[key] >= 0 for key in ("cuda_ms", "host_ms", "wait_ms")
                ),
                "invalid highlight timing cold sample",
            )
            registers = row.get("registers")
            require(
                isinstance(registers, dict) and set(registers) == {"partial", "final", "apply"},
                "highlight kernel register fields",
            )
            require(
                all(type(value) is int and value >= 0 for value in registers.values()),
                "invalid highlight kernel register count",
            )
            report_row = {
                "pair": pair_index,
                "order": order,
                "variant": row["variant"],
                "finite_pass": row["finite_pass"],
                "baseline_pass": row["baseline_pass"],
                "baseline_failures": row["baseline_failures"],
                "max_absolute": row["max_absolute"],
                "registers": registers,
                "first_sample": first,
            }
            for name in ("cuda_stage", "host_launch_readback_wait", "wait"):
                timing = row.get(name)
                samples = timing.get("raw_ms") if isinstance(timing, dict) else None
                require(
                    isinstance(samples, list)
                    and len(samples) == 30
                    and all(_finite_number(value) and value >= 0 for value in samples),
                    "invalid highlight timing samples",
                )
                ordered = sorted(samples)
                for percentile, fraction in (("p50_ms", 0.5), ("p95_ms", 0.95)):
                    expected_value = ordered[math.ceil(fraction * len(ordered)) - 1]
                    require(
                        timing.get(percentile) == expected_value, "highlight timing percentile disagrees with samples"
                    )
                report_row[name] = {"p50_ms": timing["p50_ms"], "p95_ms": timing["p95_ms"]}
            summary.append(report_row)
        production = next(row for row in variants if row["variant"] == "production")
        require(
            production["finite_pass"] is True and production["baseline_pass"] is True,
            "highlight timing production baseline failed",
        )
    require(report.get("production_pass") is True, "highlight timing production baseline failed")
    production_pass = variant_pass["production"]
    experimental_pass = all(variant_pass[name] for name in HIGHLIGHT_VARIANTS[1:])
    require(production_pass, "highlight timing production baseline failed")
    return {
        "production_pass": production_pass,
        "experimental_pass": experimental_pass,
        "variant_pass": variant_pass,
        "source_algorithm": report["source_algorithm"],
        "descriptor": descriptor,
        "device": report.get("device"),
        "driver_version": report.get("driver_version"),
        "runtime_version": report.get("runtime_version"),
        "workspace_bytes": report.get("workspace_bytes"),
        "pairs": summary,
    }


def combine_highlight_passes(accuracy_summary, timing_summaries=()):
    """Combine accuracy and every timing workload into one pass state per variant."""
    variant_pass = accuracy_summary.get("variant_pass")
    require(isinstance(variant_pass, dict) and set(variant_pass) == set(HIGHLIGHT_VARIANTS), "accuracy pass map")
    combined = dict(variant_pass)
    for timing_summary in timing_summaries:
        timing_pass = timing_summary.get("variant_pass")
        require(isinstance(timing_pass, dict) and set(timing_pass) == set(HIGHLIGHT_VARIANTS), "timing pass map")
        for variant in HIGHLIGHT_VARIANTS:
            combined[variant] = combined[variant] and timing_pass[variant]
    production_pass = combined["production"]
    require(production_pass, "baseline production characterization failed")
    return {
        "variant_pass": combined,
        "production_pass": production_pass,
        "experimental_pass": all(combined[name] for name in HIGHLIGHT_VARIANTS[1:]),
    }


def _copy_file(source, destination):
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError("expected a regular file: " + str(source))
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return execution.digest(destination)


def check_highlight(args):
    if args.preset not in ("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"):
        raise ValueError("highlight check requires a native preset")
    if args.preset.startswith("windows") != (os.name == "nt"):
        raise ValueError("run the benchmark on the preset's native platform")
    run_id = execution.name_component(args.run_id)
    directory = ROOT / "out/validation" / args.preset / "results" / run_id
    build = ROOT / "out/build" / args.preset
    build_run_id = execution.name_component(run_id + "-build")
    build_result_path = ROOT / "out/validation" / args.preset / "results" / build_run_id / "result.json"
    executable = build / "tests/bin" / (HIGHLIGHT_TARGET + (".exe" if os.name == "nt" else ""))
    with execution.OutputLease(ROOT, build):
        if directory.exists() or build_result_path.parent.exists():
            raise ValueError("run or build result directory already exists; choose a new --run-id")
        directory.mkdir(parents=True, exist_ok=False)
        run = execution.NativeRun(ROOT, directory)
        source, patch = native_source_snapshot(domain_tests=True)
        fixture_members = regular_inventory(HIGHLIGHT_FIXTURE_ROOT)
        (directory / "candidate.json").write_text(json.dumps(source, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (directory / "candidate.patch").write_bytes(patch)
        result = {
            "schema": "highlight-production-run.v1",
            "status": "failed",
            "preset": args.preset,
            "run_id": run_id,
            "candidate_head": source["head"],
            "source_identity": source_identity(source, patch),
            "build_source_identity": None,
            "build_result_sha256": None,
            "build_run_id": build_run_id,
            "build_target": HIGHLIGHT_TARGET,
            "probe_sha256": None,
            "fixture_members": fixture_members,
            "commands": run.commands,
            "accuracy": None,
            "limitations": "Standalone CUDA production operator check; not full product qualification or installed Resolve evidence.",
        }
        error = None
        build_result = None
        build_candidate = None
        build_candidate_patch = None
        build_source = None
        build_patch = None
        probe_hash = None
        case_ids = None
        try:
            run.command(
                "build-probe",
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts/verify.py"),
                    "prepare",
                    "--preset",
                    args.preset,
                    "--targets",
                    HIGHLIGHT_TARGET,
                    "--testing",
                    "ON",
                    "--run-id",
                    build_run_id,
                ],
            )
            build_source, build_patch = native_source_snapshot(domain_tests=False, literal_name="scripts/verify.py")
            build_result, build_candidate, build_candidate_patch = read_build_binding(
                build_result_path, args.preset, build_source
            )
            if source.get("head") != build_candidate.get("head"):
                raise ValueError("run candidate and prepared build have different HEAD revisions")
            source_after_build, patch_after_build = native_source_snapshot(domain_tests=True)
            fixtures_after_build = regular_inventory(HIGHLIGHT_FIXTURE_ROOT)
            if source_after_build != source or patch_after_build != patch or fixtures_after_build != fixture_members:
                raise ValueError("source or fixtures changed during probe preparation")
            if not executable.is_file():
                raise ValueError("prepared highlight precision probe is missing")

            (directory / "build-candidate.json").write_text(
                json.dumps(build_candidate, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            (directory / "build-candidate.patch").write_bytes(build_candidate_patch)
            build_directory = directory / "build"
            build_directory.mkdir()
            for member in ("result.json", *build_result.get("members", {})):
                _copy_file(build_result_path.parent / member, build_directory / member)
            probe_hash = _copy_file(executable, directory / executable.name)
            result["probe_sha256"] = probe_hash
            reference = importlib.import_module("highlight_reference")
            supplement = reference.generate()
            if supplement.get("schema") != "boost-precision-supplement.v1":
                raise ValueError("unexpected highlight reference supplement schema")
            _, case_ids = _accuracy_case_ids(supplement)
            (directory / "supplement.json").write_text(
                json.dumps(supplement, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            (directory / "fixtures.json").write_text(
                json.dumps(fixture_members, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            result["build_source_identity"] = source_identity(build_source, build_patch)
            result["build_result_sha256"] = execution.digest(build_directory / "result.json")

            with execution.OutputLease(ROOT):
                nvidia_smi = shutil.which("nvidia-smi")
                if not nvidia_smi:
                    raise ValueError("nvidia-smi is required to retain the GPU/driver context")
                try:
                    run.command("device-before", [nvidia_smi])
                    supplement_path = directory / "supplement.json"
                    accuracy_path = directory / "accuracy.json"
                    run.command(
                        "accuracy",
                        [
                            str(directory / executable.name),
                            "--mode",
                            "accuracy",
                            "--fixture-root",
                            str(HIGHLIGHT_FIXTURE_ROOT),
                            "--supplement",
                            str(supplement_path),
                            "--output",
                            str(accuracy_path),
                        ],
                    )
                    accuracy_summary = validate_highlight_accuracy(
                        json.loads(accuracy_path.read_text(encoding="utf-8")), case_ids, production=True
                    )
                    result["accuracy"] = accuracy_summary
                    result["production_pass"] = accuracy_summary["production_pass"]
                    result["status"] = "passed"
                finally:
                    try:
                        run.command("device-after", [nvidia_smi])
                    except Exception as exc:
                        result["status"] = "failed"
                        result["device_after_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result["error"] = error
        finally:
            final_source, final_patch = native_source_snapshot(domain_tests=True)
            final_fixtures = regular_inventory(HIGHLIGHT_FIXTURE_ROOT)
            result["source_unchanged"] = final_source == source and final_patch == patch
            result["fixtures_unchanged"] = final_fixtures == fixture_members
            result["probe_unchanged"] = (
                probe_hash is None or execution.digest(directory / executable.name) == probe_hash
            )
            if not result["source_unchanged"] or not result["fixtures_unchanged"] or not result["probe_unchanged"]:
                result["status"] = "failed"
                result["completion_error"] = "source, fixtures, or retained probe changed during run"
            if error is not None and result.get("status") != "passed":
                result["status"] = "failed"
            if all(command.get("drained") is True for command in run.commands):
                result["members"] = inventory(directory)
                run.publish(result)
            else:
                execution.write_json(directory / "incomplete.json", result)
        if result["status"] != "passed":
            raise ValueError(result.get("error", result.get("completion_error", "highlight precision run failed")))
        print(
            json.dumps(
                {"status": result["status"], "production_pass": result["production_pass"], "bundle": str(directory)}
            )
        )
    return 0


def highlight_run_kind(result):
    production = result.get("schema") == "highlight-production-run.v1"
    require(
        (production and result.get("status") == "passed")
        or (result.get("schema") == "highlight-precision-run.v1" and result.get("status") == "completed"),
        "incomplete highlight run",
    )
    presets = (
        ("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release")
        if production
        else ("linux-release", "windows-clang-release")
    )
    require(result.get("preset") in presets, "unsupported highlight preset")
    return production


def collect_highlight(args):
    directory = args.bundle.resolve(strict=True)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    production = highlight_run_kind(result)
    require(
        all(result.get(key) is True for key in ("source_unchanged", "fixtures_unchanged", "probe_unchanged")),
        "highlight input identity changed",
    )
    require(inventory(directory) == result.get("members"), "highlight bundle member identity")
    require(result.get("build_target") == HIGHLIGHT_TARGET, "highlight build target mismatch")
    build_directory = directory / "build"
    build_result_path = build_directory / "result.json"
    require(execution.digest(build_result_path) == result.get("build_result_sha256"), "build result hash mismatch")
    qualification = importlib.import_module("qualification")
    build_summary = qualification.read_native_result(build_directory)
    build_result = json.loads(build_result_path.read_text(encoding="utf-8"))
    require(build_summary.get("status") == "passed", "missing successful build closure")
    require(
        build_result.get("preset") == result["preset"] and HIGHLIGHT_TARGET in build_result.get("targets", []),
        "build closure mismatch",
    )
    build_candidate = json.loads((build_directory / "candidate.json").read_text(encoding="utf-8"))
    if production:
        config = build_candidate.get("configuration", {})
        expected_type = "Debug" if result["preset"].endswith("-debug") else "Release"
        require(
            config.get("BUILD_TESTING") == "ON" and config.get("CMAKE_BUILD_TYPE") == expected_type,
            "build configuration mismatch",
        )
    require(
        source_identity(
            json.loads((directory / "build-candidate.json").read_text(encoding="utf-8")),
            (directory / "build-candidate.patch").read_bytes(),
        )
        == source_identity(build_candidate, (build_directory / "candidate.patch").read_bytes()),
        "retained build source mismatch",
    )
    require(
        source_identity(
            json.loads((directory / "candidate.json").read_text(encoding="utf-8")),
            (directory / "candidate.patch").read_bytes(),
        )
        == result.get("source_identity"),
        "retained run source mismatch",
    )
    require(
        source_identity(build_candidate, (build_directory / "candidate.patch").read_bytes())
        == result.get("build_source_identity"),
        "retained product source mismatch",
    )
    probe_name = HIGHLIGHT_TARGET + (".exe" if result["preset"].startswith("windows") else "")
    require(execution.digest(directory / probe_name) == result.get("probe_sha256"), "retained probe hash mismatch")
    fixtures = regular_inventory(HIGHLIGHT_FIXTURE_ROOT)
    require(
        fixtures
        == result.get("fixture_members")
        == json.loads((directory / "fixtures.json").read_text(encoding="utf-8")),
        "frozen fixture identity mismatch",
    )
    supplement = json.loads((directory / "supplement.json").read_text(encoding="utf-8"))
    require(supplement.get("schema") == "boost-precision-supplement.v1", "supplement schema")
    _, case_ids = _accuracy_case_ids(supplement)
    accuracy_summary = validate_highlight_accuracy(
        json.loads((directory / "accuracy.json").read_text(encoding="utf-8")), case_ids, production=production
    )
    require(accuracy_summary == result.get("accuracy"), "accuracy summary mismatch")
    require(accuracy_summary["production_pass"] is True, "baseline production failure is fatal")
    expected_commands = ["build-probe", "device-before", "accuracy"]
    if not production and not result.get("accuracy_only"):
        expected_commands += [
            f"timing-{size}-{density}" for size in ("1080p", "uhd") for density in ("sparse", "dense")
        ]
    expected_commands += ["device-after"]
    require(
        [command.get("name") for command in result.get("commands", [])] == expected_commands,
        "highlight command membership",
    )
    require(
        all(
            command.get("returncode") == 0
            and command.get("drained") is True
            and command.get("process_status") == "completed"
            for command in result["commands"]
        ),
        "failed or undrained highlight child",
    )
    if production:
        require(result.get("production_pass") is True, "production accuracy failure")
        print(json.dumps({"status": "passed", "production_pass": True, "bundle": str(directory)}))
        return 0
    timings = {}
    timing_summaries = []
    if not result.get("accuracy_only"):
        for name, width, height, density in (
            ("timing-1080p-sparse", 1920, 1080, "sparse"),
            ("timing-1080p-dense", 1920, 1080, "dense"),
            ("timing-uhd-sparse", 3840, 2160, "sparse"),
            ("timing-uhd-dense", 3840, 2160, "dense"),
        ):
            timings[name] = validate_highlight_timing(
                json.loads((directory / (name + ".json")).read_text(encoding="utf-8")), width, height, density
            )
            timing_summaries.append(timings[name])
        require(timings == result.get("timings"), "highlight timing summary mismatch")
    combined = combine_highlight_passes(accuracy_summary, timing_summaries)
    require(combined == {key: result.get(key) for key in combined}, "combined variant pass summary mismatch")
    print(
        json.dumps(
            {
                "status": "completed",
                "variant_pass": combined["variant_pass"],
                "experimental_pass": combined["experimental_pass"],
                "bundle": str(directory),
            }
        )
    )
    return 0


def expected_accuracy(manifest):
    expected = set()
    for row in manifest["rows"]:
        if "pixel_size_um" not in row:
            continue
        name = row["id"]
        if name == "wide-yvv-1024-above":
            expected.add((name, "admission", None, None))
            continue
        stages = ["post_reflection"]
        settings = row["parameters"]
        if settings["scatter_amount"] != 0 and settings["halation_amount"] != 0:
            stages.append("post_scatter")
        expected.update(
            (name, stage, variant, padding) for stage in stages for variant in VARIANTS for padding in (0, 7)
        )
    return expected


def validate_accuracy(report, manifest):
    require(report["schema"] == "scatter-halation-precision-accuracy.v1", "accuracy schema")
    rows = report["records"]
    keys = [(r["id"], r["stage"], r.get("variant"), r.get("padding")) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected_accuracy(manifest), "accuracy membership")
    require(report["existing_gate_pass"] is True, "existing precision failed fixture gate")
    digests = {}
    fixtures = {row["id"]: row for row in manifest["rows"] if "pixel_size_um" in row}
    for row in rows:
        if row["stage"] == "admission":
            require(row["rejected"] is True, "radius rejection")
            continue
        require(len(row["channels"]) == 3 and all(c["count"] > 0 for c in row["channels"]), "RGB metrics")
        require(
            all(
                len(row[key]) == 64 and all(c in "0123456789abcdef" for c in row[key])
                for key in ("input_sha256", "output_sha256")
            ),
            "plane digest",
        )
        fixture = fixtures[row["id"]]
        require(
            row["reference_input_sha256"] == fixture["input"]["sha256"]
            and row["reference_output_sha256"] == fixture[row["stage"]]["sha256"],
            "reference digest",
        )
        key = (row["id"], row["stage"], row["variant"])
        digests.setdefault(key, []).append((row["input_sha256"], row["output_sha256"]))
        if row["variant"] == "existing":
            require(
                row["fixture_gate_pass"] is True and row["existing_ordinary_equal"] is True,
                "existing fixture or ordinary-launch failure",
            )
    require(all(len(values) == 2 and values[0] == values[1] for values in digests.values()), "compact/padded mismatch")
    require(
        all(len({digests[(row, stage, variant)][0][0] for variant in VARIANTS}) == 1 for row, stage, _ in digests),
        "variant source mismatch",
    )
    return {
        variant: sum(not r["fixture_gate_pass"] for r in rows if r.get("variant") == variant) for variant in VARIANTS
    }


def validate_timing(report, width, height):
    require(
        report["schema"] == "scatter-halation-precision-timing.v1" and report["mode"] == "precision-timing",
        "timing schema",
    )
    require((report["width"], report["height"]) == (width, height), "timing extent")
    require(
        report["warmupCount"] == 10 and report["precision_sample_count"] == 30 and report["admittedBytes"] > 0,
        "timing preparation",
    )
    rows = report["precision_rows"]
    keys = [(r["pair"], r["variant"]) for r in rows]
    require(
        len(keys) == 9 and len(set(keys)) == 9 and set(keys) == {(p, v) for p in range(3) for v in VARIANTS},
        "timing membership",
    )
    require(
        [(r["pair"], r["order"], r["variant"]) for r in rows]
        == [
            (pair, position, variant)
            for pair in range(3)
            for position, variant in enumerate(VARIANTS if pair % 2 == 0 else tuple(reversed(VARIANTS)))
        ],
        "timing order",
    )
    require(
        len(report["sourceChecksum"]) == 64 and all(c in "0123456789abcdef" for c in report["sourceChecksum"]),
        "timing source checksum",
    )
    require(bool(report["host_factor_derivation"]) and bool(report["gpu_event_interval"]), "timing interpretation")
    for row in rows:
        require(len(row["stage_samples_ms"]) == len(row["host_samples_ms"]) == 30, "timing sample count")
        require(
            all(math.isfinite(v) and v >= 0 for v in row["stage_samples_ms"] + row["host_samples_ms"]), "invalid timing"
        )
        for clock in ("stage", "host"):
            require(math.isfinite(row[f"cold_{clock}_ms"]) and row[f"cold_{clock}_ms"] >= 0, "invalid first sample")
            samples = sorted(row[f"{clock}_samples_ms"])
            for percentile in (50, 95):
                expected = samples[math.ceil(percentile * len(samples) / 100) - 1]
                require(row[f"{clock}_p{percentile}_ms"] == expected, "timing percentile disagrees with samples")
        require(
            set(row["registers"]) == {"horizontal", "vertical_accumulation", "vertical_core"}
            and all(isinstance(v, int) and v > 0 for v in row["registers"].values()),
            "kernel register attributes",
        )
    return {"source_checksum": report["sourceChecksum"], "recipe_hash": report["descriptorRecipeHash"]}


def collect(args):
    directory = args.bundle.resolve(strict=True)
    result = json.loads((directory / "result.json").read_text())
    require(result["schema"] == "scatter-precision-run.v1" and result["status"] == "passed", "incomplete run")
    require(all(result[key] for key in ("source_unchanged", "fixtures_unchanged", "probe_unchanged")), "changed inputs")
    require(inventory(directory) == result["members"], "bundle member identity")
    expected = (
        ["build-probe", "device-before", "accuracy"]
        + ([] if result["accuracy_only"] else ["timing-1080p", "timing-uhd"])
        + ["device-after"]
    )
    require(
        [c["name"] for c in result["commands"]] == expected
        and all(c["returncode"] == 0 and c["drained"] for c in result["commands"]),
        "command completion",
    )
    inputs = json.loads((directory / "inputs.json").read_text())
    require(execution.digest(Path(inputs["probe"]["retained_path"])) == inputs["probe"]["sha256"], "probe identity")
    for path, digest in inputs["fixtures"].items():
        require(execution.digest(Path(path)) == digest, "fixture identity")
    manifest = json.loads((ROOT / "tests/scatter_halation/parity-fixtures/manifest.json").read_text())
    failures = validate_accuracy(json.loads((directory / "accuracy.json").read_text()), manifest)
    require(failures == result["fixture_failures"], "accuracy summary")
    if not result["accuracy_only"]:
        for name, width, height in (("timing-1080p", 1920, 1080), ("timing-uhd", 3840, 2160)):
            require(
                validate_timing(json.loads((directory / f"{name}.json").read_text()), width, height)
                == result["timings"][name],
                "timing summary",
            )
    print(json.dumps({"status": "passed", "fixture_failures": failures, "bundle": str(directory)}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--bundle", type=Path, required=True)
    run_parser = commands.add_parser("highlight-check")
    run_parser.add_argument(
        "--preset",
        required=True,
        choices=("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"),
    )
    run_parser.add_argument("--run-id", required=True)
    collect_highlight_parser = commands.add_parser("highlight-collect")
    collect_highlight_parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        return collect(args)
    if args.command == "highlight-check":
        return check_highlight(args)
    return collect_highlight(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Scatter precision failed: {error}", file=sys.stderr)
        raise SystemExit(1)
