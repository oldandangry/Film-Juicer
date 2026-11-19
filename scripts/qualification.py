"""Pure native outcome and membership validation; standard library and bundles only."""

from __future__ import annotations

# Native result reading uses only the standard library and bundled bytes.
# The complete module can be copied and run offline.
from datetime import datetime
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
import time
import tomllib
import xml.etree.ElementTree as ET


PRESET_TARGETS = {
    "linux-debug": "x86_64-unknown-linux-gnu",
    "linux-release": "x86_64-unknown-linux-gnu",
    "windows-clang-debug": "x86_64-pc-windows-msvc",
    "windows-clang-release": "x86_64-pc-windows-msvc",
}
RUST_QUALITY_PACKAGES = ("film-juicer-core", "film-juicer-plugin")
RUST_POLICY_FILES = {"Cargo.lock", "Cargo.toml", "clippy.toml", "rust-toolchain.toml", "rustfmt.toml"}
RUST_BUILD_FILES = {"CMakeLists.txt", "cmake/JuicerRust.cmake"}
SOURCE_FILE_POLICY = "source_file_policy.json"
NATIVE_POLICY_FILES = {".clang-format", ".clang-tidy", "scripts/check-quality.py", "scripts/" + SOURCE_FILE_POLICY}
PRODUCT_PREVIEW_KEYS = (
    "BUILD_TESTING",
    "JUICER_GRAIN_PREVIEW",
    "JUICER_GRAIN_PREVIEW_CONFIGURED",
    "JUICER_GRAIN_MOTION_REPLAY_CONFIGURED",
)
MOTION_REPLAY_TARGET = "JuicerGrainMotionReplay"
REPLAY_EXECUTION_SUFFIX = ".replay-execution.json"
MOTION_REPLAY_RENDER = "tests/ffi/production-ofx-render.cpp"
MOTION_REPLAY_HEADER_CONSUMERS = {
    "tests/grain/film-ofx-apertures.inc": (MOTION_REPLAY_RENDER,),
    "tests/grain/motion-density.inc": (MOTION_REPLAY_RENDER,),
    "tests/grain/motion-replay.h": (MOTION_REPLAY_RENDER, "src/Cuda/JuicerCudaExecutor.cpp"),
    "tests/grain/motion-replay.inc": (MOTION_REPLAY_RENDER,),
}
MOTION_REPLAY_EXCLUSION = "CMake did not configure the JuicerGrainMotionReplay target"
OPTIONAL_PRODUCT_REGISTRATIONS = {
    "Ffi.Host.GrainPreviewDiscovery": (
        "grain_preview",
        "host",
        "JuicerProductionPreparedRenderProbe",
        "--grain-preview-discovery",
    ),
    "Ffi.Gpu.GrainPreviewStaged": (
        "grain_preview",
        "gpu",
        "JuicerProductionPreparedRenderProbe",
        "--grain-preview-witness",
    ),
    "Grain.Gpu.FilmOfxApertures": (
        "grain_preview_motion_replay",
        "gpu",
        MOTION_REPLAY_TARGET,
        "--grain-preview-apertures",
    ),
}


def is_motion_replay_target_output(source, output):
    expected = PurePosixPath(f"CMakeFiles/{MOTION_REPLAY_TARGET}.dir/{source}.o").parts
    actual = PurePosixPath(output.replace("\\", "/")).parts
    return actual[-len(expected) :] == expected


def is_motion_replay_compile_entry(source, output, arguments):
    """Identify the CMake target entry that actually enables the guarded includes."""
    macro = "JUICER_GRAIN_MOTION_REPLAY"
    options = [arg for arg in arguments if macro in arg]
    return is_motion_replay_target_output(source, output) and options == ["-D" + macro + "=1"]


def motion_replay_tidy_label(source, *, legacy=False):
    name = re.sub(r"[^a-z0-9-]+", "-", PurePosixPath(source).name.lower()).strip("-")
    # Two configured source paths can have the same filename.
    identity = "" if legacy else hashlib.sha256(source.encode()).hexdigest()[:12] + "-"
    return "clang-tidy-motion-replay-" + identity + name


def read_preview_and_replay_configuration(configuration):
    """Read CMake's effective product and replay decisions, retaining the raw option."""
    if not isinstance(configuration, dict):
        raise ValueError("missing configured product preview and replay state")
    testing, raw_preview, preview, replay = (configuration.get(key) for key in PRODUCT_PREVIEW_KEYS)
    if (
        not isinstance(testing, str)
        or not isinstance(raw_preview, str)
        or any(value not in ("ON", "OFF") for value in (preview, replay))
    ):
        raise ValueError("missing configured product preview and replay state")
    preview_enabled = preview == "ON"
    replay_configured = replay == "ON"
    if replay_configured and (not preview_enabled or not grain_preview_enabled(testing)):
        raise ValueError("replay target requires a product preview with testing enabled")
    return preview_enabled, replay_configured


def quality_scope(selected, *, qualification):
    """Derive representative native coverage and Rust quality from the selected paths."""
    expanded = qualification or any(
        path.startswith(("tests/quality/", ".github/workflows/")) or path in NATIVE_POLICY_FILES for path in selected
    )
    rust_required = expanded or any(
        path in RUST_BUILD_FILES or path in RUST_POLICY_FILES or path.endswith((".rs", "/Cargo.toml"))
        for path in selected
    )
    return expanded, rust_required


def quality_native_source(path, extensions):
    return (
        PurePosixPath(path).suffix.lower() in extensions
        and path.startswith(("src/", "native/", "tests/"))
        and "/fixtures/" not in path
    )


def grain_preview_enabled(value: str | None) -> bool:
    """Interpret the cached value as CMake's if(JUICER_GRAIN_PREVIEW)."""
    if value is None or value in ("", "0", "NOTFOUND") or value.endswith("-NOTFOUND"):
        return False
    return value.upper() not in {"OFF", "NO", "FALSE", "N", "IGNORE"}


def clippy_forms(preset, packages, raw_preview, *, preview=None):
    """Return the required labels and package arguments for Rust quality."""
    profile = "release" if preset.endswith("-release") else "dev"
    profile_args = ["--release"] if profile == "release" else []
    if preview is None:
        preview = grain_preview_enabled(raw_preview)
    for package in packages:
        test_support = package == "film-juicer-plugin"
        variants = [()]
        if test_support:
            variants.append(("test-support",))
        if preview:
            variants.append(("grain-preview",))
            if test_support:
                variants.append(("grain-preview", "test-support"))
        for features in variants:
            label = "-".join(("clippy", package, *features, profile))
            feature_args = ["--features", ",".join(features)] if features else []
            yield label, ["-p", package, "--all-targets", *feature_args, *profile_args, "--", "-D", "warnings"]


OPTICS_FUNCTION = "optics::tests::matches_frozen_native_words_identities_and_failures"
PREVIEW_PUBLICATION_FUNCTION = "state::tests::film_proxy_publication_uses_full_grid_and_encloses_its_densities"
RUST_TEST_PACKAGES = {"core": "film-juicer-core", "bridge": "film-juicer-plugin"}
RUST_TEST_REGISTRATIONS = {"Rust.Core": "core", "Rust.Bridge": "bridge"}
RUST_TEST_BUILD_TARGET = "juicer_rust_tests"
RUST_TEST_PATH_FLAGS = ("cargo", "nextest", "binaries-metadata", "cargo-metadata", "config-file")
RUST_JUNIT_RUN_ID_PROPERTY = "juicer_nextest_run_id"
RUST_EXECUTION_SCHEMAS = {
    "native-tests-v2": "rust-test-execution-v3",
    "native-tests-v3": "rust-test-execution-v4",
    "native-tests-v4": "rust-test-execution-v4",
    "native-tests-v5": "rust-test-execution-v4",
    "native-tests-v6": "rust-test-execution-v4",
}
OPTICS_TESTS = ["Ffi.Host.SpatialOptics", "Rust.Core"]
GPU_WITNESS_TEST = "Ffi.Gpu.PreparedBoundary.negative-direct"
OPTICS_MEMBERS = {
    "candidate.json",
    "candidate.patch",
    "native-list.jsonl",
    "discovery.json",
    "nextest.xml",
    "ctest.xml",
    "rust.jsonl",
    "native.jsonl",
    "nextest.toml",
    "rust-execution.json",
}
PRESETS = tuple(PRESET_TARGETS)
GROUPS = {"product", "boundary", "tooling", "corpus"}


def optics_keys(rows):
    keys = [(r["caller"], r["id"]) for r in rows]
    if any(not all(isinstance(v, str) and v for v in k) for k in keys) or len(keys) != len(set(keys)):
        raise ValueError("invalid or duplicate case membership")
    return set(keys)


def optics_discovery(document):
    selected = []
    filtered = set()
    for binary, suite in document["rust-suites"].items():
        for name, case in suite["testcases"].items():
            if case["filter-match"]["status"] == "matches":
                if case["ignored"]:
                    raise ValueError("ignored pilot function")
                selected.append((binary, name))
            else:
                filtered.add((binary, name))
    if len(selected) != 1 or selected[0][1] != OPTICS_FUNCTION:
        raise ValueError("missing or unexpected Rust function")
    return selected[0], filtered


def case_events(payload, expected=(), *, cohorts=0, functions=None):
    """Validate independent cohort declarations and actual row terminals in one pass."""
    if payload and not payload.endswith(b"\n"):
        raise ValueError("truncated case output")
    known = {(("native", caller), case_id) for caller, case_id in expected}
    started, terminal, headers, finished, declared = set(), set(), {}, set(), set()
    counts = {}
    for line in payload.splitlines():
        row = json.loads(line)
        rust = "binary" in row
        context = ("rust", row["binary"], row["function"], row["cohort"]) if rust else ("native", row["caller"])
        if rust and (
            row.get("caller") != "rust" or functions is None or (row["binary"], row["function"]) not in functions
        ):
            raise ValueError("unexpected Rust row context")
        event = row["event"]
        if event == "membership":
            ids = row["ids"]
            if (
                context in headers
                or not ids
                or len(ids) != len(set(ids))
                or any(not isinstance(i, str) or not i for i in ids)
            ):
                raise ValueError("invalid or duplicate cohort membership")
            header_keys = {(context, i) for i in ids}
            if declared & header_keys:
                raise ValueError("duplicate declared case")
            headers[context] = header_keys
            declared.update(header_keys)
            known.update(header_keys)
            if rust:
                owner = row["binary"]
            elif isinstance(cohorts, dict):
                owners = [name for name in cohorts if row["caller"].startswith(name + "/")]
                if len(owners) != 1:
                    raise ValueError("unexpected GoogleTest row context")
                owner = owners[0]
            else:
                owner = "native"
            counts[owner] = counts.get(owner, 0) + 1
            continue
        if event == "cohort-terminal":
            if (
                context not in headers
                or context in finished
                or row["outcome"] != "passed"
                or not headers[context] <= terminal
            ):
                raise ValueError("failed, premature or duplicate cohort terminal")
            finished.add(context)
            continue
        key = (context, row["id"])
        if key not in known:
            raise ValueError("unexpected case: " + repr(key))
        if event == "start" and key not in started:
            started.add(key)
        elif event == "terminal" and key in started and key not in terminal:
            if row["outcome"] != "passed":
                raise ValueError("failed or skipped case: " + repr(key))
            terminal.add(key)
        else:
            raise ValueError("duplicate or unordered case event: " + repr(key))
    if terminal != known or started != known:
        raise ValueError("missing case outcomes: " + repr(sorted(known - terminal)[:4]))
    if any(context[0] == "rust" and context not in finished for context in headers):
        raise ValueError("missing Rust cohort completion")
    if isinstance(cohorts, int):
        if len(headers) != cohorts:
            raise ValueError("missing or unexpected required cohort")
    elif {name: count for name, count in cohorts.items() if count} != counts:
        raise ValueError("missing or unexpected required cohort context")
    return len(known)


def optics_events(payload, expected):
    return case_events(payload, expected)


def rust_case_events(payload, contract, functions):
    binaries = {binary for binary, _ in functions}
    packages = {binary.split("::", 1)[0] for binary in binaries}
    if len(packages) != 1:
        raise ValueError("Rust row selection must own one package")
    package = next(iter(packages))
    required = {
        binary: count for binary, count in contract["rust_cohorts"].items() if binary.split("::", 1)[0] == package
    }
    if set(required) != binaries:
        raise ValueError("missing required Rust row binary")
    return case_events(payload, cohorts=required, functions=functions)


def _native_metadata(result):
    if (
        result.get("schema") != "native-spatial-optics-v1"
        or result.get("preset") not in ("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release")
        or result.get("status") not in ("passed", "failed")
        or result.get("shipping_isolation") is not False
    ):
        raise ValueError("invalid native result metadata")
    rows = result["membership"]["rows"]
    if len(rows) != 553 or any(not isinstance(r, str) or not r for r in rows) or len(set(rows)) != 553:
        raise ValueError("invalid 553-row membership")
    operations = optics_keys(result["membership"]["native_operations"])
    if (not operations and result["status"] == "passed") or any(c not in {"c11", "cpp"} for c, _ in operations):
        raise ValueError("invalid native operation membership")
    if (
        result["membership"]["ctest"] not in (OPTICS_TESTS, ["Ffi.Host.SpatialOptics", "Rust.Core.SpatialOpticsPilot"])
        or result["membership"]["rust_function"] != OPTICS_FUNCTION
    ):
        raise ValueError("invalid pilot selection")
    members = result["members"]
    required = OPTICS_MEMBERS if result["status"] == "passed" else {"candidate.json", "candidate.patch"}
    allowed = OPTICS_MEMBERS
    witness = result.get("gpu_witness")
    if witness is not None:
        if witness.get("test") != GPU_WITNESS_TEST or witness.get("scope") != "containment-and-shared-lock-only":
            raise ValueError("invalid GPU portability witness")
        allowed = allowed | {"gpu-witness.xml"}
        if result["status"] == "passed":
            required = required | {"gpu-witness.xml"}
    if not required <= members.keys() or set(members) - allowed:
        raise ValueError("missing required result members")
    # Reject the full reference list before opening any payload.
    for name, spec in members.items():
        path = PurePosixPath(name)
        if path.is_absolute() or path.parts != (name,) or any(c in name for c in "\\:") or name in {".", ".."}:
            raise ValueError("unconfined bundle reference")
        if type(spec["bytes"]) is not int or spec["bytes"] < 0 or len(spec["sha256"]) != 64:
            raise ValueError("invalid member identity")
    return rows, operations


def validate_native_result(result, payloads):
    """Validate captured bundle bytes; shared by publication and offline reading."""
    rows, operations = _native_metadata(result)
    if payloads.keys() != result["members"].keys():
        raise ValueError("missing required result members")
    for name, spec in result["members"].items():
        data = payloads[name]
        if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError("damaged bundle member: " + name)
    candidate = json.loads(payloads["candidate.json"])
    if candidate["preset"] != result["preset"] or candidate["head"] != result["candidate_head"]:
        raise ValueError("candidate/configuration disagreement")
    if result["status"] == "passed":
        expected_type = "Debug" if result["preset"].endswith("debug") else "Release"
        if (
            candidate["configuration"].get("CMAKE_BUILD_TYPE") != expected_type
            or candidate["configuration"].get("BUILD_TESTING") != "ON"
        ):
            raise ValueError("actual build configuration disagrees with preset")
    if not candidate["inputs"] or candidate["fixtures"]["rows"] != rows:
        raise ValueError("candidate fixture membership disagreement")
    if not result["commands"] or any(c["drained"] is not True for c in result["commands"]):
        raise ValueError("uncompleted descendant drainage")
    if result["status"] == "passed":
        if result.get("candidate_unchanged") is not True:
            raise ValueError("candidate changed or completion comparison missing")
        if any(c["returncode"] != 0 or c["process_status"] != "completed" for c in result["commands"]):
            raise ValueError("failed command cannot pass")
        native = [{"caller": c, "id": r} for c in ("c11", "cpp") for r in rows]
        expected_native = optics_keys(native) | operations
        if optics_keys([json.loads(line) for line in payloads["native-list.jsonl"].splitlines()]) != expected_native:
            raise ValueError("native listing omitted caller/row/operation")
        optics_events(payloads["rust.jsonl"], {("rust", r) for r in rows})
        optics_events(payloads["native.jsonl"], expected_native)
        rust_execution = json.loads(payloads["rust-execution.json"])
        if len(rust_execution["commands"]) != 2 or any(c["returncode"] != 0 for c in rust_execution["commands"]):
            raise ValueError("missing/failed nextest invocation")
        selected, filtered = optics_discovery(json.loads(payloads["discovery.json"]))
        seen = set()
        for suite in ET.fromstring(payloads["nextest.xml"]).iter("testsuite"):
            for case in suite.findall("testcase"):
                key = (suite.get("name"), case.get("name"))
                if key in seen or any(
                    case.find(n) is not None for n in ("failure", "error", "flakyFailure", "flakyError")
                ):
                    raise ValueError("duplicate/failed Rust function terminal")
                seen.add(key)
                if key == selected:
                    if case.find("skipped") is not None:
                        raise ValueError("selected Rust function skipped")
                elif key not in filtered or case.find("skipped") is None:
                    raise ValueError("unexpected Rust function terminal")
        if seen != filtered | {selected}:
            raise ValueError("missing Rust function terminal")
        cases = list(ET.fromstring(payloads["ctest.xml"]).iter("testcase"))
        if sorted(c.get("name") for c in cases) != result["membership"]["ctest"] or any(
            c.get("status") != "run" or any(c.find(n) is not None for n in ("failure", "error", "skipped"))
            for c in cases
        ):
            raise ValueError("missing/failed/skipped CTest case")
    witness = result.get("gpu_witness")
    if result["status"] == "passed" and witness is not None:
        if (
            witness.get("status") != "passed"
            or witness.get("contender_excluded") is not True
            or witness.get("drained_while_locked") is not True
        ):
            raise ValueError("GPU witness did not complete under shared exclusion")
        if not any(name.startswith("Resources/") for name in candidate["inputs"]):
            raise ValueError("GPU runtime resources missing from candidate")
        cases = list(ET.fromstring(payloads["gpu-witness.xml"]).iter("testcase"))
        if (
            len(cases) != 1
            or cases[0].get("name") != GPU_WITNESS_TEST
            or cases[0].get("status") != "run"
            or any(cases[0].find(n) is not None for n in ("failure", "error", "skipped"))
        ):
            raise ValueError("missing/failed GPU portability witness")
    summary = {
        "status": result["status"],
        "preset": result["preset"],
        "rust_rows": 553 if result["status"] == "passed" else 0,
        "native_rows": 1106 if result["status"] == "passed" else 0,
        "native_operations": len(operations),
        "shipping_isolation": False,
    }
    if witness is not None:
        summary["gpu_witness"] = {"test": witness["test"], "scope": witness["scope"], "status": witness["status"]}
    return summary


def configured_product_contract(contract, runtime, configuration):
    """Apply optional catalog sections selected by CMake to the base inventory."""
    preview_enabled, replay_configured = read_preview_and_replay_configuration(configuration)
    enabled = {"grain_preview": preview_enabled, "grain_preview_motion_replay": replay_configured}
    merged = {
        **contract,
        "required": list(contract["required"]),
        **{field: dict(contract.get(field, {})) for field in ("targets", "native_cohorts", "native_membership")},
    }
    for name, (section, expected_runtime, owner, _) in OPTIONAL_PRODUCT_REGISTRATIONS.items():
        if runtime != expected_runtime:
            continue
        if (
            name in merged["required"]
            or name in merged["native_cohorts"]
            or name in merged["native_membership"]
            or (section == "grain_preview_motion_replay" and owner in merged["targets"])
        ):
            raise ValueError("optional product registration in the base catalog: " + name)
        if not enabled[section]:
            continue
        scope = contract.get(section, {})
        if (
            scope.get("required") != [name]
            or set(scope.get("targets", {})) != {owner}
            or set(scope.get("native_cohorts", {})) != {name}
            or set(scope.get("native_membership", {})) != {name}
        ):
            raise ValueError("missing configured product registration contract: " + name)
        merged["required"].append(name)
        for field in ("targets", "native_cohorts", "native_membership"):
            merged[field].update(scope[field])
    return merged


def native_registration(contract, paths, registration, *, runtime="host", cuda_free=False, artifact_consumers=None):
    """Check independent owning target counts before selecting execution."""
    if cuda_free:
        targets = {
            name: row for name, row in contract["targets"].items() if runtime == "host" and row.get("cuda_free") is True
        }
        required = contract.get("cuda_free_required", []) if runtime == "host" else []
        if runtime == "host" and (not targets or not required or not set(required) <= set(contract["required"])):
            raise ValueError("missing CUDA-free host contract")
        contract = {**contract, "targets": targets, "required": required}
    if contract.get("schema") != f"native-{runtime}-contracts-v1" or set(paths) != set(contract["targets"]):
        raise ValueError("invalid host target contract")
    found = {}
    counts = dict.fromkeys(paths, 0)
    for test in registration["tests"]:
        properties = {p["name"]: p["value"] for p in test["properties"]}
        if runtime not in properties.get("LABELS", []):
            continue
        command = test.get("command", [])
        if "native-case" not in command:
            continue
        split = command.index("--")
        native = command[split + 1 :]
        owners = [
            target
            for target, path in paths.items()
            if path.replace("\\", "/").casefold() == native[0].replace("\\", "/").casefold()
        ]
        if len(owners) != 1:
            raise ValueError("unowned native host executable")
        target = owners[0]
        if {"host", "gpu"} & set(properties.get("LABELS", [])) != {runtime} or properties.get("DISABLED"):
            raise ValueError("disabled or incorrect native host runtime")
        command_runtime = command[command.index("--runtime") + 1] if "--runtime" in command[:split] else "host"
        if command_runtime != runtime:
            raise ValueError("native launcher runtime disagrees with registration")
        timeout = properties.get("TIMEOUT")
        if runtime == "gpu" and (
            type(timeout) not in (int, float)
            or not 0 <= timeout < float("inf")
            or properties.get("RESOURCE_LOCK") != ["juicer_gpu"]
        ):
            raise ValueError("GPU registration requires a finite timeout and physical-device lock")
        if test["name"] in found:
            raise ValueError("duplicate native host registration")
        protocol = contract["targets"][target]["protocol"]
        if command[command.index("--protocol") + 1] != protocol:
            raise ValueError("native host protocol disagrees with target owner")
        key = "host-" + hashlib.sha256(json.dumps(native, ensure_ascii=False).encode()).hexdigest()[:16]
        found[test["name"]] = {
            "target": target,
            "protocol": protocol,
            "command": native,
            "key": key,
            "rows": test["name"] in contract["targets"][target].get("rows", []),
        }
        counts[target] += 1
    for name, (_, expected_runtime, owner, selector) in OPTIONAL_PRODUCT_REGISTRATIONS.items():
        if name not in contract["required"]:
            continue
        entry = found.get(name)
        if (
            runtime != expected_runtime
            or entry is None
            or entry["target"] != owner
            or entry["command"][1:2] != [selector]
            or (owner == MOTION_REPLAY_TARGET and len(entry["command"]) != 2)
        ):
            raise ValueError("optional product registration differs from its owning command: " + name)
        if artifact_consumers is not None:
            binary = artifact_consumers.get(owner, {}).get("binary")
            if paths[owner] != binary or entry["command"][0] != binary:
                raise ValueError("optional product executable differs from configured target output: " + name)
    if counts != {t: s["registrations"] for t, s in contract["targets"].items()}:
        raise ValueError("missing native host registration")
    names = [t["name"] for t in registration["tests"]]
    if not set(contract["required"]) <= set(names) or len(names) != len(set(names)):
        raise ValueError("missing/duplicate required host contract")
    return found


def gtest_discovery(payload, *, excluded=frozenset()):
    document = json.loads(payload)
    names = []
    for suite in document["testsuites"]:
        for case in suite["testsuite"]:
            key = (suite["name"], case["name"])
            disabled = suite["name"].startswith("DISABLED_") or case["name"].startswith("DISABLED_")
            if disabled != (key in excluded):
                raise ValueError("disabled native host case")
            names.append(key)
    if not names or len(names) != len(set(names)):
        raise ValueError("empty/duplicate GoogleTest selection")
    return set(names)


def native_cohort_count(contract, name, preset):
    count = contract["native_cohorts"].get(name, 0)
    if isinstance(count, dict):
        count = count["windows" if preset.startswith("windows") else "linux"]
    if type(count) is not int or count < 0:
        raise ValueError("invalid native cohort count")
    return count


def native_membership(contract, name, listed, terminals):
    specification = contract.get("native_membership", {}).get(name)
    if specification is None:
        return
    required = specification["required"]
    if (
        type(specification["listed"]) is not int
        or type(specification["terminal"]) is not int
        or not isinstance(required, list)
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(value, str) or not value for value in pair)
            for pair in required
        )
    ):
        raise ValueError("invalid native membership contract")
    required_keys = {tuple(pair) for pair in required}
    if (
        len(required_keys) != len(required)
        or len(listed) != specification["listed"]
        or terminals != specification["terminal"]
        or not required_keys <= listed
    ):
        raise ValueError("missing or unexpected native membership")


def framework_terminals(payload, expected, *, nextest=False, filtered=frozenset()):
    report = ET.fromstring(payload)
    seen = set()
    for suite in report.iter("testsuite"):
        for case in suite.findall("testcase"):
            key = (suite.get("name"), case.get("name"))
            if key in seen or any(
                case.find(tag) is not None for tag in ("failure", "error", "flakyFailure", "flakyError")
            ):
                raise ValueError("duplicate/failed framework case")
            seen.add(key)
            if key in expected:
                if case.find("skipped") is not None or (not nextest and case.get("status") != "run"):
                    raise ValueError("required framework case skipped")
            elif key not in filtered or (
                case.find("skipped") is None
                if nextest
                else case.get("status") != "notrun" or case.get("result") != "suppressed"
            ):
                raise ValueError("unexpected framework terminal")
    if seen != expected | filtered:
        raise ValueError("missing framework case terminal")
    return report


def rust_package_features(cargo, package_id):
    matches = [node["features"] for node in cargo["resolve"]["nodes"] if node["id"] == package_id]
    if len(matches) != 1 or not isinstance(matches[0], list):
        raise ValueError("missing resolved Rust package features")
    features = matches[0]
    if any(not isinstance(feature, str) or not feature for feature in features) or len(set(features)) != len(features):
        raise ValueError("invalid resolved Rust package features")
    return sorted(features)


def rust_test_package(cargo, package):
    owners = [owner for owner in cargo["packages"] if owner["name"] == RUST_TEST_PACKAGES[package]]
    if len(owners) != 1:
        raise ValueError("missing or ambiguous Rust test package: " + RUST_TEST_PACKAGES[package])
    return owners[0]


def validate_rust_test_request(request, preset, build_kind, preview):
    features = "film-juicer-plugin/test-support"
    if preview:
        features += ",film-juicer-plugin/grain-preview"
    if (
        build_kind not in {"product", "cuda-free"}
        or type(preview) is not bool
        or build_kind == "cuda-free"
        and preview
        or request
        != {
            "build_kind": build_kind,
            "grain_preview": preview,
            "target": PRESET_TARGETS[preset],
            "profile": "release" if preset.endswith("-release") else "debug",
            "cargo_features": features,
        }
        or type(request["grain_preview"]) is not bool
    ):
        raise ValueError("Rust test request differs from the configured build")


def validate_rust_workspace_features(cargo, preview):
    expected = ["grain-preview", "test-support"] if preview else ["test-support"]
    for package, name in RUST_TEST_PACKAGES.items():
        owner = rust_test_package(cargo, package)
        if owner["id"] not in cargo.get("workspace_members", ()):
            raise ValueError("Rust test package does not belong to the workspace: " + name)
        if rust_package_features(cargo, owner["id"]) != expected:
            raise ValueError("Rust workspace features differ from configured build: " + name)
    return expected


def rust_test_path(preset):
    return PureWindowsPath if preset.startswith("windows-") else PurePosixPath


def rust_test_target_directory(source, preset):
    return source / "out" / "build" / preset / "cargo" / "test-support"


def validate_rust_test_paths(paths, source, preset):
    path = rust_test_path(preset)
    metadata = rust_test_target_directory(source, preset) / "metadata"
    if (
        path(paths["binaries-metadata"]) != metadata / "binaries.json"
        or path(paths["cargo-metadata"]) != metadata / "cargo.json"
        or path(paths["config-file"]) != source / ".config" / "nextest.toml"
    ):
        raise ValueError("Rust test input paths differ from the configured workspace or build")


def validate_rust_test_build(request, preset, source, cargo, binaries):
    features = validate_rust_workspace_features(cargo, request["grain_preview"])
    path = rust_test_path(preset)
    target_dir = rust_test_target_directory(source, preset)
    if path(cargo.get("workspace_root", "")) != source or path(cargo.get("target_directory", "")) != target_dir:
        raise ValueError("Cargo metadata differs from configured workspace or target directory")

    build = binaries["rust-build-meta"]
    target, profile = request["target"], request["profile"]
    output = path(target, profile)
    platforms = build.get("platforms", {})
    if (
        build.get("target-platform") != target
        or [entry.get("triple") for entry in build.get("target-platforms", [])] != [target]
        or platforms.get("host", {}).get("platform", {}).get("triple") != target
        or [entry.get("platform", {}).get("triple") for entry in platforms.get("targets", [])] != [target]
        or [path(entry) for entry in build.get("base-output-directories", [])] != [output]
        or path(build.get("target-directory", "")) != target_dir
    ):
        raise ValueError("prepared Rust binaries differ from configured target or profile, or target directory")

    owners = [rust_test_package(cargo, package) for package in RUST_TEST_PACKAGES]
    owner_ids = {owner["id"] for owner in owners}
    if len(owner_ids) != len(owners) or sorted(cargo["workspace_members"]) != sorted(owner_ids):
        raise ValueError("Rust test packages differ from the configured workspace")
    expected = {}
    for owner in owners:
        manifest = source / "rust" / owner["name"] / "Cargo.toml"
        if path(owner.get("manifest_path", "")) != manifest:
            raise ValueError("Rust package manifest differs from configured workspace: " + owner["name"])
        targets = rust_test_binaries(owner)
        if not targets:
            raise ValueError("missing Rust test targets: " + owner["name"])
        expected.update({(owner["id"], target_name): kind for target_name, kind in targets.items()})

    prepared = {}
    executables = set()
    for binary_id, binary in binaries["rust-binaries"].items():
        executable = path(binary.get("binary-path", ""))
        if (
            binary.get("binary-id") != binary_id
            or binary.get("build-platform") != "target"
            or executable.parent != target_dir / output / "deps"
        ):
            raise ValueError("prepared Rust test binary differs from configured build: " + binary_id)
        if executable in executables:
            raise ValueError("duplicate prepared Rust test executable path: " + str(executable))
        executables.add(executable)
        key = (binary["package-id"], binary["binary-name"])
        if key in prepared:
            raise ValueError("duplicate prepared Rust test binary: " + binary_id)
        prepared[key] = binary["kind"]
    if prepared != expected:
        raise ValueError("missing or unexpected prepared Rust test binaries")
    return features


def validate_rust_execution_features(execution, package, cargo, *, require_resolved=False):
    if any(key in execution for key in ("request", "package_id", "junit_sha256")):
        raise ValueError("current Rust execution binding in a historical result")
    if "schema" not in execution and execution.get("features") == "test-support":
        if require_resolved:
            raise ValueError("preview Rust execution requires resolved feature evidence")
        return  # Historical receipts carry the original hard-coded label.
    package_name = {"core": "film-juicer-core", "bridge": "film-juicer-plugin"}[package]
    packages = [owner["id"] for owner in cargo["packages"] if owner["name"] == package_name]
    if execution.get("schema") != "rust-test-execution-v2" or len(packages) != 1:
        raise ValueError("invalid Rust execution feature receipt")
    if execution.get("package") != package_name or execution.get("features") != rust_package_features(
        cargo, packages[0]
    ):
        raise ValueError("Rust execution features differ from the selected resolved package")


def rust_test_binaries(owner):
    return {
        target["name"]: "test" if "test" in target["kind"] else "lib"
        for target in owner["targets"]
        if target.get("test") and set(target["kind"]) & {"test", "lib", "staticlib"}
    }


def rust_membership(document, package, cargo, binaries, *, bound=False):
    owner = rust_test_package(cargo, package)
    prepared = binaries["rust-binaries"]
    if set(document["rust-suites"]) != set(prepared):
        raise ValueError("missing or unexpected Rust test binary")
    if bound:
        if document.get("rust-build-meta") != binaries["rust-build-meta"]:
            raise ValueError("nextest discovery build metadata differs from prepared Rust binaries")
    else:
        expected = rust_test_binaries(owner)
        actual = {b["binary-name"] for b in prepared.values() if b["package-id"] == owner["id"]}
        if not expected or set(expected) != actual:
            raise ValueError("missing or unexpected Rust test binary")
    selected, filtered = set(), set()
    for binary, suite in document["rust-suites"].items():
        built = prepared[binary]
        package_id = built["package-id"]
        if not isinstance(package_id, str) or not package_id or suite.get("package-id") != package_id:
            raise ValueError("nextest discovery package ID differs from prepared binary: " + binary)
        if bound and any(
            suite.get(key) != built[key]
            for key in ("binary-id", "binary-name", "binary-path", "kind", "build-platform")
        ):
            raise ValueError("nextest discovery differs from prepared Rust test binary: " + binary)
        owned = package_id == owner["id"]
        for name, case in suite["testcases"].items():
            key = (binary, name)
            matches = case["filter-match"]["status"] == "matches"
            if matches != owned:
                raise ValueError("missing or unexpected Rust package function")
            if matches:
                if case["ignored"]:
                    raise ValueError("ignored required Rust function")
                selected.add(key)
            else:
                filtered.add(key)
    if not selected or (package == "core" and not any(n == OPTICS_FUNCTION for _, n in selected)):
        raise ValueError("missing required Rust function")
    return selected, filtered


def rust_core_publication(contract):
    if contract.get("grain_preview", {}).get("rust_core_publication") != PREVIEW_PUBLICATION_FUNCTION:
        raise ValueError("missing or invalid Rust core publication contract")
    return PREVIEW_PUBLICATION_FUNCTION


def require_rust_core_publication(publication, selected, cargo, binaries):
    owner = rust_test_package(cargo, "core")
    targets = rust_test_binaries(owner)
    unit_binaries = [
        binary_id
        for binary_id, binary in binaries["rust-binaries"].items()
        if binary["package-id"] == owner["id"] and targets.get(binary["binary-name"]) == binary.get("kind") == "lib"
    ]
    if len(unit_binaries) != 1 or [key for key in selected if key[1] == publication] != [
        (unit_binaries[0], publication)
    ]:
        raise ValueError("missing or duplicate required Rust core publication test: " + publication)


def rust_nextest_commands(paths, package, config):
    common = [paths["nextest"], "nextest"]
    options = [
        "--config-file",
        config,
        "--user-config-file",
        "none",
        "--profile",
        "default",
        "--binaries-metadata",
        paths["binaries-metadata"],
        "--cargo-metadata",
        paths["cargo-metadata"],
        "-E",
        "package(=" + RUST_TEST_PACKAGES[package] + ")",
    ]
    return [
        [*common, "list", *options, "--message-format", "json"],
        [*common, "run", *options, "--test-threads", "1", "--retries", "0", "--no-fail-fast", "--no-tests", "fail"],
    ]


def native_ctest_command(executable, build, junit, names):
    return [
        executable,
        "--test-dir",
        str(build),
        "--no-tests=error",
        "--output-on-failure",
        "--output-junit",
        str(junit),
        "-j",
        "1",
        "-R",
        "^(" + "|".join(map(re.escape, sorted(names))) + ")$",
    ]


def validate_historical_rust_bindings(configuration, registrations, contract):
    if (
        "rust_tests" in configuration
        or "rust_core_publication" in contract.get("grain_preview", {})
        or any(
            "--request" in registrations[name].get("command", ())
            for name in RUST_TEST_REGISTRATIONS.keys() & registrations.keys()
        )
    ):
        raise ValueError("current Rust test bindings in a historical result")


def validate_rust_test_versions(versions):
    for tool, prefix in (
        ("cargo", "cargo 1.98.1 "),
        ("rustc", "rustc 1.98.1 "),
        ("nextest", "cargo-nextest 0.9.143 "),
    ):
        output = versions.get(tool)
        if not isinstance(output, str) or not output.startswith(prefix):
            raise ValueError("qualified Rust/Cargo 1.98.1 and nextest 0.9.143 required")


def rust_test_tool_paths(candidate, version_probe, source, path):
    records = candidate["version_commands"]
    versions = {record["name"]: record["output"] for record in records}
    invocation = version_probe["command"]
    if (
        len(versions) != len(records)
        or versions != candidate["versions"]
        or any(record["returncode"] != 0 for record in records)
        or len(invocation) < 4
        or invocation[1] != "-B"
        or path(invocation[2]) != source / "scripts" / "verify.py"
        or invocation[3] != "tool-versions"
        or path(version_probe["directory"]) != source
        or invocation[4:] != [arg for row in records for arg in ("--tool", row["name"], row["command"][0])]
    ):
        raise ValueError("native tool probes differ from the recorded Rust test tool versions")
    validate_rust_test_versions(versions)
    tools = {}
    required = {"cmake", "ctest", "cargo", "nextest", "rustc"}
    for record in records:
        name = record["name"]
        if name in required:
            executable = path(record["command"][0])
            arguments = record["command"][1:]
            if name == "rustc" and arguments != ["-Vv"]:
                raise ValueError("recorded rustc probe must use -Vv")
            if not executable.is_absolute() or (name != "rustc" and arguments != ["--version"]):
                raise ValueError("invalid native tool probe for Rust test execution")
            tools[name] = executable
    if tools.keys() != required or not path(invocation[0]).is_absolute():
        raise ValueError("missing qualified tool path for Rust test execution")
    preset = candidate["preset"]
    rustc = "rustc.exe" if preset.startswith("windows-") else "rustc"
    if tools["rustc"] != tools["cargo"].with_name(rustc):
        raise ValueError("recorded rustc probe must use the executable beside Cargo")
    target = PRESET_TARGETS[preset]
    hosts = [line for line in versions["rustc"].splitlines() if line.startswith("host:")]
    if hosts != [f"host: {target}"]:
        raise ValueError("rustc host differs from configured native target: " + target)
    tools["python"] = path(invocation[0])
    return tools


def native_test_build(result, configuration):
    source = rust_test_path(result["preset"])(configuration.get("CMAKE_HOME_DIRECTORY") or "")
    cuda_free = result["build_kind"] == "cuda-free"
    if not source.is_absolute() or ".." in source.parts or (cuda_free and source.parts[-2:] != ("tests", "host")):
        raise ValueError("invalid CMake source for Rust test receipt")
    if cuda_free:
        source = source.parent.parent
    build = source / "out" / "build" / result["preset"]
    if cuda_free:
        build /= "host-tests"
    return source, build


def product_artifact_consumers(artifacts, build, replay_configured):
    """Read CMake target outputs from the configured build's artifact description."""
    path = type(build)
    if artifacts.get("schema") != "rust-artifacts-v2" or path(artifacts["build"]) != build:
        raise ValueError("product artifact description differs from the configured build")
    consumers = artifacts.get("consumers")
    if not isinstance(consumers, dict) or (MOTION_REPLAY_TARGET in consumers) != replay_configured:
        raise ValueError("configured replay availability disagrees with CMake artifact consumers")
    if replay_configured:
        binary = path(consumers[MOTION_REPLAY_TARGET]["binary"])
        if not binary.is_relative_to(build) or binary == build or ".." in binary.parts:
            raise ValueError("replay artifact consumer is outside the configured build")
    return consumers


def validate_rust_artifact_paths(artifacts, result, configuration):
    preset = result["preset"]
    path = rust_test_path(preset)
    _, build = native_test_build(result, configuration)
    windows = preset.startswith("windows-")
    native_archive = "juicer_core.lib" if windows else "libjuicer_core.a"
    if path(artifacts["build"]) != build or path(artifacts["native_archive"]) != build / native_archive:
        raise ValueError("Rust artifact manifest differs from the configured build")
    archive = "film_juicer_plugin.lib" if windows else "libfilm_juicer_plugin.a"
    target = PRESET_TARGETS[preset]
    profile = "release" if preset.endswith("-release") else "debug"
    for variant, row in artifacts["variants"].items():
        directory = build / "cargo" / variant
        if (
            path(row["target_dir"]) != directory
            or path(row["archive"]) != directory / target / profile / archive
            or path(row["feature_input"]) != directory / "features.txt"
            or path(row["stamp"]) != directory / "build.stamp"
        ):
            raise ValueError("Rust artifact paths differ from the configured build: " + variant)
    for name, consumer in artifacts["consumers"].items():
        binary = path(consumer["binary"])
        if not binary.is_relative_to(build) or binary == build or ".." in binary.parts:
            raise ValueError("Rust artifact consumer is outside the configured build: " + name)


def rust_test_invocation(result, candidate, registrations):
    path = rust_test_path(result["preset"])
    source, build = native_test_build(result, candidate["configuration"])
    phases = ["registration", "build", "versions", "ctest-host"]
    commands = [entry for entry in result["commands"] if entry["name"] in phases]
    if [entry["name"] for entry in commands] != phases:
        raise ValueError("missing, duplicate or out-of-order Rust test command")
    inventory, preparation, version_probe, ctest = commands
    tools = rust_test_tool_paths(candidate, version_probe, source, path)
    command = inventory["command"]
    if (
        len(command) != 4
        or path(command[0]) != tools["ctest"]
        or command[1] != "--test-dir"
        or path(command[2]) != build
        or command[3] != "--show-only=json-v1"
        or path(inventory["directory"]) != source
    ):
        raise ValueError("Rust registration did not use the configured build and CTest executable")
    host_names = {
        name
        for name in result["selection"]
        if any(p["name"] == "LABELS" and "host" in p["value"] for p in registrations[name]["properties"])
    }
    command = ctest["command"]
    if (
        len(command) != 11
        or path(command[0]) != tools["ctest"]
        or path(command[2]) != build
        or path(ctest["directory"]) != source
        or not set(result["selection"]).intersection(RUST_TEST_REGISTRATIONS) <= host_names
        or command != native_ctest_command(command[0], command[2], command[6], host_names)
    ):
        raise ValueError("Rust test result differs from the configured outer CTest invocation")
    junit = path(command[6])
    results = source / "out" / "validation" / result["preset"] / "results"
    if junit.name != "ctest-host.xml" or junit.parent.parent != results or ".." in junit.parts:
        raise ValueError("outer CTest JUnit path differs from the configured Rust result directory")

    command = preparation["command"]
    targets = command[3:-2]
    if targets[:1] == ["--clean-first"]:
        targets = targets[1:]
    if (
        command[1:2] != ["--build"]
        or len(command) < 7
        or path(command[0]) != tools["cmake"]
        or path(command[2]) != build
        or path(preparation["directory"]) != source
        or targets[:1] != ["--target"]
        or RUST_TEST_BUILD_TARGET not in targets[1:]
        or any(target.startswith("-") for target in targets[1:])
        or command[-2:] != ["--", "-k0"]
    ):
        raise ValueError("outer build did not prepare Rust test binaries for the configured CTest invocation")
    return source, junit.parent, tools


def rust_test_registration(test, preset, request, source, tools):
    command = test.get("command", [])
    values = command[3:]
    options = dict(zip(values[::2], values[1::2]))
    path = rust_test_path(preset)
    if (
        len(command) < 3
        or path(command[0]) != path(tools["python"])
        or path(command[1]) != source / "scripts" / "verify.py"
        or command[2] != "rust-tests"
        or len(values) != 2 * len(options)
        or options.keys() != {"--package", "--preset", "--request", *("--" + flag for flag in RUST_TEST_PATH_FLAGS)}
        or options["--package"] != RUST_TEST_REGISTRATIONS[test["name"]]
        or options["--preset"] != preset
        or json.loads(options["--request"]) != request
    ):
        raise ValueError("Rust CTest registration differs from configured request: " + test["name"])
    paths = {flag: str(path(options["--" + flag])) for flag in RUST_TEST_PATH_FLAGS}
    if any(path(paths[tool]) != path(tools[tool]) for tool in ("cargo", "nextest")):
        raise ValueError("Rust CTest tools differ from version probes: " + test["name"])
    validate_rust_test_paths(paths, source, preset)
    return paths


def rust_junit_run_id(report):
    run_id = report.get("uuid", "")
    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", run_id):
        raise ValueError("missing or invalid nextest JUnit run ID")
    return run_id


def ctest_rust_run_ids(case):
    return [prop.get("value") for prop in case.findall(f"./properties/property[@name='{RUST_JUNIT_RUN_ID_PROPERTY}']")]


def validate_retained_rust_junit_times(report, suite, case):
    # Older receipts have CTest's local, whole-second start and duration,
    # the individual CTest case duration, and nextest's finer-grained times.
    ctest_start = datetime.fromisoformat(suite.attrib["timestamp"])
    nextest_start = datetime.fromisoformat(report.attrib["timestamp"])
    started_after = (nextest_start.replace(tzinfo=None) - ctest_start).total_seconds()
    suite_seconds = float(suite.attrib["time"])
    if not 0 <= started_after < suite_seconds + 1:
        raise ValueError("nextest JUnit timestamp is outside the retained CTest run")
    cases = suite.findall("testcase")
    preceding_times = [float(previous.attrib["time"]) for previous in cases[: cases.index(case)]]
    if any(not 0 <= seconds < float("inf") for seconds in preceding_times):
        raise ValueError("invalid preceding CTest case duration")
    # The recorded host invocation is serial. Allow for whole-second CTest timing.
    if started_after + 1 < sum(preceding_times):
        raise ValueError("nextest JUnit starts before preceding CTest cases finish")
    nextest_seconds = float(report.attrib["time"])
    if not 0 <= nextest_seconds <= float(case.attrib["time"]):
        raise ValueError("nextest JUnit duration exceeds the retained CTest case")
    if started_after + nextest_seconds > suite_seconds + 2:
        raise ValueError("nextest JUnit report ends after the retained CTest run")
    for terminal in report.iter("testcase"):
        if terminal.find("skipped") is not None:
            continue
        started_after = (datetime.fromisoformat(terminal.attrib["timestamp"]) - nextest_start).total_seconds()
        if not 0 <= started_after <= nextest_seconds:
            raise ValueError("nextest JUnit case timestamp is outside the reported run")


def rust_receipt_membership(
    payloads,
    name,
    *,
    preview,
    require_resolved=False,
    request=None,
    preset=None,
    source=None,
    run_dir=None,
    paths=None,
    contract=None,
    ctest_witness=None,
    receipt_schema=None,
):
    package = RUST_TEST_REGISTRATIONS[name]
    execution = json.loads(payloads[name + ".execution.json"])
    cargo = json.loads(payloads["rust-cargo.json"])
    binaries = json.loads(payloads["rust-binaries.json"])
    discovery = json.loads(payloads[name + ".discovery.json"])
    junit = payloads[name + ".xml"]
    publication = None
    execution_schema = RUST_EXECUTION_SCHEMAS.get(receipt_schema)
    if request is not None:
        owner = rust_test_package(cargo, package)
        features = validate_rust_test_build(request, preset, source, cargo, binaries)
        expected_binaries = sorted([owner["id"], target] for target in rust_test_binaries(owner))
        if (
            execution.get("schema") != execution_schema
            or execution.get("registration") != name
            or execution.get("request") != request
            or type(execution["request"]["grain_preview"]) is not bool
            or execution.get("package") != RUST_TEST_PACKAGES[package]
            or execution.get("package_id") != owner["id"]
            or execution.get("features") != features
            or execution.get("cargo") != paths["cargo"]
            or execution.get("required_binaries") != expected_binaries
            or execution.get("discovered_binaries") != sorted(discovery["rust-suites"])
        ):
            raise ValueError("Rust execution binding differs from configured registration, package or binaries")
        config_name = name + ".nextest.toml"
        commands = [entry["command"] for entry in execution["commands"]]
        if commands != rust_nextest_commands(paths, package, str(run_dir / config_name)):
            raise ValueError("Rust nextest commands differ from configured CTest registration")
        config = tomllib.loads(payloads[config_name].decode("utf-8"))
        junit_path = config["profile"]["default"]["junit"]["path"]
        if rust_test_path(preset)(junit_path) != run_dir / (name + ".xml"):
            raise ValueError("Rust nextest JUnit path differs from the configured result directory")
        if execution_schema == "rust-test-execution-v4":
            if execution.get("junit_sha256") != hashlib.sha256(junit).hexdigest():
                raise ValueError("Rust JUnit digest differs from the recorded execution")
        elif "junit_sha256" in execution:
            raise ValueError("current Rust JUnit binding in an older result")
        if preview and package == "core":
            publication = rust_core_publication(contract)
    else:
        validate_rust_execution_features(execution, package, cargo, require_resolved=require_resolved)
        if preview and execution.get("schema") == "rust-test-execution-v2":
            validate_rust_workspace_features(cargo, preview)
            if package == "core":
                publication = PREVIEW_PUBLICATION_FUNCTION
    expected, filtered = rust_membership(discovery, package, cargo, binaries, bound=request is not None)
    report = framework_terminals(junit, expected, nextest=True, filtered=filtered)
    if ctest_witness is not None:
        suite, case = ctest_witness
        run_ids = ctest_rust_run_ids(case)
        if receipt_schema in {"native-tests-v4", "native-tests-v5", "native-tests-v6"}:
            if run_ids != [rust_junit_run_id(report)]:
                raise ValueError("Rust JUnit run ID differs from the retained CTest case")
        elif run_ids:
            raise ValueError("current Rust JUnit run ID in an older result")
        elif execution_schema is not None:
            validate_retained_rust_junit_times(report, suite, case)
    if publication is not None:
        require_rust_core_publication(publication, expected, cargo, binaries)
    if (
        len(execution["commands"]) != 2
        or any(c["returncode"] != 0 for c in execution["commands"])
        or execution["doctests"]["status"] != "none"
        or execution["doctests"]["code_fence_sources"]
        or execution["doctests"]["source_files_scanned"] < 1
    ):
        raise ValueError("incomplete Rust execution")
    return expected


def domain_result_members(result):
    runtime = "gpu" if result.get("schema") == "native-gpu-v1" else "host"
    if (
        result.get("schema") not in {"native-host-v2", "native-gpu-v1"}
        or result.get("status") not in {"passed", "failed"}
        or result.get("preset") not in {"linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"}
        or result.get("shipping_isolation") is not False
        or result.get("scope") not in {"focused", runtime + "-domains"}
    ):
        raise ValueError("invalid host result metadata")
    members = result["members"]
    if (
        not {
            "candidate.json",
            "candidate.patch",
            runtime + "-contracts.json",
            runtime + "-targets.json",
            "registration.json",
        }
        <= members.keys()
    ):
        raise ValueError("missing required host metadata")
    selection = result["selection"]
    if (
        not selection
        or len(selection) != len(set(selection))
        or any(not isinstance(n, str) or not n for n in selection)
    ):
        raise ValueError("invalid host selection")
    required = {
        "candidate.json",
        "candidate.patch",
        runtime + "-contracts.json",
        runtime + "-targets.json",
        "registration.json",
    }
    if result["status"] == "passed":
        if runtime == "gpu":
            required.add("gpu-exclusion.json")
        required.add("ctest.xml")
        required.add("target-discovery.json")
        case_files = result["case_files"]
        if set(case_files) != set(selection) - {"Rust.Core", "Rust.Bridge"}:
            raise ValueError("missing host case file contract")
        keys = set()
        for entry in case_files.values():
            key = entry["key"]
            if (
                not key.startswith("host-")
                or len(key) != 21
                or any(c not in "0123456789abcdef" for c in key[5:])
                or key in keys
                or entry["protocol"] not in {"gtest", "custom"}
                or not isinstance(entry["rows"], bool)
            ):
                raise ValueError("invalid host case file contract")
            keys.add(key)
            required.add(key + ".execution.json")
            required.add(key + ".jsonl")
            if entry["protocol"] == "gtest":
                required.update({key + ".discovery.json", key + ".xml"})
            if entry["protocol"] == "custom" or entry["rows"]:
                required.add(key + ".list.jsonl")
        for name in set(selection) & {"Rust.Core", "Rust.Bridge"}:
            required.update(
                name + suffix
                for suffix in (".discovery.json", ".xml", ".execution.json", ".nextest.toml", ".rows.jsonl")
            )
            required.update({"rust-cargo.json", "rust-binaries.json"})
        if "Rust.Core" in selection and "optics_rows" in result:
            # Earlier completed bundles used a second optics stream (offline only).
            rows = result["optics_rows"]
            if len(rows) != 553 or len(set(rows)) != 553 or any(not isinstance(r, str) or not r for r in rows):
                raise ValueError("invalid Rust optics row membership")
            required.add("rust.jsonl")
    if not required <= members.keys():
        raise ValueError("missing required host evidence")
    for name, spec in members.items():
        if (
            PurePosixPath(name).parts != (name,)
            or any(c in name for c in "\\:")
            or name in {".", ".."}
            or PurePosixPath(name).is_absolute()
        ):
            raise ValueError("unconfined host result reference")
        if type(spec["bytes"]) is not int or spec["bytes"] < 0 or len(spec["sha256"]) != 64:
            raise ValueError("invalid host member identity")
    return members


def validate_domain_result(result, payloads):
    members = domain_result_members(result)
    if members.keys() != payloads.keys():
        raise ValueError("missing required host member")
    for name, spec in members.items():
        if len(payloads[name]) != spec["bytes"] or hashlib.sha256(payloads[name]).hexdigest() != spec["sha256"]:
            raise ValueError("damaged host result member: " + name)
    candidate = json.loads(payloads["candidate.json"])
    if candidate["head"] != result["candidate_head"] or candidate["preset"] != result["preset"]:
        raise ValueError("host candidate disagreement")
    if result["status"] == "failed":
        if any(c["drained"] is not True for c in result["commands"]):
            raise ValueError("undrained host execution")
        return {"status": "failed", "preset": result["preset"], "scope": result["scope"], "shipping_isolation": False}
    runtime = "gpu" if result["schema"] == "native-gpu-v1" else "host"
    contract = json.loads(payloads[runtime + "-contracts.json"])
    inventory = json.loads(payloads["registration.json"])
    native = native_registration(
        contract,
        json.loads(payloads[runtime + "-targets.json"]),
        inventory,
        runtime=runtime,
    )
    available = set(native) | ({"Rust.Core", "Rust.Bridge"} if runtime == "host" else set())
    selected = set(result["selection"])
    if not selected <= available or (result["scope"] == runtime + "-domains" and selected != available):
        raise ValueError("missing selected host domain")
    if selected & RUST_TEST_REGISTRATIONS.keys():
        validate_historical_rust_bindings(
            candidate["configuration"], {test["name"]: test for test in inventory["tests"]}, contract
        )
    if not result["commands"] or any(c["drained"] is not True for c in result["commands"]):
        raise ValueError("undrained host execution")
    counts = {"rust_functions": 0, "native_functions": 0, "custom_cases": 0, "rust_cases": 0}
    excluded_benchmarks = {tuple(n.split(".", 1)) for n in contract.get("excluded_gtest", {})}
    excluded_seen = set()
    if result["status"] == "passed":
        expected_type = "Debug" if result["preset"].endswith("debug") else "Release"
        if (
            candidate["configuration"].get("BUILD_TESTING") != "ON"
            or candidate["configuration"].get("CMAKE_BUILD_TYPE") != expected_type
            or result.get("candidate_unchanged") is not True
        ):
            raise ValueError("changed/incorrect host candidate configuration")
        if any(c["returncode"] != 0 or c["process_status"] != "completed" for c in result["commands"]):
            raise ValueError("failed command cannot establish host success")
        if runtime == "gpu":
            if (
                result.get("physical_gpu") != {"contender_excluded": True, "drained_while_locked": True}
                or json.loads(payloads["gpu-exclusion.json"]) != {"physical_gpu": "excluded"}
                or not any(c["name"] == "gpu-exclusion" for c in result["commands"])
                or not any(name.startswith("Resources/") for name in candidate["inputs"])
            ):
                raise ValueError("GPU execution lacks physical-device exclusion/drainage or runtime resources")
        ctest = list(ET.fromstring(payloads["ctest.xml"]).iter("testcase"))
        if sorted(c.get("name") for c in ctest) != sorted(selected) or any(
            c.get("status") != "run" or any(c.find(t) is not None for t in ("failure", "error", "skipped"))
            for c in ctest
        ):
            raise ValueError("missing/failed/skipped host CTest case")
        if any(ctest_rust_run_ids(case) for case in ctest if case.get("name") in RUST_TEST_REGISTRATIONS):
            raise ValueError("current Rust JUnit run ID in an older result")
        target_discovery = json.loads(payloads["target-discovery.json"])
        target_seen = {target: set() for target in target_discovery}
        for name in sorted(selected):
            if name in RUST_TEST_REGISTRATIONS:
                expected = rust_receipt_membership(
                    payloads,
                    name,
                    preview=grain_preview_enabled(candidate["configuration"].get("JUICER_GRAIN_PREVIEW")),
                )
                counts["rust_functions"] += len(expected)
                counts["rust_cases"] += rust_case_events(payloads[name + ".rows.jsonl"], contract, expected)
                if name == "Rust.Core" and "optics_rows" in result:
                    counts["rust_cases"] += optics_events(
                        payloads["rust.jsonl"], {("rust", r) for r in result["optics_rows"]}
                    )
                continue
            entry = native[name]
            if result["case_files"][name] != {k: entry[k] for k in ("key", "protocol", "rows")}:
                raise ValueError("host case file contract disagrees with registration")
            key = entry["key"]
            execution = json.loads(payloads[key + ".execution.json"])
            if (
                execution["returncode"] != 0
                or execution["command"] != entry["command"]
                or execution.get("outcomes_valid") is not True
            ):
                raise ValueError("failed/mismatched native host command")
            if entry["protocol"] == "gtest":
                discovered = gtest_discovery(payloads[key + ".discovery.json"], excluded=excluded_benchmarks)
                filtered = discovered & excluded_benchmarks
                expected = discovered - filtered
                if not expected:
                    raise ValueError("empty native framework selection")
                excluded_seen.update(filtered)
                framework_terminals(payloads[key + ".xml"], expected, filtered=filtered)
                if entry["target"] not in target_seen or target_seen[entry["target"]] & expected:
                    raise ValueError("missing target discovery or duplicate GoogleTest coverage")
                target_seen[entry["target"]].update(expected)
                counts["native_functions"] += len(expected)
                cohorts = {s + "." + n: contract["gtest_cohorts"].get(s + "." + n, 0) for s, n in expected}
            else:
                cohorts = native_cohort_count(contract, name, result["preset"])
            expected_rows = set()
            if entry["protocol"] == "custom" or entry["rows"]:
                expected_rows = optics_keys([json.loads(line) for line in payloads[key + ".list.jsonl"].splitlines()])
                if not expected_rows:
                    raise ValueError("empty custom host case selection")
            terminals = case_events(payloads[key + ".jsonl"], expected_rows, cohorts=cohorts)
            if entry["protocol"] == "custom":
                native_membership(contract, name, expected_rows, terminals)
            counts["custom_cases"] += terminals
        for target, discovery in target_discovery.items():
            expected = gtest_discovery(json.dumps(discovery), excluded=excluded_benchmarks) - excluded_benchmarks
            if not target_seen[target] <= expected or (
                result["scope"] == runtime + "-domains" and target_seen[target] != expected
            ):
                raise ValueError("GoogleTest registration omitted required functions")
    if runtime == "gpu":
        if result["scope"] == "gpu-domains" and excluded_seen != excluded_benchmarks:
            raise ValueError("missing declared benchmark discovery")
        counts["excluded_benchmarks"] = sorted(".".join(n) for n in excluded_seen)
        counts["unselected_registrations"] = sorted(available - selected)
    return {
        "status": result["status"],
        "preset": result["preset"],
        "scope": result["scope"],
        "ctest_cases": len(selected),
        **counts,
        "shipping_isolation": False,
    }


def read_native_result(bundle, *, result_bytes=None):
    """Read each required member once, without tools or originating paths."""
    started = time.perf_counter()
    bundle = Path(bundle).resolve(strict=True)
    result_path = bundle / "result.json"

    def regular(path):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            return None
        return info

    if regular(result_path) is None:
        raise ValueError("linked result metadata")
    raw = result_path.read_bytes() if result_bytes is None else result_bytes
    result = json.loads(raw)
    validators = {
        "native-tests-v1": (tests_result_members, validate_tests_result),
        "native-tests-v2": (tests_result_members, validate_tests_result),
        "native-tests-v3": (tests_result_members, validate_tests_result),
        "native-tests-v4": (tests_result_members, validate_tests_result),
        "native-tests-v5": (tests_result_members, validate_tests_result),
        "native-tests-v6": (tests_result_members, validate_tests_result),
        "native-quality-v1": (quality_result_members, validate_quality_result),
        "native-quality-v2": (quality_result_members, validate_quality_result),
        "native-quality-v3": (quality_result_members, validate_quality_result),
        "native-build-v1": (build_result_members, validate_build_result),
    }
    host = result.get("schema", "").startswith("native-host-") or result.get("schema") == "native-gpu-v1"
    metadata, validate = validators.get(
        result.get("schema"),
        (domain_result_members, validate_domain_result) if host else (_native_metadata, validate_native_result),
    )
    metadata(result)  # Reject incomplete/unconfined metadata before payload I/O.
    for name, spec in result["members"].items():
        target = bundle / name
        info = regular(target)
        if info is None:
            raise ValueError("linked bundle member")
        if info.st_size != spec["bytes"]:
            raise ValueError("damaged bundle member size: " + name)
    payloads = {name: (bundle / name).read_bytes() for name in result["members"]}
    summary = validate(result, payloads)
    candidate = json.loads(payloads["candidate.json"])
    source = {key: candidate.get(key) for key in ("head", "inputs", "new_files_base64")}
    summary["candidate_source"] = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode() + b"\0" + payloads["candidate.patch"]
    ).hexdigest()
    summary.update(
        bookkeeping_read_bytes=len(raw) + sum(map(len, payloads.values())), report_seconds=time.perf_counter() - started
    )
    return summary


def validate_python_result(discovery, functions_payload, rows_payload):
    """Standard unittest function outcomes plus independently declared row tables."""
    functions = discovery["functions"]
    names = {test["id"]: test for test in functions}
    if not names or len(names) != len(functions) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("empty or duplicate Python function membership")
    if functions_payload and not functions_payload.endswith(b"\n"):
        raise ValueError("truncated Python function outcomes")
    started, finished, skipped = set(), set(), set()
    for line in functions_payload.splitlines():
        event = json.loads(line)
        name = event["id"]
        if name not in names:
            raise ValueError("unexpected Python function")
        if event["event"] == "start" and name not in started:
            started.add(name)
        elif event["event"] == "terminal" and name in started and name not in finished:
            intended = names[name]
            if intended["skip"]:
                if event["outcome"] != "skipped" or event.get("reason") != intended["skip_reason"]:
                    raise ValueError("unexpected Python skip outcome")
                skipped.add(name)
            elif event["outcome"] != "passed":
                raise ValueError("failed or unexpected skipped Python function")
            finished.add(name)
        else:
            raise ValueError("duplicate or unordered Python function outcome")
    if started != names.keys() or finished != names.keys():
        raise ValueError("missing Python function outcomes")
    rows = case_events(
        rows_payload, cohorts={name: 1 for name, test in names.items() if test["rows"] and name not in skipped}
    )
    return {"python_functions": len(names) - len(skipped), "python_rows": rows, "python_skips": sorted(skipped)}


def membership(inventory, groups, runtime):
    """Select current CTest relevance/runtime; no source or machine-path access."""
    if not set(groups) <= GROUPS or not groups or runtime not in {"host", "gpu", "all"}:
        raise ValueError("invalid checkpoint selection")
    selected = {group: [] for group in groups}
    seen, scratch = set(), {}
    for test in inventory["tests"]:
        name = test["name"]
        if name in seen:
            raise ValueError("duplicate CTest registration: " + name)
        seen.add(name)
        properties = {p["name"]: p["value"] for p in test.get("properties", ())}
        labels = set(properties.get("LABELS", ()))
        relevance, prerequisite = labels & GROUPS, labels & {"host", "gpu"}
        if len(relevance) != 1 or len(prerequisite) != 1:
            raise ValueError("CTest group needs one relevance and a runtime prerequisite: " + name)
        group = next(iter(relevance))
        if group not in selected or runtime != "all" and runtime not in prerequisite:
            continue
        if properties.get("DISABLED") or properties.get("RUN_SERIAL"):
            raise ValueError("required CTest group is disabled or globally serialized: " + name)
        command = test.get("command", ())
        if "--scratch-root" in command:
            i = command.index("--scratch-root")
            if command.count("--scratch-root") != 1 or i + 1 == len(command):
                raise ValueError("checkpoint group needs one scratch root")
            value = command[i + 1]
            path = PureWindowsPath(value) if PureWindowsPath(value).drive else PurePosixPath(value)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("checkpoint scratch root must be absolute")
            key = path.as_posix().casefold() if isinstance(path, PureWindowsPath) else path.as_posix()
            if key in scratch:
                raise ValueError("checkpoint groups have colliding scratch roots")
            scratch[key] = name
        selected[group].append(name)
    if any(not names for names in selected.values()):
        raise ValueError("required checkpoint selection is empty")
    return {group: sorted(names) for group, names in selected.items()}


def python_registration(contract, inventory, preset, *, standalone=False, cuda_free=False):
    if contract.get("schema") != "python-contracts-v1":
        raise ValueError("invalid Python registration contract")
    configuration = "Debug" if preset.endswith("debug") else "Release"
    platform = "windows" if preset.startswith("windows") else "linux"
    expected = {
        name: row
        for name, row in contract["registrations"].items()
        if configuration in row.get("configurations", ["Debug", "Release"])
        and platform in row.get("platforms", ["linux", "windows"])
        and (not standalone or row.get("tooling", False))
        and (not cuda_free or row.get("tooling", False) or row.get("cuda_free", False))
    }
    actual = {}
    for test in inventory["tests"]:
        command = test.get("command", [])
        if "python-tests" not in command:
            continue
        name = test["name"]
        if name in actual or name not in expected:
            raise ValueError("duplicate or unexpected Python registration: " + name)
        original = command[command.index("--") + 1 :]
        owner = expected[name]["source"]
        if "-m" in original and "unittest" in original:
            if "-p" not in original or original[original.index("-p") + 1] != PurePosixPath(owner).name:
                raise ValueError("Python registration changed its owning suite")
            source = original[original.index("-s") + 1].replace("\\", "/")
            if not source.endswith(PurePosixPath(owner).parent.as_posix()):
                raise ValueError("Python registration changed its owning directory")
        else:
            script_arguments = original[1:]
            while script_arguments and script_arguments[0] in {"-B", "-u"}:
                script_arguments.pop(0)
            if (
                not script_arguments
                or script_arguments[0].replace("\\", "/").split("/")[-1] != PurePosixPath(owner).name
            ):
                raise ValueError("Python registration changed its owning script")
        key = "host-" + hashlib.sha256(json.dumps(original, ensure_ascii=False).encode()).hexdigest()[:16]
        actual[name] = {"command": original, "key": key, "protocol": "python", "rows": False}
    if actual.keys() != expected.keys():
        raise ValueError("missing required Python registration: " + ", ".join(sorted(expected.keys() - actual.keys())))
    return actual


def _flat_members(result, required):
    if (
        result.get("preset") not in PRESETS
        or result.get("status") not in {"passed", "failed"}
        or result.get("shipping_isolation") is not False
    ):
        raise ValueError("invalid native run metadata")
    members = result["members"]
    if not required <= members.keys():
        raise ValueError("missing required native run members")
    for name, spec in members.items():
        if (
            PurePosixPath(name).parts != (name,)
            or any(c in name for c in "\\:")
            or name in {".", ".."}
            or PurePosixPath(name).is_absolute()
        ):
            raise ValueError("unconfined result reference")
        if (
            type(spec["bytes"]) is not int
            or spec["bytes"] < 0
            or len(spec["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in spec["sha256"])
        ):
            raise ValueError("invalid result member identity")
    return members


def _payloads(result, payloads):
    if payloads.keys() != result["members"].keys():
        raise ValueError("missing result payload")
    for name, data in payloads.items():
        spec = result["members"][name]
        if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError("result payload identity mismatch: " + name)
    candidate = json.loads(payloads["candidate.json"])
    if candidate["head"] != result["candidate_head"] or candidate["preset"] != result["preset"]:
        raise ValueError("candidate metadata mismatch")
    if result["status"] == "passed":
        if (
            result.get("candidate_unchanged") is not True
            or not result["commands"]
            or not isinstance(candidate.get("inputs"), dict)
            or not candidate["inputs"]
        ):
            raise ValueError("changed or incomplete native candidate")
        if any(
            c.get("drained") is not True or c.get("returncode") != 0 or c.get("process_status") != "completed"
            for c in result["commands"]
        ):
            raise ValueError("failed or undrained native command")
    return candidate


def require_candidate_source(candidate, payloads, source):
    """Bind a retained source member to the independently captured candidate."""
    data = payloads[PurePosixPath(source).name]
    identity = candidate["inputs"].get(source)
    if not isinstance(identity, dict):
        raise ValueError("missing candidate source identity: " + source)
    if "worktree_sha256" in identity or "sha256" in identity:
        matches = hashlib.sha256(data).hexdigest() == identity.get("worktree_sha256", identity.get("sha256"))
    else:
        # Earlier receipts retained the Git index and the full-index patch against HEAD.
        # These JSON sources are tracked as LF text.
        name = source.encode()
        header = b"\ndiff --git a/" + name + b" b/" + name + b"\n"
        section = (b"\n" + payloads["candidate.patch"]).partition(header)[2].split(b"\ndiff --git ", 1)[0]
        postimage = re.search(rb"^index [0-9a-f]{40}\.\.([0-9a-f]{40})\b", section, re.MULTILINE)
        expected = postimage[1].decode() if postimage else identity.get("git_blob")
        data = data.replace(b"\r\n", b"\n")
        blob = b"blob " + str(len(data)).encode() + b"\0" + data
        matches = hashlib.sha1(blob, usedforsecurity=False).hexdigest() == expected
    if not matches:
        raise ValueError("retained source differs from candidate: " + source)


def tests_result_members(result):
    if result.get("scope") not in {"focused", "checkpoint"} or result.get("runtime") not in {"host", "gpu", "all"}:
        raise ValueError("invalid native test scope")
    required = {"candidate.json", "candidate.patch", "registration.json", "target-discovery.json"}
    for runtime in ("host", "gpu", "python"):
        required.update({runtime + "-contracts.json", runtime + "-targets.json"})
    if result.get("rust_artifacts"):
        required.add("rust-artifacts.json")
    selected = result["selection"]
    if result.get("build_kind") == "cuda-free" and result["scope"] != "focused":
        raise ValueError("CUDA-free results must remain focused; they are not product checkpoints")
    if (
        result.get("build_kind") not in {"product", "tooling", "cuda-free"}
        or not set(result["groups"]) <= GROUPS
        or len(selected) != len(set(selected))
        or result["status"] == "passed"
        and not selected
    ):
        raise ValueError("invalid or empty native test membership")
    if result["status"] == "passed":
        required.add("ctest.xml")
        quality_files = result.get("quality_files", [])
        quality_commands = [command for command in result["commands"] if command["name"] == "quality"]
        if bool(quality_files) != bool(quality_commands) or len(quality_commands) > 1:
            raise ValueError("missing or unexpected selected-source quality execution")
        if quality_files:
            required.add("quality.checks.json")
            required.update(
                "quality-boundary" + suffix
                for suffix in (".execution.json", ".discovery.json", ".functions.jsonl", ".rows.jsonl")
            )
        if result.get("physical_gpu"):
            required.add("gpu-exclusion.json")
        files = result["case_files"]
        if files.keys() != set(selected) - {"Rust.Core", "Rust.Bridge"}:
            raise ValueError("missing case file membership")
        for entry in files.values():
            key = entry["key"]
            if not key.startswith("host-") or len(key) != 21 or any(c not in "0123456789abcdef" for c in key[5:]):
                raise ValueError("invalid case member key")
            required.add(key + ".execution.json")
            if entry["protocol"] == "python":
                required.update(key + suffix for suffix in (".discovery.json", ".functions.jsonl", ".rows.jsonl"))
            elif entry["protocol"] in {"gtest", "custom"}:
                required.add(key + ".jsonl")
                if entry["protocol"] == "gtest":
                    required.update({key + ".discovery.json", key + ".xml"})
                if entry["protocol"] == "custom" or entry["rows"]:
                    required.add(key + ".list.jsonl")
            else:
                raise ValueError("unknown native case protocol")
        for name in set(selected) & {"Rust.Core", "Rust.Bridge"}:
            required.update(
                name + suffix
                for suffix in (".discovery.json", ".xml", ".execution.json", ".nextest.toml", ".rows.jsonl")
            )
            required.update({"rust-cargo.json", "rust-binaries.json"})
        if "Rust.Core" in selected and "optics_rows" in result:
            # Earlier completed bundles used a second optics stream (offline only).
            if len(result["optics_rows"]) != 553 or len(set(result["optics_rows"])) != 553:
                raise ValueError("invalid optics membership")
            required.add("rust.jsonl")
    return _flat_members(result, required)


def _native_cases(result, payloads, entries, contract, selected, rust_receipts):
    counts = {"rust_functions": 0, "native_functions": 0, "custom_cases": 0, "rust_cases": 0}
    full_discovery = json.loads(payloads["target-discovery.json"])
    excluded = {tuple(name.split(".", 1)) for name in contract.get("excluded_gtest", {})}
    seen, excluded_seen = {}, set()
    for name in sorted(selected):
        if name in rust_receipts:
            expected = rust_receipts[name]
            counts["rust_functions"] += len(expected)
            counts["rust_cases"] += rust_case_events(payloads[name + ".rows.jsonl"], contract, expected)
            if name == "Rust.Core" and "optics_rows" in result:
                counts["rust_cases"] += optics_events(
                    payloads["rust.jsonl"], {("rust", row) for row in result["optics_rows"]}
                )
            continue
        entry = entries[name]
        if result["case_files"][name] != {k: entry[k] for k in ("key", "protocol", "rows")}:
            raise ValueError("native case contract differs from registration")
        key = entry["key"]
        execution = json.loads(payloads[key + ".execution.json"])
        if (
            execution["returncode"] != 0
            or execution["command"] != entry["command"]
            or execution.get("outcomes_valid") is not True
        ):
            raise ValueError("failed or mismatched native command")
        if entry["protocol"] == "gtest":
            discovered = gtest_discovery(payloads[key + ".discovery.json"], excluded=excluded)
            filtered = discovered & excluded
            expected = discovered - filtered
            if not expected or seen.get(entry["target"], set()) & expected:
                raise ValueError("empty or duplicate native function coverage")
            framework_terminals(payloads[key + ".xml"], expected, filtered=filtered)
            seen.setdefault(entry["target"], set()).update(expected)
            excluded_seen.update(filtered)
            counts["native_functions"] += len(expected)
            cohorts = {
                suite + "." + function: contract["gtest_cohorts"].get(suite + "." + function, 0)
                for suite, function in expected
            }
        else:
            cohorts = native_cohort_count(contract, name, result["preset"])
        intended = set()
        if entry["protocol"] == "custom" or entry["rows"]:
            intended = optics_keys([json.loads(line) for line in payloads[key + ".list.jsonl"].splitlines()])
            if not intended:
                raise ValueError("empty custom case discovery")
        terminals = case_events(payloads[key + ".jsonl"], intended, cohorts=cohorts)
        if entry["protocol"] == "custom":
            native_membership(contract, name, intended, terminals)
        counts["custom_cases"] += terminals
    for target, actual in seen.items():
        if target not in full_discovery:
            raise ValueError("missing native target discovery")
        expected = gtest_discovery(json.dumps(full_discovery[target]), excluded=excluded) - excluded
        whole_target = {n for n, entry in entries.items() if entry["target"] == target} <= set(selected)
        if not actual <= expected or whole_target and actual != expected:
            raise ValueError("GoogleTest registration omitted required functions")
    return counts, excluded_seen


def validate_tests_result(result, payloads):
    tests_result_members(result)
    candidate = _payloads(result, payloads)
    preview_enabled = grain_preview_enabled(candidate.get("configuration", {}).get("JUICER_GRAIN_PREVIEW"))
    current_product = (
        result.get("schema") in {"native-tests-v5", "native-tests-v6"} and result["build_kind"] == "product"
    )
    if current_product and result["status"] == "passed":
        preview_enabled, replay_configured = read_preview_and_replay_configuration(candidate.get("configuration"))
    rust_selected = set(result["selection"]) & RUST_TEST_REGISTRATIONS.keys()
    artifacts_retained = "rust-artifacts.json" in payloads
    if (
        result["status"] == "passed"
        and result["build_kind"] == "product"
        and (result["schema"] in RUST_EXECUTION_SCHEMAS or (preview_enabled and rust_selected))
        and not artifacts_retained
    ):
        raise ValueError("product Rust receipt requires its separate artifact description")
    if (
        bool(candidate.get("features", {}).get("rust_artifacts")) != artifacts_retained
        or bool(result.get("rust_artifacts")) != artifacts_retained
    ):
        raise ValueError("missing separate Rust artifact description")
    artifact_schema = None
    if artifacts_retained:
        artifacts = json.loads(payloads["rust-artifacts.json"])
        variants = artifacts["variants"]
        artifact_schema = artifacts["schema"]
        production_features = ["grain-preview"] if artifact_schema == "rust-artifacts-v2" and preview_enabled else []
        if (
            artifact_schema not in {"rust-artifacts-v1", "rust-artifacts-v2"}
            or set(variants) != {"production", "test-support"}
            or variants["production"]["archive"] == variants["test-support"]["archive"]
            or variants["production"]["target_dir"] == variants["test-support"]["target_dir"]
            or variants["production"]["features"] != production_features
            or variants["test-support"]["features"] != ["test-support"]
            or candidate["features"].get("rust_artifacts") != "rust-artifacts.json"
        ):
            raise ValueError("invalid separate Rust artifact description")
    if result["status"] == "failed":
        return {
            "status": "failed",
            "preset": result["preset"],
            "scope": result["scope"],
            "groups": result["groups"],
            "runtime": result["runtime"],
            "ctest_cases": 0,
            "shipping_isolation": False,
            "error": result.get("error", "native execution failed"),
        }
    if result["build_kind"] == "product":
        if result["schema"] in RUST_EXECUTION_SCHEMAS and artifact_schema != "rust-artifacts-v2":
            raise ValueError("current product receipt requires the v2 Rust artifact description")
        if artifact_schema == "rust-artifacts-v2":
            validate_rust_artifact_paths(artifacts, result, candidate["configuration"])
    artifact_consumers = None
    if current_product:
        _, build = native_test_build(result, candidate["configuration"])
        artifact_consumers = product_artifact_consumers(artifacts, build, replay_configured)
    inventory = json.loads(payloads["registration.json"])
    if current_product:
        for runtime in ("host", "gpu", "python"):
            require_candidate_source(candidate, payloads, f"tests/{runtime}-contracts.json")
    contracts = {runtime: json.loads(payloads[runtime + "-contracts.json"]) for runtime in ("host", "gpu", "python")}
    if current_product:
        for runtime in ("host", "gpu"):
            contracts[runtime] = configured_product_contract(contracts[runtime], runtime, candidate["configuration"])
    standalone = result["build_kind"] == "tooling"
    cuda_free = result["build_kind"] == "cuda-free"
    native = (
        {}
        if standalone
        else {
            runtime: native_registration(
                contracts[runtime],
                json.loads(payloads[runtime + "-targets.json"]),
                inventory,
                runtime=runtime,
                cuda_free=cuda_free,
                artifact_consumers=artifact_consumers,
            )
            for runtime in ("host", "gpu")
        }
    )
    python = python_registration(
        contracts["python"], inventory, result["preset"], standalone=standalone, cuda_free=cuda_free
    )
    available = set(python)
    if not standalone:
        available.update({"Rust.Core", "Rust.Bridge"})
        available.update(native["host"])
        available.update(native["gpu"])
    registered = {test["name"] for test in inventory["tests"]}
    if registered != available:
        raise ValueError("missing or unowned public CTest registration")
    selected = set(result["selection"])
    if not selected <= available:
        raise ValueError("unknown selected test")
    requested = membership(inventory, result["groups"], result["runtime"]) if result["scope"] == "checkpoint" else None
    if requested is not None and selected != {name for names in requested.values() for name in names}:
        raise ValueError("checkpoint omitted required registration")
    counts = {
        "rust_functions": 0,
        "native_functions": 0,
        "custom_cases": 0,
        "rust_cases": 0,
        "python_functions": 0,
        "python_rows": 0,
        "python_skips": [],
    }
    excluded = set()
    if result["status"] == "passed":
        configuration = candidate["configuration"]
        if configuration["BUILD_TESTING"] != "ON" or configuration["CMAKE_BUILD_TYPE"] != (
            "Debug" if result["preset"].endswith("debug") else "Release"
        ):
            raise ValueError("incorrect testing configuration")
        preview = not cuda_free and preview_enabled
        request = None
        source = run_dir = tools = None
        registrations = {test["name"]: test for test in inventory["tests"]}
        if rust_selected:
            if result["schema"] in RUST_EXECUTION_SCHEMAS:
                request = configuration.get("rust_tests")
                validate_rust_test_request(request, result["preset"], result["build_kind"], preview)
                source, run_dir, tools = rust_test_invocation(result, candidate, registrations)
            else:
                validate_historical_rust_bindings(configuration, registrations, contracts["host"])
        ctest = ET.fromstring(payloads["ctest.xml"])
        cases = list(ctest.iter("testcase"))
        if sorted(c.get("name") for c in cases) != sorted(selected) or any(
            c.get("status") != "run" or any(c.find(t) is not None for t in ("failure", "error", "skipped"))
            for c in cases
        ):
            raise ValueError("missing, failed or skipped CTest outcomes")
        ctest_witnesses = {
            case.get("name"): (suite, case)
            for suite in ctest.iter("testsuite")
            for case in suite.findall("testcase")
            if case.get("name") in rust_selected
        }
        # Preview artifact v2 postdates the hard-coded execution label. Its
        # presence requires resolved Rust test evidence, which the archive cannot supply.
        require_resolved = preview and artifact_schema == "rust-artifacts-v2"
        rust_receipts = {
            name: rust_receipt_membership(
                payloads,
                name,
                preview=preview,
                require_resolved=require_resolved,
                request=request,
                preset=result["preset"],
                source=source,
                run_dir=run_dir,
                paths=rust_test_registration(registrations[name], result["preset"], request, source, tools)
                if request
                else None,
                contract=contracts["host"],
                ctest_witness=ctest_witnesses[name],
                receipt_schema=result["schema"],
            )
            for name in sorted(rust_selected)
        }
        gpu_selected = selected & set(native.get("gpu", {}))
        if gpu_selected and (
            not any(name.startswith("Resources/") for name in candidate["inputs"])
            or result.get("physical_gpu") != {"contender_excluded": True, "drained_while_locked": True}
            or json.loads(payloads["gpu-exclusion.json"]) != {"physical_gpu": "excluded"}
            or not any(command["name"] == "gpu-exclusion" for command in result["commands"])
        ):
            raise ValueError("missing shared physical GPU exclusion/drainage")
        for runtime, entries in native.items():
            chosen = selected & set(entries)
            if runtime == "host":
                chosen |= selected & {"Rust.Core", "Rust.Bridge"}
            found, filtered = _native_cases(result, payloads, entries, contracts[runtime], chosen, rust_receipts)
            for name, number in found.items():
                counts[name] += number
            excluded.update(".".join(name) for name in filtered)
        for name in selected & python.keys():
            entry = python[name]
            if result["case_files"][name] != {k: entry[k] for k in ("key", "protocol", "rows")}:
                raise ValueError("Python case contract differs from registration")
            key = entry["key"]
            execution = json.loads(payloads[key + ".execution.json"])
            if (
                execution["command"] != entry["command"]
                or execution["returncode"] != 0
                or execution.get("outcomes_valid") is not True
            ):
                raise ValueError("failed or mismatched Python execution")
            found = validate_python_result(
                json.loads(payloads[key + ".discovery.json"]),
                payloads[key + ".functions.jsonl"],
                payloads[key + ".rows.jsonl"],
            )
            counts["python_functions"] += found["python_functions"]
            counts["python_rows"] += found["python_rows"]
            counts["python_skips"].extend(name + ":" + function for function in found["python_skips"])
        if result.get("quality_files"):
            checks = json.loads(payloads["quality.checks.json"])
            if checks["selected"] != result["quality_files"] or checks["preset"] != result["preset"]:
                raise ValueError("selected-source quality scope differs from execution")
            if result["schema"] in {"native-tests-v5", "native-tests-v6"}:
                counts["replay_compiler_analysis"] = validate_replay_compiler_analysis(
                    checks,
                    payloads,
                    candidate,
                    result.get("run_id"),
                    prefix="quality-",
                    execution_evidence=result["schema"] == "native-tests-v6",
                )
            if checks["schema"] in {"native-quality-checks-v2", "native-quality-checks-v3", "native-quality-checks-v4"}:
                invocation = next(command["command"] for command in result["commands"] if command["name"] == "quality")
                packages = selected_source_quality_packages(invocation, result["preset"], result["quality_files"])
                if checks.get("mode") != "development":
                    raise ValueError("quality mode differs from the verifier invocation")
                _, rust_required = quality_scope(result["quality_files"], qualification=False)
                validate_clippy_forms(
                    checks,
                    rust_required=rust_required,
                    required_packages=packages,
                    preview=preview_enabled if current_product else None,
                )
                if rust_required and checks["rust_quality"]["JUICER_GRAIN_PREVIEW"] != candidate["configuration"].get(
                    "JUICER_GRAIN_PREVIEW"
                ):
                    raise ValueError("quality preview value differs from the native configuration")
            validate_quality_checks(checks, payloads)
    return {
        "status": result["status"],
        "preset": result["preset"],
        "scope": result["scope"],
        "groups": result["groups"],
        "runtime": result["runtime"],
        "build_kind": result["build_kind"],
        "ctest_cases": len(selected),
        **counts,
        "excluded_benchmarks": sorted(excluded),
        "unselected_registrations": sorted(available - selected),
        "shipping_isolation": False,
    }


def quality_result_members(result):
    required = {"candidate.json", "candidate.patch", "quality.checks.json"}
    if result["status"] == "passed":
        required.update(
            "quality-boundary" + suffix
            for suffix in (".execution.json", ".discovery.json", ".functions.jsonl", ".rows.jsonl")
        )
    return _flat_members(result, required)


def selected_source_quality_packages(invocation, preset, files):
    """Read the selected-source command emitted by verify.quality_command."""
    prefix = ["-B", "scripts/check-quality.py", "--preset", preset, "--development", "--files", *files]
    end = len(prefix) + 1
    if invocation[1:end] != prefix:
        raise ValueError("invalid selected-source quality invocation")
    options = invocation[end:]
    if options[:1] == ["--compiler-consumers"]:
        end = next((index for index, arg in enumerate(options[1:], 1) if arg.startswith("-")), len(options))
        if end == 1:
            raise ValueError("missing selected-source compiler consumers")
        options = options[end:]
    if not options:
        return RUST_QUALITY_PACKAGES
    selector, *packages = options
    if (
        selector != "--rust-packages"
        or not packages
        or any(package not in RUST_QUALITY_PACKAGES for package in packages)
    ):
        raise ValueError("invalid Rust package selector in the verifier invocation")
    return packages


def validate_clippy_forms(checks, *, rust_required, required_packages, preview=None):
    if ("rust-quality" in checks["expected_checks"]) != rust_required:
        raise ValueError("declared Rust quality differs from the execution scope")
    binding = checks.get("rust_quality")
    required = {}
    if rust_required:
        if not isinstance(binding, dict) or set(binding) != {"JUICER_GRAIN_PREVIEW", "packages"}:
            raise ValueError("missing Rust quality binding")
        raw_preview = binding["JUICER_GRAIN_PREVIEW"]
        packages = binding["packages"]
        if raw_preview is not None and not isinstance(raw_preview, str):
            raise ValueError("invalid raw CMake preview value")
        if (
            not isinstance(packages, list)
            or not packages
            or any(package not in RUST_QUALITY_PACKAGES for package in packages)
            or len(packages) != len(set(packages))
        ):
            raise ValueError("invalid Rust quality package selection")
        if required_packages is not None and set(packages) != set(required_packages):
            raise ValueError("Rust quality packages differ from the execution scope")
        required = dict(clippy_forms(checks["preset"], packages, raw_preview, preview=preview))
    elif binding is not None:
        raise ValueError("unexpected Rust quality binding")
    prefix = ["clippy", "--locked", "--target", PRESET_TARGETS[checks["preset"]], "--target-dir"]
    for operation in checks["commands"]:
        name = operation["name"].removeprefix("quality-")
        command = operation["command"]
        if not name.startswith("clippy-film-juicer-") and (
            command[1:2] != ["clippy"] or command[1:] == ["clippy", "--version"]
        ):
            continue
        args = required.pop(name, None)
        if args is None or command[1:6] != prefix or len(command) < 7 or not command[6] or command[7:] != args:
            raise ValueError("unexpected Clippy form: " + name)
    if required:
        raise ValueError("missing required Clippy forms: " + ", ".join(required))


def require_ordinary_compiler_analysis(formatted, translation_extensions, commands, root, build):
    """Check ordinary analysis remained in scope alongside selected replay headers."""
    path = type(root)
    sources = set()
    for command in commands:
        if len(command) == 5 and command[1] == "-p" and command[3] == "--warnings-as-errors=*":
            if path(command[2]) == build:
                sources.add(root / command[4])
        elif command[1:2] == ["--warnings-as-errors=*"] and command[4:5] == ["--"]:
            sources.add(root / command[3])
    required = {root / source for source in formatted if PurePosixPath(source).suffix.lower() in translation_extensions}
    if not sources or not required <= sources:
        raise ValueError("missing ordinary compiler analysis in mixed replay header selection")


def replay_database_sources(database, root, build):
    if not isinstance(database, list) or not database:
        raise ValueError("invalid motion replay compilation database")
    path = type(root)
    sources = set()
    for entry in database:
        file = path(entry["file"])
        if not file.is_relative_to(root) or ".." in file.parts:
            raise ValueError("incorrect motion replay compile variant")
        source = file.relative_to(root).as_posix()
        arguments = entry["arguments"]
        output = path(entry["output"])
        if (
            source in sources
            or path(entry["directory"]) != build
            or output != build / f"CMakeFiles/{MOTION_REPLAY_TARGET}.dir/{source}.o"
            or not isinstance(arguments, list)
            or not all(isinstance(arg, str) for arg in arguments)
            or not is_motion_replay_compile_entry(source, str(output), arguments)
            or not any(arg == "-o" and build / operand == output for arg, operand in zip(arguments, arguments[1:]))
            or not any(arg == "-c" and path(operand) == file for arg, operand in zip(arguments, arguments[1:]))
        ):
            raise ValueError("incorrect motion replay compile variant")
        sources.add(source)
    return sources


def validate_replay_compiler_analysis(checks, payloads, candidate, run_id, *, prefix="", execution_evidence=False):
    expected_schema = "native-quality-checks-v4" if execution_evidence else "native-quality-checks-v3"
    if checks.get("schema") != expected_schema:
        raise ValueError("current quality result requires compiler analysis evidence")
    analysis = checks.get("replay_compiler_analysis")
    headers = MOTION_REPLAY_HEADER_CONSUMERS.keys() & set(checks["selected"])
    if not isinstance(analysis, dict) or analysis.keys() != headers:
        raise ValueError("missing or invented replay header compiler analysis")
    commands = {item["name"]: item for item in checks["commands"]}
    recorded = {name for name in commands if name.startswith(prefix + "clang-tidy-motion-replay-")}
    executions = {
        name.removesuffix(REPLAY_EXECUTION_SUFFIX) for name in payloads if name.endswith(REPLAY_EXECUTION_SUFFIX)
    }
    database_name = "compile_commands.json"
    if not headers:
        if recorded or executions or database_name in payloads or SOURCE_FILE_POLICY in payloads:
            raise ValueError("unexpected replay compiler invocation")
        return analysis

    configuration = candidate["configuration"]
    _, replay_configured = read_preview_and_replay_configuration(configuration)
    formatter = commands.get(prefix + "clang-format", {}).get("command", [])
    if SOURCE_FILE_POLICY not in payloads:
        raise ValueError("missing source file policy for the replay compiler receipt")
    require_candidate_source(candidate, payloads, "scripts/" + SOURCE_FILE_POLICY)
    policy = json.loads(payloads[SOURCE_FILE_POLICY])
    extensions = policy.get("supportedExtensions")
    translation_units = policy.get("translationUnitExtensions")
    if (
        policy.get("schemaVersion") != 1
        or not isinstance(extensions, list)
        or not isinstance(translation_units, list)
        or not extensions
        or not translation_units
    ):
        raise ValueError("invalid source file policy for the replay compiler receipt")
    selected_native = {source for source in checks["selected"] if quality_native_source(source, extensions)}
    formatted = set(formatter[4:])
    if (
        "native-quality" not in checks["expected_checks"]
        or formatter[1:4] != ["--dry-run", "--Werror", "--style=file"]
        or not (headers | selected_native) <= formatted
    ):
        raise ValueError("missing formatting for a selected native file in the replay compiler receipt")
    path = rust_test_path(checks["preset"])
    root = path(configuration["CMAKE_HOME_DIRECTORY"])
    build = root / "out/build" / checks["preset"]
    if not root.is_absolute():
        raise ValueError("invalid configured source directory for the replay compiler receipt")
    if execution_evidence:
        if "rust-artifacts.json" not in payloads:
            raise ValueError("missing CMake artifact description for replay quality")
        product_artifact_consumers(json.loads(payloads["rust-artifacts.json"]), build, replay_configured)
    if ordinary := formatted - headers:
        ordinary_commands = [
            operation["command"]
            for name, operation in commands.items()
            if name.startswith(prefix + "clang-tidy-") and name not in recorded
        ]
        require_ordinary_compiler_analysis(ordinary, translation_units, ordinary_commands, root, build)
    if not replay_configured:
        if (
            recorded
            or executions
            or database_name in payloads
            or any(evidence != {"excluded": MOTION_REPLAY_EXCLUSION} for evidence in analysis.values())
        ):
            raise ValueError("incorrect compiler exclusion for unavailable motion replay")
        return analysis

    minimum = {header: MOTION_REPLAY_HEADER_CONSUMERS[header] for header in headers}
    required_sources = {source for sources in minimum.values() for source in sources}
    if database_name not in payloads:
        raise ValueError("missing replay compilation database")
    database = payloads[database_name]
    sources = replay_database_sources(json.loads(database), root, build)
    if not required_sources <= sources or (not execution_evidence and sources != required_sources):
        raise ValueError("invalid motion replay compilation database")
    labels = {source: prefix + motion_replay_tidy_label(source, legacy=not execution_evidence) for source in sources}
    if set(labels.values()) != recorded:
        raise ValueError("missing or unexpected replay compiler command")
    if execution_evidence and executions != recorded:
        raise ValueError("missing or unexpected replay compiler execution evidence")
    if not isinstance(run_id, str) or run_id in {".", ".."} or path(run_id).parts != (run_id,):
        raise ValueError("missing replay compiler result directory identity")
    database_directory = root / "out/validation" / checks["preset"] / "results" / run_id
    database_identity = {
        "path": str(database_directory / database_name),
        "sha256": hashlib.sha256(database).hexdigest(),
    }
    verifier = commands.get(prefix + "clang-tidy-config", {}).get("command", [])
    consumers = {header: [] for header in headers}
    for source, name in sorted(labels.items()):
        operation = commands[name]
        command = operation["command"]
        if (
            len(command) != 5
            or verifier != [command[0], "--verify-config"]
            or command[1] != "-p"
            or command[3:] != ["--warnings-as-errors=*", str(root / source)]
            or path(operation["directory"]) != root
        ):
            raise ValueError("incorrect motion replay compiler invocation: " + name)
        if path(command[2]) != database_directory:
            raise ValueError("motion replay compiler invocation did not use the retained database")
        if execution_evidence:
            execution = json.loads(payloads[name + REPLAY_EXECUTION_SUFFIX])
            if (
                execution.get("command") != command
                or execution.get("directory") != operation["directory"]
                or execution.get("database") != database_identity
            ):
                raise ValueError(
                    "motion replay compiler invocation differs from its recorded database execution: " + name
                )
            covered = execution.get("headers")
            if (
                not isinstance(covered, list)
                or not covered
                or not set(covered) <= headers
                or len(set(covered)) != len(covered)
            ):
                raise ValueError("invalid replay header compiler execution coverage: " + name)
            for header in covered:
                consumers[header].append(name)
    for header, sources in minimum.items():
        required = [labels[source] for source in sources]
        if not execution_evidence:
            consumers[header] = required
        elif not set(required) <= set(consumers[header]):
            raise ValueError("missing required replay header compiler execution coverage: " + header)
    if analysis != {header: {"commands": names} for header, names in consumers.items()}:
        raise ValueError("missing or unexpected replay compiler command")
    return analysis


def validate_quality_checks(checks, payloads=None):
    if (
        checks.get("schema")
        not in {
            "native-quality-checks-v1",
            "native-quality-checks-v2",
            "native-quality-checks-v3",
            "native-quality-checks-v4",
        }
        or checks["status"] != "passed"
        or not checks["expected_checks"]
        or checks.get("python_cases") != ["quality-boundary"]
    ):
        raise ValueError("failed or empty quality checks")
    expected = checks["expected_checks"]
    completed = checks["completed_checks"]
    commands = checks["commands"]
    planned = checks["expected_commands"]
    if (
        len(expected) != len(set(expected))
        or set(expected) != set(completed)
        or len(completed) != len(expected)
        or len(planned) != len(set(planned))
        or {command["name"] for command in commands} != set(planned)
        or len(commands) != len(planned)
    ):
        raise ValueError("missing or duplicate quality operation")
    if any(
        command.get("returncode") != 0
        or command.get("drained") is not True
        or command.get("process_status") != "completed"
        for command in commands
    ):
        raise ValueError("failed or undrained quality operation")
    counts = {"quality_checks": len(expected), "python_functions": 0, "python_rows": 0}
    if checks.get("python_cases"):
        if payloads is None:
            raise ValueError("missing quality Python outcomes")
        for key in checks["python_cases"]:
            execution = json.loads(payloads[key + ".execution.json"])
            if execution["returncode"] != 0 or execution.get("outcomes_valid") is not True:
                raise ValueError("failed quality Python execution")
            found = validate_python_result(
                json.loads(payloads[key + ".discovery.json"]),
                payloads[key + ".functions.jsonl"],
                payloads[key + ".rows.jsonl"],
            )
            counts["python_functions"] += found["python_functions"]
            counts["python_rows"] += found["python_rows"]
    return counts


def validate_quality_result(result, payloads):
    quality_result_members(result)
    candidate = _payloads(result, payloads)
    checks = json.loads(payloads["quality.checks.json"])
    if (
        checks["preset"] != result["preset"]
        or checks["selected"] != result["selected"]
        or checks["excluded"] != result["excluded"]
        or checks["commands"] != result["commands"]
    ):
        raise ValueError("quality result differs from its executed checks")
    counts = {"quality_checks": 0}
    configuration = candidate.get("configuration", {})
    if result["status"] == "passed" and result["schema"] in {"native-quality-v2", "native-quality-v3"}:
        counts["replay_compiler_analysis"] = validate_replay_compiler_analysis(
            checks,
            payloads,
            candidate,
            result.get("run_id"),
            execution_evidence=result["schema"] == "native-quality-v3",
        )
    if result["status"] == "passed" and checks["schema"] in {
        "native-quality-checks-v2",
        "native-quality-checks-v3",
        "native-quality-checks-v4",
    }:
        mode = result.get("mode")
        if mode not in {"qualification", "development"} or checks.get("mode") != mode:
            raise ValueError("quality mode differs from the standalone result")
        qualification_requested = result.get("qualification_requested")
        if "qualification_requested" in result and (
            type(qualification_requested) is not bool or (qualification_requested and mode != "qualification")
        ):
            raise ValueError("invalid standalone qualification request")
        binding = checks.get("rust_quality")
        # Older standalone results record the mode but omit the explicit flag.
        # They can still prove full qualification coverage with their Rust binding and commands.
        if qualification_requested is None:
            qualification_requested = mode == "qualification" and binding is not None
        _, rust_required = quality_scope(result["selected"], qualification=qualification_requested)
        packages = RUST_QUALITY_PACKAGES if mode == "qualification" else None
        preview_enabled = None
        if result["schema"] in {"native-quality-v2", "native-quality-v3"} and configuration:
            preview_enabled, _ = read_preview_and_replay_configuration(configuration)
            if (
                rust_required
                and binding
                and binding.get("JUICER_GRAIN_PREVIEW") != configuration["JUICER_GRAIN_PREVIEW"]
            ):
                raise ValueError("quality preview value differs from the native configuration")
        validate_clippy_forms(checks, rust_required=rust_required, required_packages=packages, preview=preview_enabled)
        counts["rust_quality"] = binding
    if result["status"] == "passed":
        counts.update(validate_quality_checks(checks, payloads))
    return {
        "status": result["status"],
        "preset": result["preset"],
        "scope": "quality",
        **counts,
        "shipping_isolation": False,
    }


def validate_production_contract(preset, payloads):
    """Read the completed OFF gate's assertions, never its original file paths."""
    report = json.loads(payloads["production-contract.json"])
    if report.get("schema") == "production-contract-v2":
        return _validate_production_contract_v2(preset, payloads, report)
    return _validate_production_contract_v1(preset, payloads, report)


def _validate_production_contract_v1(preset, payloads, report):
    commands_data = payloads["production-commands.json"]
    commands = json.loads(commands_data)
    if (
        preset not in {"linux-release", "windows-clang-release"}
        or report.get("schema") != "production-contract-v1"
        or report.get("preset") != preset
        or report.get("build_testing") != "OFF"
        or report.get("test_support") != "OFF"
        or report.get("exports") != ["OfxGetNumberOfPlugins", "OfxGetPlugin"]
        or report.get("hidden_device_load") is not True
    ):
        raise ValueError("incomplete production OFF assertions")
    expected = {
        "native-symbols",
        "rust-symbols",
        "module-exports",
        "film-compile",
        "film-values",
        "accepted-output-color-compile",
        "accepted-output-color-values",
        "prepared-render",
        "pdb-symbols" if preset.startswith("windows") else "module-symbols",
    }
    if (
        len(commands) != len(expected)
        or {row["name"] for row in commands} != expected
        or any(type(row["returncode"]) is not int or row["returncode"] != 0 for row in commands)
        or hashlib.sha256(commands_data).hexdigest() != report["commands"]["sha256"]
    ):
        raise ValueError("missing or failed production command")
    if len(report["scanned"]) != 4 or any(
        row["symbols_and_fault_strings_absent"] is not True for row in report["scanned"]
    ):
        raise ValueError("incomplete production isolation scan")
    probes = report["probes"]
    if (
        len(probes) != 2
        or {row["name"] for row in probes} != {"film", "accepted-output-color"}
        or any(
            row["linked_sha256"] != row["before_run_sha256"] or row["linked_sha256"] != row["after_run_sha256"]
            for row in probes
        )
    ):
        raise ValueError("changed or missing production C11 probe")
    prepared = report["prepared_render"]
    if (
        prepared["executable_sha256_before"] != prepared["executable_sha256_after"]
        or [row["case_id"] for row in prepared["rows"]]
        != ["negative-direct/cold", "negative-direct/warm", "negative-print/cold", "negative-print/warm"]
        or any(
            row["accepted_fixture_match"] is not True
            or row["diagnostic"] != ""
            or any(type(row[key]) is not int or row[key] != 0 for key in ("status", "api", "native_code"))
            for row in prepared["rows"]
        )
    ):
        raise ValueError("incomplete or failed production prepared rendering")
    return {"production_off": True, "prepared_render_cases": len(prepared["rows"])}


def _validate_production_contract_v2(preset, payloads, report):
    def digest(value):
        return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)

    def unchanged(row):
        return digest(row["sha256_before"]) and row["sha256_before"] == row["sha256_after"]

    if (
        preset not in {"linux-release", "windows-clang-release"}
        or report.get("preset") != preset
        or report.get("build_testing") != "OFF"
        or report.get("test_support") != "OFF"
        or report.get("exports") != ["OfxGetNumberOfPlugins", "OfxGetPlugin"]
        or report.get("hidden_device_load") is not True
    ):
        raise ValueError("incomplete production OFF assertions")
    commands_data = payloads["production-commands.json"]
    commands = json.loads(commands_data)
    expected = {
        "native-symbols",
        "rust-symbols",
        "module-exports",
        "module-symbols",
        "ofx-probe-build",
        "ofx-probe",
        "cuda-abi-probe-build",
        "cuda-abi-probe",
        "module-probe",
        "prepared-render",
    }
    if (
        len(commands) != len(expected)
        or {row["name"] for row in commands} != expected
        or any(
            type(row["returncode"]) is not int
            or row["returncode"] != 0
            or row["process_status"] != "completed"
            or row["drained"] is not True
            or not isinstance(row["command"], list)
            or not row["command"]
            or any(not isinstance(arg, str) or not arg for arg in row["command"])
            or not digest(row["stdout_sha256"])
            or not digest(row["stderr_sha256"])
            for row in commands
        )
        or hashlib.sha256(commands_data).hexdigest() != report["commands"]["sha256"]
    ):
        raise ValueError("missing, failed or undrained production command")
    scans = report["scanned"]
    scan_names = {"native-symbols", "rust-symbols", "module-exports", "module-symbols"}
    if (
        len(scans) != 4
        or {row["name"] for row in scans} != scan_names
        or any(
            not digest(row["sha256"])
            or row["symbols_and_fault_strings_absent"] is not True
            or row["legacy_symbols_absent"] is not True
            for row in scans
        )
    ):
        raise ValueError("incomplete production isolation scan")
    build_inputs = report["build_inputs"]
    consumed = report["consumed_inputs"]
    if (
        not isinstance(build_inputs, dict)
        or not build_inputs
        or not isinstance(consumed, dict)
        or not consumed
        or any(not isinstance(path, str) or not path or not unchanged(row) for path, row in build_inputs.items())
        or any(not isinstance(path, str) or not path or not unchanged(row) for path, row in consumed.items())
    ):
        raise ValueError("changed or missing production input")
    if any(row["sha256"] != build_inputs[row["file"]]["sha256_before"] for row in scans):
        raise ValueError("changed production scan input")
    loaded = report["loaded_module"]
    module_hash = build_inputs[loaded["file"]]["sha256_before"]
    if (
        loaded["sha256"] != module_hash
        or build_inputs[loaded["build_file"]]["sha256_before"] != module_hash
        or any(
            row["sha256"] != module_hash
            for row in scans
            if row["name"] == "module-exports" or (preset == "linux-release" and row["name"] == "module-symbols")
        )
    ):
        raise ValueError("changed production module")
    probes = report["probes"]
    if (
        len(probes) != 2
        or {row["name"] for row in probes} != {"ofx", "cuda-abi"}
        or any(
            not digest(row["linked_sha256"])
            or row["linked_sha256"] != row["before_run_sha256"]
            or row["linked_sha256"] != row["after_run_sha256"]
            or row["linked_sha256"] != build_inputs[row["executable"]]["sha256_before"]
            for row in probes
        )
    ):
        raise ValueError("changed or missing production C11 probe")
    commands_by_name = {row["name"]: row for row in commands}
    probes_by_name = {row["name"]: row for row in probes}
    variant = report.get("variant", "production")
    if variant not in {"production", "grain-preview"}:
        raise ValueError("unsupported production module variant")
    ofx_command = [probes_by_name["ofx"]["executable"], loaded["file"]]
    if variant == "grain-preview":
        ofx_command.append("preview")
    if (
        commands_by_name["ofx-probe"]["command"] != ofx_command
        or commands_by_name["cuda-abi-probe"]["command"] != [probes_by_name["cuda-abi"]["executable"]]
        or loaded["file"] not in commands_by_name["module-probe"]["command"]
        or any(commands_by_name[row["name"]]["command"][-1] != row["file"] for row in scans)
    ):
        raise ValueError("production command does not bind inspected inputs")
    prepared = report["prepared_render"]
    command = next(row for row in commands if row["name"] == "prepared-render")
    if (
        prepared["module"] != loaded["file"]
        or prepared["module_sha256"] != module_hash
        or not digest(prepared["executable_sha256_before"])
        or prepared["executable_sha256_before"] != prepared["executable_sha256_after"]
        or prepared["executable_sha256_before"] != build_inputs[prepared["executable"]]["sha256_before"]
        or prepared["fixture_sha256"] != consumed[prepared["fixture"]]["sha256_before"]
        or prepared["command"] != command["command"]
        or len(prepared["command"]) != 6
        or prepared["command"][0] != prepared["executable"]
        or prepared["command"][1] != loaded["file"]
        or prepared["command"][3] != prepared["fixture"]
        or prepared["command"][4] != preset
        or prepared["stdout_sha256"] != command["stdout_sha256"]
        or prepared["stderr_sha256"] != command["stderr_sha256"]
    ):
        raise ValueError("changed production render identity")
    rows = prepared["rows"]
    ids = ["negative-direct/cold", "negative-direct/warm", "negative-print/cold", "negative-print/warm"]
    forbidden = {
        "recipe_hash",
        "submitted",
        "state_released_after_submit",
        "views_released_after_submit",
        "status",
        "api",
        "native_code",
    }
    observations = (
        "accepted_fixture_match",
        "finite_output",
        "alpha_preserved",
        "padding_preserved",
        "stream_completed",
        "readback_complete",
        "context_preserved",
        "context_restored",
    )
    if not isinstance(rows, list) or [row["case_id"] for row in rows] != ids:
        raise ValueError("incomplete production render rows")
    for row, case_id in zip(rows, ids, strict=True):
        route, render_pass = case_id.split("/")
        channels = 3 if route == "negative-direct" else 4
        shape = [5, 7, channels]
        if (
            forbidden.intersection(row)
            or row["route"] != route
            or row["pass"] != render_pass
            or any(
                not isinstance(row[key], list)
                or len(row[key]) != 3
                or any(type(value) is not int for value in row[key])
                or row[key] != shape
                for key in ("input_shape", "output_shape")
            )
            or type(row["ofx_status"]) is not int
            or row["ofx_status"] != 0
            or row["messages"] != []
            or type(row["samples_compared"]) is not int
            or row["samples_compared"] != 5 * 7 * channels
            or not digest(row["input_sha256"])
            or not digest(row["output_f32_words_sha256"])
            or any(row[key] is not True for key in observations)
            or any(
                type(row[key]) is not int or row[key] != 1 for key in ("source_image_releases", "output_image_releases")
            )
        ):
            raise ValueError("incomplete or failed production rendering observation")
    if any(rows[index]["output_f32_words_sha256"] != rows[index + 1]["output_f32_words_sha256"] for index in (0, 2)):
        raise ValueError("production warm output differs")
    lifecycle = prepared["lifecycle"]
    if (
        any(
            type(lifecycle[key]) is not int or lifecycle[key] != value
            for key, value in (
                ("module_load_status", 0),
                ("effects_created", 2),
                ("effects_destroyed", 2),
                ("module_unload_status", 0),
            )
        )
        or not isinstance(lifecycle["destroy_statuses"], list)
        or len(lifecycle["destroy_statuses"]) != 2
        or any(type(value) is not int or value != 0 for value in lifecycle["destroy_statuses"])
        or lifecycle["context_restored"] is not True
    ):
        raise ValueError("incomplete production lifecycle")
    return {"production_off": True, "prepared_render_cases": len(rows)}


def build_result_members(result):
    required = {"candidate.json", "candidate.patch"}
    if result.get("production_contract") is True and result["status"] == "passed":
        required.update({"production-contract.json", "production-commands.json"})
    return _flat_members(result, required)


def validate_build_result(result, payloads):
    build_result_members(result)
    candidate = _payloads(result, payloads)
    assertions = {}
    if result.get("production_contract") is True and result["status"] == "passed":
        if candidate["configuration"]["BUILD_TESTING"] != "OFF":
            raise ValueError("production contract requires a testing-OFF candidate")
        assertions = validate_production_contract(result["preset"], payloads)
    return {
        "status": result["status"],
        "preset": result["preset"],
        "scope": "build",
        "shipping_isolation": False,
        **assertions,
    }


def collect_bundles(args):
    """Read named completed runs; never assemble a checkpoint from focused passes."""
    if args.output.exists():
        raise ValueError("retain the existing report and use a new output path")
    summaries = [read_native_result(bundle) for bundle in args.bundles]
    status = "passed" if summaries and all(summary["status"] == "passed" for summary in summaries) else "failed"
    requested = set(getattr(args, "selection", None) or ())
    if requested:
        expected_presets = set(getattr(args, "presets", None) or (summary["preset"] for summary in summaries))
        if {summary["preset"] for summary in summaries} != expected_presets:
            raise ValueError("missing requested native preset")
        for preset in expected_presets:
            complete = [
                summary
                for summary in summaries
                if summary["preset"] == preset
                and summary["scope"] == "checkpoint"
                and requested <= set(summary["groups"])
                and (summary["runtime"] == "all" or summary["runtime"] == args.runtime)
            ]
            if len(complete) != 1:
                raise ValueError("requested checkpoint requires one complete fresh execution per preset")
        if len({summary["candidate_source"] for summary in summaries}) != 1:
            raise ValueError("requested checkpoint contains different source candidates")
    result = {
        "status": status,
        "scope": "checkpoint" if requested else "reported-runs",
        "shipping_isolation": False,
        "results": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print("Offline result report:", args.output)
    return 0 if status == "passed" else 1


def collection_main(arguments):
    import argparse

    parser = argparse.ArgumentParser(description="Offline reading of named native bundles")
    parser.add_argument("--bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", nargs="+", choices=sorted(GROUPS))
    parser.add_argument("--runtime", choices=("host", "gpu", "all"), default="host")
    parser.add_argument("--presets", nargs="+", choices=PRESETS)
    try:
        return collect_bundles(parser.parse_args(arguments))
    except (ValueError, KeyError, TypeError, OSError, ET.ParseError) as error:
        parser.exit(1, f"Invalid/incomplete offline report: {error}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline native result reader (stdlib only).")
    parser.add_argument("--bundle", type=Path, required=True)
    options = parser.parse_args()
    try:
        summary = read_native_result(options.bundle)
        print(json.dumps(summary, indent=2))
        raise SystemExit(0 if summary["status"] == "passed" else 1)
    except (ValueError, KeyError, TypeError, OSError, ET.ParseError) as error:
        parser.exit(1, f"Invalid/incomplete native result: {error}\n")
