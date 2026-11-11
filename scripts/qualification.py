"""Pure native outcome and membership validation; standard library and bundles only."""

from __future__ import annotations

# Native result reading uses only the standard library and bundled bytes.
# The complete module can be copied and run offline.
import hashlib
import json
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
import time
import xml.etree.ElementTree as ET

OPTICS_FUNCTION = "optics::tests::matches_frozen_native_words_identities_and_failures"
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
PRESETS = ("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release")
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


def native_registration(contract, paths, registration, *, runtime="host", cuda_free=False):
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
    seen = set()
    for suite in ET.fromstring(payload).iter("testsuite"):
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


def rust_membership(document, package, cargo, binaries):
    package_name = {"core": "film-juicer-core", "bridge": "film-juicer-plugin"}[package]
    owners = [p for p in cargo["packages"] if p["name"] == package_name]
    if len(owners) != 1:
        raise ValueError("missing Rust package contract")
    owner = owners[0]
    expected = {
        (owner["id"], t["name"])
        for t in owner["targets"]
        if t.get("test") and set(t["kind"]) & {"test", "lib", "staticlib"}
    }
    actual = {
        (b["package-id"], b["binary-name"])
        for b in binaries["rust-binaries"].values()
        if b["package-id"] == owner["id"]
    }
    if not expected or expected != actual or set(document["rust-suites"]) != set(binaries["rust-binaries"]):
        raise ValueError("missing or unexpected Rust test binary")
    selected, filtered = set(), set()
    for binary, suite in document["rust-suites"].items():
        owned = binaries["rust-binaries"][binary]["package-id"] == owner["id"]
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
    native = native_registration(
        contract,
        json.loads(payloads[runtime + "-targets.json"]),
        json.loads(payloads["registration.json"]),
        runtime=runtime,
    )
    available = set(native) | ({"Rust.Core", "Rust.Bridge"} if runtime == "host" else set())
    selected = set(result["selection"])
    if not selected <= available or (result["scope"] == runtime + "-domains" and selected != available):
        raise ValueError("missing selected host domain")
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
        target_discovery = json.loads(payloads["target-discovery.json"])
        target_seen = {target: set() for target in target_discovery}
        for name in sorted(selected):
            if name in {"Rust.Core", "Rust.Bridge"}:
                package = "core" if name == "Rust.Core" else "bridge"
                expected, filtered = rust_membership(
                    json.loads(payloads[name + ".discovery.json"]),
                    package,
                    json.loads(payloads["rust-cargo.json"]),
                    json.loads(payloads["rust-binaries.json"]),
                )
                framework_terminals(payloads[name + ".xml"], expected, nextest=True, filtered=filtered)
                execution = json.loads(payloads[name + ".execution.json"])
                if (
                    len(execution["commands"]) != 2
                    or any(c["returncode"] != 0 for c in execution["commands"])
                    or execution["doctests"]["status"] != "none"
                    or execution["doctests"]["code_fence_sources"]
                    or execution["doctests"]["source_files_scanned"] < 1
                ):
                    raise ValueError("incomplete Rust discovery/execution")
                counts["rust_functions"] += len(expected)
                counts["rust_cases"] += rust_case_events(payloads[name + ".rows.jsonl"], contract, expected)
                if package == "core" and "optics_rows" in result:
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
        "native-quality-v1": (quality_result_members, validate_quality_result),
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


def _native_cases(result, payloads, entries, contract, selected):
    counts = {"rust_functions": 0, "native_functions": 0, "custom_cases": 0, "rust_cases": 0}
    full_discovery = json.loads(payloads["target-discovery.json"])
    excluded = {tuple(name.split(".", 1)) for name in contract.get("excluded_gtest", {})}
    seen, excluded_seen = {}, set()
    for name in sorted(selected):
        if name in {"Rust.Core", "Rust.Bridge"}:
            package = "core" if name == "Rust.Core" else "bridge"
            expected, filtered = rust_membership(
                json.loads(payloads[name + ".discovery.json"]),
                package,
                json.loads(payloads["rust-cargo.json"]),
                json.loads(payloads["rust-binaries.json"]),
            )
            framework_terminals(payloads[name + ".xml"], expected, nextest=True, filtered=filtered)
            execution = json.loads(payloads[name + ".execution.json"])
            if (
                len(execution["commands"]) != 2
                or any(c["returncode"] != 0 for c in execution["commands"])
                or execution["doctests"]["status"] != "none"
                or execution["doctests"]["code_fence_sources"]
                or execution["doctests"]["source_files_scanned"] < 1
            ):
                raise ValueError("incomplete Rust execution")
            counts["rust_functions"] += len(expected)
            counts["rust_cases"] += rust_case_events(payloads[name + ".rows.jsonl"], contract, expected)
            if package == "core" and "optics_rows" in result:
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
    if bool(candidate.get("features", {}).get("rust_artifacts")) != bool(result.get("rust_artifacts")):
        raise ValueError("missing separate Rust artifact description")
    if result.get("rust_artifacts"):
        artifacts = json.loads(payloads["rust-artifacts.json"])
        variants = artifacts["variants"]
        if (
            artifacts["schema"] != "rust-artifacts-v1"
            or set(variants) != {"production", "test-support"}
            or variants["production"]["archive"] == variants["test-support"]["archive"]
            or variants["production"]["target_dir"] == variants["test-support"]["target_dir"]
            or variants["production"]["features"] != []
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
    inventory = json.loads(payloads["registration.json"])
    contracts = {runtime: json.loads(payloads[runtime + "-contracts.json"]) for runtime in ("host", "gpu", "python")}
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
        if candidate["configuration"]["BUILD_TESTING"] != "ON" or candidate["configuration"]["CMAKE_BUILD_TYPE"] != (
            "Debug" if result["preset"].endswith("debug") else "Release"
        ):
            raise ValueError("incorrect testing configuration")
        cases = list(ET.fromstring(payloads["ctest.xml"]).iter("testcase"))
        if sorted(c.get("name") for c in cases) != sorted(selected) or any(
            c.get("status") != "run" or any(c.find(t) is not None for t in ("failure", "error", "skipped"))
            for c in cases
        ):
            raise ValueError("missing, failed or skipped CTest outcomes")
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
            found, filtered = _native_cases(result, payloads, entries, contracts[runtime], chosen)
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


def validate_quality_checks(checks, payloads=None):
    if (
        checks.get("schema") != "native-quality-checks-v1"
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
    _payloads(result, payloads)
    checks = json.loads(payloads["quality.checks.json"])
    if (
        checks["preset"] != result["preset"]
        or checks["selected"] != result["selected"]
        or checks["excluded"] != result["excluded"]
        or checks["commands"] != result["commands"]
    ):
        raise ValueError("quality result differs from its executed checks")
    counts = validate_quality_checks(checks, payloads) if result["status"] == "passed" else {"quality_checks": 0}
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
    if (
        commands_by_name["ofx-probe"]["command"] != [probes_by_name["ofx"]["executable"], loaded["file"]]
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
