"""Explicit native checkpoints; direct CTest groups and existing command records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib
import uuid

from build_evidence import check_preparation_inputs, check_preparation_owner, check_scoped_preparation, preparation_inventory, prepare_build, registration_files
from execution import Campaign, OutputLease, digest, snapshot, tool_snapshot, write_json

PRESETS = ("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release")
GROUPS = {"product", "boundary", "tooling", "corpus", "production"}

# CMake 4.3.1 LF/CRLF distributions: these bootstrap file() calls only convert
# environment paths or create CTest directories. The script template is data for
# configure_file, with an unsubstituted VERSION that prevents direct execution.
# Unknown bytes keep the ordinary conservative scan; receipts always hash raw bytes.
CMAKE_BOOTSTRAP_INPUTS = {
    "Modules/CMakeGenericSystem.cmake": {
        "51d5fba7b8e3492a6246ba29d192f8bb66888541ef987d1e41dff37213c2d2e9",
        "996f5d82cf4776db17d5399c1475fd4e09f8b76aa15975d96ca9115ca45bcb15"},
    "Modules/CTestTargets.cmake": {
        "2a870c04ff090e2cbd1bd181bbc8d8cd82d393115bb71f6798b5dc602b95a9dc",
        "b7d94b4ef08f50468cbafa0df58c5c0a0f9146c4c1afeca6c78c2df0f167f4d2"},
    "Modules/Platform/WindowsPaths.cmake": {
        "8228d73394f0945e64c966ebd8936a80d7ed76d066a6846a97b84c89c1981255",
        "2bd1111c39c43d858dd108c12bd1c735cc69066a90bf611336b852a6f6578f7d"},
    "Templates/CTestScript.cmake.in": {
        "3ccc0724081d7df661ad80f1d2ac6427d3e5aef7473687f926b7a08155b16cb7",
        "998a14031c9a14418d0a390d6328bdf5392ac3391a16132c30ffbe2da77146d6"},
}


def require_checkpoint(record):
    # A completed selection can coexist with unrelated pending work. The collector
    # checks current owners before admitting a complete aggregate.
    selected = set(record.get("selection", ()))
    expected = selected | ({"prepare", "quality"} if "product" in selected else set())
    if (record.get("mode") != "checkpoint" or record.get("status") not in {"passed", "incomplete"}
            or record.get("scope_status") != "passed" or not record.get("reason")
            or not selected or selected - GROUPS or set(record.get("required", ())) != expected
            or set(record.get("stages", {})) != expected
            or any(s.get("status") != "passed" for s in record.get("stages", {}).values())):
        raise ValueError("a completed explicit checkpoint is required; development/partial evidence is inadmissible")


def validate_ctest_report(report, names):
    import xml.etree.ElementTree as ET
    cases = list(ET.parse(report).getroot().iter("testcase"))
    if sorted(c.get("name") for c in cases) != sorted(names) or any(
            c.get("status") != "run" or any(c.find(n) is not None for n in ("failure", "error", "skipped"))
            for c in cases):
        raise ValueError("checkpoint CTest membership did not execute completely")


def membership(inventory, groups, runtime):
    """CTest registration is the only source of relevance and runtime membership."""
    result = {group: [] for group in groups if group != "production"}
    seen = set()
    scratch_owners = {}
    for test in inventory.get("tests", []):
        name = test["name"]
        if name in seen:
            raise ValueError("duplicate CTest registration: " + name)
        seen.add(name)
        properties = {p["name"]: p["value"] for p in test.get("properties", [])}
        labels = set(properties.get("LABELS", []))
        relevance = labels & (GROUPS - {"production"})
        if len(relevance) != 1 or len(labels & {"host", "gpu"}) != 1:
            raise ValueError("CTest group needs one relevance and a runtime prerequisite: " + name)
        group = next(iter(relevance))
        if group in result and (runtime == "all" or "host" in labels):
            if properties.get("DISABLED"):
                raise ValueError("required CTest group is disabled: " + name)
            if properties.get("RUN_SERIAL"):
                raise ValueError("checkpoint group has unexplained global serialization; use owned scratch or a named resource lock: " + name)
            command = test.get("command", [])
            if "--scratch-root" in command:
                index = command.index("--scratch-root")
                if command.count("--scratch-root") != 1 or index + 1 == len(command):
                    raise ValueError("checkpoint group needs one explicit scratch destination: " + name)
                destination = Path(command[index + 1])
                if not destination.is_absolute():
                    raise ValueError("checkpoint scratch destination must be absolute: " + name)
                scratch = os.path.normcase(str(destination.resolve()))
                if scratch in scratch_owners:
                    raise ValueError("checkpoint groups have colliding scratch roots: " + scratch_owners[scratch] + ", " + name)
                scratch_owners[scratch] = name
            result[group].append(name)
    if any(not names for names in result.values()):
        raise ValueError("required checkpoint selection is empty")
    return {group: sorted(names) for group, names in result.items()}


def ctest_identity(command):
    identity = list(command)
    identity[identity.index("--output-junit") + 1] = "<junit>"
    return identity


def stage_receipts(result, resolve=Path):
    return [resolve(result["receipt"]), *(resolve(p) for p in result.get("additional_receipts", {}))]


def validate_ctest_stage(paths, parts, names, commands, resolve=Path):
    """Both runtime partitions must prove the one selected relevance group."""
    paths = paths if isinstance(paths, list) else [paths]
    selected = [name for members in parts.values() for name in members]
    if (not parts or list(parts) != [role for role in ("host", "gpu") if role in parts]
            or any(not members for members in parts.values())
            or len(selected) != len(set(selected)) or sorted(selected) != sorted(names)
            or len(paths) != len(parts) or len(commands) != len(parts)):
        raise ValueError("checkpoint runtime partition membership is incomplete or duplicated")
    for path, (role, members), expected in zip(paths, parts.items(), commands, strict=True):
        command = json.loads(path.read_text())["identity"]["command"]
        pattern = "^(" + "|".join(re.escape(n) for n in members) + ")$"
        runtime_flag = "-LE" if role == "host" else "-L"
        if (ctest_identity(command) != expected or command[command.index("-R") + 1] != pattern
                or command[command.index(runtime_flag) + 1] != "gpu"):
            raise ValueError("checkpoint CTest command does not bind its runtime partition")
        validate_ctest_report(resolve(command[command.index("--output-junit") + 1]), members)


def run_ctest_parts(campaign, group, commands, roles, *, inputs, tools, environment,
                    timeout, repeat_reason):
    # A single pending stage owns both invocations and their common deadline.
    deadline = time.monotonic() + timeout
    expected_inputs = snapshot(campaign.root, inputs)
    present_tools = [p for p in tools if Path(p).is_file()]
    expected_tools = tool_snapshot(present_tools)
    paths = []
    for role, command in zip(roles, commands, strict=True):
        def run():
            if time.monotonic() >= deadline:
                raise TimeoutError("checkpoint CTest stage deadline expired")
            return campaign.run(group, command, inputs=inputs, tools=present_tools,
                expected_inputs=expected_inputs, expected_tools=expected_tools,
                env=environment, timeout=deadline - time.monotonic(),
                repeat_reason=repeat_reason)
        if role == "gpu":
            with OutputLease(campaign.root, timeout=max(0, deadline - time.monotonic())):
                paths.append(run())
        else:
            paths.append(run())
        # Exit zero alone cannot admit the next runtime partition.
        validate_ctest_report(Path(command[command.index("--output-junit") + 1]), roles[role])
        if time.monotonic() > deadline:
            raise TimeoutError("checkpoint CTest stage deadline expired")
    return paths


def production_manifest_identity(receipt, *, root, resolve=Path):
    """Read the identity emitted by the successful production process."""
    record = json.loads(resolve(str(receipt)).read_text())
    command = record["identity"]["command"]
    if command.count("--artifacts") != 1:
        raise ValueError("production command does not declare retained execution evidence")
    output = resolve(str(root / record["output"]))
    identities = re.findall(r"^PRODUCTION_MANIFEST_SHA256 ([0-9a-f]{64})$", output.read_text(), re.M)
    if len(identities) != 1:
        raise ValueError("production manifest lacks its execution-time identity; fresh execution required")
    return resolve(command[command.index("--artifacts") + 1]) / "production.json", identities[0]


def production_payloads(manifest, expected_manifest_sha256, *, root, inputs, resolve=Path, retained=None):
    """Recheck retained OFF execution files independently of the current ON tree."""
    manifest = resolve(str(manifest))
    if not manifest.is_file() or digest(manifest) != expected_manifest_sha256:
        raise ValueError("production manifest execution identity changed")
    document = json.loads(manifest.read_text())
    probes = document.get("probes", [])
    expected_names = {"film", "accepted-output-color"}
    if len(probes) != 2 or {row["name"] for row in probes} != expected_names:
        raise ValueError("production retained probe membership is incomplete or duplicated")
    suffix = ".exe" if resolve(probes[0]["executable"]).suffix == ".exe" else ""
    expected = {str(manifest): expected_manifest_sha256}

    def admit(name, checksum):
        path = resolve(name)
        key = str(path)
        if key in expected and expected[key] != checksum:
            raise ValueError("production execution identities disagree: " + key)
        if not path.is_file() or digest(path) != checksum:
            raise ValueError("production execution payload missing or changed: " + key)
        expected[key] = checksum

    build = root / "out/build" / document["preset"]
    declared = {str(resolve(str(root / name))): checksum for name, checksum in inputs.items()
                if name.startswith("out/build/")}
    copies = {str(resolve(name)): row for name, row in document.get("build_inputs", {}).items()}
    if (not declared or set(copies) != set(declared)
            or str(build / "CMakeCache.txt") not in declared
            or not any(Path(name).suffix.lower() in {".a", ".lib", ".ofx", ".pdb"} for name in declared)):
        raise ValueError("production retained build-input membership is incomplete; fresh OFF execution required")
    for name, checksum in declared.items():
        row = copies[name]
        destination = manifest.parent / "build" / Path(name).relative_to(build)
        if resolve(row["file"]) != destination or row["sha256"] != checksum:
            raise ValueError("production retained build-input execution identities disagree")
        admit(row["file"], checksum)
    cache = manifest.parent / "build/CMakeCache.txt"
    if document.get("build_testing") != "OFF" or not re.search(r"^BUILD_TESTING:BOOL=OFF$", cache.read_text(), re.M):
        raise ValueError("production retained configuration is not testing-OFF")

    for row in probes:
        name = row["name"]
        executable = resolve(row["executable"])
        source = resolve(row["source"])
        if (executable != manifest.parent / (name + suffix)
                or source != manifest.parent / (name + ".c")
                or len({row["linked_sha256"], row["before_run_sha256"], row["after_run_sha256"]}) != 1):
            raise ValueError("production probe execution identities disagree: " + name)
        admit(row["executable"], row["after_run_sha256"])
        admit(row["source"], row["source_sha256"])
    consumed = document.get("consumed_inputs", {})
    reference = document.get("reference_fixture", {})
    reference_manifest = root / "tests/film/reference-fixtures/manifest.json"
    spec = json.loads(reference_manifest.read_text())
    reference_payload = reference_manifest.parent / spec["presets"][document["preset"]]
    reference_consumed = {str(resolve(name)): checksum for name, checksum in consumed.items()}
    if (reference.get("logical_preset") != document["preset"]
            or resolve(reference.get("manifest", "")) != reference_manifest
            or resolve(reference.get("payload", "")) != reference_payload):
        raise ValueError("production film-reference fixture membership is incomplete")
    for path in (reference_manifest, reference_payload):
        checksum = digest(path)
        if inputs.get(path.relative_to(root).as_posix()) != checksum or reference_consumed.get(str(path)) != checksum:
            raise ValueError("production film-reference consumed fixture identity changed")
    if digest(reference_payload) != spec["payload_sha256"][reference_payload.name]:
        raise ValueError("production film-reference frozen payload changed")
    expected_reference = []
    for row in json.loads(reference_payload.read_text())["products"]:
        if "input" not in row:
            continue
        for case in range(8 if row["method"] == 1 else 7):
            envelope = ([0, 0, 0, 0, 0, 0, 0] if case in (0, 5) else
                        [2, 0, 0, 1, 0, 0, 0] if case in (1, 4) else
                        [3, 0, 0, 2, 1, 0, 0] if case == 2 else
                        [3, 0, 0, 2, 2, 1, 0] if case == 3 else
                        [3, 0, 0, 3, 0, 7, 0] if case == 6 else
                        [3, 0, 0, 4, 1, 8, 1])
            expected_reference.append({"case": f"reference-{row['method']}-{case}",
                                       "envelope": envelope, "hash": row["reference"]["hash"] if case in (0, 5) else 0,
                                       "fields": "13-pass"})
    if len(expected_reference) != 15 or document.get("reference_results") != expected_reference:
        raise ValueError("production film-reference value/failure results are incomplete")
    descriptor_manifest = root / "tests/scanner/descriptor-fixtures/manifest.json"
    descriptor_spec = json.loads(descriptor_manifest.read_bytes())
    logical = descriptor_spec["presets"][document["preset"]]["native"]
    descriptor_payload = descriptor_manifest.parent / descriptor_spec["payloads"][logical]
    descriptor_context = descriptor_manifest.parent / descriptor_spec["contexts"][logical]
    descriptor = document.get("scanner_descriptor_fixture", {})
    if (descriptor.get("logical_preset") != document["preset"] or
            any(resolve(descriptor.get(key, "")) != path for key, path in
                (("manifest", descriptor_manifest), ("payload", descriptor_payload), ("context", descriptor_context)))):
        raise ValueError("production scanner descriptor fixture membership is incomplete")
    for path in (descriptor_manifest, descriptor_payload, descriptor_context):
        checksum = digest(path)
        if inputs.get(path.relative_to(root).as_posix()) != checksum or reference_consumed.get(str(path)) != checksum:
            raise ValueError("production scanner descriptor consumed fixture identity changed")
        if path != descriptor_manifest and checksum != descriptor_spec["sha256"][path.name]:
            raise ValueError("production scanner descriptor frozen payload/context changed")
    expected_descriptors = descriptor_spec["production_cases"] + descriptor_spec["production_malformed"]
    if len(expected_descriptors) != 41 or document.get("scanner_descriptor_results") != expected_descriptors:
        raise ValueError("production scanner descriptor value/failure results are incomplete")
    post_manifest = root / "tests/scanner/post-effects-fixtures/manifest.json"
    post_spec = json.loads(post_manifest.read_bytes())
    logical = post_spec["presets"][document["preset"]]["native"]
    post_payload = post_manifest.parent / post_spec["payloads"][logical]
    post_context = post_manifest.parent / post_spec["contexts"][logical]
    post = document.get("scanner_post_fixture", {})
    if (post.get("logical_preset") != document["preset"] or
            any(resolve(post.get(key, "")) != path for key, path in
                (("manifest", post_manifest), ("payload", post_payload), ("context", post_context)))):
        raise ValueError("production scanner post-effects fixture membership is incomplete")
    for path in (post_manifest, post_payload, post_context):
        checksum = digest(path)
        if inputs.get(path.relative_to(root).as_posix()) != checksum or reference_consumed.get(str(path)) != checksum:
            raise ValueError("production scanner post-effects consumed fixture identity changed")
        if path != post_manifest and checksum != post_spec["sha256"][path.name]:
            raise ValueError("production scanner post-effects frozen payload/context changed")
    expected_post = post_spec["production_cases"]
    if len(expected_post) != 410 or document.get("scanner_post_results") != expected_post:
        raise ValueError("production scanner post-effects value/failure results are incomplete")
    gaussian_manifest = root / "tests/optics/gaussian-fixtures/manifest.json"
    gaussian_spec = json.loads(gaussian_manifest.read_bytes())
    logical = gaussian_spec["presets"][document["preset"]]["native"]
    gaussian_payload = gaussian_manifest.parent / gaussian_spec["payloads"][logical]
    gaussian_context = gaussian_manifest.parent / gaussian_spec["contexts"][logical]
    gaussian = document.get("gaussian_fixture", {})
    if gaussian.get("logical_preset") != document["preset"] or any(
            resolve(gaussian.get(key, "")) != path for key, path in
            (("manifest", gaussian_manifest), ("payload", gaussian_payload), ("context", gaussian_context))):
        raise ValueError("production Gaussian fixture membership is incomplete")
    for path in (gaussian_manifest, gaussian_payload, gaussian_context):
        checksum = digest(path)
        if inputs.get(path.relative_to(root).as_posix()) != checksum or reference_consumed.get(str(path)) != checksum or (path != gaussian_manifest and checksum != gaussian_spec["sha256"][path.name]):
            raise ValueError("production Gaussian consumed fixture changed")
    expected_gaussian = []
    for row in json.loads(gaussian_payload.read_bytes())["cases"]:
        radius, sigma = row["radius"], row["sigma_bits"]
        reason = 2 if radius < 1 or radius > 2048 else (3 if sigma & 0x80000000 or sigma == 0 or sigma >= 0x7f800000 else 0)
        expected_gaussian.append({"case": row["name"], "envelope": [3 if reason else 0, 0, 0, reason],
                                  "radius": 0 if reason else radius, "sigma_bits": 0 if reason else sigma,
                                  "weights": [] if reason else row["weights"]})
    for name in gaussian_spec["production_cases"][188:]:
        reason = 2 if name == "radius-upper-reject" else (0 if name in {"zero-capacity", "null-release"} else 1)
        expected_gaussian.append({"case": name, "envelope": [3 if reason == 2 else 2 if reason else 0, 0, 0, reason],
                                  "radius": 0, "sigma_bits": 0, "weights": []})
    if len(expected_gaussian) != 201 or document.get("gaussian_results") != expected_gaussian:
        raise ValueError("production Gaussian full arrays/metadata/failure results are incomplete")

    frame_manifest = root / "tests/frame/geometry-fixtures/manifest.json"
    frame_spec = json.loads(frame_manifest.read_bytes())
    frame_logical = frame_spec["presets"][document["preset"]]["native"]
    frame_payload = frame_manifest.parent / frame_spec["payloads"][frame_logical]
    frame_context = frame_manifest.parent / frame_spec["contexts"][frame_logical]
    frame = document.get("frame_fixture", {})
    if frame.get("logical_preset") != document["preset"] or any(
            resolve(frame.get(key, "")) != path for key, path in
            (("manifest", frame_manifest), ("payload", frame_payload), ("context", frame_context))):
        raise ValueError("production frame geometry fixture membership is incomplete")
    for path in (frame_manifest, frame_payload, frame_context, frame_manifest.parent / "inputs.json"):
        checksum = digest(path)
        if (inputs.get(path.relative_to(root).as_posix()) != checksum
                or reference_consumed.get(str(path)) != checksum
                or (path != frame_manifest and checksum != frame_spec["sha256"][path.name])):
            raise ValueError("production frame geometry consumed fixture identity changed")
    frame_expected = frame_spec["production_cases"]
    if len(frame_expected) != 69 or len({row["case"] for row in frame_expected}) != 69 or document.get("frame_results") != frame_expected:
        raise ValueError("production frame geometry full fields/words/failure results are incomplete")

    halation_manifest = root / "tests/scatter_halation/frame-fixtures/manifest.json"
    halation_spec = json.loads(halation_manifest.read_bytes())
    halation_logical = halation_spec["presets"][document["preset"]]["native"]
    halation_paths = {
        "manifest": halation_manifest,
        "payload": halation_manifest.parent / halation_spec["payloads"][halation_logical],
        "context": halation_manifest.parent / halation_spec["contexts"][halation_logical],
        "normative": halation_manifest.parent / halation_spec["checked"],
        "input_spec": halation_manifest.parent / "inputs.json",
    }
    halation = document.get("halation_frame_fixture", {})
    if halation.get("logical_preset") != document["preset"] or any(
            resolve(halation.get(key, "")) != path for key, path in halation_paths.items()):
        raise ValueError("production halation frame fixture membership is incomplete")
    for path in halation_paths.values():
        checksum = digest(path)
        if (inputs.get(path.relative_to(root).as_posix()) != checksum
                or reference_consumed.get(str(path)) != checksum
                or (path != halation_manifest and checksum != halation_spec["sha256"][path.name])):
            raise ValueError("production halation frame consumed fixture identity changed")
    halation_expected = halation_spec["production_cases"]
    if (len(halation_expected) != 86 or len({row["case"] for row in halation_expected}) != 86
            or document.get("halation_frame_results") != halation_expected):
        raise ValueError("production halation frame full fields/words/failure results are incomplete")

    diffusion_manifest = root / "tests/diffusion/frame-fixtures/manifest.json"
    diffusion_spec = json.loads(diffusion_manifest.read_bytes())
    diffusion_logical = diffusion_spec["presets"][document["preset"]]["native"]
    diffusion_paths = {
        "manifest": diffusion_manifest,
        "payload": diffusion_manifest.parent / diffusion_spec["payloads"][diffusion_logical],
        "context": diffusion_manifest.parent / diffusion_spec["contexts"][diffusion_logical],
        "normative": diffusion_manifest.parent / diffusion_spec["checked"],
        "input_spec": diffusion_manifest.parent / "inputs.json",
        "edges_input_spec": diffusion_manifest.parent / "edge-inputs.json",
        "edges_payload": diffusion_manifest.parent / diffusion_spec["payloads"][diffusion_spec["presets"][document["preset"]]["edges"]],
        "edges_context": diffusion_manifest.parent / diffusion_spec["contexts"][diffusion_spec["presets"][document["preset"]]["edges"]],
    }
    diffusion = document.get("diffusion_frame_fixture", {})
    if diffusion.get("logical_preset") != document["preset"] or any(
            resolve(diffusion.get(key, "")) != path for key, path in diffusion_paths.items()):
        raise ValueError("production diffusion frame fixture membership is incomplete")
    for path in diffusion_paths.values():
        checksum = digest(path)
        if (inputs.get(path.relative_to(root).as_posix()) != checksum
                or reference_consumed.get(str(path)) != checksum
                or (path != diffusion_manifest and checksum != diffusion_spec["sha256"][path.name])):
            raise ValueError("production diffusion frame consumed fixture identity changed")
    diffusion_expected = diffusion_spec["production_cases"]
    if (len(diffusion_expected) != 507 or len({row["case"] for row in diffusion_expected}) != 507
            or document.get("diffusion_frame_results") != diffusion_expected):
        raise ValueError("production diffusion frame full fields/words/failure results are incomplete")

    dir_manifest = root / "tests/dir/frame-fixtures/manifest.json"
    dir_spec = json.loads(dir_manifest.read_bytes())
    dir_logical = dir_spec["presets"][document["preset"]]["native"]
    dir_paths = {
        "manifest": dir_manifest,
        "payload": dir_manifest.parent / dir_spec["payloads"][dir_logical],
        "normative": dir_manifest.parent / dir_spec["checked"],
        "input_spec": dir_manifest.parent / "inputs.json",
    }
    dir_frame = document.get("dir_frame_fixture", {})
    if dir_frame.get("logical_preset") != document["preset"] or any(
            resolve(dir_frame.get(key, "")) != path for key, path in dir_paths.items()):
        raise ValueError("production DIR frame fixture membership is incomplete")
    for path in dir_paths.values():
        checksum = digest(path)
        if (inputs.get(path.relative_to(root).as_posix()) != checksum
                or reference_consumed.get(str(path)) != checksum
                or (path != dir_manifest and checksum != dir_spec["sha256"][path.name])):
            raise ValueError("production DIR frame consumed fixture identity changed")
    dir_expected = dir_spec["production_cases"]
    if (len(dir_expected) != 121 or len({row["case"] for row in dir_expected}) != 121
            or document.get("dir_frame_results") != dir_expected):
        raise ValueError("production DIR frame full fields/words/failure trace results are incomplete")

    headers = {name: checksum for name, checksum in consumed.items()
               if resolve(name).parent == manifest.parent and resolve(name).suffix == ".h"}
    if str(manifest.parent / "production-film-expected.h") not in {str(resolve(name)) for name in headers}:
        raise ValueError("production retained expectation header membership is incomplete")
    for name, checksum in headers.items():
        admit(name, checksum)
    command_record = document["commands"]
    if resolve(command_record["file"]) != manifest.parent / "commands.json":
        raise ValueError("production retained command record is not the declared run")
    admit(command_record["file"], command_record["sha256"])
    commands = json.loads(resolve(command_record["file"]).read_text())
    names = [row["name"] for row in commands]
    if len(names) != len(set(names)) or not {name + role for name in expected_names for role in ("-compile", "-values")} <= set(names):
        raise ValueError("production retained command membership is incomplete or duplicated")
    outputs = {}
    for row in commands:
        if row["returncode"] != 0:
            raise ValueError("production retained command did not succeed")
        for extension in ("stdout", "stderr"):
            name = str(manifest.parent / (row["name"] + "." + extension))
            outputs[name] = row[extension + "_sha256"]
        if row["name"] in {name + "-values" for name in expected_names}:
            name = row["name"].removesuffix("-values")
            if resolve(row["command"][0]) != manifest.parent / (name + suffix):
                raise ValueError("production command did not execute the linked probe")
    recorded_outputs = {str(resolve(name)): checksum for name, checksum in document.get("command_outputs", {}).items()}
    if recorded_outputs != outputs:
        raise ValueError("production retained detailed output membership is incomplete")
    for name, checksum in outputs.items():
        admit(name, checksum)
    film_stdout = manifest.parent / "film-values.stdout"
    dir_rows = [json.loads(line.removeprefix("DIR_FRAME_RESULT "))
                for line in film_stdout.read_text().splitlines() if line.startswith("DIR_FRAME_RESULT ")]
    if dir_rows != dir_expected:
        raise ValueError("production DIR frame executed stdout results disagree")
    diffusion_rows = [json.loads(line.removeprefix("DIFFUSION_FRAME_RESULT "))
                      for line in film_stdout.read_text().splitlines() if line.startswith("DIFFUSION_FRAME_RESULT ")]
    if diffusion_rows != diffusion_expected:
        raise ValueError("production diffusion frame executed stdout results disagree")
    if retained is not None and {str(resolve(name)): checksum for name, checksum in retained.items()} != expected:
        raise ValueError("production retained payload membership or execution identities disagree; fresh execution required")
    return expected


def run_stage(campaign, state_path, state, name, command, *, inputs, tools, environment,
          timeout, semantic_command=None, action=None, validate=None, payloads=(),
          dependencies=(), repeat_reason="", refresh_inputs=None, admission_observations=None):
    """Reuse the latest gate across attempt names; pending work is never a pass."""
    campaign.check_run_request(name, command, inputs=inputs, timeout=timeout)
    identity = {"command": list(semantic_command or command),
                "inputs": snapshot(campaign.root, inputs), "tools": tool_snapshot(tools, allow_missing=True), "tool_paths": list(tools),
                "environment": hashlib.sha256(json.dumps({k: v for k, v in environment.items()
                    if k not in {"JUICER_OUTPUT_LEASES", "JUICER_TEST_ARTIFACT_DIR"}}, sort_keys=True).encode()).hexdigest(),
                "dependencies": list(dependencies)}
    previous = state.get(name)
    if previous and previous["identity"] == identity and not repeat_reason:
        if previous["status"] != "passed":
            raise ValueError("pending checkpoint gate requires a relevant repair or a diagnosed repeat reason: " + name)
        try:
            from verify import check_receipt_payloads
            paths = stage_receipts(previous)
            for path, expected in zip(paths, [previous["sha256"], *previous.get("additional_receipts", {}).values()], strict=True):
                if digest(path) != expected:
                    raise ValueError("checkpoint command receipt changed")
                check_receipt_payloads(campaign.root, path)
            path = paths[0]
            if name == "production" and payloads:
                production_payloads(*production_manifest_identity(path, root=campaign.root),
                                    root=campaign.root, inputs=previous["identity"]["inputs"],
                                    retained=previous.get("payloads", {}))
            if any(digest(Path(p)) != value for p, value in previous.get("payloads", {}).items()):
                raise ValueError("checkpoint result payload changed")
            if validate:
                validate(paths if len(paths) > 1 else path)
            if snapshot(campaign.root, inputs) != identity["inputs"] or tool_snapshot(tools, allow_missing=True) != identity["tools"]:
                raise ValueError("checkpoint inputs changed during reuse")
            print("Reused checkpoint gate:", name, path, flush=True)
            return {**previous, "reused_from": str(path)}
        except BaseException as error:
            state[name] = {**previous, "status": "failed", "error": str(error)}
            write_json(state_path, state)
            raise
    state[name] = {"status": "pending", "identity": identity}
    write_json(state_path, state)
    try:
        executed = action() if action else campaign.run(name, command, inputs=inputs, tools=[p for p in tools if Path(p).is_file()],
            env=environment, timeout=timeout, repeat_reason=repeat_reason)
        paths = executed if isinstance(executed, list) else [executed]
        if not paths:
            raise ValueError("checkpoint stage produced no command evidence")
        if validate:
            validate(paths if len(paths) > 1 else paths[0])
        actual = snapshot(campaign.root, inputs)
        # Preparation owns generated configuration; source/tool mutations still fail.
        generated = set()
        if name == "prepare" and action:
            build = campaign.root / "out/build" / campaign.directory.parent.name
            generated = {p for p in identity["inputs"] if (campaign.root / p).is_relative_to(build)}
        if (any(actual.get(p) != value for p, value in identity["inputs"].items() if p not in generated)
                or tool_snapshot(tools, allow_missing=True) != identity["tools"]):
            raise ValueError("checkpoint relevant inputs changed")
        refreshed = snapshot(campaign.root, refresh_inputs()) if refresh_inputs else actual
        if name == "prepare" and action:
            generated.update(p for p in refreshed if (campaign.root / p).is_relative_to(build))
        if admission_observations is not None and any(
                value != admission_observations.get(p) for p, value in refreshed.items()
                if p not in identity["inputs"] and p not in generated):
            raise ValueError("checkpoint newly discovered preparation inputs changed")
        identity["inputs"] = refreshed
        retained = (production_payloads(*production_manifest_identity(paths[0], root=campaign.root),
                                       root=campaign.root, inputs=identity["inputs"])
                    if name == "production" and payloads else {str(p): digest(Path(p)) for p in payloads})
        result = {"status": "passed", "identity": identity, "receipt": str(paths[0]),
                  "sha256": digest(paths[0]),
                  "additional_receipts": {str(p): digest(p) for p in paths[1:]}, "payloads": retained}
        state[name] = result
        write_json(state_path, state)
        return result
    except BaseException as error:
        state[name].update(status="failed", error=str(error))
        write_json(state_path, state)
        raise


def checkpoint_inputs(root, build, group, *, source_files=None, preparation=None, resolve=Path):
    from verify import is_documentation, source_snapshot

    paths = {root / name for name in ("CMakeLists.txt", "CMakePresets.json", "tests/CMakeLists.txt",
             "scripts/qualification.py", "scripts/execution.py", "scripts/process_tree.py")}
    paths.add(build / "CMakeCache.txt")
    if group == "production":
        paths.update(build / name for name in ("build.ninja", "compile_commands.json", "cargo-features.txt"))
    else:
        paths.update(registration_files(build, allow_missing=True))
    if group in {"boundary", "tooling"}:
        paths.update(root / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "clippy.toml", "rustfmt.toml", "cmake/JuicerRust.cmake"))
        paths.update(p for p in (root / "rust").rglob("*") if p.is_file())
        paths.update(p for p in (root / ".cargo").glob("*") if p.is_file())
        if group == "boundary":
            paths.add(root / "scripts/check-quality.py")
            for directory in ("src", "native"):
                paths.update(p for p in (root / directory).rglob("*") if p.is_file() and p.suffix in {".h", ".cpp", ".cu", ".cuh"})
            paths.add(root / "tests/ffi/color_preparation_test.cpp")
            paths.update(root / "tests/ffi" / name for name in (
                "film_recipe_test.cpp", "exposure_fixture.h", "film_recipe_abi_c.c", "film_recipe_abi_cpp.inc",
                "film_recipe_abi_facts.inc", "dir_frame_test.inc", "diffusion_frame_test.inc", "halation_frame_test.inc", "frame_geometry_test.inc", "gaussian_kernel_test.inc", "film_reference_test.inc", "print_illuminant_test.inc", "print_illuminant_fixture.h", "print_preflash_test.inc", "print_balance_test.inc", "scanner_correction_test.inc", "scanner_lut_test.inc", "scanner_lut_descriptor_test.inc", "output_color_test.cpp", "output_color_abi_c.c", "scanner_color_test.inc"))
            paths.update((root / "tests/film/reference-fixtures").glob("*.json"))
            paths.update((root / "tests/print/illuminant-fixtures").glob("*.json"))
            paths.update((root / "tests/print/preflash-fixtures").glob("*.json"))
            paths.update((root / "tests/print/balance-fixtures").glob("*.json"))
            paths.update((root / "tests/scanner/correction-fixtures").glob("*.json"))
            paths.update((root / "tests/scanner/lut-fixtures").glob("*.json"))
            paths.update((root / "tests/scanner/descriptor-fixtures").glob("*.json"))
            paths.update((root / "tests/scanner/post-effects-fixtures").glob("*.json"))
            paths.update((root / "tests/optics/gaussian-fixtures").glob("*.json"))
            paths.update((root / "tests/frame/geometry-fixtures").glob("*.json"))
            paths.update((root / "tests/scatter_halation/frame-fixtures").glob("*.json"))
            paths.update((root / "tests/diffusion/frame-fixtures").glob("*.json"))
            paths.update((root / "tests/dir/frame-fixtures").glob("*.json"))
            paths.update((root / "tests/scanner/color-fixtures").glob("*.json"))
            paths.update((root / "tests/quality").glob("test_rust_*.py"))
            paths.update(p for p in (root / "tests/quality/fixtures").rglob("*.rs"))
        else:
            for directory in ("scripts", "tests/quality", "tests/ctest"):
                paths.update(p for p in (root / directory).rglob("*.py") if "__pycache__" not in p.parts)
            paths.add(root / "tests/ffi/run_rust_build_contract.py")
    else:
        paths.update(root / name for name in (source_files if source_files is not None else source_snapshot(root)))
        for directory in ("Resources", "tests"):
            paths.update(p for p in (root / directory).rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts
                         and not is_documentation(p.relative_to(root).as_posix()))
    if group == "prepare":
        paths = preparation_inputs(root, build, {p for p in paths if p.is_file()}, context=preparation, resolve=resolve)
    return sorted(p.relative_to(root).as_posix() for p in paths if p.is_file())


def has_opaque_configure_commands(paths, *, build=None, configuration=None, resolve=Path):
    """File operations, processes and dynamic evaluation have no declared closure."""
    sources = {path for path in paths if path.name == "CMakeLists.txt" or path.suffix == ".cmake"}
    paths, bootstrap = set(sources), {}
    if configuration is not None:
        cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
        match = re.search(r"^CMAKE_ROOT:INTERNAL=(.+)$", cache, re.M)
        if match:
            cmake_root = resolve(match[1].strip()).resolve()
            bootstrap = {cmake_root / name: checksums for name, checksums in CMAKE_BOOTSTRAP_INPUTS.items()}
        # include() accepts arbitrary filenames. Regeneration data that cannot
        # be read safely as CMake text also keeps the preparation scope broad.
        paths.update(resolve(name) for name in configuration["regeneration_mtimes"])
    for path in paths:
        content = path.read_bytes()
        if path not in sources and hashlib.sha256(content).hexdigest() in bootstrap.get(path, ()):
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return True
        if re.search(r"\b(?:file|execute_process|cmake_language)\s*\(", text, re.I):
            return True
    return False


def preparation_inputs(root, build, paths, *, context=None, resolve=Path):
    """Exclude only registered runtime files outside the owning build closure.

    Collection uses the producer's sealed closure after validating its original
    requirements. It never runs a foreign native tool or projects an old receipt.
    """
    from build_dependencies import NinjaGraph
    from build_evidence import scoped_context

    if not (build / "CTestTestfile.cmake").is_file():
        return paths
    runtime = set()
    for registration in registration_files(build):
        for line in registration.read_text(encoding="utf-8").splitlines():
            # Generated CTest command operands are explicit runtime owners. Do
            # not infer directory contents, manifest payloads or Python imports.
            if not line.startswith("add_test("):
                continue
            for operand in re.findall(r'"([^"\\$;]+)"', line):
                path = resolve(operand)
                if path.is_absolute() and path.is_file() and path.is_relative_to(root / "tests"):
                    runtime.add(path)
    if not runtime or has_opaque_configure_commands(paths):
        return paths
    if context is None:
        graph = NinjaGraph(build, ["all"])
        context = scoped_context(root, root, build, graph, [], dict(os.environ))
    if has_opaque_configure_commands(paths, build=build, configuration=context["configuration"], resolve=resolve):
        return paths
    # A custom build command may consume a script/file without a DEPENDS
    # edge. Such operands are not proven runtime-only.
    commands = context["commands"].replace("\\", "/").casefold()
    runtime = {p for p in runtime if p.name.casefold() not in commands
               and not any(re.search(re.escape(parent.as_posix().casefold()) + r'(?=["\s;]|$)', commands)
                           for parent in p.parents if parent.is_relative_to(root / "tests"))}
    protected = {resolve(name).resolve() for name in context["files"]}
    protected.update((root / name).resolve() for name in context["configuration"]["inputs"])
    # Cargo test embeddings/build scripts and configure-time file operations
    # can escape Ninja's dependency graph. Unresolved consumers keep broad scope.
    for path in paths:
        if path.suffix == ".rs":
            if path.name == "build.rs" or re.search(r"\binclude(?:_str|_bytes)?\s*!", path.read_text(encoding="utf-8")):
                return paths
            runtime.discard(path)
        if path.name == "Cargo.toml":
            if tomllib.loads(path.read_text(encoding="utf-8")).get("package", {}).get("build"):
                return paths
        if path.name != "CMakeLists.txt" and path.suffix != ".cmake":
            continue
        content = path.read_text(encoding="utf-8")
        # A source-side configure/tool operand may be absent from Ninja. Only
        # recognize ordinary add_test syntax; anything else stays conservative.
        consumers = re.sub(r'\badd_test\s*\((?:"(?:[^"\\]|\\.)*"|[^()"])*\)', "", content, flags=re.I)
        runtime = {p for p in runtime if p.name.casefold() not in consumers.casefold()}
    return {p for p in paths if p not in runtime or p.resolve() in protected}


def checkpoint_tools(build, environment, group=None):
    tools = {Path(sys.executable)}
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    tools.update(Path(value) for value in re.findall(r"^(?:CMAKE_(?:(?:C|CXX|CUDA|CUDA_HOST)_COMPILER|LINKER|AR|RANLIB)|JUICER_CARGO_EXECUTABLE):[^=]+=(.+)$", cache, re.M))
    for name in ("cmake", "ctest", "cargo", "rustc"):
        value = shutil.which(name, path=environment.get("PATH"))
        if value:
            tools.add(Path(value))
    if group in {None, "product", "tooling", "production"}:
        llvm = Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin") if os.name == "nt" else Path("/usr/lib/llvm-22/bin")
        suffix = ".exe" if os.name == "nt" else ""
        for name in ("clang-format", "clang-tidy", "clang", "clang-cl", "llvm-nm"):
            choices = [environment["JUICER_" + name.upper().replace("-", "_")]] if "JUICER_" + name.upper().replace("-", "_") in environment else [name + "-22", name, str(llvm / (name + suffix))]
            for choice in choices:
                resolved = shutil.which(choice, path=environment.get("PATH"))
                if resolved or Path(choice).is_file():
                    tools.add(Path(resolved or choice))
                    break
        if group == "production":
            tools.add(llvm / ("llvm-nm" + suffix))
            if os.name == "nt":
                tools.update(llvm / name for name in ("clang-cl.exe", "llvm-readobj.exe", "llvm-pdbutil.exe"))
    configs = {directory / ".cargo" / name for directory in (build.parents[2], *build.parents[2].parents)
               for name in ("config", "config.toml")}
    for probe in (("rust-naming", "rust-boundaries") if group in {None, "boundary", "tooling"} else ()):
        artifact = build.parents[2] / "out/validation" / build.name / "quality" / probe
        configs.update(parent / ".cargo" / name for parent in (artifact, *artifact.parents)
                       for name in ("config", "config.toml"))
    configs.update(Path(environment.get("CARGO_HOME", Path.home() / ".cargo")) / name for name in ("config", "config.toml"))
    configs.add(Path(environment.get("RUSTUP_HOME", Path.home() / ".rustup")) / "settings.toml")
    tools.update(configs)
    def bind(value, directory=None):
        if value:
            candidate = Path(value)
            if not candidate.is_absolute() and candidate.parent != Path(".") and directory:
                candidate = directory / candidate
            resolved = shutil.which(str(candidate), path=environment.get("PATH"))
            tools.add(Path(resolved) if resolved else candidate)
    for name, value in environment.items():
        if name in {"RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTDOC", "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER"} or name.startswith("CARGO_TARGET_") and name.endswith("_LINKER"):
            bind(value)
    for path in configs:
        if not path.is_file() or path.name == "settings.toml":
            continue
        config = tomllib.loads(path.read_text())
        for name in ("rustc", "rustc-wrapper", "rustc-workspace-wrapper", "rustdoc"):
            bind(config.get("build", {}).get(name), path.parent.parent)
        for target in config.get("target", {}).values():
            bind(target.get("linker"), path.parent.parent)
    rustup = shutil.which("rustup", path=environment.get("PATH"))
    if rustup:
        for name in ("cargo", "rustc", "clippy-driver"):
            tools.add(Path(subprocess.check_output([rustup, "which", name], env=environment, text=True, timeout=30).strip()))
    return [str(p.absolute()) for p in sorted(tools)]


def qualify(root, args):
    if args is None or not args.reason.strip() or not args.selection or set(args.selection) - GROUPS:
        raise ValueError("checkpoint reason and explicit required selections are mandatory")
    if args.preset not in PRESETS or args.runtime not in {"host", "all"}:
        raise ValueError("checkpoint requires a supported native preset and explicit host/all runtime")
    if args.preset.startswith("windows") != (os.name == "nt"):
        raise ValueError("checkpoint must execute on its named native platform")
    build = root / "out/build" / args.preset
    with OutputLease(root, build):
        return _qualify_owned(root, build, args)


def prepare_checkpoint(campaign, build, state_path, state, environment, *, repeat_reason="", timeout=1800, source_files=None):
    """Preparation repair and ordinary admission share the same producer and artifacts."""
    inputs = checkpoint_inputs(campaign.root, build, "prepare", source_files=source_files)
    outputs = []
    prepared_context = None
    admission_observations = {}
    previous = state.get("prepare")
    def prepare():
        if previous and previous.get("payloads"):
            try:
                path = Path(previous["receipt"])
                if digest(path) != previous["sha256"]:
                    raise ValueError("checkpoint preparation receipt changed")
                try:
                    check_preparation_owner(build, path)
                except ValueError:
                    pass  # A superseded/missing owner still needs ordinary preparation.
                else:
                    from verify import check_receipt_payloads
                    check_receipt_payloads(campaign.root, path)
                    try:
                        check_scoped_preparation(campaign, build, path, environment)
                    except ValueError:
                        # Changed inputs or missing outputs require the real producer.
                        pass
                    else:
                        if any(digest(Path(p)) != checksum for p, checksum in previous["payloads"].items()):
                            raise ValueError("checkpoint preparation artifacts changed under the completed producer")
            except BaseException:
                # Keep the original artifact facts attached to failed admission.
                state["prepare"] = previous.copy()
                raise
        # A reconfigure can reveal a previously unseen consumer. Observe omitted
        # candidates before commands, but bind them only if refresh admits them.
        candidates = set(checkpoint_inputs(campaign.root, build, "product", source_files=source_files)) - set(inputs)
        admission_observations.update(snapshot(campaign.root, sorted(candidates)))
        configuration = preparation_inventory(campaign.root, build, environment) if (build / "CMakeCache.txt").is_file() else None
        return prepare_build(campaign, "prepare", source=campaign.root, build=build,
            inputs=[p for p in inputs if not p.startswith("out/build/")], timeout=timeout,
            targets=["all"], configure_args=["-DBUILD_TESTING=ON"], env=environment, checkpoint=True,
            reconfigure=has_opaque_configure_commands((campaign.root / name for name in inputs), build=build,
                configuration=configuration))[1]
    def validate(path):
        nonlocal prepared_context
        record = check_scoped_preparation(campaign, build, path, environment)
        outputs[:] = record["scoped_preparation"]["outputs"]
        prepared_context = record["scoped_preparation"]["context"]
    return run_stage(campaign, state_path, state, "prepare", ["cmake", "--build", str(build)],
        inputs=inputs, tools=checkpoint_tools(build, environment, "product"), environment=environment,
        timeout=timeout, repeat_reason=repeat_reason,
        action=prepare,
        admission_observations=admission_observations,
        refresh_inputs=lambda: checkpoint_inputs(campaign.root, build, "prepare", source_files=source_files,
                                                 preparation=prepared_context),
        validate=validate, payloads=outputs)


def preparation_identity(build, stage, *, resolve=Path):
    """Bind admitted artifact facts; producer provenance remains in the original receipt.

    Callers validate raw command/process evidence through run_stage or collection.
    The artifact owner and exact output bytes must still match at consumption.
    """
    path = resolve(stage["receipt"])
    if digest(path) != stage["sha256"]:
        raise ValueError("checkpoint preparation receipt changed")
    record = check_preparation_owner(build, path, resolve=resolve)
    if record.get("status") != "passed" or record.get("returncode") != 0 or not record.get("process_tree", {}).get("drained"):
        raise ValueError("checkpoint preparation lacks successful process completion")
    proof = record["scoped_preparation"]
    check_preparation_inputs(build, proof, resolve=resolve)
    artifacts = stage.get("payloads", {})
    if not artifacts or set(artifacts) != set(proof["outputs"]):
        raise ValueError("checkpoint preparation lacks exact artifact identities; repeat preparation")
    if any(digest(resolve(p)) != checksum for p, checksum in artifacts.items()):
        raise ValueError("checkpoint preparation artifact bytes changed")
    facts = {name: value for name, value in proof.items()
             if name not in {"producer", "invalidated", "rust_invalidated_packages"}}
    return hashlib.sha256(json.dumps({"preparation": facts, "checkpoint": stage["identity"],
                                     "artifacts": artifacts}, sort_keys=True).encode()).hexdigest()


def _qualify_owned(root, build, args):
    from verify import source_snapshot
    cache = (build / "CMakeCache.txt").read_text()
    home = re.search(r"^CMAKE_HOME_DIRECTORY:[^=]+=(.+)$", cache, re.M)
    if not home or Path(home[1]).resolve() != root.resolve():
        raise ValueError("configured checkpoint tree belongs to another source directory")
    groups = set(args.selection)
    if "product" in groups and not args.base:
        raise ValueError("product checkpoints require the accepted diff base")
    if "production" in groups and (not args.preset.endswith("release") or groups != {"production"}):
        raise ValueError("production isolation is a separate explicit Release checkpoint on a testing-OFF tree")
    required = (["prepare", "quality"] if "product" in groups else []) + sorted(groups)
    print(json.dumps({"reason": args.reason, "preset": args.preset, "runtime": args.runtime,
                      "required_stages": required, "scope": "explicit preset checkpoint; full phase/release acceptance requires all retained native/profile and production gates"}, indent=2), flush=True)
    campaign = Campaign(root, args.preset, args.campaign or "qualification-" + uuid.uuid4().hex, required)
    state_path = build / "verification-checkpoint.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    current = source_snapshot(root)
    record = {"mode": "checkpoint", "status": "running", "scope_status": "running", "reason": args.reason,
              "preset": args.preset, "runtime": args.runtime, "selection": sorted(groups), "required": required,
              "source": current, "stages": {}}
    receipt = campaign.directory / "qualification.json"
    if receipt.exists():
        raise ValueError("checkpoint attempt already exists; retain it and use a new attempt name")
    started = time.monotonic()
    environment = dict(os.environ)
    def run(name, command, inputs, *, tools=None, **kwargs):
        if tools is None:
            tools = checkpoint_tools(build, environment, name if name in GROUPS else "product")
        result = run_stage(campaign, state_path, state, name, command, inputs=inputs, tools=tools,
                       environment=environment, timeout=1800, repeat_reason=args.repeat_reason, **kwargs)
        record["stages"][name] = result
        return result
    try:
        built = None
        prepared_identity = None
        if "product" in groups:
            built = prepare_checkpoint(campaign, build, state_path, state, environment,
                                       repeat_reason=args.repeat_reason, source_files=current)
            record["stages"]["prepare"] = built
            prepared_identity = preparation_identity(build, built)
            inputs = checkpoint_inputs(root, build, "product", source_files=current)
            run("quality", [sys.executable, "-B", "scripts/check-quality.py", "--preset", args.preset,
                "--qualification", "--base", args.base], inputs, dependencies=[prepared_identity])
        if groups != {"production"}:
            query = subprocess.check_output(["ctest", "--test-dir", str(build), "--show-only=json-v1"], cwd=root, env=environment, text=True, timeout=30)
            inventory = json.loads(query)
            selected = membership(inventory, groups, args.runtime)
            gpu_tests = {test["name"] for test in inventory["tests"]
                         if any(p["name"] == "LABELS" and "gpu" in p["value"] for p in test.get("properties", []))}
            record["membership"] = selected
            print("CTest membership:", json.dumps(selected, sort_keys=True), flush=True)
            record["runtime_membership"] = {}
            for group, names in selected.items():
                parts = {role: [n for n in names if (n in gpu_tests) == (role == "gpu")]
                         for role in ("host", "gpu")}
                parts = {role: members for role, members in parts.items() if members}
                record["runtime_membership"][group] = parts
                commands, reports = [], []
                for role, members in parts.items():
                    junit = campaign.directory / (group + "-" + role + ".xml")
                    pattern = "^(" + "|".join(re.escape(n) for n in members) + ")$"
                    commands.append(["ctest", "--test-dir", str(build), "-R", pattern,
                        "-LE" if role == "host" else "-L", "gpu", "--no-tests=error",
                        "--output-on-failure", "--output-junit", str(junit)])
                    reports.append(junit)
                identities = [ctest_identity(command) for command in commands]
                def validate(paths, parts=parts, names=names, identities=identities, group=group):
                    validate_ctest_stage(paths, parts, names, identities)
                    if group == "product":
                        check_scoped_preparation(campaign, build, Path(built["receipt"]))
                inputs = checkpoint_inputs(root, build, group, source_files=current)
                tools = checkpoint_tools(build, environment, group)
                run(group, commands[0], inputs, tools=tools, semantic_command=identities,
                    action=lambda group=group, commands=commands, parts=parts, inputs=inputs, tools=tools:
                        run_ctest_parts(campaign, group, commands, parts, inputs=inputs, tools=tools,
                            environment=environment, timeout=1800, repeat_reason=args.repeat_reason),
                    validate=validate, payloads=reports,
                    dependencies=[prepared_identity] if built and group == "product" else [])
        else:
            cache = (build / "CMakeCache.txt").read_text()
            if not re.search(r"^BUILD_TESTING:BOOL=OFF$", cache, re.M):
                raise ValueError("production isolation requires a separately prepared testing-OFF Release tree")
            from build_dependencies import NinjaGraph
            artifacts = sorted(p for n in NinjaGraph(build, ["juicer"], environment).outputs
                               if (p := Path(n)).is_file() and p.suffix.lower() in {".a", ".lib", ".ofx", ".pdb"})
            # TARGET_PDB is not a Ninja output. Its removal changes membership;
            # the harness then fails under the existing pending stage owner.
            pdb = build / "juicer.pdb"
            if os.name == "nt" and pdb.is_file():
                artifacts = sorted(set(artifacts) | {pdb})
            if not artifacts:
                raise ValueError("final production artifacts are missing")
            out = campaign.directory / "production"
            inputs = checkpoint_inputs(root, build, "production", source_files=current) + [p.relative_to(root).as_posix() for p in artifacts]
            command = [sys.executable, "-B", "tests/ffi/production_contract.py", "--preset", args.preset, "--artifacts", str(out)]
            run("production", command, inputs, semantic_command=[*command[:-1], "<artifacts>"], payloads=[out / "production.json"])
        if built:
            try:
                if preparation_identity(build, built) != prepared_identity:
                    raise ValueError("checkpoint preparation changed during consumption")
            except BaseException as error:
                state["prepare"] = {**built, "status": "failed", "error": str(error)}
                write_json(state_path, state)
                raise
        if source_snapshot(root) != current:
            raise ValueError("checkpoint source changed during execution")
        record["scope_status"] = "passed"
        record["pending"] = sorted(name for name, value in state.items() if value["status"] != "passed")
        record["status"] = "incomplete" if record["pending"] else "passed"
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        write_json(receipt, record)
    print("Checkpoint receipt:", receipt, flush=True)
    return receipt


def collect_qualification(root, paths):
    """Full phase/release aggregation is explicit; scoped passes cannot stand in."""
    if not paths:
        raise ValueError("all native/profile checkpoint receipts are required")
    from contextlib import ExitStack
    with ExitStack() as leases:
        for preset in PRESETS:
            leases.enter_context(OutputLease(root, root / "out/build" / preset))
        return _collect_owned(root, paths)


def _collect_owned(root, paths):
    def local(name):
        normalized = str(name).replace("\\", "/")
        if Path(name).is_absolute():
            return Path(name)
        if "/out/" in normalized:
            return root / "out" / normalized.split("/out/", 1)[1]
        if os.name != "nt" and re.match(r"^[A-Za-z]:/", normalized):
            return Path(subprocess.check_output(["wslpath", "-u", normalized], text=True).strip())
        return Path(normalized)
    records = [json.loads(Path(p).read_text()) for p in paths]
    for record in records:
        require_checkpoint(record)
    from verify import source_snapshot
    observed_source = source_snapshot(root) if all("source" in record for record in records) else None
    if observed_source is not None and any(record["source"] != observed_source for record in records):
        raise ValueError("checkpoint records do not bind one current candidate")
    stages = [stage for record in records for stage in record["stages"].values()]
    source_paths = sorted({p for stage in stages for p in stage["identity"]["inputs"]
                           if not p.startswith("out/build/")})
    tool_paths = sorted({str(local(p)) for stage in stages
                         for p in [*stage["identity"]["tool_paths"], *stage["identity"]["tools"]]})
    def observe_tools():
        bindings = {p: str(Path(p).resolve()) for p in tool_paths}
        values = tool_snapshot(tool_paths, allow_missing=True)
        if (set(values) != set(bindings.values())
                or bindings != {p: str(Path(p).resolve()) for p in tool_paths}):
            raise ValueError("checkpoint tool paths changed during collection")
        return bindings, values
    # One read-only operation shares reads, never facts from an earlier attempt.
    # The terminal scan seals content and lexical-to-resolved path bindings.
    bindings, tools = observe_tools()
    inputs = snapshot(root, source_paths)
    members, preparations = {}, {}
    for record in records:
        owner = root / "out/build" / record["preset"] / "verification-checkpoint.json"
        state = json.loads(owner.read_text())
        if any(value.get("status") != "passed" for value in state.values()):
            raise ValueError("pending checkpoint work prevents aggregate collection")
        for name, result in record["stages"].items():
            current = state.get(name, {})
            if current.get("status") != "passed" or any(current.get(key) != result.get(key)
                    for key in ("sha256", "identity", "additional_receipts", "payloads")):
                raise ValueError("pending or newer checkpoint ownership prevents collection: " + name)
            raw_paths = stage_receipts(result, local)
            for path, expected in zip(raw_paths, [result["sha256"], *result.get("additional_receipts", {}).values()], strict=True):
                if digest(path) != expected:
                    raise ValueError("checkpoint command receipt changed")
                command = json.loads(path.read_text())
                process = json.loads(local(command["process_tree"]["receipt"]).read_text())
                if (command.get("status") != "passed" or command.get("returncode") != 0
                        or not process.get("drained") or any(command["process_tree"].get(k) != v for k, v in process.items())
                        or digest(root / command["output"]) != command["output_sha256"]):
                    raise ValueError("checkpoint terminal evidence is invalid")
            if name == "production":
                production_payloads(*production_manifest_identity(raw_paths[0], root=root, resolve=local),
                                    root=root, inputs=result["identity"]["inputs"],
                                    resolve=local, retained=result.get("payloads", {}))
            for path, expected in result.get("payloads", {}).items():
                if digest(local(path)) != expected:
                    raise ValueError("checkpoint result payload changed")
            if name in record.get("membership", {}):
                validate_ctest_stage(raw_paths, record.get("runtime_membership", {}).get(name, {}),
                    record["membership"][name], result["identity"]["command"], local)
            declared = result["identity"]["inputs"]
            source_inputs = {path: checksum for path, checksum in declared.items() if not path.startswith("out/build/")}
            if {p: inputs[p] for p in source_inputs} != source_inputs:
                raise ValueError("checkpoint declared source/data inputs changed")
            group = name if name in GROUPS | {"prepare"} else "product"
            if name == "prepare":
                preparation_identity(root / "out/build" / record["preset"], result, resolve=local)
                preparations[record["preset"]] = json.loads(raw_paths[0].read_text())["scoped_preparation"]["context"]
            expected_members = {path for path in declared if not path.startswith("out/build/")}
            key = record["preset"], group
            if key not in members:
                discovery = {"source_files": observed_source}
                if group == "prepare":
                    discovery.update(preparation=preparations[record["preset"]], resolve=local)
                members[key] = {path for path in checkpoint_inputs(root, root / "out/build" / record["preset"], group,
                                                                  **discovery)
                                if not path.startswith("out/build/")}
            actual_members = members[key]
            if expected_members != actual_members:
                raise ValueError("checkpoint source/data membership changed")
            expected_tools = {bindings[str(local(path))]: value for path, value in result["identity"]["tools"].items()}
            actual_tools = {bindings[str(local(p))]: tools[bindings[str(local(p))]]
                            for p in result["identity"]["tool_paths"]}
            if actual_tools != expected_tools:
                raise ValueError("checkpoint toolchain/configuration changed")

        if "product" in record["selection"]:
            identity = preparation_identity(root / "out/build" / record["preset"], record["stages"]["prepare"], resolve=local)
            if any(record["stages"][name]["identity"]["dependencies"] != [identity] for name in ("quality", "product")):
                raise ValueError("checkpoint stages do not bind the current preparation artifacts")
    if observe_tools() != (bindings, tools):
        raise ValueError("checkpoint toolchain/configuration changed during collection")
    if snapshot(root, source_paths) != inputs:
        raise ValueError("checkpoint declared source/data inputs changed during collection")
    current = source_snapshot(root) if observed_source is not None else None
    if observed_source is not None and current != observed_source:
        raise ValueError("checkpoint source/data membership changed during collection")
    # Git selection omits ignored test/resource inputs that checkpoint_inputs owns.
    # Seal each shared membership observation independently of that selection.
    for (preset, group), expected in members.items():
        discovery = {"source_files": current}
        if group == "prepare":
            discovery.update(preparation=preparations[preset], resolve=local)
        actual_members = {path for path in checkpoint_inputs(root, root / "out/build" / preset, group,
                                                            **discovery)
                          if not path.startswith("out/build/")}
        if actual_members != expected:
            raise ValueError("checkpoint source/data membership changed during collection")
    required = {(p, group) for p in PRESETS for group in ("product", "boundary")}
    required |= {(p, "tooling") for p in ("linux-debug", "windows-clang-debug")}
    required |= {(p, "production") for p in ("linux-release", "windows-clang-release")}
    actual = [(r["preset"], g) for r in records for g in r["selection"] if g != "corpus"]
    if len(actual) != len(set(actual)) or set(actual) != required or sum("corpus" in r["selection"] for r in records) != 1:
        raise ValueError("full checkpoint membership is incomplete or duplicated")
    if any(r["runtime"] != "all" for r in records if "product" in r["selection"]):
        raise ValueError("host-only product checks do not establish full qualification")
    if current is None:
        current = source_snapshot(root)
    if any(r["source"] != current for r in records):
        raise ValueError("checkpoint records do not bind one current candidate")
    return {"mode": "qualification", "status": "passed", "source": current,
            "receipts": {str(p): digest(Path(p)) for p in paths}, "scope": "explicit complete retained checkpoint; independent review and host acceptance remain separate"}
