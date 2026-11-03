"""Explicit native checkpoints; direct CTest groups and existing command records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import struct
import subprocess
import sys
import time
import tomllib
import uuid

from build_evidence import check_preparation_inputs, check_preparation_owner, check_scoped_preparation, ctest_command, linker_tools, preparation_inventory, preparation_paths, prepare_build, registration_files
from execution import (Campaign, OutputLease, command_environment, digest, snapshot, tool_snapshot, write_json,
                       prepare_windows_identity_launch, mapped_windows_path, windows_runtime_paths,
                       windows_identity_interpreter, windows_path, read_bytes, read_text, read_json,
                       read_transcript, command_result_payloads, FileObservations, COMMAND_TRANSCRIPT_PROTOCOL,
                       _decode_command_receipt)
from build_dependencies import identity_json, read_identity_request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/ffi"))
from grain_probe_contract import GRAIN_FRAME_CASE_COUNT, GRAIN_FRAME_CASE_IDS
from effects_frame_probe_contract import EFFECTS_FRAME_CASE_COUNT, EFFECTS_FRAME_CASE_IDS

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
    cases = list(ET.fromstring(read_bytes(report)).iter("testcase"))
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


def artifact_membership(directories, resolve=Path):
    return {str(directory): sorted(str(path) for path in resolve(str(directory)).rglob("*") if path.is_file())
            if resolve(str(directory)).is_dir() else None for directory in directories}


def ctest_artifacts(inventory, names, build):
    """Bind registered runtime consumers, excluding explicit output destinations."""
    files, directories = set(), set()
    output_options = {"--target-dir", "--artifacts", "--scratch-root", "--output-dir", "--output-junit"}
    for test in inventory["tests"]:
        if test["name"] not in names:
            continue
        properties = {prop["name"]: prop["value"] for prop in test.get("properties", [])}
        working = Path(properties.get("WORKING_DIRECTORY", build))
        if not working.is_absolute():
            working = build / working
        command = test.get("command", [])
        for index, operand in enumerate(command):
            if (index and command[index - 1] in output_options
                    or operand.split("=", 1)[0] in output_options
                    or operand.startswith("CARGO_TARGET_DIR=")):
                continue
            path = Path(shutil.which(operand) or operand) if index == 0 else Path(operand)
            path = Path(os.path.abspath(path if path.is_absolute() else working / path))
            if not path.is_relative_to(build):
                continue
            if path.is_file():
                files.add(str(path))
            elif index and path.is_dir():
                directories.add(str(path))
                files.update(str(member) for member in path.rglob("*") if member.is_file())
    return sorted(files), sorted(directories)


def stage_receipts(result, resolve=Path):
    return [resolve(result["receipt"]), *(resolve(p) for p in result.get("additional_receipts", {}))]


def ctest_chunk(path, parts, commands, resolve=Path):
    """Read an exact invocation; a failed invocation can still contain passed cases."""
    import xml.etree.ElementTree as ET
    raw = read_json(str(path), resolve=resolve)
    command = raw["identity"]["command"]
    role = "host" if "-LE" in command else "gpu"
    if role not in parts:
        raise ValueError("checkpoint runtime partition is unexpected")
    report = resolve(command[command.index("--output-junit") + 1])
    cases = list(ET.fromstring(read_bytes(report)).iter("testcase"))
    members = sorted(c.get("name") for c in cases)
    if not members or len(members) != len(set(members)) or set(members) - set(parts[role]):
        raise ValueError("checkpoint CTest partition membership is incomplete or duplicated")
    expected = list(commands[list(parts).index(role)])
    expected[expected.index("-R") + 1] = "^(" + "|".join(re.escape(n) for n in members) + ")$"
    if ctest_identity(command) != expected:
        raise ValueError("checkpoint CTest command does not bind its runtime partition")
    passed = {c.get("name") for c in cases if c.get("status") == "run"
              and not any(c.find(n) is not None for n in ("failure", "error", "skipped"))}
    valid = (raw.get("status") == "passed" and raw.get("returncode") == 0 and passed == set(members)
             or raw.get("status") == "failed" and raw.get("returncode") == 8 and passed != set(members)
             and raw.get("process_tree", {}).get("status") == "completed")
    if not valid or not raw.get("process_tree", {}).get("drained"):
        raise ValueError("checkpoint CTest did not execute completely")
    return role, passed, report


def validate_ctest_stage(paths, parts, names, commands, resolve=Path):
    """Compose exact passed cases while retaining original failed invocations."""
    paths = paths if isinstance(paths, list) else [paths]
    selected = [name for members in parts.values() for name in members]
    if (not parts or list(parts) != [role for role in ("host", "gpu") if role in parts]
            or any(not members for members in parts.values())
            or len(selected) != len(set(selected)) or sorted(selected) != sorted(names)
            or len(commands) != len(parts)):
        raise ValueError("checkpoint runtime partition membership is incomplete or duplicated")
    complete = set()
    for path in paths:
        role, passed, _ = ctest_chunk(path, parts, commands, resolve)
        if role == "gpu" and set(parts.get("host", [])) - complete:
            raise ValueError("checkpoint GPU partition precedes completed host cases")
        if complete & passed:
            raise ValueError("checkpoint CTest success is duplicated")
        complete.update(passed)
    if complete != set(names):
        raise ValueError("checkpoint CTest membership did not execute completely")


def run_ctest_parts(campaign, group, commands, roles, *, build, inputs, tools, environment,
                    timeout, repeat_reason, state=None, state_path=None, payloads=None):
    # Normal qualification has no elapsed deadline. A finite timeout is only
    # supplied by deliberate failure controls.
    deadline = None if timeout is None else time.monotonic() + timeout
    expected_inputs = snapshot(campaign.root, inputs)
    present_tools = [p for p in tools if Path(p).is_file()]
    expected_tools = tool_snapshot(present_tools)
    chunks = state[group].setdefault("ctest_chunks", []) if state is not None else []
    identities = [ctest_identity(command) for command in commands]
    paths, complete, retained_payloads = [], set(), []
    from verify import check_receipt_payloads
    for chunk in chunks:
        path = Path(chunk["receipt"])
        log = path.with_name("LastTest.log")
        if (digest(path) != chunk["sha256"] or digest(Path(chunk["report"])) != chunk["report_sha256"]
                or not log.is_file() or digest(log) != chunk.get("last_test_log_sha256")):
            raise ValueError("checkpoint partial result payload changed")
        raw = check_receipt_payloads(campaign.root, path, allow_ctest_failure=True)
        if (raw["identity"]["inputs"] != expected_inputs or raw["identity"]["tools"] != expected_tools):
            raise ValueError("checkpoint partial result inputs changed")
        role, passed, report = ctest_chunk(path, roles, identities)
        if role == "gpu" and set(roles.get("host", [])) - complete:
            raise ValueError("checkpoint GPU partition precedes completed host cases")
        if passed & complete:
            raise ValueError("checkpoint partial success is duplicated")
        complete.update(passed)
        paths.append(path)
        retained_payloads.extend((report, log))
    for role, command in zip(roles, commands, strict=True):
        remaining = sorted(set(roles[role]) - complete)
        if not remaining:
            continue
        command = list(command)
        command[command.index("-R") + 1] = "^(" + "|".join(re.escape(n) for n in remaining) + ")$"
        def run():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("checkpoint CTest stage deadline expired")
            return campaign.run(group, command, inputs=inputs, tools=present_tools,
                expected_inputs=expected_inputs, expected_tools=expected_tools,
                env=environment, timeout=None if deadline is None else deadline - time.monotonic(),
                repeat_reason=repeat_reason)
        failure = None
        try:
            if role == "gpu":
                with OutputLease(campaign.root, timeout=None if deadline is None else max(0, deadline - time.monotonic())):
                    path = run()
            else:
                path = run()
        except subprocess.CalledProcessError as error:
            if error.returncode != 8:
                raise
            failure = error
            raw = campaign.attempts(group)[-1]
            path = (campaign.root / raw["output"]).with_name("result.json")
        # The next CTest partition replaces this build-local log.
        log = path.with_name("LastTest.log")
        shutil.copyfile(build / "Testing/Temporary/LastTest.log", log)
        check_receipt_payloads(campaign.root, path, allow_ctest_failure=True)
        if snapshot(campaign.root, inputs) != expected_inputs or tool_snapshot(present_tools) != expected_tools:
            raise ValueError("checkpoint inputs changed during CTest")
        _, passed, report = ctest_chunk(path, roles, identities)
        complete.update(passed)
        paths.append(path)
        retained_payloads.extend((report, log))
        chunks.append({"receipt": str(path), "sha256": digest(path),
                       "report": str(report), "report_sha256": digest(report),
                       "last_test_log_sha256": digest(log)})
        if state_path is not None:
            write_json(state_path, state)
        if failure is not None:
            raise failure
    if payloads is not None:
        payloads[:] = retained_payloads
    return paths


def production_manifest_identity(receipt, *, root, resolve=Path):
    """Read the identity emitted by the successful production process."""
    record = read_json(str(receipt), resolve=resolve)
    command = record["identity"]["command"]
    if command.count("--artifacts") != 1:
        raise ValueError("production command does not declare retained execution evidence")
    output = resolve(str(root / record["output"]))
    identities = re.findall(r"^PRODUCTION_MANIFEST_SHA256 ([0-9a-f]{64})$", read_text(output), re.M)
    if len(identities) != 1:
        raise ValueError("production manifest lacks its execution-time identity; fresh execution required")
    return resolve(command[command.index("--artifacts") + 1]) / "production.json", identities[0]


def production_transcript(row, role, directory, *, protocol=None, resolve=Path):
    """Validate one complete production command stream in its declared encoding."""
    path = directory / (row["name"] + "." + role)
    if protocol is None:
        try:
            payload = read_bytes(resolve(str(path)))
        except OSError as error:
            raise ValueError("production execution payload missing or changed: " + str(path)) from error
        if hashlib.sha256(payload).hexdigest() != row[role + "_sha256"]:
            raise ValueError("production execution payload missing or changed: " + str(path))
        return payload, str(path), row[role + "_sha256"]
    if protocol != COMMAND_TRANSCRIPT_PROTOCOL:
        raise ValueError("unsupported production transcript protocol")
    streams = row.get("streams")
    if not isinstance(streams, dict) or set(streams) != {"stdout", "stderr"}:
        raise ValueError("production command stream membership changed")
    reference = streams[role]
    if (not isinstance(reference, dict) or resolve(reference.get("path", "")) != resolve(str(path))
            or reference.get("decoded_sha256") != row[role + "_sha256"]):
        raise ValueError("production command stream path or decoded identity changed")
    payload = read_transcript(reference, resolve=resolve)
    return payload, str(path), reference["encoded_sha256"]



def production_payloads(manifest, expected_manifest_sha256, *, root, inputs, resolve=Path, retained=None):
    """Recheck retained OFF execution files independently of the current ON tree."""
    manifest = resolve(str(manifest))
    if not manifest.is_file() or digest(manifest) != expected_manifest_sha256:
        raise ValueError("production manifest execution identity changed")
    document = json.loads(read_text(manifest))
    transcript_protocol = document.get("transcript_protocol")
    if transcript_protocol not in (None, COMMAND_TRANSCRIPT_PROTOCOL):
        raise ValueError("unsupported production transcript protocol")
    for field in (
        "scanner_lut_results", "scanner_lut_fixture", "gaussian_results", "gaussian_fixture",
        "optics_results", "optics_fixture", "print_illuminant_results",
        "print_illuminant_fixture", "preflash_results", "preflash_fixture",
    ):
        if field in document:
            raise ValueError("production retired mathematical owner field remains: " + field)
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

    command_rows = None
    decoded_streams = {}

    def commands_for_run():
        nonlocal command_rows
        if command_rows is None:
            reference = document["commands"]
            if resolve(reference["file"]) != manifest.parent / "commands.json":
                raise ValueError("production retained command record is not the declared run")
            admit(reference["file"], reference["sha256"])
            command_rows = identity_json(read_text(resolve(reference["file"])))
            names = [row["name"] for row in command_rows]
            if len(names) != len(set(names)):
                raise ValueError("production retained command membership is incomplete or duplicated")
        return command_rows

    def command_stream(name, role):
        key = name, role
        if key not in decoded_streams:
            rows = [row for row in commands_for_run() if row["name"] == name]
            if len(rows) != 1:
                raise ValueError("production retained command membership is incomplete or duplicated")
            payload, path, checksum = production_transcript(
                rows[0], role, manifest.parent, protocol=transcript_protocol, resolve=resolve)
            admit(path, checksum)
            decoded_streams[key] = payload
        return decoded_streams[key]

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
    if document.get("build_testing") != "OFF" or not re.search(r"^BUILD_TESTING:BOOL=OFF$", read_text(cache), re.M):
        raise ValueError("production retained configuration is not testing-OFF")

    prepared_render = document.get("prepared_render")
    render_keys = {
        "executable", "executable_sha256_before", "executable_sha256_after",
        "command", "fixture", "fixture_sha256", "stdout_sha256",
        "stderr_sha256", "rows",
    }
    if not isinstance(prepared_render, dict) or set(prepared_render) != render_keys:
        raise ValueError("production prepared-render schema is incomplete or contains unknown fields")
    preset = document["preset"]
    target = "JuicerProductionPreparedRenderProbe"
    executable_name = target + suffix
    original_executable = build / "tests/bin" / executable_name
    retained_executable = manifest.parent / "build/tests/bin" / executable_name
    fixture = root / "tests/ofx/fixtures/exposure" / f"processor-{preset}.dat"
    source = root / "tests/ffi/production-prepared-render.cpp"
    overlay = root / "tests/print/illuminant-fixtures/product-updates.json"
    command = prepared_render["command"]
    matching_command = (isinstance(command, list) and len(command) == 5
                        and resolve(command[0]) == original_executable
                        and resolve(command[1]) == root / "Resources"
                        and resolve(command[2]) == fixture and command[3] == preset
                        and resolve(command[4]) == overlay)
    if (resolve(prepared_render["executable"]) != retained_executable
            or not matching_command
            or resolve(prepared_render["fixture"]) != fixture
            or len({prepared_render["executable_sha256_before"],
                    prepared_render["executable_sha256_after"]}) != 1):
        raise ValueError("production prepared-render execution identities disagree")
    target_key = str(resolve(str(original_executable)))
    if target_key not in declared or declared[target_key] != prepared_render["executable_sha256_before"]:
        raise ValueError("production prepared-render build artifact identity is missing or changed")
    admit(prepared_render["executable"], prepared_render["executable_sha256_after"])

    consumed = document.get("consumed_inputs", {})
    reference_consumed = {str(resolve(name)): checksum for name, checksum in consumed.items()}
    fixture_manifest = root / "tests/ofx/fixtures/exposure/manifest.json"
    for path in (root / "CMakeLists.txt", source, fixture_manifest, fixture,
                 root / "tests/print/illuminant-fixtures/manifest.json", overlay):
        checksum = inputs.get(path.relative_to(root).as_posix())
        if (checksum is None or digest(path) != checksum
                or reference_consumed.get(str(path.resolve())) != checksum):
            raise ValueError("production prepared-render consumed input identity changed")
    fixture_spec = json.loads(read_text(fixture_manifest))
    if (fixture_spec.get("files", {}).get(fixture.name) != prepared_render["fixture_sha256"]
            or digest(fixture) != prepared_render["fixture_sha256"]):
        raise ValueError("production prepared-render fixture identity changed")
    overlay_spec = json.loads(read_text(overlay, encoding="utf-8"))
    fixture_values = {}
    tokens = read_text(fixture, encoding="utf-8").split()
    position = 0
    while position < len(tokens):
        name = tokens[position]
        try:
            count = int(tokens[position + 1])
        except (IndexError, ValueError) as error:
            raise ValueError("production prepared-render fixture is malformed") from error
        position += 2
        if count < 0 or position + count > len(tokens):
            raise ValueError("production prepared-render fixture is truncated")
        try:
            values = [struct.unpack("<I", struct.pack("<f", float(value)))[0]
                      for value in tokens[position:position + count]]
        except (ValueError, OverflowError, struct.error) as error:
            raise ValueError("production prepared-render fixture contains an invalid sample") from error
        position += count
        if name in {"negative-direct", "negative-print"}:
            if name in fixture_values:
                raise ValueError("production prepared-render fixture has duplicate route rows")
            fixture_values[name] = values
    if set(fixture_values) != {"negative-direct", "negative-print"}:
        raise ValueError("production prepared-render fixture route membership is incomplete")
    for route, values in fixture_values.items():
        key = f"ofx/fixtures/exposure/{fixture.name}.{route}"
        patches = overlay_spec.get("fixtures", {}).get(key, [])
        if not isinstance(patches, list):
            raise ValueError("production prepared-render fixture overlay is malformed")
        for patch in patches:
            if (not isinstance(patch, dict) or set(patch) != {"op", "path", "value"}
                    or patch["op"] not in {"test", "replace"}
                    or not isinstance(patch["path"], str)
                    or not re.fullmatch(r"/(0|[1-9][0-9]*)", patch["path"])
                    or type(patch["value"]) is not int or not 0 <= patch["value"] <= 0xffffffff):
                raise ValueError("production prepared-render fixture overlay operation is malformed")
            sample = int(patch["path"][1:])
            if sample >= len(values):
                raise ValueError("production prepared-render fixture overlay index is out of range")
            if patch["op"] == "test":
                if values[sample] != patch["value"]:
                    raise ValueError("production prepared-render fixture overlay test failed")
            else:
                values[sample] = patch["value"]

    stdout = manifest.parent / "prepared-render.stdout"
    stderr = manifest.parent / "prepared-render.stderr"
    for path, checksum in ((stdout, prepared_render["stdout_sha256"]),
                           (stderr, prepared_render["stderr_sha256"])):
        if transcript_protocol:
            payload = command_stream("prepared-render", path.suffix.removeprefix("."))
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError("production prepared-render retained output missing or changed")
            continue
        if not path.is_file() or digest(path) != checksum:
            raise ValueError("production prepared-render retained output missing or changed")
        admit(str(path), checksum)

    row_fields = {
        "case_id", "route", "pass", "input_sha256", "recipe_hash", "status", "api",
        "native_code", "diagnostic", "input_shape", "output_shape",
        "output_f32_words_sha256", "accepted_fixture_match", "submitted",
        "readback_complete", "state_released_after_submit", "views_released_after_submit",
        "stream_completed", "context_preserved", "context_restored",
    }
    raw_fields = (row_fields - {"input_sha256", "output_f32_words_sha256"}) | {
        "input_f32_words", "output_f32_words",
    }
    expected_ids = ("negative-direct/cold", "negative-direct/warm",
                    "negative-print/cold", "negative-print/warm")
    expected_rows = prepared_render["rows"]
    prepared_stdout = command_stream("prepared-render", "stdout")
    try:
        raw_document = json.loads(prepared_stdout.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise ValueError("production prepared-render stdout is not its declared JSON result") from error
    raw_rows = raw_document.get("rows") if isinstance(raw_document, dict) and set(raw_document) == {"rows"} else None
    if (not isinstance(expected_rows, list) or len(expected_rows) != 4
            or tuple(row.get("case_id") for row in expected_rows if isinstance(row, dict)) != expected_ids
            or not isinstance(raw_rows, list) or len(raw_rows) != 4):
        raise ValueError("production prepared-render row membership is incomplete or duplicated")

    def word_digest(words):
        if (not isinstance(words, list) or any(type(word) is not int or not 0 <= word <= 0xffffffff
                                               for word in words)):
            raise ValueError("production prepared-render word array is malformed")
        return hashlib.sha256(b"".join(struct.pack("<I", word) for word in words)).hexdigest()

    def f32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    for index, (case_id, final, raw) in enumerate(zip(expected_ids, expected_rows, raw_rows, strict=True)):
        if (not isinstance(final, dict) or set(final) != row_fields
                or not isinstance(raw, dict) or set(raw) != raw_fields
                or final["case_id"] != case_id or raw["case_id"] != case_id):
            raise ValueError("production prepared-render row schema or order changed")
        route = "negative-direct" if index < 2 else "negative-print"
        components = 3 if route == "negative-direct" else 4
        shape = [5, 7, components]
        input_hash = word_digest(raw.get("input_f32_words"))
        output_hash = word_digest(raw.get("output_f32_words"))
        shapes = (final["input_shape"], final["output_shape"], raw["input_shape"], raw["output_shape"])
        if (final["route"] != route or raw["route"] != route
                or any(not isinstance(value, list) for value in shapes)
                or any(type(value) is not int for shape_values in shapes for value in shape_values)
                or final["input_shape"] != shape or final["output_shape"] != shape
                or raw["input_shape"] != shape or raw["output_shape"] != shape
                or len(raw["input_f32_words"]) != 5 * 7 * components
                or len(raw["output_f32_words"]) != 5 * 7 * components):
            raise ValueError("production prepared-render input/output shape changed")
        if (final["input_sha256"] != input_hash
                or final["output_f32_words_sha256"] != output_hash):
            raise ValueError("production prepared-render row word hashes disagree with stdout")
        raw_final = {key: value for key, value in raw.items()
                     if key not in {"input_f32_words", "output_f32_words"}}
        if final != {**raw_final, "input_sha256": input_hash,
                     "output_f32_words_sha256": output_hash}:
            raise ValueError("production prepared-render manifest rows disagree with stdout")
        if (final["pass"] != ("cold" if index in (0, 2) else "warm")
                or type(final["status"]) is not int or final["status"] != 0
                or type(final["api"]) is not int or final["api"] != 0
                or type(final["native_code"]) is not int or final["native_code"] != 0
                or final["accepted_fixture_match"] is not True
                or any(final[name] is not True for name in (
                    "submitted", "readback_complete", "state_released_after_submit",
                    "views_released_after_submit", "stream_completed", "context_preserved",
                    "context_restored"))
                or type(final["recipe_hash"]) is not int or not 0 <= final["recipe_hash"] < 2**64
                or not isinstance(final["diagnostic"], str)):
            raise ValueError("production prepared-render row did not prove successful completion")
        expected_words = fixture_values[route]
        actual_words = raw["output_f32_words"]
        if len(expected_words) != len(actual_words):
            raise ValueError("production prepared-render fixture output shape changed")
        for actual_word, expected_word in zip(actual_words, expected_words, strict=True):
            actual_value = struct.unpack("<f", struct.pack("<I", actual_word))[0]
            expected_value = struct.unpack("<f", struct.pack("<I", expected_word))[0]
            difference = abs(f32(actual_value - expected_value))
            allowed = f32(f32(2e-4) + f32(f32(3e-4) * abs(expected_value)))
            if not difference <= allowed:
                raise ValueError("production prepared-render output does not match the selected fixture")
    if (expected_rows[0]["output_f32_words_sha256"] != expected_rows[1]["output_f32_words_sha256"]
            or expected_rows[2]["output_f32_words_sha256"] != expected_rows[3]["output_f32_words_sha256"]):
        raise ValueError("production prepared-render warm output differs from its cold route")

    command_record = document["commands"]
    if resolve(command_record["file"]) != manifest.parent / "commands.json":
        raise ValueError("production retained command record is not the declared run")
    admit(command_record["file"], command_record["sha256"])
    commands = commands_for_run()
    names = [row["name"] for row in commands]
    if len(names) != len(set(names)) or names.count("prepared-render") != 1:
        raise ValueError("production retained command membership is incomplete or duplicated")
    prepared_command = next(row for row in commands if row["name"] == "prepared-render")
    if (prepared_command["command"] != prepared_render["command"]
            or prepared_command["stdout_sha256"] != prepared_render["stdout_sha256"]
            or prepared_command["stderr_sha256"] != prepared_render["stderr_sha256"]):
        raise ValueError("production prepared-render command execution identity changed")

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
    reference_consumed = {str(resolve(name)): checksum for name, checksum in consumed.items()}
    for path in (root / "tests/ffi/grain_frame_abi_assertions.inc",
                 root / "tests/ffi/grain_probe_contract.py",
                 root / "tests/ffi/effects_frame_abi_assertions.inc",
                 root / "tests/ffi/effects_frame_probe_contract.py"):
        checksum = digest(path)
        if (inputs.get(path.relative_to(root).as_posix()) != checksum
                or reference_consumed.get(str(path)) != checksum):
            raise ValueError("production grain C11 assertion input identity changed")
    reference = document.get("reference_fixture", {})
    reference_manifest = root / "tests/film/reference-fixtures/manifest.json"
    spec = json.loads(read_text(reference_manifest))
    reference_payload = reference_manifest.parent / spec["presets"][document["preset"]]
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
    for row in json.loads(read_text(reference_payload))["products"]:
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
    descriptor_spec = json.loads(read_bytes(descriptor_manifest))
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
    post_spec = json.loads(read_bytes(post_manifest))
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
    frame_manifest = root / "tests/frame/geometry-fixtures/manifest.json"
    frame_spec = json.loads(read_bytes(frame_manifest))
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
    halation_spec = json.loads(read_bytes(halation_manifest))
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
    diffusion_spec = json.loads(read_bytes(diffusion_manifest))
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
    dir_spec = json.loads(read_bytes(dir_manifest))
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

    effects_manifest = root / "tests/film/effects-frame-fixtures/manifest.json"
    effects_spec = json.loads(read_bytes(effects_manifest))
    effects_logical = effects_spec["presets"][document["preset"]]["native"]
    effects_paths = {
        "manifest": effects_manifest,
        "payload": effects_manifest.parent / effects_spec["payloads"][effects_logical],
        "context": effects_manifest.parent / effects_spec["contexts"][effects_logical],
        "normative": effects_manifest.parent / effects_spec["checked"],
        "input_spec": effects_manifest.parent / "inputs.json",
    }
    effects = document.get("effects_frame_fixture", {})
    if effects.get("logical_preset") != document["preset"] or any(
            resolve(effects.get(key, "")) != path for key, path in effects_paths.items()):
        raise ValueError("production effects frame fixture membership is incomplete")
    for path in effects_paths.values():
        checksum = digest(path)
        if (inputs.get(path.relative_to(root).as_posix()) != checksum
                or reference_consumed.get(str(path)) != checksum
                or (path != effects_manifest and checksum != effects_spec["sha256"][path.name])):
            raise ValueError("production effects frame consumed fixture identity changed")
    effects_expected = effects_spec["production_cases"][document["preset"]]
    if (tuple(row["case"] for row in effects_expected) != EFFECTS_FRAME_CASE_IDS
            or len(effects_expected) != EFFECTS_FRAME_CASE_COUNT
            or len({row["case"] for row in effects_expected}) != EFFECTS_FRAME_CASE_COUNT
            or document.get("effects_frame_results") != effects_expected):
        raise ValueError("production effects frame full values/failure/diagnostic results are incomplete")

    grain_manifest = root / "tests/grain/frame-fixtures/manifest.json"
    grain_spec = json.loads(read_bytes(grain_manifest))
    grain_paths = {
        "manifest": grain_manifest,
        "payload": grain_manifest.parent / grain_spec["presets"][document["preset"]]["native"],
        "normative": grain_manifest.parent / grain_spec["checked"],
        "input_spec": grain_manifest.parent / "inputs.json",
    }
    grain = document.get("grain_frame_fixture", {})
    if grain.get("logical_preset") != document["preset"] or any(
            resolve(grain.get(key, "")) != path for key, path in grain_paths.items()):
        raise ValueError("production grain frame fixture membership is incomplete")
    for path in grain_paths.values():
        checksum = digest(path)
        if (inputs.get(path.relative_to(root).as_posix()) != checksum
                or reference_consumed.get(str(path)) != checksum
                or (path != grain_manifest and checksum != grain_spec["sha256"][path.name])):
            raise ValueError("production grain frame consumed fixture identity changed")
    grain_expected = grain_spec["production_cases"]
    if (tuple(row["case"] for row in grain_expected) != GRAIN_FRAME_CASE_IDS
            or len(grain_expected) != GRAIN_FRAME_CASE_COUNT
            or len({row["case"] for row in grain_expected}) != GRAIN_FRAME_CASE_COUNT
            or document.get("grain_frame_results") != grain_expected):
        raise ValueError("production grain frame full values/metadata/failure/diagnostic results are incomplete")

    boundary_manifest = root / "tests/dir/boundary-fixtures/manifest.json"
    boundary_spec = json.loads(read_bytes(boundary_manifest))
    boundary_logical = boundary_spec["presets"][document["preset"]]["native"]
    boundary_paths = {
        "manifest": boundary_manifest,
        "payload": boundary_manifest.parent / boundary_logical,
        "normative": boundary_manifest.parent / boundary_spec["checked"],
        "input_spec": boundary_manifest.parent / "inputs.json",
    }
    boundary = document.get("dir_boundary_fixture", {})
    if boundary.get("logical_preset") != document["preset"] or any(
            resolve(boundary.get(key, "")) != path for key, path in boundary_paths.items()):
        raise ValueError("production DIR boundary fixture membership is incomplete")
    for path in boundary_paths.values():
        checksum = digest(path)
        if (inputs.get(path.relative_to(root).as_posix()) != checksum
                or reference_consumed.get(str(path)) != checksum
                or (path != boundary_manifest and checksum != boundary_spec["sha256"][path.name])):
            raise ValueError("production DIR boundary consumed fixture identity changed")
    boundary_expected = boundary_spec["production_cases"]
    if (len(boundary_expected) != 171 or len({row["case"] for row in boundary_expected}) != 171
            or document.get("dir_boundary_results") != boundary_expected):
        raise ValueError("production DIR boundary full values/key/metadata/raw/lifetime results are incomplete")

    headers = {name: checksum for name, checksum in consumed.items()
               if resolve(name).parent == manifest.parent and resolve(name).suffix == ".h"}
    if str(manifest.parent / "production-film-expected.h") not in {str(resolve(name)) for name in headers}:
        raise ValueError("production retained expectation header membership is incomplete")
    for name, checksum in headers.items():
        admit(name, checksum)
    if len(names) != len(set(names)) or not ({name + role for name in expected_names for role in ("-compile", "-values")} | {"prepared-render"}) <= set(names):
        raise ValueError("production retained command membership is incomplete or duplicated")
    outputs = {}
    for row in commands:
        if row["returncode"] != 0:
            raise ValueError("production retained command did not succeed")
        for extension in ("stdout", "stderr"):
            name = str(manifest.parent / (row["name"] + "." + extension))
            command_stream(row["name"], extension)
            outputs[name] = (row["streams"][extension]["encoded_sha256"] if transcript_protocol
                             else row[extension + "_sha256"])
        if row["name"] in {name + "-values" for name in expected_names}:
            name = row["name"].removesuffix("-values")
            if resolve(row["command"][0]) != manifest.parent / (name + suffix):
                raise ValueError("production command did not execute the linked probe")
    recorded_outputs = {str(resolve(name)): checksum for name, checksum in document.get("command_outputs", {}).items()}
    if recorded_outputs != outputs:
        raise ValueError("production retained detailed output membership is incomplete")
    for name, checksum in outputs.items():
        admit(name, checksum)
    film_stdout = command_stream("film-values", "stdout").decode("utf-8")
    effects_rows = [json.loads(line.removeprefix("EFFECTS_FRAME_RESULT "))
                    for line in film_stdout.splitlines()
                    if line.startswith("EFFECTS_FRAME_RESULT ")]
    if effects_rows != effects_expected:
        raise ValueError("production effects frame executed stdout results disagree")
    boundary_rows = [json.loads(line.removeprefix("DIR_BOUNDARY_RESULT "))
                     for line in film_stdout.splitlines()
                     if line.startswith("DIR_BOUNDARY_RESULT ")]
    if boundary_rows != boundary_expected:
        raise ValueError("production DIR boundary executed stdout does not match frozen full results")
    grain_rows = [json.loads(line.removeprefix("GRAIN_FRAME_RESULT "))
                  for line in film_stdout.splitlines() if line.startswith("GRAIN_FRAME_RESULT ")]
    if grain_rows != grain_expected:
        raise ValueError("production grain frame executed stdout results disagree")
    dir_rows = [json.loads(line.removeprefix("DIR_FRAME_RESULT "))
                for line in film_stdout.splitlines() if line.startswith("DIR_FRAME_RESULT ")]
    if dir_rows != dir_expected:
        raise ValueError("production DIR frame executed stdout results disagree")
    diffusion_rows = [json.loads(line.removeprefix("DIFFUSION_FRAME_RESULT "))
                      for line in film_stdout.splitlines() if line.startswith("DIFFUSION_FRAME_RESULT ")]
    if diffusion_rows != diffusion_expected:
        raise ValueError("production diffusion frame executed stdout results disagree")
    if retained is not None and {str(resolve(name)): checksum for name, checksum in retained.items()} != expected:
        raise ValueError("production retained payload membership or execution identities disagree; fresh execution required")
    return expected


def _stage_identity(root, command, inputs, tools, environment, dependencies, artifact_directories=()):
    identity = {"command": list(command), "inputs": snapshot(root, inputs),
                "tools": tool_snapshot(tools, allow_missing=True), "tool_paths": list(tools),
                "environment": hashlib.sha256(json.dumps({k: v for k, v in command_environment(environment).items()
                    if k not in {"JUICER_OUTPUT_LEASES", "JUICER_TEST_ARTIFACT_DIR"}}, sort_keys=True).encode()).hexdigest(),
                "dependencies": list(dependencies)}
    if artifact_directories:
        identity["artifact_directories"] = artifact_membership(artifact_directories)
    return identity


def _bounded_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _bounded_hash(value):
    return hashlib.sha256(_bounded_bytes(value)).hexdigest()


def _bounded_reference(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("bounded evidence is not a file")
    return {"path": str(path), "sha256": digest(path)}


def _bounded_payload(reference):
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str) or not Path(reference["path"]).is_absolute()
            or ".." in Path(reference["path"]).parts or not isinstance(reference["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])):
        raise ValueError("malformed bounded evidence reference")
    path = Path(reference["path"])
    if path.resolve(strict=True) != path:
        raise ValueError("bounded evidence is missing, retargeted or changed")
    raw = read_bytes(path)
    if hashlib.sha256(raw).hexdigest() != reference["sha256"] or path.resolve(strict=True) != path:
        raise ValueError("bounded evidence is missing, retargeted or changed")
    return raw


_POLICY_RECIPE_V1 = "native-c11-policy-miniature-v1"
_POLICY_RECIPE = "native-c11-policy-miniature-v2"
_POLICY_READ_RECIPES = {_POLICY_RECIPE_V1, _POLICY_RECIPE}
_POLICY_ASSURANCE = "synthetic-c11-policy-quality"
_POLICY_GROUPS = {"product", "boundary", "corpus", "production"}
_POLICY_MODULE_SOURCES = {
    "execution": "scripts/execution.py",
    "build_dependencies": "scripts/build_dependencies.py",
    "verify": "scripts/verify.py",
    "grain_probe_contract": "tests/ffi/grain_probe_contract.py",
    "effects_frame_probe_contract": "tests/ffi/effects_frame_probe_contract.py"
}


def _policy_namespaces():
    namespaces = _bounded_cli_namespaces()
    for name in ("build_evidence.py", "build_dependencies.py", "execution.py", "process_tree.py"):
        namespaces["scripts"][name] = ["file"]
    return namespaces


def _policy_reader_before(source):
    """The prospective fixture keeps the API and reverses only foreign resolution."""
    current = '''                if Path(operand).is_absolute() or PureWindowsPath(operand).is_absolute():
                    # A foreign absolute operand must reach its native-path
                    # resolver before a local parent can hide that identity.
                    target = resolve(operand)
                else:
                    target = path.parent / (operand.replace("\\\\", "/") if resolve is not Path else operand)
'''
    if source.count(current) != 1:
        raise ValueError("unsupported policy reader source; reviewed resolution body changed")
    return source.replace(current, "                target = resolve(path.parent / operand)\n", 1)


def _policy_templates():
    directory = Path(__file__).resolve().parents[1] / "tests/quality/evidence/policy"
    templates = {name: directory / filename for name, filename in (
        ("CMakeLists.txt", "CMakeLists.txt"), ("src/evidence_child.c", "evidence_child.c"),
        ("scripts/check-quality.py", "check_quality.py"),
        ("tests/quality/test_miniature.py", "test_miniature.py"))}
    templates.update({"scripts/" + name: Path(__file__).with_name(name)
                      for name in ("build_dependencies.py", "execution.py", "process_tree.py")})
    return templates


def _policy_policy(recipe=_POLICY_RECIPE):
    if recipe not in _POLICY_READ_RECIPES:
        raise ValueError("unsupported_protocol: unknown policy recipe")
    return {"profile": "declared-dependencies-v1", "recipe": recipe,
            "templates": {name: digest(path) for name, path in _policy_templates().items()},
            "controllers": ["scripts/build_evidence.py"], "source_namespaces": _policy_namespaces(),
            "environment_premise": ["qualified-os-driver-hardware", "ordinary-credentials-access-policy",
                                    "installed-system-libraries", "qualified-tool-runtime-distributions"],
            "scope": "synthetic policy contexts; actual host recorded separately; no GPU or product qualification"}


def _policy_presets(root, compiler, *, recipe=_POLICY_RECIPE):
    if recipe not in _POLICY_READ_RECIPES:
        raise ValueError("unsupported_protocol: unknown policy recipe")
    rows = []
    for preset in PRESETS:
        profile = "Release" if preset.endswith("release") else "Debug"
        variables = {"CMAKE_C_COMPILER": str(compiler), "CMAKE_BUILD_TYPE": profile,
                     "BUILD_TESTING": "ON", "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
                     "Python3_EXECUTABLE": sys.executable, "CMAKE_C_FLAGS": ""}
        if os.name == "nt":
            ranlib = Path(compiler).with_name("llvm-ranlib.exe")
            if not ranlib.is_file():
                raise ValueError("policy miniature requires the qualified native llvm-ranlib")
            variables.update(CMAKE_LINKER_TYPE="LLD", CMAKE_RANLIB=ranlib.as_posix(),
                CMAKE_C_FLAGS="/Brepro", CMAKE_C_FLAGS_DEBUG="/Od" + (" /Z7" if recipe == _POLICY_RECIPE_V1 else ""),
                CMAKE_C_FLAGS_RELEASE="/O2 /DNDEBUG" + (" /Z7" if recipe == _POLICY_RECIPE_V1 else ""),
                CMAKE_EXE_LINKER_FLAGS="/MANIFEST:NO /INCREMENTAL:NO /Brepro",
                CMAKE_EXE_LINKER_FLAGS_DEBUG=("/DEBUG " if recipe == _POLICY_RECIPE_V1 else "") + "/INCREMENTAL:NO /Brepro",
                CMAKE_EXE_LINKER_FLAGS_RELEASE=("/DEBUG " if recipe == _POLICY_RECIPE_V1 else "") + "/INCREMENTAL:NO /Brepro",
                CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded" + ("Debug" if profile == "Debug" else ""))
        rows.append({"name": preset, "generator": "Ninja", "binaryDir": str(root / "out/build" / preset),
                     "cacheVariables": variables})
    return {"version": 6, "configurePresets": rows,
            "testPresets": [{"name": preset, "configurePreset": preset,
                             "execution": {"jobs": 4, "noTestsAction": "error"}} for preset in PRESETS]}


def _policy_candidate(root, *, original=False):
    from build_evidence import capture_bounded_dependencies
    namespaces = _policy_namespaces()
    observed = capture_bounded_dependencies(root, files=[], directories=[root / name for name in namespaces])
    for name, members in namespaces.items():
        path = root / name
        row = observed["directories"][str(path)]
        if (row["resolved"] != str(path) or {r["name"] for r in row["members"]} != set(members)
                or any(r["kind"] not in members[r["name"]] or r["resolved"] != str(path / r["name"])
                       for r in row["members"])):
            raise ValueError("subject_changed: policy source/import namespace changed")
    from verify import source_snapshot
    names = {".gitignore", "CMakeLists.txt", "CMakePresets.json", "src/evidence_child.c", "input.txt",
             "scripts/qualification.py", "scripts/check-quality.py", "scripts/build_evidence.py",
             "scripts/build_dependencies.py", "scripts/execution.py", "scripts/process_tree.py",
             "tests/quality/test_miniature.py"}
    selected = set(subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                         cwd=root).decode().split("\0")) - {""}
    observed_source = source_snapshot(root)
    files = {name: observed_source.get(name) for name in names}
    if selected != names or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in files.values()):
        raise ValueError("subject_changed: unsupported policy source membership")
    for name, path in _policy_templates().items():
        if files[name] != digest(path):
            raise ValueError("subject_changed: policy template changed: " + name)
    native_reader = read_text(Path(__file__).with_name("build_evidence.py"), encoding="utf-8")
    if read_text(root / "scripts/build_evidence.py", encoding="utf-8") not in {native_reader, _policy_reader_before(native_reader)}:
        raise ValueError("unreviewed_transition: policy reader is neither reviewed source form")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip()
    if original and any(hashlib.sha256(subprocess.check_output(["git", "show", head + ":" + name], cwd=root)).hexdigest()
                        != checksum for name, checksum in files.items()):
        raise ValueError("incomplete_original: policy original must match its retained Git revision")
    return {"root": str(root), "head": head, "tree": tree, "files": files}


def _policy_producer(root):
    import build_evidence
    import importlib.machinery
    selected = Path(build_evidence.__file__).absolute()
    expected = root / "scripts/build_evidence.py"
    if selected != expected or selected.resolve(strict=True) != expected:
        raise ValueError("unsupported_protocol: policy fixture must execute its declared reader module")
    for name in ("build_evidence", "execution", "build_dependencies", "qualification", "verify",
                 "grain_probe_contract", "effects_frame_probe_contract"):
        module = sys.modules[name]
        if type(module.__spec__.loader) is not importlib.machinery.SourceFileLoader or module.__spec__.origin != module.__file__:
            raise ValueError("unsupported_protocol: policy producer has an unsupported project source loader: " + name)
    external = _bounded_producer()
    modules = {}
    for name, relative in _POLICY_MODULE_SOURCES.items():
        module = sys.modules[name]
        path = Path(module.__file__).absolute()
        expected_module = Path(external["root"]) / relative
        if (path != expected_module or path.resolve(strict=True) != expected_module
                or digest(path) != external["files"][relative]):
            raise ValueError("unsupported_protocol: policy producer did not load its fixed subject module: " + name)
        modules[name] = {"path": str(path), "resolved": str(path.resolve(strict=True)), "sha256": digest(path)}
    native_supervisor = root / "scripts/process_tree.py"
    supervisor = Path(sys.modules["execution"].__file__).absolute().with_name("process_tree.py")
    if native_supervisor.resolve(strict=True) != native_supervisor or digest(native_supervisor) != digest(supervisor):
        raise ValueError("unsupported_protocol: policy supervisor subject changed")
    if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13)
            or sys.pycache_prefix is not None or sys.flags.optimize != 0):
        raise ValueError("unsupported_protocol: policy producer needs its fixed CPython3.13 source selection")
    cache_policy = {"cache_tag": sys.implementation.cache_tag, "optimization": sys.flags.optimize, "pycache_prefix": None}
    sources = {Path(external["root"]) / name for name in external["files"]}
    sources.update({selected, *[Path(row["path"]) for row in modules.values()]})
    caches = {}
    import importlib.util
    for path in sorted(sources):
        cache = Path(importlib.util.cache_from_source(str(path)))
        if cache.exists() or cache.is_symlink():
            raise ValueError("unsupported_protocol: policy producer source cache is present: " + str(cache))
        caches[str(cache)] = {"resolved": str(cache.resolve(strict=False))}
    return {"external": external, "reader": {"path": str(selected), "resolved": str(selected.resolve()),
                                                        "sha256": digest(selected)}, "modules": modules,
            "supervisor": {"path": str(supervisor), "resolved": str(supervisor.resolve(strict=True)), "sha256": digest(supervisor)},
            "native_supervisor": {"path": str(native_supervisor), "resolved": str(native_supervisor.resolve(strict=True)),
                                  "sha256": digest(native_supervisor)},
            "cache_policy": cache_policy, "cache_absent": caches}


def _policy_tools(build, environment, group, *, executable=None):
    values = set(checkpoint_tools(build, environment, group))
    values.update({sys.executable, sys._base_executable})
    values.add(str(build.parents[2] / "scripts/build_evidence.py"))
    if group in {"product", "boundary", "corpus"}:
        values.add(str(build / ("BoundedEvidenceChild.exe" if os.name == "nt" else "BoundedEvidenceChild")))
    if executable is not None:
        values.add(str(executable))
    return sorted(values)


def _policy_payload(reference, resolve=Path):
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str)
            or not (Path(reference["path"]).is_absolute() or PureWindowsPath(reference["path"]).is_absolute())
            or not isinstance(reference["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])):
        raise ValueError("evidence_missing: malformed policy reference")
    if "\0" in reference["path"] or ".." in PureWindowsPath(reference["path"]).parts or ".." in Path(reference["path"]).parts:
        raise ValueError("evidence_missing: policy reference has an unsupported lexical path")
    path = resolve(reference["path"])
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("terminal_changed: policy reference path was retargeted: " + reference["path"])
    raw = read_bytes(path)
    if hashlib.sha256(raw).hexdigest() != reference["sha256"] or path.resolve(strict=True) != path:
        raise ValueError("terminal_changed: policy evidence changed: " + reference["path"])
    return raw


def _policy_write_shared_facts(campaign, candidate, producer, support):
    owner = str(campaign.directory.resolve(strict=True))
    record = {"protocol": "synthetic-policy-facts-v1", "owner_campaign": owner,
              "candidate": candidate, "producer": producer, "support": support}
    content = _bounded_bytes(record)
    checksum = hashlib.sha256(content).hexdigest()
    directory = campaign.directory / "policy-facts"
    if directory.is_symlink():
        raise ValueError("owner_conflict: policy shared-facts directory is a symlink")
    directory.mkdir(exist_ok=True)
    if directory.resolve(strict=True) != directory:
        raise ValueError("owner_conflict: policy shared-facts directory was retargeted")
    path = directory / (checksum + ".json")
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or read_bytes(path) != content:
            raise ValueError("owner_conflict: existing policy shared-facts blob differs")
    return {"path": str(path.resolve(strict=True)), "sha256": checksum}


def _policy_read_shared_facts(reference, owner_campaign, resolve=Path):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("owner_conflict: malformed policy shared-facts reference")
    if (not isinstance(owner_campaign, str)
            or not (Path(owner_campaign).is_absolute() or PureWindowsPath(owner_campaign).is_absolute())):
        raise ValueError("owner_conflict: malformed policy shared-facts campaign owner")
    logical_path = PureWindowsPath if PureWindowsPath(owner_campaign).is_absolute() else Path
    owner = logical_path(owner_campaign)
    checksum = reference.get("sha256") if isinstance(reference, dict) else None
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("owner_conflict: malformed policy shared-facts digest")
    expected = owner / "policy-facts" / (checksum + ".json")
    reference_path = logical_path(reference["path"])
    if (not owner.is_absolute() or not reference_path.is_absolute() or reference_path != expected
            or ".." in reference_path.parts or ".." in owner.parts):
        raise ValueError("owner_conflict: policy shared-facts path is outside its campaign owner")
    content = _policy_payload(reference, resolve)
    if hashlib.sha256(content).hexdigest() != checksum:
        raise ValueError("owner_conflict: policy shared-facts content address differs")
    record = identity_json(content.decode("utf-8"))
    if (content != _bounded_bytes(record) or set(record) != {"protocol", "owner_campaign", "candidate", "producer", "support"}
            or record["protocol"] != "synthetic-policy-facts-v1" or record["owner_campaign"] != owner_campaign):
        raise ValueError("owner_conflict: malformed or cross-owner policy shared-facts blob")
    support = record["support"]
    if (set(record["candidate"]) != {"root", "head", "tree", "files"}
            or set(record["producer"]) != {"external", "reader", "modules", "supervisor", "native_supervisor",
                                            "cache_policy", "cache_absent"}
            or set(support) != {"tooling", "tooling_stage", "preparation", "quality_stage", "quality"}
            or set(support["preparation"]) != {"identity", "stage"}
            or any(set(support[key]) != {"path", "sha256"} for key in ("tooling", "quality"))):
        raise ValueError("owner_conflict: malformed policy shared-facts document shape")
    return record


def _policy_read_admission(reference, *, resolve=Path):
    raw_bytes = _policy_payload(reference, resolve)
    record = identity_json(raw_bytes.decode("utf-8"))
    if not isinstance(record, dict):
        raise ValueError("owner_conflict: malformed policy admission")
    if record.get("protocol") == "synthetic-policy-admission-v1":
        fields = {"protocol", "assurance", "admitted_at", "declaration", "original", "current", "comparison",
                  "support", "owner", "chronology"}
        if set(record) != fields or record["assurance"] != _POLICY_ASSURANCE:
            raise ValueError("owner_conflict: malformed legacy policy admission")
        semantic = {key: value for key, value in record.items() if key not in {"protocol", "assurance", "admitted_at"}}
        return semantic, {"protocol": record["protocol"], "owner_campaign": None, "shared_facts": None}
    fields = {"protocol", "assurance", "admitted_at", "declaration", "original", "current", "comparison",
              "chronology", "owner_campaign", "shared_facts", "current_preparation"}
    if (raw_bytes != _bounded_bytes(record) or record.get("protocol") != "synthetic-policy-admission-v2" or set(record) != fields
            or record["assurance"] != _POLICY_ASSURANCE or not isinstance(record["current"], dict)
            or set(record["current"]) != {"policy", "context", "subject", "stage_identity", "counter"}
            or not isinstance(record["admitted_at"], str)):
        raise ValueError("owner_conflict: malformed policy admission v2")
    shared = _policy_read_shared_facts(record["shared_facts"], record["owner_campaign"], resolve)
    current = dict(record["current"])
    mode = record["current_preparation"]
    if mode == {"mode": "shared"}:
        preparation = identity_json(_bounded_bytes(shared["support"]["preparation"]).decode("utf-8"))
    elif isinstance(mode, dict) and set(mode) == {"mode", "value"} and mode["mode"] == "override":
        preparation = mode["value"]
        if preparation == shared["support"]["preparation"]:
            raise ValueError("owner_conflict: redundant policy preparation override")
    else:
        raise ValueError("owner_conflict: malformed policy preparation storage mode")
    current["candidate"] = shared["candidate"]
    current["producer"] = shared["producer"]
    current["preparation"] = preparation
    original_reference = record["original"]
    original_bytes = _policy_payload(original_reference, resolve)
    original_record = identity_json(original_bytes.decode("utf-8"))
    if isinstance(original_record, dict) and original_record.get("protocol") == "product-evidence-policy-v1":
        original = read_policy_contract(original_reference, resolve=resolve)["contract"]
    else:
        original, _ = _policy_decode_contract_envelope(original_reference, resolve=resolve)
    original_result = identity_json(_bounded_bytes(original["stage"]["result"]).decode("utf-8"))
    owner = {**original_result, "execution_contract": identity_json(_bounded_bytes(record["original"]).decode("utf-8"))}
    semantic = {key: value for key, value in record.items()
                if key not in {"protocol", "assurance", "admitted_at", "owner_campaign", "shared_facts", "current_preparation"}}
    semantic["current"] = current
    semantic["support"] = identity_json(_bounded_bytes(shared["support"]).decode("utf-8"))
    semantic["owner"] = owner
    stable = {**semantic, "owner_campaign": record["owner_campaign"], "shared_facts": record["shared_facts"]}
    logical_path = PureWindowsPath if PureWindowsPath(reference["path"]).is_absolute() else Path
    original_facts = original["facts"]
    native_path = PureWindowsPath if PureWindowsPath(original_facts["candidate"]["root"]).is_absolute() else Path
    expected_build = native_path(original_facts["candidate"]["root"]) / "out/build" / original_facts["context"]["preset"]
    expected_path = expected_build / "verification-admissions" / _bounded_hash(stable) / "admission.json"
    if logical_path(reference["path"]) != expected_path:
        raise ValueError("owner_conflict: policy admission content path differs")
    return semantic, {"protocol": record["protocol"], "owner_campaign": record["owner_campaign"],
                      "shared_facts": record["shared_facts"], "shared_record": shared}


def _policy_encode_admission(campaign, facts, admitted_at, *, admission_root, owner_campaign=None, shared_reference=None):
    owner_campaign = owner_campaign or str(campaign.directory.resolve(strict=True))
    shared_values = {"candidate": facts["current"]["candidate"], "producer": facts["current"]["producer"],
                     "support": facts["support"]}
    if shared_reference is None:
        shared_reference = _policy_write_shared_facts(campaign, **shared_values)
    stored_current = {key: value for key, value in facts["current"].items()
                      if key not in {"candidate", "producer", "preparation"}}
    if facts["current"]["preparation"] == facts["support"]["preparation"]:
        preparation_storage = {"mode": "shared"}
    else:
        preparation_storage = {"mode": "override", "value": facts["current"]["preparation"]}
    stable = {**facts, "owner_campaign": owner_campaign, "shared_facts": shared_reference}
    record = {"protocol": "synthetic-policy-admission-v2", "assurance": _POLICY_ASSURANCE,
              "admitted_at": admitted_at,
              "declaration": facts["declaration"], "original": facts["original"], "current": stored_current,
              "comparison": facts["comparison"], "chronology": facts["chronology"],
              "owner_campaign": owner_campaign, "shared_facts": shared_reference,
              "current_preparation": preparation_storage}
    path = Path(admission_root) / "verification-admissions" / _bounded_hash(stable) / "admission.json"
    return record, path


def _policy_receipt_decode_matches(retained, decoded, logical_path, resolve=Path):
    if not isinstance(retained, dict):
        return False
    if retained.get("protocol") != "command-receipt-v2":
        return decoded == retained
    expected = _decode_command_receipt(retained, logical_path, resolve=resolve,
                                       physical_receipt=resolve(logical_path))
    return decoded == expected


def _policy_raw(root, reference, resolve=Path, *, preparation=None):
    retained = identity_json(_policy_payload(reference, resolve).decode("utf-8"))
    raw = read_json(reference["path"], resolve=resolve)
    if not _policy_receipt_decode_matches(retained, raw, reference["path"], resolve):
        raise ValueError("incomplete_original: policy command receipt decode changed its retained record")
    process = read_json(raw["process_tree"]["receipt"], resolve=resolve)
    executable = raw["identity"]["command"][0]
    if not (Path(executable).is_absolute() or PureWindowsPath(executable).is_absolute()):
        stem = Path(executable).stem.casefold()
        matches = [name for name, checksum in raw["identity"]["tools"].items()
                   if Path(name.replace("\\", "/")).stem.casefold() == stem
                   and checksum == raw["identity"]["executable_sha256"]]
        if not matches and preparation is not None:
            if (not isinstance(preparation, dict) or preparation.get("status") != "passed"
                    or not isinstance(preparation.get("identity", {}).get("tools"), dict)):
                raise ValueError("incomplete_original: policy preparation tool anchor is not a passed sealed stage")
            anchor_ref = {"path": preparation["receipt"], "sha256": preparation["sha256"]}
            anchor_retained = identity_json(_policy_payload(anchor_ref, resolve).decode("utf-8"))
            anchor = read_json(anchor_ref["path"], resolve=resolve)
            if not _policy_receipt_decode_matches(anchor_retained, anchor, anchor_ref["path"], resolve):
                raise ValueError("incomplete_original: policy preparation receipt decode changed its retained record")
            proof = anchor.get("scoped_preparation", {})
            anchor_process = read_json(anchor["process_tree"]["receipt"], resolve=resolve)
            if (anchor.get("status") != "passed" or anchor.get("returncode") != 0
                    or anchor_process.get("status") != "completed" or anchor_process.get("returncode") != 0
                    or not anchor_process.get("drained")
                    or any(anchor["process_tree"].get(key) != value for key, value in anchor_process.items())
                    or digest(root / anchor["output"]) != anchor["output_sha256"]
                    or anchor.get("stage") != "prepare" or raw.get("stage") != "prepare"
                    or preparation["identity"]["command"][:3] != ["cmake", "--build", proof.get("build")]
                    or anchor["identity"]["command"][:3] != preparation["identity"]["command"][:3]
                    or executable != anchor["identity"]["command"][0]
                    or raw["identity"]["executable_sha256"] != anchor["identity"]["executable_sha256"]
                    or resolve(reference["path"]).parent.parent != resolve(anchor_ref["path"]).parent.parent
                    or raw["identity"]["environment_sha256"] != anchor["identity"]["environment_sha256"]):
                raise ValueError("incomplete_original: policy raw command does not belong to its preparation tool anchor")
            matches = [name for name, checksum in preparation["identity"]["tools"].items()
                       if Path(name.replace("\\", "/")).stem.casefold() == stem
                       and checksum == raw["identity"]["executable_sha256"]]
        if len(matches) != 1:
            raise ValueError("incomplete_original: policy command executable has no unique sealed tool binding")
        executable = matches[0]
    if (raw.get("status") != "passed" or raw.get("returncode") != 0 or not process.get("drained")
            or process.get("status") != "completed" or process.get("returncode") != 0
            or any(raw["process_tree"].get(k) != v for k, v in process.items())
            or digest(root / raw["output"]) != raw["output_sha256"]
            or digest(resolve(executable)) != raw["identity"]["executable_sha256"]):
        raise ValueError("incomplete_original: policy raw command/process evidence is invalid")
    return raw


def _policy_off_paths(proof):
    native_path = PureWindowsPath if PureWindowsPath(proof["source"]).is_absolute() else Path
    source, build = native_path(proof["source"]), native_path(proof["build"])
    context = proof["context"]
    configuration = context["configuration"]
    paths = set(proof["outputs"]) | set(context["files"]) | set(configuration["tools"])
    paths.update(str(source / name) for name in configuration["inputs"])
    paths.update(configuration["registrations"])
    paths.update(configuration["regeneration_mtimes"])
    local = {path for path in paths if native_path(path).is_relative_to(build)}
    return local, paths - local


def _policy_configuration(root, reference, *, preparation, resolve=Path):
    from build_evidence import SCOPED_PROTOCOL
    manifest = identity_json(_policy_payload(reference, resolve).decode("utf-8"))
    if set(manifest) != {"protocol", "producer", "producer_sha256", "outputs", "trace", "trace_sha256"} or manifest["protocol"] != SCOPED_PROTOCOL:
        raise ValueError("evidence_missing: retained configuration owner schema differs")
    producer = {"path": manifest["producer"], "sha256": manifest["producer_sha256"]}
    raw = _policy_raw(root, producer, resolve, preparation=preparation)
    if raw.get("configuration_evidence") != {key: manifest[key] for key in ("outputs", "trace", "trace_sha256")}:
        raise ValueError("evidence_missing: retained configuration producer does not bind its generated inputs")
    process = resolve(raw["process_tree"]["receipt"])
    refs = [reference, producer, {"path": manifest["trace"], "sha256": manifest["trace_sha256"]},
            {"path": str(root / raw["output"]), "sha256": raw["output_sha256"]},
            {"path": str(process), "sha256": digest(process)}]
    for ref in refs:
        _policy_payload(ref, resolve)
    return refs


def _policy_off_preparation(campaign, state):
    """OFF uses its normal scoped owner, without inventing an ON registration root."""
    build = campaign.root / "out/build" / campaign.directory.parent.name
    prepared = state["prepare"]
    _check_stage_owner(build / "verification-checkpoint.json", state, "prepare", prepared)
    record = check_scoped_preparation(campaign, build, Path(prepared["receipt"]), dict(os.environ))
    proof = record["scoped_preparation"]
    if (proof["configure_args"] != ["-DBUILD_TESTING=OFF"] or proof["targets"] != ["all"]
            or prepared.get("payloads") != {name: row["sha256"] for name, row in proof["output_identities"].items()}
            or any(digest(Path(path)) != checksum for path, checksum in prepared["payloads"].items())):
        raise ValueError("preparation_changed: actual OFF preparation/output owner differs")
    facts = {k: v for k, v in proof.items() if k not in {"producer", "invalidated", "rust_invalidated_packages"}}
    return {"identity": _preparation_facts_identity(facts, prepared), "stage": prepared}, proof


def policy_retain_off(campaign, state):
    """Retain the complete tiny OFF preparation before running its retained child."""
    build = campaign.root / "out/build" / campaign.directory.parent.name
    preparation, proof = _policy_off_preparation(campaign, state)
    if proof["configure_args"] != ["-DBUILD_TESTING=OFF"]:
        raise ValueError("subject_changed: policy OFF retention needs actual OFF preparation")
    destination = campaign.directory / "retained-off"
    destination.mkdir()
    paths, _ = _policy_off_paths(proof)
    rows = {}
    for index, name in enumerate(sorted(paths)):
        source = Path(name)
        retained = destination / (str(index) + "-" + source.name)
        shutil.copyfile(source, retained)
        if os.name == "posix":
            retained.chmod(source.stat().st_mode & 0o777)
        rows[name] = {"path": str(retained), "sha256": digest(retained)}
        if digest(source) != rows[name]["sha256"]:
            raise ValueError("terminal_changed: OFF retention changed while copying")
    binary = str(build / ("BoundedEvidenceChild.exe" if os.name == "nt" else "BoundedEvidenceChild"))
    record = {"protocol": "synthetic-policy-off-v1", "preset": build.name, "testing": "OFF",
              "preparation": preparation,
              "files": rows, "binary": rows[binary], "counter": str(destination / "counter.txt")}
    manifest = destination / "retained.json"
    manifest.write_bytes(_bounded_bytes(record))
    return _bounded_reference(manifest)


def _policy_off(reference, resolve=Path):
    record = identity_json(_policy_payload(reference, resolve).decode("utf-8"))
    if (set(record) != {"protocol", "preset", "testing", "preparation", "files", "binary", "counter"}
            or record["protocol"] != "synthetic-policy-off-v1" or record["testing"] != "OFF"
            or record["preset"] not in {"linux-release", "windows-clang-release"}):
        raise ValueError("evidence_missing: malformed synthetic OFF closure")
    prepared = record["preparation"]["stage"]
    raw = read_json(prepared["receipt"], resolve=resolve)
    proof = raw["scoped_preparation"]
    _policy_raw(resolve(proof["source"]), {"path": prepared["receipt"], "sha256": prepared["sha256"]},
                resolve, preparation=prepared)
    source_path = PureWindowsPath if PureWindowsPath(proof["source"]).is_absolute() else Path
    source = source_path(proof["source"])
    build = source_path(proof["build"])
    historical = {k: v for k, v in proof.items() if k not in {"producer", "invalidated", "rust_invalidated_packages"}}
    if (build != source / "out/build" / record["preset"]
            or record["preparation"]["identity"] != _preparation_facts_identity(historical, prepared)):
        raise ValueError("preparation_changed: retained OFF source/build/preparation identity differs")
    expected, _ = _policy_off_paths(proof)
    if set(record["files"]) != expected or proof["configure_args"] != ["-DBUILD_TESTING=OFF"]:
        raise ValueError("artifact_changed: synthetic OFF retained membership changed")
    for name, binding in record["files"].items():
        _policy_payload(binding, resolve)
        if name in proof["output_identities"] and binding["sha256"] != proof["output_identities"][name]["sha256"]:
            raise ValueError("artifact_changed: ON bytes substituted for OFF payload")
        if name not in proof["output_identities"]:
            relative = source_path(name).relative_to(source).as_posix()
            if proof["context"]["configuration"]["inputs"].get(relative) != binding["sha256"]:
                raise ValueError("artifact_changed: retained OFF configuration differs from original preparation")
    binaries = [row for name, row in record["files"].items() if Path(name.replace("\\", "/")).name in
                {"BoundedEvidenceChild", "BoundedEvidenceChild.exe"}]
    if len(binaries) != 1 or record["binary"] != binaries[0]:
        raise ValueError("artifact_changed: synthetic OFF binary does not bind retained output")
    cache_name = str(build / "CMakeCache.txt")
    cache = _policy_payload(record["files"][cache_name], resolve).decode("utf-8").splitlines()
    if "BUILD_TESTING:BOOL=OFF" not in cache or "CMAKE_BUILD_TYPE:STRING=Release" not in cache:
        raise ValueError("artifact_changed: retained OFF configuration has the wrong actual feature/profile")
    registry = str(build / "verification-config-outputs.json")
    if registry in record["files"]:
        _policy_configuration(resolve(proof["source"]), record["files"][registry], preparation=prepared, resolve=resolve)
    return record


def _policy_build_subject(root, build, environment, registrations):
    from build_dependencies import NinjaGraph
    reader = root / "scripts/build_evidence.py"
    graph = NinjaGraph(build, ["all"], environment)
    actual = _bounded_build_subject(root, build, environment)
    consumed = {str(p) for p in graph.source_files()} | set(actual["configuration"]["regeneration_mtimes"])
    if (str(reader) in consumed or str(reader.resolve()) in consumed or "build_evidence.py" in graph.commands
            or any("build_evidence.py" in read_text(Path(path), encoding="utf-8") for path in registrations)):
        raise ValueError("subject_changed: reader is an actual graph/regeneration/registered runtime consumer")
    # preparation_inventory adds its producer helpers after querying Ninja. The
    # full broad proof remains intact; this separately checked view removes only
    # the named reader's outer evidence contribution.
    actual = identity_json(_bounded_bytes(actual).decode("utf-8"))
    actual["files"].pop(str(reader), None)
    actual["configuration"]["inputs"].pop("scripts/build_evidence.py", None)
    actual["configuration"]["tools"].pop(str(reader), None)
    return actual


def _policy_capture(campaign, state, name, command, inputs, tools, environment, *, off=None):
    from build_evidence import capture_bounded_dependencies
    import sysconfig
    root = campaign.root
    build = root / "out/build" / campaign.directory.parent.name
    OutputLease.check_held()
    _check_bounded_cli_environment(environment)
    injection = {"LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
                 "CPATH", "C_INCLUDE_PATH", "GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH", "CL", "_CL_",
                 "CMAKE_TOOLCHAIN_FILE", "CMAKE_PROJECT_TOP_LEVEL_INCLUDES"}
    if any(environment.get(key) for key in injection):
        raise ValueError("unsupported bounded project compiler/runtime injection")
    for key in ("PATH", "INCLUDE", "LIB"):
        if any(value and Path(value).resolve().is_relative_to(root)
               for value in environment.get(key, "").split(os.pathsep)):
            raise ValueError("bounded environment searches the mutable miniature project")
    candidate = _policy_candidate(root)
    compiler = shutil.which("clang-cl.exe", path=environment.get("PATH")) if os.name == "nt" else "/usr/bin/gcc-13"
    if identity_json(read_text(root / "CMakePresets.json")) != _policy_presets(root, compiler):
        raise ValueError("subject_changed: policy compiler/configuration injection")
    cache = read_text(build / "CMakeCache.txt", encoding="utf-8")
    profile = "Release" if build.name.endswith("release") else "Debug"
    testing = "OFF" if name == "production" else "ON"
    if (not re.search(r"^CMAKE_BUILD_TYPE:STRING=" + profile + r"$", cache, re.M)
            or not re.search(r"^BUILD_TESTING:BOOL=" + testing + r"$", cache, re.M)):
        raise ValueError("subject_changed: policy actual profile/testing cache differs")
    if name not in _POLICY_GROUPS:
        raise ValueError("unsupported_protocol: policy original group is unsupported")
    if name == "production":
        if off is None:
            off = _bounded_reference(Path(command[0]).parent / "retained.json")
        retained = _policy_off(off)
        preparation, _ = _policy_off_preparation(campaign, state)
        if preparation != retained["preparation"]:
            raise ValueError("owner_conflict: original OFF execution does not bind its current retained preparation")
        prepared = retained["preparation"]["stage"]
        prepared_identity = retained["preparation"]["identity"]
        proof = read_json(prepared["receipt"])["scoped_preparation"]
        binary, counter = Path(retained["binary"]["path"]), Path(retained["counter"])
        if list(command) != [str(binary), str(root / "input.txt"), str(counter)]:
            raise ValueError("subject_changed: policy OFF command differs from its retained child")
        registrations, registered = {}, None
        files = {Path(row["path"]) for row in retained["files"].values()}
        _, external = _policy_off_paths(proof)
        files.update(Path(path) for path in external)
    else:
        retained = None
        prepared = state["prepare"]
        if prepared.get("status") != "passed":
            raise ValueError("owner_conflict: current policy preparation is not passed")
        prepared_identity = preparation_identity(build, prepared)
        proof = read_json(prepared["receipt"])["scoped_preparation"]
        if proof["configure_args"] != ["-DBUILD_TESTING=ON"] or proof["targets"] != ["all"]:
            raise ValueError("preparation_changed: policy stage requires ordinary ON all-target preparation")
        inventory = json.loads(subprocess.check_output(["ctest", "--test-dir", str(build), "--show-only=json-v1"],
                                                       cwd=root, env=environment, text=True))
        expected_members = {group: ["Policy." + group] for group in ("product", "boundary", "corpus")}
        expected_members["tooling"] = ["Tooling.Miniature"]
        if membership(inventory, set(expected_members), "all") != expected_members or len(inventory["tests"]) != 4:
            raise ValueError("incomplete_original: policy exact registered membership changed")
        registered = next(t for t in inventory["tests"] if t["name"] == "Policy." + name)
        binary, counter = build / ("BoundedEvidenceChild.exe" if os.name == "nt" else "BoundedEvidenceChild"), build / ("counter-" + name + ".txt")
        properties = {row["name"]: row["value"] for row in registered["properties"]}
        if ([Path(p) for p in registered["command"]] != [binary, root / "input.txt", counter]
                or set(properties) != {"LABELS", "TIMEOUT", "WORKING_DIRECTORY"}
                or set(properties["LABELS"]) != {"host", name} or properties["TIMEOUT"] != 0
                or Path(properties["WORKING_DIRECTORY"]) != build):
            raise ValueError("subject_changed: policy child command/properties changed")
        expected_command = [*ctest_command(root, build), "-R", "^(Policy\\." + name + ")$", "-LE", "gpu",
                            "--no-tests=error", "--output-on-failure", "--output-junit", command[-1]]
        if list(command) != expected_command:
            raise ValueError("subject_changed: policy CTest command form changed")
        registrations = {str(p): digest(p) for p in registration_files(build)}
        if len(registrations) != 4:
            raise ValueError("incomplete_original: policy generated registration chain is incomplete")
        files = {Path(p) for p in proof["context"]["files"]} | {Path(p) for p in proof["outputs"]}
        files.update(Path(p) for p in registrations)
        files.update(root / p for p in proof["context"]["configuration"]["inputs"])
        files.update(Path(p) for p in proof["context"]["configuration"]["tools"])
    files.update(root / p for p in candidate["files"])
    files.update(Path(p) for p in tools if Path(p).is_file())
    files.discard(root / "scripts/build_evidence.py")
    absent = {Path(p) for p in tools if not Path(p).exists()}
    absent.update({build / "CTestCustom.cmake", build / "CTestCustom.ctest"})
    files.difference_update(absent)
    executables = {binary}
    for path in files:
        if (name != "production" and path.name.lower() in {"ctest", "ctest.exe"}) or path == Path(sys.executable):
            executables.add(path)
    dependencies = capture_bounded_dependencies(root, files=sorted(files), absent=sorted(absent),
        directories=[root / n for n in _policy_namespaces()], executables=sorted(executables))
    actual = _policy_build_subject(root, build, environment, registrations) if name != "production" else None
    if actual is not None:
        reader = str(root / "scripts/build_evidence.py")
        if reader in actual["files"] or "scripts/build_evidence.py" in actual["configuration"]["inputs"]:
            raise ValueError("subject_changed: reader is an actual build consumer")
    facts = {k: v for k, v in proof.items() if k not in {"producer", "invalidated", "rust_invalidated_packages"}}
    return {"policy": _policy_policy(), "candidate": candidate,
            "context": {"preset": build.name, "native_target": sysconfig.get_platform(),
                        "profile": "Release" if build.name.endswith("release") else "Debug",
                        "testing": "OFF" if name == "production" else "ON", "features": [],
                        "runtime": "all", "group": name, "scope": "synthetic-policy"},
            "subject": {"dependencies": dependencies, "preparation_facts": facts, "build_subject": actual,
                        "registrations": registrations, "registration": registered, "off": off,
                        "runtime": {"python": sys.executable, "base_python": sys._base_executable}},
            "preparation": {"identity": prepared_identity, "stage": prepared}, "producer": _policy_producer(root),
            "stage_identity": _stage_identity(root, list(command) if name == "production" else [ctest_identity(command)],
                                               inputs, tools, environment, [prepared_identity]),
            "counter": str(counter)}


def _policy_entry(campaign, state, name, command, identity, inputs, tools, environment):
    facts = _policy_capture(campaign, state, name, command, inputs, tools, environment)
    if facts["stage_identity"] != identity or Path(facts["counter"]).exists() or Path(facts["counter"]).is_symlink():
        raise ValueError("incomplete_original: policy original requires an unexecuted counter and exact identity")
    facts["candidate"] = _policy_candidate(campaign.root, original=True)
    return {"policy_facts": facts, "launch": _bounded_launch(environment),
            "prior": [r["output"] for r in campaign.attempts(name)]}


def _policy_finish(campaign, state, name, command, result, entry, inputs, tools, environment):
    facts = entry["policy_facts"]
    paths = stage_receipts(result)
    if len(paths) != 1 or not paths[0].is_relative_to(campaign.directory / "attempts"):
        raise ValueError("incomplete_original: policy original requires one fresh complete execution")
    reference = _bounded_reference(paths[0])
    raw = _policy_raw(campaign.root, reference)
    if raw["output"] in entry["prior"] or raw["stage"] != name:
        raise ValueError("incomplete_original: policy execution was replayed or relabelled")
    if read_bytes(Path(facts["counter"])) != b"1\n":
        raise ValueError("incomplete_original: policy child did not assert exactly once")
    if name != "production":
        validate_ctest_stage(paths, {"host": ["Policy." + name]}, ["Policy." + name], [ctest_identity(command)])
    transcript = read_text(paths[0].with_name("LastTest.log") if name != "production" else campaign.root / raw["output"], encoding="utf-8")
    if not re.search(r"(?m)^result=14\r?$", transcript) or not re.search(r"(?m)^testing=" + facts["context"]["testing"] + r"\r?$", transcript):
        raise ValueError("incomplete_original: policy child result/testing output was not retained")
    terminal = _policy_capture(campaign, state, name, command, inputs, tools, environment)
    if terminal != facts or _bounded_launch(environment) != entry["launch"]:
        raise ValueError("terminal_changed: policy subject or producer changed during execution")
    references = {"receipt": reference, "output": _bounded_reference(campaign.root / raw["output"]),
                  "process": _bounded_reference(raw["process_tree"]["receipt"]),
                  "counter": _bounded_reference(facts["counter"])}
    if name != "production":
        references.update(junit=_bounded_reference(command[-1]), last_test=_bounded_reference(paths[0].with_name("LastTest.log")))
    stored_facts = facts
    stored_result = {key: value for key, value in result.items() if key != "identity"}
    document = {"protocol": "product-evidence-policy-v2", "assurance": _POLICY_ASSURANCE,
                "facts": stored_facts, "stage": {"name": name, "result": stored_result}, "execution": references,
                "started_at": raw["started_at"], "finished_at": raw["finished_at"], "launch": entry["launch"],
                "seals": {"entry": _bounded_hash(stored_facts), "terminal": _bounded_hash(stored_facts)}}
    reader_bytes = read_bytes(Path(facts["producer"]["reader"]["path"]))
    reader_sha256 = hashlib.sha256(reader_bytes).hexdigest()
    if reader_sha256 != facts["producer"]["reader"]["sha256"]:
        raise ValueError("terminal_changed: actual original reader source changed")
    document["producer_source"] = _write_policy_reader_source(campaign.root, reader_bytes)
    sidecar = paths[0].with_name("execution-contract.json")
    with sidecar.open("xb") as stream:
        stream.write(_bounded_bytes(document))
    return _bounded_reference(sidecar)


def _policy_reader_source_path(owner_root, checksum):
    owner_root = Path(owner_root)
    if not owner_root.is_dir() or owner_root.resolve(strict=True) != owner_root:
        raise ValueError("owner_conflict: policy reader-source owner is missing or retargeted")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("owner_conflict: malformed policy reader-source identity")
    return owner_root / "out/policy-reader-sources" / (checksum + ".py")


def _write_policy_reader_source(owner_root, content):
    checksum = hashlib.sha256(content).hexdigest()
    path = _policy_reader_source_path(owner_root, checksum)
    output_root = path.parent.parent
    if output_root.is_symlink():
        raise ValueError("owner_conflict: policy output directory is a symlink")
    output_root.mkdir(exist_ok=True)
    if output_root.resolve(strict=True) != output_root:
        raise ValueError("owner_conflict: policy output directory was retargeted")
    directory = path.parent
    if directory.is_symlink():
        raise ValueError("owner_conflict: policy reader-source directory is a symlink")
    directory.mkdir(exist_ok=True)
    if directory.resolve(strict=True) != directory:
        raise ValueError("owner_conflict: policy reader-source directory was retargeted")
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        existing = read_bytes(path) if path.is_file() and not path.is_symlink() else None
        if (existing is None or hashlib.sha256(existing).hexdigest() != checksum or existing != content):
            raise ValueError("owner_conflict: existing policy reader-source blob differs")
    return _bounded_reference(path)


def _policy_decode_contract_envelope(reference, *, resolve=Path):
    raw_bytes = _policy_payload(reference, resolve)
    document = identity_json(raw_bytes.decode("utf-8"))
    protocols = {"product-evidence-policy-v1", "product-evidence-policy-v2"}
    if (not isinstance(document, dict) or set(document) != {"protocol", "assurance", "facts", "stage", "execution",
            "started_at", "finished_at", "launch", "seals", "producer_source"} or document.get("protocol") not in protocols
            or document["assurance"] != _POLICY_ASSURANCE):
        raise ValueError("unsupported_protocol: malformed prospective policy original")
    version = document["protocol"]
    if (not isinstance(document["facts"], dict) or not isinstance(document["stage"], dict)
            or not {"name", "result"}.issubset(document["stage"])
            or not isinstance(document["stage"]["name"], str)
            or not isinstance(document["stage"]["result"], dict)
            or (version == "product-evidence-policy-v2" and set(document["stage"]) != {"name", "result"})
            or raw_bytes != _bounded_bytes(document)):
        raise ValueError("unsupported_protocol: malformed prospective policy original")
    facts, result = document["facts"], document["stage"]["result"]
    if version == "product-evidence-policy-v2":
        if (set(facts) != {"policy", "candidate", "context", "subject", "preparation", "producer", "stage_identity", "counter"}
                or not isinstance(facts["stage_identity"], dict) or "identity" in result
                or document["seals"] != {"entry": _bounded_hash(facts), "terminal": _bounded_hash(facts)}):
            raise ValueError("incomplete_original: malformed compact policy original facts or seals")
        result["identity"] = identity_json(_bounded_bytes(facts["stage_identity"]).decode("utf-8"))
    else:
        if document["seals"] != {"entry": _bounded_hash(facts), "terminal": _bounded_hash(facts)}:
            raise ValueError("incomplete_original: policy original seals disagree")
    return document, version


def read_policy_contract(reference, *, resolve=Path):
    document, version = _policy_decode_contract_envelope(reference, resolve=resolve)
    facts, result = document["facts"], document["stage"]["result"]
    name = document["stage"]["name"]
    context = facts.get("context", {})
    candidate = facts.get("candidate", {})
    if (set(context) != {"preset", "native_target", "profile", "testing", "features", "runtime", "group", "scope"}
            or set(candidate) != {"root", "head", "tree", "files"}
            or not all(isinstance(candidate.get(k), str) and re.fullmatch(r"[0-9a-f]{40}", candidate[k]) for k in ("head", "tree"))
            or context.get("profile") != ("Release" if context.get("preset", "").endswith("release") else "Debug")
            or context.get("testing") != ("OFF" if name == "production" else "ON")
            or context.get("native_target") != ("win-amd64" if PureWindowsPath(candidate.get("root", "")).is_absolute() else "linux-x86_64")
            or set(result) != {"status", "identity", "receipt", "sha256", "additional_receipts", "payloads"} | (set() if name == "production" else {"ctest_chunks"})
            or set(result["identity"]) != {"command", "inputs", "tools", "tool_paths", "environment", "dependencies"}
            or set(facts.get("preparation", {})) != {"identity", "stage"}):
        raise ValueError("incomplete_original: policy context/stage/preparation schema differs")
    recipe = facts.get("policy", {}).get("recipe") if isinstance(facts.get("policy"), dict) else None
    if (set(facts) != {"policy", "candidate", "context", "subject", "preparation", "producer", "stage_identity", "counter"}
            or recipe not in _POLICY_READ_RECIPES or facts["policy"] != _policy_policy(recipe) or name not in _POLICY_GROUPS
            or facts["context"]["group"] != name or facts["context"]["scope"] != "synthetic-policy"
            or facts["context"]["preset"] not in PRESETS or facts["context"]["runtime"] != "all"
            or facts["context"]["features"] != [] or result.get("status") != "passed"
            or result.get("reuse_admission") or result.get("additional_receipts")
            or facts["stage_identity"] != result["identity"]
            or result["identity"]["dependencies"] != [facts["preparation"]["identity"]]
            or (version == "product-evidence-policy-v1"
                and document["seals"] != {"entry": _bounded_hash(facts), "terminal": _bounded_hash(facts)})):
        raise ValueError("incomplete_original: policy original scope, identity or seals disagree")
    root = resolve(facts["candidate"]["root"])
    _policy_payload(document["producer_source"], resolve)
    if document["producer_source"]["sha256"] != facts["producer"]["reader"]["sha256"]:
        raise ValueError("incomplete_original: original reader source bytes are not retained")
    execution = document["execution"]
    required = {"receipt", "output", "process", "counter"} | (set() if name == "production" else {"junit", "last_test"})
    if (set(execution) != required or execution["receipt"] != {"path": result["receipt"], "sha256": result["sha256"]}
            or resolve(reference["path"]) != resolve(result["receipt"]).with_name("execution-contract.json")
            or _policy_payload(execution["counter"], resolve) != b"1\n"):
        raise ValueError("incomplete_original: policy execution references disagree")
    if recipe == _POLICY_RECIPE_V1:
        expected_reader_source = resolve(execution["receipt"]["path"]).with_name("reader-source.py")
    else:
        expected_reader_source = resolve(facts["candidate"]["root"]) / "out/policy-reader-sources" / (
            facts["producer"]["reader"]["sha256"] + ".py")
    if resolve(document["producer_source"]["path"]) != expected_reader_source:
        raise ValueError("incomplete_original: reader-source path is outside its fixed policy owner")
    for value in execution.values():
        _policy_payload(value, resolve)
    raw = _policy_raw(root, execution["receipt"], resolve)
    original_root = facts["candidate"]["root"]
    native_path = PureWindowsPath if PureWindowsPath(original_root).is_absolute() else Path
    producer = facts["producer"]
    producer_keys = {"external", "reader", "modules", "supervisor", "native_supervisor", "cache_policy", "cache_absent"}
    if (set(producer) != producer_keys or set(producer["external"]) != {"root", "files"}
            or set(producer["modules"]) != set(_POLICY_MODULE_SOURCES)
            or producer["cache_policy"] != {"cache_tag": "cpython-313", "optimization": 0, "pycache_prefix": None}):
        raise ValueError("incomplete_original: policy producer/module/cache schema differs")
    sources = {native_path(producer["external"]["root"]) / path for path in producer["external"]["files"]}
    sources.add(native_path(producer["reader"]["path"]))
    for module_name, binding in producer["modules"].items():
        relative = _POLICY_MODULE_SOURCES[module_name]
        expected = str(native_path(producer["external"]["root"]) / relative)
        if (set(binding) != {"path", "resolved", "sha256"} or binding["path"] != expected or binding["resolved"] != expected
                or binding["sha256"] != producer["external"]["files"][relative]):
            raise ValueError("incomplete_original: policy imported module origin differs")
    for role, filename in (("reader", "build_evidence.py"), ("native_supervisor", "process_tree.py")):
        binding = producer[role]
        expected = str(native_path(original_root) / "scripts" / filename)
        if (set(binding) != {"path", "resolved", "sha256"} or binding["path"] != expected or binding["resolved"] != expected
                or binding["sha256"] != facts["candidate"]["files"]["scripts/" + filename]):
            raise ValueError("incomplete_original: policy actual reader/supervisor origin differs")
    expected_supervisor = str(native_path(producer["external"]["root"]) / "scripts/process_tree.py")
    if (producer["supervisor"] != {"path": expected_supervisor, "resolved": expected_supervisor,
            "sha256": producer["external"]["files"]["scripts/process_tree.py"]}
            or producer["native_supervisor"]["sha256"] != producer["supervisor"]["sha256"]):
        raise ValueError("incomplete_original: policy ordinary/native supervisor source differs")
    expected_caches = {str(path.parent / "__pycache__" / (path.stem + ".cpython-313.pyc")) for path in sources}
    if set(producer["cache_absent"]) != expected_caches or any(set(row) != {"resolved"} or row["resolved"] != path
            for path, row in producer["cache_absent"].items()):
        raise ValueError("incomplete_original: policy project source cache absence is incomplete")
    if (raw["stage"] != name or raw["started_at"] != document["started_at"] or raw["finished_at"] != document["finished_at"]
            or raw["identity"]["inputs"] != result["identity"]["inputs"]
            or raw["identity"]["tools"] != {k: v for k, v in result["identity"]["tools"].items() if v is not None}
            or any(raw["identity"].get(k) != v for k, v in document["launch"].items())
            or execution["output"]["path"] != str(native_path(original_root) / raw["output"])
            or execution["process"]["path"] != raw["process_tree"]["receipt"]
            or execution["counter"]["path"] != facts["counter"]
            or (raw["identity"]["command"] if name == "production" else [ctest_identity(raw["identity"]["command"])])
                != result["identity"]["command"]):
        raise ValueError("incomplete_original: policy raw command correspondence changed")
    if name != "production":
        validate_ctest_stage(stage_receipts(result, resolve), {"host": ["Policy." + name]}, ["Policy." + name],
                             result["identity"]["command"], resolve)
    prepared = facts["preparation"]["stage"]
    prepared_raw = _policy_raw(root, {"path": prepared["receipt"], "sha256": prepared["sha256"]}, resolve,
                               preparation=prepared)
    historical = {k: v for k, v in prepared_raw["scoped_preparation"].items()
                  if k not in {"producer", "invalidated", "rust_invalidated_packages"}}
    if (historical != facts["subject"]["preparation_facts"]
            or _preparation_facts_identity(historical, prepared) != facts["preparation"]["identity"]
            or prepared.get("status") != "passed" or prepared.get("additional_receipts")
            or prepared.get("payloads") != {path: row["sha256"] for path, row in historical["output_identities"].items()}
            or set(prepared.get("payloads", {})) != set(historical["outputs"])):
        raise ValueError("preparation_changed: policy historical preparation correspondence changed")
    dependencies = facts["subject"]["dependencies"]
    paths = native_path(original_root)
    if (set(dependencies) != {"protocol", "root", "files", "directories", "absent", "python"}
            or dependencies["protocol"] != "declared-dependencies-v1" or dependencies["root"] != original_root
            or dependencies["python"] is not None
            or set(dependencies["directories"]) != {str(paths / n) for n in _policy_namespaces()}):
        raise ValueError("incomplete_original: policy dependency protocol/root/namespace schema differs")
    for dirname, expected_members in _policy_namespaces().items():
        directory_path = paths / dirname
        row = dependencies["directories"][str(directory_path)]
        if (set(row) != {"scope", "resolved", "members"} or row["scope"] != "immediate"
                or row["resolved"] != str(directory_path) or len(row["members"]) != len(expected_members)
                or {child["name"] for child in row["members"]} != set(expected_members)
                or any(set(child) != {"name", "kind", "resolved"}
                       or child["kind"] not in expected_members[child["name"]]
                       or child["resolved"] != str(directory_path / child["name"]) for child in row["members"])):
            raise ValueError("incomplete_original: historical policy namespace membership/targets differ")
    absent = {str(paths / "out/build" / context["preset"] / n) for n in ("CTestCustom.cmake", "CTestCustom.ctest")}
    absent.update(path for path in result["identity"]["tool_paths"] if path not in dependencies["files"]
                  and path != facts["producer"]["reader"]["path"])
    if set(dependencies["absent"]) != absent or any(set(row) != {"resolved"} or not native_path(row["resolved"]).is_absolute()
                                                  for row in dependencies["absent"].values()):
        raise ValueError("incomplete_original: historical policy absence membership/targets differ")
    for path, row in dependencies["files"].items():
        fields = {"kind", "resolved", "sha256", "executable"}
        if row.get("executable") and context["native_target"] == "linux-x86_64":
            fields.add("execute_bits")
        if (set(row) != fields or row["kind"] != "file" or type(row["executable"]) is not bool
                or not native_path(path).is_absolute() or not native_path(row["resolved"]).is_absolute()
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                or ("execute_bits" in row and (type(row["execute_bits"]) is not int or not row["execute_bits"] or row["execute_bits"] & ~0o111))):
            raise ValueError("incomplete_original: malformed historical policy file/executable facts")
    if set(facts["subject"]) != {"dependencies", "preparation_facts", "build_subject", "registrations", "registration", "off", "runtime"}:
        raise ValueError("incomplete_original: policy subject schema differs")
    required_files = {str(paths / name) for name in facts["candidate"]["files"]}
    if set(facts["candidate"]["files"]) != set(_policy_namespaces()["."]) - {".git", "out", "scripts", "src", "tests"} | {
            "src/evidence_child.c", "scripts/qualification.py", "scripts/check-quality.py", "scripts/build_evidence.py",
            "scripts/build_dependencies.py", "scripts/execution.py", "scripts/process_tree.py", "tests/quality/test_miniature.py"}:
        raise ValueError("incomplete_original: policy source membership differs")
    required_files.discard(str(paths / "scripts/build_evidence.py"))
    if name == "production":
        retained = _policy_off(facts["subject"]["off"], resolve)
        required_files.update(row["path"] for row in retained["files"].values())
        _, external = _policy_off_paths(historical)
        required_files.update(external)
    else:
        configuration = historical["context"]["configuration"]
        if facts["subject"]["registrations"] != configuration["registrations"] or len(configuration["registrations"]) != 4:
            raise ValueError("incomplete_original: original omitted a generated registration consumer")
        required_files.update(historical["context"]["files"])
        required_files.update(historical["outputs"])
        required_files.update(configuration["registrations"])
        required_files.update(str(paths / path) for path in configuration["inputs"])
        required_files.update(configuration["tools"])
        actual = facts["subject"]["build_subject"]
        reader = str(paths / "scripts/build_evidence.py")
        if (reader in actual["files"] or "scripts/build_evidence.py" in actual["configuration"]["inputs"]
                or "scripts/build_evidence.py" in actual["commands"]
                or str(paths / "src/evidence_child.c") not in actual["files"]):
            raise ValueError("incomplete_original: policy reader was an actual build consumer")
        registration = facts["subject"]["registration"]
        command = registration.get("command")
        property_rows = registration.get("properties")
        if (not isinstance(command, list) or len(command) != 3 or any(not isinstance(value, str) for value in command)
                or not isinstance(property_rows, list) or len(property_rows) != 3
                or any(not isinstance(row, dict) or set(row) != {"name", "value"} for row in property_rows)):
            raise ValueError("incomplete_original: historical policy registration/command differs")
        properties = {row["name"]: row["value"] for row in property_rows}
        native_build = paths / "out/build" / context["preset"]
        binary = native_build / ("BoundedEvidenceChild.exe" if context["native_target"] == "win-amd64" else "BoundedEvidenceChild")
        counter = native_build / ("counter-" + name + ".txt")
        if (registration["name"] != "Policy." + name or set(properties) != {"LABELS", "TIMEOUT", "WORKING_DIRECTORY"}
                or set(properties["LABELS"]) != {"host", name} or properties["TIMEOUT"] != 0
                or not isinstance(properties["WORKING_DIRECTORY"], str)
                or native_path(properties["WORKING_DIRECTORY"]) != native_build
                or [native_path(value) for value in command] != [binary, paths / "input.txt", counter]
                or native_path(facts["counter"]) != counter):
            raise ValueError("incomplete_original: historical policy registration/command differs")
    required_files.update(p for p in result["identity"]["tool_paths"] if p not in dependencies["absent"])
    required_files.discard(str(paths / "scripts/build_evidence.py"))
    if set(dependencies["files"]) != required_files:
        raise ValueError("incomplete_original: policy declared dependency membership is incomplete")
    for path in facts["candidate"]["files"]:
        if path != "scripts/build_evidence.py" and dependencies["files"][str(paths / path)]["sha256"] != facts["candidate"]["files"][path]:
            raise ValueError("incomplete_original: historical policy source bytes disagree")
    bindings = {}
    for path in result["identity"]["tool_paths"]:
        if path == str(paths / "scripts/build_evidence.py"):
            row = facts["producer"]["reader"]
        else:
            row = dependencies["files"].get(path, dependencies["absent"].get(path))
        if row is None:
            raise ValueError("incomplete_original: policy tool has no declared fact")
        bindings[row["resolved"]] = row.get("sha256")
    if bindings != result["identity"]["tools"]:
        raise ValueError("incomplete_original: original native tool bindings disagree")
    runtime = facts["subject"]["runtime"]
    if set(runtime) != {"python", "base_python"} or not set(runtime.values()) <= set(dependencies["files"]):
        raise ValueError("incomplete_original: original Python/base runtime roles are missing")
    runnable = {runtime["python"]}
    if name == "production":
        runnable.add(retained["binary"]["path"])
    else:
        runnable.add(str(binary))
        runnable.update(path for path in dependencies["files"] if native_path(path).name.casefold() in {"ctest", "ctest.exe"})
    if {path for path, row in dependencies["files"].items() if row["executable"]} != runnable:
        raise ValueError("incomplete_original: policy runnable role membership differs")
    if name == "production":
        retained = _policy_off(facts["subject"]["off"], resolve)
        expected_command = [retained["binary"]["path"], str(paths / "input.txt"), retained["counter"]]
        if (retained["preset"] != facts["context"]["preset"] or retained["preparation"] != facts["preparation"]
                or historical["source"] != original_root
                or historical["build"] != str(paths / "out/build" / facts["context"]["preset"])
                or facts["counter"] != retained["counter"] or raw["identity"]["command"] != expected_command):
            raise ValueError("preparation_changed: original OFF closure/command belongs to another context")
        expected_payloads = {row["path"]: row["sha256"] for row in retained["files"].values()}
        expected_payloads.update({facts["subject"]["off"]["path"]: facts["subject"]["off"]["sha256"],
                                 execution["counter"]["path"]: execution["counter"]["sha256"]})
    else:
        expected_payloads = {execution[key]["path"]: execution[key]["sha256"] for key in ("junit", "counter")}
    if result["payloads"] != expected_payloads:
        raise ValueError("incomplete_original: policy retained payload membership differs")
    for path, checksum in result.get("payloads", {}).items():
        _policy_payload({"path": path, "sha256": checksum}, resolve)
    document["protocol"] = "product-evidence-policy-v1"
    return {"reference": dict(reference), "contract": document}


def _policy_support_stage(root, stage, *, resolve=Path, producer=None):
    if stage.get("status") != "passed" or stage.get("reuse_admission") or stage.get("additional_receipts"):
        raise ValueError("evidence_missing: policy support must be independent whole-success current execution")
    launch = stage.get("execution_launch", {})
    expected_keys = {"environment_sha256", "supervisor_sha256", "stage_environment", "producer", "policy_producer"}
    if (set(launch) != expected_keys or launch["stage_environment"] != stage["identity"]["environment"]
            or launch["policy_producer"] != (producer if producer is not None else _policy_producer(root))
            or launch["producer"] != launch["policy_producer"]["external"]
            or launch["supervisor_sha256"] != launch["policy_producer"]["supervisor"]["sha256"]):
        raise ValueError("evidence_missing: policy support does not bind the actual loaded producer")
    raw = _policy_raw(root, {"path": stage["receipt"], "sha256": stage["sha256"]}, resolve)
    if (raw["identity"]["inputs"] != stage["identity"]["inputs"]
            or raw["identity"]["tools"] != {k: v for k, v in stage["identity"]["tools"].items() if v is not None}
            or any(raw["identity"].get(k) != launch[k] for k in ("environment_sha256", "supervisor_sha256"))):
        raise ValueError("evidence_missing: policy support launch differs from its completed raw execution")
    for name, checksum in stage.get("payloads", {}).items():
        _policy_payload({"path": name, "sha256": checksum}, resolve)
    return [raw]


def _policy_declaration(reference, *, resolve=Path):
    declaration = identity_json(_policy_payload(reference, resolve).decode("utf-8"))
    fields = {"protocol", "recipe", "original_candidate", "current_candidate", "changes", "originals", "compatibility",
              "producer_transition", "fresh_evidence", "chronology", "review", "owner_decision"}
    if (not isinstance(declaration, dict) or set(declaration) != fields
            or declaration["protocol"] != "tooling-evidence-policy-transition-v1" or declaration["recipe"] != _POLICY_RECIPE):
        raise ValueError("unsupported_protocol: malformed fixed policy transition")
    old, current = declaration["original_candidate"], declaration["current_candidate"]
    if (set(old) != {"root", "head", "tree", "files"} or set(current) != set(old)
            or set(old["files"]) != set(current["files"]) or old["root"] != current["root"]):
        raise ValueError("unreviewed_transition: policy candidate membership/source pair disagrees")
    originals = declaration["originals"]
    if (not isinstance(originals, list) or not originals
            or any(not isinstance(row, dict) or set(row) != {"preset", "group", "contract"}
                   or row["preset"] not in PRESETS or row["group"] not in _POLICY_GROUPS for row in originals)
            or len({(r["preset"], r["group"]) for r in originals}) != len(originals)):
        raise ValueError("unreviewed_transition: policy direct original selection is incomplete or duplicated")
    groups = sorted({row["group"] for row in originals})
    changes = [{"path": name, "old_sha256": old["files"][name], "new_sha256": current["files"][name], "groups": groups}
               for name in sorted(old["files"]) if old["files"][name] != current["files"][name]]
    if (len(changes) != 1 or changes[0]["path"] != "scripts/build_evidence.py" or declaration["changes"] != changes
            or declaration["compatibility"] != [{"path": "scripts/build_evidence.py", "group": group,
                    "role": "controller-only", "outcome_integrity": "preserved"} for group in groups]):
        raise ValueError("unreviewed_transition: policy source delta or historical trustworthiness disagrees")
    producers = declaration["producer_transition"]
    if (not isinstance(producers, dict) or set(producers) != {"original", "current"}
            or set(producers["original"]) != {"external", "reader", "modules", "supervisor", "native_supervisor", "cache_policy", "cache_absent"}
            or set(producers["current"]) != {"external", "reader", "modules", "supervisor", "native_supervisor", "cache_policy", "cache_absent"}
            or {k: v for k, v in producers["original"].items() if k != "reader"}
                != {k: v for k, v in producers["current"].items() if k != "reader"}
            or {k: v for k, v in producers["original"]["reader"].items() if k != "sha256"}
                != {k: v for k, v in producers["current"]["reader"].items() if k != "sha256"}
            or producers["original"]["reader"]["sha256"] != changes[0]["old_sha256"]
            or producers["current"]["reader"]["sha256"] != changes[0]["new_sha256"]):
        raise ValueError("unreviewed_transition: actual reader/producer transition differs")
    evidence = declaration["fresh_evidence"]
    if (not isinstance(evidence, dict) or set(evidence) != {"tooling", "quality", "quality_base"}
            or len(evidence["tooling"]) != 1 or len(evidence["quality"]) != 1
            or not re.fullmatch(r"[0-9a-f]{40}", evidence["quality_base"])):
        raise ValueError("evidence_missing: policy transition requires complete current tooling and quality")
    core = _bounded_hash({k: v for k, v in declaration.items() if k not in {"review", "owner_decision"}})
    for key, decision in (("review", "compatible"), ("owner_decision", "approve")):
        approval = identity_json(_policy_payload(declaration[key], resolve).decode("utf-8"))
        if approval != {"protocol": "synthetic-policy-" + key + "-v1", "transition_sha256": core,
                        "original_candidate_sha256": _bounded_hash(old), "current_candidate_sha256": _bounded_hash(current),
                        "originals": originals, "producer_transition": producers, "decision": decision}:
            raise ValueError("unreviewed_transition: stale or incorrectly scoped policy " + key)
    for ref in [*evidence["tooling"], *evidence["quality"]]:
        _policy_payload(ref, resolve)
    return declaration


def _read_policy_request(root, args, reference):
    declaration = _policy_declaration(reference)
    if args.runtime != "all" or set(args.selection) - (_POLICY_GROUPS | {"tooling"}):
        raise ValueError("unsupported_protocol: policy admission requires its exact all-runtime synthetic context")
    originals = {row["group"]: read_policy_contract(row["contract"]) for row in declaration["originals"]
                 if row["preset"] == args.preset}
    if (set(originals) != set(args.selection) - {"tooling"}
            or len(originals) != len(declaration["originals"]) or declaration["current_candidate"] != _policy_candidate(root)
            or declaration["producer_transition"]["current"] != _policy_producer(root)):
        raise ValueError("unreviewed_transition: policy selection/current source/producer differs")
    for original in originals.values():
        facts = original["contract"]["facts"]
        if (facts["candidate"] != declaration["original_candidate"]
                or facts["producer"] != declaration["producer_transition"]["original"]):
            raise ValueError("unreviewed_transition: policy direct original/source provenance differs")
    current = read_text(root / "scripts/build_evidence.py", encoding="utf-8")
    actual = read_text(Path(__file__).with_name("build_evidence.py"), encoding="utf-8")
    old = subprocess.check_output(["git", "show", declaration["original_candidate"]["head"] + ":scripts/build_evidence.py"], cwd=root)
    if current != actual or old.decode("utf-8") != _policy_reader_before(actual):
        raise ValueError("unreviewed_transition: actual repair/source pair is not the reviewed registration correction")
    return {"reference": reference, "declaration": declaration, "originals": originals, "policy_recipe": _POLICY_RECIPE}


def _policy_support(root, build, state, declaration, *, resolve=Path, native_maps=None,
                    collection=False, recorded_support=None):
    from datetime import datetime
    producer = declaration["producer_transition"]["current"]
    support_ref = declaration["fresh_evidence"]["tooling"][0]
    wrapper = identity_json(_policy_payload(support_ref, resolve).decode("utf-8"))
    require_checkpoint(wrapper)
    expected_source = declaration["current_candidate"]["files"] | {
        "rust/film-juicer-plugin/src/cuda/sys.rs": None, "src/GeneratedColorSpaces.cpp": None,
        "src/GeneratedColorSpaces.h": None}
    if (wrapper["preset"] != build.name or wrapper["selection"] != ["tooling"] or wrapper["runtime"] != "all"
            or wrapper.get("source") != expected_source or wrapper.get("reuse_declaration")
            or wrapper.get("membership") != {"tooling": ["Tooling.Miniature"]}
            or wrapper.get("runtime_membership") != {"tooling": {"host": ["Tooling.Miniature"]}}):
        raise ValueError("evidence_missing: policy tooling wrapper scope/source differs")
    tooling = {k: v for k, v in wrapper["stages"]["tooling"].items() if k != "reused_from"}
    if state.get("tooling") != tooling or state.get("quality", {}).get("status") != "passed":
        raise ValueError("owner_conflict: current policy supporting owners differ")
    tooling_raw = _policy_support_stage(root, tooling, resolve=resolve, producer=producer)[0]
    if tooling["identity"]["dependencies"]:
        raise ValueError("unreviewed_transition: circular policy tooling support")
    validate_ctest_stage(stage_receipts(tooling, resolve), {"host": ["Tooling.Miniature"]}, ["Tooling.Miniature"],
                         tooling["identity"]["command"], resolve)
    prepared, quality = state["prepare"], state["quality"]
    if collection and (not isinstance(recorded_support, dict) or recorded_support.get("tooling_stage") != tooling
            or recorded_support.get("quality_stage") != quality or recorded_support.get("preparation", {}).get("stage") != prepared):
        raise ValueError("terminal_changed: policy support differs from its strongly admitted closure")
    preparation = preparation_identity(build, prepared, resolve=resolve,
        native_file_identities=(native_maps or {}).get(prepared["sha256"]))
    if (quality["identity"]["dependencies"] != [preparation]
            or declaration["fresh_evidence"]["quality"] != [{"path": quality["receipt"], "sha256": quality["sha256"]}]):
        raise ValueError("preparation_changed: policy quality/current preparation differs")
    quality_raw = _policy_support_stage(root, quality, resolve=resolve, producer=producer)[0]
    expected_quality = [str(PureWindowsPath(quality_raw["identity"]["command"][0]))
        if PureWindowsPath(quality_raw["identity"]["command"][0]).is_absolute() else quality_raw["identity"]["command"][0],
        "-B", "scripts/check-quality.py", "--preset", build.name, "--qualification", "--base",
        declaration["fresh_evidence"]["quality_base"]]
    if quality["identity"]["command"] != expected_quality or quality_raw["identity"]["command"] != expected_quality:
        raise ValueError("evidence_missing: policy quality command/base differs")
    evidence = _read_synthetic_quality(root, build, quality, declaration["fresh_evidence"]["quality_base"],
        recipe=_POLICY_RECIPE, resolve=resolve, native_maps=native_maps, producer=producer,
        candidate=declaration["current_candidate"])
    if quality.get("payloads") != {ref["path"]: ref["sha256"] for ref in evidence["payloads"]}:
        raise ValueError("evidence_missing: policy quality has incomplete immutable stream closure")
    for group, stage, commands, dependencies in (
            ("tooling", tooling, tooling["identity"]["command"], []),
            ("product", quality, expected_quality, [preparation])):
        expected_inputs = snapshot(root, checkpoint_inputs(root, build, group,
            source_files=declaration["current_candidate"]["files"], resolve=resolve))
        identity = stage["identity"]
        if identity["inputs"] != expected_inputs or identity["dependencies"] != dependencies or identity["command"] != commands:
            raise ValueError("terminal_changed: current policy supporting identity differs")
        foreign = os.name == "posix" and PureWindowsPath(declaration["current_candidate"]["root"]).is_absolute()
        if not foreign:
            tools = stage["identity"]["tool_paths"] if collection else _policy_tools(build, dict(os.environ), group)
            if _stage_identity(root, commands, expected_inputs, tools, dict(os.environ), dependencies) != identity:
                raise ValueError("tools_changed: current policy support tool/environment identity differs")
        else:
            for path, checksum in identity["tools"].items():
                if checksum is None:
                    if resolve(path).exists() or resolve(path).is_symlink():
                        raise ValueError("tools_changed: absent supporting policy tool appeared")
                else:
                    rows = [maps["files"][path] for maps in (native_maps or {}).values() if path in maps["files"]]
                    if not rows:
                        raise ValueError("tools_changed: supporting policy tool lacks fresh native identity")
                    _policy_live_file(path, rows[0], resolve=resolve, native_maps=native_maps)
    prep_raw = _policy_raw(root, {"path": prepared["receipt"], "sha256": prepared["sha256"]}, resolve,
                          preparation=prepared)
    if not (datetime.fromisoformat(tooling_raw["finished_at"]) <= datetime.fromisoformat(prep_raw["started_at"])
            <= datetime.fromisoformat(prep_raw["finished_at"]) <= datetime.fromisoformat(quality_raw["started_at"])):
        raise ValueError("chronology_missing: current policy tooling/preparation/quality ordering differs")
    support = {"tooling": support_ref, "tooling_stage": tooling, "preparation": {"identity": preparation, "stage": prepared},
               "quality_stage": quality, "quality": declaration["fresh_evidence"]["quality"][0]}
    if collection and support != recorded_support:
        raise ValueError("terminal_changed: current policy support identity/reference differs")
    return support


def _policy_current_identity(original, candidate, preparation):
    facts = original["contract"]["facts"]
    identity = identity_json(_bounded_bytes(facts["stage_identity"]).decode("utf-8"))
    old_reader = facts["producer"]["reader"]
    new_hash = candidate["files"]["scripts/build_evidence.py"]
    if "scripts/build_evidence.py" in identity["inputs"]:
        identity["inputs"]["scripts/build_evidence.py"] = new_hash
    if old_reader["resolved"] in identity["tools"]:
        identity["tools"][old_reader["resolved"]] = new_hash
    if facts["context"]["group"] != "production":
        identity["dependencies"] = [preparation["identity"]]
    return identity


def _policy_compare(original, current, declaration):
    document = original["contract"]
    old = document["facts"]
    if (old["candidate"] != declaration["original_candidate"] or current["candidate"] != declaration["current_candidate"]
            or old["producer"] != declaration["producer_transition"]["original"]
            or current["producer"] != declaration["producer_transition"]["current"]
            or old["policy"] != current["policy"] or old["context"] != current["context"]):
        raise ValueError("subject_changed: policy/context/source/producer correspondence differs")
    expected = identity_json(_bounded_bytes(old["subject"]).decode("utf-8"))
    reader = old["producer"]["reader"]
    row = (expected["preparation_facts"]["context"]["files"].get(reader["path"])
           if old["context"]["group"] != "production" else None)
    if row is not None:
        if row["sha256"] != reader["sha256"]:
            raise ValueError("preparation_changed: original reader observation differs")
        row["sha256"] = current["producer"]["reader"]["sha256"]
    if old["context"]["group"] != "production":
        config = expected["preparation_facts"]["context"]["configuration"]
        if "scripts/build_evidence.py" in config["inputs"]:
            if config["inputs"]["scripts/build_evidence.py"] != reader["sha256"]:
                raise ValueError("preparation_changed: original broad reader configuration observation differs")
            config["inputs"]["scripts/build_evidence.py"] = current["producer"]["reader"]["sha256"]
    if expected != current["subject"]:
        raise ValueError("subject_changed: policy declared dependency/output/registration facts differ")
    expected_identity = _policy_current_identity(original, current["candidate"], current["preparation"])
    if current["stage_identity"] != expected_identity:
        raise ValueError("tools_changed: policy stage command/tool/environment/dependency differs")
    return {"protocol": "synthetic-policy-comparison-v1", "original": original["reference"],
            "current_candidate_sha256": _bounded_hash(current["candidate"]),
            "original_preparation_identity": old["preparation"]["identity"],
            "current_preparation_identity": current["preparation"]["identity"],
            "declaration_sha256": _bounded_hash(declaration), "review_authenticated": False}


def _policy_admit(campaign, state_path, state, name, reuse, support):
    original = reuse["originals"][name]
    expected = identity_json(_bounded_bytes({**original["contract"]["stage"]["result"],
                                             "execution_contract": original["reference"]}).decode("utf-8"))
    previous = state.get(name, {})
    OutputLease.check_held()
    _check_stage_owner(state_path, state, name, previous)
    if {k: v for k, v in previous.items() if k != "reuse_admission"} != expected:
        raise ValueError("owner_conflict: policy latest owner is not the direct original")
    snapshot_state = identity_json(read_text(state_path))
    if snapshot_state != state:
        raise ValueError("owner_conflict: supplied policy state is not the current complete owner state")
    def observe():
        OutputLease.check_held()
        if identity_json(read_text(state_path)) != snapshot_state:
            raise ValueError("owner_conflict: policy latest owner changed")
        declaration = _policy_declaration(reuse["reference"])
        if read_policy_contract(original["reference"]) != original:
            raise ValueError("terminal_changed: policy direct original changed")
        current_support = _policy_support(campaign.root, state_path.parent, state, declaration)
        if current_support != support:
            raise ValueError("terminal_changed: policy current support changed")
        old = original["contract"]["facts"]
        command = old["stage_identity"]["command"] if name == "production" else list(old["stage_identity"]["command"][0])
        if name != "production":
            command[-1] = original["contract"]["execution"]["junit"]["path"]
        if name == "production":
            current = identity_json(_bounded_bytes(old).decode("utf-8"))
            current["candidate"] = _policy_candidate(campaign.root)
            current["producer"] = _policy_producer(campaign.root)
            current["stage_identity"] = _policy_current_identity(original, current["candidate"], current["preparation"])
            _policy_off(old["subject"]["off"])
            from build_evidence import capture_bounded_dependencies
            dependencies = old["subject"]["dependencies"]
            observed = capture_bounded_dependencies(campaign.root,
                files=[Path(p) for p in dependencies["files"]],
                absent=[Path(p) for p in dependencies["absent"]],
                directories=[Path(p) for p in dependencies["directories"]],
                executables=[Path(p) for p, row in dependencies["files"].items() if row["executable"]])
            if observed != dependencies:
                raise ValueError("subject_changed: retained OFF subject/tool/absence/namespace changed")
            current["subject"]["dependencies"] = observed
        else:
            inputs = checkpoint_inputs(campaign.root, state_path.parent, name)
            tools = _policy_tools(state_path.parent, dict(os.environ), name)
            current = _policy_capture(campaign, state, name, command, inputs, tools, dict(os.environ))
        comparison = _policy_compare(original, current, declaration)
        return {"declaration": reuse["reference"], "original": original["reference"], "current": current,
                "comparison": comparison, "support": support, "owner": expected,
                "chronology": declaration["chronology"]}
    facts = observe()
    owner_campaign = str(campaign.directory.resolve(strict=True))
    prior_reference = previous.get("reuse_admission")
    if prior_reference:
        prior_facts, prior_meta = _policy_read_admission(prior_reference)
        prior_path = (mapped_windows_path(prior_reference["path"])
                      if os.name == "posix" and PureWindowsPath(prior_reference["path"]).is_absolute()
                      else Path(prior_reference["path"]))
        if prior_path.resolve(strict=True).parent.parent != state_path.parent / "verification-admissions":
            raise ValueError("owner_conflict: policy prior admission is outside its fixed build owner")
        if prior_facts != facts:
            raise ValueError("owner_conflict: policy prior admission facts changed")
        if prior_meta["protocol"] != "synthetic-policy-admission-v2":
            expected_path = state_path.parent / "verification-admissions" / _bounded_hash(facts) / "admission.json"
            completion = prior_path.with_name("completion.json")
            if prior_path != expected_path or not (completion.is_file() and not completion.is_symlink()):
                raise ValueError("owner_conflict: incomplete legacy policy admission cannot be safely reused")
            _policy_completion(prior_reference, state_path.parent)
            if observe() != facts or identity_json(read_text(state_path)) != snapshot_state:
                raise ValueError("terminal_changed: legacy policy admission facts changed during reuse")
            _check_stage_owner(state_path, state, name, previous)
            OutputLease.check_held()
            return previous
        owner_campaign = prior_meta["owner_campaign"]
        expected_shared = {"protocol": "synthetic-policy-facts-v1", "owner_campaign": owner_campaign,
                           "candidate": facts["current"]["candidate"], "producer": facts["current"]["producer"],
                           "support": facts["support"]}
        if prior_meta["shared_record"] != expected_shared:
            raise ValueError("owner_conflict: policy prior shared-facts payload changed")
        completion = prior_path.with_name("completion.json")
        if completion.exists() or completion.is_symlink():
            _policy_completion(prior_reference, state_path.parent)
        prior_owner = (mapped_windows_path(owner_campaign)
                       if os.name == "posix" and PureWindowsPath(owner_campaign).is_absolute()
                       else Path(owner_campaign))
        if not (completion.exists() or completion.is_symlink()) and prior_owner.resolve(strict=True) != campaign.directory.resolve(strict=True):
            raise ValueError("owner_conflict: incomplete policy admission belongs to another campaign")
        shared_reference = prior_meta["shared_facts"]
    else:
        shared_reference = None
    from datetime import datetime, timezone
    record, path = _policy_encode_admission(campaign, facts, datetime.now(timezone.utc).isoformat(),
        admission_root=state_path.parent, owner_campaign=owner_campaign, shared_reference=shared_reference)
    if path.exists():
        retained_facts, retained_meta = _policy_read_admission(_bounded_reference(path))
        if retained_facts != facts or retained_meta["owner_campaign"] != owner_campaign:
            raise ValueError("owner_conflict: policy immutable admission conflicts")
        record = identity_json(read_text(path))
    else:
        _check_stage_owner(state_path, state, name, previous)
        OutputLease.check_held()
        _write_admission_record(path, record)
    if observe() != facts:
        raise ValueError("terminal_changed: policy admission facts changed")
    reference = _bounded_reference(path)
    committed_facts, committed_meta = _policy_read_admission(reference)
    if committed_facts != facts or committed_meta["owner_campaign"] != owner_campaign:
        raise ValueError("terminal_changed: policy committed admission changed")
    _check_stage_owner(state_path, state, name, previous)
    OutputLease.check_held()
    state[name] = {**expected, "reuse_admission": reference}
    write_json(state_path, state)
    return state[name]


def _policy_validate_completion_header(admission, metadata, completed):
    legacy = metadata["protocol"] == "synthetic-policy-admission-v1"
    expected_fields = {"protocol", "admission", "wrapper"} if legacy else {
        "protocol", "admission", "wrapper", "shared_facts"}
    expected_protocol = "synthetic-policy-completion-v1" if legacy else "synthetic-policy-completion-v2"
    if (set(completed) != expected_fields or completed["protocol"] != expected_protocol
            or completed["admission"] != admission
            or (not legacy and completed["shared_facts"] != metadata["shared_facts"])):
        raise ValueError("owner_conflict: malformed or conflicting policy completion")
    if not legacy:
        logical_path = PureWindowsPath if PureWindowsPath(completed["wrapper"]["path"]).is_absolute() else Path
        if logical_path(completed["wrapper"]["path"]).parent != logical_path(metadata["owner_campaign"]):
            raise ValueError("owner_conflict: policy completion wrapper differs from shared-facts campaign owner")
    return legacy


def _policy_validate_completion_admission_owner(admission, metadata, record, declaration, retained):
    if metadata["protocol"] != "synthetic-policy-admission-v2":
        return
    matches = [row for row in declaration["originals"] if row.get("contract") == record.get("original")]
    if len(matches) != 1:
        raise ValueError("owner_conflict: policy completion admission has no unique original group")
    stage = retained.get("stages", {}).get(matches[0]["group"], {})
    if stage.get("reuse_admission") != admission:
        raise ValueError("owner_conflict: policy completion creator wrapper does not retain this exact admission")


def _policy_completion(admission, build, *, resolve=Path):
    record, metadata = _policy_read_admission(admission, resolve=resolve)
    admission_path = resolve(admission["path"])
    path = admission_path.with_name("completion.json")
    reference = {"path": str(path), "sha256": digest(path)}
    completed = identity_json(_policy_payload(reference, resolve).decode("utf-8"))
    _policy_validate_completion_header(admission, metadata, completed)
    retained = identity_json(_policy_payload(completed["wrapper"], resolve).decode("utf-8"))
    require_checkpoint(retained)
    declaration = _policy_declaration(record["declaration"], resolve=resolve)
    groups = {row["group"] for row in declaration["originals"]}
    stages = retained["stages"]
    _policy_validate_completion_admission_owner(admission, metadata, record, declaration, retained)
    expected_source = declaration["current_candidate"]["files"] | {
        "rust/film-juicer-plugin/src/cuda/sys.rs": None, "src/GeneratedColorSpaces.cpp": None,
        "src/GeneratedColorSpaces.h": None}
    if (retained["status"] != "passed" or retained.get("pending")
            or retained.get("preset") != build.name or retained.get("runtime") != "all"
            or retained.get("source") != expected_source or retained.get("assurance") != _POLICY_ASSURANCE
            or retained.get("reuse_declaration") != record["declaration"]
            or set(retained["selection"]) - {"tooling"} != groups
            or set(retained.get("stage_provenance", {})) != set(stages)):
        raise ValueError("owner_conflict: policy completion wrapper source/context/stages differ")
    for row in declaration["originals"]:
        original = read_policy_contract(row["contract"], resolve=resolve)
        actual = stages.get(row["group"], {})
        if ({k: v for k, v in actual.items() if k not in {"reuse_admission", "reused_from"}}
                != {**original["contract"]["stage"]["result"], "execution_contract": row["contract"]}
                or not actual.get("reuse_admission") or retained["stage_provenance"].get(row["group"]) != "admitted"):
            raise ValueError("owner_conflict: policy completion stage does not retain its direct original")
        admitted, _ = _policy_read_admission(actual["reuse_admission"], resolve=resolve)
        if admitted.get("declaration") != record["declaration"] or admitted.get("original") != row["contract"]:
            raise ValueError("owner_conflict: policy completion selects a different admission")
    if "product" in groups:
        if stages.get("prepare") != record["support"]["preparation"]["stage"] or stages.get("quality") != record["support"]["quality_stage"]:
            raise ValueError("owner_conflict: policy completion supporting preparation/quality differs")
    if "tooling" in retained["selection"] and stages.get("tooling") != record["support"]["tooling_stage"]:
        raise ValueError("owner_conflict: policy completion tooling differs")
    memberships = {group: ["Tooling.Miniature" if group == "tooling" else "Policy." + group]
                   for group in retained["selection"] if group != "production"}
    if (retained.get("membership") != memberships
            or retained.get("runtime_membership") != {group: {"host": names} for group, names in memberships.items()}):
        raise ValueError("owner_conflict: policy completion membership differs")
    wrapper_directory = resolve(completed["wrapper"]["path"]).parent
    for name, stage in stages.items():
        provenance = ("admitted" if stage.get("reuse_admission") else "exact-reused"
            if stage.get("reused_from") or not resolve(stage["receipt"]).is_relative_to(wrapper_directory / "attempts")
            else "executed")
        if retained["stage_provenance"][name] != provenance:
            raise ValueError("owner_conflict: policy completion provenance differs from actual retained execution")
    return completed, reference


def _qualify_policy_owned(root, build, args, reuse):
    from verify import source_snapshot
    state_path = build / "verification-checkpoint.json"
    state = identity_json(read_text(state_path))
    if any(stage.get("status") != "passed" for stage in state.values()):
        raise ValueError("owner_conflict: pending/failed/blocked policy state prevents admission")
    selected = sorted(set(args.selection))
    if "production" in selected and selected != ["production"]:
        raise ValueError("unsupported_protocol: policy OFF is a separate outer context")
    required = sorted(set(selected) | ({"prepare", "quality"} if "product" in selected else set()))
    campaign = Campaign(root, build.name, args.campaign or "policy-admission-" + uuid.uuid4().hex, required)
    path = campaign.directory / "qualification.json"
    if path.exists():
        raise ValueError("owner_conflict: policy wrapper attempt already exists")
    record = {"mode": "checkpoint", "status": "running", "scope_status": "running", "reason": args.reason,
              "preset": build.name, "runtime": "all", "selection": selected, "required": required,
              "source": source_snapshot(root), "stages": {}, "reuse_declaration": reuse["reference"],
              "assurance": _POLICY_ASSURANCE, "stage_provenance": {}, "membership": {}, "runtime_membership": {}}
    try:
        support = _policy_support(root, build, state, reuse["declaration"])
        for name in required:
            if name in {"prepare", "quality", "tooling"}:
                result = state[name]
                record["stage_provenance"][name] = ("executed" if Path(result["receipt"]).is_relative_to(
                    campaign.directory / "attempts") and not result.get("reused_from") else "exact-reused")
            else:
                result = _policy_admit(campaign, state_path, state, name, reuse, support)
                record["stage_provenance"][name] = "admitted"
            record["stages"][name] = result
            if name in {"product", "boundary", "corpus", "tooling"}:
                members = ["Tooling.Miniature" if name == "tooling" else "Policy." + name]
                record["membership"][name] = members
                record["runtime_membership"][name] = {"host": members}
        if (_policy_support(root, build, state, reuse["declaration"]) != support
                or source_snapshot(root) != record["source"] or identity_json(read_text(state_path)) != state):
            raise ValueError("terminal_changed: policy wrapper source/support/owners changed")
        OutputLease.check_held()
        record.update(status="passed", scope_status="passed", pending=[])
        for name in selected:
            admission = record["stages"][name].get("reuse_admission")
            if admission:
                completion = Path(admission["path"]).with_name("completion.json")
                if completion.exists() or completion.is_symlink():
                    _policy_completion(admission, build)
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        write_json(path, record)
    for name in selected:
        admission = record["stages"][name].get("reuse_admission")
        if admission:
            completion = Path(admission["path"]).with_name("completion.json")
            if completion.exists():
                _policy_completion(admission, build)
            else:
                _, admission_meta = _policy_read_admission(admission)
                if admission_meta["protocol"] != "synthetic-policy-admission-v2":
                    raise ValueError("owner_conflict: incomplete legacy policy admission cannot be completed by the v2 writer")
                completed = {"protocol": "synthetic-policy-completion-v2", "admission": admission,
                             "wrapper": _bounded_reference(path), "shared_facts": admission_meta["shared_facts"]}
                _write_admission_record(completion, completed)
    return path


def policy_collection_stage(root, build, wrapper, name, result, state, *, resolve=Path, native_maps=None,
                            wrapper_path=None):
    reference = result.get("execution_contract")
    if not reference:
        return None
    data = identity_json(_policy_payload(reference, resolve).decode("utf-8"))
    if data.get("protocol") not in {"product-evidence-policy-v1", "product-evidence-policy-v2"}:
        return None
    original = read_policy_contract(reference, resolve=resolve)
    facts = original["contract"]["facts"]
    reader_source = _policy_payload(original["contract"]["producer_source"], resolve).decode("utf-8")
    public_reader = str((PureWindowsPath if PureWindowsPath(facts["producer"]["external"]["root"]).is_absolute() else Path)(
        facts["producer"]["external"]["root"]) / "scripts/build_evidence.py")
    public_source = _policy_payload({"path": public_reader,
        "sha256": facts["producer"]["external"]["files"]["scripts/build_evidence.py"]}, resolve).decode("utf-8")
    if reader_source not in {public_source, _policy_reader_before(public_source)}:
        raise ValueError("unreviewed_transition: actual original reader source is not a fixed policy source form")
    if (facts["context"]["preset"] != build.name or facts["context"]["group"] != name
            or wrapper["runtime"] != "all" or name not in wrapper["selection"]
            or {k: v for k, v in result.items() if k not in {"reuse_admission", "execution_contract", "reused_from"}}
                != original["contract"]["stage"]["result"]):
        raise ValueError("incomplete_original: policy wrapper/result context differs from actual original")
    references = [reference, original["contract"]["producer_source"], *original["contract"]["execution"].values()]
    policy_recipe = facts["policy"]["recipe"]
    info = {"recipe": policy_recipe, "assurance": _POLICY_ASSURANCE, "context": facts["context"],
            "current_identity": result["identity"], "original": reference, "provenance": "executed",
            "off": facts["subject"]["off"], "references": references, "chronology": wrapper.get("policy_chronology"),
            "finished_at": original["contract"]["finished_at"],
            "execution_candidate": facts["candidate"], "current_candidate": facts["candidate"],
            "original_preparation": facts["preparation"]}
    admission = result.get("reuse_admission")
    if admission:
        admitted, admission_meta = _policy_read_admission(admission, resolve=resolve)
        if admitted["original"] != reference:
            raise ValueError("owner_conflict: malformed policy admission")
        if admission_meta["protocol"] == "synthetic-policy-admission-v1":
            stable = admitted
            expected_path = build / "verification-admissions" / _bounded_hash(stable) / "admission.json"
            if resolve(admission["path"]) != expected_path:
                raise ValueError("owner_conflict: legacy policy admission path/identity differs")
        if admission_meta["protocol"] == "synthetic-policy-admission-v2":
            if wrapper_path is None:
                raise ValueError("owner_conflict: policy admission lacks its fixed build/current wrapper owner")
            retained_wrapper = identity_json(read_bytes(resolve(wrapper_path)).decode("utf-8"))
            if retained_wrapper != wrapper:
                raise ValueError("owner_conflict: current policy wrapper path/content differs")
        declaration = _policy_declaration(admitted["declaration"], resolve=resolve)
        if (wrapper.get("reuse_declaration") != admitted["declaration"] or wrapper.get("assurance") != _POLICY_ASSURANCE
                or wrapper.get("stage_provenance", {}).get(name) != "admitted"
                or admitted["current"]["candidate"] != declaration["current_candidate"]
                or admitted["owner"] != {**original["contract"]["stage"]["result"], "execution_contract": reference}
                or state.get(name) != {**admitted["owner"], "reuse_admission": admission}
                or admitted["comparison"] != _policy_compare(original, admitted["current"], declaration)
                or admitted["support"] != _policy_support(root, build, state, declaration, resolve=resolve, native_maps=native_maps,
                                                          collection=True, recorded_support=admitted["support"])):
            raise ValueError("terminal_changed: policy admission/current support/original relation differs")
        completion, completion_ref = _policy_completion(admission, build, resolve=resolve)
        completion_path = resolve(completion_ref["path"])
        references += [admission, admitted["declaration"], declaration["review"], declaration["owner_decision"],
                       *declaration["fresh_evidence"]["tooling"], *declaration["fresh_evidence"]["quality"], completion["wrapper"]]
        if admission_meta["shared_facts"] is not None:
            references.append(admission_meta["shared_facts"])
        references.append(completion_ref)
        info.update(current_identity=admitted["current"]["stage_identity"], admission=admission, provenance="admitted",
                    chronology=admitted["chronology"], completion_sha256=digest(completion_path),
                    producer_transition=declaration["producer_transition"], current_candidate=admitted["current"]["candidate"])
        for stage in (admitted["support"]["tooling_stage"], admitted["support"]["preparation"]["stage"],
                      admitted["support"]["quality_stage"]):
            references += _policy_stage_references(root, stage, resolve)
        reader = admitted["current"]["producer"]["reader"]
        if (reader_source != _policy_reader_before(public_source)
                or _policy_payload({"path": reader["path"], "sha256": reader["sha256"]}, resolve).decode("utf-8") != public_source):
            raise ValueError("unreviewed_transition: actual original/current reader repair differs")
        _policy_live_file(reader["path"], reader, resolve=resolve, native_maps=native_maps)
        references.append({"path": reader["path"], "sha256": reader["sha256"]})
    elif wrapper.get("source") != facts["candidate"]["files"] | {
            "rust/film-juicer-plugin/src/cuda/sys.rs": None, "src/GeneratedColorSpaces.cpp": None, "src/GeneratedColorSpaces.h": None}:
        raise ValueError("subject_changed: ordinary policy wrapper does not bind its original current candidate")
    if facts["context"]["group"] == "production":
        info["off_record"] = _policy_off(facts["subject"]["off"], resolve)
        references.append(facts["subject"]["off"])
        references += list(info["off_record"]["files"].values())
        original_build = (PureWindowsPath if PureWindowsPath(facts["candidate"]["root"]).is_absolute() else Path)(
            facts["candidate"]["root"]) / "out/build" / build.name
        registry = str(original_build / "verification-config-outputs.json")
        if registry in info["off_record"]["files"]:
            references += _policy_configuration(root, info["off_record"]["files"][registry],
                                                preparation=info["off_record"]["preparation"]["stage"], resolve=resolve)
        chronology = info["chronology"]
        if chronology is not None:
            try:
                binding = chronology["on"][build.name]
                if set(binding) != {"preparation", "started"}:
                    raise ValueError("chronology_missing: ON restoration must bind its original preparation and actual start")
                restoration_stage = binding["preparation"]
                restoration_ref = {"path": restoration_stage["receipt"], "sha256": restoration_stage["sha256"]}
                restoration = _policy_raw(root, restoration_ref, resolve, preparation=restoration_stage)
                started = _policy_raw(root, binding["started"], resolve, preparation=restoration_stage)
                proof = restoration["scoped_preparation"]
                off_commands = facts["subject"]["preparation_facts"]["context"]["commands"]
                expected_on_commands = off_commands.replace("POLICY_TESTING_ON=0", "POLICY_TESTING_ON=1")
                if (proof["configure_args"] != ["-DBUILD_TESTING=ON"] or proof["targets"] != ["all"]
                        or resolve(proof["build"]) != build or proof["source"] != facts["candidate"]["root"]
                        or off_commands.count("POLICY_TESTING_ON=0") != 1 or proof["context"]["commands"] != expected_on_commands
                        or any(proof["context"]["configuration"]["tools"].get(name) != checksum
                               for name, checksum in facts["subject"]["preparation_facts"]["context"]["configuration"]["tools"].items())
                        or restoration_stage["identity"]["environment"] != facts["preparation"]["stage"]["identity"]["environment"]
                        or resolve(binding["started"]["path"]).parent.parent != resolve(restoration_ref["path"]).parent.parent
                        or started["identity"]["command"][:3] != ["cmake", "--preset", build.name]
                        or "-DBUILD_TESTING=ON" not in started["identity"]["command"]
                        or started["identity"]["executable_sha256"] != restoration["identity"]["executable_sha256"]
                        or any(started["identity"]["inputs"].get(name) != checksum
                               or restoration_stage["identity"]["inputs"].get(name) != checksum
                               for name, checksum in facts["candidate"]["files"].items())):
                    raise ValueError("chronology_missing: ON restoration belongs to another context")
                from datetime import datetime
                if not datetime.fromisoformat(started["started_at"]) <= datetime.fromisoformat(started["finished_at"]) <= datetime.fromisoformat(restoration["finished_at"]):
                    raise ValueError("chronology_missing: ON restoration start/finish ordering differs")
                info["on_started_at"] = started["started_at"]
                references += _policy_stage_references(root, restoration_stage, resolve)
                references += _policy_stage_references(root, {"receipt": binding["started"]["path"],
                    "sha256": binding["started"]["sha256"], "payloads": {}}, resolve, preparation=restoration_stage)
            except (KeyError, TypeError) as error:
                raise ValueError("chronology_missing: incomplete original ON restoration references") from error
    subject = admitted["current"]["subject"] if admission else facts["subject"]
    producer = admitted["current"]["producer"] if admission else facts["producer"]
    info["producer"] = producer
    observed_candidate = _policy_candidate(root)
    # Mapping a Windows checkout for reading does not change its native source
    # identity. Compare the actual Git facts while retaining that original root.
    observed_candidate["root"] = info["current_candidate"]["root"]
    if observed_candidate != info["current_candidate"]:
        raise ValueError("subject_changed: current policy Git head/tree/source differs")
    native_path = PureWindowsPath if PureWindowsPath(producer["external"]["root"]).is_absolute() else Path
    for path, checksum in producer["external"]["files"].items():
        reference = {"path": str(native_path(producer["external"]["root"]) / path), "sha256": checksum}
        _policy_payload(reference, resolve)
        references.append(reference)
    for path, binding in producer["cache_absent"].items():
        local = resolve(path)
        if local.exists() or local.is_symlink() or local.resolve(strict=False) != resolve(binding["resolved"]):
            raise ValueError("terminal_changed: actual policy producer project cache appeared or retargeted")
    for path, binding in subject["dependencies"]["files"].items():
        _policy_live_file(path, binding, resolve=resolve, native_maps=native_maps)
    for path in subject["dependencies"]["absent"]:
        expected_target = resolve(subject["dependencies"]["absent"][path]["resolved"])
        if resolve(path).exists() or resolve(path).is_symlink() or resolve(path).resolve(strict=False) != expected_target:
            raise ValueError("subject_changed: absent policy dependency appeared")
    for path, directory in subject["dependencies"]["directories"].items():
        expected = {row["name"] for row in directory["members"]}
        local = resolve(path)
        if (local.resolve(strict=True) != resolve(directory["resolved"])
                or {p.name for p in local.iterdir()} != expected
                or any((local / row["name"]).resolve(strict=True) != resolve(row["resolved"])
                       or ("file" if (local / row["name"]).is_file() else "directory") != row["kind"]
                       for row in directory["members"])):
            raise ValueError("subject_changed: policy source/import namespace membership changed")
    existing = {(reference["path"], reference["sha256"]) for reference in references}
    references.extend(reference for reference in _policy_command_map_references(references, resolve)
                      if (reference["path"], reference["sha256"]) not in existing)
    return info


def _policy_stage_references(root, stage, resolve, *, preparation=None):
    reference = {"path": stage["receipt"], "sha256": stage["sha256"]}
    raw = _policy_raw(root, reference, resolve, preparation=stage if preparation is None else preparation)
    process = resolve(raw["process_tree"]["receipt"])
    return [reference, {"path": str(root / raw["output"]), "sha256": raw["output_sha256"]},
            {"path": str(process), "sha256": digest(process)},
            *[{"path": path, "sha256": checksum} for path, checksum in stage.get("payloads", {}).items()],
            *[{"path": path, "sha256": checksum}
              for path, checksum in command_result_payloads(reference["path"], resolve=resolve).items()]]


def _policy_command_map_references(references, resolve):
    blobs = {}
    for reference in references:
        logical = PureWindowsPath(reference["path"]) if PureWindowsPath(reference["path"]).is_absolute() else Path(reference["path"])
        if logical.name == "result.json" and logical.parent.parent.name == "attempts":
            blobs.update(command_result_payloads(reference["path"], resolve=resolve))
    return [{"path": path, "sha256": checksum} for path, checksum in sorted(blobs.items())]


def _policy_live_file(path, binding, *, resolve=Path, native_maps=None):
    local = resolve(path)
    if not local.is_file() or digest(local) != binding["sha256"]:
        raise ValueError("terminal_changed: policy dependency/tool contents changed: " + path)
    foreign = os.name == "posix" and PureWindowsPath(path).is_absolute()
    expected = {"resolved": binding["resolved"], "sha256": binding["sha256"]}
    if foreign:
        rows = [groups["files"][path] for groups in (native_maps or {}).values() if path in groups["files"]]
        if not rows or any(row != expected for row in rows):
            raise ValueError("terminal_changed: policy dependency lacks matching fresh native identity: " + path)
    elif str(local.resolve(strict=True)) != binding["resolved"]:
        raise ValueError("terminal_changed: policy dependency/tool target changed: " + path)
    if binding.get("execute_bits") is not None and local.stat().st_mode & 0o111 != binding["execute_bits"]:
        raise ValueError("terminal_changed: policy executable permission changed")
    if binding.get("execute_mask") is not None and local.stat().st_mode & 0o111 != binding["execute_mask"]:
        raise ValueError("terminal_changed: policy quality helper permission changed")
    if binding.get("executable") or "launch_access" in binding:
        if not foreign:
            if os.name == "posix":
                if os.access not in os.supports_effective_ids or not os.access(local, os.X_OK, effective_ids=True):
                    raise ValueError("terminal_changed: policy executable lacks effective-id launch access")
            elif not os.access(local, os.X_OK):
                raise ValueError("terminal_changed: policy executable lacks native launch access")


def _policy_native_partitions(files, runtime):
    """Preserve each Windows spelling without putting aliases in one read."""
    if not runtime or any(files.get(path) != binding for path, binding in runtime.items()):
        raise ValueError("evidence_missing: policy native partitions lack their fixed runtime bindings")
    runtime_names = {PureWindowsPath(path): path for path in runtime}
    if len(runtime_names) != len(runtime):
        raise ValueError("unsupported_protocol: policy native runtime paths have conflicting aliases")
    aliases = {}
    for path, binding in sorted(files.items()):
        if (not isinstance(path, str) or not PureWindowsPath(path).is_absolute()
                or not isinstance(binding, dict) or set(binding) != {"resolved", "sha256"}
                or not isinstance(binding["resolved"], str) or not PureWindowsPath(binding["resolved"]).is_absolute()
                or not isinstance(binding["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", binding["sha256"])):
            raise ValueError("evidence_missing: malformed policy native partition binding")
        key = PureWindowsPath(path)
        names = aliases.setdefault(key, [])
        if names and files[names[0]] != binding:
            raise ValueError("terminal_changed: equivalent policy native paths have conflicting bindings")
        if key in runtime_names and path != runtime_names[key]:
            raise ValueError("unsupported_protocol: policy native runtime paths have conflicting aliases")
        names.append(path)
    partitions = [{path: dict(runtime[path]) for path in sorted(runtime)}
                  for _ in range(max(map(len, aliases.values())))]
    for key, names in aliases.items():
        if key not in runtime_names:
            for index, path in enumerate(names):
                partitions[index][path] = dict(files[path])
    return partitions


def policy_native_collection_bindings(root, records, resolve):
    """Additional native reads keep preparation proof membership unchanged."""
    expected = {}
    for wrapper in records:
        state = None
        for stage in wrapper["stages"].values():
            reference = stage.get("execution_contract")
            if not reference:
                continue
            document = identity_json(_policy_payload(reference, resolve).decode("utf-8"))
            if document.get("protocol") not in {"product-evidence-policy-v1", "product-evidence-policy-v2"}:
                continue
            _check_bounded_cli_environment(dict(os.environ))
            facts = read_policy_contract(reference, resolve=resolve)["contract"]["facts"]
            if not PureWindowsPath(facts["candidate"]["root"]).is_absolute():
                continue
            if state is None:
                state_path = root / "out/build" / wrapper["preset"] / "verification-checkpoint.json"
                state = identity_json(read_text(state_path, encoding="utf-8"))
            selected = facts
            admitted = None
            if stage.get("reuse_admission"):
                admitted, _ = _policy_read_admission(stage["reuse_admission"], resolve=resolve)
                selected = admitted["current"]
            prepared = state.get("prepare", facts["preparation"]["stage"])
            proof = identity_json(_policy_payload({"path": prepared["receipt"], "sha256": prepared["sha256"]}, resolve).decode("utf-8"))["scoped_preparation"]
            files = dict(proof["context"]["files"])
            def add(path, binding):
                identity = {"resolved": binding["resolved"], "sha256": binding["sha256"]}
                if path in files and files[path] != identity:
                    raise ValueError("terminal_changed: policy native identity contributions disagree")
                files[path] = identity
            for path, binding in selected["subject"]["dependencies"]["files"].items():
                add(path, binding)
            reader = selected["producer"]["reader"]
            add(reader["path"], reader)
            for binding in [*selected["producer"]["modules"].values(), selected["producer"]["supervisor"], selected["producer"]["native_supervisor"]]:
                add(binding["path"], binding)
            if admitted:
                quality = admitted["support"]["quality_stage"]
                raw = _policy_raw(root, {"path": quality["receipt"], "sha256": quality["sha256"]}, resolve)
                reports = [line.removeprefix("Synthetic quality report: ") for line in read_text(root / raw["output"]).splitlines()
                           if line.startswith("Synthetic quality report: ")]
                if len(reports) != 1:
                    raise ValueError("evidence_missing: policy native quality report is incomplete")
                report_ref = {"path": reports[0], "sha256": quality["payloads"].get(reports[0])}
                report = identity_json(_policy_payload(report_ref, resolve).decode("utf-8"))
                for path, binding in report["inputs"].items():
                    add(path, binding)
                for binding in report["tools"].values():
                    add(binding["path"], binding)
            for binding in list(files.values()):
                add(binding["resolved"], binding)
            runtime = {}
            for path in selected["subject"]["runtime"].values():
                if path not in files:
                    raise ValueError("evidence_missing: policy native collection lacks its fixed runtime binding")
                runtime[path] = files[path]
            for partition in _policy_native_partitions(files, runtime):
                groups = {"files": partition, "rust_configuration": dict(proof["context"]["rust_configuration"])}
                key = _bounded_hash({"policy_original": reference, "producer": selected["producer"], "observations": groups})
                expected[key] = groups
    return expected


def policy_collection_chronology(infos):
    from datetime import datetime
    selected = [info for info in infos if info.get("recipe") in _POLICY_READ_RECIPES]
    off = [info for info in selected if info["context"]["group"] == "production"]
    if not off:
        return
    if {info["context"]["preset"] for info in off} != {"linux-release", "windows-clang-release"}:
        raise ValueError("chronology_missing: both synthetic OFF contexts are required")
    chronologies = [info.get("chronology") for info in off]
    if any(value is None or value != chronologies[0] for value in chronologies):
        raise ValueError("chronology_missing: original OFF/ON barrier references are missing or disagree")
    chronology = chronologies[0]
    if (not isinstance(chronology, dict) or set(chronology) != {"off", "on"}
            or set(chronology["off"]) != {"linux-release", "windows-clang-release"}
            or set(chronology["on"]) != {"linux-release", "windows-clang-release"}):
        raise ValueError("chronology_missing: exact synthetic OFF/ON chronology contexts are required")
    ends, starts = [], []
    for info in off:
        preset = info["context"]["preset"]
        if chronology["off"][preset] != info["original"]:
            raise ValueError("chronology_missing: original OFF chronology selects another execution")
        products = [stage for stage in selected if stage["context"]["preset"] == preset and stage["context"]["group"] == "product"]
        if (len(products) != 1 or products[0]["execution_candidate"] != info["execution_candidate"]
                or products[0]["context"]["native_target"] != info["context"]["native_target"]
                or chronology["on"][preset]["preparation"] != products[0]["original_preparation"]["stage"]):
            raise ValueError("chronology_missing: ON restoration does not bind the same original subject/native preparation")
        # Collection already resolves and validates these immutable originals.
        # Keep timestamps as explicit sealed evidence in the policy info.
        ends.append(datetime.fromisoformat(info["finished_at"]))
        starts.append(datetime.fromisoformat(info["on_started_at"]))
    if max(ends) > min(starts):
        raise ValueError("chronology_missing: both original OFF completions must precede either ON restoration")


def _bounded_source_names(recipe):
    names = {".gitignore", "CMakeLists.txt", "CMakePresets.json", "src/evidence_child.c", "input.txt",
             "scripts/qualification.py"}
    if recipe == "native-c11-cli-miniature-v1":
        names.update({"scripts/check-quality.py", "tests/quality/test_miniature.py"})
    elif recipe != "native-c11-miniature-v1":
        raise ValueError("unsupported bounded original capture recipe")
    return names


_BOUNDED_CLI_PYTHON_OVERRIDES = (
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONUSERBASE",
    "PYTHONPYCACHEPREFIX", "PYTHONPLATLIBDIR", "PYTHONEXECUTABLE", "PYTHONWARNINGS"
)


def _bounded_cli_namespaces():
    return {
        ".": {".git": ["directory", "file"], ".gitignore": ["file"], "CMakeLists.txt": ["file"],
              "CMakePresets.json": ["file"], "input.txt": ["file"], "out": ["directory"],
              "scripts": ["directory"], "src": ["directory"], "tests": ["directory"]},
        "scripts": {"check-quality.py": ["file"], "qualification.py": ["file"]},
        "src": {"evidence_child.c": ["file"]},
        "tests": {"quality": ["directory"]},
        "tests/quality": {"test_miniature.py": ["file"]}
    }


def _check_bounded_cli_namespaces(root, directories):
    expected = _bounded_cli_namespaces()
    if set(directories) != {str(root / name) for name in expected}:
        raise ValueError("bounded CLI source namespace membership changed")
    for name, members in expected.items():
        path = root / name
        row = directories[str(path)]
        if (set(row) != {"scope", "resolved", "members"} or row["scope"] != "immediate"
                or Path(row["resolved"]) != path or not isinstance(row["members"], list)):
            raise ValueError("bounded CLI source namespace target changed: " + str(path))
        observed = row["members"]
        if (any(not isinstance(child, dict) or set(child) != {"name", "kind", "resolved"}
                or not isinstance(child["name"], str) for child in observed)
                or len(observed) != len(members) or {child["name"] for child in observed} != set(members)):
            raise ValueError("bounded CLI source namespace membership changed: " + str(path))
        for child in observed:
            if child["kind"] not in members[child["name"]] or Path(child["resolved"]) != path / child["name"]:
                raise ValueError("bounded CLI source namespace kind or target changed: " + str(path / child["name"]))


def _check_bounded_cli_environment(environment):
    for name in _BOUNDED_CLI_PYTHON_OVERRIDES:
        if environment.get(name):
            raise ValueError("unsupported bounded CLI Python environment override: " + name)


def _bounded_policy(templates=None, *, recipe="native-c11-miniature-v1"):
    _bounded_source_names(recipe)
    if templates is None:
        directory = Path(__file__).resolve().parents[1] / "tests/quality/evidence"
        locations = {name: directory / name for name in ("CMakeLists.txt", "src/evidence_child.c")}
        if recipe == "native-c11-cli-miniature-v1":
            locations.update({"CMakeLists.txt": directory / "cli/CMakeLists.txt",
                              "src/evidence_child.c": directory / "cli/evidence_child.c",
                              "scripts/check-quality.py": directory / "cli/check_quality.py",
                              "tests/quality/test_miniature.py": directory / "cli/test_miniature.py"})
        templates = {name: digest(path) for name, path in locations.items()}
    policy = {"profile": "declared-dependencies-v1", "recipe": recipe,
            "templates": templates,
            "controllers": ["scripts/qualification.py"],
            "environment_premise": ["qualified-os-driver-hardware", "ordinary-credentials-access-policy",
                                    "installed-system-libraries", "qualified-tool-runtime-distributions"],
            "scope": "declared project dependencies; unobserved external environment changes may escape detection"}
    if recipe == "native-c11-cli-miniature-v1":
        policy["source_namespaces"] = _bounded_cli_namespaces()
        policy["unsupported_python_overrides"] = list(_BOUNDED_CLI_PYTHON_OVERRIDES)
    return policy


def _bounded_producer():
    root = Path(__file__).resolve().parents[1]
    names = ("scripts/qualification.py", "scripts/build_evidence.py", "scripts/build_dependencies.py",
             "scripts/execution.py", "scripts/process_tree.py", "scripts/verify.py", "scripts/check-quality.py",
             "tests/ffi/grain_probe_contract.py", "tests/ffi/effects_frame_probe_contract.py")
    return {"root": str(root), "files": {name: digest(root / name) for name in names}}


def _bounded_launch(environment):
    environment = command_environment(environment)
    if "JUICER_OUTPUT_LEASES" in os.environ:
        environment["JUICER_OUTPUT_LEASES"] = os.environ["JUICER_OUTPUT_LEASES"]
    return {"environment_sha256": hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
            "supervisor_sha256": digest(Path(__file__).with_name("process_tree.py"))}


def _bounded_candidate(root, *, original=False, recipe="native-c11-miniature-v1"):
    from verify import source_snapshot
    if recipe == "native-c11-cli-miniature-v1":
        from build_evidence import capture_bounded_dependencies
        namespaces = capture_bounded_dependencies(root, files=[],
            directories=[root / name for name in _bounded_cli_namespaces()])
        _check_bounded_cli_namespaces(root, namespaces["directories"])
    files = source_snapshot(root)
    required = _bounded_source_names(recipe)
    names = set(subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                        cwd=root).decode().split("\0")) - {""}
    # source_snapshot also includes three product-only generated paths. They
    # are not members of this miniature's explicit Git source recipe.
    files = {name: files.get(name) for name in names}
    if names != required or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                                     for value in files.values()):
        raise ValueError("bounded miniature has unsupported source membership or unresolved state")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip()
    if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (head, tree)):
        raise ValueError("bounded miniature has no reconstructible source revision")
    if original:
        for name, checksum in files.items():
            raw = subprocess.check_output(["git", "show", head + ":" + name], cwd=root)
            if hashlib.sha256(raw).hexdigest() != checksum:
                raise ValueError("bounded miniature original must match its retained Git revision")
    return {"root": str(root), "head": head, "tree": tree, "files": files}


def _bounded_presets(root, preset, compiler, *, recipe="native-c11-miniature-v1"):
    _bounded_source_names(recipe)
    build = root / "out/build" / preset
    variables = {"CMAKE_C_COMPILER": str(compiler), "CMAKE_BUILD_TYPE": "Debug", "BUILD_TESTING": "ON",
                 "CMAKE_EXPORT_COMPILE_COMMANDS": "ON", "CMAKE_C_FLAGS": ""}
    if recipe == "native-c11-cli-miniature-v1":
        variables["Python3_EXECUTABLE"] = sys.executable
    if preset == "windows-clang-debug":
        variables.update(CMAKE_LINKER_TYPE="LLD", CMAKE_C_FLAGS="/Brepro", CMAKE_C_FLAGS_DEBUG="/Od /Z7",
                         CMAKE_EXE_LINKER_FLAGS="/MANIFEST:NO /INCREMENTAL:NO /Brepro",
                         CMAKE_EXE_LINKER_FLAGS_DEBUG="/DEBUG /INCREMENTAL:NO /Brepro",
                         CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreadedDebug")
        if recipe == "native-c11-cli-miniature-v1":
            ranlib = Path(compiler).with_name("llvm-ranlib.exe")
            if not ranlib.is_file():
                raise ValueError("bounded CLI Windows recipe requires its qualified compiler-sibling llvm-ranlib")
            variables["CMAKE_RANLIB"] = ranlib.as_posix()
    return {"version": 6, "configurePresets": [
        {"name": preset, "generator": "Ninja", "binaryDir": str(build), "cacheVariables": variables}],
        "testPresets": [{"name": preset, "configurePreset": preset,
                         "execution": {"jobs": 4, "noTestsAction": "error"}}]}


def _bounded_cli_tools(build, environment):
    binary = build / ("BoundedEvidenceChild.exe" if os.name == "nt" else "BoundedEvidenceChild")
    ctest = shutil.which("ctest", path=environment.get("PATH"))
    if ctest is None:
        raise ValueError("bounded CTest tool is unavailable")
    tools = set(checkpoint_tools(build, environment, "product")) | {str(binary), sys._base_executable}
    target = Path(ctest).resolve(strict=True)
    if target not in {Path(name) for name in tools}:
        tools.add(str(target))
    return sorted(tools)


def _bounded_files(root, candidate, proof, tools, build, *, recipe="native-c11-miniature-v1"):
    files = {root / name for name in candidate["files"] if name != "scripts/qualification.py"}
    files.update(Path(name) for name in proof["context"]["files"])
    files.update(root / name for name in proof["context"]["configuration"]["inputs"])
    files.update(Path(name) for name in proof["context"]["configuration"]["tools"])
    files.update(Path(name) for name in proof["outputs"])
    files.add(build / "CTestTestfile.cmake")
    files.update(Path(name) for name in tools)
    if recipe == "native-c11-cli-miniature-v1":
        files.discard(root / "scripts/qualification.py")
    return files


def _bounded_build_subject(root, build, environment):
    from build_dependencies import NinjaGraph
    from build_evidence import scoped_context

    graph = NinjaGraph(build, ["all"], environment)
    context = scoped_context(root, root, build, graph, [], environment)
    controller = root / "scripts/qualification.py"
    consumed = {*context["files"], *context["configuration"]["tools"]}
    consumed.update(str(root / name) for name in context["configuration"]["inputs"])
    if (str(controller) in consumed or str(controller.resolve()) in consumed
            or "qualification.py" in context["commands"].casefold()):
        raise ValueError("bounded controller is an actual build consumer")
    return context


def capture_execution_subject(campaign, state, command, *, inputs, tools, environment,
                              recipe="native-c11-miniature-v1"):
    """Read current facts for an explicit audited miniature; never execute or admit a stage."""
    from build_evidence import capture_bounded_dependencies
    import sysconfig

    root = campaign.root
    preset = campaign.directory.parent.name
    native = "windows-clang-debug" if os.name == "nt" else "linux-debug"
    if preset != native:
        raise ValueError("bounded capture supports only its native Debug miniature")
    build = root / "out/build" / preset
    lease = OutputLease(root, build)
    if lease.path not in OutputLease.held:
        raise ValueError("bounded facts require the current ordinary build owner")
    OutputLease.check_held()
    if recipe == "native-c11-cli-miniature-v1":
        _check_bounded_cli_environment(environment)
    policy = _bounded_policy(recipe=recipe)
    for name, checksum in policy["templates"].items():
        if digest(root / name) != checksum:
            raise ValueError("unsupported bounded miniature source recipe")
    candidate = _bounded_candidate(root, recipe=recipe)
    compiler = shutil.which("clang-cl.exe", path=environment.get("PATH")) if os.name == "nt" else "/usr/bin/gcc-13"
    if (compiler is None or Path(compiler).resolve().is_relative_to(root)
            or identity_json(read_text(root / "CMakePresets.json")) != _bounded_presets(root, preset, compiler, recipe=recipe)):
        raise ValueError("unsupported bounded compiler/configuration injection")
    injection = {"LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
                 "CPATH", "C_INCLUDE_PATH", "GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH", "CL", "_CL_",
                 "CMAKE_TOOLCHAIN_FILE", "CMAKE_PROJECT_TOP_LEVEL_INCLUDES"}
    if any(environment.get(name) for name in injection):
        raise ValueError("unsupported bounded project compiler/runtime injection")
    for name in ("PATH", "INCLUDE", "LIB"):
        for value in environment.get(name, "").split(os.pathsep):
            if value and Path(value).resolve().is_relative_to(root):
                raise ValueError("bounded environment searches the mutable miniature project")
    prepared = state.get("prepare")
    if not isinstance(prepared, dict) or prepared.get("status") != "passed":
        raise ValueError("bounded original requires completed current preparation")
    prepared_identity = preparation_identity(build, prepared)
    preparation_record = check_scoped_preparation(campaign, build, Path(prepared["receipt"]), environment)
    proof = preparation_record["scoped_preparation"]
    if (proof["targets"] != ["all"] or proof["configure_args"] != ["-DBUILD_TESTING=ON"]
            or (recipe == "native-c11-miniature-v1"
                and set(proof["inputs"]) != {"CMakeLists.txt", "CMakePresets.json", "src/evidence_child.c"})):
        raise ValueError("bounded miniature needs its explicit actual build input map")
    build_subject = None
    if recipe == "native-c11-cli-miniature-v1":
        required_preparation = [name for name in checkpoint_inputs(root, build, "prepare",
            source_files=candidate["files"], preparation=proof["context"]) if not name.startswith("out/build/")]
        if sorted(proof["inputs"]) != required_preparation:
            raise ValueError("bounded CLI miniature needs the ordinary complete preparation input map")
        build_subject = _bounded_build_subject(root, build, environment)
        if (any(proof["context"].get(key) != value for key, value in build_subject.items() if key != "files")
                or any(proof["context"]["files"].get(name) != row for name, row in build_subject["files"].items())):
            raise ValueError("bounded actual build subject disagrees with current preparation")
    cache = read_text(build / "CMakeCache.txt", encoding="utf-8")
    if not re.search(r"^CMAKE_BUILD_TYPE:STRING=Debug$", cache, re.M):
        raise ValueError("bounded miniature requires Debug preparation")
    binary = build / ("BoundedEvidenceChild.exe" if os.name == "nt" else "BoundedEvidenceChild")
    counter = build / "assertion-count.txt"
    if str(binary) not in proof["outputs"]:
        raise ValueError("bounded child has no current preparation output owner")
    command = list(command)
    if not command or "--output-junit" not in command or command.count("--output-junit") != 1:
        raise ValueError("bounded original needs its exact CTest command")
    junit = Path(command[-1])
    if not junit.is_absolute() or not junit.is_relative_to(root / "out/validation" / preset):
        raise ValueError("bounded JUnit destination is outside its owned campaign")
    expected = [*ctest_command(root, build), "-R", "^(Bounded\\.Read)$", "-LE", "gpu", "--no-tests=error",
                "--output-on-failure", "--output-junit", str(junit)]
    if command != expected:
        raise ValueError("unsupported bounded CTest command form")
    inventory = json.loads(subprocess.check_output(["ctest", "--test-dir", str(build), "--show-only=json-v1"],
                                                  cwd=root, env=environment, text=True))
    required_members = {"product": ["Bounded.Read"]}
    if recipe == "native-c11-cli-miniature-v1":
        required_members["tooling"] = ["Tooling.Miniature"]
    if (membership(inventory, set(required_members), "all") != required_members
            or len(inventory["tests"]) != len(required_members)):
        raise ValueError("bounded miniature registration membership changed")
    test = next(test for test in inventory["tests"] if test["name"] == "Bounded.Read")
    properties = {row["name"]: row["value"] for row in test.get("properties", [])}
    if ([Path(value) for value in test["command"]] != [binary, root / "input.txt", counter]
            or set(properties.get("LABELS", [])) != {"host", "product"}
            or set(properties) != {"LABELS", "TIMEOUT", "WORKING_DIRECTORY"} or properties["TIMEOUT"] != 0
            or Path(properties["WORKING_DIRECTORY"]) != build
            or registration_files(build) != [(build / "CTestTestfile.cmake").resolve()]):
        raise ValueError("bounded miniature registration does not bind its native child")
    if recipe == "native-c11-cli-miniature-v1":
        tooling = next(test for test in inventory["tests"] if test["name"] == "Tooling.Miniature")
        tooling_properties = {row["name"]: row["value"] for row in tooling.get("properties", [])}
        if (len(tooling["command"]) != 3 or Path(tooling["command"][0]) != Path(sys.executable)
                or tooling["command"][1] != "-B" or Path(tooling["command"][2]) != root / "tests/quality/test_miniature.py"
                or set(tooling_properties) != {"LABELS", "TIMEOUT", "WORKING_DIRECTORY"}
                or tooling_properties["LABELS"] != ["host", "tooling"] or tooling_properties["TIMEOUT"] != 0
                or Path(tooling_properties["WORKING_DIRECTORY"]) != build):
            raise ValueError("bounded CLI tooling registration changed")
    required_inputs = (checkpoint_inputs(root, build, "product", source_files=candidate["files"])
                       if recipe == "native-c11-cli-miniature-v1" else sorted(candidate["files"]))
    if sorted(inputs) != required_inputs:
        raise ValueError("bounded stage needs its complete declared source input map")
    if recipe == "native-c11-cli-miniature-v1" and sorted(tools) != _bounded_cli_tools(build, environment):
        raise ValueError("bounded CLI stage needs the ordinary complete tool map and native child")
    selected_ctest = shutil.which("ctest", path=environment.get("PATH"))
    if selected_ctest is None:
        raise ValueError("bounded CTest tool is unavailable")
    ctest = Path(selected_ctest).resolve(strict=True)
    required_tools = {binary, ctest, Path(sys.executable), Path(sys._base_executable)}
    if not (required_tools <= {Path(name) for name in tools} if recipe == "native-c11-cli-miniature-v1"
            else {str(binary), str(ctest), sys.executable, sys._base_executable} <= set(tools)):
        raise ValueError("bounded stage must bind the CTest and child executables")
    files = _bounded_files(root, candidate, proof, tools, build, recipe=recipe)
    if root / "scripts/qualification.py" in files:
        raise ValueError("bounded controller is an actual preparation consumer")
    absent = {build / "CTestCustom.cmake", build / "CTestCustom.ctest"}
    if recipe == "native-c11-cli-miniature-v1":
        missing_tools = {Path(name) for name in tools if not Path(name).exists()}
        absent.update(missing_tools)
        files.difference_update(missing_tools)
    executables = {binary, ctest}
    if recipe == "native-c11-cli-miniature-v1":
        executables.add(Path(sys.executable))
        # Keep the declaration's lexical spelling on case-insensitive native
        # paths; the resolved target is a separate required dependency fact.
        executables = {next(path for path in files if path == executable) for executable in executables}
    directories = ([root / name for name in _bounded_cli_namespaces()]
                   if recipe == "native-c11-cli-miniature-v1" else [root])
    dependencies = capture_bounded_dependencies(root, files=sorted(files), directories=directories,
        absent=sorted(absent), executables=sorted(executables))
    if recipe == "native-c11-cli-miniature-v1":
        _check_bounded_cli_namespaces(root, dependencies["directories"])
    actual_build = {key: value for key, value in proof.items()
                    if key not in {"producer", "invalidated", "rust_invalidated_packages"}}
    subject = {"dependencies": dependencies, "preparation_facts": actual_build,
               "registration": {"name": test["name"], "command": test["command"],
                                "properties": test.get("properties", [])}}
    if build_subject is not None:
        subject["build_subject"] = build_subject
        subject["tooling_registration"] = {"name": tooling["name"], "command": tooling["command"],
                                            "properties": tooling.get("properties", [])}
    return {"policy": policy, "candidate": candidate,
            "context": {"preset": preset, "native_target": sysconfig.get_platform(), "profile": "Debug",
                        "testing": "ON", "features": [], "runtime": "host", "group": "product"},
            "controller_paths": policy["controllers"],
            "subject": subject,
            "preparation": {"identity": prepared_identity, "stage": prepared},
            "stage_identity": _stage_identity(root, [ctest_identity(command)], inputs, tools, environment,
                                               [prepared_identity])}


def _bounded_entry(campaign, state, name, command, identity, inputs, tools, environment,
                   recipe="native-c11-miniature-v1"):
    if recipe == _POLICY_RECIPE:
        return _policy_entry(campaign, state, name, command, identity, inputs, tools, environment)
    if name != "product":
        raise ValueError("real product/boundary/corpus/OFF bounded profiles are not enabled")
    facts = capture_execution_subject(campaign, state, command, inputs=inputs, tools=tools, environment=environment,
                                      recipe=recipe)
    if facts["stage_identity"] != identity:
        raise ValueError("bounded original stage identity does not match its recipe")
    facts["candidate"] = _bounded_candidate(campaign.root, original=True, recipe=recipe)
    counter = campaign.root / "out/build" / facts["context"]["preset"] / "assertion-count.txt"
    if counter.exists() or counter.is_symlink():
        raise ValueError("bounded original requires an unexecuted counter")
    facts = identity_json(_bounded_bytes(facts).decode("utf-8"))
    return (facts, {str(campaign.root / record["output"]): record["attempt_number"]
                    for record in campaign.attempts(name)}, _bounded_producer(), _bounded_launch(environment))


def _bounded_finish(campaign, state, command, result, entry, inputs, tools, environment):
    if isinstance(entry, dict) and "policy_facts" in entry:
        return _policy_finish(campaign, state, entry["policy_facts"]["context"]["group"], command, result,
                              entry, inputs, tools, environment)
    from verify import check_receipt_payloads
    facts, prior, producer, launch = entry
    paths = stage_receipts(result)
    if len(paths) != 1 or str(paths[0].with_name("output.log")) in prior:
        raise ValueError("bounded original needs one fresh whole command, not partial/replayed evidence")
    path = paths[0]
    if not path.is_relative_to(campaign.directory / "attempts"):
        raise ValueError("bounded original command belongs to another campaign")
    raw = check_receipt_payloads(campaign.root, path)
    expected_tools = result["identity"]["tools"]
    if facts["policy"]["recipe"] == "native-c11-cli-miniature-v1":
        expected_tools = {name: checksum for name, checksum in expected_tools.items() if checksum is not None}
    if (raw["stage"] != "product" or raw["identity"]["command"] != list(command)
            or raw["identity"]["inputs"] != result["identity"]["inputs"]
            or raw["identity"]["tools"] != expected_tools
            or any(raw["identity"].get(key) != value for key, value in launch.items())):
        raise ValueError("bounded original command/source provenance disagrees")
    junit = Path(command[-1])
    validate_ctest_stage(path, {"host": ["Bounded.Read"]}, ["Bounded.Read"], [ctest_identity(command)])
    counter = campaign.root / "out/build" / facts["context"]["preset"] / "assertion-count.txt"
    if read_bytes(counter) != b"1\n":
        raise ValueError("bounded original did not assert exactly once")
    terminal = capture_execution_subject(campaign, state, command, inputs=inputs, tools=tools, environment=environment,
                                         recipe=facts["policy"]["recipe"])
    if terminal != facts or producer != _bounded_producer() or launch != _bounded_launch(environment):
        raise ValueError("bounded subject changed at terminal capture")
    execution = {"receipt": _bounded_reference(path), "output": _bounded_reference(campaign.root / raw["output"]),
                 "process": _bounded_reference(raw["process_tree"]["receipt"]), "junit": _bounded_reference(junit),
                 "last_test": _bounded_reference(path.with_name("LastTest.log")), "counter": _bounded_reference(counter),
                 "started_at": raw["started_at"], "finished_at": raw["finished_at"]}
    document = {"protocol": "product-evidence-bounded-v1", "assurance": "declared-dependencies-v1",
                **{key: facts[key] for key in ("policy", "candidate", "context", "controller_paths", "subject", "preparation")},
                "stage": {"name": "product", "result": result}, "execution": execution,
                "producer": producer, "launch": launch,
                "seals": {"entry": _bounded_hash({"facts": facts, "producer": producer, "launch": launch}),
                          "terminal": _bounded_hash({"facts": terminal, "producer": producer, "launch": launch})}}
    sidecar = path.with_name("execution-contract.json")
    OutputLease.check_held()
    with sidecar.open("xb") as stream:
        stream.write(_bounded_bytes(document))
    return _bounded_reference(sidecar)


def _check_bounded_contract(document):
    fields = {"protocol", "assurance", "policy", "candidate", "context", "controller_paths", "subject",
              "preparation", "stage", "execution", "seals", "producer", "launch"}
    templates = document.get("policy", {}).get("templates", {}) if isinstance(document, dict) and isinstance(document.get("policy"), dict) else {}
    recipe = document.get("policy", {}).get("recipe") if isinstance(document, dict) and isinstance(document.get("policy"), dict) else None
    source_names = _bounded_source_names(recipe)
    template_names = {"CMakeLists.txt", "src/evidence_child.c"}
    subject_names = {"dependencies", "preparation_facts", "registration"}
    if recipe == "native-c11-cli-miniature-v1":
        template_names.update({"scripts/check-quality.py", "tests/quality/test_miniature.py"})
        subject_names.update({"build_subject", "tooling_registration"})
    if (not isinstance(document, dict) or set(document) != fields
            or document["protocol"] != "product-evidence-bounded-v1"
            or document["assurance"] != "declared-dependencies-v1"
            or not isinstance(templates, dict) or set(templates) != template_names
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in templates.values())
            or document["policy"] != _bounded_policy(templates, recipe=recipe)
            or document["controller_paths"] != ["scripts/qualification.py"]):
        raise ValueError("unsupported or malformed bounded original contract")
    try:
        result = document["stage"]["result"]
        candidate = document["candidate"]
        dependencies = document["subject"]["dependencies"]
        if (set(document["stage"]) != {"name", "result"} or document["stage"]["name"] != "product"
                or set(result) != {"status", "identity", "receipt", "sha256", "additional_receipts", "payloads", "ctest_chunks"}
                or result["status"] != "passed" or result["additional_receipts"]
                or set(result["identity"]) != {"command", "inputs", "tools", "tool_paths", "environment", "dependencies"}
                or set(document["context"]) != {"preset", "native_target", "profile", "testing", "features", "runtime", "group"}
                or document["context"]["runtime"] != "host" or document["context"]["group"] != "product"
                or document["context"]["testing"] != "ON" or document["context"]["profile"] != "Debug"
                or document["context"]["preset"] not in {"linux-debug", "windows-clang-debug"}
                or document["context"]["features"] != []
                or set(document["subject"]) != subject_names
                or set(document["preparation"]) != {"identity", "stage"}
                or set(candidate) != {"root", "head", "tree", "files"}
                or not Path(candidate["root"]).is_absolute()
                or not all(re.fullmatch(r"[0-9a-f]{40}", candidate[key]) for key in ("head", "tree"))
                or set(candidate["files"]) != source_names
                or (recipe == "native-c11-miniature-v1" and result["identity"]["inputs"] != candidate["files"])
                or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in candidate["files"].values())
                or set(dependencies) != {"protocol", "root", "files", "directories", "absent", "python"}
                or dependencies["protocol"] != "declared-dependencies-v1" or dependencies["root"] != candidate["root"]
                or dependencies["python"] is not None):
            raise ValueError("bounded original is partial, admitted or missing its subject facts")
        producer = document["producer"]
        if (set(producer) != {"root", "files"} or not Path(producer["root"]).is_absolute()
                or set(producer["files"]) != {"scripts/qualification.py", "scripts/build_evidence.py", "scripts/build_dependencies.py",
                                              "scripts/execution.py", "scripts/process_tree.py", "scripts/verify.py", "scripts/check-quality.py",
                                              "tests/ffi/grain_probe_contract.py", "tests/ffi/effects_frame_probe_contract.py"}
                or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in producer["files"].values())
                or set(document["launch"]) != {"environment_sha256", "supervisor_sha256"}
                or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in document["launch"].values())
                or producer["files"]["scripts/process_tree.py"] != document["launch"]["supervisor_sha256"]):
            raise ValueError("bounded original lacks its producer or actual launch identity")
        root = Path(candidate["root"])
        build = root / "out/build" / document["context"]["preset"]
        binary = build / ("BoundedEvidenceChild.exe" if document["context"]["preset"].startswith("windows") else "BoundedEvidenceChild")
        registration = document["subject"]["registration"]
        properties = {row["name"]: row["value"] for row in registration["properties"]}
        if (set(registration) != {"name", "command", "properties"} or registration["name"] != "Bounded.Read"
                or [Path(value) for value in registration["command"]] != [binary, root / "input.txt", build / "assertion-count.txt"]
                or len(registration["properties"]) != 3 or set(properties) != {"LABELS", "TIMEOUT", "WORKING_DIRECTORY"}
                or properties["LABELS"] != ["host", "product"] or properties["TIMEOUT"] != 0
                or Path(properties["WORKING_DIRECTORY"]) != build):
            raise ValueError("bounded original registration facts disagree with its recipe")
        runnable = {binary}
        if recipe == "native-c11-cli-miniature-v1":
            context = document["subject"]["preparation_facts"]["context"]
            actual = document["subject"]["build_subject"]
            controller = root / "scripts/qualification.py"
            if (set(actual) != set(context)
                    or any(context.get(key) != value for key, value in actual.items() if key != "files")
                    or any(context["files"].get(name) != row for name, row in actual["files"].items())
                    or str(controller) in actual["files"]
                    or "scripts/qualification.py" in actual["configuration"]["inputs"]
                    or str(controller) in actual["configuration"]["tools"]
                    or "qualification.py" in actual["commands"].casefold()
                    or str(root / "src/evidence_child.c") not in actual["files"]
                    or "CMakeLists.txt" not in actual["configuration"]["inputs"]):
                raise ValueError("bounded CLI original lacks its separately resolved actual build subject")
            expected_inputs = dict(candidate["files"])
            for filename in ("CMakeCache.txt", "CTestTestfile.cmake"):
                path = build / filename
                expected_inputs[path.relative_to(root).as_posix()] = dependencies["files"][str(path)]["sha256"]
            if result["identity"]["inputs"] != expected_inputs:
                raise ValueError("bounded CLI original lacks its complete ordinary input membership")
            tooling = document["subject"]["tooling_registration"]
            tooling_properties = {row["name"]: row["value"] for row in tooling["properties"]}
            if (set(tooling) != {"name", "command", "properties"} or tooling["name"] != "Tooling.Miniature"
                    or len(tooling["command"]) != 3 or not Path(tooling["command"][0]).is_absolute()
                    or Path(tooling["command"][0]) not in {Path(name) for name in result["identity"]["tool_paths"]}
                    or tooling["command"][1] != "-B" or Path(tooling["command"][2]) != root / "tests/quality/test_miniature.py"
                    or len(tooling["properties"]) != 3 or set(tooling_properties) != {"LABELS", "TIMEOUT", "WORKING_DIRECTORY"}
                    or tooling_properties["LABELS"] != ["host", "tooling"] or tooling_properties["TIMEOUT"] != 0
                    or Path(tooling_properties["WORKING_DIRECTORY"]) != build):
                raise ValueError("bounded CLI original lacks its independent tooling registration")
            runnable.add(Path(tooling["command"][0]))
        expected = _bounded_files(root, candidate, document["subject"]["preparation_facts"], result["identity"]["tool_paths"], build,
                                  recipe=recipe)
        ctests = [Path(name) for name in result["identity"]["tool_paths"] if Path(name).name.lower() in {"ctest", "ctest.exe"}]
        ctest_targets = ({Path(dependencies["files"][str(path)]["resolved"]) for path in ctests}
                         if recipe == "native-c11-cli-miniature-v1" else set(ctests))
        absent = {str(build / "CTestCustom.cmake"), str(build / "CTestCustom.ctest")}
        if recipe == "native-c11-cli-miniature-v1":
            missing_tools = {name for name in result["identity"]["tool_paths"] if name not in dependencies["files"]}
            if not missing_tools <= set(dependencies["absent"]):
                raise ValueError("bounded CLI original lacks its absent tool/configuration facts")
            absent.update(missing_tools)
            expected.difference_update(Path(name) for name in missing_tools)
            bindings = {}
            for name in result["identity"]["tool_paths"]:
                row = dependencies["files"].get(name, dependencies["absent"].get(name))
                bindings[row["resolved"]] = row.get("sha256")
            if bindings != result["identity"]["tools"]:
                raise ValueError("bounded CLI original tool bindings disagree with the ordinary stage")
        if (set(dependencies["files"]) != {str(path) for path in expected}
                or (len(ctest_targets) != 1 if recipe == "native-c11-cli-miniature-v1" else len(ctests) != 1)
                or set(dependencies["directories"]) != ({str(root / name) for name in _bounded_cli_namespaces()}
                    if recipe == "native-c11-cli-miniature-v1" else {str(root)})
                or set(dependencies["absent"]) != absent):
            raise ValueError("bounded original lacks its complete declared dependency membership")
        if recipe == "native-c11-cli-miniature-v1":
            _check_bounded_cli_namespaces(root, dependencies["directories"])
        runnable.update(ctest_targets)
        for name, row in dependencies["files"].items():
            executable = Path(name) in runnable
            keys = {"kind", "resolved", "sha256", "executable"}
            if executable and not document["context"]["preset"].startswith("windows"):
                keys.add("execute_bits")
                if not isinstance(row.get("execute_bits"), int) or not row["execute_bits"] or row["execute_bits"] & ~0o111:
                    raise ValueError("bounded original lacks executable permission facts")
            if (set(row) != keys or row["kind"] != "file" or row["executable"] is not executable
                    or not Path(name).is_absolute() or not Path(row["resolved"]).is_absolute()
                    or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
                raise ValueError("bounded original has malformed declared file facts")
        directory = dependencies["directories"][str(root)]
        if (set(directory) != {"scope", "resolved", "members"} or directory["scope"] != "immediate"
                or not Path(directory["resolved"]).is_absolute() or not directory["members"]):
            raise ValueError("bounded original lacks its namespace facts")
        members = directory["members"]
        names = {row["name"] for row in members}
        if len(names) != len(members) or not {".git", ".gitignore", "CMakeLists.txt", "CMakePresets.json", "input.txt", "out", "scripts", "src"} <= names:
            raise ValueError("bounded original namespace membership is incomplete")
        if recipe == "native-c11-cli-miniature-v1" and "tests" not in names:
            raise ValueError("bounded CLI original lacks its tooling source namespace")
        for row in members:
            if (set(row) != {"name", "kind", "resolved"} or Path(row["name"]).name != row["name"]
                    or row["name"] in {".", ".."} or row["kind"] not in {"file", "directory"}
                    or not Path(row["resolved"]).is_absolute()):
                raise ValueError("bounded original has malformed namespace facts")
        for row in dependencies["absent"].values():
            if set(row) != {"resolved"} or not Path(row["resolved"]).is_absolute():
                raise ValueError("bounded original has malformed absence facts")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("malformed bounded original facts") from error
    facts = {key: document[key] for key in ("policy", "candidate", "context", "controller_paths", "subject", "preparation")}
    facts["stage_identity"] = result["identity"]
    sealed = _bounded_hash({"facts": facts, "producer": document["producer"], "launch": document["launch"]})
    if document["seals"] != {"entry": sealed, "terminal": sealed}:
        raise ValueError("bounded original entry/terminal seals disagree")
    return result


def read_execution_contract(reference):
    """Validate immutable original payloads; never require an old preparation to own current output."""
    from verify import check_receipt_payloads
    raw_bytes = _bounded_payload(reference)
    document = identity_json(raw_bytes.decode("utf-8"))
    if document.get("protocol") in {"product-evidence-policy-v1", "product-evidence-policy-v2"}:
        return read_policy_contract(reference)
    result = _check_bounded_contract(document)
    if document["policy"] != _bounded_policy(recipe=document["policy"]["recipe"]):
        raise ValueError("bounded original recipe policy changed")
    if raw_bytes != _bounded_bytes(document):
        raise ValueError("bounded contract is not canonical JSON")
    root = Path(document["candidate"]["root"])
    execution = document["execution"]
    if (set(execution) != {"receipt", "output", "process", "junit", "last_test", "counter", "started_at", "finished_at"}
            or Path(reference["path"]) != Path(result["receipt"]).with_name("execution-contract.json")
            or execution["receipt"] != {"path": result["receipt"], "sha256": result["sha256"]}):
        raise ValueError("bounded original references do not bind the actual execution")
    for key in ("receipt", "output", "process", "junit", "last_test", "counter"):
        _bounded_payload(execution[key])
    raw = check_receipt_payloads(root, result["receipt"])
    expected_tools = result["identity"]["tools"]
    if document["policy"]["recipe"] == "native-c11-cli-miniature-v1":
        expected_tools = {name: checksum for name, checksum in expected_tools.items() if checksum is not None}
    if (raw["stage"] != "product" or raw["started_at"] != execution["started_at"]
            or raw["finished_at"] != execution["finished_at"]
            or raw["identity"]["inputs"] != result["identity"]["inputs"]
            or raw["identity"]["tools"] != expected_tools
            or any(raw["identity"].get(key) != value for key, value in document["launch"].items())
            or [ctest_identity(raw["identity"]["command"])] != result["identity"]["command"]
            or execution["output"]["path"] != str(root / raw["output"])
            or execution["process"]["path"] != raw["process_tree"]["receipt"]
            or execution["junit"]["path"] != raw["identity"]["command"][-1]
            or execution["last_test"]["path"] != str(Path(result["receipt"]).with_name("LastTest.log"))
            or _bounded_payload(execution["counter"]) != b"1\n"):
        raise ValueError("bounded original raw execution correspondence changed")
    validate_ctest_stage(Path(result["receipt"]), {"host": ["Bounded.Read"]}, ["Bounded.Read"], result["identity"]["command"])
    prepared = document["preparation"]["stage"]
    prep_ref = {"path": prepared["receipt"], "sha256": prepared["sha256"]}
    prepared_raw = identity_json(_bounded_payload(prep_ref).decode("utf-8"))
    check_receipt_payloads(root, prepared["receipt"])
    historical = {key: value for key, value in prepared_raw["scoped_preparation"].items()
                  if key not in {"producer", "invalidated", "rust_invalidated_packages"}}
    artifacts = prepared.get("payloads", {})
    if (prepared.get("status") != "passed" or prepared.get("additional_receipts")
            or not artifacts or set(artifacts) != set(historical["outputs"])
            or artifacts != {name: row["sha256"] for name, row in historical["output_identities"].items()}
            or historical != document["subject"]["preparation_facts"]
            or _preparation_facts_identity(historical, prepared) != document["preparation"]["identity"]
            or result["identity"]["dependencies"] != [document["preparation"]["identity"]]):
        raise ValueError("bounded original preparation dependency changed")
    for name, checksum in result["payloads"].items():
        _bounded_payload({"path": name, "sha256": checksum})
    return {"reference": dict(reference), "contract": document}


def _check_reuse_declaration(declaration):
    required = {"protocol", "original_candidate", "current_candidate", "changes", "originals", "compatibility",
                "fresh_evidence", "review", "owner_decision"}
    if (not isinstance(declaration, dict) or set(declaration) != required
            or declaration["protocol"] != "tooling-evidence-transition-bounded-v1"):
        raise ValueError("unsupported or malformed bounded transition declaration")
    evidence = declaration["fresh_evidence"]
    if (not isinstance(evidence, dict) or set(evidence) != {"tooling", "quality", "quality_base"}
            or not isinstance(evidence["tooling"], list) or not evidence["tooling"]
            or not isinstance(evidence["quality"], list) or not isinstance(evidence["quality_base"], str)
            or not evidence["quality_base"]):
        raise ValueError("bounded transition lacks supporting reference scope")
    references = [declaration["review"], declaration["owner_decision"], *evidence["tooling"], *evidence["quality"]]
    for reference in references:
        if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
                or not isinstance(reference["path"], str) or not Path(reference["path"]).is_absolute()
                or not isinstance(reference["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])):
            raise ValueError("malformed bounded review/supporting reference")
    _bounded_bytes(declaration)
    return references


def read_reuse_declaration(reference):
    """Check reference integrity only; R3 owns supporting evidence admission and human approval."""
    declaration = identity_json(_bounded_payload(reference).decode("utf-8"))
    for evidence in _check_reuse_declaration(declaration):
        _bounded_payload(evidence)
    return declaration


def compare_execution_subject(original, current, declaration):
    """Compare captured values only. This authenticates no review and admits nothing."""
    if not isinstance(original, dict) or set(original) != {"reference", "contract"}:
        raise ValueError("comparison requires a direct original contract reference")
    document = original["contract"]
    result = _check_bounded_contract(document)
    if _bounded_hash(document) != original["reference"]["sha256"]:
        raise ValueError("original comparison document changed")
    _check_reuse_declaration(declaration)
    if (not isinstance(current, dict) or set(current) != {"policy", "candidate", "context", "controller_paths",
                                                        "subject", "preparation", "stage_identity"}
            or current["controller_paths"] != document["controller_paths"]):
        raise ValueError("malformed bounded current subject facts")
    if (declaration["original_candidate"] != document["candidate"]
            or declaration["current_candidate"] != current["candidate"]
            or declaration["originals"] != [{"preset": document["context"]["preset"], "group": "product",
                                             "contract": original["reference"]}]):
        raise ValueError("unreviewed transition: source pair or direct original reference disagrees")
    old, new = document["candidate"]["files"], current["candidate"]["files"]
    if set(old) != set(new):
        raise ValueError("subject_changed: candidate membership changed")
    changes = [{"path": name, "old_sha256": old[name], "new_sha256": new[name], "groups": ["product"]}
               for name in sorted(old) if old[name] != new[name]]
    if (not changes or declaration["changes"] != changes
            or any(row["path"] not in document["controller_paths"] for row in changes)
            or declaration["compatibility"] != [{"path": row["path"], "group": "product", "role": "controller-only",
                                                   "outcome_integrity": "preserved"} for row in changes]):
        raise ValueError("unreviewed transition: changed subject, consumer or historical trustworthiness")
    if current["policy"] != document["policy"] or current["context"] != document["context"]:
        raise ValueError("subject_changed: policy or native context changed")
    expected_subject = identity_json(_bounded_bytes(document["subject"]).decode("utf-8"))
    if document["policy"]["recipe"] == "native-c11-cli-miniature-v1":
        # Broad preparation observations are retained verbatim in both proofs.
        # The separately constructed build subject must exclude this controller.
        for row in changes:
            path = str(Path(document["candidate"]["root"]) / row["path"])
            observed = expected_subject["preparation_facts"]["context"]["files"].get(path)
            if observed is not None:
                if observed["sha256"] != row["old_sha256"]:
                    raise ValueError("preparation_changed: original controller observation disagrees")
                observed["sha256"] = row["new_sha256"]
    if current["subject"] != expected_subject:
        raise ValueError("subject_changed: declared dependencies or preparation facts changed")
    expected_identity = {**result["identity"], "inputs": {**result["identity"]["inputs"],
                        **{row["path"]: row["new_sha256"] for row in changes}},
                        "dependencies": [current["preparation"]["identity"]]}
    if current["stage_identity"] != expected_identity:
        raise ValueError("tools_changed: command, tool, environment or stage dependency facts changed")
    # Reference semantics/authentication and current quality admission belong to R3.
    # R2 binds the data supplied for later review without declaring it approved.
    evidence = {"fresh_evidence": declaration["fresh_evidence"], "review": declaration["review"],
                "owner_decision": declaration["owner_decision"]}
    return {"protocol": "product-evidence-comparison-bounded-v1", "status": "compatible", "scope": "comparison-only",
            "original": original["reference"], "declaration_sha256": _bounded_hash(declaration),
            "current_candidate_sha256": _bounded_hash(current["candidate"]),
            "original_preparation_identity": document["preparation"]["identity"],
            "current_preparation_identity": current["preparation"]["identity"],
            "supporting_references_sha256": _bounded_hash(evidence), "review_authenticated": False}


def _read_admission_request(root, args):
    """Read strict opt-in data before configuration queries or stage mutation."""
    path = Path(args.reuse_declaration).absolute()
    reference = {"path": str(path), "sha256": digest(path)}
    if identity_json(_bounded_payload(reference).decode("utf-8")).get("protocol") == "tooling-evidence-policy-transition-v1":
        return _read_policy_request(root, args, reference)
    declaration = read_reuse_declaration(reference)
    _check_bounded_cli_environment(dict(os.environ))
    if set(args.selection) != {"product"} or args.runtime != "host":
        raise ValueError("unsupported_protocol: only the synthetic native host product recipe is eligible")
    candidates = (declaration["original_candidate"], declaration["current_candidate"])
    names = {".gitignore", "CMakeLists.txt", "CMakePresets.json", "src/evidence_child.c", "input.txt",
             "scripts/qualification.py", "scripts/check-quality.py", "tests/quality/test_miniature.py"}
    for candidate in candidates:
        if (not isinstance(candidate, dict) or set(candidate) != {"root", "head", "tree", "files"}
                or candidate["root"] != str(root.resolve())
                or any(not isinstance(candidate[key], str) or not re.fullmatch(r"[0-9a-f]{40}", candidate[key])
                       for key in ("head", "tree"))
                or not isinstance(candidate["files"], dict) or set(candidate["files"]) != names
                or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                       for value in candidate["files"].values())):
            raise ValueError("unreviewed_transition: malformed candidate membership or identity")
    old, new = (candidate["files"] for candidate in candidates)
    changes = [{"path": name, "old_sha256": old[name], "new_sha256": new[name], "groups": ["product"]}
               for name in sorted(names) if old[name] != new[name]]
    if (not changes or declaration["changes"] != changes
            or any(row["path"] != "scripts/qualification.py" for row in changes)
            or declaration["compatibility"] != [{"path": row["path"], "group": "product", "role": "controller-only",
                                                "outcome_integrity": "preserved"} for row in changes]):
        raise ValueError("unreviewed_transition: exact controller delta or historical integrity disagrees")
    originals = declaration["originals"]
    if (not isinstance(originals, list) or len(originals) != 1 or not isinstance(originals[0], dict)
            or set(originals[0]) != {"preset", "group", "contract"}
            or originals[0]["preset"] != args.preset or originals[0]["group"] != "product"):
        raise ValueError("unsupported_protocol: one direct original in this native preset is required")
    core = _bounded_hash({key: value for key, value in declaration.items() if key not in {"review", "owner_decision"}})
    for key, protocol, decision in (("review", "tooling-evidence-review-bounded-v1", "compatible"),
                                    ("owner_decision", "tooling-evidence-owner-bounded-v1", "approve")):
        approval = identity_json(_bounded_payload(declaration[key]).decode("utf-8"))
        expected = {"protocol": protocol, "transition_sha256": core,
                    "original_candidate_sha256": _bounded_hash(candidates[0]),
                    "current_candidate_sha256": _bounded_hash(candidates[1]), "originals": originals,
                    "scope": {"preset": args.preset, "groups": ["product"]}, "decision": decision}
        if approval != expected:
            raise ValueError("unreviewed_transition: stale or incorrectly scoped " + key)
    if (not isinstance(args.base, str) or not re.fullmatch(r"[0-9a-f]{40}", args.base)
            or declaration["fresh_evidence"]["quality_base"] != args.base
            or subprocess.check_output(["git", "rev-parse", args.base + "^{commit}"], cwd=root, text=True).strip() != args.base):
        raise ValueError("unreviewed_transition: quality requires the declared immutable accepted base")
    original = read_execution_contract(originals[0]["contract"])
    if (original["contract"]["policy"]["recipe"] != "native-c11-cli-miniature-v1"
            or original["contract"]["candidate"] != candidates[0]
            or original["contract"]["context"]["preset"] != args.preset):
        raise ValueError("unsupported_protocol: original is not this prospective synthetic CLI recipe")
    if _bounded_candidate(root, recipe="native-c11-cli-miniature-v1") != candidates[1]:
        raise ValueError("subject_changed: current candidate differs from the reviewed transition")
    return {"reference": reference, "declaration": declaration, "original": original,
            "validator": _bounded_producer()}


def _check_stage_owner(state_path, state, name, expected):
    if state_path.resolve(strict=True) != state_path:
        raise ValueError("owner_conflict: checkpoint owner path was retargeted")
    current = identity_json(read_text(state_path, encoding="utf-8"))
    expected = {key: value for key, value in expected.items() if key != "reused_from"}
    if current != state or current.get(name) != expected or expected.get("status") != "passed":
        raise ValueError("owner_conflict: pending, failed or newer checkpoint owner: " + name)


def _check_stage_evidence(root, stage):
    from verify import check_receipt_payloads
    if not isinstance(stage, dict) or stage.get("status") != "passed" or stage.get("reuse_admission"):
        raise ValueError("incomplete_original: independent ordinary successful support is required")
    launch = stage.get("execution_launch")
    if (not isinstance(launch, dict) or set(launch) != {"environment_sha256", "supervisor_sha256", "stage_environment", "producer"}
            or launch["stage_environment"] != stage["identity"]["environment"]
            or launch["producer"] != _bounded_producer()
            or launch["supervisor_sha256"] != digest(Path(__file__).with_name("process_tree.py"))):
        raise ValueError("evidence_missing: support lacks prospective actual launch provenance")
    paths = stage_receipts(stage)
    references = [{"path": str(path), "sha256": checksum} for path, checksum in zip(
        paths, [stage["sha256"], *stage.get("additional_receipts", {}).values()], strict=True)]
    raw = []
    for reference in references:
        _bounded_payload(reference)
        command = check_receipt_payloads(root, reference["path"])
        if (command["identity"]["inputs"] != stage["identity"]["inputs"]
                or command["identity"]["tools"] != {k: v for k, v in stage["identity"]["tools"].items() if v is not None}
                or any(command["identity"].get(key) != launch[key] for key in ("environment_sha256", "supervisor_sha256"))):
            raise ValueError("evidence_missing: support command does not bind its stage inputs/tools")
        raw.append(command)
    for name, checksum in stage.get("payloads", {}).items():
        _bounded_payload({"path": name, "sha256": checksum})
    return raw


def _read_synthetic_quality(root, build, stage, base, *, recipe="native-c11-cli-miniature-v1",
                            resolve=Path, native_maps=None, producer=None, candidate=None):
    """Bind the finite checker report to its completed ordinary Campaign command."""
    from datetime import datetime
    import runpy
    if recipe not in {"native-c11-cli-miniature-v1", *_POLICY_READ_RECIPES}:
        raise ValueError("unsupported_protocol: unknown finite quality recipe")
    policy = recipe in _POLICY_READ_RECIPES
    raw = _policy_support_stage(root, stage, resolve=resolve, producer=producer) if policy else _check_stage_evidence(root, stage)
    interpreter = stage["identity"]["command"][0] if policy else sys.executable
    expected_command = [interpreter, "-B", "scripts/check-quality.py", "--preset", build.name,
                        "--qualification", "--base", base]
    if (len(raw) != 1 or stage["identity"]["command"] != expected_command
            or raw[0]["identity"]["command"] != expected_command):
        raise ValueError("evidence_missing: quality command/base does not match integrated qualification")
    lines = read_text(root / raw[0]["output"], encoding="utf-8").splitlines()
    reports = [line.removeprefix("Synthetic quality report: ") for line in lines
               if line.startswith("Synthetic quality report: ")]
    seals = [line.removeprefix("Synthetic quality seal: ") for line in lines
             if line.startswith("Synthetic quality seal: ")]
    if len(reports) != 1 or len(seals) != 1:
        raise ValueError("evidence_missing: one independently completed synthetic quality report is required")
    reference = {"path": reports[0], "sha256": seals[0]} if policy else _bounded_reference(reports[0])
    if reference["sha256"] != seals[0]:
        raise ValueError("evidence_missing: synthetic quality report differs from its executed seal")
    path = resolve(reference["path"])
    if not path.is_relative_to(root / "out/validation" / build.name / "synthetic-quality"):
        raise ValueError("evidence_missing: synthetic quality report has the wrong owner")
    report = identity_json((_policy_payload(reference, resolve) if policy else _bounded_payload(reference)).decode("utf-8"))
    fields = {"protocol", "assurance", "status", "root", "preset", "base", "candidate", "inputs", "tools",
              "policy", "checks", "started_at", "finished_at"}
    template = Path(__file__).resolve().parents[1] / ("tests/quality/evidence/policy/check_quality.py" if policy
                                                   else "tests/quality/evidence/cli/check_quality.py")
    policies = runpy.run_path(str(template), run_name="synthetic_quality_policy")
    candidate = candidate or (_policy_candidate(root) if policy else _bounded_candidate(root, recipe="native-c11-cli-miniature-v1"))
    original_root = candidate["root"] if policy else str(root)
    native_path = PureWindowsPath if PureWindowsPath(original_root).is_absolute() else Path
    native_root = native_path(original_root)
    native_build = native_root / "out/build" / build.name
    expected_inputs = [native_root / name for name in policies["SOURCE_FILES"]] + [native_build / "compile_commands.json"]
    if (not isinstance(report, dict) or set(report) != fields
            or report["protocol"] != ("synthetic-policy-quality-v1" if policy else "synthetic-cli-quality-v1")
            or report["assurance"] != (_POLICY_ASSURANCE if policy else "synthetic-c11-cli-quality") or report["status"] != "passed"
            or report["root"] != original_root or report["preset"] != build.name or report["base"] != base
            or report["policy"] != policies["POLICY"]
            or report["candidate"] != candidate["files"]
            or set(report["inputs"]) != {str(p) for p in expected_inputs}
            or set(report["tools"]) != {"python", "git", "ruff", "clang-format", "clang-tidy"}
            or not isinstance(report["checks"], list)
            or [row.get("name") for row in report["checks"]] != ["diff", "syntax", "ruff", "format", "tidy"]):
        raise ValueError("evidence_missing: incomplete or incorrectly scoped synthetic quality report")
    if not (datetime.fromisoformat(raw[0]["started_at"]) <= datetime.fromisoformat(report["started_at"])
            <= datetime.fromisoformat(report["finished_at"]) <= datetime.fromisoformat(raw[0]["finished_at"])):
        raise ValueError("evidence_missing: synthetic quality completion is outside its command lifetime")
    for name, row in report["inputs"].items():
        if policy:
            if set(row) != {"resolved", "sha256"}:
                raise ValueError("terminal_changed: malformed synthetic quality input")
            _policy_live_file(name, row, resolve=resolve, native_maps=native_maps)
        elif row != {"resolved": str(Path(name).resolve(strict=True)), "sha256": digest(Path(name))}:
            raise ValueError("terminal_changed: synthetic quality input changed: " + name)
    references = [reference]
    for name, row in report["tools"].items():
        if not isinstance(row, dict) or "path" not in row:
            raise ValueError("tools_changed: malformed synthetic quality tool: " + name)
        if policy and os.name == "posix" and PureWindowsPath(row["path"]).is_absolute():
            if set(row) != {"path", "version", "resolved", "sha256", "execute_mask", "launch_access"} or row["execute_mask"] is not None or row["launch_access"] is not True:
                raise ValueError("tools_changed: malformed native Windows quality helper facts")
            _policy_live_file(row["path"], row, resolve=resolve, native_maps=native_maps)
            metadata = {key: row[key] for key in ("resolved", "sha256", "execute_mask", "launch_access")}
        else:
            metadata = policies["executable_identity"](row["path"])
        if (set(row) != {"path", "version", *metadata}
                or {key: row[key] for key in metadata} != metadata or not isinstance(row["version"], str)
                or not row["version"]):
            raise ValueError("tools_changed: synthetic quality tool changed: " + name)
    if policy and report["tools"]["python"]["path"] != interpreter:
        raise ValueError("tools_changed: policy quality interpreter differs from its raw executable")
    commands = policies["quality_commands"](native_root, native_build, base, report["tools"])
    for check in report["checks"]:
        if (set(check) != {"name", "command", "cwd", "returncode", "stdout", "stderr"}
                or check["returncode"] != 0 or check["cwd"] != original_root
                or check["command"] != commands[check["name"]]):
            raise ValueError("incomplete_original: synthetic quality check is incomplete")
        for stream in ("stdout", "stderr"):
            if not resolve(check[stream]["path"]).is_relative_to(path.parent):
                raise ValueError("evidence_missing: quality check output has the wrong owner")
            _policy_payload(check[stream], resolve) if policy else _bounded_payload(check[stream])
            references.append(check[stream])
    return {"report": reference, "payloads": references, "tools": report["tools"]}


def _check_synthetic_quality(root, build, stage, base, *, recipe="native-c11-cli-miniature-v1"):
    evidence = _read_synthetic_quality(root, build, stage, base, recipe=recipe)
    if stage.get("payloads") != {ref["path"]: ref["sha256"] for ref in evidence["payloads"]}:
        raise ValueError("evidence_missing: quality lacks its immutable completed payload closure")
    return evidence


def _validate_reuse_support(campaign, state_path, state, reuse, environment, quality=None, *, collection=False, recorded_support=None):
    from verify import source_snapshot
    root, build = campaign.root, state_path.parent
    declaration = read_reuse_declaration(reuse["reference"])
    if declaration != reuse["declaration"] or _bounded_producer() != reuse["validator"]:
        raise ValueError("terminal_changed: declaration or admission validator changed")
    tooling = declaration["fresh_evidence"]["tooling"]
    if len(tooling) != 1:
        raise ValueError("evidence_missing: one complete current native tooling wrapper is required")
    wrapper = identity_json(_bounded_payload(tooling[0]).decode("utf-8"))
    require_checkpoint(wrapper)
    if (wrapper["selection"] != ["tooling"] or wrapper["preset"] != build.name or wrapper["runtime"] != "host"
            or wrapper.get("source") != source_snapshot(root) or wrapper.get("reuse_declaration")
            or wrapper.get("membership") != {"tooling": ["Tooling.Miniature"]}
            or wrapper.get("runtime_membership") != {"tooling": {"host": ["Tooling.Miniature"]}}):
        raise ValueError("evidence_missing: tooling source, native context or complete membership disagrees")
    stage = {key: value for key, value in wrapper["stages"]["tooling"].items() if key != "reused_from"}
    if collection and (not isinstance(recorded_support, dict) or recorded_support.get("tooling_stage") != stage):
        raise ValueError("terminal_changed: tooling differs from its strongly admitted closure")
    _check_stage_owner(state_path, state, "tooling", stage)
    raw = _check_stage_evidence(root, stage)
    if stage["identity"]["dependencies"] or any(row["stage"] != "tooling" for row in raw):
        raise ValueError("unreviewed_transition: circular tooling support")
    commands = stage["identity"]["command"]
    expected = [*ctest_command(root, build), "-R", "^(Tooling\\.Miniature)$", "-LE", "gpu", "--no-tests=error",
                "--output-on-failure", "--output-junit", "<junit>"]
    if commands != [expected]:
        raise ValueError("evidence_missing: tooling command or runtime changed")
    validate_ctest_stage(stage_receipts(stage), {"host": ["Tooling.Miniature"]}, ["Tooling.Miniature"], commands)
    inputs = checkpoint_inputs(root, build, "tooling", source_files=source_snapshot(root))
    tools = (stage["identity"]["tool_paths"] if collection else sorted(set(checkpoint_tools(build, environment, "tooling")) | set(
        ctest_artifacts({"tests": [{"name": "Tooling.Miniature", "command": [sys.executable, "-B",
            str(root / "tests/quality/test_miniature.py")]}]}, ["Tooling.Miniature"], build)[0])))
    if _stage_identity(root, commands, inputs, tools, environment, []) != stage["identity"]:
        raise ValueError("tools_changed: current tooling inputs, tools or environment changed")
    supporting = {"tooling": tooling[0], "tooling_stage": stage}
    if quality is not None:
        quality = {key: value for key, value in quality.items() if key != "reused_from"}
        if collection and recorded_support.get("quality_stage") != quality:
            raise ValueError("terminal_changed: quality differs from its strongly admitted closure")
        _check_stage_owner(state_path, state, "prepare", state["prepare"])
        _check_stage_owner(state_path, state, "quality", quality)
        prepared_identity = preparation_identity(build, state["prepare"])
        baseline, pending = preparation_paths(build, ["all"])
        if baseline.resolve(strict=True) != baseline or pending.exists() or pending.is_symlink():
            raise ValueError("owner_conflict: current preparation owner path changed or is pending")
        supporting["preparation_owner"] = _bounded_reference(baseline)
        if quality["identity"]["dependencies"] != [prepared_identity]:
            raise ValueError("preparation_changed: quality does not depend on current preparation")
        expected_quality = _stage_identity(root, [sys.executable, "-B", "scripts/check-quality.py", "--preset",
            build.name, "--qualification", "--base", declaration["fresh_evidence"]["quality_base"]],
            checkpoint_inputs(root, build, "product", source_files=source_snapshot(root)),
            quality["identity"]["tool_paths"] if collection else checkpoint_tools(build, environment, "product"), environment, [prepared_identity])
        if quality["identity"] != expected_quality:
            raise ValueError("tools_changed: current integrated quality identity changed")
        supporting["quality_stage"] = quality
        supporting["quality"] = _check_synthetic_quality(root, build, quality,
                                                         declaration["fresh_evidence"]["quality_base"])
        for reference in declaration["fresh_evidence"]["quality"]:
            if reference != {"path": quality["receipt"], "sha256": quality["sha256"]}:
                raise ValueError("unreviewed_transition: quality reference is not the current independent command")
    if collection and supporting != recorded_support:
        raise ValueError("terminal_changed: current supporting identity/reference differs from admission")
    return supporting


def _read_committed_admission(reference, build):
    from datetime import datetime
    payload = _bounded_payload(reference)
    record = identity_json(payload.decode("utf-8"))
    facts = {"declaration", "original", "current_candidate", "current_preparation", "comparison", "support", "validator", "owner"}
    if (not isinstance(record, dict) or set(record) != {"protocol", "assurance", "admitted_at", *facts}
            or record["protocol"] != "product-evidence-admission-bounded-v1"
            or record["assurance"] != "synthetic-c11-cli-quality" or payload != _bounded_bytes(record)
            or not isinstance(record["admitted_at"], str)):
        raise ValueError("owner_conflict: malformed committed admission")
    datetime.fromisoformat(record["admitted_at"])
    expected_path = build / "verification-admissions" / _bounded_hash({key: record[key] for key in facts}) / "admission.json"
    if str(expected_path) != reference["path"]:
        raise ValueError("owner_conflict: committed admission identity/path disagrees")
    declaration = read_reuse_declaration(record["declaration"])
    original = read_execution_contract(record["original"])
    if (record["current_candidate"] != declaration["current_candidate"]
            or original["contract"]["candidate"] != declaration["original_candidate"]
            or declaration["originals"] != [{"preset": build.name, "group": "product", "contract": record["original"]}]
            or record["owner"] != {**original["contract"]["stage"]["result"], "execution_contract": record["original"]}
            or not isinstance(record["current_preparation"], dict)
            or set(record["current_preparation"]) != {"identity", "stage"}
            or record["current_preparation"]["stage"].get("status") != "passed"):
        raise ValueError("owner_conflict: committed admission source/original/preparation binding disagrees")
    prepared = record["current_preparation"]["stage"]
    proof = identity_json(_bounded_payload({"path": prepared["receipt"], "sha256": prepared["sha256"]}).decode("utf-8"))["scoped_preparation"]
    proof = {key: value for key, value in proof.items() if key not in {"producer", "invalidated", "rust_invalidated_packages"}}
    if _preparation_facts_identity(proof, prepared) != record["current_preparation"]["identity"]:
        raise ValueError("owner_conflict: committed admission preparation identity disagrees")
    evidence = {"fresh_evidence": declaration["fresh_evidence"], "review": declaration["review"],
                "owner_decision": declaration["owner_decision"]}
    expected_comparison = {"protocol": "product-evidence-comparison-bounded-v1", "status": "compatible", "scope": "comparison-only",
        "original": record["original"], "declaration_sha256": _bounded_hash(declaration),
        "current_candidate_sha256": _bounded_hash(record["current_candidate"]),
        "original_preparation_identity": original["contract"]["preparation"]["identity"],
        "current_preparation_identity": record["current_preparation"]["identity"],
        "supporting_references_sha256": _bounded_hash(evidence), "review_authenticated": False}
    if record["comparison"] != expected_comparison:
        raise ValueError("owner_conflict: committed comparison does not bind its declaration and preparations")
    return record


def _read_admission_completion(admission, build):
    record = _read_committed_admission(admission, build)
    path = Path(admission["path"]).with_name("completion.json")
    completed = identity_json(_bounded_payload({"path": str(path), "sha256": digest(path)}).decode("utf-8"))
    if not isinstance(completed, dict) or set(completed) != {"admission", "wrapper"} or completed["admission"] != admission:
        raise ValueError("owner_conflict: admission completion is malformed or names another admission")
    wrapper = identity_json(_bounded_payload(completed["wrapper"]).decode("utf-8"))
    require_checkpoint(wrapper)
    candidate = record["current_candidate"]
    expected_stages = {"prepare": record["current_preparation"]["stage"],
                       "quality": record["support"]["quality_stage"],
                       "product": {**record["owner"], "reuse_admission": admission}}
    stages = {name: {key: value for key, value in stage.items() if key != "reused_from"}
              for name, stage in wrapper["stages"].items()}
    if (wrapper.get("reuse_declaration") != record["declaration"]
            or wrapper["stages"].get("product", {}).get("reuse_admission") != admission
            or wrapper.get("preset") != build.name or wrapper.get("runtime") != "host"
            or wrapper.get("selection") != ["product"] or wrapper.get("assurance") != "synthetic-c11-cli-quality"
            or wrapper.get("stage_provenance", {}).get("product") != "admitted"
            or set(wrapper.get("stage_provenance", {})) != {"prepare", "quality", "product"}
            or wrapper["stage_provenance"]["prepare"] != "exact-reused"
            or wrapper["stage_provenance"]["quality"] not in {"executed", "exact-reused"}
            or stages != expected_stages
            or wrapper.get("membership") != {"product": ["Bounded.Read"]}
            or wrapper.get("runtime_membership") != {"product": {"host": ["Bounded.Read"]}}
            or wrapper.get("source") != candidate["files"] | {
                "rust/film-juicer-plugin/src/cuda/sys.rs": None, "src/GeneratedColorSpaces.cpp": None,
                "src/GeneratedColorSpaces.h": None}):
        raise ValueError("owner_conflict: completed wrapper does not bind this admission and candidate")
    return completed


def _check_original_stage_owner(state_path, state, reuse):
    _check_stage_owner(state_path, state, "product", state.get("product", {}))
    expected = {**reuse["original"]["contract"]["stage"]["result"],
                "execution_contract": reuse["original"]["reference"]}
    current = {key: value for key, value in state["product"].items() if key != "reuse_admission"}
    if current != expected:
        raise ValueError("owner_conflict: latest product owner is not the direct original execution")
    prior = state["product"].get("reuse_admission")
    if prior:
        record = _read_committed_admission(prior, state_path.parent)
        if record.get("original") != reuse["original"]["reference"]:
            raise ValueError("owner_conflict: committed admission names another original")
        completion = Path(prior["path"]).with_name("completion.json")
        if record.get("declaration") != reuse["reference"] or completion.exists() or completion.is_symlink():
            _read_admission_completion(prior, state_path.parent)


def _write_admission_record(path, record):
    """Publish complete bytes under the existing exclusive build lease."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve(strict=True) != path.parent:
        raise ValueError("owner_conflict: admission directory was retargeted")
    temporary = path.with_name("record-" + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(_bounded_bytes(record))
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise ValueError("owner_conflict: immutable admission already exists")
    temporary.replace(path)


def admit_stage_evidence(campaign, state_path, state, name, command, *, inputs, tools, environment,
                         dependencies, semantic_command, reuse, validate=None):
    """Admit one direct original without invoking its product action or changing its identity."""
    if name != "product":
        raise ValueError("unsupported_protocol: no real product/boundary/corpus/OFF admission recipe is enabled")
    OutputLease.check_held()
    expected_state = identity_json(read_text(state_path, encoding="utf-8"))
    if expected_state != state:
        raise ValueError("owner_conflict: current checkpoint state changed")
    _check_original_stage_owner(state_path, state, reuse)
    def observe():
        if identity_json(read_text(state_path, encoding="utf-8")) != expected_state:
            raise ValueError("owner_conflict: latest stage changed during admission")
        original = read_execution_contract(reuse["original"]["reference"])
        if original != reuse["original"]:
            raise ValueError("terminal_changed: original execution changed")
        support = _validate_reuse_support(campaign, state_path, state, reuse, environment, state["quality"])
        current = capture_execution_subject(campaign, state, command, inputs=inputs, tools=tools,
            environment=environment, recipe="native-c11-cli-miniature-v1")
        actual_identity = _stage_identity(campaign.root, semantic_command, inputs, tools, environment, dependencies)
        if current["stage_identity"] != actual_identity:
            raise ValueError("tools_changed: requested product identity differs from current subject")
        comparison = compare_execution_subject(original, current, reuse["declaration"])
        if validate:
            validate(stage_receipts(state["product"])[0])
        return {"declaration": reuse["reference"], "original": original["reference"],
                "current_candidate": current["candidate"], "current_preparation": current["preparation"],
                "comparison": comparison, "support": support, "validator": reuse["validator"],
                "owner": expected_state["product"]}
    facts = observe()
    # The previous pointer is an observation, not part of the stable admission ID.
    facts["owner"] = {k: v for k, v in facts["owner"].items() if k != "reuse_admission"}
    key = _bounded_hash(facts)
    path = state_path.parent / "verification-admissions" / key / "admission.json"
    prior = state["product"].get("reuse_admission")
    if prior:
        prior_record = _read_committed_admission(prior, state_path.parent)
        if prior_record["declaration"] == reuse["reference"] and prior["path"] != str(path):
            raise ValueError("owner_conflict: committed admission differs from this exact recovery")
    if prior and Path(prior["path"]) == path:
        record = identity_json(_bounded_payload(prior).decode("utf-8"))
    elif path.exists():
        record = identity_json(_bounded_payload({"path": str(path), "sha256": digest(path)}).decode("utf-8"))
    else:
        from datetime import datetime, timezone
        record = {"protocol": "product-evidence-admission-bounded-v1", "assurance": "synthetic-c11-cli-quality",
                  "admitted_at": datetime.now(timezone.utc).isoformat(), **facts}
        _write_admission_record(path, record)
    if (set(record) != {"protocol", "assurance", "admitted_at", *facts}
            or record["protocol"] != "product-evidence-admission-bounded-v1"
            or record["assurance"] != "synthetic-c11-cli-quality"
            or {key: record[key] for key in facts} != facts):
        raise ValueError("terminal_changed: retained admission does not match this exact request")
    terminal = observe()
    terminal["owner"] = {k: v for k, v in terminal["owner"].items() if k != "reuse_admission"}
    if terminal != facts:
        raise ValueError("terminal_changed: admission facts changed before publication")
    reference = {"path": str(path), "sha256": digest(path)}
    if identity_json(_bounded_payload(reference).decode("utf-8")) != record:
        raise ValueError("terminal_changed: admission record changed before publication")
    _check_original_stage_owner(state_path, state, reuse)
    result = {**state[name], "reuse_admission": reference}
    state[name] = result
    write_json(state_path, state)
    return result


def run_stage(campaign, state_path, state, name, command, *, inputs, tools, environment,
          timeout, semantic_command=None, action=None, validate=None, payloads=(),
          dependencies=(), repeat_reason="", refresh_inputs=None, admission_observations=None, ctest=False,
          artifact_directories=(), bounded_capture=None, reuse=None, synthetic_quality_base=None,
          synthetic_quality_recipe="native-c11-cli-miniature-v1", policy_support=False):
    """Reuse the latest gate across attempt names; pending work is never a pass."""
    campaign.check_run_request(name, command, inputs=inputs, timeout=timeout)
    if bounded_capture is not None and bounded_capture not in {"native-c11-miniature-v1", "native-c11-cli-miniature-v1", *_POLICY_READ_RECIPES}:
        raise ValueError("unsupported bounded original capture recipe")
    if synthetic_quality_base is not None and name != "quality":
        raise ValueError("synthetic quality capture requires its quality stage")
    if synthetic_quality_recipe not in {"native-c11-cli-miniature-v1", *_POLICY_READ_RECIPES}:
        raise ValueError("unsupported_protocol: unknown finite quality recipe")
    identity = _stage_identity(campaign.root, semantic_command or command, inputs, tools, environment,
                               dependencies, artifact_directories)
    previous = state.get(name)
    matching = previous and previous["identity"] == identity
    if matching and previous["status"] != "passed" and not repeat_reason:
        raise ValueError("pending checkpoint gate requires a relevant repair or a diagnosed repeat reason: " + name)
    if matching and previous["status"] == "passed":
        try:
            from verify import check_receipt_payloads
            paths = stage_receipts(previous)
            for path, expected in zip(paths, [previous["sha256"], *previous.get("additional_receipts", {}).values()], strict=True):
                if digest(path) != expected:
                    raise ValueError("checkpoint command receipt changed")
                check_receipt_payloads(campaign.root, path, allow_ctest_failure=ctest)
            path = paths[0]
            if name == "production" and payloads:
                policy_original = None
                reference = previous.get("execution_contract")
                if reference and identity_json(_bounded_payload(reference).decode("utf-8")).get("protocol") in {
                        "product-evidence-policy-v1", "product-evidence-policy-v2"}:
                    policy_original = read_policy_contract(reference)
                    if (policy_original["contract"]["facts"]["context"]["group"] != "production"
                            or _policy_candidate(campaign.root) != policy_original["contract"]["facts"]["candidate"]):
                        raise ValueError("subject_changed: synthetic OFF exact reuse has a different candidate")
                    _policy_off(policy_original["contract"]["facts"]["subject"]["off"])
                if policy_original is None:
                    production_payloads(*production_manifest_identity(path, root=campaign.root),
                                        root=campaign.root, inputs=previous["identity"]["inputs"],
                                        retained=previous.get("payloads", {}))
            if any(digest(Path(p)) != value for p, value in previous.get("payloads", {}).items()):
                raise ValueError("checkpoint result payload changed")
            if validate:
                validate(paths if len(paths) > 1 else path)
            if synthetic_quality_base is not None:
                _check_synthetic_quality(campaign.root, state_path.parent, previous, synthetic_quality_base,
                                         recipe=synthetic_quality_recipe)
            if snapshot(campaign.root, inputs) != identity["inputs"] or tool_snapshot(tools, allow_missing=True) != identity["tools"]:
                raise ValueError("checkpoint inputs changed during reuse")
            if artifact_directories and artifact_membership(artifact_directories) != identity["artifact_directories"]:
                raise ValueError("checkpoint runtime artifact membership changed during reuse")
            print("Reused checkpoint gate:", name, path, flush=True)
            return {**previous, "reused_from": str(path)}
        except BaseException as error:
            state[name] = {**previous, "status": "failed", "error": str(error)}
            write_json(state_path, state)
            raise
    if reuse is not None:
        return admit_stage_evidence(campaign, state_path, state, name, command, inputs=inputs, tools=tools,
            environment=environment, dependencies=dependencies, semantic_command=semantic_command or command,
            reuse=reuse, validate=validate)
    state[name] = {"status": "pending", "identity": identity}
    if ctest and matching:
        state[name]["ctest_chunks"] = previous.get("ctest_chunks", []).copy()
    write_json(state_path, state)
    try:
        support_launch = ({**_bounded_launch(environment), "stage_environment": identity["environment"],
                           "producer": _bounded_producer()} if name in {"tooling", "quality"} else None)
        if policy_support:
            if support_launch is None:
                raise ValueError("unsupported_protocol: policy support is only tooling or quality")
            support_launch["policy_producer"] = _policy_producer(campaign.root)
        bounded_entry = (_bounded_entry(campaign, state, name, command, identity, inputs, tools, environment,
                                       recipe=bounded_capture)
                         if bounded_capture is not None else None)
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
        if artifact_directories and artifact_membership(artifact_directories) != identity["artifact_directories"]:
            raise ValueError("checkpoint runtime artifact membership changed")
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
                    if name == "production" and payloads and not (isinstance(bounded_entry, dict) and "policy_facts" in bounded_entry)
                    else {str(p): digest(Path(p)) for p in payloads})
        result = {"status": "passed", "identity": identity, "receipt": str(paths[0]),
                  "sha256": digest(paths[0]),
                  "additional_receipts": {str(p): digest(p) for p in paths[1:]}, "payloads": retained}
        if ctest:
            result["ctest_chunks"] = state[name].get("ctest_chunks", [])
        if support_launch is not None:
            # Cross-attempt partial chains retain ordinary retry semantics, but
            # cannot supply independently whole-success transition support.
            from verify import check_receipt_payloads
            current_launch = True
            for path in paths:
                raw = check_receipt_payloads(campaign.root, path, allow_ctest_failure=ctest)
                actual_launch = {key: raw["identity"].get(key) for key in ("environment_sha256", "supervisor_sha256")}
                if actual_launch != {key: support_launch[key] for key in actual_launch}:
                    if ctest and len(paths) > 1:
                        current_launch = False
                    else:
                        raise ValueError("checkpoint support action launch differs from its declared environment/supervisor")
            if current_launch:
                result["execution_launch"] = support_launch
        if synthetic_quality_base is not None:
            evidence = _read_synthetic_quality(campaign.root, state_path.parent, result, synthetic_quality_base,
                                              recipe=synthetic_quality_recipe)
            result["payloads"] = {ref["path"]: ref["sha256"] for ref in evidence["payloads"]}
        if bounded_entry is not None:
            try:
                result["execution_contract"] = _bounded_finish(campaign, state, command, result.copy(), bounded_entry,
                                                                inputs, tools, environment)
            except (OSError, ValueError) as error:
                result["bounded_ineligible"] = str(error)
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
        paths.update(root / name for name in (
            "tests/ffi/production_contract.py",
            "tests/ffi/production-film.c.in",
            "tests/ffi/production-prepared-render.cpp",
            "tests/ofx/fixtures/exposure/manifest.json",
            f"tests/ofx/fixtures/exposure/processor-{build.name}.dat",
            "tests/print/illuminant-fixtures/manifest.json",
            "tests/print/illuminant-fixtures/product-updates.json",
            "tests/ffi/film_recipe_abi_c.c",
            "tests/ffi/grain_frame_abi_assertions.inc",
            "tests/ffi/grain_probe_contract.py",
            "tests/ffi/effects_frame_abi_assertions.inc",
            "tests/ffi/effects_frame_probe_contract.py",
        ))
    else:
        paths.update(registration_files(build, allow_missing=True, resolve=resolve))
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
                "film_recipe_abi_facts.inc", "effects_frame_test.inc", "effects_frame_abi_assertions.inc", "effects_frame_probe_contract.py", "grain_frame_test.inc", "dir_frame_test.inc", "diffusion_frame_test.inc", "halation_frame_test.inc", "frame_geometry_test.inc", "gaussian_kernel_test.inc", "film_reference_test.inc", "print_illuminant_test.inc", "print_illuminant_fixture.h", "print_preflash_test.inc", "print_balance_test.inc", "scanner_correction_test.inc", "scanner_lut_test.inc", "scanner_lut_descriptor_test.inc", "output_color_test.cpp", "output_color_abi_c.c", "scanner_color_test.inc"))
            paths.update((root / "tests/film/effects-frame-fixtures").glob("*.json"))
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
            paths.update((root / "tests/grain/frame-fixtures").glob("*.json"))
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
        cache = read_text(build / "CMakeCache.txt", encoding="utf-8")
        match = re.search(r"^CMAKE_ROOT:INTERNAL=(.+)$", cache, re.M)
        if match:
            cmake_root = resolve(match[1].strip()).resolve()
            bootstrap = {cmake_root / name: checksums for name, checksums in CMAKE_BOOTSTRAP_INPUTS.items()}
        # include() accepts arbitrary filenames. Regeneration data that cannot
        # be read safely as CMake text also keeps the preparation scope broad.
        paths.update(resolve(name) for name in configuration["regeneration_mtimes"])
    for path in paths:
        content = read_bytes(path)
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
    for registration in registration_files(build, resolve=resolve):
        for line in read_text(registration, encoding="utf-8").splitlines():
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
            if path.name == "build.rs" or re.search(r"\binclude(?:_str|_bytes)?\s*!", read_text(path, encoding="utf-8")):
                return paths
            runtime.discard(path)
        if path.name == "Cargo.toml":
            if tomllib.loads(read_text(path, encoding="utf-8")).get("package", {}).get("build"):
                return paths
        if path.name != "CMakeLists.txt" and path.suffix != ".cmake":
            continue
        content = read_text(path, encoding="utf-8")
        # A source-side configure/tool operand may be absent from Ninja. Only
        # recognize ordinary add_test syntax; anything else stays conservative.
        consumers = re.sub(r'\badd_test\s*\((?:"(?:[^"\\]|\\.)*"|[^()"])*\)', "", content, flags=re.I)
        runtime = {p for p in runtime if p.name.casefold() not in consumers.casefold()}
    return {p for p in paths if p not in runtime or p.resolve() in protected}


def checkpoint_tools(build, environment, group=None):
    tools = {Path(sys.executable)}
    if os.name == "nt":
        tools.update(p.resolve(strict=True) for p in windows_runtime_paths().values() if p is not None)
    elif group == "tooling" and environment.get("JUICER_WINDOWS_PYTHON"):
        runtime = windows_identity_interpreter(environment["JUICER_WINDOWS_PYTHON"])
        tools.update((runtime["executable"], mapped_windows_path(runtime["base_python"])))
        if runtime["config"] is not None:
            tools.add(runtime["config"])
        if environment.get("JUICER_WINDOWS_IDENTITY_PROOF"):
            tools.add(Path(environment["JUICER_WINDOWS_IDENTITY_PROOF"]))
    cache = read_text(build / "CMakeCache.txt", encoding="utf-8")
    tools.update(linker_tools(build, cache, environment))
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
        config = tomllib.loads(read_text(path))
        for name in ("rustc", "rustc-wrapper", "rustc-workspace-wrapper", "rustdoc"):
            bind(config.get("build", {}).get(name), path.parent.parent)
        for target in config.get("target", {}).values():
            bind(target.get("linker"), path.parent.parent)
    rustup = shutil.which("rustup", path=environment.get("PATH"))
    if rustup:
        for name in ("cargo", "rustc", "clippy-driver"):
            tools.add(Path(subprocess.check_output([rustup, "which", name], env=environment, text=True, timeout=None).strip()))
    return [str(p.absolute()) for p in sorted(tools)]


def qualify(root, args):
    if args is None or not args.reason.strip() or not args.selection or set(args.selection) - GROUPS:
        raise ValueError("checkpoint reason and explicit required selections are mandatory")
    if args.preset not in PRESETS or args.runtime not in {"host", "all"}:
        raise ValueError("checkpoint requires a supported native preset and explicit host/all runtime")
    build = root / "out/build" / args.preset
    reuse = _read_admission_request(root, args) if getattr(args, "reuse_declaration", None) is not None else None
    if args.preset.startswith("windows") != (os.name == "nt") and not (reuse and reuse.get("policy_recipe") == _POLICY_RECIPE):
        raise ValueError("checkpoint must execute on its named native platform")
    with OutputLease(root, build):
        return _qualify_owned(root, build, args, reuse=reuse)


def prepare_checkpoint(campaign, build, state_path, state, environment, *, repeat_reason="", timeout=None,
                       source_files=None, allow_build=False):
    """Admit completed preparation; building requires explicit authorization."""
    admitted = None
    try:
        baseline, pending = preparation_paths(build, ["all"])
        if pending.exists():
            raise ValueError("pending preparation still owns the build outputs")
        owner = read_json(baseline)
        admitted = Path(owner["producer"])
        proof = check_scoped_preparation(campaign, build, admitted, environment)["scoped_preparation"]
        from build_dependencies import file_identities
        if file_identities([Path(name) for name in proof["outputs"]]) != proof["output_identities"]:
            raise ValueError("completed preparation artifact bytes changed")
        if proof["targets"] != ["all"] or proof["configure_args"] != ["-DBUILD_TESTING=ON"]:
            raise ValueError("checkpoint requires all-target testing-ON preparation")
        required = checkpoint_inputs(campaign.root, build, "prepare", source_files=source_files,
                                     preparation=proof["context"])
        missing = [name for name in required if not name.startswith("out/build/")
                   and str((campaign.root / name).resolve()) not in proof["context"]["files"]]
        if missing:
            raise ValueError("preparation input membership is incomplete: " + ", ".join(missing[:5]))
    except (OSError, ValueError, KeyError) as error:
        admitted = None
        if not allow_build:
            raise ValueError(f"preparation admission failed without changing outputs: {error}; "
                             f"inspect the difference, then explicitly run verify.py prepare --preset {build.name} "
                             "--targets all --testing ON if a build is needed") from error
    inputs = checkpoint_inputs(campaign.root, build, "prepare", source_files=source_files)
    outputs = []
    prepared_context = None
    admission_observations = {}
    previous = state.get("prepare")
    def prepare():
        if admitted is not None:
            return admitted
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


def _preparation_facts_identity(facts, stage):
    return hashlib.sha256(json.dumps({"preparation": facts, "checkpoint": stage["identity"],
                                     "artifacts": stage["payloads"]}, sort_keys=True).encode()).hexdigest()


def preparation_identity(build, stage, *, resolve=Path, native_file_identities=None):
    """Bind admitted artifact facts; producer provenance remains in the original receipt.

    Callers validate raw command/process evidence through run_stage or collection.
    The artifact owner and exact output bytes must still match at consumption.
    """
    path = resolve(stage["receipt"])
    if digest(path) != stage["sha256"]:
        raise ValueError("checkpoint preparation receipt changed")
    record = check_preparation_owner(build, stage["receipt"], resolve=resolve)
    if record.get("status") != "passed" or record.get("returncode") != 0 or not record.get("process_tree", {}).get("drained"):
        raise ValueError("checkpoint preparation lacks successful process completion")
    proof = record["scoped_preparation"]
    check_preparation_inputs(build, proof, resolve=resolve, native_file_identities=native_file_identities)
    artifacts = stage.get("payloads", {})
    if not artifacts or set(artifacts) != set(proof["outputs"]):
        raise ValueError("checkpoint preparation lacks exact artifact identities; repeat preparation")
    if any(digest(resolve(p)) != checksum for p, checksum in artifacts.items()):
        raise ValueError("checkpoint preparation artifact bytes changed")
    facts = {name: value for name, value in proof.items()
             if name not in {"producer", "invalidated", "rust_invalidated_packages"}}
    return _preparation_facts_identity(facts, stage)


def _qualify_owned(root, build, args, *, reuse=None):
    if reuse is not None and reuse.get("policy_recipe") == _POLICY_RECIPE:
        return _qualify_policy_owned(root, build, args, reuse)
    from verify import source_snapshot
    cache = read_text(build / "CMakeCache.txt")
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
    state = read_json(state_path) if state_path.exists() else {}
    current = source_snapshot(root)
    record = {"mode": "checkpoint", "status": "running", "scope_status": "running", "reason": args.reason,
              "preset": args.preset, "runtime": args.runtime, "selection": sorted(groups), "required": required,
              "source": current, "stages": {}}
    if reuse is not None:
        record.update(reuse_declaration=reuse["reference"], assurance="synthetic-c11-cli-quality", stage_provenance={})
    receipt = campaign.directory / "qualification.json"
    if receipt.exists():
        raise ValueError("checkpoint attempt already exists; retain it and use a new attempt name")
    started = time.monotonic()
    environment = dict(os.environ)
    def run(name, command, inputs, *, tools=None, **kwargs):
        if tools is None:
            tools = checkpoint_tools(build, environment, name if name in GROUPS else "product")
        result = run_stage(campaign, state_path, state, name, command, inputs=inputs, tools=tools,
                       environment=environment, timeout=None, repeat_reason=args.repeat_reason, **kwargs)
        record["stages"][name] = result
        if reuse is not None:
            record["stage_provenance"][name] = ("admitted" if result.get("reuse_admission") else
                "exact-reused" if result.get("reused_from") or not Path(result["receipt"]).is_relative_to(
                    campaign.directory / "attempts") else "executed")
        return result
    try:
        if reuse is not None:
            _check_original_stage_owner(state_path, state, reuse)
            _validate_reuse_support(campaign, state_path, state, reuse, environment)
            if not isinstance(state.get("prepare"), dict) or state["prepare"].get("status") != "passed":
                raise ValueError("owner_conflict: current preparation stage is pending, failed or blocked")
        built = None
        prepared_identity = None
        if "product" in groups:
            built = prepare_checkpoint(campaign, build, state_path, state, environment,
                                       repeat_reason=args.repeat_reason, source_files=current,
                                       allow_build=False if reuse is not None else getattr(args, "prepare", False))
            record["stages"]["prepare"] = built
            if reuse is not None:
                record["stage_provenance"]["prepare"] = ("exact-reused" if built.get("reused_from")
                    or not Path(built["receipt"]).is_relative_to(campaign.directory / "attempts") else "executed")
            prepared_identity = preparation_identity(build, built)
            inputs = checkpoint_inputs(root, build, "product", source_files=current)
            run("quality", [sys.executable, "-B", "scripts/check-quality.py", "--preset", args.preset,
                "--qualification", "--base", args.base], inputs, dependencies=[prepared_identity],
                synthetic_quality_base=args.base if reuse is not None else None)
        if groups != {"production"}:
            query = subprocess.check_output(["ctest", "--test-dir", str(build), "--show-only=json-v1"], cwd=root, env=environment, text=True, timeout=None)
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
                commands, payloads = [], []
                for role, members in parts.items():
                    junit = campaign.directory / (group + "-" + role + ".xml")
                    pattern = "^(" + "|".join(re.escape(n) for n in members) + ")$"
                    commands.append([*ctest_command(root, build), "-R", pattern,
                        "-LE" if role == "host" else "-L", "gpu", "--no-tests=error",
                        "--output-on-failure", "--output-junit", str(junit)])
                    payloads.append(junit)
                identities = [ctest_identity(command) for command in commands]
                def validate(paths, parts=parts, names=names, identities=identities, group=group):
                    validate_ctest_stage(paths, parts, names, identities)
                    if group == "product":
                        check_scoped_preparation(campaign, build, Path(built["receipt"]))
                inputs = checkpoint_inputs(root, build, group, source_files=current)
                tools = checkpoint_tools(build, environment, group)
                if reuse is not None and group == "product":
                    tools = _bounded_cli_tools(build, environment)
                artifacts, directories = ctest_artifacts(inventory, names, build) if group != "product" else ([], [])
                tools = sorted(set(tools) | set(artifacts))
                run(group, commands[0], inputs, tools=tools, semantic_command=identities,
                    action=lambda group=group, commands=commands, parts=parts, inputs=inputs, tools=tools:
                        run_ctest_parts(campaign, group, commands, parts, build=build, inputs=inputs, tools=tools,
                            environment=environment, timeout=None, repeat_reason=args.repeat_reason,
                            state=state, state_path=state_path, payloads=payloads),
                    validate=validate, payloads=payloads,
                    ctest=True, artifact_directories=directories,
                    dependencies=[prepared_identity] if built and group == "product" else [],
                    reuse=reuse if group == "product" else None)
        else:
            cache = read_text(build / "CMakeCache.txt")
            if not re.search(r"^BUILD_TESTING:BOOL=OFF$", cache, re.M):
                raise ValueError("production isolation requires a separately prepared testing-OFF Release tree")
            from build_dependencies import NinjaGraph
            graph = NinjaGraph(build, ["juicer", "JuicerProductionPreparedRenderProbe"], environment)
            executable = build / "tests/bin" / (
                "JuicerProductionPreparedRenderProbe.exe" if os.name == "nt"
                else "JuicerProductionPreparedRenderProbe")
            if str(executable.resolve()) not in graph.outputs or not executable.is_file():
                raise ValueError("prepared-render production executable is missing from the target graph")
            artifacts = sorted({p for n in graph.outputs
                                if (p := Path(n)).is_file() and p.suffix.lower() in {".a", ".lib", ".ofx", ".pdb"}}
                               | {executable})
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
        if reuse is not None:
            terminal_support = _validate_reuse_support(campaign, state_path, state, reuse, environment,
                                                        record["stages"]["quality"])
            _check_original_stage_owner(state_path, state, reuse)
            admission_ref = record["stages"]["product"].get("reuse_admission")
            if admission_ref:
                admitted = identity_json(_bounded_payload(admission_ref).decode("utf-8"))
                read_execution_contract(reuse["original"]["reference"])
                terminal_subject = capture_execution_subject(campaign, state, commands[0], inputs=inputs, tools=tools,
                    environment=environment, recipe="native-c11-cli-miniature-v1")
                if (admitted["current_candidate"] != terminal_subject["candidate"]
                        or admitted["current_preparation"] != terminal_subject["preparation"]
                        or admitted["support"] != terminal_support
                        or admitted["comparison"] != compare_execution_subject(reuse["original"], terminal_subject,
                                                                               reuse["declaration"])):
                    raise ValueError("terminal_changed: committed admission no longer supports wrapper completion")
                # The final subject scan is another external-read boundary.
                # Seal its already declared evidence closure before success.
                if (_validate_reuse_support(campaign, state_path, state, reuse, environment,
                                             record["stages"]["quality"]) != terminal_support
                        or read_execution_contract(reuse["original"]["reference"]) != reuse["original"]
                        or identity_json(_bounded_payload(admission_ref).decode("utf-8")) != admitted):
                    raise ValueError("terminal_changed: evidence changed during final wrapper scan")
                completion = Path(admission_ref["path"]).with_name("completion.json")
                if completion.exists() or completion.is_symlink():
                    _read_admission_completion(admission_ref, build)
            _check_stage_owner(state_path, state, "product", record["stages"]["product"])
            _check_stage_owner(state_path, state, "prepare", record["stages"]["prepare"])
            _check_stage_owner(state_path, state, "quality", record["stages"]["quality"])
            OutputLease.check_held()
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
    if reuse is not None and record["scope_status"] == "passed":
        admission = record["stages"]["product"].get("reuse_admission")
        if admission:
            completion = Path(admission["path"]).with_name("completion.json")
            completed = {"admission": admission, "wrapper": _bounded_reference(receipt)}
            if not completion.exists():
                _write_admission_record(completion, completed)
            else:
                _read_admission_completion(admission, build)
    return receipt


def native_runtime_binding(expected, name):
    bindings = [groups["files"].get(name) for groups in expected.values()]
    if not bindings or any(value is None or value != bindings[0] for value in bindings):
        raise ValueError("Windows runtime tool is not bound by every preparation proof")
    return bindings[0]


def native_identity_result(root, entry, expected):
    if (set(entry) != {"phase", "preparations", "receipt", "sha256", "payloads"}
            or entry["phase"] not in {"entry", "terminal"}
            or entry["preparations"] != sorted(expected)):
        raise ValueError("malformed native identity evidence")
    path = Path(entry["receipt"])
    if digest(path) != entry["sha256"]:
        raise ValueError("native identity command receipt changed")
    record = read_json(path)
    native = record["identity"]["windows_identity"]
    request_path = root / native["request"]
    request = read_identity_request(request_path, native["request_sha256"])
    base = native_runtime_binding(expected, native["base_python"])
    launch = prepare_windows_identity_launch(root, native["python"], request_path, base_python=base["resolved"])
    phase = entry["phase"]
    if (request["phase"] != phase or record["stage"] != "windows-" + phase
            or request["proofs"] != {p: {g: list(v) for g, v in groups.items()}
                                       for p, groups in expected.items()}
            or native != launch or record["identity"]["command"] != launch["command"]
            or record.get("status") != "passed" or record.get("returncode") != 0):
        raise ValueError("native identity command/request does not bind the collection")
    tree = record["process_tree"]
    process_path = Path(tree["receipt"])
    process = read_json(process_path)
    correlation = {"collection_id": request["collection_id"], "phase": phase,
                   "request_sha256": native["request_sha256"]}
    if (tree.get("drained") is not True or process.get("drained") is not True
            or process.get("status") != "completed" or process.get("returncode") != 0
            or process.get("identity_request") != correlation
            or process.get("command") != launch["command"]
            or set(process.get("creation", {})) != {"windows_creation_time_100ns"}
            or type(process["creation"]["windows_creation_time_100ns"]) is not int
            or process["creation"]["windows_creation_time_100ns"] <= 0
            or any(type(process.get(k)) is not int or process[k] <= 0
                   for k in ("supervisor_pid", "command_pid", "gated_pid"))
            or type(tree.get("interop_pid")) is not int or tree["interop_pid"] <= 0
            or any(tree.get(k) != v for k, v in process.items())):
        raise ValueError("native identity process completion is invalid")
    output = root / record["output"]
    required = {str(path), str(output), str(process_path),
                str(process_path.with_suffix(".gate.json")),
                str(process_path.with_suffix(".seal.json")), str(request_path)}
    required.update(command_result_payloads(path))
    if set(entry["payloads"]) != required:
        raise ValueError("native identity payload membership changed")
    if any(digest(Path(p)) != value for p, value in entry["payloads"].items()):
        raise ValueError("native identity payload changed")
    if int(read_text(process_path.with_suffix(".gate.json"))) != process["gated_pid"]:
        raise ValueError("native identity gated process changed")
    if read_json(process_path.with_suffix(".seal.json")) != {
            "receipt_sha256": digest(process_path), "identity_request": correlation}:
        raise ValueError("native identity terminal process seal changed")
    if (digest(output) != record["output_sha256"]
            or snapshot(root, list(record["identity"]["inputs"])) != record["identity"]["inputs"]
            or tool_snapshot(list(record["identity"]["tools"])) != record["identity"]["tools"]
            or digest(Path(launch["executable"])) != record["identity"]["executable_sha256"]
            or digest(root / "scripts/process_tree.py") != record["identity"]["supervisor_sha256"]):
        raise ValueError("native identity input/tool evidence changed")
    response = identity_json(read_text(output))
    response_keys = {"protocol", "collection_id", "phase", "request_sha256", "runtime", "proofs"}
    if request["protocol"] == "windows-file-identities-v2":
        response_keys.add("file_observations")
    if (not isinstance(response, dict)
            or set(response) != response_keys
            or any(response[k] != request[k] for k in ("protocol", "collection_id", "phase"))
            or response["request_sha256"] != native["request_sha256"]
            or response["proofs"] != expected
            or not isinstance(response["runtime"], dict)
            or set(response["runtime"]) != {"python", "base_python"}):
        raise ValueError("native identity result does not match original proof maps")
    if "file_observations" in response:
        counts = response["file_observations"]
        if (not isinstance(counts, dict) or set(counts) != {
                "content_bytes", "content_reads", "transcript_decoded_bytes"}
                or any(type(value) is not int or value < 0 for value in counts.values())):
            raise ValueError("native identity read observations are malformed")
    for role, runtime in response["runtime"].items():
        binding = native_runtime_binding(expected, launch[role])
        if (not isinstance(runtime, dict) or set(runtime) != {"path", "resolved", "sha256"}
                or runtime["path"] != launch[role]
                or runtime["resolved"] != binding["resolved"] or runtime["sha256"] != binding["sha256"]
                or digest(mapped_windows_path(runtime["path"])) != runtime["sha256"]
                or digest(mapped_windows_path(runtime["resolved"])) != runtime["sha256"]):
            raise ValueError("native identity interpreter binding changed")
    return response


def check_native_identity_evidence(root, entries, expected):
    if not isinstance(entries, list) or [e.get("phase") for e in entries] != ["entry", "terminal"]:
        raise ValueError("native identity entry/terminal evidence is incomplete")
    first, last = [native_identity_result(root, entry, expected) for entry in entries]
    if (first["collection_id"] != last["collection_id"] or first["proofs"] != last["proofs"]
            or first["runtime"] != last["runtime"]):
        raise ValueError("native identity terminal target/tool facts changed")


def run_native_identity_phase(root, native, phase):
    campaign = native["campaign"]
    request = {"protocol": "windows-file-identities-v2", "collection_id": native["collection_id"],
               "phase": phase, "proofs": {p: {g: list(v) for g, v in groups.items()}
                                         for p, groups in native["expected"].items()}}
    request_path = campaign.directory / ("windows-" + phase + "-request.json")
    with request_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(request) + "\n")
    launch = prepare_windows_identity_launch(root, native["python"], request_path)
    base = native_runtime_binding(native["expected"], launch["base_python"])
    launch = prepare_windows_identity_launch(root, native["python"], request_path, base_python=base["resolved"])
    inputs = [request_path.relative_to(root).as_posix(), "scripts/execution.py",
              "scripts/process_tree.py", "scripts/build_dependencies.py", *native["receipts"]]
    path = campaign.run("windows-" + phase, launch["command"], inputs=inputs,
        tools=launch["tools"], windows_python=native["python"], identity_request=request_path,
        windows_base_python=base["resolved"], expected_inputs=native["receipts"])
    raw = read_json(path)
    process = Path(raw["process_tree"]["receipt"])
    payloads = [path, root / raw["output"], process, process.with_suffix(".gate.json"),
                process.with_suffix(".seal.json"), request_path]
    payloads.extend(Path(name) for name in command_result_payloads(path))
    entry = {"phase": phase, "preparations": sorted(native["expected"]), "receipt": str(path),
             "sha256": digest(path), "payloads": {str(p): digest(p) for p in payloads}}
    response = native_identity_result(root, entry, native["expected"])
    native["stages"].append(entry)
    return response["proofs"]


def prepare_native_collection(root, records, windows_python, resolve):
    expected, receipts, interpreters = {}, {}, []
    for record in records:
        if "product" not in record["selection"]:
            continue
        stage = record["stages"]["prepare"]
        path = resolve(stage["receipt"])
        if digest(path) != stage["sha256"]:
            raise ValueError("checkpoint preparation receipt changed")
        proof = read_json(stage["receipt"], resolve=resolve).get("scoped_preparation")
        if not proof or not PureWindowsPath(proof["source"]).is_absolute():
            continue
        if os.name != "posix":
            continue
        if not windows_python:
            raise ValueError("Windows preparation collection requires explicit --windows-python")
        selected = str(PureWindowsPath(windows_python))
        qualified = record["stages"]["quality"]["identity"]
        declared = qualified["tools"]
        if selected not in qualified["tool_paths"]:
            raise ValueError("Windows identity interpreter does not match qualified tool identities")
        runtime = windows_identity_interpreter(selected)
        roles = [runtime["python"], runtime["base_python"]]
        if runtime["config"] is not None:
            roles.append(windows_path(runtime["config"]))
        for name in roles:
            binding = proof["context"]["files"].get(name)
            if (binding is None or declared.get(binding["resolved"]) != binding["sha256"]
                    or digest(mapped_windows_path(name)) != binding["sha256"]):
                raise ValueError("Windows runtime tool/config is not bound by qualified native proof")
        interpreters.append(selected)
        expected[stage["sha256"]] = {group: proof["context"][group] for group in ("files", "rust_configuration")}
        receipts[path.relative_to(root).as_posix()] = stage["sha256"]
    if not expected:
        if windows_python is not None:
            raise ValueError("Windows identity operand has no foreign preparation consumer")
        return None
    subjects = policy_native_collection_bindings(root, records, resolve)
    if set(subjects) & set(expected):
        raise ValueError("Windows policy subject proof conflicts with a preparation identity")
    expected.update(subjects)
    if len(set(interpreters)) != 1:
        raise ValueError("Windows preparation interpreters differ")
    collection_id = uuid.uuid4().hex
    return {"collection_id": collection_id, "python": interpreters[0], "expected": expected,
            "receipts": receipts, "stages": [],
            "campaign": Campaign(root, "linux-debug", "collection-" + collection_id,
                                 ["windows-entry", "windows-terminal"])}


def _collect_cli_admission(root, build, wrapper, name, result, state):
    """Revalidate the fixed CLI original without executing its CTest command."""
    from types import SimpleNamespace
    from build_evidence import capture_bounded_dependencies

    reference = result["reuse_admission"]
    admitted = _read_committed_admission(reference, build)
    completed = _read_admission_completion(reference, build)
    completion = _bounded_reference(Path(reference["path"]).with_name("completion.json"))
    declaration = read_reuse_declaration(admitted["declaration"])
    args = SimpleNamespace(reuse_declaration=admitted["declaration"]["path"], selection=["product"],
                           runtime="host", preset=build.name, base=declaration["fresh_evidence"]["quality_base"])
    reuse = _read_admission_request(root, args)
    retained_stages = {"prepare": admitted["current_preparation"]["stage"],
                       "quality": admitted["support"]["quality_stage"],
                       "product": {**admitted["owner"], "reuse_admission": reference}}
    supplied_stages = {key: {field: value for field, value in stage.items() if field != "reused_from"}
                       for key, stage in wrapper["stages"].items()}
    if (name != "product" or wrapper.get("reuse_declaration") != reuse["reference"]
            or wrapper.get("stage_provenance", {}).get(name) != "admitted"
            or wrapper.get("assurance") != "synthetic-c11-cli-quality"
            or wrapper.get("preset") != build.name or wrapper.get("runtime") != "host"
            or wrapper.get("runtime") != reuse["original"]["contract"]["context"]["runtime"]
            or wrapper.get("selection") != ["product"] or supplied_stages != retained_stages
            or wrapper.get("membership") != {"product": ["Bounded.Read"]}
            or wrapper.get("runtime_membership") != {"product": {"host": ["Bounded.Read"]}}
            or wrapper.get("stage_provenance", {}).get("prepare") != "exact-reused"
            or wrapper.get("stage_provenance", {}).get("quality") not in {"executed", "exact-reused"}
            or set(wrapper.get("stage_provenance", {})) != set(retained_stages)
            or admitted["validator"] != reuse["validator"]
            or {key: value for key, value in result.items() if key not in {"reuse_admission", "reused_from"}}
                != admitted["owner"]):
        raise ValueError("owner_conflict: wrapper does not bind its completed direct admission")
    _check_original_stage_owner(build / "verification-checkpoint.json", state, reuse)
    support = _validate_reuse_support(SimpleNamespace(root=root), build / "verification-checkpoint.json",
                                      state, reuse, dict(os.environ), state["quality"],
                                      collection=True, recorded_support=admitted["support"])
    if support != admitted["support"]:
        raise ValueError("terminal_changed: current independent admission support changed")
    original = reuse["original"]["contract"]
    dependencies = original["subject"]["dependencies"]
    observed = capture_bounded_dependencies(root, files=sorted(dependencies["files"]),
        directories=sorted(dependencies["directories"]), absent=sorted(dependencies["absent"]),
        executables=sorted(path for path, row in dependencies["files"].items() if row["executable"]))
    prepared = state["prepare"]
    identity = preparation_identity(build, prepared)
    raw = identity_json(_bounded_payload({"path": prepared["receipt"], "sha256": prepared["sha256"]}).decode("utf-8"))
    proof = {key: value for key, value in raw["scoped_preparation"].items()
             if key not in {"producer", "invalidated", "rust_invalidated_packages"}}
    actual_build = original["subject"]["build_subject"]
    context = proof["context"]
    if (any(context.get(key) != value for key, value in actual_build.items() if key != "files")
            or any(context["files"].get(path) != row for path, row in actual_build["files"].items())):
        raise ValueError("preparation_changed: current preparation no longer binds the original build subject")
    current_identity = {**result["identity"],
        "inputs": snapshot(root, checkpoint_inputs(root, build, name)), "dependencies": [identity]}
    current = {"policy": original["policy"], "candidate": _bounded_candidate(root, recipe="native-c11-cli-miniature-v1"),
        "context": original["context"], "controller_paths": original["controller_paths"],
        "subject": {**original["subject"], "dependencies": observed, "preparation_facts": proof},
        "preparation": {"identity": identity, "stage": prepared}, "stage_identity": current_identity}
    comparison = compare_execution_subject(reuse["original"], current, declaration)
    if (comparison != admitted["comparison"] or current["preparation"] != admitted["current_preparation"]
            or current["candidate"] != admitted["current_candidate"]):
        raise ValueError("terminal_changed: admission no longer corresponds to current preparation and subject")
    references = [reference, admitted["declaration"], completion, completed["wrapper"], reuse["original"]["reference"],
                  declaration["review"], declaration["owner_decision"], *declaration["fresh_evidence"]["tooling"],
                  *declaration["fresh_evidence"]["quality"]]
    references.extend(value for value in original["execution"].values() if isinstance(value, dict))
    for retained in (original["preparation"]["stage"], prepared, support["tooling_stage"], support["quality_stage"]):
        for receipt in stage_receipts(retained):
            references.append(_bounded_reference(receipt))
            raw_retained = read_json(receipt)
            references.extend((_bounded_reference(raw_retained["process_tree"]["receipt"]),
                               _bounded_reference(root / raw_retained["output"])))
        references.extend({"path": path, "sha256": checksum} for path, checksum in retained.get("payloads", {}).items())
    references.extend({"path": str(Path(reuse["validator"]["root"]) / path), "sha256": checksum}
                      for path, checksum in reuse["validator"]["files"].items())
    return {"current_identity": current_identity, "original": reuse["original"]["reference"],
            "admission": reference, "assurance": admitted["assurance"],
            "completion": completion, "completion_wrapper": completed["wrapper"],
            "references": references,
            "execution_candidate": original["candidate"], "current_candidate": current["candidate"]}


def validate_qualification_membership(records):
    """Check policy membership after each selected stage's evidence is validated."""
    required = {(preset, group) for preset in PRESETS for group in ("product", "boundary")}
    required |= {(preset, "tooling") for preset in ("linux-debug", "windows-clang-debug")}
    required |= {(preset, "production") for preset in ("linux-release", "windows-clang-release")}
    required.add(("linux-debug", "corpus"))
    actual = [(record["preset"], group) for record in records for group in record["selection"]]
    contexts = [(record["preset"], "OFF" if "production" in record["selection"] else "ON") for record in records]
    expected_contexts = {(preset, "ON") for preset in PRESETS} | {
        (preset, "OFF") for preset in ("linux-release", "windows-clang-release")}
    corpus = [preset for preset, group in actual if group == "corpus"]
    if (len(actual) != 13 or len(set(actual)) != 13 or set(actual) != required
            or len(contexts) != 6 or len(set(contexts)) != 6 or set(contexts) != expected_contexts):
        if (len(actual) == len(set(actual)) == 13 and len(corpus) == 1
                and {role for role in actual if role[1] != "corpus"} == required - {("linux-debug", "corpus")}
                and len(contexts) == len(set(contexts)) == 6 and set(contexts) == expected_contexts):
            raise ValueError("corpus evidence is required only on Linux Debug")
        raise ValueError("full checkpoint membership is incomplete or duplicated")
    if any(record["selection"] != ["production"] for record in records if "production" in record["selection"]):
        raise ValueError("production isolation requires a separate OFF context")
    if any(record["runtime"] != "all" for record in records if "product" in record["selection"]):
        raise ValueError("host-only product checks do not establish full qualification")


def _collection_tool_key(path, *, resolve, native_maps, bindings, tools, expected_target=None):
    """Bind foreign lexical tools through native facts into the local seals."""
    local = str(resolve(path))
    target = bindings[local]
    checksum = tools[target]
    if os.name != "posix" or not PureWindowsPath(path).is_absolute() or checksum is None:
        return target
    rows = [groups["files"][path] for groups in native_maps.values() if path in groups["files"]]
    # Ordinary preparation maps may not observe this tool. Preserve their local
    # comparison; policy dependencies separately require complete native rows.
    if not rows:
        return target
    if (any(row != rows[0] for row in rows)
            or set(rows[0]) != {"resolved", "sha256"}):
        raise ValueError("checkpoint Windows tool lacks a consistent fresh native identity")
    native = rows[0]
    if expected_target is not None and native["resolved"] != expected_target:
        raise ValueError("checkpoint Windows tool target changed")
    canonical = str(resolve(native["resolved"]))
    if (canonical not in bindings or native["sha256"] != checksum
            or tools[bindings[canonical]] != checksum):
        raise ValueError("checkpoint Windows tool target/hash is outside the current local seals")
    return bindings[canonical]


def collect_qualification(root, paths, *, windows_python=None):
    """Full phase/release aggregation is explicit; scoped passes cannot stand in."""
    if not paths:
        raise ValueError("all native/profile checkpoint receipts are required")
    from contextlib import ExitStack
    with ExitStack() as leases:
        for preset in PRESETS:
            leases.enter_context(OutputLease(root, root / "out/build" / preset))
        return _collect_owned(root, paths, windows_python=windows_python)


def _collect_owned(root, paths, *, windows_python=None):
    def local(name):
        normalized = str(name).replace("\\", "/")
        if os.name != "nt" and PureWindowsPath(name).is_absolute():
            return mapped_windows_path(name)
        if Path(name).is_absolute():
            return Path(name)
        if "/out/" in normalized:
            return root / "out" / normalized.split("/out/", 1)[1]
        return Path(normalized)
    with FileObservations() as initial_reads:
        owners, evidence = {}, {}

        def retain(path, raw=None):
            path = Path(path)
            binding = str(path.resolve(strict=True))
            checksum = hashlib.sha256(raw).hexdigest() if raw is not None else digest(path)
            facts = (binding, checksum)
            if (str(path.resolve(strict=True)) != binding
                    or (path in evidence and evidence[path] != facts)):
                raise ValueError("checkpoint immutable evidence changed during collection")
            evidence[path] = facts

        def retain_command_maps(logical_receipt):
            for blob, checksum in command_result_payloads(logical_receipt, resolve=local).items():
                blob_path = local(blob)
                retain(blob_path)
                if evidence[blob_path][1] != checksum:
                    raise ValueError("checkpoint command identity-map blob changed")

        records = []
        for path in paths:
            raw = read_bytes(path)
            retain(path, raw)
            records.append(identity_json(raw.decode("utf-8")))
        for record in records:
            require_checkpoint(record)
            if record.get("reuse_declaration") and not any(stage.get("reuse_admission") for stage in record["stages"].values()):
                raise ValueError("owner_conflict: admission-bearing wrapper has no completed admission")
            provenance = record.get("stage_provenance")
            if provenance is not None and (not isinstance(provenance, dict)
                    or set(provenance) != set(record["stages"])):
                raise ValueError("checkpoint provenance does not bind every retained stage")
        from verify import source_snapshot
        observed_source = source_snapshot(root) if all("source" in record for record in records) else None
        if observed_source is not None and any(record["source"] != observed_source for record in records):
            raise ValueError("checkpoint records do not bind one current candidate")
        for record in records:
            owner = root / "out/build" / record["preset"] / "verification-checkpoint.json"
            raw = read_bytes(owner)
            retain(owner, raw)
            state = read_json(owner)
            owners[owner] = state
            if any(value.get("status") != "passed" for value in state.values()):
                raise ValueError("pending checkpoint work prevents aggregate collection")
        native = prepare_native_collection(root, records, windows_python, local)
        native_maps = run_native_identity_phase(root, native, "entry") if native else {}
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
        members, preparations, preparation_ids, artifact_seals = {}, {}, {}, []
        admissions, provenance, candidates, transitions = [], [], {}, {}
        for wrapper_path, record in zip(paths, records, strict=True):
            owner = root / "out/build" / record["preset"] / "verification-checkpoint.json"
            retain(owner)
            state = read_json(owner)
            if owner in owners and owners[owner] != state:
                raise ValueError("pending or newer checkpoint ownership prevents collection")
            owners[owner] = state
            if any(value.get("status") != "passed" for value in state.values()):
                raise ValueError("pending checkpoint work prevents aggregate collection")
            for name, result in record["stages"].items():
                current = state.get(name, {})
                if (current.get("status") != "passed"
                        or current != {key: value for key, value in result.items() if key != "reused_from"}):
                    raise ValueError("pending or newer checkpoint ownership prevents collection: " + name)
                policy = policy_collection_stage(root, owner.parent, record, name, result, state,
                                                 resolve=local, native_maps=native_maps, wrapper_path=wrapper_path)
                admitted = policy
                if result.get("reuse_admission") and admitted is None:
                    admitted = _collect_cli_admission(root, owner.parent, record, name, result, state)
                if admitted is not None:
                    for reference in admitted["references"]:
                        path = local(reference["path"])
                        retain(path)
                        if evidence[path][1] != reference["sha256"]:
                            raise ValueError("terminal_changed: admission immutable reference changed")
                    logical = PureWindowsPath(reference["path"]) if PureWindowsPath(reference["path"]).is_absolute() else Path(reference["path"])
                    if logical.name == "result.json" and logical.parent.parent.name == "attempts":
                        retain_command_maps(reference["path"])
                    admissions.append((record, name, result, state, admitted, wrapper_path))
                elif record.get("reuse_declaration"):
                    if record.get("stage_provenance", {}).get(name) not in {"executed", "exact-reused"}:
                        raise ValueError("owner_conflict: incomplete admission-bearing wrapper")
                checked_identity = admitted["current_identity"] if admitted is not None else result["identity"]
                wrapper_path = Path(wrapper_path)
                stage_provenance = ("admitted" if result.get("reuse_admission") else "exact-reused"
                    if result.get("reused_from") or not local(result["receipt"]).is_relative_to(wrapper_path.parent / "attempts")
                    else "executed")
                if record.get("stage_provenance", {}).get(name, stage_provenance) != stage_provenance:
                    raise ValueError("checkpoint provenance disagrees with retained execution or admission")
                context = {}
                if admitted is not None:
                    if admitted.get("context") is not None:
                        context["context"] = admitted["context"]
                    for kind in ("execution_candidate", "current_candidate"):
                        candidate = admitted.get(kind)
                        if candidate is not None:
                            key = _bounded_hash(candidate)
                            candidates[key] = candidate
                            context[kind] = key
                    transition = admitted.get("producer_transition")
                    if transition is not None:
                        key = _bounded_hash(transition)
                        transitions[key] = transition
                        context["producer_transition"] = key
                provenance.append({"preset": record["preset"], "group": name,
                    "provenance": stage_provenance,
                    "execution": {"path": result["receipt"], "sha256": result["sha256"]},
                    **context,
                    **({"original": admitted["original"], "admission": admitted["admission"]}
                       if admitted is not None and result.get("reuse_admission") else {})})
                directories = result["identity"].get("artifact_directories", {})
                expected_directories = {str(local(directory)): sorted(str(local(path)) for path in paths) if paths is not None else None
                                        for directory, paths in directories.items()}
                if artifact_membership(expected_directories) != expected_directories:
                    raise ValueError("checkpoint runtime artifact membership changed")
                artifact_seals.append(expected_directories)
                raw_paths = stage_receipts(result, local)
                logical_paths = [result["receipt"], *result.get("additional_receipts", {})]
                for logical_path, path, expected in zip(
                        logical_paths, raw_paths,
                        [result["sha256"], *result.get("additional_receipts", {}).values()], strict=True):
                    retain(path)
                    if digest(path) != expected:
                        raise ValueError("checkpoint command receipt changed")
                    command = read_json(logical_path, resolve=local)
                    retain_command_maps(logical_path)
                    process_path = local(command["process_tree"]["receipt"])
                    retain(process_path)
                    retain(root / command["output"])
                    process = read_json(command["process_tree"]["receipt"], resolve=local)
                    ctest_failure = (name in record.get("membership", {}) and command.get("status") == "failed"
                                     and command.get("returncode") == 8 and process.get("status") == "completed")
                    if (not ctest_failure and (command.get("status") != "passed" or command.get("returncode") != 0)
                            or not process.get("drained") or any(command["process_tree"].get(k) != v for k, v in process.items())
                            or digest(root / command["output"]) != command["output_sha256"]):
                        raise ValueError("checkpoint terminal evidence is invalid")
                if name == "production" and policy is None:
                    production_payloads(*production_manifest_identity(raw_paths[0], root=root, resolve=local),
                                        root=root, inputs=result["identity"]["inputs"],
                                        resolve=local, retained=result.get("payloads", {}))
                for path, expected in result.get("payloads", {}).items():
                    retain(local(path))
                    if digest(local(path)) != expected:
                        raise ValueError("checkpoint result payload changed")
                if name in record.get("membership", {}):
                    validate_ctest_stage(raw_paths, record.get("runtime_membership", {}).get(name, {}),
                        record["membership"][name], result["identity"]["command"], local)
                declared = checked_identity["inputs"]
                source_inputs = {path: checksum for path, checksum in declared.items() if not path.startswith("out/build/")}
                if {p: inputs[p] for p in source_inputs} != source_inputs:
                    raise ValueError("checkpoint declared source/data inputs changed")
                group = name if name in GROUPS | {"prepare"} else "product"
                if name == "prepare":
                    preparation_ids[record["preset"]] = preparation_identity(
                        root / "out/build" / record["preset"], result, resolve=local,
                        native_file_identities=native_maps.get(result["sha256"]))
                    preparations[record["preset"]] = read_json(result["receipt"], resolve=local)["scoped_preparation"]["context"]
                expected_members = {path for path in declared if not path.startswith("out/build/")}
                key = record["preset"], group
                if key not in members:
                    discovery = {"source_files": observed_source, "resolve": local}
                    if group == "prepare":
                        discovery.update(preparation=preparations[record["preset"]], resolve=local)
                    members[key] = {path for path in checkpoint_inputs(root, root / "out/build" / record["preset"], group,
                                                                      **discovery)
                                    if not path.startswith("out/build/")}
                actual_members = members[key]
                if expected_members != actual_members:
                    raise ValueError("checkpoint source/data membership changed")
                expected_tools = {_collection_tool_key(path, resolve=local, native_maps=native_maps,
                                  bindings=bindings, tools=tools, expected_target=path): value
                                  for path, value in checked_identity["tools"].items()}
                actual_tools = {_collection_tool_key(path, resolve=local, native_maps=native_maps,
                                bindings=bindings, tools=tools): tools[bindings[str(local(path))]]
                                for path in result["identity"]["tool_paths"]}
                if actual_tools != expected_tools:
                    raise ValueError("checkpoint toolchain/configuration changed")

            if "product" in record["selection"]:
                identity = preparation_ids[record["preset"]]
                if (record["stages"]["quality"]["identity"]["dependencies"] != [identity]
                        or (not record["stages"]["product"].get("reuse_admission")
                            and record["stages"]["product"]["identity"]["dependencies"] != [identity])):
                    raise ValueError("checkpoint stages do not bind the current preparation artifacts")
    terminal_native_maps = run_native_identity_phase(root, native, "terminal") if native else {}
    with FileObservations() as semantic_reads:
        policy_collection_chronology([entry[4] for entry in admissions])
        if native:
            check_native_identity_evidence(root, native["stages"], native["expected"])
        for record, name, stage, state, entry, wrapper_path in admissions:
            build = root / "out/build" / record["preset"]
            fresh = policy_collection_stage(root, build, record, name, stage, state,
                                                resolve=local, native_maps=terminal_native_maps, wrapper_path=wrapper_path)
            if fresh is None:
                fresh = _collect_cli_admission(root, build, record, name, stage, state)
            if fresh != entry:
                raise ValueError("terminal_changed: admission facts changed during collection")
        for record in records:
            if "product" in record["selection"]:
                prepared = record["stages"]["prepare"]
                identity = preparation_identity(root / "out/build" / record["preset"], prepared, resolve=local,
                    native_file_identities=terminal_native_maps.get(prepared["sha256"]))
                if identity != preparation_ids[record["preset"]]:
                    raise ValueError("checkpoint preparation changed during collection")
    with FileObservations() as publication_reads:
        if native:
            check_native_identity_evidence(root, native["stages"], native["expected"])
        if observe_tools() != (bindings, tools):
            raise ValueError("checkpoint toolchain/configuration changed during collection")
        if any(artifact_membership(expected) != expected for expected in artifact_seals):
            raise ValueError("checkpoint runtime artifact membership changed during collection")
        if snapshot(root, source_paths) != inputs:
            raise ValueError("checkpoint declared source/data inputs changed during collection")
        current = source_snapshot(root) if observed_source is not None else None
        if observed_source is not None and current != observed_source:
            raise ValueError("checkpoint source/data membership changed during collection")
        # Git selection omits ignored test/resource inputs that checkpoint_inputs owns.
        # Seal each shared membership observation independently of that selection.
        for (preset, group), expected in members.items():
            discovery = {"source_files": current, "resolve": local}
            if group == "prepare":
                discovery.update(preparation=preparations[preset], resolve=local)
            actual_members = {path for path in checkpoint_inputs(root, root / "out/build" / preset, group,
                                                                **discovery)
                              if not path.startswith("out/build/")}
            if actual_members != expected:
                raise ValueError("checkpoint source/data membership changed during collection")
        if any(read_json(path) != state for path, state in owners.items()):
            raise ValueError("pending or newer checkpoint ownership prevents terminal collection")
        if any((str(path.resolve(strict=True)), digest(path)) != facts for path, facts in evidence.items()):
            raise ValueError("checkpoint immutable evidence changed during collection")
        for observations in (initial_reads, semantic_reads):
            if any(str(Path(name).resolve(strict=True)) != target
                   for name, target in observations.bindings.items()):
                raise ValueError("checkpoint evidence path changed during collection")
            if any(digest(Path(name)) != checksum for name, checksum in observations.digests.items()):
                raise ValueError("checkpoint immutable evidence changed during collection")
        validate_qualification_membership(records)
        if current is None:
            current = source_snapshot(root)
        if any(r["source"] != current for r in records):
            raise ValueError("checkpoint records do not bind one current candidate")
        current_candidates = {stage["current_candidate"] for stage in provenance if "current_candidate" in stage}
        if candidates and len(current_candidates) != 1:
            raise ValueError("checkpoint admissions do not bind one complete current candidate")
        result = {"mode": "qualification", "status": "passed", "source": current,
                  "receipts": {str(p): evidence[Path(p)][1] for p in paths}, "stages": provenance,
                  "assurance": sorted({entry[4]["assurance"] for entry in admissions}) or ["ordinary-current-evidence"],
                  "scope": "explicit complete retained checkpoint; independent review and host acceptance remain separate"}
        if candidates:
            result["candidates"] = candidates
            result["current_candidate"] = next(iter(current_candidates))
        if transitions:
            result["producer_transitions"] = transitions
        if any(entry[4].get("recipe") == _POLICY_RECIPE for entry in admissions):
            result["scope"] = "synthetic miniature policy collection; no real product/GPU qualification or evidence adoption"
        if native:
            result["native_identities"] = native["stages"]
        if observed_source is not None and source_snapshot(root) != current:
            raise ValueError("checkpoint source/data membership changed during collection")
        result["file_observations"] = {"entry": initial_reads.counts(), "terminal": semantic_reads.counts(),
                                       "publication": publication_reads.counts()}
        OutputLease.check_held()
        return result
