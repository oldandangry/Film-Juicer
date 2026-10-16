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

from build_evidence import check_scoped_preparation, prepare_build, registration_files
from execution import Campaign, OutputLease, digest, snapshot, tool_snapshot, write_json

PRESETS = ("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release")
GROUPS = {"product", "boundary", "tooling", "corpus", "production"}


def require_checkpoint(record):
    selected = set(record.get("selection", ()))
    expected = selected | ({"prepare", "quality"} if "product" in selected else set())
    if (record.get("mode") != "checkpoint" or record.get("status") != "passed"
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


def run_stage(campaign, state_path, state, name, command, *, inputs, tools, environment,
          timeout, semantic_command=None, action=None, validate=None, payloads=(),
          dependencies=(), repeat_reason="", refresh_inputs=None):
    """Reuse the latest gate across attempt names; pending work is never a pass."""
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
        identity["inputs"] = snapshot(campaign.root, refresh_inputs()) if refresh_inputs else actual
        result = {"status": "passed", "identity": identity, "receipt": str(paths[0]),
                  "sha256": digest(paths[0]),
                  "additional_receipts": {str(p): digest(p) for p in paths[1:]}, "payloads": {str(p): digest(Path(p)) for p in payloads}}
        state[name] = result
        write_json(state_path, state)
        return result
    except BaseException as error:
        state[name].update(status="failed", error=str(error))
        write_json(state_path, state)
        raise


def checkpoint_inputs(root, build, group):
    paths = {root / name for name in ("CMakeLists.txt", "CMakePresets.json", "tests/CMakeLists.txt",
             "scripts/qualification.py", "scripts/execution.py", "scripts/process_tree.py")}
    paths.add(build / "CMakeCache.txt")
    if group != "production":
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
            paths.update((root / "tests/quality").glob("test_rust_*.py"))
            paths.update(p for p in (root / "tests/quality/fixtures").rglob("*.rs"))
        else:
            for directory in ("scripts", "tests/quality", "tests/ctest"):
                paths.update(p for p in (root / directory).rglob("*.py") if "__pycache__" not in p.parts)
            paths.add(root / "tests/ffi/run_rust_build_contract.py")
    else:
        from verify import source_snapshot
        paths.update(root / name for name in source_snapshot(root))
        for directory in ("Resources", "tests"):
            paths.update(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return sorted(p.relative_to(root).as_posix() for p in paths if p.is_file())


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
    def run(name, command, inputs, **kwargs):
        tools = checkpoint_tools(build, environment, name if name in GROUPS else "product")
        result = run_stage(campaign, state_path, state, name, command, inputs=inputs, tools=tools,
                       environment=environment, timeout=1800, repeat_reason=args.repeat_reason, **kwargs)
        record["stages"][name] = result
        return result
    try:
        built = None
        if "product" in groups:
            inputs = checkpoint_inputs(root, build, "product")
            built = run("prepare", ["cmake", "--build", str(build)], inputs,
                action=lambda: prepare_build(campaign, "prepare", source=root, build=build,
                    inputs=[p for p in inputs if not p.startswith("out/build/")], timeout=1800, targets=["all"], configure_args=["-DBUILD_TESTING=ON"], checkpoint=True)[1],
                refresh_inputs=lambda: checkpoint_inputs(root, build, "product"),
                validate=lambda path: check_scoped_preparation(campaign, build, path))
            inputs = checkpoint_inputs(root, build, "product")
            run("quality", [sys.executable, "-B", "scripts/check-quality.py", "--preset", args.preset,
                "--qualification", "--base", args.base], inputs, dependencies=[built["sha256"]])
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
                inputs = checkpoint_inputs(root, build, group)
                tools = checkpoint_tools(build, environment, group)
                run(group, commands[0], inputs, semantic_command=identities,
                    action=lambda group=group, commands=commands, parts=parts, inputs=inputs, tools=tools:
                        run_ctest_parts(campaign, group, commands, parts, inputs=inputs, tools=tools,
                            environment=environment, timeout=1800, repeat_reason=args.repeat_reason),
                    validate=validate, payloads=reports,
                    dependencies=[built["sha256"]] if built and group == "product" else [])
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
            inputs = checkpoint_inputs(root, build, "production") + [p.relative_to(root).as_posix() for p in artifacts]
            command = [sys.executable, "-B", "tests/ffi/production_contract.py", "--preset", args.preset, "--artifacts", str(out)]
            run("production", command, inputs, semantic_command=[*command[:-1], "<artifacts>"], payloads=[out / "production.json"])
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
            for path, expected in result.get("payloads", {}).items():
                if digest(local(path)) != expected:
                    raise ValueError("checkpoint result payload changed")
            if name in record.get("membership", {}):
                validate_ctest_stage(raw_paths, record.get("runtime_membership", {}).get(name, {}),
                    record["membership"][name], result["identity"]["command"], local)
            declared = result["identity"]["inputs"]
            source_inputs = {path: checksum for path, checksum in declared.items() if not path.startswith("out/build/")}
            if snapshot(root, list(source_inputs)) != source_inputs:
                raise ValueError("checkpoint declared source/data inputs changed")
            group = name if name in GROUPS else "product"
            expected_members = {path for path in declared if not path.startswith("out/build/")}
            actual_members = {path for path in checkpoint_inputs(root, root / "out/build" / record["preset"], group)
                              if not path.startswith("out/build/")}
            if expected_members != actual_members:
                raise ValueError("checkpoint source/data membership changed")
            expected_tools = {str(local(path).resolve()): value for path, value in result["identity"]["tools"].items()}
            if tool_snapshot([str(local(p)) for p in result["identity"]["tool_paths"]], allow_missing=True) != expected_tools:
                raise ValueError("checkpoint toolchain/configuration changed")
            if name == "production":
                artifacts = {path: checksum for path, checksum in result["identity"]["inputs"].items()
                             if Path(path).suffix.lower() in {".a", ".lib", ".ofx", ".pdb"}}
                if not artifacts or any(digest(root / path) != checksum for path, checksum in artifacts.items()):
                    raise ValueError("final inspected production artifacts changed")
    required = {(p, group) for p in PRESETS for group in ("product", "boundary")}
    required |= {(p, "tooling") for p in ("linux-debug", "windows-clang-debug")}
    required |= {(p, "production") for p in ("linux-release", "windows-clang-release")}
    actual = [(r["preset"], g) for r in records for g in r["selection"] if g != "corpus"]
    if len(actual) != len(set(actual)) or set(actual) != required or sum("corpus" in r["selection"] for r in records) != 1:
        raise ValueError("full checkpoint membership is incomplete or duplicated")
    if any(r["runtime"] != "all" for r in records if "product" in r["selection"]):
        raise ValueError("host-only product checks do not establish full qualification")
    from verify import source_snapshot
    current = source_snapshot(root)
    if any(r["source"] != current for r in records):
        raise ValueError("checkpoint records do not bind one current candidate")
    return {"mode": "qualification", "status": "passed", "source": current,
            "receipts": {str(p): digest(Path(p)) for p in paths}, "scope": "explicit complete retained checkpoint; independent review and host acceptance remain separate"}
