"""Build preparation and complete CTest launch inputs for verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from execution import OutputLease, digest, snapshot, tool_snapshot, write_json
from build_dependencies import (
    CompilationInputs,
    NinjaGraph,
    file_identities,
    include_membership,
    configuration_outputs,
)


def configuration_files(build: Path) -> list[Path]:
    names = ["CMakeCache.txt", "build.ninja", "CMakeFiles/rules.ninja"]
    paths = [build / name for name in names]
    for path in paths:
        if not path.is_file():
            raise ValueError(
                f"configure the Ninja build before qualification: missing {path}"
            )
    for name in ("compile_commands.json", "verification-config-outputs.json"):
        optional = build / name
        if optional.is_file():
            paths.append(optional)
    return paths


def registration_files(build: Path, *, allow_missing=False) -> list[Path]:
    """Follow generated subdirs/includes only; never scan unrelated build trees."""
    pending = [build / "CTestTestfile.cmake"]
    visited = set()
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        if not path.is_file():
            if allow_missing:
                continue
            raise ValueError(
                f"build registered tests before qualification: missing {path}"
            )
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.match(r"\s*(include|subdirs)\s*\(", line, re.IGNORECASE):
                match = re.fullmatch(
                    r'\s*(include|subdirs)\("([^"$]+)"\)\s*', line, re.IGNORECASE
                )
                if not match:
                    raise ValueError(
                        f"unsupported generated test include in {path}: {line}"
                    )
                target = path.parent / match[2]
                pending.append(
                    target / "CTestTestfile.cmake"
                    if match[1].lower() == "subdirs"
                    else target
                )
    return sorted(visited)


def split_inputs(root: Path, paths) -> tuple[list[str], list[str]]:
    local, external = set(), set()
    for path in paths:
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"input is not a file: {path}")
        try:
            local.add(path.relative_to(root).as_posix())
        except ValueError:
            external.add(str(path))
    return sorted(local), sorted(external)


def ctest_command(root: Path, build: Path) -> list[str]:
    if build.parent == root / "out/build":
        return ["ctest", "--preset", build.name]
    return ["ctest", "--test-dir", str(build), "--parallel", "4", "--no-tests=error"]


def preparation_inventory(root: Path, build: Path, environment) -> dict:
    """Bind Ninja's regeneration dependencies, including external CMake modules."""
    paths = configuration_files(build)
    cache = paths[0].read_text(encoding="utf-8")
    if not re.search(r"^CMAKE_GENERATOR:INTERNAL=Ninja\s*$", cache, re.MULTILINE):
        raise ValueError("build preparation requires the project Ninja generator")
    match = re.search(r"^CMAKE_MAKE_PROGRAM:[^=]+=(.+)$", cache, re.MULTILINE)
    if not match or not Path(match[1]).is_file():
        raise ValueError("configured Ninja executable is missing")
    ninja = Path(match[1]).resolve()
    query = subprocess.run(
        [str(ninja), "-C", str(build), "-t", "query", "build.ninja"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    lines = query.stdout.splitlines()
    if (
        len(lines) < 3
        or lines[1] != "  input: RERUN_CMAKE"
        or "  outputs:" not in lines
    ):
        raise ValueError("unsupported Ninja regeneration dependency query")
    dependencies = [
        build / line.strip().removeprefix("| ").removeprefix("|| ")
        for line in lines[2 : lines.index("  outputs:")]
    ]
    helpers = [
        Path(__file__).with_name(name)
        for name in ("build_evidence.py", "build_dependencies.py", "execution.py", "process_tree.py")
    ]
    local, external = split_inputs(root, [*paths, *dependencies, ninja, *helpers])
    registrations = (
        registration_files(build, allow_missing=True)
        if (build / "CTestTestfile.cmake").is_file()
        else []
    )
    compiler_tools = [Path(value) for value in re.findall(r"^CMAKE_(?:C|CXX|CUDA)_COMPILER:[^=]+=(.+)$", cache, re.M)]
    cargo = re.search(r"^JUICER_CARGO_EXECUTABLE:[^=]+=(.+)$", cache, re.M)
    if cargo:
        cargo = Path(cargo[1])
        rustc = cargo.with_name("rustc.exe" if os.name == "nt" else "rustc")
        sysroot = Path(subprocess.check_output([str(rustc), "--print", "sysroot"], cwd=root, env=environment, text=True, timeout=30).strip())
        compiler_tools += [cargo, rustc, sysroot / "bin" / cargo.name, sysroot / "bin" / rustc.name]
    external.extend(str(path) for path in compiler_tools if path.is_file())
    return {
        "inputs": snapshot(root, local),
        "tools": tool_snapshot(external),
        "regeneration_mtimes": {
            str(p.resolve()): p.stat().st_mtime_ns for p in dependencies
        },
        "registrations": {
            str(p): digest(p) if p.is_file() else None for p in registrations
        },
    }


def redact_configure_trace(trace, source, build):
    facts, outputs = [], set()
    for line in trace.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        arguments = row.get("args", [])
        output = None
        if row.get("cmd") == "configure_file" and len(arguments) >= 2:
            output = arguments[1]
        elif (
            row.get("cmd") == "file"
            and arguments
            and arguments[0] in {"CONFIGURE", "GENERATE"}
            and "OUTPUT" in arguments
        ):
            output = arguments[arguments.index("OUTPUT") + 1]
        if output is None:
            continue
        path = Path(output)
        if not path.is_absolute() and Path(row["file"]).name == "CMakeLists.txt":
            path = build / Path(row["file"]).parent.relative_to(source) / path
        if path.is_absolute() and "$<" not in output:
            path = path.resolve()
            path.relative_to(build)
            outputs.add(str(path))
        facts.append(
            {
                "file": row["file"],
                "line": row["line"],
                "command": row["cmd"],
                "arguments_sha256": hashlib.sha256(
                    json.dumps(arguments).encode()
                ).hexdigest(),
            }
        )
    redacted = trace.with_suffix(".redacted.json")
    write_json(redacted, facts)
    trace.unlink()
    return outputs, redacted


def configure_with_evidence(
    campaign, stage, command, source, build, inputs, tools, environment, timeout
):
    """Retain only generated-output facts from CMake's expanded source trace."""
    import uuid

    directory = campaign.directory / "preparation"
    directory.mkdir(exist_ok=True)
    trace = directory / ("cmake-trace-" + uuid.uuid4().hex + ".jsonl")
    sources = {source / "CMakeLists.txt"}
    parents = (
        [source / name for name in ("cmake", "tests")]
        if source == campaign.root
        else [source]
    )
    for parent in parents:
        if parent.is_dir():
            sources.update(
                path
                for path in parent.rglob("*")
                if path.is_file()
                and (path.suffix == ".cmake" or path.name == "CMakeLists.txt")
            )
    config_inputs, config_tools = split_inputs(
        campaign.root, sources | {campaign.root / name for name in inputs}
    )
    flags = [
        "--trace-expand",
        "--trace-format=json-v1",
        "--trace-redirect=" + str(trace),
    ]
    flags += ["--trace-source=" + str(path) for path in sorted(sources)]
    primary_error = None
    try:
        configured = campaign.run(
            stage,
            [*command, *flags],
            inputs=config_inputs,
            tools=sorted(set(tools) | set(config_tools)),
            env=environment,
            timeout=timeout,
            repeat_reason="configuration changed or has no content proof",
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        records = campaign.attempts(stage)
        if (
            records
            and records[-1].get("process_tree", {}).get("drained")
            and trace.is_file()
        ):
            try:
                outputs, redacted = redact_configure_trace(trace, source, build)
            except (OSError, ValueError, KeyError) as trace_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "CMake trace redaction also failed: " + str(trace_error)
                )
                write_json(
                    trace.with_suffix(".redaction-error.json"),
                    {"error": str(trace_error), "trace_sha256": digest(trace)},
                )
                trace.unlink()
    record = json.loads(configured.read_text(encoding="utf-8"))
    record["configuration_evidence"] = {
        "outputs": sorted(outputs),
        "trace": str(redacted),
        "trace_sha256": digest(redacted),
    }
    write_json(configured, record)
    write_json(
        build / "verification-config-outputs.json",
        {
            "protocol": SCOPED_PROTOCOL,
            "producer": str(configured),
            "producer_sha256": digest(configured),
            "outputs": sorted(outputs),
            "trace": str(redacted),
            "trace_sha256": digest(redacted),
        },
    )
    return configured




LEGACY_ADMISSION_UNSUPPORTED = "legacy sealed-build preparation/CTest is unsupported; use explicit owning development or checkpoint admission"
SCOPED_PROTOCOL = "scoped-development-v1"






def dependency_edges_digest(graph):
    return hashlib.sha256(
        json.dumps(
            {name: sorted(children) for name, children in sorted(graph.inputs.items())},
            sort_keys=True,
        ).encode()
    ).hexdigest()


def environment_digest(environment):
    return hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in environment.items()
                if key
                not in {
                    "JUICER_OUTPUT_LEASES",
                }
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def verify_dependency_selection(root, build, graph, env=None):
    """A usable configured owning closure needs no complete-build receipt."""
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    match = re.search(r"^CMAKE_HOME_DIRECTORY:[^=]+=(.+)$", cache, re.M)
    if not match or Path(match[1]).resolve() != root.resolve():
        raise ValueError("configured build tree belongs to another source directory")
    if not graph.outputs or not graph.commands:
        raise ValueError("unresolved owning build dependencies")
    return preparation_inventory(root, build, dict(os.environ if env is None else env))


def scoped_context(root, source, build, graph, inputs, environment, compilation=None):
    """Relevant source/dependency content and configuration; outputs are local cache."""
    configuration = verify_dependency_selection(source, build, graph, environment)
    paths = set(graph.source_files()) | {root / name for name in inputs}
    # Missing depfiles are supplemented by the existing ordered include traversal.
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "juicer_scoped_includes", Path(__file__).with_name("check-quality.py")
    )
    quality = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = quality
    spec.loader.exec_module(quality)
    database = build / "compile_commands.json"
    if database.is_file():
        compilation = compilation or CompilationInputs(source, build)
        for row in compilation.rows:
            output = str(graph.path(row.get("output", "")))
            if output not in graph.outputs:
                continue
            paths.update(source / name for name in compilation.paths(row))
    probe_configs = []
    for group, harness in (("rust-naming", "test_rust_naming.py"), ("rust-boundaries", "test_rust_boundaries.py")):
        if "tests/quality/" + harness in inputs:
            artifact = source / "out/validation" / build.name / "quality" / group
            quality.check_probe_configuration(source, artifact)
            probe_configs.extend(parent / ".cargo" / name for parent in (artifact, *artifact.parents)
                                 for name in ("config", "config.toml"))
    # Linked runtime binaries and archives are not recursively sealed.
    generated = set(configuration_outputs(build, environment))
    for path in paths:
        if (path.is_relative_to(build) and not path.is_relative_to(build / "_deps")
                and path.suffix.lower() in {".h", ".hpp", ".inc"}
                and path not in generated and str(path) not in graph.outputs):
            raise ValueError("unresolved generated build input: " + str(path))
    paths = {path for path in paths if path.suffix.lower() not in {".a", ".lib", ".o", ".obj", ".so"} and ".so." not in path.name}
    return {
        "configuration": configuration,
        "files": file_identities(paths),
        "include_membership": include_membership(build, graph, source, compilation),
        "commands": graph.commands,
        "rust_configuration": file_identities(
            [*probe_configs, *[directory / ".cargo" / name for directory in (source, *source.parents) for name in ("config", "config.toml")]]
            + [Path(environment.get("CARGO_HOME", Path.home() / ".cargo")) / name for name in ("config", "config.toml")]
            + ([Path(environment["RUSTUP_HOME"]) / "settings.toml"] if environment.get("RUSTUP_HOME") else [])
        ) if any(Path(name).name in {"libfilm_juicer_plugin.a", "film_juicer_plugin.lib"} for name in graph.outputs) else {},
        "rust_timestamps": {str(path): path.stat().st_mtime_ns for path in paths if path.suffix == ".rs" and path.is_file()},
        "environment_sha256": environment_digest(environment),
    }


def preparation_paths(build, targets):
    key = hashlib.sha256(json.dumps(sorted(set(targets))).encode()).hexdigest()[:24]
    baseline = build / ("verification-build-" + key + ".json")
    return baseline, baseline.with_suffix(".pending.json")


def check_scoped_preparation(campaign, build, prepared, env=None, *, compilation=None):
    record = json.loads(Path(prepared).read_text(encoding="utf-8"))
    proof = record.get("scoped_preparation")
    if not proof or proof.get("protocol") != SCOPED_PROTOCOL:
        raise ValueError(LEGACY_ADMISSION_UNSUPPORTED)
    baseline, pending = preparation_paths(build, proof["targets"])
    if pending.exists():
        raise ValueError("pending preparation invalidates the prior scoped pass")
    try:
        current = json.loads(baseline.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("latest completed preparation is unavailable; repeat preparation") from error
    if proof.get("producer") != str(Path(prepared).resolve()) or current != proof:
        raise ValueError("scoped pass does not own the latest completed preparation")
    return _check_preparation_result(campaign, build, prepared, env, compilation=compilation)


def _check_preparation_result(campaign, build, prepared, env=None, *, compilation=None, fresh_context=None):
    record = json.loads(Path(prepared).read_text(encoding="utf-8"))
    proof = record.get("scoped_preparation")
    if not proof or proof.get("protocol") != SCOPED_PROTOCOL:
        raise ValueError(LEGACY_ADMISSION_UNSUPPORTED)
    if record.get("status") != "passed" or record.get("returncode") != 0 or not record.get("process_tree", {}).get("drained"):
        raise ValueError("scoped preparation did not pass with proved process completion")
    process = json.loads(Path(record["process_tree"]["receipt"]).read_text(encoding="utf-8"))
    if any(record["process_tree"].get(key) != value for key, value in process.items()) or digest(campaign.root / record["output"]) != record["output_sha256"]:
        raise ValueError("scoped preparation command evidence changed")
    if proof["build"] != str(build.resolve()):
        raise ValueError("scoped preparation belongs to another build tree")
    expected_command = ["cmake", "--build", str(build.resolve()), "--parallel", "2", "--target", *proof["targets"]]
    if record["identity"]["command"] != expected_command:
        raise ValueError("scoped preparation command/targets changed")
    try:
        if fresh_context is None:
            graph = NinjaGraph(build, proof["targets"], env)
            fresh_context = scoped_context(campaign.root, Path(proof["source"]), build, graph, proof["inputs"], dict(os.environ if env is None else env), compilation)
    except ValueError as error:
        raise ValueError("relevant dependency contents or membership changed: " + str(error)) from error
    if fresh_context != proof["context"]:
        raise ValueError("relevant dependency contents or membership changed; repeat preparation")
    if any(not Path(name).is_file() for name in proof["outputs"]):
        raise ValueError("scoped build output is missing; repeat preparation")
    return record


def development_selection(root, build, inputs, *, compilation=None):
    from verify import SUPPORTED, select_consumers

    return select_consumers(root, build, sorted(set(inputs) & SUPPORTED), compilation=compilation)


def invalidate_preparation_owners(build, baseline, outputs):
    """Overlapping target sets cannot keep proof of outputs this writer may replace."""
    for path in build.glob("verification-build-*.json"):
        if path == baseline or path.name.endswith(".pending.json"):
            continue
        try:
            proof = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # An unreadable owner is already unusable by every reader.
        if proof.get("protocol") == SCOPED_PROTOCOL and outputs & set(proof["outputs"]):
            path.unlink()


def invalidate_rust(campaign, stage, source, build, changed, old, environment, timeout):
    """Use Cargo incrementally except for content edits its timestamps could miss."""
    changed = {name for name in changed if name.endswith(".rs") and Path(name).is_relative_to(source / "rust")}
    if not changed:
        return []
    stamp = build / "cargo-build.stamp"
    previous_times = old.get("context", {}).get("rust_timestamps", {}) if old else {}
    stale = {name for name in changed if Path(name).stat().st_mtime_ns <= previous_times.get(name, 0)
             or stamp.is_file() and Path(name).stat().st_mtime_ns <= stamp.stat().st_mtime_ns}
    if not stale:
        return []
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--offline", "--format-version", "1"],
        cwd=source, env=environment, text=True, timeout=30))
    packages = {package["id"]: package for package in metadata["packages"]}
    affected = {key for key, package in packages.items()
                if any(Path(name).is_relative_to(Path(package["manifest_path"]).parent) for name in stale)}
    while True:
        expanded = affected | {node["id"] for node in metadata["resolve"]["nodes"]
                               if any(dependency["pkg"] in affected for dependency in node["deps"])}
        if expanded == affected:
            break
        affected = expanded
    if not affected:
        raise ValueError("unresolved package ownership for restored-time Rust inputs")
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    profile = "release" if re.search(r"^CMAKE_BUILD_TYPE:STRING=Release$", cache, re.M) else "dev"
    target = "x86_64-pc-windows-msvc" if os.name == "nt" else "x86_64-unknown-linux-gnu"
    cargo = re.search(r"^JUICER_CARGO_EXECUTABLE:[^=]+=(.+)$", cache, re.M)
    if not cargo:
        raise ValueError("missing configured Cargo owner")
    command = [cargo[1], "clean", "--manifest-path", str(source / "Cargo.toml"), "--locked", "--offline",
               "--target-dir", str(build / "cargo"), "--profile", profile]
    names = sorted({packages[key]["name"] for key in affected})
    package_args = [item for name in names for item in ("--package", name)]
    local, external = split_inputs(campaign.root, [source / "Cargo.toml", source / "Cargo.lock", *(Path(name) for name in stale)])
    campaign.run(stage, [*command, "--target", target, *package_args], inputs=local, tools=external, env=environment,
                 timeout=timeout, repeat_reason="restored-time content edit requires active-package Cargo invalidation")
    host = sorted({packages[key]["name"] for key in affected
                   if any(set(target["kind"]) & {"custom-build", "proc-macro"} for target in packages[key]["targets"])})
    if host:
        campaign.run(stage, [*command, *[item for name in host for item in ("--package", name)]],
                     inputs=local, tools=external, env=environment, timeout=timeout,
                     repeat_reason="invalidate affected active-profile host build-script/proc-macro products")
    stamp.unlink(missing_ok=True)
    return names


def rust_context(source, build, context):
    """The existing preparation owner's relevant Cargo source/configuration identity."""
    controls = {source / name for name in ("Cargo.toml", "Cargo.lock", "clippy.toml", "rust-toolchain.toml", "rustfmt.toml")}
    controls.add(build / "cargo-features.txt")
    return {
        "files": {name: identity for name, identity in context["files"].items()
                  if Path(name).is_relative_to(source / "rust") or Path(name) in controls},
        "configuration": context.get("rust_configuration"),
        "tools": {name: identity for name, identity in context["configuration"]["tools"].items()
                  if Path(name).name in {"rustup", "rustup.exe", "cargo", "cargo.exe", "rustc", "rustc.exe"}},
        "environment": context["environment_sha256"],
    }


def rust_changes(current, previous):
    files, old = current["files"], previous["files"]
    changed = {name for name in files.keys() | old.keys() if files.get(name) != old.get(name)}
    if any(current[name] != previous[name] for name in current if name != "files") or any(not name.endswith(".rs") for name in changed):
        changed.update(name for name in files if name.endswith(".rs"))
    return changed


def _prepare_build_owned(campaign, stage, *, source, build, inputs, timeout,
                         configure_args=(), targets=(), env=None, tools=(), resume=None, checkpoint=False):
    if not targets or resume is not None:
        raise ValueError(LEGACY_ADMISSION_UNSUPPORTED)
    source, build = source.resolve(), build.resolve()
    if not checkpoint and source == campaign.root and build.parent == campaign.root / "out/build":
        if build.name != "linux-debug" or any(name not in {"JuicerRouteBridgeTests", "JuicerOutputColorTests"}
                                             and not (".dir/" in name and name.endswith((".o", ".obj"))) for name in targets):
            raise ValueError(LEGACY_ADMISSION_UNSUPPORTED)
        selected = development_selection(source, build, inputs)
        if sorted(targets) != selected["targets"]:
            raise ValueError(LEGACY_ADMISSION_UNSUPPORTED)
    targets, inputs = sorted(set(targets)), sorted(set(inputs))
    environment = dict(os.environ if env is None else env)
    with OutputLease(campaign.root, build):
        configured = None
        baseline, pending_path = preparation_paths(build, targets)
        interrupted = pending_path.is_file()
        old = None
        if baseline.is_file():
            candidate = json.loads(baseline.read_text(encoding="utf-8"))
            if candidate.get("protocol") == SCOPED_PROTOCOL:
                old = candidate
        build.mkdir(parents=True, exist_ok=True)
        write_json(pending_path, {"protocol": SCOPED_PROTOCOL, "targets": targets})
        cache = build / "CMakeCache.txt"
        configure = not cache.is_file() or old is None or old.get("configure_args") != list(configure_args)
        if cache.is_file():
            inventory = preparation_inventory(source, build, environment)
            if old and old["context"]["configuration"] != inventory:
                configure = True
        if configure:
            command = (["cmake", "--preset", build.name, *configure_args]
                       if source == campaign.root and build.parent == campaign.root / "out/build"
                       else ["cmake", "-S", str(source), "-B", str(build), *configure_args])
            build.mkdir(parents=True, exist_ok=True)
            configured = configure_with_evidence(campaign, stage, command, source, build, inputs, tools, environment, timeout)
        graph = NinjaGraph(build, targets, environment)
        before = scoped_context(campaign.root, source, build, graph, inputs, environment)
        changed = set()
        if old:
            previous, actual = old["context"]["files"], before["files"]
            changed = {name for name in previous.keys() | actual.keys() if previous.get(name) != actual.get(name)}
        broad = old is None or interrupted or any(old["context"].get(name) != before.get(name) for name in ("configuration", "include_membership", "commands", "environment_sha256", "rust_configuration"))
        rust_files = {name: identity for name, identity in before["files"].items() if name.endswith(".rs") and Path(name).is_relative_to(source / "rust")}
        current_rust = rust_context(source, build, before)
        if old:
            changed.update(rust_changes(current_rust, rust_context(source, build, old["context"])))
        if old is None or interrupted:
            archive_names = {name for name in graph.outputs if Path(name).name in {"libfilm_juicer_plugin.a", "film_juicer_plugin.lib"}}
            shared = []
            for owner in build.glob("verification-build-*.json"):
                if owner.name.endswith(".pending.json") or owner == baseline:
                    continue
                try:
                    proof = json.loads(owner.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                owner_pending = owner.with_suffix(".pending.json")
                if proof.get("protocol") == SCOPED_PROTOCOL and proof.get("source") == str(source) and proof.get("build") == str(build) and archive_names & set(proof["outputs"]) and not owner_pending.exists():
                    shared.append(rust_context(source, build, proof["context"]))
            # All valid overlapping owners must describe the same archive inputs.
            # No owner (or an interrupted write) is unknown, even for clean HEAD.
            if not interrupted and shared and all(value == shared[0] for value in shared):
                changed.update(rust_changes(current_rust, shared[0]))
            elif any(Path(name).is_file() for name in archive_names):
                changed.update(rust_files)
        mutable = {name for name in graph.outputs if Path(name).suffix.lower() not in {".a", ".lib"} and Path(name).name != "cargo-build.stamp"}
        affected = mutable if broad else graph.affected(changed) & mutable
        potential = set(affected) | {name for name in graph.outputs if not Path(name).is_file()}
        if any(name.endswith(".rs") for name in changed):
            potential.update(graph.affected(changed))
        invalidate_preparation_owners(build, baseline, potential)
        rust_invalidated = invalidate_rust(campaign, stage, source, build, changed, old, environment, timeout)
        invalidated = graph.invalidate(affected)
        command = ["cmake", "--build", str(build), "--parallel", "2", "--target", *targets]
        write_json(pending_path, {"protocol": SCOPED_PROTOCOL, "targets": targets, "outputs": sorted(mutable)})
        for attempt in range(2):
            local, external = split_inputs(campaign.root, [Path(name) for name in before["files"]])
            built = campaign.run(stage, command, inputs=local, tools=[*tools, *external], env=environment,
                                 timeout=timeout, repeat_reason="build only explicit owning targets and dependencies")
            fresh_graph = NinjaGraph(build, targets, environment)
            after = scoped_context(campaign.root, source, build, fresh_graph, inputs, environment)
            if file_identities([Path(name) for name in before["files"]]) != before["files"]:
                raise ValueError("relevant inputs mutated during preparation")
            if before == after:
                break
            if attempt or any(before[name] != after[name] for name in before if name != "files"):
                raise ValueError("unresolved dependency/configuration change during preparation")
            discovered = set(after["files"]) ^ before["files"].keys()
            invalidated += fresh_graph.invalidate(fresh_graph.affected(discovered) & mutable)
            before, graph = after, fresh_graph
        proof = {"protocol": SCOPED_PROTOCOL, "producer": str(built.resolve()), "source": str(source), "build": str(build),
                 "targets": targets, "inputs": inputs, "configure_args": list(configure_args), "context": after,
                 "outputs": sorted(fresh_graph.outputs), "invalidated": invalidated, "rust_invalidated_packages": rust_invalidated}
        record = json.loads(built.read_text(encoding="utf-8"))
        record["scoped_preparation"] = proof
        write_json(built, record)
        # The writer just established this post-build context. Only its owned
        # result was written since; independent readers always rediscover it.
        _check_preparation_result(campaign, build, built, environment, fresh_context=after)
        write_json(baseline, proof)
        pending_path.unlink()
        return configured, built


def run_ctest(campaign, stage, *, build, prepared, inputs, timeout, env=None,
              tools=(), tests=(), junit=None):
    if not tests or build.parent == campaign.root / "out/build" and (
            build.name != "linux-debug" or not set(tests) <= {"Route.Host.Bridge", "Ffi.Host.OutputColor", "Quality.RustBoundaries"}):
        raise ValueError(LEGACY_ADMISSION_UNSUPPORTED)
    with OutputLease(campaign.root, build):
        compilation = CompilationInputs(campaign.root, build) if build.parent == campaign.root / "out/build" else None
        prepared_record = check_scoped_preparation(campaign, build, prepared, env, compilation=compilation)
        if build.parent == campaign.root / "out/build":
            proof = prepared_record["scoped_preparation"]
            selected_scope = development_selection(campaign.root, build, proof["inputs"], compilation=compilation)
            if sorted(tests) != selected_scope["tests"] or proof["targets"] != selected_scope["targets"]:
                raise ValueError(LEGACY_ADMISSION_UNSUPPORTED)
        # Query only the named groups. Do not seal test executables or runtime trees.
        pattern = "^(" + "|".join(re.escape(name) for name in sorted(tests)) + ")$"
        command = [*ctest_command(campaign.root, build), "-R", pattern]
        inventory = json.loads(subprocess.check_output([*command, "--show-only=json-v1"], cwd=campaign.root, env=env, text=True, timeout=30))
        selected = inventory.get("tests", [])
        if sorted(test["name"] for test in selected) != sorted(tests) or not selected:
            raise ValueError("selected CTest membership changed or is empty")
        if any(any(prop["name"] == "LABELS" and "gpu" in prop["value"] for prop in test.get("properties", [])) for test in selected):
            raise ValueError("GPU tests require explicit checkpoint admission; " + LEGACY_ADMISSION_UNSUPPORTED)
        if any(any(prop["name"] == "DISABLED" and prop["value"] for prop in test.get("properties", [])) for test in selected):
            raise ValueError("selected CTest group is disabled")
        if junit is None:
            junit = campaign.directory / (stage + ".xml")
        check_scoped_preparation(campaign, build, prepared, env)
        result = campaign.run(stage, [*command, "--output-on-failure", "--output-junit", str(junit)],
                              inputs=sorted(set(inputs) | {Path(prepared).relative_to(campaign.root).as_posix()}),
                              tools=tools, env=env, timeout=timeout, repeat_reason="execute exact owning CTest groups")
        import xml.etree.ElementTree as ET

        cases = ET.parse(junit).getroot().findall(".//testcase")
        if sorted(case.get("name") for case in cases) != sorted(tests) or any(case.find(name) is not None for case in cases for name in ("failure", "error", "skipped")):
            raise ValueError("selected CTest did not execute a nonempty successful result")
        check_scoped_preparation(campaign, build, prepared, env)
        return result


def prepare_build(campaign, stage, **arguments):
    before = len(campaign.attempts(stage))
    try:
        return _prepare_build_owned(campaign, stage, **arguments)
    except BaseException as error:
        records = campaign.attempts(stage)
        if len(records) > before:
            record = records[-1]
            if (
                record["status"] == "passed"
                and "--build" in record["identity"]["command"]
            ):
                record.update(
                    status="failed", error="build provenance failed: " + str(error)
                )
                write_json(
                    campaign.root / Path(record["output"]).with_name("result.json"),
                    record,
                )
        raise
