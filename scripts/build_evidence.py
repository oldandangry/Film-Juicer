"""Build preparation and complete CTest launch inputs for verification."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import uuid

from execution import Campaign, OutputLease, command_environment, digest, snapshot, tool_snapshot, write_json
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


def windows_vctip(environment) -> str | None:
    if os.name != "nt":
        return None
    tools = next(
        (value for key, value in environment.items() if key.upper() == "VCTOOLSINSTALLDIR"),
        None,
    )
    if not tools:
        raise ValueError("Windows build needs the selected VCToolsInstallDir")
    return str(Path(tools) / "bin/Hostx64/x64/VCTIP.EXE")


def ctest_command(root: Path, build: Path) -> list[str]:
    if build.parent == root / "out/build":
        return ["ctest", "--preset", build.name]
    return ["ctest", "--test-dir", str(build), "--parallel", "4", "--no-tests=error"]


def linker_tools(build: Path, cache: str, environment) -> list[Path]:
    """Bind configured linkers and explicit Linux compiler-driver selections."""
    configured = re.search(r"^CMAKE_LINKER:[^=]+=(.+)$", cache, re.M)
    tools = {Path(configured[1])} if configured else set()
    if os.name == "nt":
        return sorted(tools)
    compilers = {value for value in re.findall(
        r"^CMAKE_(?:C|CXX)_COMPILER:[^=]+=(.+)$", cache, re.M)}
    if not compilers:
        return sorted(tools)
    ninja = re.search(r"^CMAKE_MAKE_PROGRAM:[^=]+=(.+)$", cache, re.M)
    if not ninja or not (build / "build.ninja").is_file():
        raise ValueError("selected linker discovery requires the configured Ninja commands")
    commands = subprocess.check_output(
        [ninja[1], "-C", str(build), "-t", "commands"], env=environment,
        text=True, encoding="utf-8", timeout=None)
    selected = set()
    for command in commands.splitlines():
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        tokens = list(lexer)
        for index, token in enumerate(tokens):
            if token in {"&&", "||", ";", "|"} or not token.startswith("/"):
                continue
            if token not in compilers:
                continue
            end = next((i for i in range(index + 1, len(tokens))
                        if tokens[i] in {"&&", "||", ";", "|"}), len(tokens))
            arguments = tokens[index + 1:end]
            if "-c" in arguments or "-E" in arguments or "-S" in arguments:
                continue
            selector = None
            explicit = None
            lookup = []
            position = 0
            while position < len(arguments):
                argument = arguments[position]
                if argument.startswith("-fuse-ld="):
                    selector = argument.split("=", 1)[1]
                elif argument.startswith("--ld-path="):
                    explicit = argument.split("=", 1)[1]
                elif argument == "--ld-path":
                    if position + 1 == len(arguments):
                        raise ValueError("selected linker option has no operand: " + argument)
                    position += 1
                    explicit = arguments[position]
                elif argument in {"-B", "--sysroot", "--target", "--gcc-toolchain"}:
                    if position + 1 == len(arguments):
                        raise ValueError("selected linker option has no operand: " + argument)
                    position += 1
                    lookup.extend((argument, arguments[position]))
                elif (argument.startswith(("-B", "--sysroot=", "--target=", "--gcc-toolchain="))
                      or argument in {"-m32", "-m64"}):
                    lookup.append(argument)
                position += 1
            if selector is None and explicit is None:
                continue
            start = next((i + 1 for i in range(index - 1, -1, -1)
                          if tokens[i] in {"&&", "||", ";", "|"}), 0)
            if index != start or any(a.startswith(("@", "-specs", "--specs", "-wrapper", "--config")) for a in arguments):
                raise ValueError("unsupported selected-linker launch or search configuration")
            selected.add((token, selector or "", explicit or "", tuple(lookup)))
    for compiler, selector, explicit, lookup in sorted(selected):
        if explicit:
            name = explicit
        elif Path(selector).is_absolute():
            name = selector
        else:
            name = subprocess.check_output(
                [compiler, *lookup, "-fuse-ld=" + selector, "-print-prog-name=ld." + selector],
                cwd=build, env=environment, text=True, timeout=None).strip()
        path = Path(name)
        if not path.is_absolute():
            found = shutil.which(name, path=environment.get("PATH")) if path.parent == Path(".") else str(build / path)
            if not found:
                raise ValueError("selected linker executable is missing: " + name)
            path = Path(found)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError("selected linker executable is unavailable: " + str(path))
        tools.add(path)
    return sorted(tools)


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
        timeout=None,
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
    compiler_tools += linker_tools(build, cache, environment)
    cargo = re.search(r"^JUICER_CARGO_EXECUTABLE:[^=]+=(.+)$", cache, re.M)
    if cargo:
        cargo = Path(cargo[1])
        rustc = cargo.with_name("rustc.exe" if os.name == "nt" else "rustc")
        sysroot = Path(subprocess.check_output([str(rustc), "--print", "sysroot"], cwd=root, env=environment, text=True, timeout=None).strip())
        compiler_tools += [cargo, rustc, sysroot / "bin" / cargo.name, sysroot / "bin" / rustc.name]
    external.extend(str(path) for path in compiler_tools if path.is_file())
    return {
        "inputs": snapshot(root, local),
        "tools": tool_snapshot(external),
        "regeneration_mtimes": {
            str(p.resolve()): p.stat().st_mtime_ns for p in dependencies
            if p.resolve() != (build / "CMakeFiles/cmake.verify_globs").resolve()
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
            vctip_executable=windows_vctip(environment),
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
    registry = build / "verification-config-outputs.json"
    if registry.is_file():
        try:
            previous = json.loads(registry.read_text(encoding="utf-8"))
            if (isinstance(previous, dict)
                    and previous.get("protocol") == SCOPED_PROTOCOL
                    and previous.get("outputs") == record["configuration_evidence"]["outputs"]
                    and previous.get("trace_sha256") == record["configuration_evidence"]["trace_sha256"]):
                # Fresh configure evidence remains in its own receipt. Retain the
                # validated registry when only producer/trace paths would change.
                configuration_outputs(build, environment)
                return configured
        except (OSError, ValueError, KeyError, TypeError):
            # The completed configure can replace evidence that cannot be retained.
            pass
    write_json(
        registry,
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
                for key, value in command_environment(environment).items()
                if key
                not in {
                    "JUICER_OUTPUT_LEASES",
                    "CMAKE_BUILD_PARALLEL_LEVEL",
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
    paths = {path for path in paths if str(path) not in graph.outputs
             and path.suffix.lower() not in {".a", ".lib", ".o", ".obj", ".so"}
             and ".so." not in path.name}
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


def check_preparation_owner(build, prepared, *, resolve=Path):
    """Check the current producer, including when collecting another native lane."""
    prepared = resolve(str(prepared))
    record = json.loads(prepared.read_text(encoding="utf-8"))
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
    producer_sha256 = current.pop("producer_sha256", None)
    if producer_sha256 is not None and digest(prepared) != producer_sha256:
        raise ValueError("scoped preparation command receipt changed")
    if not proof.get("producer") or resolve(proof["producer"]).resolve() != prepared.resolve() or current != proof:
        raise ValueError("scoped pass does not own the latest completed preparation")
    if resolve(proof["build"]).resolve() != build.resolve():
        raise ValueError("scoped preparation belongs to another build tree")
    return record


def check_scoped_preparation(campaign, build, prepared, env=None, *, compilation=None):
    check_preparation_owner(build, prepared)
    return _check_preparation_result(campaign, build, prepared, env, compilation=compilation)


def check_preparation_inputs(build, proof, *, resolve=Path):
    """Recheck recorded native facts without executing another lane's tools."""
    source = resolve(proof["source"])
    context = proof["context"]
    configuration = context["configuration"]
    for files in (context["files"], context["rust_configuration"]):
        expected = {str(resolve(name)): {**identity, "resolved": str(resolve(identity["resolved"]))}
                    for name, identity in files.items()}
        if file_identities([resolve(name) for name in files]) != expected:
            raise ValueError("preparation dependency contents or paths changed")
    if snapshot(source, list(configuration["inputs"])) != configuration["inputs"]:
        raise ValueError("preparation configuration inputs changed")
    expected_tools = {str(resolve(name).resolve()): checksum for name, checksum in configuration["tools"].items()}
    if tool_snapshot([str(resolve(name)) for name in configuration["tools"]]) != expected_tools:
        raise ValueError("preparation configuration tools changed")
    if any(resolve(name).stat().st_mtime_ns != stamp for name, stamp in configuration["regeneration_mtimes"].items()):
        raise ValueError("preparation regeneration inputs changed")
    registrations = {str(resolve(name)): checksum for name, checksum in configuration["registrations"].items()}
    if {str(p): digest(p) for p in registration_files(build)} != registrations:
        raise ValueError("preparation test registration changed")
    if any(p.relative_to(source).as_posix() not in configuration["inputs"] for p in configuration_files(build)):
        raise ValueError("preparation configuration membership changed")
    # The producer already resolved the ordered include scopes. Recheck those
    # namespaces, including new shadow files, without parsing foreign commands.
    outputs = {resolve(name) for name in proof["outputs"]}
    for name, rows in context["include_membership"].items():
        directory = resolve(name)
        expected = {relative.replace("\\", "/"): str(resolve(target)) if target is not None else None
                    for relative, target in rows}
        members, visited = {}, set()
        for parent, children, files in os.walk(directory, followlinks=True):
            children[:] = sorted(n for n in children if n not in {"__pycache__", ".git", ".ruff_cache"})
            parent = Path(parent)
            resolved = parent.resolve()
            if resolved in visited:
                children[:] = []
                continue
            visited.add(resolved)
            members[parent.relative_to(directory).as_posix()] = str(resolved)
            members.update({(parent / n).relative_to(directory).as_posix(): None for n in files if parent / n not in outputs})
        if members != expected:
            raise ValueError("preparation include membership changed")


def _check_preparation_command(campaign, build, prepared):
    record = json.loads(Path(prepared).read_text(encoding="utf-8"))
    proof = record.get("scoped_preparation")
    if not proof or proof.get("protocol") != SCOPED_PROTOCOL:
        raise ValueError(LEGACY_ADMISSION_UNSUPPORTED)
    if record.get("status") != "passed" or record.get("returncode") != 0 or not record.get("process_tree", {}).get("drained"):
        raise ValueError("scoped preparation did not pass with proved process completion")
    process = json.loads(Path(record["process_tree"]["receipt"]).read_text(encoding="utf-8"))
    expected_process = {key: value for key, value in record["process_tree"].items() if key != "receipt"}
    if process != expected_process or digest(campaign.root / record["output"]) != record["output_sha256"]:
        raise ValueError("scoped preparation command evidence changed")
    if proof["build"] != str(build.resolve()):
        raise ValueError("scoped preparation belongs to another build tree")
    jobs = proof.get("build_jobs", 2)
    if type(jobs) is not int or jobs < 1:
        raise ValueError("invalid scoped preparation build worker count")
    expected_command = ["cmake", "--build", str(build.resolve()), "--parallel", str(jobs), "--target", *proof["targets"]]
    if record["identity"]["command"] != expected_command:
        raise ValueError("scoped preparation command/targets changed")
    return record


def _check_preparation_result(campaign, build, prepared, env=None, *, compilation=None, fresh_context=None):
    record = _check_preparation_command(campaign, build, prepared)
    proof = record["scoped_preparation"]
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


def invalidate_rust(campaign, stage, source, build, changed, old, environment, timeout, *, force=False):
    """Use Cargo incrementally except for content edits its timestamps could miss."""
    changed = {name for name in changed if name.endswith(".rs") and Path(name).is_relative_to(source / "rust")}
    if not changed:
        return []
    stamp = build / "cargo-build.stamp"
    previous_times = old.get("context", {}).get("rust_timestamps", {}) if old else {}
    stale = {name for name in changed if force or Path(name).stat().st_mtime_ns <= previous_times.get(name, 0)
             or stamp.is_file() and Path(name).stat().st_mtime_ns <= stamp.stat().st_mtime_ns}
    if not stale:
        return []
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--offline", "--format-version", "1"],
        cwd=source, env=environment, text=True, timeout=None))
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


def _admit_missing_scope(campaign, source, build, baseline, targets, inputs, configure_args, environment):
    """A completed overlapping producer can prove artifacts, never test success."""
    owners = []
    for path in sorted(build.glob("verification-build-*.json")):
        if path == baseline or path == baseline.with_suffix(".pending.json"):
            continue
        if path.name.endswith(".pending.json"):
            return None  # Pending writers remain outside missing-scope admission.
        try:
            proof = json.loads(path.read_text(encoding="utf-8"))
            owners.append((path, proof, proof.pop("producer_sha256", None)))
        except (OSError, ValueError):
            return None
    if not owners or not any(owner.get("output_identities") and checksum for _, owner, checksum in owners):
        return None  # Older proofs did not bind output bytes.
    graph = NinjaGraph(build, targets, environment)
    before = scoped_context(campaign.root, source, build, graph, inputs, environment)
    admitted = {}
    output_identities = None
    for path, proof, producer_sha256 in owners:
        if not graph.outputs & set(proof.get("outputs", ())):
            continue
        if (proof.get("protocol") != SCOPED_PROTOCOL or proof.get("source") != str(source)
                or proof.get("build") != str(build) or proof.get("configure_args") != list(configure_args)
                or not producer_sha256 or not graph.outputs <= set(proof.get("output_identities", ()))
                or path != preparation_paths(build, proof.get("targets", ()))[0]):
            return None
        selected_outputs = {name: proof["output_identities"][name] for name in graph.outputs}
        if output_identities is not None and output_identities != selected_outputs:
            return None
        output_identities = selected_outputs
        try:
            producer = Path(proof["producer"])
            if check_preparation_owner(build, producer)["scoped_preparation"] != proof:
                return None
            # The same output closure needs only the selected scope's discovery.
            # A broader closure must rediscover all of its own consumed facts.
            if set(proof["outputs"]) == graph.outputs and proof["context"] == before:
                context = before
            else:
                owner_graph = NinjaGraph(build, proof["targets"], environment)
                if not graph.outputs <= owner_graph.outputs or any(
                        graph.inputs[name] != owner_graph.inputs.get(name) for name in graph.outputs):
                    return None
                context = scoped_context(campaign.root, source, build, owner_graph, proof["inputs"], environment)
            record = _check_preparation_result(campaign, build, producer, environment, fresh_context=context)
            executable = shutil.which(record["identity"]["command"][0], path=environment.get("PATH", ""))
            if executable is None or digest(Path(executable)) != record["identity"]["executable_sha256"]:
                return None
            if (snapshot(campaign.root, list(record["identity"]["inputs"])) != record["identity"]["inputs"]
                    or tool_snapshot(list(record["identity"]["tools"])) != record["identity"]["tools"]):
                return None
            if any(before[name] != context[name] for name in ("configuration", "environment_sha256", "rust_configuration")):
                return None
            if any(any(context[name].get(key) != value for key, value in before[name].items())
                   for name in ("files", "include_membership", "rust_timestamps")):
                return None
            if not set(before["commands"].splitlines()) <= set(context["commands"].splitlines()):
                return None
        except (OSError, ValueError, KeyError):
            return None
        if file_identities([Path(name) for name in proof["outputs"]]) != proof["output_identities"]:
            raise ValueError("completed preparation artifact bytes changed; refusing scope adoption")
        admitted[str(producer)] = producer_sha256
    return (graph, before, admitted, output_identities) if admitted else None


def _preparation_attempts(directory, stage):
    return {str(path) for path in (directory / "attempts").glob(stage + "-*")}


def _read_recovery_file(binding, expected):
    if (not isinstance(binding, dict) or binding.get("file") != str(expected)
            or expected.is_symlink() or not expected.is_file()
            or digest(expected) != binding.get("sha256")):
        raise ValueError("pending preparation recovery evidence is missing or changed")
    return json.loads(expected.read_text(encoding="utf-8"))


def _write_recovery_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = value if isinstance(value, bytes) else (json.dumps(value, indent=2) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(content)
    return {"file": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def _drained_preparation_attempt(root, directory, stage, path, number):
    """A launch sentinel is not proof: bind the supervisor's terminal receipt too."""
    path = Path(path)
    if (path.parent != directory / "attempts" or not path.name.startswith(stage + "-")
            or path.is_symlink() or not path.is_dir()):
        raise ValueError("pending preparation attempt does not belong to its invocation")
    result, process, output = (path / name for name in ("result.json", "process.json", "output.log"))
    if any(item.is_symlink() or not item.is_file() for item in (result, process, output)):
        raise ValueError("pending preparation attempt has incomplete process evidence")
    record = json.loads(result.read_text(encoding="utf-8"))
    tree = json.loads(process.read_text(encoding="utf-8"))
    output_sha256 = digest(output)
    if (not isinstance(record, dict) or not isinstance(tree, dict)
            or record.get("stage") != stage or record.get("directory") != str(root)
            or record.get("attempt_number") != number or record.get("status") not in {"passed", "failed"}
            or not record.get("finished_at") or not record.get("identity", {}).get("command")
            or record.get("output") != output.relative_to(root).as_posix()
            or record.get("output_sha256") != output_sha256
            or tree.get("drained") is not True
            or tree.get("status") not in {"completed", "failed-parent", "timeout", "interrupted"}
            or not tree.get("supervisor_pid") or not tree.get("creation")
            or record.get("process_tree") != {**tree, "receipt": str(process)}):
        raise ValueError("pending preparation command has no matching terminal drained proof")
    return {"directory": str(path), "result_sha256": digest(result),
            "process_sha256": digest(process), "output_sha256": output_sha256}


def _check_pending_preparation(campaign, source, build, targets, pending):
    try:
        if (not isinstance(pending, dict) or pending.get("pending_protocol") != "drained-preparation-v1"
                or pending.get("protocol") != SCOPED_PROTOCOL or pending.get("state") != "retryable"
                or pending.get("root") != str(campaign.root) or pending.get("source") != str(source)
                or pending.get("build") != str(build) or pending.get("targets") != targets):
            raise ValueError("pending preparation has no sealed recovery proof")
        invocation = str(uuid.UUID(pending["invocation"]))
        directory = Path(pending["campaign"])
        directory.relative_to(campaign.root / "out/validation")
        stage = pending["stage"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", stage):
            raise ValueError("invalid pending preparation stage")
        contract = json.loads((directory / "campaign.json").read_text(encoding="utf-8"))
        if contract["root"] != str(campaign.root) or stage not in contract["required"]:
            raise ValueError("pending preparation campaign changed")
        started = _read_recovery_file(pending["invocation_record"], directory / "preparation" / (invocation + ".started.json"))
        if any(pending.get(key) != value for key, value in started.items()):
            raise ValueError("pending preparation invocation changed")
        failed = _read_recovery_file(pending["failure_proof"], directory / "preparation" / (invocation + ".failed.json"))
        if failed["pending"] != {key: value for key, value in pending.items() if key != "failure_proof"}:
            raise ValueError("pending preparation failure proof changed")
        attempts = failed["attempts"]
        if (not isinstance(attempts, list)
                or failed["completion"] != ("commands-drained" if attempts else "no-command-started")
                or (directory / "running.lock").exists() or (directory / "running.lock").is_symlink()):
            raise ValueError("pending preparation still has an unproved command owner")
        seen = set(started["attempts_before"])
        for number, proof in enumerate(attempts, started["next_attempt"]):
            if proof["directory"] in seen:
                raise ValueError("pending preparation substituted a prior command attempt")
            seen.add(proof["directory"])
            if _drained_preparation_attempt(campaign.root, directory, stage, proof["directory"], number) != proof:
                raise ValueError("pending preparation command evidence changed")
        if seen != _preparation_attempts(directory, stage):
            raise ValueError("pending preparation command attempt membership changed")
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("cannot recover pending preparation: " + str(error)) from error


@contextmanager
def _pending_preparation(campaign, stage, source, build, targets, pending_path, lease):
    """Only this live, healthy lease owner may seal a failed invocation for retry."""
    recovered = None
    if pending_path.exists() or pending_path.is_symlink():
        if pending_path.is_symlink() or not pending_path.is_file():
            raise ValueError("cannot recover malformed pending preparation")
        content = pending_path.read_bytes()
        previous = json.loads(content)
        _check_pending_preparation(campaign, source, build, targets, previous)
        recovered = _write_recovery_file(
            campaign.directory / "preparation" / (str(uuid.uuid4()) + ".pending.json"), content)
    before = _preparation_attempts(campaign.directory, stage)
    prior = campaign.attempts(stage)
    invocation = str(uuid.uuid4())
    started = {"protocol": SCOPED_PROTOCOL, "pending_protocol": "drained-preparation-v1",
               "invocation": invocation, "root": str(campaign.root), "campaign": str(campaign.directory),
               "stage": stage, "source": str(source), "build": str(build), "targets": targets,
               "lease_token": lease.token, "attempts_before": sorted(before),
               "next_attempt": max((item.get("attempt_number", 0) for item in prior), default=0) + 1}
    invocation_record = _write_recovery_file(
        campaign.directory / "preparation" / (invocation + ".started.json"), started)
    pending = {**started, "invocation_record": invocation_record, "state": "running",
               "recovered_pending": recovered}
    lease.check_admission()
    write_json(pending_path, pending)
    try:
        yield pending
    except BaseException as error:
        try:
            lease.check_admission()
            if (campaign.directory / "running.lock").exists() or (campaign.directory / "running.lock").is_symlink():
                raise ValueError("campaign still has an unproved command owner")
            created = _preparation_attempts(campaign.directory, stage) - before
            records = []
            for name in created:
                record = json.loads((Path(name) / "result.json").read_text(encoding="utf-8"))
                records.append((record["attempt_number"], Path(name), record))
            records.sort(key=lambda item: item[0])
            if records:
                _, directory, record = records[-1]
                if record["status"] == "passed" and "--build" in record["identity"]["command"]:
                    record.update(status="failed", error="build provenance failed: " + str(error))
                    write_json(directory / "result.json", record)
            attempts = [_drained_preparation_attempt(campaign.root, campaign.directory, stage, path, number)
                        for number, (_, path, _) in enumerate(records, started["next_attempt"])]
            if json.loads(pending_path.read_text(encoding="utf-8")) != pending:
                raise ValueError("pending preparation changed while owned")
            sealed = {**pending, "state": "retryable"}
            proof = {"pending": sealed, "completion": "commands-drained" if attempts else "no-command-started",
                     "attempts": attempts, "error": {"type": type(error).__name__, "message": str(error)}}
            lease.check_admission()
            if before | created != _preparation_attempts(campaign.directory, stage):
                raise ValueError("preparation command attempts changed while sealing")
            binding = _write_recovery_file(
                campaign.directory / "preparation" / (invocation + ".failed.json"), proof)
            write_json(pending_path, {**sealed, "failure_proof": binding})
        except BaseException as sealing_error:
            error.add_note("preparation remains pending without retry proof: " + str(sealing_error))
        raise


# One-time owner amendment, 2026-10-21; no generic legacy admission.
S4C1_PENDING = {'linux-debug': {'marker': 'verification-build-4a0c80d1e3a0698beeea82ac.pending.json',
                 'source': '/mnt/c/Dev/Codex',
                 'sha256': '6ac1fb8649dc35f6572313ba2a41db6eeef1e34ade881143bc6bfae952646662',
                 'campaign': 'out/validation/linux-debug/s4-b-slice-5-core-gate-2026-10-18',
                 'evidence': {'campaign.json': '418ee13c04267ef4c26c55b25eaf56f31a25301fe0f664a0cef854a7874d9430',
                              'attempts/prepare-0qlifko3/output.log': '910a8c428b30c1ceadd3da1a2198e057d36c121b3d3b4e2c12843b3ba4b87c5e',
                              'attempts/prepare-0qlifko3/process.json': '5a1084120b69820cf32c210297444e799f3e9605c4a2fc16a9f5fc2e6b403ab4',
                              'attempts/prepare-0qlifko3/result.json': '0445b6e25951047125bb34fcd50b90632252477ba0dbe4f50f5a0577803e07f2',
                              'attempts/prepare-6dyxysjf/output.log': '58d9ec2986c6b41c9062f0cf041fe43409eb4bdf749ee6d894efbed2b101878a',
                              'attempts/prepare-6dyxysjf/process.json': 'ed3e61aee3de6779bb692925d5548ca25e24deeb11db7deeea7adc40f4e9dda3',
                              'attempts/prepare-6dyxysjf/result.json': '30a78f5c7ab92c2f6af5cb4ba0ac3489fc7f2afd58f6362c6bcb830456342c1a',
                              'attempts/prepare-h42d44bf/output.log': '5b9531c3850c50b6d548033d955317c66b790ecc688ef35ddc731ab91ffb0e88',
                              'attempts/prepare-h42d44bf/process.json': '91c2f2dc9897dbb3fd5387a7b58050e4daa6fbac461a8546c64ee87ab9a5561b',
                              'attempts/prepare-h42d44bf/result.json': 'bac9745278ece15d8ea7e5c6d8bea477260adfe0e285573f692aa82313314f99',
                              'attempts/prepare-ls_e8bxk/output.log': 'ada22006c703769d454295bcd246e567c37e5902391f755970a95f9ce47a568a',
                              'attempts/prepare-ls_e8bxk/process.json': 'cc9fc4b56a5851c51f72e284a84fe4fe7d094515b141337a0561cb3f406fad48',
                              'attempts/prepare-ls_e8bxk/result.json': '5532c3082dd9ce3d55f64015a57c633381f111010832a51f939481c05794d0cc',
                              'attempts/prepare-u24ootbo/output.log': '5c538a452900221f19ad2bff2436fc0a23156e508b6d1d38824e4eb788c40a9c',
                              'attempts/prepare-u24ootbo/process.json': 'f1d507ea41d8eb042c57dc47383b469881e7f43bc0ab4b8aae3abbe9f9c02d23',
                              'attempts/prepare-u24ootbo/result.json': '5ac017b3407463aa4c65f3f344f8efffc67ae00854049b90421824dd7e8da117',
                              'attempts/prepare-ya1whx8o/output.log': '89811688036109a2adf32ab5502a4f48402e2415459a527339fa5a50507c3e87',
                              'attempts/prepare-ya1whx8o/process.json': '83800362914ce64801d50a39134e22b63c9225d955e6643967a1833f6ee98143',
                              'attempts/prepare-ya1whx8o/result.json': '40aa20b5363444a7fdd68868dd5dd67131deda430bf0505703972f031c7351ee'}},
 'linux-release': {'marker': 'verification-build-bea0e3ec4c32132ca0641ce9.pending.json',
                   'source': '/mnt/c/Dev/Codex',
                   'sha256': 'a5060139283fbf2f6f9443113f1e56988434ba62691fc7addfbc294b1da10a31',
                   'campaign': 'out/validation/linux-release/s4-c-slice-1-final-restore-on-v3-2026-10-20',
                   'evidence': {'campaign.json': '4110e5814b84cd67f68872abb4cc19d54f7eb2d0a5d140b13e8221d9382aa37c',
                                'attempts/prepare-ja13lxlk/output.log': '15e9ba5aae39511d661edebe87488c0e1b98eb18e7624175f12dd55563a84ff1',
                                'attempts/prepare-ja13lxlk/process.json': '753fdeda93eaae169636c26cddd0a4ee00b2fb661a94676893eb678d735d5a31',
                                'attempts/prepare-ja13lxlk/result.json': 'b4e8dd2258c35675e412ebd01d42e1c744a26fb34fdb53a3a318a98f588650ed',
                                'attempts/prepare-x7w2cr2h/output.log': 'ab91540258338d0cb42de3f21256cf5fdc4265c83910d61c3314770b53b82914',
                                'attempts/prepare-x7w2cr2h/process.json': 'e5a414354cff33234e1088e3972f34a357353ff5decec61d630b4cebde0cebee',
                                'attempts/prepare-x7w2cr2h/result.json': '6d019a78e52ecfacb6567df318b50213da3005633e3ba953fd041de94f129e30',
                                'attempts/prepare-xy2t8kpz/output.log': '099b1b885e768be08c15d87c9d27ad8a4c61619ea89580069057c2516bf3bd38',
                                'attempts/prepare-xy2t8kpz/process.json': '1aa769d4d4d686f8c34b6b4a51bacc9c8ecbc95770621f3119fea7c165de777f',
                                'attempts/prepare-xy2t8kpz/result.json': '8b9358c4601b90da6c5c06795b2108280907cbc536b42f53d0f243caebf63bf9'}}}


def retire_s4c1_pending(root, preset):
    """Retire only the exact authorized records, without manufacturing recovery proof."""
    if os.name == "nt" or preset not in S4C1_PENDING:
        raise ValueError("legacy disposition is limited to the two named Linux trees")
    root = root.resolve()
    build = root / "out/build" / preset
    expected = S4C1_PENDING[preset]
    marker = build / expected["marker"]
    directory = root / expected["campaign"]
    with OutputLease(root, build) as lease:
        if str(root) != expected["source"]:
            raise ValueError("legacy disposition requires the exact authorized source/build path")
        if marker.is_symlink() or not marker.is_file() or digest(marker) != expected["sha256"]:
            raise ValueError("legacy marker is unknown, changed or already retired")
        if {p.name for p in build.glob("*.pending.json")} != {marker.name}:
            raise ValueError("additional pending owner blocks legacy disposition")
        for name, checksum in expected["evidence"].items():
            path = directory / name
            if path.is_symlink() or not path.is_file() or digest(path) != checksum:
                raise ValueError("legacy drainage evidence is missing or changed: " + str(path))
        attempts = sorted((directory / "attempts").glob("prepare-*"))
        if {p.name for p in attempts} != {Path(p).parts[1] for p in expected["evidence"] if p.startswith("attempts/")}:
            raise ValueError("legacy preparation attempt membership changed")
        if (directory / "running.lock").exists() or (directory / "running.lock").is_symlink():
            raise ValueError("legacy command still has an active or unproved owner")
        proofs = []
        for attempt in attempts:
            record = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
            proofs.append(_drained_preparation_attempt(root, directory, "prepare", attempt, record["attempt_number"]))
        # The Debug marker predates these commands and has no invocation binding.
        # The previous boot's retained drainage plus this boot boundary is required.
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        for attempt in attempts:
            tree = json.loads((attempt / "process.json").read_text(encoding="utf-8"))
            if preset == "linux-debug" and tree["creation"]["boot_id"] == boot:
                raise ValueError("Debug legacy disposition requires the verified later boot")
            if tree["creation"]["boot_id"] == boot:
                stat = Path("/proc") / str(tree["supervisor_pid"]) / "stat"
                if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[19] == tree["creation"]["linux_start_ticks"]:
                    raise ValueError("legacy supervisor is still active")
        campaign = Campaign(root, preset, "legacy-retirement-" + uuid.uuid4().hex, ["retire"])
        original = _write_recovery_file(campaign.directory / marker.name, marker.read_bytes())
        baseline = marker.with_name(marker.name.replace(".pending.json", ".json"))
        retained_baseline = (_write_recovery_file(campaign.directory / baseline.name, baseline.read_bytes())
                             if baseline.is_file() else None)
        record = {"disposition": "retired-unqualified", "marker": str(marker), "original": original,
                  "baseline": retained_baseline, "evidence": expected["evidence"], "campaign": str(directory),
                  "drained_attempts": proofs, "current_boot_id": boot, "lease_token": lease.token,
                  "fresh_preparation_required": True}
        result = campaign.directory / "retirement.json"
        _write_recovery_file(result, record)
        lease.check_admission()
        if digest(marker) != expected["sha256"]:
            raise ValueError("legacy marker changed during disposition")
        # Removing the old completed owner prevents reuse after marker retirement.
        if retained_baseline:
            if digest(baseline) != retained_baseline["sha256"]:
                raise ValueError("legacy completed owner changed during disposition")
            baseline.unlink()
        marker.unlink()
        return result


PREPARATION_TARGETS = {"all", "juicer", "juicer_analysis_inputs", "JuicerFilmRecipeTests", "JuicerRouteBridgeTests", "JuicerOutputColorTests"}


def prepare_targets(root, preset, targets, testing="ON"):
    """Prepare a bounded native target scope; never dispatch quality or runtime tests."""
    from qualification import PRESETS

    if (preset not in PRESETS or preset.startswith("windows") != (os.name == "nt")
            or not targets or set(targets) - PREPARATION_TARGETS or testing not in {"ON", "OFF"}
            or "all" in targets and len(targets) != 1):
        raise ValueError("preparation requires a native preset and explicit maintained build targets")
    if testing == "OFF" and (not preset.endswith("release") or set(targets) - {"all", "juicer"}):
        raise ValueError("testing OFF preparation requires Release product targets")
    root = root.resolve()
    build = root / "out/build" / preset
    cache = build / "CMakeCache.txt"
    if cache.is_file():
        home = re.search(r"^CMAKE_HOME_DIRECTORY:[^=]+=(.+)$", cache.read_text(encoding="utf-8"), re.M)
        if not home or str(Path(home[1].strip())) != str(root):
            raise ValueError("use the configured source path spelling for preparation")
    with OutputLease(root, build):
        own_pending = preparation_paths(build, targets)[1]
        if any(path != own_pending for path in build.glob("*.pending.json")):
            raise ValueError("another pending preparation must be resolved by its existing owner first")
        campaign = Campaign(root, preset, "preparation-" + uuid.uuid4().hex, ["prepare"])
        from qualification import checkpoint_inputs
        inputs = (checkpoint_inputs(root, build, "prepare") if targets == ["all"] and testing == "ON"
                  else ["scripts/verify.py", "scripts/build_evidence.py"])
        _, prepared = prepare_build(campaign, "prepare", source=root, build=build,
                                    inputs=[name for name in inputs if not name.startswith("out/build/")], timeout=None,
                                    configure_args=["-DBUILD_TESTING=" + testing], targets=targets,
                                    checkpoint=True, reconfigure=True)
        check_scoped_preparation(campaign, build, prepared)
        return prepared


def prepare_build(campaign, stage, *, source, build, inputs, timeout,
                  configure_args=(), targets=(), env=None, tools=(), resume=None, checkpoint=False, reconfigure=False):
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
    environment = command_environment(os.environ if env is None else env)
    workers = environment.get("CMAKE_BUILD_PARALLEL_LEVEL", "")
    if workers and (not workers.isascii() or not workers.isdecimal() or int(workers) < 1):
        raise ValueError("CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer")
    jobs = int(workers) if workers else min(os.cpu_count() or 2, 8)
    # Scheduling is bound by the explicit command, not an input available to
    # configure scripts, compilers or build scripts whose outputs we retain.
    environment.pop("CMAKE_BUILD_PARALLEL_LEVEL", None)
    build_command = ["cmake", "--build", str(build), "--parallel", str(jobs), "--target", *targets]
    campaign.check_run_request(stage, build_command, inputs=inputs, timeout=timeout)
    with OutputLease(campaign.root, build) as lease, ExitStack() as preparation:
        configured = None
        baseline, pending_path = preparation_paths(build, targets)
        interrupted = pending_path.exists() or pending_path.is_symlink()
        old = None
        if baseline.is_file():
            candidate = json.loads(baseline.read_text(encoding="utf-8"))
            if candidate.get("protocol") == SCOPED_PROTOCOL:
                old = candidate
        previous_outputs = {}
        superseded_artifacts = {}
        superseded_producers = {}
        force_rust = False
        unbound = bool(old and (not old.get("producer_sha256")
                               or not old.get("output_identities")
                               or set(old["output_identities"]) != set(old["outputs"])))
        if old and not interrupted and not unbound:
            check_preparation_owner(build, Path(old["producer"]))
            _check_preparation_command(campaign, build, old["producer"])
            previous_outputs = file_identities([Path(name) for name in old["outputs"]])
            if any(old["output_identities"].get(name) != identity for name, identity in previous_outputs.items()):
                # A direct build may have replaced outputs after an authored edit.
                # Rebuild the whole scope; these bytes cannot establish any reuse.
                files = old["context"]["files"]
                if (file_identities([Path(name) for name in files]) == files
                        and old.get("configure_args") == list(configure_args)):
                    raise ValueError("completed preparation artifact bytes changed; refusing renewal")
                superseded_artifacts[old["producer"]] = previous_outputs
                superseded_producers[old["producer"]] = old["producer_sha256"]
                previous_outputs = {}
                force_rust = True
        build.mkdir(parents=True, exist_ok=True)
        pending = preparation.enter_context(
            _pending_preparation(campaign, stage, source, build, targets, pending_path, lease))
        cache = build / "CMakeCache.txt"
        admission_start = time.perf_counter()
        admission = (_admit_missing_scope(campaign, source, build, baseline, targets, inputs, configure_args, environment)
                     if old is None and not interrupted and cache.is_file() and not reconfigure else None)
        admission_seconds = time.perf_counter() - admission_start
        configure = reconfigure or not cache.is_file() or (old is None and admission is None)
        if old and old.get("configure_args") != list(configure_args):
            configure = True
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
        if admission:
            graph, before, _, admitted_outputs = admission
        else:
            graph = NinjaGraph(build, targets, environment)
            before = scoped_context(campaign.root, source, build, graph, inputs, environment)
        changed = set()
        if old:
            previous, actual = old["context"]["files"], before["files"]
            changed = {name for name in previous.keys() | actual.keys() if previous.get(name) != actual.get(name)}
        broad = interrupted or unbound or force_rust or (old is None and admission is None)
        if old and any(old["context"].get(name) != before.get(name) for name in ("configuration", "include_membership", "commands", "environment_sha256", "rust_configuration")):
            broad = True
        rust_files = {name: identity for name, identity in before["files"].items() if name.endswith(".rs") and Path(name).is_relative_to(source / "rust")}
        if unbound:
            changed.update(rust_files)
        current_rust = rust_context(source, build, before)
        if old:
            changed.update(rust_changes(current_rust, rust_context(source, build, old["context"])))
        if force_rust:
            changed.update(rust_files)
        archive_owners = []
        if (old is None or interrupted) and admission is None:
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
                    archives = archive_names & set(proof["outputs"])
                    retained = archives | {name for name in proof["outputs"]
                                           if Path(name).name == "cargo-build.stamp" and Path(name).is_file()}
                    if not proof.get("producer_sha256") or not retained <= set(proof.get("output_identities", ())):
                        continue
                    check_preparation_owner(build, Path(proof["producer"]))
                    _check_preparation_command(campaign, build, proof["producer"])
                    identities = {name: proof["output_identities"][name] for name in retained}
                    actual = file_identities([Path(name) for name in retained])
                    if actual != identities:
                        if current_rust == rust_context(source, build, proof["context"]):
                            raise ValueError("completed preparation artifact bytes changed; refusing archive reuse")
                        # Changed Cargo inputs require a fresh active-package build,
                        # even if an intervening build left misleading timestamps.
                        superseded_artifacts[proof["producer"]] = actual
                        superseded_producers[proof["producer"]] = proof["producer_sha256"]
                        force_rust = True
                        continue
                    shared.append((owner, proof, identities))
            # All valid overlapping owners must describe the same archive inputs.
            # No owner (or an interrupted write) is unknown, even for clean HEAD.
            contexts = [rust_context(source, build, proof["context"]) for _, proof, _ in shared]
            if not force_rust and not interrupted and contexts and all(value == contexts[0] for value in contexts):
                changed.update(rust_changes(current_rust, contexts[0]))
                if current_rust == contexts[0]:
                    archive_owners = shared
            elif any(Path(name).is_file() for name in archive_names):
                changed.update(rust_files)
        mutable = {name for name in graph.outputs if Path(name).suffix.lower() not in {".a", ".lib"} and Path(name).name != "cargo-build.stamp"}
        affected = mutable if broad else graph.affected(changed) & mutable
        potential = set(affected) | {name for name in graph.outputs if not Path(name).is_file()}
        if any(name.endswith(".rs") for name in changed):
            potential.update(graph.affected(changed))
        retained_names = previous_outputs.keys() & graph.outputs - graph.affected(potential | changed)
        retained_outputs = {name: previous_outputs[name] for name in retained_names}
        invalidate_preparation_owners(build, baseline, potential)
        rust_invalidated = invalidate_rust(campaign, stage, source, build, changed, old, environment, timeout, force=force_rust)
        invalidated = graph.invalidate(affected)
        pending["outputs"] = sorted(mutable)
        write_json(pending_path, pending)
        for attempt in range(2):
            local, external = split_inputs(campaign.root, [Path(name) for name in before["files"]])
            built = campaign.run(stage, build_command, inputs=local, tools=[*tools, *external], env=environment,
                                 timeout=timeout, repeat_reason="build only explicit owning targets and dependencies",
                                 vctip_executable=windows_vctip(environment))
            fresh_graph = NinjaGraph(build, targets, environment)
            after = scoped_context(campaign.root, source, build, fresh_graph, inputs, environment)
            if file_identities([Path(name) for name in before["files"]]) != before["files"]:
                raise ValueError("relevant inputs mutated during preparation")
            if before == after:
                break
            if attempt or any(before[name] != after[name] for name in before if name != "files"):
                raise ValueError("unresolved dependency/configuration change during preparation")
            discovered = set(after["files"]) ^ before["files"].keys()
            discovered_outputs = fresh_graph.affected(discovered)
            retained_outputs = {name: identity for name, identity in retained_outputs.items() if name not in discovered_outputs}
            invalidated += fresh_graph.invalidate(discovered_outputs & mutable)
            before, graph = after, fresh_graph
        output_identities = file_identities([Path(name) for name in fresh_graph.outputs])
        if any(output_identities.get(name) != identity for name, identity in retained_outputs.items()):
            raise ValueError("retained artifact bytes changed during preparation")
        if retained_outputs:
            if (digest(Path(old["producer"])) != old["producer_sha256"]
                    or json.loads(baseline.read_text(encoding="utf-8")) != old):
                raise ValueError("retained preparation owner changed during renewal")
            _check_preparation_command(campaign, build, old["producer"])
        if admission and output_identities != admitted_outputs:
            raise ValueError("admitted artifacts changed during missing-scope preparation")
        if admission:
            for producer, checksum in admission[2].items():
                if digest(Path(producer)) != checksum:
                    raise ValueError("admitted preparation command receipt changed")
                check_preparation_owner(build, producer)
                _check_preparation_command(campaign, build, producer)
        for owner, previous, identities in archive_owners:
            if any(output_identities.get(name) != identity for name, identity in identities.items()):
                raise ValueError("retained shared Rust artifact bytes changed during preparation")
            if digest(Path(previous["producer"])) != previous["producer_sha256"]:
                raise ValueError("retained shared Rust preparation receipt changed")
            owner_pending = owner.with_suffix(".pending.json")
            if owner_pending.exists() or owner_pending.is_symlink():
                raise ValueError("retained shared Rust preparation owner changed")
            # This writer may supersede the native owner while retaining its archive.
            if potential & set(previous["outputs"]):
                if owner.exists() or owner.is_symlink():
                    raise ValueError("retained shared Rust preparation owner changed")
            elif not owner.is_file() or json.loads(owner.read_text(encoding="utf-8")) != previous:
                raise ValueError("retained shared Rust preparation owner changed")
            _check_preparation_command(campaign, build, previous["producer"])
        for producer, checksum in superseded_producers.items():
            if digest(Path(producer)) != checksum:
                raise ValueError("superseded preparation command receipt changed")
            _check_preparation_command(campaign, build, producer)
        proof = {"protocol": SCOPED_PROTOCOL, "producer": str(built.resolve()), "source": str(source), "build": str(build),
                 "targets": targets, "build_jobs": jobs, "inputs": inputs, "configure_args": list(configure_args), "context": after,
                 "output_identities": output_identities,
                 "outputs": sorted(fresh_graph.outputs), "invalidated": invalidated, "rust_invalidated_packages": rust_invalidated}
        record = json.loads(built.read_text(encoding="utf-8"))
        record["preparation_admission"] = {"scope_missing": old is None, "artifact_owners": admission[2] if admission else {},
                                           "archive_owners": {proof["producer"]: proof["producer_sha256"] for _, proof, _ in archive_owners},
                                           "superseded_artifacts": superseded_artifacts,
                                           "superseded_producers": superseded_producers,
                                           "configured": configured is not None, "seconds": admission_seconds,
                                           "recovered_pending": pending["recovered_pending"]}
        record["scoped_preparation"] = proof
        write_json(built, record)
        # The writer just established this post-build context. Only its owned
        # result was written since; independent readers always rediscover it.
        _check_preparation_result(campaign, build, built, environment, fresh_context=after)
        # Receipt provenance stays outside the stable preparation facts consumed
        # by checkpoint identity; the current owner still binds the exact bytes.
        write_json(baseline, {**proof, "producer_sha256": digest(built)})
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
        inventory = json.loads(subprocess.check_output([*command, "--show-only=json-v1"], cwd=campaign.root, env=env, text=True, timeout=None))
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
