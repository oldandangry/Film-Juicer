"""Ninja graph and actual compiler dependencies for build provenance and selection."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time

from execution import digest


class NinjaGraph:
    def __init__(self, build: Path, targets=(), env=None):
        self.build = build.resolve()
        cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
        match = re.search(r"^CMAKE_MAKE_PROGRAM:[^=]+=(.+)$", cache, re.M)
        if not match:
            raise ValueError("Ninja configuration is required")
        self.ninja = match[1]
        self.environment = env
        graph = self.query("graph", *(targets or ["all"]))
        labels, rules, edges = {}, set(), []
        for line in graph.splitlines():
            node = re.fullmatch(r'"([^"]+)" \[label=("(?:[^"\\]|\\.)*")(.*)\]', line)
            edge = re.match(r'"([^"]+)" -> "([^"]+)"', line)
            if node:
                labels[node[1]] = json.loads(node[2])
                if "shape=ellipse" in node[3]:
                    rules.add(node[1])
            elif edge:
                edges.append(
                    (edge[1], edge[2], " phony" in line, "style=dotted" in line)
                )
        incoming = {}
        phony_edges = {(left, right) for left, right, phony, _ in edges if phony}
        for left, right, _, order_only in edges:
            if not order_only:
                incoming.setdefault(right, []).append(left)
        self.inputs, self.outputs = {}, set()
        for identifier, name in labels.items():
            if identifier in rules:
                continue
            dependencies = set()
            for parent in incoming.get(identifier, []):
                if parent in rules:
                    dependencies.update(
                        labels[p] for p in incoming.get(parent, []) if p not in rules
                    )
                    if labels[parent] != "phony":
                        self.outputs.add(name)
                else:
                    dependencies.add(labels[parent])
                    if (parent, identifier) not in phony_edges:
                        self.outputs.add(name)
            self.inputs[name] = dependencies
        # CMake wraps always-run utility commands in a matching phony target.
        # Their reserved CMakeFiles/<target> node is symbolic, not a file output.
        symbolic = {
            child for name, children in self.inputs.items() for child in children
            if Path(child).parent.name == "CMakeFiles"
            and Path(name) == Path(child).parent.parent / Path(child).name
            and name not in self.outputs
        }
        self.outputs.difference_update(symbolic)
        self.compiler_inputs = {}
        current = None
        for line in self.query("deps").splitlines():
            if ": #deps " in line:
                current = line.split(": #deps ", 1)[0]
                self.compiler_inputs[current] = set()
            elif line.startswith("    ") and current is not None:
                self.compiler_inputs[current].add(line.strip())
        for name in self.inputs:
            self.inputs[name].update(self.compiler_inputs.get(name, ()))
        self.inputs = {
            str(self.path(name)): {str(self.path(child)) for child in children}
            for name, children in self.inputs.items()
        }
        self.outputs = {str(self.path(name)) for name in self.outputs}
        self.compiler_inputs = {
            str(self.path(name)): {str(self.path(child)) for child in children}
            for name, children in self.compiler_inputs.items()
        }
        self.rust_sources = rust_product_sources(self.build)
        if self.rust_sources is not None:
            for name in self.outputs:
                if Path(name).name in {
                    "film_juicer_plugin.lib",
                    "libfilm_juicer_plugin.a",
                }:
                    self.inputs[name] = {
                        child
                        for child in self.inputs[name]
                        if Path(child).suffix != ".rs"
                    } | set(self.rust_sources)
        self.parents = {}
        for parent, children in self.inputs.items():
            for child in children:
                self.parents.setdefault(child, set()).add(parent)
        self.leaf_cache = {}
        self.commands = self.query("commands", *(targets or ["all"]))

    def query(self, *arguments):
        return subprocess.check_output(
            [self.ninja, "-C", str(self.build), "-t", *arguments],
            env=self.environment,
            text=True,
            encoding="utf-8",
            timeout=60,
        )

    def path(self, name):
        return Path(os.path.abspath(self.build / name))

    def leaves(self, node, seen=None):
        if node in self.leaf_cache:
            return self.leaf_cache[node]
        seen = set() if seen is None else seen
        if node in seen:
            return set()
        seen.add(node)
        children = self.inputs.get(node, set())
        if not children:
            result = {self.path(node)} if self.path(node).is_file() else set()
        else:
            result = set().union(*(self.leaves(child, seen) for child in children))
        self.leaf_cache[node] = result
        return result

    def affected(self, paths):
        pending = [
            name
            for path in paths
            for name in (str(Path(os.path.abspath(path))), str(Path(path).resolve()))
        ]
        reached = set()
        while pending:
            name = pending.pop()
            if name not in reached:
                reached.add(name)
                pending.extend(self.parents.get(name, ()))
        return reached & self.outputs

    def artifacts(self):
        return sorted(
            {self.path(name) for name in self.outputs if self.path(name).is_file()}
        )

    def source_files(self):
        return sorted(set().union(*(self.leaves(name) for name in self.outputs)))

    def invalidate(self, names):
        removed = []
        for name in sorted(names):
            path = self.path(name)
            if path.is_file():
                if not path.is_relative_to(self.build):
                    raise ValueError(
                        f"unresolved generated output outside build directory: {path}"
                    )
                path.unlink()
                removed.append(str(path))
        return removed


def file_identities(paths, timings=None):
    """Freshly resolve and read each file; do not retain identities across calls."""
    paths = sorted(set(paths))

    def resolve_paths():
        parents = {}

        def resolve(path):
            if path not in parents:
                # Windows resolution owns junctions and canonical name casing.
                # On POSIX, share common parent traversal within this read only.
                if os.name == "nt" or path.parent == path or path.name == ".." or path.is_symlink():
                    parents[path] = path.resolve(strict=True)
                else:
                    parents[path] = resolve(path.parent) / path.name
            return parents[path]

        return {path: resolve(path) for path in paths if path.is_file()}

    started = time.perf_counter()
    before = resolve_paths()
    resolved_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = {str(path): {"resolved": str(resolved), "sha256": digest(path)} for path, resolved in before.items()}
    hash_seconds = time.perf_counter() - started
    started = time.perf_counter()
    # This is a new traversal: a directory/file symlink retargeted while reading
    # cannot be hidden by the entry boundary's shared parent resolutions.
    if resolve_paths() != before:
        raise ValueError("input path changed while hashing")
    resolved_seconds += time.perf_counter() - started
    if timings is not None:
        timings["path_resolution_seconds"] = timings.get("path_resolution_seconds", 0) + resolved_seconds
        timings["content_hash_seconds"] = timings.get("content_hash_seconds", 0) + hash_seconds
    return result


class CompilationInputs:
    """Ordered native include discovery for one validation boundary only."""

    def __init__(self, root, build):
        import importlib.util
        import sys

        self.root = root
        spec = importlib.util.spec_from_file_location(
            "juicer_compilation_inputs", Path(__file__).with_name("check-quality.py")
        )
        self.quality = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.quality
        spec.loader.exec_module(self.quality)
        self.rows = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
        self.resolved = {}
        self.includes, self.edges, self.discovered = {}, {}, {}
        self.entries, self.searches = {}, {}
        self.file_presence = {}

    def resolve(self, path):
        if path not in self.resolved:
            if os.name == "nt" or path.parent == path or path.name == ".." or path.is_symlink():
                self.resolved[path] = path.resolve()
            else:
                self.resolved[path] = self.resolve(path.parent) / path.name
        return self.resolved[path]

    def entry(self, row):
        path = self.resolve(Path(row["directory"]) / row["file"])
        arguments = tuple(row.get("arguments", ())) or self.quality.split_compiler_command(row["command"])
        key = (path, row["directory"], arguments)
        if key not in self.entries:
            self.entries[key] = self.quality.CompilationEntry(
                path.relative_to(self.root).as_posix(), row.get("command", ""), Path(row["directory"]), arguments
            )
        return self.entries[key]

    def is_file(self, path):
        if path not in self.file_presence:
            self.file_presence[path] = path.is_file()
        return self.file_presence[path]

    def search(self, entry):
        key = (entry.directory, entry.arguments)
        if key not in self.searches:
            self.searches[key] = self.quality.include_search(self.root, entry, resolve=self.resolve)
        return self.searches[key]

    def paths(self, row):
        entry = self.entry(row)
        search = self.search(entry)
        key = (entry.path, search)
        if key not in self.discovered:
            self.discovered[key] = self.quality.included_paths(
                self.root, entry, self.includes, self.edges, search=search, resolve=self.resolve, is_file=self.is_file
            )
        return self.discovered[key]


def include_membership(build, graph, source_root, compilation=None):
    """Watch current ordered include namespaces as well as compiler-read files.

    A new shadow header must invalidate proof even when no old dependency has
    changed. This inventory is invocation-local; output/evidence JSON is not a
    native include input. Mutable build-root include scopes remain unresolved.
    """
    database = build / "compile_commands.json"
    if not database.is_file():
        return {}
    compilation = compilation or CompilationInputs(source_root, build)
    sources = {compilation.resolve(path) for path in graph.source_files()}
    directories = set()
    for row in compilation.rows:
        path = Path(row["directory"]) / row["file"]
        if compilation.resolve(path) not in sources:
            continue
        search = compilation.search(compilation.entry(row))
        directories.update([*search.quoted, *search.ordinary, path.parent])
    result = {}
    outputs = {Path(name) for name in graph.outputs}
    for directory in sorted(directories):
        # Observe mutable project/generated namespaces, not compiler installations.
        if not directory.is_relative_to(source_root) and not directory.is_relative_to(build):
            continue
        if build == directory or build.is_relative_to(directory):
            raise ValueError(
                "unresolved include scope contains mutable build outputs: "
                + str(directory)
            )
        # Declared outputs can disappear during invalidation. Keep watching
        # their include directory for new, unowned shadow headers.
        members, visited = [], set()
        for parent, children, files in os.walk(directory, followlinks=True):
            children[:] = sorted(
                name
                for name in children
                if name not in {"__pycache__", ".git", ".ruff_cache"}
            )
            resolved = Path(parent).resolve()
            if resolved in visited:
                children[:] = []
                continue
            visited.add(resolved)
            members.append([str(Path(parent).relative_to(directory)), str(resolved)])
            members.extend(
                [str((Path(parent) / name).relative_to(directory)), None]
                for name in sorted(files) if Path(parent) / name not in outputs
            )
        result[str(directory)] = sorted(members, key=lambda item: item[0])
    return result


def configuration_outputs(build, environment=None):
    """CMake declares every generated configuration file as a RERUN_CMAKE output."""
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    ninja = re.search(r"^CMAKE_MAKE_PROGRAM:[^=]+=(.+)$", cache, re.M)[1]
    payload = subprocess.check_output(
        [ninja, "-C", str(build), "-t", "targets", "all"],
        env=environment,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    names = [
        line.rsplit(": ", 1)[0]
        for line in payload.splitlines()
        if line.endswith(": RERUN_CMAKE")
    ]
    if not names:
        raise ValueError("unsupported CMake regeneration outputs")
    outputs = {Path(os.path.abspath(build / name)) for name in names}
    manifest = build / "verification-config-outputs.json"
    if manifest.is_file():
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("configuration producer evidence changed; configure again")
        if document.get("protocol") != "scoped-development-v1":
            return sorted(outputs)
        producer = Path(document["producer"])
        record = json.loads(producer.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or digest(producer) != document["producer_sha256"]:
            raise ValueError("configuration producer evidence changed; configure again")
        tree = record.get("process_tree")
        if not isinstance(tree, dict) or not isinstance(tree.get("receipt"), str):
            raise ValueError("configuration producer evidence changed; configure again")
        evidence = record.get("configuration_evidence")
        expected = {key: document[key] for key in ("outputs", "trace", "trace_sha256")}
        process = json.loads(
            Path(tree["receipt"]).read_text(encoding="utf-8")
        )
        if (
            evidence != expected
            or record.get("status") != "passed"
            or record.get("returncode") != 0
            or tree.get("drained") is not True
            or tree.get("status") != "completed"
            or tree.get("returncode") != 0
            or not isinstance(process, dict)
            or process.get("drained") is not True
            or process != {key: value for key, value in tree.items() if key != "receipt"}
            or digest(Path(document["trace"])) != document["trace_sha256"]
            or digest(Path(record["directory"]) / record["output"])
            != record["output_sha256"]
        ):
            raise ValueError("configuration producer evidence changed; configure again")
        outputs.update(Path(name) for name in document["outputs"])
    return sorted(outputs)


def rust_product_sources(build):
    """Cargo's compiler-produced depfile distinguishes product inputs from test-only stamp inputs."""
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    profile = (
        "release"
        if re.search(r"^CMAKE_BUILD_TYPE:STRING=Release$", cache, re.M)
        else "debug"
    )
    target = "x86_64-pc-windows-msvc" if os.name == "nt" else "x86_64-unknown-linux-gnu"
    name = "film_juicer_plugin.d" if os.name == "nt" else "libfilm_juicer_plugin.d"
    path = build / "cargo" / target / profile / name
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").replace("\\\n", "").splitlines()
    first = lines[0] if lines else ""
    parts = re.split(r":\s+", first, maxsplit=1)
    if len(parts) != 2 or "\\ " in parts[1] or "\\#" in parts[1]:
        raise ValueError("unresolved Cargo depfile representation: " + str(path))
    return sorted({str(Path(value).resolve()) for value in parts[1].split()})


def header_consumers(root, build, headers, *, require_compiled=True, compilation=None):
    """Combine compiler dependencies and ordered includes across configured variants."""
    compilation = compilation or CompilationInputs(root, build)
    graph = NinjaGraph(build)
    entries = compilation.rows
    wanted = {str(compilation.resolve(root / name)) for name in headers}
    outputs = [str(compilation.resolve(Path(row["directory"]) / row["output"]).relative_to(build)) for row in entries if row.get("output")]
    current = subprocess.run([graph.ninja, "-C", str(build), "-n", "-d", "explain", *outputs],
                             env=graph.environment, text=True, capture_output=True, timeout=60, check=True)
    changed_commands = {str((build / name).resolve()) for name in re.findall(r"^ninja explain: command line changed for (.+)$", current.stderr, re.M)}
    targets, sources, found, macros = set(), set(), set(), {}
    for row in entries:
        if not row.get("output"):
            raise ValueError("unresolved compile variant without an output: " + row["file"])
        output = str(compilation.resolve(Path(row["directory"]) / row["output"]))
        path = compilation.resolve(Path(row["directory"]) / row["file"])
        relative = path.relative_to(root).as_posix()
        discovered = {str(root / name) for name in compilation.paths(row)}
        known = graph.compiler_inputs.get(output) if Path(output).is_file() and output not in changed_commands else None
        opaque = False
        for name in discovered | graph.compiler_inputs.get(output, set()):
            if name not in macros:
                candidate = Path(name)
                macros[name] = (
                    candidate.is_relative_to(root) and candidate.is_file()
                    and bool(re.search(r"^\s*#\s*include\s+[^<\"\s]", candidate.read_text(encoding="utf-8"), re.M))
                )
            opaque |= macros[name]
        # Opaque includes remain conservative compile candidates even with warm
        # depfiles: restored macro edits can change their consumer membership.
        if opaque:
            targets.add(str(Path(output).relative_to(build)))
        found.update(wanted & (known or set()))
        if wanted & (discovered | (known or set())):
            targets.add(str(Path(output).relative_to(build)))
            sources.add(relative)
    if not targets or require_compiled and found != wanted:
        raise ValueError("unresolved shared header without configured consumers")
    return sorted(targets), sorted(sources)
