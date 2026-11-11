"""Native compiler dependencies and ordered header-consumer selection."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess


class NinjaGraph:
    """Compiler-reported dependency edges used by native header selection."""

    def __init__(self, build: Path):
        self.build = build.resolve()
        cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
        match = re.search(r"^CMAKE_MAKE_PROGRAM:[^=]+=(.+)$", cache, re.M)
        if not match:
            raise ValueError("Ninja configuration is required")
        self.ninja = match[1]
        self.compiler_inputs = {}
        current = None
        dependencies = subprocess.check_output(
            [self.ninja, "-C", str(self.build), "-t", "deps"],
            text=True,
            encoding="utf-8",
            timeout=None,
        )
        for line in dependencies.splitlines():
            if ": #deps " in line:
                current = str(self.path(line.split(": #deps ", 1)[0]))
                self.compiler_inputs[current] = set()
            elif line.startswith("    ") and current is not None:
                self.compiler_inputs[current].add(str(self.path(line.strip())))

    def path(self, name):
        return Path(os.path.abspath(self.build / name))


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


def header_consumers(root, build, headers, *, require_compiled=True, compilation=None):
    """Combine compiler dependencies and ordered includes across configured variants."""
    compilation = compilation or CompilationInputs(root, build)
    graph = NinjaGraph(build)
    entries = compilation.rows
    wanted = {str(compilation.resolve(root / name)) for name in headers}
    outputs = [
        str(compilation.resolve(Path(row["directory"]) / row["output"]).relative_to(build))
        for row in entries
        if row.get("output")
    ]
    current = subprocess.run(
        [graph.ninja, "-C", str(build), "-n", "-d", "explain", *outputs],
        text=True,
        capture_output=True,
        timeout=None,
        check=True,
    )
    changed_commands = {
        str((build / name).resolve())
        for name in re.findall(r"^ninja explain: command line changed for (.+)$", current.stderr, re.M)
    }
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
                    candidate.is_relative_to(root)
                    and candidate.is_file()
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
