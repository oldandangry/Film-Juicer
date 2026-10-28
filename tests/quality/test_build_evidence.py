"""Behavior controls through real native/Cargo preparation and CTest entry points."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/ctest"))
from control_artifacts import MARKER, record_fixture
import build_evidence
from execution import Campaign, OutputLease, digest, write_json

ROOT = Path(__file__).resolve().parents[2]


class CompilationInputsTests(unittest.TestCase):
    def test_fresh_boundary_observes_include_directory_symlink_retarget(self):
        from build_dependencies import CompilationInputs

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            build = root / "build"
            build.mkdir()
            for name in ("first", "second"):
                (root / name).mkdir()
            (root / "first/entry.h").write_text("// no selected consumer\n")
            (root / "second/entry.h").write_text('#include "../selected.h"\n')
            (root / "selected.h").write_text("// selected\n")
            (root / "main.cpp").write_text('#include <entry.h>\n')
            link = root / "current"

            def link_to(directory):
                if os.name == "nt":
                    subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(directory)], check=True, capture_output=True, timeout=10)
                else:
                    link.symlink_to(directory, target_is_directory=True)

            link_to(root / "first")
            row = {"file": "main.cpp", "directory": str(root), "arguments": ["c++", "-I" + str(link)]}
            (build / "compile_commands.json").write_text(json.dumps([row]))
            first = CompilationInputs(root, build)
            self.assertEqual(first.paths(row), {"main.cpp", "first/entry.h"})
            os.rmdir(link) if os.name == "nt" else link.unlink()
            link_to(root / "second")
            final = CompilationInputs(root, build)
            self.assertEqual(final.paths(row), {"main.cpp", "second/entry.h", "selected.h"})

    def test_ordered_variants_and_fresh_boundary_observe_changed_includes(self):
        from build_dependencies import CompilationInputs

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            build = root / "build"
            build.mkdir()
            for directory in ("first", "second"):
                (root / directory).mkdir()
                (root / directory / "chosen.h").write_text("// " + directory)
            source = root / "main.cpp"
            source.write_text('#include <chosen.h>\n')
            forced = root / "forced.h"
            forced.write_text("// forced\n")
            rows = [
                {"file": "main.cpp", "directory": str(root), "arguments": ["c++", "-I" + str(root / a), "-I" + str(root / b), *extra]}
                for a, b, extra in [("first", "second", []), ("second", "first", ["-include", "forced.h"])]
            ]
            (build / "compile_commands.json").write_text(json.dumps(rows))
            boundary = CompilationInputs(root, build)
            self.assertEqual(boundary.paths(rows[0]), {"main.cpp", "first/chosen.h"})
            self.assertEqual(boundary.paths(rows[1]), {"main.cpp", "second/chosen.h", "forced.h"})
            self.assertEqual(boundary.paths(rows[0]), {"main.cpp", "first/chosen.h"})
            stamp = source.stat()
            source.write_text('#include <other.h>\n')
            os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            (root / "first/other.h").write_text("// next boundary\n")
            fresh = CompilationInputs(root, build)
            self.assertEqual(fresh.paths(rows[0]), {"main.cpp", "first/other.h"})
            # Membership in the earlier namespace changes lookup even when the
            # source bytes and the formerly resolved header remain unchanged.
            (root / "second/other.h").write_text("// new earlier match\n")
            final = CompilationInputs(root, build)
            self.assertEqual(final.paths(rows[1]), {"main.cpp", "second/other.h", "forced.h"})


class BuildEvidenceTests(unittest.TestCase):
    def setUp(self):
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        artifact_root = os.environ.get("JUICER_TEST_ARTIFACT_DIR")
        retain_fixtures = bool(os.environ.get("JUICER_TEST_RETAIN_FIXTURES"))
        parent = ((Path(artifact_root) if artifact_root else ROOT / "out/validation" / preset)
                  / "build-evidence-controls")
        parent.mkdir(parents=True, exist_ok=True)
        temporary_parent = str(parent.resolve())
        if os.name == "nt" and not temporary_parent.startswith("\\\\?\\"):
            # Cargo creates incremental objects beyond MAX_PATH even when the
            # host disables ordinary long paths. Keep this spelling in cleanup.
            temporary_parent = ("\\\\?\\UNC\\" + temporary_parent[2:]
                                if temporary_parent.startswith("\\\\") else "\\\\?\\" + temporary_parent)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_parent, delete=False)
        self.root = (parent / Path(self.temporary.name).name).resolve()

        def finish_fixture():
            outcome = record_fixture(self.root, self)
            failed = outcome != "passed"
            blocked = list(self.root.glob("out/verification-locks/*/owner.json"))
            if failed or blocked or retain_fixtures:
                # Keep configured paths and unproved owners in place for diagnosis.
                retained = self.root / MARKER
                print(f"Retained build-evidence fixture: {retained}", file=sys.stderr)
            else:
                try:
                    self.temporary.cleanup()
                except BaseException:
                    record_fixture(self.root, self, "failed")
                    raise

        self.addCleanup(finish_fixture)
        record_fixture(self.root, self, "running")
        self.source = self.root / "source"
        self.source.mkdir()
        self.build = self.root / "build"
        self.campaign = Campaign(self.root, preset, "control", ["prepare", "tests"])
        files = {
            "CMakeLists.txt": "cmake_minimum_required(VERSION 3.24)\nproject(ContentProof LANGUAGES C CXX)\n"
            "set(CMAKE_CXX_STANDARD 20)\nset(CMAKE_C_STANDARD 11)\n"
            'enable_testing()\nif(ENABLE_RUST)\ninclude("'
            + (self.source / "cmake/JuicerRust.cmake").as_posix()
            + '")\nendif()\n'
            "configure_file(binding.h.in generated/binding.h COPYONLY)\n"
            "add_executable(behavior main.cpp native.cpp abi.c)\n"
            'target_include_directories(behavior PRIVATE "${CMAKE_BINARY_DIR}/shadow" "${CMAKE_BINARY_DIR}/generated")\n'
            "if(ENABLE_RUST)\ntarget_link_libraries(behavior PRIVATE film_juicer_rust)\nelse()\ntarget_sources(behavior PRIVATE rust_stub.c)\nendif()\n"
            "add_test(NAME behavior COMMAND behavior 10)\n",
            "main.cpp": '#include <cstdio>\n#include <cstdlib>\n#include "middle.h"\n'
            'int native();\nextern "C" int rust_value();\n'
            "int main(int argc,char** argv){int value=native()+header()+rust_value();"
            'std::printf("value=%d\\n",value);return argc!=2 || value!=std::atoi(argv[1]);}\n',
            "rust_stub.c": "int rust_value(void){return 3;}\n",
            "native.cpp": "int native(){return 1;}\n",
            "middle.h": '#include "detail.h"\n',
            "detail.h": '#include "binding.h"\ninline int header(){return 2+BINDING;}\n',
            "binding.h.in": "#define BINDING 4\n",
            "abi.c": 'extern int rust_value(void);\n_Static_assert(_Generic(&rust_value,int(*)(void):1,default:0),"C11 witness");\n',
        }
        (self.source / "cmake").mkdir()
        (self.source / "cmake/JuicerRust.cmake").write_bytes(
            (ROOT / "cmake/JuicerRust.cmake").read_bytes()
        )
        for name, content in files.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.args = [
            "-G",
            "Ninja",
            "-DBUILD_TESTING=ON",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DENABLE_RUST=OFF",
        ]
        if os.name == "nt":
            self.args += [
                "-DCMAKE_C_COMPILER=clang-cl.exe",
                "-DCMAKE_CXX_COMPILER=clang-cl.exe",
                "-DCMAKE_LINKER_TYPE=LLD",
            ]
        else:
            self.args += [
                "-DCMAKE_C_COMPILER=/usr/bin/gcc-13",
                "-DCMAKE_CXX_COMPILER=/usr/bin/g++-13",
            ]

    def enable_rust(self):
        self.assertIn("-DENABLE_RUST=OFF", self.args)
        self.assertFalse((self.source / "Cargo.toml").exists())
        # Construct once before authored mutations; preparation must not repair
        # Rust inputs that a test later edits or deletes.
        files = {
            "Cargo.toml": '[workspace]\nmembers=["rust/film-juicer-core","rust/film-juicer-plugin"]\nresolver="3"\n',
            "rust/film-juicer-core/Cargo.toml": '[package]\nname="film-juicer-core"\nversion="0.1.0"\nedition="2024"\n',
            "rust/film-juicer-core/src/lib.rs": "pub fn value()->i32{3}\n",
            "rust/film-juicer-plugin/Cargo.toml": '[package]\nname="film-juicer-plugin"\nversion="0.1.0"\nedition="2024"\n'
            '[lib]\ncrate-type=["staticlib"]\n[dependencies]\nfilm-juicer-core={path="../film-juicer-core"}\n'
            "[features]\ntest-support=[]\n",
            "rust/film-juicer-plugin/src/lib.rs": '#[unsafe(no_mangle)]\npub extern "C" fn rust_value()->i32{film_juicer_core::value()}\n',
            "rust-toolchain.toml": '[toolchain]\nchannel="1.98.1"\n',
            "rustfmt.toml": "",
            "clippy.toml": "",
        }
        for name, content in files.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        subprocess.run(
            [
                "cargo",
                "generate-lockfile",
                "--manifest-path",
                str(self.source / "Cargo.toml"),
            ],
            check=True,
            capture_output=True,
        )
        self.args[self.args.index("-DENABLE_RUST=OFF")] = "-DENABLE_RUST=ON"

    def inputs(self):
        return [
            path.relative_to(self.root).as_posix()
            for path in self.source.rglob("*")
            if path.is_file()
        ]

    def prepare_restorable_baseline(self):
        run = self.campaign.run

        def seed_configuration_times(stage, command, **kwargs):
            result = run(stage, command, **kwargs)
            if command[0] == "cmake" and "-S" in command:
                inventory = build_evidence.preparation_inventory(self.source, self.build, dict(os.environ))
                for name in inventory["regeneration_mtimes"]:
                    path = Path(name)
                    if path.is_relative_to(self.root):
                        stamp = path.stat()
                        mtime = stamp.st_mtime_ns // 1_000_000_000 * 1_000_000_000
                        os.utime(path, ns=(stamp.st_atime_ns, mtime))
                        self.assertEqual(path.stat().st_mtime_ns, mtime)
            return result

        # Seed representable configuration times before the real producer binds
        # them, rather than modifying a completed preparation's bound context.
        with patch.object(self.campaign, "run", side_effect=seed_configuration_times):
            return self.prepare()

    def snapshot_preparation(self, targets):
        # Save only this fixture's owned inputs, artifacts and provenance at its
        # configured path. Each fault still traverses real build publication.
        producer = self.prepared
        record = json.loads(producer.read_text())
        proof = record["scoped_preparation"]
        paths = {producer, Path(record["process_tree"]["receipt"]), self.root / record["output"]}
        paths.update(Path(name) for name in proof["outputs"])
        configuration = proof["context"]["configuration"]
        paths.update(self.source / name for name in configuration["inputs"])
        paths.update(Path(name) for name in configuration["tools"] if Path(name).is_relative_to(self.root))
        paths.update(Path(name) for name in configuration["registrations"])
        paths.update(Path(name) for name in proof["context"]["files"] if Path(name).is_relative_to(self.root))
        paths.update(build_evidence.preparation_paths(self.build, proof["targets"]))
        paths.update(build_evidence.preparation_paths(self.build, targets))
        self.assertTrue(all(path.is_relative_to(self.root) for path in paths))
        # Seed saved files with exactly restorable times, including on mounted
        # filesystems that truncate explicit timestamp updates to whole seconds.
        # Never alter timestamps that are part of the producer's bound context.
        bound_times = configuration["regeneration_mtimes"].keys() | proof["context"]["rust_timestamps"].keys()
        with OutputLease(self.root, self.build):
            for path in paths:
                if path.is_file() and str(path) not in bound_times:
                    stamp = path.stat()
                    mtime = stamp.st_mtime_ns // 1_000_000_000 * 1_000_000_000
                    os.utime(path, ns=(stamp.st_atime_ns, mtime))
                    self.assertEqual(path.stat().st_mtime_ns, mtime)
            build_evidence.check_scoped_preparation(self.campaign, self.build, producer)
            return producer, {path: (path.read_bytes(), path.stat()) if path.is_file() else None for path in paths}

    def restore_preparation(self, snapshot):
        producer, files = snapshot
        with OutputLease(self.root, self.build):
            for path, saved in files.items():
                if saved is None:
                    path.unlink(missing_ok=True)
                else:
                    content, stamp = saved
                    if not path.is_file() or path.read_bytes() != content:
                        path.write_bytes(content)
                    if path.stat().st_mtime_ns != stamp.st_mtime_ns:
                        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    self.assertEqual(path.read_bytes(), content, str(path))
                    self.assertEqual(path.stat().st_mtime_ns, stamp.st_mtime_ns, str(path))
            self.prepared = producer
            build_evidence.check_scoped_preparation(self.campaign, self.build, producer)

    def prepare(self, **kwargs):
        _, self.prepared = build_evidence.prepare_build(
            self.campaign,
            "prepare",
            source=self.source,
            build=self.build,
            inputs=self.inputs(),
            configure_args=self.args,
            timeout=120,
            **{"targets": ["behavior"], **kwargs},
        )
        return self.prepared

    def test_unchanged_reconfigure_preserves_registry_and_artifacts(self):
        self.prepare()
        # The first refresh retires CMake's cold compiler-detection inputs.
        # Establish the post-discovery configuration before proof-only renewal.
        first = self.prepare(reconfigure=True)
        proof = json.loads(first.read_text())["scoped_preparation"]
        registry = self.build / "verification-config-outputs.json"
        original = registry.read_bytes()
        self.prepare(reconfigure=True)
        renewed = json.loads(self.prepared.read_text())
        self.assertEqual(registry.read_bytes(), original)
        self.assertEqual(renewed["scoped_preparation"]["context"], proof["context"])
        self.assertEqual(renewed["scoped_preparation"]["output_identities"], proof["output_identities"])
        self.assertEqual(renewed["scoped_preparation"]["invalidated"], [])
        self.assertTrue(renewed["preparation_admission"]["configured"])
        self.assertIn("no work to do", (self.root / renewed["output"]).read_text())
        configurations = [row for row in self.campaign.attempts("prepare") if "configuration_evidence" in row]
        self.assertEqual(len(configurations), 3)
        self.assertNotEqual(configurations[-2]["configuration_evidence"]["trace"],
                            configurations[-1]["configuration_evidence"]["trace"])
        self.assertEqual(configurations[-2]["configuration_evidence"]["trace_sha256"],
                         configurations[-1]["configuration_evidence"]["trace_sha256"])
        self.assertEqual(renewed["scoped_preparation"]["outputs"], proof["outputs"])
        self.assertIn(self.build / "generated/binding.h", build_evidence.configuration_outputs(self.build))

    def test_call_id_and_worker_changes_preserve_native_and_rust_artifacts(self):
        self.enable_rust()
        with patch.dict(os.environ, {"CODEX_TOOL_CALL_ID": "first", "CMAKE_BUILD_PARALLEL_LEVEL": "2"}):
            self.prepare()
            self.prepare(reconfigure=True)
            first = json.loads(self.prepared.read_text())["scoped_preparation"]
            stamps = {name: Path(name).stat().st_mtime_ns for name in first["outputs"]}
            with patch.dict(os.environ, {"CODEX_TOOL_CALL_ID": "second", "CMAKE_BUILD_PARALLEL_LEVEL": "3"}):
                self.prepare(reconfigure=True)
                record = json.loads(self.prepared.read_text())
                proof = record["scoped_preparation"]
                self.assertEqual(proof["invalidated"], [])
                self.assertEqual(proof["rust_invalidated_packages"], [])
                self.assertEqual(proof["context"], first["context"])
                self.assertEqual(proof["output_identities"], first["output_identities"])
                self.assertEqual(stamps, {name: Path(name).stat().st_mtime_ns for name in first["outputs"]})
                self.assertEqual(proof["build_jobs"], 3)
                command = record["identity"]["command"]
                self.assertEqual(command[command.index("--parallel") + 1], "3")
                build_evidence.check_scoped_preparation(self.campaign, self.build, self.prepared)

    def test_invalid_worker_count_starts_no_preparation(self):
        for workers in ("0", "-1", "1.5", "unlimited", "２"):
            with self.subTest(workers=workers), patch.dict(os.environ, {"CMAKE_BUILD_PARALLEL_LEVEL": workers}):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.prepare()
        self.assertEqual(self.campaign.attempts("prepare"), [])

    def test_worker_count_cannot_become_an_untracked_compiler_input(self):
        with (self.source / "CMakeLists.txt").open("a") as stream:
            stream.write(
                'if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})\n'
                'target_compile_definitions(behavior PRIVATE WORKERS=$ENV{CMAKE_BUILD_PARALLEL_LEVEL})\n'
                'else()\ntarget_compile_definitions(behavior PRIVATE WORKERS=7)\nendif()\n'
            )
        (self.source / "native.cpp").write_text("int native(){return WORKERS;}\n")
        with patch.dict(os.environ, {"CMAKE_BUILD_PARALLEL_LEVEL": "2"}):
            self.prepare()
            self.prepare(reconfigure=True)
            first = json.loads(self.prepared.read_text())["scoped_preparation"]
            with patch.dict(os.environ, {"CMAKE_BUILD_PARALLEL_LEVEL": "3"}):
                self.prepare()
                record = json.loads(self.prepared.read_text())
                proof = record["scoped_preparation"]
                self.assertFalse(record["preparation_admission"]["configured"])
                self.assertEqual(proof["build_jobs"], 3)
                self.assertEqual(proof["invalidated"], [])
                self.assertEqual(proof["output_identities"], first["output_identities"])
                build_evidence.check_scoped_preparation(self.campaign, self.build, self.prepared)
                binary = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
                result = subprocess.run([str(binary), "16"], capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout.strip(), "value=16")

    def test_reconfigure_replaces_unproved_registry(self):
        self.prepare()
        registry = self.build / "verification-config-outputs.json"
        previous = json.loads(registry.read_text())
        trace = Path(previous["trace"])
        trace.write_bytes(trace.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "configuration producer evidence changed"):
            build_evidence.configuration_outputs(self.build)
        self.prepare(reconfigure=True)
        current = json.loads(registry.read_text())
        self.assertNotEqual(current["producer"], previous["producer"])
        self.assertEqual(current["outputs"], previous["outputs"])
        self.assertIn(self.build / "generated/binding.h", build_evidence.configuration_outputs(self.build))

    def test_reconfigure_replaces_registry_with_empty_process_receipt(self):
        self.assert_registry_replaced(lambda process: {})

    def test_reconfigure_replaces_registry_with_subset_process_receipt(self):
        self.assert_registry_replaced(lambda process: {"drained": True})

    def test_reconfigure_replaces_registry_with_incomplete_process_receipt(self):
        self.assert_registry_replaced(lambda process: {
            key: value for key, value in process.items() if key != "creation"
        })

    def test_reconfigure_replaces_registry_with_nonobject_process_receipt(self):
        self.assert_registry_replaced(lambda process: [])

    def assert_registry_replaced(self, mutate_process):
        self.prepare()
        self.prepare(reconfigure=True)
        registry = self.build / "verification-config-outputs.json"
        previous = json.loads(registry.read_text())
        producer = Path(previous["producer"])
        original_producer = producer.read_bytes()
        process = Path(json.loads(original_producer)["process_tree"]["receipt"])
        original_process = process.read_bytes()
        process.with_suffix(".before.json").write_bytes(original_process)
        build_evidence.write_json(process, mutate_process(json.loads(original_process)))
        invalid_process = process.read_bytes()
        with self.assertRaisesRegex(ValueError, "configuration producer evidence changed"):
            build_evidence.configuration_outputs(self.build)

        self.prepare(reconfigure=True)
        current = json.loads(registry.read_text())
        self.assertNotEqual(current["producer"], previous["producer"])
        self.assertEqual(current["outputs"], previous["outputs"])
        self.assertEqual(current["trace_sha256"], previous["trace_sha256"])
        self.assertEqual(producer.read_bytes(), original_producer)
        self.assertEqual(process.read_bytes(), invalid_process)
        build_evidence.check_scoped_preparation(self.campaign, self.build, self.prepared)
        _, pending = build_evidence.preparation_paths(self.build, ["behavior"])
        self.assertFalse(pending.exists())
        self.assertIn(self.build / "generated/binding.h", build_evidence.configuration_outputs(self.build))

        # Intact replacement evidence must support the ordinary no-work renewal.
        original_registry = registry.read_bytes()
        proof = json.loads(self.prepared.read_text())["scoped_preparation"]
        self.prepare(reconfigure=True)
        renewed = json.loads(self.prepared.read_text())
        self.assertEqual(registry.read_bytes(), original_registry)
        self.assertEqual(renewed["scoped_preparation"]["output_identities"], proof["output_identities"])
        self.assertEqual(renewed["scoped_preparation"]["invalidated"], [])
        self.assertIn("no work to do", (self.root / renewed["output"]).read_text())

    def test_reconfigure_replaces_registry_for_changed_trace_or_outputs(self):
        self.prepare()
        registry = self.build / "verification-config-outputs.json"
        cmake = self.source / "CMakeLists.txt"
        for change in ("trace", "outputs"):
            with self.subTest(change=change):
                previous = json.loads(registry.read_text())
                if change == "trace":
                    cmake.write_text(cmake.read_text().replace("COPYONLY", "@ONLY"))
                else:
                    cmake.write_text(cmake.read_text() +
                        "configure_file(binding.h.in generated/second.h COPYONLY)\n")
                self.prepare(reconfigure=True)
                current = json.loads(registry.read_text())
                self.assertNotEqual(current["producer"], previous["producer"])
                self.assertNotEqual(current["trace_sha256"], previous["trace_sha256"])
                if change == "trace":
                    self.assertEqual(current["outputs"], previous["outputs"])
                else:
                    self.assertEqual(set(current["outputs"]),
                                     {*previous["outputs"], str(self.build / "generated/second.h")})
                self.assertEqual((self.build / "generated/binding.h").read_bytes(),
                                 (self.source / "binding.h.in").read_bytes())
                self.assertIn(self.build / "generated/binding.h", build_evidence.configuration_outputs(self.build))

    def test_target_preparation_transitions_without_running_runtime_tests(self):
        preset = "windows-clang-release" if os.name == "nt" else "linux-release"
        root = self.source
        (root / "src").mkdir()
        for path in list(root.iterdir()):
            if path.suffix in {".cpp", ".c", ".h"}:
                path.rename(root / "src" / path.name)
        cmake = root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text().replace("main.cpp native.cpp abi.c", "src/main.cpp src/native.cpp src/abi.c")
                         .replace("rust_stub.c", "src/rust_stub.c"), encoding="utf-8")
        for name in ("verify.py", "build_evidence.py"):
            path = root / "scripts" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes((ROOT / "scripts" / name).read_bytes())
        with (root / "CMakeLists.txt").open("a") as stream:
            stream.write('\nadd_custom_target(juicer DEPENDS behavior)\n'
                         'add_test(NAME must-not-run COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_BINARY_DIR}/runtime-ran")\n')
        variables = {arg[2:].split("=", 1)[0]: arg.split("=", 1)[1] for arg in self.args if arg.startswith("-D")}
        variables["CMAKE_BUILD_TYPE"] = "Release"
        (root / "CMakePresets.json").write_text(json.dumps({"version": 6, "configurePresets": [{
            "name": preset, "generator": "Ninja", "binaryDir": "${sourceDir}/out/build/" + preset,
            "cacheVariables": variables}]}), encoding="utf-8")
        build = root / "out/build" / preset
        development = build / "verification-development.json"
        build.mkdir(parents=True)
        development.write_text('{"pending":["preserve-me"]}\n', encoding="utf-8")
        saved = development.read_bytes()
        for testing in ("ON", "OFF", "ON"):
            prepared = build_evidence.prepare_targets(root, preset, ["juicer"], testing)
            record = json.loads(prepared.read_text())
            self.assertEqual(record["scoped_preparation"]["targets"], ["juicer"])
            self.assertEqual(record["scoped_preparation"]["configure_args"], ["-DBUILD_TESTING=" + testing])
            self.assertTrue(record["process_tree"]["drained"])
            self.assertFalse((build / "runtime-ran").exists())
            self.assertEqual(development.read_bytes(), saved)
            self.assertTrue(all(json.loads(p.read_text())["stage"] == "prepare"
                                for p in prepared.parents[2].glob("attempts/*/result.json")))
        campaign = Campaign(root, preset, "freshness-reader", ["prepare"])
        build_evidence.check_scoped_preparation(campaign, build, prepared)
        (root / "src/native.cpp").write_text("int native(){return 9;}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dependency contents"):
            build_evidence.check_scoped_preparation(campaign, build, prepared)
        pending = build_evidence.preparation_paths(build, ["all"])[1]
        pending.write_text('{"protocol":"legacy"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "another pending"):
            build_evidence.prepare_targets(root, preset, ["juicer"])
        self.assertEqual(pending.read_text(), '{"protocol":"legacy"}\n')

    def assert_legacy_disposition_is_exact_drained(self):
        preset = "linux-release"
        build = self.root / "out/build" / preset
        build.mkdir(parents=True)
        marker = build / "verification-build-authorized.pending.json"
        marker.write_bytes(b'{"protocol":"legacy","targets":["all"]}\n')
        original = marker.read_bytes()
        baseline = marker.with_name("verification-build-authorized.json")
        baseline.write_bytes(b'{"old":"unqualified"}\n')
        previous = baseline.read_bytes()
        directory = self.campaign.directory
        evidence = {str(p.relative_to(directory)): digest(p) for p in
                    [directory / "campaign.json", *directory.glob("attempts/prepare-*/*")]
                    if p.name in {"campaign.json", "result.json", "process.json", "output.log"}}
        expected = {"source": str(self.root), "marker": marker.name, "sha256": digest(marker),
                    "campaign": str(directory.relative_to(self.root)), "evidence": evidence}
        protected = {directory / name: (directory / name).read_bytes() for name in evidence}
        attempt = next(directory.glob("attempts/prepare-*/process.json"))
        for fault in ("changed-marker", "malformed-marker", "missing-evidence", "changed-evidence", "undrained",
                      "running-command", "new-attempt", "additional-marker", "active-supervisor"):
            with self.subTest(fault=fault):
                binding = json.loads(json.dumps(expected))
                extras = []
                if fault in {"changed-marker", "malformed-marker"}:
                    marker.write_bytes(original + b" " if fault == "changed-marker" else b"{")
                elif fault == "missing-evidence":
                    attempt.unlink()
                elif fault == "changed-evidence":
                    attempt.write_bytes(protected[attempt] + b" ")
                elif fault in {"undrained", "active-supervisor"}:
                    tree = json.loads(attempt.read_text())
                    if fault == "undrained":
                        tree["drained"] = False
                    else:
                        tree["supervisor_pid"] = os.getpid()
                        tree["creation"]["linux_start_ticks"] = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
                    write_json(attempt, tree)
                    result = attempt.with_name("result.json")
                    record = json.loads(result.read_text())
                    record["process_tree"] = {**tree, "receipt": str(attempt)}
                    write_json(result, record)
                    for path in (attempt, result):
                        binding["evidence"][str(path.relative_to(directory))] = digest(path)
                else:
                    extra = (directory / "running.lock" if fault == "running-command" else
                             directory / "attempts/prepare-unproved" if fault == "new-attempt" else
                             build / "unknown.pending.json")
                    extra.write_text("unknown")
                    extras.append(extra)
                try:
                    presented = marker.read_bytes()
                    with patch.object(build_evidence, "S4C1_PENDING", {preset: binding}), self.assertRaises(ValueError):
                        build_evidence.retire_s4c1_pending(self.root, preset)
                    self.assertEqual(marker.read_bytes(), presented)
                    self.assertEqual(baseline.read_bytes(), previous)
                finally:
                    marker.write_bytes(original)
                    for path, content in protected.items():
                        path.write_bytes(content)
                    for path in extras:
                        path.unlink()
        lease = OutputLease(self.root, build, timeout=.02)
        for status in ("owned", "blocked"):
            lease.path.mkdir(parents=True)
            owner = lease.path / "owner.json"
            write_json(owner, {"owner_pid": 999999999, "status": status, "token": "unproved"})
            saved = owner.read_bytes()
            error, message = ((TimeoutError, "output remains owned") if status == "owned" else
                              (RuntimeError, "output owner is blocked"))
            try:
                with patch.object(build_evidence, "S4C1_PENDING", {preset: expected}), \
                        patch.object(build_evidence, "OutputLease", return_value=lease), \
                        self.assertRaisesRegex(error, message):
                    build_evidence.retire_s4c1_pending(self.root, preset)
                self.assertEqual(owner.read_bytes(), saved)
                self.assertEqual(marker.read_bytes(), original)
            finally:
                owner.unlink()
                lease.path.rmdir()
        with patch.object(build_evidence, "S4C1_PENDING", {preset: expected}):
            retired = build_evidence.retire_s4c1_pending(self.root, preset)
            self.assertFalse(marker.exists())
            self.assertFalse(baseline.exists())
            record = json.loads(retired.read_text())
            self.assertEqual(record["disposition"], "retired-unqualified")
            self.assertTrue(record["fresh_preparation_required"])
            self.assertEqual(Path(record["original"]["file"]).read_bytes(), original)
            self.assertEqual(Path(record["baseline"]["file"]).read_bytes(), previous)
            self.assertTrue(all(path.read_bytes() == content for path, content in protected.items()))
            with self.assertRaisesRegex(ValueError, "already retired"):
                build_evidence.retire_s4c1_pending(self.root, preset)

    def execute(self, **kwargs):
        receipt = build_evidence.run_ctest(
            self.campaign,
            "tests",
            build=self.build,
            prepared=self.prepared,
            inputs=self.inputs(),
            timeout=30,
            **{"tests": ["behavior"], **kwargs},
        )
        output = (self.root / json.loads(receipt.read_text())["output"]).read_text()
        self.assertIn("100% tests passed", output)
        return receipt

    def test_native_transitive_generated_and_configuration_edits_execute_new_behavior(
        self,
    ):
        self.prepare()
        self.execute()
        # Finite declared controls: each semantic mutation must fail the original
        # independent value witness after a same-size/restored-time source edit.
        for name, before, after in [
            ("native.cpp", "return 1", "return 8"),
            ("detail.h", "return 2", "return 8"),
            ("binding.h.in", "BINDING 4", "BINDING 8"),
            ("CMakeLists.txt", "behavior 10", "behavior 11"),
        ]:
            with self.subTest(name=name):
                path = self.source / name
                original, stat = path.read_bytes(), path.stat()
                path.write_bytes(original.replace(before.encode(), after.encode()))
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.prepare()
                with self.assertRaises(subprocess.CalledProcessError):
                    self.execute()
                path.write_bytes(original)
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.prepare()
                self.execute()

    def test_intervening_preparation_rejects_earlier_pass_after_source_restore(self):
        first = self.prepare()
        self.execute()
        path = self.source / "native.cpp"
        original, stamp = path.read_bytes(), path.stat()
        try:
            path.write_bytes(original.replace(b"return 1", b"return 8"))
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.prepare()
        finally:
            path.write_bytes(original)
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        attempts = len(self.campaign.attempts("tests"))
        with OutputLease(self.root, self.build):
            # The current baseline binds the replacement producer's raw receipt.
            with self.assertRaisesRegex(ValueError, "command receipt changed"):
                build_evidence.check_scoped_preparation(self.campaign, self.build, first)
            with self.assertRaisesRegex(ValueError, "command receipt changed"):
                build_evidence.run_ctest(self.campaign, "tests", build=self.build, prepared=first,
                                         inputs=self.inputs(), timeout=30, tests=["behavior"])
        self.assertEqual(len(self.campaign.attempts("tests")), attempts)
        self.prepare()
        self.execute()

    def test_undeclared_preparation_preserves_reusable_results(self):
        first = self.prepare()
        baseline, pending = build_evidence.preparation_paths(self.build, ["behavior"])
        record = json.loads(first.read_text())
        paths = {
            baseline, first, Path(record["process_tree"]["receipt"]),
            self.root / record["output"],
            *(Path(name) for name in record["scoped_preparation"]["outputs"]),
        }
        saved = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
        caller = Campaign(self.root, self.campaign.directory.parent.name, "invalid-stage", ["quality"])
        with patch.object(caller, "run", wraps=caller.run) as run:
            with self.assertRaisesRegex(ValueError, "declared stage"):
                build_evidence.prepare_build(
                    caller, "prepare", source=self.source, build=self.build,
                    inputs=self.inputs(), configure_args=self.args,
                    targets=["behavior"], timeout=120,
                )
            self.assertFalse(pending.exists())
            run.assert_not_called()
        self.assertFalse((caller.directory / "attempts").exists())
        self.assertEqual(
            saved, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
        )
        with OutputLease(self.root, self.build):
            build_evidence.check_scoped_preparation(self.campaign, self.build, first)
        self.execute()

    def test_invalid_preparation_request_preserves_absent_or_pending_owner(self):
        baseline, pending = build_evidence.preparation_paths(self.build, ["behavior"])
        requests = [{"inputs": []}, *({"timeout": value} for value in (0, -1, float("nan"), float("inf"), -float("inf")))]
        for existing in (False, True):
            if existing:
                self.build.mkdir()
                pending.write_text("unfinished preparation\n")
                saved = pending.read_bytes(), pending.stat().st_mtime_ns
            for invalid in requests:
                with self.subTest(existing=existing, invalid=invalid):
                    arguments = dict(inputs=self.inputs(), timeout=120)
                    arguments.update(invalid)
                    with patch.object(self.campaign, "run", wraps=self.campaign.run) as run:
                        with self.assertRaisesRegex(ValueError, "declared stage"):
                            build_evidence.prepare_build(
                                self.campaign, "prepare", source=self.source, build=self.build,
                                configure_args=self.args, targets=["behavior"], **arguments,
                            )
                        run.assert_not_called()
                    self.assertFalse(baseline.exists())
                    if existing:
                        self.assertEqual(saved, (pending.read_bytes(), pending.stat().st_mtime_ns))
                    else:
                        self.assertFalse(self.build.exists())
        self.assertFalse((self.campaign.directory / "attempts").exists())

    def test_pending_or_missing_preparation_owner_rejects_prior_pass(self):
        first = self.prepare()
        baseline = next(self.build.glob("verification-build-*.json"))
        original = baseline.read_bytes()
        pending = baseline.with_suffix(".pending.json")
        with OutputLease(self.root, self.build):
            try:
                pending.write_text("unfinished preparation")
                with self.assertRaisesRegex(ValueError, "pending preparation"):
                    build_evidence.check_scoped_preparation(self.campaign, self.build, first)
            finally:
                pending.unlink(missing_ok=True)
            try:
                baseline.unlink()
                with self.assertRaisesRegex(ValueError, "latest completed preparation"):
                    build_evidence.check_scoped_preparation(self.campaign, self.build, first)
                baseline.write_text("unreadable owner")
                with self.assertRaisesRegex(ValueError, "latest completed preparation"):
                    build_evidence.check_scoped_preparation(self.campaign, self.build, first)
            finally:
                baseline.write_bytes(original)
        self.execute()
        if os.name != "nt":
            self.assert_legacy_disposition_is_exact_drained()

    def test_restored_timestamp_rust_body_is_observed_through_native_executable(self):
        self.enable_rust()
        self.prepare()
        self.execute()
        path = self.source / "rust/film-juicer-core/src/lib.rs"
        original, stamp = path.read_bytes(), path.stat()
        path.write_bytes(original.replace(b"{3}", b"{8}"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.prepare()
        proof = json.loads(self.prepared.read_text())["scoped_preparation"]
        self.assertEqual(proof["rust_invalidated_packages"], ["film-juicer-core", "film-juicer-plugin"])
        clean = [row["identity"]["command"] for row in self.campaign.attempts("prepare") if Path(row["identity"]["command"][0]).name in {"cargo", "cargo.exe"} and row["identity"]["command"][1] == "clean"]
        self.assertEqual(len(clean), 1)
        self.assertIn("--target", clean[0])
        self.assertEqual(clean[0][clean[0].index("--profile") + 1], "dev")
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        path.write_bytes(original)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.prepare()
        self.execute()

    def test_overlapping_target_preparation_cannot_leave_a_reusable_owner(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + "add_custom_target(overlap DEPENDS behavior)\n")
        first = self.prepare()
        self.execute()
        path = self.source / "native.cpp"
        original, stamp = path.read_bytes(), path.stat()
        try:
            path.write_bytes(original.replace(b"return 1", b"return 8"))
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.prepare(targets=["overlap"])
        finally:
            path.write_bytes(original)
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with OutputLease(self.root, self.build), self.assertRaisesRegex(ValueError, "latest completed preparation"):
            build_evidence.check_scoped_preparation(self.campaign, self.build, first)
        self.prepare()
        self.execute()

    def test_new_scope_restored_clean_rust_uses_shared_archive_inputs(self):
        self.enable_rust()
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + "add_custom_target(overlap DEPENDS behavior)\n")
        self.prepare()
        self.execute()
        path = self.source / "rust/film-juicer-core/src/lib.rs"
        original, stamp = path.read_bytes(), path.stat()
        path.write_bytes(original.replace(b"{3}", b"{8}"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.prepare()
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        path.write_bytes(original)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.prepare(targets=["overlap"])
        self.assertEqual(json.loads(self.prepared.read_text())["scoped_preparation"]["rust_invalidated_packages"], ["film-juicer-core", "film-juicer-plugin"])
        self.execute()

    def test_header_selection_keeps_cold_configured_variants(self):
        from build_dependencies import header_consumers

        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + 'add_library(extra OBJECT main.cpp)\ntarget_include_directories(extra PRIVATE "${CMAKE_BINARY_DIR}/generated")\n')
        self.prepare()
        targets, sources = header_consumers(self.source, self.build, ["detail.h"])
        self.assertEqual(sources, ["main.cpp"])
        self.assertEqual(len(targets), 2)
        self.assertTrue(any("extra.dir" in name for name in targets))
        self.prepare(targets=targets)
        self.assertTrue(all((self.build / name).is_file() for name in targets))
        (self.source / "unconsumed.h").write_text("// no consumer\n")
        with self.assertRaisesRegex(ValueError, "without configured consumers"):
            header_consumers(self.source, self.build, ["unconsumed.h"])

    def test_old_nonconsumer_becomes_header_consumer_without_header_edit(self):
        from build_dependencies import header_consumers

        other = self.source / "other.cpp"
        other.write_text("int other(){return 0;}\n")
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + 'add_library(extra OBJECT other.cpp)\ntarget_include_directories(extra PRIVATE "${CMAKE_BINARY_DIR}/generated")\n')
        self.prepare()
        before, sources = header_consumers(self.source, self.build, ["detail.h"])
        self.assertNotIn("other.cpp", sources)
        header = (self.source / "detail.h").read_bytes()
        other.write_text('#include "detail.h"\nint other(){return header();}\n')
        after, sources = header_consumers(self.source, self.build, ["detail.h"])
        self.assertIn("other.cpp", sources)
        self.assertGreater(len(after), len(before))
        self.assertEqual((self.source / "detail.h").read_bytes(), header)

    def test_selected_probe_fixture_change_invalidates_pass_without_cleaning_rust(self):
        path = self.source / "probes/reject.rs"
        path.parent.mkdir()
        path.write_text("// compiler assertion fixture\n")
        first = self.prepare()
        path.write_text("// changed compiler assertion\n")
        with OutputLease(self.root, self.build), self.assertRaisesRegex(ValueError, "dependency contents"):
            build_evidence.check_scoped_preparation(self.campaign, self.build, first)
        self.prepare()
        self.assertEqual(json.loads(self.prepared.read_text())["scoped_preparation"]["rust_invalidated_packages"], [])
        self.execute()

    def test_cargo_configuration_change_invalidates_reuse_and_active_packages(self):
        self.enable_rust()
        first = self.prepare()
        config = self.source / ".cargo/config.toml"
        config.parent.mkdir()
        config.write_text("[build]\nrustflags = []\n")
        with OutputLease(self.root, self.build), self.assertRaisesRegex(ValueError, "dependency contents"):
            build_evidence.check_scoped_preparation(self.campaign, self.build, first)
        self.prepare()
        self.assertEqual(json.loads(self.prepared.read_text())["scoped_preparation"]["rust_invalidated_packages"], ["film-juicer-core", "film-juicer-plugin"])
        self.execute()

    def test_macro_candidates_compile_before_consumer_publication(self):
        from build_dependencies import header_consumers

        middle = self.source / "middle.h"
        middle.write_text('#define DETAIL "detail.h"\n#include DETAIL\n')
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + 'add_library(extra OBJECT main.cpp)\ntarget_include_directories(extra PRIVATE "${CMAKE_BINARY_DIR}/generated")\n')
        self.prepare()
        targets, _ = header_consumers(self.source, self.build, ["detail.h"], require_compiled=False)
        self.assertEqual(len(targets), 2)
        self.prepare(targets=targets)
        self.assertEqual(len(header_consumers(self.source, self.build, ["detail.h"])[0]), 2)
        # A warm restored-time macro edit must discover a previously absent header.
        (self.source / "new.h").write_text('#include "detail.h"\n')
        original, stamp = middle.read_bytes(), middle.stat()
        middle.write_bytes(original.replace(b'"detail.h"', b'"new.h"'))
        os.utime(middle, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "without configured consumers"):
            header_consumers(self.source, self.build, ["new.h"])
        candidates, _ = header_consumers(self.source, self.build, ["new.h"], require_compiled=False)
        self.assertEqual(len(candidates), 2)
        self.prepare(targets=candidates)
        self.assertEqual(header_consumers(self.source, self.build, ["new.h"])[1], ["main.cpp"])

    def test_new_scope_cargo_configuration_invalidates_active_packages(self):
        self.enable_rust()
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + "add_custom_target(overlap DEPENDS behavior)\n")
        self.prepare()
        config = self.source / ".cargo/config.toml"
        config.parent.mkdir()
        config.write_text("[build]\nrustflags = []\n")
        self.prepare(targets=["overlap"])
        self.assertEqual(json.loads(self.prepared.read_text())["scoped_preparation"]["rust_invalidated_packages"], ["film-juicer-core", "film-juicer-plugin"])
        self.execute()

    def test_rust_tool_identity_change_invalidates_active_packages(self):
        self.enable_rust()
        self.prepare()
        inventory = build_evidence.preparation_inventory
        def changed_tools(*args):
            value = inventory(*args)
            for name in value["tools"]:
                if Path(name).name in {"rustc", "rustc.exe"}:
                    value["tools"][name] = "replacement tool identity"
            return value
        with patch.object(build_evidence, "preparation_inventory", side_effect=changed_tools):
            self.prepare()
            self.assertEqual(json.loads(self.prepared.read_text())["scoped_preparation"]["rust_invalidated_packages"], ["film-juicer-core", "film-juicer-plugin"])
            self.execute()

    def test_normal_timestamp_rust_edit_uses_incremental_cargo(self):
        self.enable_rust()
        self.prepare()
        path = self.source / "rust/film-juicer-core/src/lib.rs"
        original = path.read_bytes()
        path.write_bytes(original.replace(b"{3}", b"{8}"))
        self.prepare()
        self.assertEqual(json.loads(self.prepared.read_text())["scoped_preparation"]["rust_invalidated_packages"], [])
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        path.write_bytes(original)
        self.prepare()
        self.execute()

    def test_unchanged_repeat_reuses_objects_and_archive_without_touching(self):
        self.enable_rust()
        self.prepare()
        record = json.loads(self.prepared.read_text())
        paths = [Path(name) for name in record["scoped_preparation"]["outputs"]]
        before = {str(path): (digest(path), path.stat().st_mtime_ns) for path in paths}
        self.prepare()
        after = {str(path): (digest(path), path.stat().st_mtime_ns) for path in paths}
        self.assertEqual(before, after)
        self.assertEqual(
            json.loads(self.prepared.read_text())["scoped_preparation"]["invalidated"], []
        )
        # A missing successful-Cargo stamp must preserve the unchanged archive
        # and native consumers; order-only prerequisites do not carry data.
        (self.build / "cargo-build.stamp").unlink()
        self.prepare()
        repeated = {
            str(path): (digest(path), path.stat().st_mtime_ns)
            for path in paths
            if path.name != "cargo-build.stamp"
        }
        self.assertEqual(
            {
                name: value
                for name, value in before.items()
                if Path(name).name != "cargo-build.stamp"
            },
            repeated,
        )
        self.execute()

    def test_missing_preparation_owner_rebuilds_existing_outputs(self):
        self.prepare()
        owners = list(self.build.glob("verification-build-*.json"))
        self.assertTrue(owners)
        for path in owners:
            path.unlink()
        binary = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        binary.write_bytes(b"stale executable")
        self.prepare()
        self.execute()

    def assert_changed_source_rebuilds_intervening_archive(self, *, new_scope=False, interrupted=False):
        self.enable_rust()
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
            "add_custom_target(overlap DEPENDS behavior)\nadd_test(NAME selected COMMAND behavior 15)\n")
        producer = self.prepare()
        original = producer.read_bytes()
        proof = json.loads(original)["scoped_preparation"]
        archive = next(Path(name) for name in proof["outputs"]
                       if Path(name).name in {"libfilm_juicer_plugin.a", "film_juicer_plugin.lib"})
        source = self.source / "rust/film-juicer-core/src/lib.rs"
        source.write_bytes(source.read_bytes().replace(b"{3}", b"{8}"))
        with OutputLease(self.root, self.build):
            external = self.campaign.run("prepare", ["cmake", "--build", str(self.build), "--parallel", "2", "--target", "behavior"],
                inputs=self.inputs(), timeout=120, repeat_reason="changed-source direct build before fresh preparation",
                vctip_executable=build_evidence.windows_vctip(os.environ))
        self.assertNotEqual(digest(archive), proof["output_identities"][str(archive)]["sha256"])
        # Prove fresh preparation does not trust Cargo's timestamps or these bytes.
        archive.write_bytes(b"unusable intervening archive")
        timestamp = max(time.time_ns(), (self.build / "cargo-build.stamp").stat().st_mtime_ns) + 2_000_000_000
        os.utime(source, ns=(timestamp, timestamp))
        targets = ["overlap"] if new_scope else ["behavior"]
        failed = None
        if interrupted:
            with patch.object(build_evidence, "invalidate_preparation_owners", side_effect=ValueError("stop before fresh build")):
                with self.assertRaisesRegex(ValueError, "stop before fresh build"):
                    self.prepare(targets=targets)
            _, pending = build_evidence.preparation_paths(self.build, targets)
            failed = json.loads(pending.read_text())["failure_proof"]
            self.assertEqual(json.loads(Path(failed["file"]).read_text())["completion"], "commands-drained")
        self.prepare(targets=targets)
        record = json.loads(self.prepared.read_text())
        self.assertEqual(record["preparation_admission"]["archive_owners"], {})
        self.assertEqual(record["preparation_admission"]["artifact_owners"], {})
        self.assertEqual(record["preparation_admission"]["superseded_producers"], {str(producer): digest(producer)})
        self.assertIn(str(archive), record["preparation_admission"]["superseded_artifacts"][str(producer)])
        self.assertEqual(record["scoped_preparation"]["rust_invalidated_packages"], ["film-juicer-core", "film-juicer-plugin"])
        self.assertEqual(producer.read_bytes(), original)
        self.assertTrue(external.is_file())
        if failed:
            self.assertEqual(digest(Path(failed["file"])), failed["sha256"])
            self.assertIsNotNone(record["preparation_admission"]["recovered_pending"])
        self.execute(tests=["selected"], junit=self.campaign.directory / "fresh-source.xml")

    def test_changed_source_renewal_rebuilds_intervening_archive(self):
        self.assert_changed_source_rebuilds_intervening_archive()

    def test_changed_source_new_scope_rebuilds_intervening_archive(self):
        self.assert_changed_source_rebuilds_intervening_archive(new_scope=True)

    def test_changed_source_retry_rebuilds_intervening_archive(self):
        self.assert_changed_source_rebuilds_intervening_archive(new_scope=True, interrupted=True)

    def test_changed_native_source_rebuilds_intervening_executable(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + "add_test(NAME selected COMMAND behavior 17)\n")
        producer = self.prepare()
        source = self.source / "native.cpp"
        source.write_bytes(source.read_bytes().replace(b"return 1", b"return 8"))
        with OutputLease(self.root, self.build):
            self.campaign.run("prepare", ["cmake", "--build", str(self.build), "--parallel", "2", "--target", "behavior"],
                inputs=self.inputs(), timeout=120, repeat_reason="changed native source before renewal",
                vctip_executable=build_evidence.windows_vctip(os.environ))
        binary = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        binary.write_bytes(b"unusable intervening executable")
        self.prepare()
        record = json.loads(self.prepared.read_text())
        self.assertEqual(record["preparation_admission"]["superseded_producers"], {str(producer): digest(producer)})
        self.assertIn(str(binary), record["scoped_preparation"]["invalidated"])
        self.execute(tests=["selected"], junit=self.campaign.directory / "fresh-native.xml")

    def assert_renewal_rejects_restored_artifact(self, artifact_kind, *, unbound=False, new_scope=False):
        rust = artifact_kind == "archive"
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
            "add_custom_target(overlap DEPENDS behavior)\nadd_test(NAME selected COMMAND behavior 10)\n")
        if rust:
            self.enable_rust()
        producer = self.prepare()
        binary = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        self.assertEqual(subprocess.check_output([str(binary), "10"], text=True, timeout=10).strip(), "value=10")
        baseline, pending = build_evidence.preparation_paths(self.build, ["behavior"])
        original_owner = baseline.read_bytes()
        (self.campaign.directory / "prior-baseline.json").write_bytes(original_owner)
        proof = json.loads(producer.read_text())["scoped_preparation"]
        originals = {Path(name): (Path(name).read_bytes(), Path(name).stat()) for name in proof["outputs"]}
        if artifact_kind == "executable":
            artifact = binary
        elif rust:
            artifact = next(path for path in originals if path.name in {"libfilm_juicer_plugin.a", "film_juicer_plugin.lib"})
        else:
            artifact = next(path for path in originals if path.name in {"native.cpp.o", "native.cpp.obj"})
        source = self.source / ("rust/film-juicer-core/src/lib.rs" if rust else "native.cpp")
        content, stamp = source.read_bytes(), source.stat()
        source.write_bytes(content.replace(b"{3}", b"{8}") if rust else content.replace(b"return 1", b"return 8"))
        with OutputLease(self.root, self.build):
            external = self.campaign.run("prepare", ["cmake", "--build", str(self.build), "--parallel", "2", "--target", "behavior"],
                inputs=self.inputs(), timeout=120, repeat_reason="fixture-local intervening artifact mutation",
                vctip_executable=build_evidence.windows_vctip(os.environ))
        expected = "15" if rust else "17"
        witness = subprocess.check_output([str(binary), expected], text=True, timeout=10).strip()
        self.assertEqual(witness, "value=" + expected)
        source.write_bytes(content)
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        for path, (data, previous) in originals.items():
            if path != artifact:
                path.write_bytes(data)
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.assertNotEqual(digest(artifact), proof["output_identities"][str(artifact)]["sha256"])
        build_evidence.write_json(self.campaign.directory / "restored-artifact.json", {
            "producer": str(producer), "intervening_build": str(external), "artifact": str(artifact),
            "expected": proof["output_identities"][str(artifact)], "mutated_sha256": digest(artifact),
            "restored_mtime_ns": artifact.stat().st_mtime_ns, "intervening_witness": witness})
        if unbound:
            legacy = json.loads(original_owner)
            legacy.pop("producer_sha256")
            build_evidence.write_json(baseline, legacy)
            self.prepare(targets=["overlap"] if new_scope else ["behavior"])
            repaired = json.loads(self.prepared.read_text())["scoped_preparation"]
            self.assertEqual(repaired["rust_invalidated_packages"], ["film-juicer-core", "film-juicer-plugin"])
            with OutputLease(self.root, self.build):
                build_evidence.check_scoped_preparation(self.campaign, self.build, self.prepared)
            self.execute(tests=["selected"] if new_scope else ["behavior"], junit=self.campaign.directory / "repaired.xml")
            return
        attempts = len(self.campaign.attempts("prepare"))
        with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
            self.prepare()
        self.assertEqual(len(self.campaign.attempts("prepare")), attempts)
        self.assertEqual(baseline.read_bytes(), original_owner)
        self.assertFalse(pending.exists())
        selected, _ = build_evidence.preparation_paths(self.build, ["overlap"])
        self.assertFalse(selected.exists())
        with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
            self.prepare(targets=["overlap"])
        self.assertFalse(selected.exists())
        self.assertEqual(baseline.read_bytes(), original_owner)

    def test_same_scope_renewal_rejects_restored_time_executable(self):
        self.assert_renewal_rejects_restored_artifact("executable")

    def test_same_scope_renewal_rejects_restored_time_object(self):
        self.assert_renewal_rejects_restored_artifact("object")

    def test_same_scope_renewal_rejects_restored_time_rust_archive(self):
        self.assert_renewal_rejects_restored_artifact("archive")

    def test_same_scope_unbound_owner_rebuilds_stale_rust_archive(self):
        self.assert_renewal_rejects_restored_artifact("archive", unbound=True)

    def test_new_scope_unbound_owner_rebuilds_stale_rust_archive(self):
        self.assert_renewal_rejects_restored_artifact("archive", unbound=True, new_scope=True)

    def test_missing_scope_rejects_incomplete_process_payload(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + "add_custom_target(overlap DEPENDS behavior)\n")
        producer = self.prepare()
        record = json.loads(producer.read_text())
        process = Path(record["process_tree"]["receipt"])
        original, stamp = process.read_bytes(), process.stat()
        complete = json.loads(original)
        payloads = [{}, {"drained": True}, {"status": "completed"},
                    {key: value for key, value in complete.items() if key != "creation"},
                    {**complete, "unexpected": True}]
        selected, _ = build_evidence.preparation_paths(self.build, ["overlap"])
        for payload in payloads:
            with self.subTest(keys=sorted(payload)), OutputLease(self.root, self.build):
                try:
                    build_evidence.write_json(process, payload)
                    with self.assertRaisesRegex(ValueError, "command evidence changed"):
                        build_evidence.check_scoped_preparation(self.campaign, self.build, producer)
                    self.assertIsNone(build_evidence._admit_missing_scope(self.campaign, self.source, self.build,
                        selected, ["overlap"], self.inputs(), self.args, dict(os.environ)))
                finally:
                    process.write_bytes(original)
                    os.utime(process, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.execute(junit=self.campaign.directory / "restored-process.xml")

    def assert_publication_mutation_is_rejected(self, *, renew=False):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + "add_custom_target(overlap DEPENDS behavior)\n")
        producer = self.prepare_restorable_baseline()
        baseline, _ = build_evidence.preparation_paths(self.build, ["behavior"])
        original_owner = baseline.read_bytes()
        targets = ["behavior"] if renew else ["overlap"]
        snapshot = self.snapshot_preparation(targets)
        faults = ("log", "process", "object") if renew else ("log", "process", "object", "receipt")
        for index, fault in enumerate(faults):
            with self.subTest(fault=fault):
                if index:
                    self.restore_preparation(snapshot)
                record = json.loads(producer.read_text())
                if fault == "process":
                    payload = Path(record["process_tree"]["receipt"])
                elif fault == "log":
                    payload = self.root / record["output"]
                elif fault == "object":
                    payload = next(Path(name) for name in record["scoped_preparation"]["outputs"] if name.endswith((".o", ".obj")))
                else:
                    payload = producer
                run = self.campaign.run

                def mutate(stage, command, **kwargs):
                    result = run(stage, command, **kwargs)
                    if "--build" in command:
                        if fault == "process":
                            build_evidence.write_json(payload, {})
                        elif fault == "log":
                            payload.write_bytes(payload.read_bytes() + b"mutated producer log\n")
                        elif fault == "object":
                            content = payload.read_bytes()
                            payload.write_bytes(content + b"mutated output" if renew else content[:-1] + bytes([content[-1] ^ 1]))
                        else:
                            build_evidence.write_json(payload, {**record, "seconds": record["seconds"] + 1})
                    return result

                if fault == "object":
                    message = "retained artifact bytes changed" if renew else "admitted artifacts changed"
                elif fault == "receipt":
                    message = "admitted preparation command receipt changed"
                else:
                    message = "command evidence changed"
                with patch.object(self.campaign, "run", side_effect=mutate), self.assertRaisesRegex(ValueError, message):
                    self.prepare(targets=targets)
                selected, pending = build_evidence.preparation_paths(self.build, targets)
                if renew:
                    self.assertEqual(selected.read_bytes(), original_owner)
                else:
                    self.assertFalse(selected.exists())
                self.assertTrue(pending.exists())
                self.assertEqual(baseline.read_bytes(), original_owner)
                self.assertEqual(self.campaign.attempts("prepare")[-1]["status"], "failed")

    def test_admission_publication_rejects_mutation(self):
        self.assert_publication_mutation_is_rejected()

    def test_renewal_publication_rejects_mutation(self):
        self.assert_publication_mutation_is_rejected(renew=True)

    def assert_shared_rust_fallback(self, faults=(None,), *, missing_stamp=False):
        self.enable_rust()
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
            "add_custom_target(overlap DEPENDS behavior)\nadd_test(NAME selected COMMAND behavior 17)\n")
        producer = self.prepare_restorable_baseline() if len(faults) > 1 else self.prepare()
        binary = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        self.assertEqual(subprocess.check_output([str(binary), "10"], text=True, timeout=10).strip(), "value=10")
        snapshot = self.snapshot_preparation(["overlap"]) if len(faults) > 1 else None
        for index, fault in enumerate(faults):
            with self.subTest(fault=fault):
                if index:
                    self.restore_preparation(snapshot)
                record = json.loads(producer.read_text())
                proof = record["scoped_preparation"]
                baseline, old_pending = build_evidence.preparation_paths(self.build, ["behavior"])
                original_owner = baseline.read_bytes()
                (self.campaign.directory / "prior-baseline.json").write_bytes(original_owner)
                archive = next(Path(name) for name in proof["outputs"]
                               if Path(name).name in {"libfilm_juicer_plugin.a", "film_juicer_plugin.lib"})
                stamp = self.build / "cargo-build.stamp"
                if missing_stamp:
                    stamp.unlink()
                retained = (archive,) if missing_stamp else (archive, stamp)
                preserved = {str(path): (digest(path), path.stat().st_mtime_ns) for path in retained}
                native = self.source / "native.cpp"
                previous_time = native.stat()
                native.write_bytes(native.read_bytes().replace(b"return 1", b"return 8"))
                os.utime(native, ns=(previous_time.st_atime_ns, previous_time.st_mtime_ns))
                selected, pending = build_evidence.preparation_paths(self.build, ["overlap"])
                self.assertFalse(selected.exists())
                self.assertFalse(pending.exists())
                run, admit = self.campaign.run, build_evidence._admit_missing_scope
                admissions, builds = [], []

                def observe_admission(*args, **kwargs):
                    result = admit(*args, **kwargs)
                    self.assertIsNone(result)  # The native input changed; whole-scope adoption is invalid.
                    admissions.append(result)
                    return result

                def mutate(stage, command, **kwargs):
                    result = run(stage, command, **kwargs)
                    if "--build" in command:
                        self.assertEqual(preserved, {str(path): (digest(path), path.stat().st_mtime_ns) for path in retained})
                        self.assertTrue(stamp.is_file())
                        self.assertFalse(baseline.exists())  # This writer legitimately superseded the native owner.
                        witness = subprocess.check_output([str(binary), "17"], text=True, timeout=10).strip()
                        self.assertEqual(witness, "value=17")
                        builds.append(str(result))
                        payload = None
                        if fault in {"archive", "stamp"}:
                            payload = archive if fault == "archive" else stamp
                            previous = payload.stat()
                            payload.write_bytes(payload.read_bytes() + b"mutated shared output")
                            os.utime(payload, ns=(previous.st_atime_ns, previous.st_mtime_ns))
                        elif fault == "process":
                            payload = Path(record["process_tree"]["receipt"])
                            build_evidence.write_json(payload, {})
                        elif fault == "log":
                            payload = self.root / record["output"]
                            payload.write_bytes(payload.read_bytes() + b"mutated shared producer log\n")
                        elif fault == "receipt":
                            payload = producer
                            build_evidence.write_json(payload, {**record, "seconds": record["seconds"] + 1})
                        elif fault == "owner":
                            payload = baseline
                            baseline.write_bytes(original_owner)
                        build_evidence.write_json(self.campaign.directory / "shared-fallback.json", {
                            "producer": str(producer), "original_owner": str(baseline), "build": str(result),
                            "preserved": preserved, "witness": witness, "fault": fault, "missing_stamp": missing_stamp,
                            "payload": str(payload) if payload else None, "payload_sha256": digest(payload) if payload else None})
                    return result

                with patch.object(build_evidence, "_admit_missing_scope", side_effect=observe_admission), patch.object(self.campaign, "run", side_effect=mutate):
                    if fault:
                        message = {"archive": "shared Rust artifact bytes changed", "stamp": "shared Rust artifact bytes changed", "process": "command evidence changed",
                                   "log": "command evidence changed", "receipt": "shared Rust preparation receipt changed",
                                   "owner": "shared Rust preparation owner changed"}[fault]
                        with self.assertRaisesRegex(ValueError, message):
                            self.prepare(targets=["overlap"])
                    else:
                        self.prepare(targets=["overlap"])
                self.assertEqual(admissions, [None])
                self.assertEqual(len(builds), 1)
                self.assertFalse(old_pending.exists())
                if fault:
                    self.assertFalse(selected.exists())
                    self.assertTrue(pending.exists())
                    self.assertEqual(self.campaign.attempts("prepare")[-1]["status"], "failed")
                    continue
                current = json.loads(self.prepared.read_text())
                self.assertEqual(current["preparation_admission"]["artifact_owners"], {})
                self.assertEqual(current["preparation_admission"]["archive_owners"], {str(producer): digest(producer)})
                self.assertEqual(current["scoped_preparation"]["rust_invalidated_packages"], [])
                self.assertTrue(current["scoped_preparation"]["invalidated"])
                self.assertFalse(baseline.exists())
                self.assertFalse(pending.exists())
                with OutputLease(self.root, self.build):
                    build_evidence.check_scoped_preparation(self.campaign, self.build, self.prepared)
                self.execute(tests=["selected"], junit=self.campaign.directory / "selected.xml")

    def test_shared_rust_fallback_rejects_mutated_producer_and_outputs(self):
        self.assert_shared_rust_fallback(("archive", "log", "owner", "process", "receipt", "stamp"))

    def test_shared_rust_fallback_preserves_archive_and_executes_new_native_behavior(self):
        self.assert_shared_rust_fallback()

    def test_shared_rust_fallback_preserves_archive_with_missing_stamp(self):
        self.assert_shared_rust_fallback(missing_stamp=True)

    def assert_fresh_scope_entry(self, *, rust):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
            "add_library(unrelated OBJECT native.cpp)\n" + "".join(
                f"add_custom_target(scope_{i} DEPENDS behavior)\n"
                f"add_test(NAME selected_{i} COMMAND behavior 10)\n" for i in range(3)))
        if rust:
            self.enable_rust()
        warm = self.prepare(targets=["behavior", "unrelated"])
        binary = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        self.assertEqual(subprocess.check_output([str(binary), "10"], text=True, timeout=10).strip(), "value=10")
        warm_baseline, _ = build_evidence.preparation_paths(self.build, ["behavior", "unrelated"])
        warm_owner = warm_baseline.read_bytes()
        proof = json.loads(warm.read_text())["scoped_preparation"]
        artifacts = {name: (digest(Path(name)), Path(name).stat().st_mtime_ns) for name in proof["outputs"]}
        self.assertEqual(set(proof["output_identities"]), set(artifacts))
        for i in range(3):
            targets = [f"scope_{i}"]
            baseline, pending = build_evidence.preparation_paths(self.build, targets)
            self.assertFalse(baseline.exists())
            self.assertFalse(pending.exists())
            start = len(self.campaign.attempts("prepare"))
            self.prepare(targets=targets)
            record = json.loads(self.prepared.read_text())
            selected = record["scoped_preparation"]
            self.assertTrue(record["preparation_admission"]["artifact_owners"])
            self.assertEqual(selected["targets"], targets)
            current = json.loads(baseline.read_text())
            self.assertEqual(current.pop("producer_sha256"), digest(self.prepared))
            self.assertEqual(current, selected)
            self.assertFalse(pending.exists())
            self.assertEqual(selected["invalidated"], [])
            self.assertEqual(selected["rust_invalidated_packages"], [])
            attempts = self.campaign.attempts("prepare")[start:]
            self.assertEqual(len(attempts), 1)
            self.assertIn("--build", attempts[0]["identity"]["command"])
            output = (self.root / attempts[0]["output"]).read_text()
            self.assertIn("no work to do", output)
            self.assertEqual(artifacts, {name: (digest(Path(name)), Path(name).stat().st_mtime_ns) for name in artifacts})
            self.assertEqual(warm_baseline.read_bytes(), warm_owner)
            with OutputLease(self.root, self.build):
                build_evidence.check_scoped_preparation(self.campaign, self.build, warm)
                build_evidence.check_scoped_preparation(self.campaign, self.build, self.prepared)
            self.execute(tests=[f"selected_{i}"], junit=self.campaign.directory / f"selected-{i}.xml")

    def test_new_native_scopes_preserve_fresh_broader_outputs_and_execute_tests(self):
        self.assert_fresh_scope_entry(rust=False)

    def test_new_rust_scopes_preserve_fresh_archive_stamp_and_execute_tests(self):
        self.assert_fresh_scope_entry(rust=True)

    def test_missing_scope_admission_rejects_changed_or_incomplete_owner_facts(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + "add_custom_target(overlap DEPENDS behavior)\n")
        self.prepare()
        baseline, pending = build_evidence.preparation_paths(self.build, ["behavior"])
        selected, _ = build_evidence.preparation_paths(self.build, ["overlap"])
        record = json.loads(self.prepared.read_text())
        proof = record["scoped_preparation"]
        process = Path(record["process_tree"]["receipt"])
        log = self.root / record["output"]
        binary = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        obj = next(Path(name) for name in proof["outputs"] if name.endswith((".o", ".obj")))
        paths = [baseline, self.prepared, process, log, binary, obj,
                 self.source / "native.cpp", self.build / "generated/binding.h",
                 self.build / "compile_commands.json"]
        originals = {p: (p.read_bytes(), p.stat()) for p in paths}

        def admit(environment=None):
            return build_evidence._admit_missing_scope(self.campaign, self.source, self.build,
                selected, ["overlap"], self.inputs(), self.args, dict(os.environ) if environment is None else environment)

        with OutputLease(self.root, self.build):
            self.assertIsNotNone(admit())
            for fault in ("legacy", "partial", "foreign", "arguments", "proof", "raw_receipt", "raw_timing", "conflict",
                          "process", "log", "pending", "object", "missing_object", "executable",
                          "restored_source", "generated", "include_order", "membership", "environment"):
                with self.subTest(fault=fault):
                    try:
                        changed = json.loads(json.dumps(proof))
                        if fault in {"legacy", "partial", "foreign", "arguments", "proof"}:
                            if fault == "legacy":
                                changed.pop("output_identities")
                            elif fault == "partial":
                                changed["output_identities"].pop(str(binary))
                            elif fault == "foreign":
                                changed["source"] = str(self.root)
                            elif fault == "arguments":
                                changed["configure_args"].append("-DOTHER_ABI=1")
                            else:
                                changed["producer"] = "superseded"
                            build_evidence.write_json(baseline, changed)
                        elif fault in {"raw_receipt", "raw_timing"}:
                            modified = json.loads(self.prepared.read_text())
                            if fault == "raw_receipt":
                                name = next(iter(modified["identity"]["inputs"]))
                                modified["identity"]["inputs"][name] = "mutated raw input receipt"
                            else:
                                modified["seconds"] += 1
                            build_evidence.write_json(self.prepared, modified)
                        elif fault == "conflict":
                            conflict, _ = build_evidence.preparation_paths(self.build, ["behavior", "overlap"])
                            changed["targets"] = ["behavior", "overlap"]
                            build_evidence.write_json(conflict, {**changed, "producer_sha256": digest(self.prepared)})
                        elif fault == "process":
                            build_evidence.write_json(process, {"drained": False})
                        elif fault == "pending":
                            pending.write_text("unfinished writer")
                        elif fault == "missing_object":
                            obj.unlink()
                        elif fault == "membership":
                            (self.source / "new-shadow.h").write_text("// new namespace member\n")
                        elif fault == "environment":
                            self.assertIsNone(admit({**os.environ, "JUICER_SCOPE_SENTINEL": "changed"}))
                            continue
                        else:
                            path = {"log": log, "object": obj, "executable": binary,
                                    "restored_source": self.source / "native.cpp",
                                    "generated": self.build / "generated/binding.h",
                                    "include_order": self.build / "compile_commands.json"}[fault]
                            content, stamp = originals[path]
                            if fault == "restored_source":
                                path.write_bytes(content.replace(b"return 1", b"return 8"))
                            elif fault == "generated":
                                path.write_bytes(content.replace(b"BINDING 4", b"BINDING 8"))
                            else:
                                path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
                            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                        try:
                            self.assertIsNone(admit())
                        except (ValueError, subprocess.CalledProcessError):
                            pass  # Rejection of corrupt configuration is conservative too.
                    finally:
                        for path, (content, stamp) in originals.items():
                            path.write_bytes(content)
                            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                        pending.unlink(missing_ok=True)
                        (self.source / "new-shadow.h").unlink(missing_ok=True)
                        conflict, _ = build_evidence.preparation_paths(self.build, ["behavior", "overlap"])
                        conflict.unlink(missing_ok=True)
            self.assertIsNotNone(admit())
        self.execute()

    def test_new_scope_rejects_mutated_rust_archive_and_stamp(self):
        self.enable_rust()
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + "add_custom_target(overlap DEPENDS behavior)\n")
        self.prepare()
        proof = json.loads(self.prepared.read_text())["scoped_preparation"]
        artifacts = [Path(name) for name in proof["outputs"]
                     if Path(name).name in {"cargo-build.stamp", "libfilm_juicer_plugin.a", "film_juicer_plugin.lib"}]
        self.assertEqual(len(artifacts), 2)
        selected, _ = build_evidence.preparation_paths(self.build, ["overlap"])
        for artifact in artifacts:
            with self.subTest(artifact=artifact.name):
                content, stamp = artifact.read_bytes(), artifact.stat()
                try:
                    artifact.write_bytes(content + b"mutated output")
                    os.utime(artifact, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    with OutputLease(self.root, self.build), self.assertRaisesRegex(ValueError, "artifact bytes changed"):
                        build_evidence._admit_missing_scope(self.campaign, self.source, self.build,
                            selected, ["overlap"], self.inputs(), self.args, dict(os.environ))
                finally:
                    artifact.write_bytes(content)
                    os.utime(artifact, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.execute()
        archive = next(path for path in artifacts if path.name != "cargo-build.stamp")
        archive.write_bytes(archive.read_bytes() + b"mutated archive")
        with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
            self.prepare(targets=["overlap"])
        self.assertFalse(selected.exists())
        self.assertTrue(selected.with_suffix(".pending.json").exists())

    def test_missing_scope_checks_cmake_in_supplied_environment(self):
        import shutil
        import shlex

        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + "add_custom_target(overlap DEPENDS behavior)\n")
        real_cmake = shutil.which("cmake")
        self.assertIsNotNone(real_cmake)
        directories = [self.root / name for name in ("supplied-tools", "agent-tools")]
        for directory in directories:
            directory.mkdir()
        first = directories[0] / ("cmake.exe" if os.name == "nt" else "cmake")
        second = directories[1] / first.name
        if os.name == "nt":
            wrapper = self.root / "cmake-wrapper.c"
            wrapper.write_text('#include <process.h>\nint main(int argc, char **argv) { (void)argc; '
                'return (int)_spawnv(_P_WAIT, ' + json.dumps(real_cmake) + ', (const char *const *)argv); }\n')
            with OutputLease(self.root, self.build):
                self.campaign.run("prepare", ["clang-cl.exe", "/nologo", "/TC", "/std:c11", "-fuse-ld=lld",
                    str(wrapper), "/Fe" + str(first), "/Fo" + str(self.root / "cmake-wrapper.obj")],
                    inputs=[wrapper.relative_to(self.root).as_posix()], timeout=30,
                    vctip_executable=build_evidence.windows_vctip(os.environ))
        else:
            first.write_text("#!/bin/sh\nexec " + shlex.quote(real_cmake) + ' "$@"\n')
            first.chmod(0o755)
        shutil.copyfile(first, second)
        second.chmod(first.stat().st_mode)
        original = first.read_bytes()
        base_path = os.environ["PATH"]
        environment = {**os.environ, "PATH": str(directories[0]) + os.pathsep + base_path}
        selected, _ = build_evidence.preparation_paths(self.build, ["overlap"])
        with patch.dict(os.environ, {"PATH": str(directories[1]) + os.pathsep + base_path}):
            self.prepare(env=environment)
            with OutputLease(self.root, self.build):
                self.assertIsNotNone(build_evidence._admit_missing_scope(self.campaign, self.source, self.build,
                    selected, ["overlap"], self.inputs(), self.args, environment))
                try:
                    first.write_bytes(original + b"\n")
                    self.assertIsNone(build_evidence._admit_missing_scope(self.campaign, self.source, self.build,
                        selected, ["overlap"], self.inputs(), self.args, environment))
                finally:
                    first.write_bytes(original)
            self.execute(env=environment)

    def test_complete_proof_preserves_unrelated_objects_for_selected_target(self):
        self.prepare()
        record = json.loads(self.prepared.read_text())
        objects = {
            Path(name): (digest(Path(name)), Path(name).stat().st_mtime_ns)
            for name in record["scoped_preparation"]["outputs"]
            if "main.cpp." in name and name.endswith((".o", ".obj"))
        }
        self.assertTrue(objects)
        path = self.source / "native.cpp"
        stamp = path.stat()
        path.write_bytes(path.read_bytes().replace(b"return 1", b"return 8"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.prepare(targets=["behavior"])
        self.assertEqual(
            objects, {path: (digest(path), path.stat().st_mtime_ns) for path in objects}
        )
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute(tests=["behavior"])
        failure = self.campaign.attempts("tests")[-1]
        output = (self.root / failure["output"]).read_text()
        self.assertIn("value=17", output)
        self.assertNotIn("RegularExpression::compile", output)

    def test_changed_compile_configuration_cannot_select_old_dependency_logs(self):
        first = self.prepare()
        with OutputLease(self.root, self.build):
            build_evidence.check_scoped_preparation(self.campaign, self.build, first)
        path = self.source / "CMakeLists.txt"
        path.write_text(path.read_text() + "set(CMAKE_CXX_FLAGS -DNEW_VARIANT)\n")
        with OutputLease(self.root, self.build), self.assertRaisesRegex(ValueError, "dependency contents"):
            build_evidence.check_scoped_preparation(self.campaign, self.build, first)
        self.prepare()
        self.execute()

    def write_interruptible_preparation(self):
        self.args = ["-G", "Ninja"]
        (self.source / "payload.txt").write_text("prepared payload\n", encoding="utf-8")
        child = (
            "import os,sys,time\nfrom pathlib import Path\n"
            "progress=Path(sys.argv[2])\n"
            "progress.write_bytes(b'preparation child writer started\\n')\n"
            "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
            "for _ in range(1500):\n"
            " with progress.open('ab') as stream:stream.write(b'preparation child writer\\n')\n"
            " time.sleep(.02)\n"
        )
        (self.source / "prepare.py").write_text(
            "import os,subprocess,sys\nfrom pathlib import Path\n"
            "payload,build=map(Path,sys.argv[1:])\n"
            "mode=build/'interrupt-mode'\n"
            "if mode.is_file() and mode.read_text()=='interrupt':\n"
            f" child={child!r}\n"
            " process=subprocess.Popen([sys.executable,'-c',child,str(build/'writer-ready'),str(build/'writer-progress.log')],start_new_session=os.name!='nt')\n"
            " process.wait()\n"
            "(build/'material.txt').write_bytes(payload.read_bytes())\n",
            encoding="utf-8",
        )
        python = str(Path(sys.base_prefix) / "python.exe") if os.name == "nt" else sys.executable
        (self.source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\nproject(PreparationRecovery NONE)\nenable_testing()\n"
            'add_custom_target(behavior ALL COMMAND "' + python.replace("\\", "/")
            + '" "${CMAKE_SOURCE_DIR}/prepare.py" "${CMAKE_SOURCE_DIR}/payload.txt" "${CMAKE_BINARY_DIR}"\n'
            ' DEPENDS "${CMAKE_SOURCE_DIR}/prepare.py" "${CMAKE_SOURCE_DIR}/payload.txt"\n'
            ' BYPRODUCTS "${CMAKE_BINARY_DIR}/material.txt")\n'
            'add_test(NAME behavior COMMAND "${CMAKE_COMMAND}" -E compare_files '
            '"${CMAKE_SOURCE_DIR}/payload.txt" "${CMAKE_BINARY_DIR}/material.txt")\n',
            encoding="utf-8",
        )

    def test_interrupted_preparation_recovers_only_after_real_descendant_drainage(self):
        self.write_interruptible_preparation()
        self.prepare()
        self.execute()
        baseline, pending = build_evidence.preparation_paths(self.build, ["behavior"])
        original_baseline = baseline.read_bytes()
        retained = {path: path.read_bytes() for path in self.campaign.directory.glob("attempts/*/*") if path.is_file()}
        (self.build / "interrupt-mode").write_text("interrupt")
        wait = subprocess.Popen.wait
        interrupted = False

        def interrupt_build(process, *args, **kwargs):
            nonlocal interrupted
            if not interrupted and "--build" in process.args and any("process_tree.py" in str(part) for part in process.args):
                deadline = time.monotonic() + 10
                while not (self.build / "writer-ready").is_file() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue((self.build / "writer-ready").is_file(), "real descendant did not start")
                interrupted = True
                raise KeyboardInterrupt
            return wait(process, *args, **kwargs)

        with patch.object(subprocess.Popen, "wait", interrupt_build), self.assertRaises(KeyboardInterrupt):
            self.prepare()
        self.assertTrue(interrupted)
        failure = self.campaign.attempts("prepare")[-1]
        self.assertEqual(failure["status"], "failed")
        self.assertTrue(failure["process_tree"]["drained"])
        self.assertEqual(failure["process_tree"]["status"], "interrupted")
        progress = self.build / "writer-progress.log"
        progress_bytes = progress.read_bytes()
        self.assertIn(b"preparation child writer started\n", progress_bytes)
        time.sleep(.1)
        self.assertEqual(progress.read_bytes(), progress_bytes)
        self.assertFalse((self.campaign.directory / "running.lock").exists())
        self.assertFalse(OutputLease(self.root, self.build).path.exists())
        self.assertEqual(baseline.read_bytes(), original_baseline)
        original_pending = pending.read_bytes()
        self.assertTrue(all(path.read_bytes() == content for path, content in retained.items()))
        retained.update({path: path.read_bytes() for path in self.campaign.directory.glob("attempts/*/*") if path.is_file()})
        with self.assertRaisesRegex(ValueError, "pending preparation"):
            self.execute()
        (self.build / "interrupt-mode").write_text("recovered")
        self.prepare()
        self.assertFalse(pending.exists())
        self.assertNotEqual(baseline.read_bytes(), original_baseline)
        recovery = json.loads(self.prepared.read_text())["preparation_admission"]["recovered_pending"]
        self.assertEqual(Path(recovery["file"]).read_bytes(), original_pending)
        self.assertEqual(digest(Path(recovery["file"])), recovery["sha256"])
        self.assertTrue(all(path.read_bytes() == content for path, content in retained.items()))
        self.execute()
        self.assertEqual(progress.read_bytes(), progress_bytes)

    def test_pending_recovery_requires_complete_unchanged_invocation_proof(self):
        self.write_interruptible_preparation()
        self.prepare()
        baseline, pending = build_evidence.preparation_paths(self.build, ["behavior"])
        attempts_before = len(self.campaign.attempts("prepare"))
        with patch.object(build_evidence, "preparation_inventory", side_effect=KeyboardInterrupt), \
                self.assertRaises(KeyboardInterrupt):
            self.prepare()
        no_command = pending.read_bytes()
        no_command_record = json.loads(no_command)
        proof = json.loads(Path(no_command_record["failure_proof"]["file"]).read_text())
        self.assertEqual(proof["completion"], "no-command-started")
        self.assertEqual(proof["attempts"], [])
        self.assertEqual(len(self.campaign.attempts("prepare")), attempts_before)
        self.prepare()
        recovered = json.loads(self.prepared.read_text())["preparation_admission"]["recovered_pending"]
        self.assertEqual(Path(recovered["file"]).read_bytes(), no_command)
        self.assertEqual(digest(Path(recovered["file"])), recovered["sha256"])
        earlier = json.loads(self.prepared.read_text())
        earlier_process = Path(earlier["process_tree"]["receipt"]).read_bytes()
        baseline_before = baseline.read_bytes()
        self.args.append("-DRECOVERY_PROBE=ON")
        with patch.object(build_evidence, "_check_preparation_result", side_effect=ValueError("controlled publication failure")), \
                self.assertRaisesRegex(ValueError, "controlled publication failure"):
            self.prepare()
        marker = json.loads(pending.read_text())
        proof_path = Path(marker["failure_proof"]["file"])
        proof = json.loads(proof_path.read_text())
        self.assertEqual(proof["completion"], "commands-drained")
        self.assertEqual(len(proof["attempts"]), 2)
        configure_attempt, build_attempt = [Path(item["directory"]) for item in proof["attempts"]]
        self.assertIn("-S", json.loads((configure_attempt / "result.json").read_text())["identity"]["command"])
        self.assertIn("--build", json.loads((build_attempt / "result.json").read_text())["identity"]["command"])
        configure_process = (configure_attempt / "process.json").read_bytes()
        attempt = build_attempt
        process = attempt / "process.json"
        self.assertEqual(json.loads(process.read_text())["status"], "completed")
        protected = [baseline, self.build / "material.txt", *self.campaign.directory.glob("attempts/*/*")]
        protected = {path: path.read_bytes() for path in protected if path.is_file()}
        originals = {path: path.read_bytes() for path in (pending, proof_path, process)}
        attempts_before = len(self.campaign.attempts("prepare"))
        for fault in ("legacy", "running", "missing-proof", "changed-proof", "substituted-proof",
                      "unknown-drainage", "substituted-process", "substituted-configure"):
            with self.subTest(fault=fault):
                candidate = json.loads(originals[pending])
                if fault == "legacy":
                    candidate = {"protocol": build_evidence.SCOPED_PROTOCOL, "targets": ["behavior"]}
                elif fault == "running":
                    candidate["state"] = "running"
                elif fault == "missing-proof":
                    candidate.pop("failure_proof")
                elif fault == "changed-proof":
                    proof_path.write_bytes(originals[proof_path] + b"\n")
                elif fault == "substituted-proof":
                    candidate["failure_proof"] = no_command_record["failure_proof"]
                elif fault == "unknown-drainage":
                    tree = json.loads(originals[process])
                    tree["drained"] = False
                    process.write_text(json.dumps(tree), encoding="utf-8")
                elif fault == "substituted-process":
                    process.write_bytes(earlier_process)
                else:
                    process.write_bytes(configure_process)
                pending.write_text(json.dumps(candidate), encoding="utf-8")
                presented = {path: path.read_bytes() for path in originals}
                try:
                    with patch.object(self.campaign, "run", side_effect=AssertionError("invalid recovery dispatched a command")), \
                            self.assertRaisesRegex(ValueError, "recover pending preparation"):
                        self.prepare()
                    self.assertEqual({path: path.read_bytes() for path in originals}, presented)
                    self.assertEqual(len(self.campaign.attempts("prepare")), attempts_before)
                    self.assertEqual(baseline.read_bytes(), baseline_before)
                    self.assertTrue(all(path.read_bytes() == content for path, content in protected.items() if path != process))
                finally:
                    for path, content in originals.items():
                        path.write_bytes(content)
        self.assertTrue(all(path.read_bytes() == content for path, content in protected.items()))
        # A later started attempt has no terminal proof. An earlier completed
        # command cannot establish drainage for this additional writer.
        later = self.campaign.directory / "attempts/prepare-unproved"
        later.mkdir()
        record = json.loads((attempt / "result.json").read_text())
        record.update(status="running", attempt_number=record["attempt_number"] + 1)
        (later / "result.json").write_text(json.dumps(record), encoding="utf-8")
        with patch.object(self.campaign, "run", side_effect=AssertionError("later unproved attempt launched recovery")), \
                self.assertRaisesRegex(ValueError, "recover pending preparation"):
            self.prepare()
        self.assertEqual(pending.read_bytes(), originals[pending])
        self.assertEqual(baseline.read_bytes(), baseline_before)
        self.assertTrue(all(path.read_bytes() == content for path, content in protected.items()))

    def test_selected_utility_failure_requires_fresh_preparation(self):
        script = self.source / "fail_once.py"
        script.write_text(
            "from pathlib import Path\nimport sys\np=Path(sys.argv[1])\nif not p.exists():\n p.write_text('failed')\n raise SystemExit(7)\n"
        )
        with (self.source / "CMakeLists.txt").open("a") as stream:
            stream.write(
                'add_custom_target(fail_once ALL COMMAND "'
                + sys.executable.replace("\\", "/")
                + '" "${CMAKE_SOURCE_DIR}/fail_once.py" "${CMAKE_BINARY_DIR}/failed-once" DEPENDS behavior)\n'
            )
        self.build.mkdir()
        marker = self.build / "failed-once"
        marker.write_text("warm setup")
        targets = ["fail_once"]
        self.prepare(targets=targets)
        record = json.loads(self.prepared.read_text())
        outputs = {
            Path(name): (digest(Path(name)), Path(name).stat().st_mtime_ns)
            for name in record["scoped_preparation"]["outputs"]
        }
        self.assertTrue(outputs)
        baseline = next(self.build.glob("verification-build-*.json"))
        saved = baseline.read_bytes()
        pending = baseline.with_suffix(".pending.json")
        marker.unlink()
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare(targets=targets)
        self.assertEqual(marker.read_text(), "failed")
        self.assertEqual(
            outputs, {path: (digest(path), path.stat().st_mtime_ns) for path in outputs}
        )
        self.assertEqual(baseline.read_bytes(), saved)
        pending_contents = pending.read_bytes()
        failure = self.campaign.attempts("prepare")[-1]
        self.assertEqual(failure["status"], "failed")
        self.assertTrue(failure["process_tree"]["drained"])
        failed = self.root / Path(failure["output"]).with_name("result.json")
        retained = {
            path: path.read_bytes()
            for path in failed.parent.iterdir()
            if path.is_file()
        }
        attempts = len(self.campaign.attempts("prepare"))
        with self.assertRaisesRegex(ValueError, "legacy sealed-build"):
            self.prepare(targets=targets, resume=failed)
        self.assertEqual(len(self.campaign.attempts("prepare")), attempts)
        self.assertEqual(baseline.read_bytes(), saved)
        self.assertEqual(pending.read_bytes(), pending_contents)
        with self.assertRaisesRegex(ValueError, "pending preparation"):
            self.execute()
        self.prepare(targets=targets)
        self.assertFalse(pending.exists())
        self.assertNotEqual(baseline.read_bytes(), saved)
        self.assertTrue(
            all(path.read_bytes() == value for path, value in retained.items())
        )
        self.execute()

    def test_failed_compile_preserves_baseline_until_a_repaired_build(self):
        self.prepare()
        self.execute()
        baseline = next(self.build.glob("verification-build-*.json"))
        saved = baseline.read_bytes()
        path = self.source / "native.cpp"
        main = self.source / "main.cpp"
        original, stamp = path.read_bytes(), path.stat()
        main_original, main_stamp = main.read_bytes(), main.stat()
        path.write_text("this cannot compile\n")
        main.write_bytes(main_original.replace(b"native()+header()", b"native()+7+header()"))
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare()
        self.assertEqual(baseline.read_bytes(), saved)
        self.assertTrue(list(self.build.glob("verification-build-*.pending.json")))
        path.write_bytes(original)
        main.write_bytes(main_original)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        os.utime(main, ns=(main_stamp.st_atime_ns, main_stamp.st_mtime_ns))
        self.prepare()
        self.assertFalse(list(self.build.glob("verification-build-*.pending.json")))
        self.execute()

    def test_mutation_does_not_advance_successful_baseline(self):
        self.prepare()
        baseline = next(self.build.glob("verification-build-*.json"))
        original = baseline.read_bytes()
        run = self.campaign.run

        def mutate(stage, command, **kwargs):
            result = run(stage, command, **kwargs)
            if "--build" in command:
                (self.source / "native.cpp").write_text("int native(){return 8;}\n")
            return result

        with (
            patch.object(self.campaign, "run", mutate),
            self.assertRaisesRegex(ValueError, "mutated"),
        ):
            self.prepare()
        self.assertEqual(baseline.read_bytes(), original)

    def test_preparation_publication_rejects_changed_process_proof_or_missing_output(self):
        self.write_interruptible_preparation()
        write = build_evidence.write_json
        for fault in ("process", "output"):
            with self.subTest(fault=fault):
                self.prepare()
                baseline = next(path for path in self.build.glob("verification-build-*.json") if not path.name.endswith(".pending.json"))
                original = baseline.read_bytes()
                process_before = {}

                def corrupt(path, document):
                    write(path, document)
                    if "scoped_preparation" in document and document.get("status") == "passed":
                        if fault == "process":
                            process = Path(document["process_tree"]["receipt"])
                            process_before[process] = process.read_bytes()
                            raise ValueError("controlled failure before preparation publication")
                        else:
                            (self.build / "material.txt").unlink()

                with patch.object(build_evidence, "write_json", side_effect=corrupt), self.assertRaises(ValueError):
                    self.prepare()
                self.assertEqual(baseline.read_bytes(), original)
                self.assertTrue(list(self.build.glob("verification-build-*.pending.json")))
                if fault == "process":
                    pending = baseline.with_suffix(".pending.json")
                    pending_before = pending.read_bytes()
                    attempts = len(self.campaign.attempts("prepare"))
                    for process in process_before:
                        write(process, {"drained": False})
                    with patch.object(self.campaign, "run", side_effect=AssertionError("unproved process launched a retry")), \
                            self.assertRaises(ValueError):
                        self.prepare()
                    self.assertEqual(baseline.read_bytes(), original)
                    self.assertEqual(pending.read_bytes(), pending_before)
                    self.assertEqual(len(self.campaign.attempts("prepare")), attempts)
                    for process, content in process_before.items():
                        process.write_bytes(content)
        self.prepare()
        self.execute()
        baseline, pending = build_evidence.preparation_paths(self.build, ["behavior"])
        baseline_before = baseline.read_bytes()

        def unproved_publication(path, document):
            write(path, document)
            if "scoped_preparation" in document and document.get("status") == "passed":
                write(Path(document["process_tree"]["receipt"]), {"drained": False})

        with patch.object(build_evidence, "write_json", side_effect=unproved_publication), self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(baseline.read_bytes(), baseline_before)
        self.assertEqual(json.loads(pending.read_text())["state"], "running")
        pending_before = pending.read_bytes()
        attempts = len(self.campaign.attempts("prepare"))
        with patch.object(self.campaign, "run", side_effect=AssertionError("unproved publication launched a retry")), \
                self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(baseline.read_bytes(), baseline_before)
        self.assertEqual(pending.read_bytes(), pending_before)
        self.assertEqual(len(self.campaign.attempts("prepare")), attempts)

    def test_symbolic_utility_preserves_real_output_checks(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
                         'add_custom_target(stage COMMAND "${CMAKE_COMMAND}" -E copy_if_different '
                         '"${CMAKE_CURRENT_SOURCE_DIR}/native.cpp" "${CMAKE_BINARY_DIR}/stage-data" '
                         'DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/native.cpp" '
                         'BYPRODUCTS "${CMAKE_BINARY_DIR}/stage-data")\nadd_dependencies(behavior stage)\n')
        self.prepare()
        proof = json.loads(self.prepared.read_text())["scoped_preparation"]
        symbolic = self.build / "CMakeFiles/stage"
        self.assertFalse(symbolic.exists())
        self.assertNotIn(str(symbolic), proof["outputs"])
        real = self.build / "stage-data"
        self.assertIn(str(real), proof["outputs"])
        self.assertEqual(real.read_bytes(), (self.source / "native.cpp").read_bytes())
        self.execute()
        for path in [next(Path(name) for name in proof["outputs"] if name.endswith((".o", ".obj"))), real]:
            path.unlink()
            with self.assertRaises(ValueError):
                self.execute()
            self.prepare()
            self.execute()

    def test_fetched_header_content_invalidates_native_value(self):
        # CMake FetchContent sources live in _deps; they are content-bound
        # inputs, unlike undeclared generated headers elsewhere in the tree.
        header = self.build / "_deps/fetched-src/value.h"
        header.parent.mkdir(parents=True)
        header.write_text("#define FETCHED_VALUE 1\n")
        (self.source / "native.cpp").write_text('#include "value.h"\nint native(){return FETCHED_VALUE;}\n')
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + 'target_include_directories(behavior PRIVATE "${CMAKE_BINARY_DIR}/_deps/fetched-src")\n')
        self.prepare()
        self.execute()
        stamp = header.stat()
        header.write_text("#define FETCHED_VALUE 8\n")
        os.utime(header, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "dependency contents or membership changed"):
            self.execute()
        self.prepare()
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        header.write_text("#define FETCHED_VALUE 1\n")
        os.utime(header, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.prepare()
        self.execute()

    def test_generated_build_header_survives_reconfiguration(self):
        (self.source / "step-source.h").write_text("#pragma once\n")
        main = self.source / "main.cpp"
        main.write_text('#include "step.h"\n' + main.read_text())
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
            'add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/generated/step.h" '
            'COMMAND "${CMAKE_COMMAND}" -E copy_if_different '
            '"${CMAKE_CURRENT_SOURCE_DIR}/step-source.h" "${CMAKE_BINARY_DIR}/generated/step.h" '
            'DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/step-source.h")\n'
            'add_custom_target(step-header DEPENDS "${CMAKE_BINARY_DIR}/generated/step.h")\n'
            'add_dependencies(behavior step-header)\n')
        self.prepare()
        generated = self.build / "generated/step.h"
        self.assertTrue(generated.is_file())
        graph = build_evidence.NinjaGraph(self.build, ["behavior"])
        consumed = {name for dependencies in graph.compiler_inputs.values()
                    for name in dependencies}
        self.assertIn(str(generated), consumed)
        self.args.append("-DRECONFIGURE_CHECK=ON")
        self.prepare()
        proof = json.loads(self.prepared.read_text())["scoped_preparation"]
        self.assertTrue(generated.is_file())
        self.assertNotIn(str(generated), proof["context"]["files"])
        unowned = self.build / "generated/unowned.h"
        unowned.write_text("// shadows a future include\n")
        with self.assertRaisesRegex(ValueError, "dependency contents or membership changed"):
            self.execute()
        unowned.unlink()
        self.prepare()
        self.execute()

    def test_generated_shadow_header_addition_cannot_reuse_old_executable(self):
        self.prepare()
        shadow = self.build / "shadow/binding.h"
        shadow.parent.mkdir(exist_ok=True)
        shadow.write_text("#define BINDING 8\n")
        with self.assertRaisesRegex(
            ValueError, "dependency contents or membership changed"
        ):
            self.execute()
        with self.assertRaisesRegex(ValueError, "unresolved generated build input"):
            self.prepare()
        shadow.unlink()
        self.prepare()
        self.execute()

    def test_partial_build_cannot_supply_unfiltered_qualification(self):
        self.prepare(targets=["behavior"])
        with self.assertRaisesRegex(ValueError, "legacy sealed-build"):
            self.execute(tests=[])


class FixtureLifetimeTests(unittest.TestCase):
    def run_fixture(self, artifact_root, retain, body):
        environment = {
            "JUICER_TEST_ARTIFACT_DIR": str(artifact_root),
            "JUICER_TEST_RETAIN_FIXTURES": "1" if retain else "",
        }
        case = BuildEvidenceTests("test_unchanged_reconfigure_preserves_registry_and_artifacts")
        case.test_unchanged_reconfigure_preserves_registry_and_artifacts = lambda: body(case)
        with patch.dict(os.environ, environment):
            result = case.run(unittest.TestResult())
        self.addCleanup(case.temporary.cleanup)
        return case, result

    def test_artifact_routing_cleans_successful_build_fixture(self):
        with tempfile.TemporaryDirectory() as artifact_root:
            case, result = self.run_fixture(artifact_root, False, lambda case: None)
            self.assertTrue(result.wasSuccessful())
            self.assertFalse(case.root.exists())

    def test_cleanup_handles_long_descendant_paths(self):
        def create_long_descendant(case):
            path = Path(case.temporary.name) / ("incremental-" + "a" * 90) / ("session-" + "b" * 90) / ("c" * 90 + ".o")
            self.assertGreater(len(str(path)), 260)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"incremental object")
            self.assertEqual(path.read_bytes(), b"incremental object")
            self.assertFalse(str(case.root).startswith("\\\\?\\"))

        with tempfile.TemporaryDirectory() as artifact_root:
            case, result = self.run_fixture(artifact_root, False, create_long_descendant)
            self.assertTrue(result.wasSuccessful(), result.errors)
            self.assertFalse(case.root.exists())

    def test_diagnostic_opt_in_retains_successful_build_fixture(self):
        with tempfile.TemporaryDirectory() as artifact_root:
            case, result = self.run_fixture(artifact_root, True, lambda case: None)
            self.assertTrue(result.wasSuccessful())
            self.assertTrue((case.root / MARKER).is_file())
            self.assertEqual(json.loads((case.root / MARKER).read_text())["outcome"], "passed")

    def test_missing_result_retains_build_fixture(self):
        case = BuildEvidenceTests("test_unchanged_reconfigure_preserves_registry_and_artifacts")
        case.setUp()
        self.addCleanup(case.temporary.cleanup)
        case.doCleanups()
        self.assertTrue((case.root / MARKER).is_file())

    def test_test_failures_and_blocked_owner_retain_build_fixtures(self):
        with tempfile.TemporaryDirectory() as artifact_root:
            def fail_test(case):
                case.fail("fixture lifecycle control")

            failed_case, result = self.run_fixture(artifact_root, False, fail_test)
            self.assertEqual(len(result.failures), 1)
            failure_marker = json.loads((failed_case.root / MARKER).read_text())
            self.assertEqual(failure_marker["test"], failed_case.id())
            self.assertEqual(failure_marker["outcome"], "failed")

            def fail_subtest(case):
                with case.subTest(control="failure"):
                    case.fail("fixture lifecycle control")

            failed_case, result = self.run_fixture(artifact_root, False, fail_subtest)
            self.assertEqual(len(result.failures), 1)
            failure_marker = json.loads((failed_case.root / MARKER).read_text())
            self.assertEqual(failure_marker["test"], failed_case.id())

            def leave_blocked_owner(case):
                owner = case.root / "out/verification-locks/control/owner.json"
                owner.parent.mkdir(parents=True)
                owner.write_text('{"status":"blocked"}\n')

            blocked_case, result = self.run_fixture(artifact_root, False, leave_blocked_owner)
            self.assertTrue(result.wasSuccessful())
            marker = json.loads((blocked_case.root / MARKER).read_text())
            owner = blocked_case.root / "out/verification-locks/control/owner.json"
            self.assertEqual(marker["blocked_owners"], [str(owner)])
            self.assertEqual(marker["outcome"], "passed")


class FileEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.build = self.root / "build"

    def test_identity_reads_reject_symlink_retarget_and_restored_time_content(self):
        from build_dependencies import file_identities

        first, second, alias = [self.root / name for name in ("first", "second", "alias")]
        for directory, text in [(first, "first"), (second, "other")]:
            directory.mkdir()
            (directory / "input.txt").write_text(text)

        def link_to(target):
            if os.name == "nt":
                subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(alias), str(target)], check=True, capture_output=True, timeout=10)
            else:
                alias.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=True)

        def unlink():
            os.rmdir(alias) if os.name == "nt" else alias.unlink()

        link_to(first)
        self.addCleanup(unlink)
        path = alias / "input.txt"
        baseline = file_identities([path])
        traversal = alias / ".." / "first" / "input.txt"
        self.assertEqual(file_identities([traversal])[str(traversal)]["resolved"], str((first / "input.txt").resolve()))
        self.assertEqual(file_identities([self.root / "absent"]), {})
        if os.name != "nt":
            leaf = self.root / "leaf.txt"
            leaf.symlink_to("first/input.txt")
            self.assertEqual(file_identities([leaf])[str(leaf)]["resolved"], str((first / "input.txt").resolve()))
            dangling = self.root / "dangling.txt"
            dangling.symlink_to("absent")
            self.assertEqual(file_identities([dangling]), {})
        stamp = path.stat()
        path.write_text("third")
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.assertNotEqual(file_identities([path]), baseline)
        original = build_evidence.digest
        (second / "input.txt").write_text("third")

        def retarget(value):
            checksum = original(value)
            unlink()
            link_to(second)
            return checksum

        with patch("build_dependencies.digest", side_effect=retarget), self.assertRaisesRegex(ValueError, "input path changed"):
            file_identities([path])
        self.assertEqual(file_identities([path])[str(path)]["resolved"], str((second / "input.txt").resolve()))

    def test_output_lease_spans_readers_and_conflicting_callers(self):
        with OutputLease(self.root, self.build):
            environment = {
                key: value
                for key, value in os.environ.items()
                if key != "JUICER_OUTPUT_LEASES"
            }
            command = [
                sys.executable,
                "-B",
                "-c",
                "import sys;sys.path.insert(0,sys.argv[1]);from execution import OutputLease;from pathlib import Path;"
                "with_lease=OutputLease(Path(sys.argv[2]),Path(sys.argv[3]),timeout=.2);with_lease.__enter__()",
                str(ROOT / "scripts"),
                str(self.root),
                str(self.build),
            ]
            result = subprocess.run(
                command, env=environment, capture_output=True, timeout=10
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"output remains owned", result.stderr)
        with OutputLease(self.root, self.build):
            pass


if __name__ == "__main__":
    unittest.main()
