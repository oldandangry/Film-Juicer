"""Real CMake dependency witnesses under the native output owner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/ctest"))

from build_dependencies import CompilationInputs, header_consumers
from execution import NativeRun, OutputLease
from control_artifacts import MARKER, record_fixture, selected_paths

ROOT = Path(__file__).resolve().parents[2]


class CompilationInputsTests(unittest.TestCase):
    def test_fresh_boundary_observes_include_directory_symlink_retarget(self):
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
                    subprocess.run(
                        ["cmd.exe", "/c", "mklink", "/J", str(link), str(directory)],
                        check=True, capture_output=True, timeout=10,
                    )
                else:
                    link.symlink_to(directory, target_is_directory=True)

            link_to(root / "first")
            row = {"file": "main.cpp", "directory": str(root),
                   "arguments": ["c++", "-I" + str(link)]}
            (build / "compile_commands.json").write_text(json.dumps([row]))
            first = CompilationInputs(root, build)
            self.assertEqual(first.paths(row), {"main.cpp", "first/entry.h"})
            os.rmdir(link) if os.name == "nt" else link.unlink()
            link_to(root / "second")
            final = CompilationInputs(root, build)
            self.assertEqual(final.paths(row), {"main.cpp", "second/entry.h", "selected.h"})

    def test_ordered_variants_and_fresh_boundary_observe_changed_includes(self):
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
                {"file": "main.cpp", "directory": str(root),
                 "arguments": ["c++", "-I" + str(root / a), "-I" + str(root / b), *extra]}
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
            (root / "second/other.h").write_text("// new earlier match\n")
            final = CompilationInputs(root, build)
            self.assertEqual(final.paths(rows[1]), {"main.cpp", "second/other.h", "forced.h"})


class NativeDependencyWitnessTests(unittest.TestCase):
    def setUp(self):
        self.preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        artifact_root = Path(os.environ.get(
            "JUICER_TEST_ARTIFACT_DIR", ROOT / "out/validation" / self.preset
        )).resolve() / "build-evidence-controls"
        artifact_root.mkdir(parents=True, exist_ok=True)
        # Nested lifetime controls must fit Windows compiler-tool path limits.
        self.scratch = Path(tempfile.mkdtemp(prefix="n ü ", dir=artifact_root))
        self.succeeded = False
        self.addCleanup(self.finish_scratch)
        self.source = self.scratch / "source"
        self.build = self.scratch / "build"
        self.logs = self.scratch / "logs"
        self.source.mkdir()
        self.logs.mkdir()
        self.native_run = NativeRun(self.scratch, self.logs)
        self.cmake = shutil.which("cmake")
        self.ninja = shutil.which("ninja")
        self.compiler = Path(shutil.which("clang-cl") or "clang-cl.exe") if os.name == "nt" else Path("/usr/bin/g++-13")

    def finish_scratch(self):
        outcome = record_fixture(self.scratch, self)
        owners = list((self.scratch / "out/verification-locks").glob("*/owner.json"))
        if self.succeeded and outcome == "passed" and not owners and not os.environ.get("JUICER_TEST_RETAIN_FIXTURES"):
            try:
                shutil.rmtree(self.scratch)
            except BaseException:
                record_fixture(self.scratch, self, "failed")
                print(f"Native dependency cleanup failed; retained: {self.scratch}", file=sys.stderr)
                raise
        else:
            print(f"Retained native dependency evidence: {self.scratch}", file=sys.stderr)

    def configure(self):
        self.assertTrue(self.cmake and self.ninja, "CMake and Ninja are required for native dependency evidence")
        self.assertTrue(self.compiler.is_file(), f"qualified native compiler is missing: {self.compiler}")
        (self.source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(NativeDependencyWitness LANGUAGES CXX)\n"
            "set(CMAKE_CXX_STANDARD 20)\n"
            "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
            "set(CMAKE_CXX_EXTENSIONS OFF)\n"
            "add_library(selected_producer STATIC producer.cpp)\n"
            "target_include_directories(selected_producer PUBLIC \"${CMAKE_CURRENT_SOURCE_DIR}\")\n"
            "add_executable(selected_consumer main.cpp)\n"
            "target_link_libraries(selected_consumer PRIVATE selected_producer)\n"
            "add_executable(unrelated_consumer unrelated.cpp)\n",
            encoding="utf-8",
        )
        (self.source / "value.h").write_text("inline constexpr int value = 1;\n", encoding="utf-8")
        (self.source / "producer.h").write_text(
            '#include "value.h"\nint produced_value();\n', encoding="utf-8"
        )
        (self.source / "producer.cpp").write_text(
            '#include "producer.h"\nint produced_value() { return value + 1; }\n', encoding="utf-8"
        )
        (self.source / "main.cpp").write_text(
            "#include <concepts>\n#include <iostream>\n#include \"producer.h\"\n"
            "static_assert(std::same_as<decltype(produced_value()), int>);\n"
            "int main() { std::cout << produced_value() << '\\n'; }\n",
            encoding="utf-8",
        )
        (self.source / "unrelated.cpp").write_text(
            "int main() { return 0; }\n", encoding="utf-8"
        )
        args = [self.cmake, "-S", str(self.source), "-B", str(self.build), "-G", "Ninja",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", f"-DCMAKE_CXX_COMPILER={self.compiler}"]
        if os.name == "nt":
            args.append("-DCMAKE_LINKER_TYPE=LLD")
        self.native_run.command("configure", args, timeout=120, cwd=self.source)

    def build_selected(self, label):
        return self.native_run.command(
            label, [self.cmake, "--build", str(self.build), "--target", "selected_consumer", "--verbose"],
            timeout=120, cwd=self.source,
        )

    def execute_selected(self, label):
        executable = self.build / ("selected_consumer.exe" if os.name == "nt" else "selected_consumer")
        return self.native_run.command(label, [str(executable)], timeout=10, cwd=self.source).read_text(
            encoding="utf-8"
        ).strip()

    def test_selected_cxx20_closure_rebuilds_only_its_consumers(self):
        with OutputLease(self.scratch, self.build):
            self.configure()
            first_build = self.build_selected("build-initial").read_text(encoding="utf-8")
            self.assertIn("producer.cpp", first_build)
            self.assertIn("main.cpp", first_build)
            self.assertNotIn("unrelated.cpp", first_build)
            self.assertEqual(self.execute_selected("run-initial"), "2")

            query = subprocess.check_output
            queries = []
            def observe_query(command, **options):
                output = query(command, **options)
                queries.append((command[command.index("-t") + 1], len(output.encode("utf-8"))))
                return output
            with patch("build_dependencies.subprocess.check_output", side_effect=observe_query):
                targets, sources = header_consumers(self.source, self.build, ["value.h"])
            self.assertEqual([tool for tool, _ in queries], ["deps"])
            print("Header selection Ninja query/output bytes:", queries)
            self.assertEqual(sources, ["main.cpp", "producer.cpp"])
            self.assertEqual(len(targets), 2)

            executable = self.build / ("selected_consumer.exe" if os.name == "nt" else "selected_consumer")
            producer_object = next(self.build.rglob("producer.cpp.obj" if os.name == "nt" else "producer.cpp.o"))
            original_times = executable.stat().st_mtime_ns, producer_object.stat().st_mtime_ns
            no_op = self.build_selected("build-no-op").read_text(encoding="utf-8").lower()
            self.assertIn("no work to do", no_op)
            self.assertEqual(
                (executable.stat().st_mtime_ns, producer_object.stat().st_mtime_ns), original_times
            )

            (self.source / "producer.cpp").write_text(
                '#include "producer.h"\nint produced_value() { return value + 10; }\n',
                encoding="utf-8",
            )
            before_producer_edit = executable.stat().st_mtime_ns, producer_object.stat().st_mtime_ns
            producer_edit = self.build_selected("build-producer-edit").read_text(encoding="utf-8")
            self.assertIn("producer.cpp", producer_edit)
            self.assertNotIn("unrelated.cpp", producer_edit)
            self.assertNotEqual(executable.stat().st_mtime_ns, before_producer_edit[0])
            self.assertNotEqual(producer_object.stat().st_mtime_ns, before_producer_edit[1])
            self.assertEqual(self.execute_selected("run-producer-edit"), "11")

            (self.source / "value.h").write_text("inline constexpr int value = 20;\n", encoding="utf-8")
            before_header_edit = executable.stat().st_mtime_ns, producer_object.stat().st_mtime_ns
            header_edit = self.build_selected("build-header-edit").read_text(encoding="utf-8")
            self.assertIn("producer.cpp", header_edit)
            self.assertIn("main.cpp", header_edit)
            self.assertNotIn("unrelated.cpp", header_edit)
            self.assertNotEqual(executable.stat().st_mtime_ns, before_header_edit[0])
            self.assertNotEqual(producer_object.stat().st_mtime_ns, before_header_edit[1])
            self.assertEqual(self.execute_selected("run-header-edit"), "30")

            self.native_run.publish({
                "schema": "native-cmake-dependency-v1",
                "preset": self.preset,
                "targets": ["selected_consumer"],
                "commands": self.native_run.commands,
            })
            self.assertEqual(json.loads((self.logs / "result.json").read_text())[
                "schema"], "native-cmake-dependency-v1")
            self.succeeded = True

    def test_failed_native_command_retains_scratch_diagnostics(self):
        with OutputLease(self.scratch, self.build):
            self.configure()
            with self.assertRaises(subprocess.CalledProcessError):
                self.native_run.command(
                    "build-missing-target",
                    [self.cmake, "--build", str(self.build), "--target", "missing_target"],
                    timeout=30,
                    cwd=self.source,
                )
            self.assertTrue((self.logs / "build-missing-target.log").is_file())
            self.assertTrue((self.logs / "build-missing-target.process.json").is_file())
            self.assertFalse((self.logs / "result.json").exists())
            self.assertTrue(self.scratch.is_dir())
            self.succeeded = True


class DependencyFixtureLifetimeTests(unittest.TestCase):
    def run_control(self, *, outer_failure=False, cleanup_failure=False):
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        parent = Path(os.environ.get("JUICER_TEST_ARTIFACT_DIR", ROOT / "out/validation" / preset)) / "build-evidence-controls"
        parent.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(prefix="l-", dir=parent, delete=False)
        case = NativeDependencyWitnessTests("test_failed_native_command_retains_scratch_diagnostics")
        original = case.test_failed_native_command_retains_scratch_diagnostics
        if outer_failure:
            def fail_after_control():
                original()
                case.fail("controlled outer failure")
            case.test_failed_native_command_retains_scratch_diagnostics = fail_after_control
        with patch.dict(os.environ, {"JUICER_TEST_ARTIFACT_DIR": directory.name, "JUICER_TEST_RETAIN_FIXTURES": ""}):
            if cleanup_failure:
                with patch.object(shutil, "rmtree", side_effect=PermissionError("controlled cleanup failure")):
                    result = case.run(unittest.TestResult())
            else:
                result = case.run(unittest.TestResult())
        return directory, case, result

    def test_expected_negative_control_cleans_successful_scratch(self):
        directory, case, result = self.run_control()
        self.assertTrue(result.wasSuccessful(), result.errors)
        self.assertFalse(case.scratch.exists())
        directory.cleanup()

    def test_outer_failure_preserves_owned_diagnostics_and_selection(self):
        directory, case, result = self.run_control(outer_failure=True)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("controlled outer failure", result.failures[0][1])
        self.assertTrue(case.scratch.is_dir())
        self.assertEqual(json.loads((case.scratch / MARKER).read_bytes())["outcome"], "failed")
        self.assertEqual(selected_paths(Path(directory.name)), [case.scratch])
        directory.cleanup()

    def test_cleanup_failure_preserves_failed_marker_and_selection(self):
        directory, case, result = self.run_control(cleanup_failure=True)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("controlled cleanup failure", result.errors[0][1])
        self.assertEqual(json.loads((case.scratch / MARKER).read_bytes())["outcome"], "failed")
        self.assertEqual(selected_paths(Path(directory.name)), [case.scratch])
        directory.cleanup()


if __name__ == "__main__":
    unittest.main()
