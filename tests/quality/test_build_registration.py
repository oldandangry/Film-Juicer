"""Real selected CTest admission and launcher contracts, without runtime sealing."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_evidence
import execution

ROOT = Path(__file__).resolve().parents[2]


class BuildTests(unittest.TestCase):
    def setUp(self):
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        parent = ROOT / "out/validation" / preset / "registration-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, delete=False)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.build = self.root / "build"
        self.python = str(Path(sys.base_prefix) / "python.exe") if os.name == "nt" else sys.executable
        cmake = ('cmake_minimum_required(VERSION 3.24)\nproject(Admission NONE)\nenable_testing()\n'
                 'add_custom_target(prepared COMMAND "${CMAKE_COMMAND}" -E copy_if_different '
                 '"${CMAKE_CURRENT_SOURCE_DIR}/required.txt" "${CMAKE_BINARY_DIR}/material.txt" '
                 'DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/required.txt" BYPRODUCTS "${CMAKE_BINARY_DIR}/material.txt")\n'
                 'add_test(NAME behavior COMMAND "${PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/case.py" '
                 '"${CMAKE_CURRENT_SOURCE_DIR}/required.txt" "${CMAKE_BINARY_DIR}/trace.log")\n')
        (self.source / "CMakeLists.txt").write_text(cmake)
        (self.source / "required.txt").write_text("expected")
        (self.source / "case.py").write_text("from pathlib import Path\nimport sys\nassert Path(sys.argv[1]).read_text()=='expected'\nPath(sys.argv[2]).write_text('trace')\n")
        self.inputs = [str(p.relative_to(self.root)) for p in self.source.iterdir()]
        self.campaign = execution.Campaign(self.root, preset, "admission", ["prepare", "tests"])
        self.lease = execution.OutputLease(self.root, self.build)
        self.lease.__enter__()
        def finish():
            self.lease.__exit__(None, None, None)
            result = self._outcome.result
            if any(test is self for test, _ in [*result.errors, *result.failures]):
                print("Retained registration fixture:", self.root, file=sys.stderr)
            else:
                self.temporary.cleanup()
        self.addCleanup(finish)

    def prepare(self):
        _, self.prepared = build_evidence.prepare_build(self.campaign, "prepare", source=self.source,
            build=self.build, inputs=self.inputs, timeout=30, targets=["prepared"],
            configure_args=["-G", "Ninja", "-DPYTHON=" + self.python])

    def execute(self):
        return build_evidence.run_ctest(self.campaign, "tests", build=self.build,
            prepared=self.prepared, inputs=self.inputs, timeout=30, tests=["behavior"])

    def test_selected_membership_executes_without_recursive_runtime_inventory(self):
        self.prepare()
        path = self.execute()
        record = json.loads(path.read_text())
        self.assertEqual(record["status"], "passed")
        self.assertTrue(record["process_tree"]["drained"])
        self.assertIn("1/1 Test", (path.parent / "output.log").read_text())
        self.assertEqual(record["identity"]["tools"], {})

    def test_local_admission_rejects_gpu_without_checkpoint_escape(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + '\nset_tests_properties(behavior PROPERTIES LABELS "gpu")\n')
        self.prepare()
        with patch.object(self.campaign, "run") as run, self.assertRaisesRegex(ValueError, "GPU tests require explicit checkpoint"):
            self.execute()
        run.assert_not_called()
        with self.assertRaisesRegex(TypeError, "checkpoint"):
            build_evidence.run_ctest(self.campaign, "tests", build=self.build, prepared=self.prepared,
                inputs=self.inputs, timeout=30, tests=["behavior"], checkpoint=True)

    def test_registration_change_rejects_before_test_launch(self):
        self.prepare()
        registration = self.build / "CTestTestfile.cmake"
        registration.write_text(registration.read_text() + '\nset(CHANGED true)\n')
        with patch.object(self.campaign, "run") as run, self.assertRaises(ValueError):
            self.execute()
        run.assert_not_called()

    def test_discovery_mutation_rejects_before_execution(self):
        self.prepare()
        original = subprocess.check_output
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            if "--show-only=json-v1" in args[0]:
                (self.source / "required.txt").write_text("changed")
            return result
        with patch.object(build_evidence.subprocess, "check_output", side_effect=mutate), patch.object(self.campaign, "run") as run, self.assertRaises(ValueError):
            self.execute()
        run.assert_not_called()

    def test_trace_output_replacement_is_not_an_input_or_a_failed_test(self):
        self.prepare()
        self.execute()
        (self.build / "trace.log").write_text("previous output")
        self.execute()
        self.assertEqual((self.build / "trace.log").read_text(), "trace")
        (self.source / "case.py").write_text("raise SystemExit(2)\n")
        self.prepare()
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()

    def test_explicit_trace_input_mutation_cannot_pass(self):
        trace = self.source / "trace.txt"
        trace.write_text("input")
        self.inputs.append(str(trace.relative_to(self.root)))
        case = self.source / "case.py"
        case.write_text(case.read_text() + "Path(__file__).with_name('trace.txt').write_text('changed')\n")
        self.prepare()
        with self.assertRaisesRegex(ValueError, "changed"):
            self.execute()

    def test_missing_or_disabled_registered_case_cannot_pass(self):
        self.prepare()
        path = self.build / "CTestTestfile.cmake"
        path.write_text(path.read_text() + '\nset_tests_properties(behavior PROPERTIES DISABLED TRUE)\n')
        with self.assertRaises(ValueError):
            self.execute()
        (self.source / "case.py").unlink()
        with self.assertRaises((ValueError, OSError)):
            self.execute()

    def test_python_file_launcher_resolves_sibling_imports_and_test_selection(self):
        (self.source / "sibling.py").write_text("VALUE=42\n")
        test = self.source / "test_import.py"
        test.write_text("import sibling,unittest\nclass ImportTest(unittest.TestCase):\n def test_value(self): self.assertEqual(sibling.VALUE,42)\nif __name__=='__main__': unittest.main()\n")
        result = subprocess.run(execution.python_command(test, "ImportTest.test_value", python=self.python),
                                cwd=self.root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Ran 1 test", result.stderr)

    def test_unsupported_registration_is_not_silently_omitted(self):
        self.prepare()
        path = self.build / "CTestTestfile.cmake"
        path.write_text(path.read_text() + '\ninclude("${UNRESOLVED}/tests.cmake")\n')
        with self.assertRaisesRegex(ValueError, "unsupported generated test include"):
            build_evidence.registration_files(self.build)

    @unittest.skipUnless(os.name == "nt", "Windows linker contract")
    def test_direct_rustc_uses_cargo_linker_and_rejects_override(self):
        linker = os.environ["CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER"]
        self.assertEqual(execution.rustc_command(["--version"])[1:3], ["-C", "linker=" + linker])
        with self.assertRaisesRegex(ValueError, "override"):
            execution.rustc_command(["-Clinker=other"])
        with self.assertRaisesRegex(ValueError, "qualified linker is missing"):
            execution.rustc_command(["--version"], env={})


if __name__ == "__main__":
    unittest.main()
