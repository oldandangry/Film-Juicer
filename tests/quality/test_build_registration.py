"""Real selected CTest admission and launcher contracts, without runtime sealing."""
from __future__ import annotations
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/ctest"))
from control_artifacts import MARKER, record_fixture
import build_evidence
import execution

ROOT = Path(__file__).resolve().parents[2]

TOOLING_TESTS = (
    "Quality.Dispatcher", "Quality.ToolingLint", "Quality.Execution", "Quality.BuildEvidence",
    "Quality.BuildRegistration", "Quality.Verify", "Quality.Checkpoint", "Quality.PreparationRenewal",
    "Ctest.Host.Report", "Ctest.Host.Repetition", "Ffi.Host.RustBuild",
)


class BuildTests(unittest.TestCase):
    def run(self, result=None):
        try:
            return super().run(result)
        except KeyboardInterrupt:
            self.interrupted = True
            self.retain = True
            if hasattr(self, "lease") and self.lease.path in execution.OutputLease.held:
                self.lease.blocked = True
            while self._cleanups:
                try:
                    self.doCleanups()
                except BaseException:
                    # Cancellation clears unittest's outcome; keep draining callbacks
                    # if its error reporter cannot attach a cleanup failure.
                    traceback.print_exc(file=sys.stderr)
            raise

    def setUp(self):
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        artifact_root = os.environ.get("JUICER_TEST_ARTIFACT_DIR")
        parent = (Path(artifact_root) if artifact_root else ROOT / "out/validation" / preset) / "registration-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, delete=False)
        self.root = Path(self.temporary.name).resolve()
        self.retain = bool(os.environ.get("JUICER_TEST_RETAIN_FIXTURES"))

        def finish_fixture():
            outcome = record_fixture(self.root, self)
            failed = outcome != "passed"
            owners = list(self.root.glob("out/verification-locks/*/owner.json"))
            if failed or self.retain or owners:
                print(f"Retained registration fixture: {self.root} test={self.id()}", file=sys.stderr, flush=True)
            else:
                try:
                    self.temporary.cleanup()
                except BaseException:
                    record_fixture(self.root, self, "failed")
                    print(f"Registration cleanup failed; remaining fixture: {self.root}", file=sys.stderr, flush=True)
                    raise

        self.addCleanup(finish_fixture)
        record_fixture(self.root, self, "running")
        if self.retain:
            print("Registration fixture:", self.root, file=sys.stderr, flush=True)
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
        self.addCleanup(self.lease.__exit__, None, None, None)

    def test_shared_tooling_entry_has_exact_members_and_rejects_an_omission(self):
        product_entry = ROOT / "tests/CMakeLists.txt"
        standalone_entry = ROOT / "tests/tooling/CMakeLists.txt"
        shared_module = ROOT / "tests/ToolingTests.cmake"
        self.assertIn('include("${CMAKE_CURRENT_SOURCE_DIR}/ToolingTests.cmake")', product_entry.read_text())
        self.assertIn('include("${JUICER_TOOLING_ROOT}/tests/ToolingTests.cmake")', standalone_entry.read_text())
        self.assertIn("juicer_register_tooling_tests(", product_entry.read_text())
        self.assertIn("juicer_register_tooling_tests(", standalone_entry.read_text())
        self.assertIn('"${JUICER_TEST_ARTIFACT_ROOT}" CACHE PATH "Tooling test artifact root"', product_entry.read_text())
        self.assertIn('ARTIFACT_ROOT "${JUICER_TOOLING_ARTIFACT_ROOT}"', product_entry.read_text())

        cmake = shutil.which("cmake")
        ctest = shutil.which("ctest")
        ninja = shutil.which("ninja")
        self.assertTrue(cmake and ctest and ninja, "standalone registration needs CMake, CTest and Ninja")

        def tooling_members(module_text, *, override_artifacts=False):
            with tempfile.TemporaryDirectory(dir=self.root) as directory:
                checkout = Path(directory) / "checkout"
                (checkout / "tests/tooling").mkdir(parents=True)
                (checkout / "tests/ToolingTests.cmake").write_text(module_text)
                (checkout / "tests/tooling/CMakeLists.txt").write_text(standalone_entry.read_text())
                build = Path(directory) / "build"
                artifacts = Path(directory) / "routed-artifacts"
                subprocess.run(
                    [cmake, "-S", str(checkout / "tests/tooling"), "-B", str(build), "-G", "Ninja",
                     "-DCMAKE_BUILD_TYPE=Debug", "-DPython3_EXECUTABLE=" + sys.executable,
                     "-DCMAKE_C_COMPILER=" + ("clang-cl.exe" if os.name == "nt" else "/usr/bin/gcc-13"),
                     "-DJUICER_TOOLING_CARGO=" + sys.executable, "-DJUICER_TOOLING_ROOT=" + str(checkout)]
                    + (["-DJUICER_TOOLING_ARTIFACT_ROOT=" + str(artifacts)] if override_artifacts else []),
                    check=True, capture_output=True, text=True, timeout=120)
                inventory = json.loads(subprocess.check_output(
                    [ctest, "--test-dir", str(build), "--show-only=json-v1"],
                    text=True, timeout=30))
                members = {test["name"] for test in inventory["tests"]}
                partitions = {}
                for shard in ("core", "BuildEvidence", "Checkpoint", "PreparationRenewal"):
                    selected = json.loads(subprocess.check_output(
                        [ctest, "--test-dir", str(build), "--show-only=json-v1",
                         "-L", "^tooling-" + shard + "$"], text=True, timeout=30))
                    names = {test["name"] for test in selected["tests"]}
                    self.assertTrue(names, "empty tooling shard: " + shard)
                    self.assertFalse(set().union(*partitions.values()) & names,
                                     "tooling test assigned to multiple shards")
                    partitions[shard] = names
                    if shard != "core":
                        self.assertEqual(names, {"Quality." + shard})
                self.assertEqual(set().union(*partitions.values()), members)
                if override_artifacts:
                    routed = set()
                    for test in inventory["tests"]:
                        properties = {row["name"]: row["value"] for row in test.get("properties", [])}
                        for variable in properties.get("ENVIRONMENT", []):
                            if variable.startswith("JUICER_TEST_ARTIFACT_DIR="):
                                self.assertTrue(Path(variable.split("=", 1)[1]).is_relative_to(artifacts))
                                routed.add(test["name"])
                        if test["name"] == "Ffi.Host.RustBuild":
                            command = test["command"]
                            self.assertEqual(Path(command[command.index("--artifacts") + 1]), artifacts / "ffi/rust-build")
                    self.assertEqual(routed, {"Quality.Execution", "Quality.BuildEvidence", "Quality.BuildRegistration",
                                              "Quality.Verify", "Quality.Checkpoint", "Quality.PreparationRenewal", "Ctest.Host.Repetition"})
                return {
                    test["name"] for test in inventory["tests"]
                    if "tooling" in next((property_["value"] for property_ in test.get("properties", [])
                                          if property_["name"] == "LABELS"), [])
                }

        source = shared_module.read_text()
        self.assertEqual(tooling_members(source), set(TOOLING_TESTS))
        self.assertEqual(tooling_members(source, override_artifacts=True), set(TOOLING_TESTS))
        omitted_source = source.replace("    list(APPEND tooling_tests Ctest.Host.Report)\n", "", 1)
        self.assertNotEqual(omitted_source, source)
        with self.assertRaises(AssertionError):
            self.assertEqual(tooling_members(omitted_source), set(TOOLING_TESTS))

    def test_passed_output_preserves_initial_diagnostics_and_duration_tail(self):
        from xml.etree import ElementTree

        presets = json.loads((ROOT / "CMakePresets.json").read_text())
        output = next(p for p in presets["testPresets"] if p["name"] == "test-base")["output"]
        (self.source / "case.py").write_text(
            "print('initial-diagnostic')\n"
            + "print('x'*" + str(output["maxPassedTestOutputSize"] * 2) + ")\n"
            + "print('final-duration-summary')\n")
        (self.source / "CMakePresets.json").write_text(json.dumps({
            "version": 6, "configurePresets": [{"name": "control", "generator": "Ninja",
                "binaryDir": str(self.build), "cacheVariables": {"PYTHON": self.python}}],
            "testPresets": [{"name": "control", "configurePreset": "control", "output": output}]}))
        subprocess.run(["cmake", "--preset", "control"], cwd=self.source,
                       check=True, capture_output=True, timeout=30)
        report = self.root / "passed-output.xml"
        subprocess.run(["ctest", "--preset", "control", "--no-tests=error", "--output-junit", str(report)],
                       cwd=self.source, check=True, capture_output=True, timeout=30)
        case = ElementTree.parse(report).getroot().find("testcase")
        self.assertEqual(case.get("status"), "run")
        self.assertIsNone(case.find("failure"))
        retained = case.findtext("system-out")
        self.assertIn("initial-diagnostic", retained)
        self.assertIn("final-duration-summary", retained)
        self.assertLess(len(retained), output["maxPassedTestOutputSize"] * 2)

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


class FixtureLifetimeTests(unittest.TestCase):
    def run_fixture(self, body, *, retain=False):
        case = BuildTests("test_selected_membership_executes_without_recursive_runtime_inventory")
        setattr(case, case._testMethodName, lambda: body(case))
        result = unittest.TestResult()
        interrupted = False
        with patch.dict(os.environ, {"JUICER_TEST_RETAIN_FIXTURES": "1" if retain else ""}):
            try:
                case.run(result)
            except KeyboardInterrupt:
                interrupted = True
        return case, result, interrupted

    def test_routing_cleans_success_and_opt_in_retains(self):
        for retain in (False, True):
            with self.subTest(retain=retain):
                case, result, interrupted = self.run_fixture(lambda case: None, retain=retain)
                self.assertTrue(result.wasSuccessful(), result.errors)
                self.assertFalse(interrupted)
                self.assertEqual(case.root.exists(), retain)
                self.assertNotIn(case.lease.path, execution.OutputLease.held)
                case.temporary.cleanup()

    def test_failures_setup_cleanup_errors_and_missing_result_retain(self):
        def fail(case):
            case.fail("outer lifecycle failure")

        def subtest(case):
            with case.subTest(control="outer failure"):
                case.fail("outer subtest failure")

        def cleanup(case):
            def error():
                raise RuntimeError("cleanup lifecycle error")
            case.addCleanup(error)

        def failure_and_cleanup(case):
            cleanup(case)
            fail(case)

        for name, body, failures, errors in (("assertion", fail, 1, 0), ("subtest", subtest, 1, 0),
                                             ("cleanup", cleanup, 0, 1), ("both", failure_and_cleanup, 1, 1),
                                             ("setup", lambda case: None, 0, 1), ("disposal", lambda case: None, 0, 1)):
            with self.subTest(outcome=name):
                fault = nullcontext()
                if name == "setup":
                    fault = patch.object(execution, "Campaign", side_effect=RuntimeError("setup lifecycle error"))
                elif name == "disposal":
                    fault = patch.object(tempfile.TemporaryDirectory, "cleanup", side_effect=OSError("disposal lifecycle error"))
                with fault:
                    case, result, _ = self.run_fixture(body)
                self.assertEqual(len(result.failures), failures)
                self.assertEqual(len(result.errors), errors)
                self.assertTrue(case.root.exists())
                self.assertEqual(json.loads((case.root / MARKER).read_text())["outcome"], "failed")
                case.temporary.cleanup()
        case = BuildTests("test_selected_membership_executes_without_recursive_runtime_inventory")
        case.setUp()
        case.doCleanups()
        self.assertTrue(case.root.exists())
        case.temporary.cleanup()

    def test_interruption_and_blocked_owner_retain_fixture(self):
        previous = os.environ.get("JUICER_OUTPUT_LEASES")
        for mode in ("before-command", "after-command", "blocked", "cleanup-error"):
            with self.subTest(mode=mode):
                def body(case):
                    if mode == "cleanup-error":
                        def error():
                            raise OSError("cancelled fixture cleanup error")
                        case.addCleanup(error)
                    if mode == "after-command":
                        case.prepare()
                    if mode == "blocked":
                        # No child is launched: this is the existing owner's unproved state.
                        case.lease.blocked = True
                    else:
                        raise KeyboardInterrupt
                case, result, interrupted = self.run_fixture(body)
                self.assertEqual(interrupted, mode != "blocked")
                self.assertTrue(result.wasSuccessful(), result.errors)
                self.assertTrue(case.root.exists())
                self.assertEqual(json.loads((case.lease.path / "owner.json").read_text())["status"], "blocked")
                self.assertNotIn(case.lease.path, execution.OutputLease.held)
                self.assertEqual(os.environ.get("JUICER_OUTPUT_LEASES"), previous)
                # Any real prepare command has drained before the synthetic cancellation.
                case.temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
