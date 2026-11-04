"""Shared tooling registration and native CTest discovery contracts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/ctest"))
import execution
import qualification
from control_artifacts import record_fixture

ROOT = Path(__file__).resolve().parents[2]

# Independently declared registration membership. Keep synchronized with the
# owning CMake contract when a quality command is added or removed.
TOOLING_TESTS = (
    "Quality.NativeResults", "Quality.HostResults", "Quality.GpuResults",
    "Quality.Dispatcher", "Quality.ToolingLint", "Quality.Execution", "Quality.BuildEvidence",
    "Quality.BuildRegistration", "Quality.Verify", "Quality.Checkpoint",
    "Ctest.Host.Report", "Ctest.Host.Repetition", "Ffi.Host.RustBuild",
)
WINDOWS_TOOLING_SHARDS = {
    "core": {
        "Quality.NativeResults", "Quality.HostResults", "Quality.GpuResults", "Quality.Dispatcher",
        "Quality.ToolingLint", "Quality.Execution", "Quality.BuildRegistration", "Quality.Verify",
        "Ctest.Host.Report", "Ctest.Host.Repetition", "Ffi.Host.RustBuild",
    },
    "BuildEvidence": {"Quality.BuildEvidence"},
    "Checkpoint": {"Quality.Checkpoint"},
}


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        parent = Path(os.environ.get("JUICER_TEST_ARTIFACT_DIR", ROOT / "out/validation" / self.preset)) / "registration-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="registration controls ü ", dir=parent, delete=False)
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.finish_fixture)
        self.build = self.root / "out/build" / self.preset
        self.build.mkdir(parents=True)
        self.directory = self.root / "out/validation" / self.preset / "results/registration"
        self.directory.mkdir(parents=True)
        self.artifacts = self.root / "routed artifacts"
        self.checkout = self.root / "checkout"
        (self.checkout / "tests/tooling").mkdir(parents=True)
        (self.checkout / "tests/ToolingTests.cmake").write_text(
            (ROOT / "tests/ToolingTests.cmake").read_text(encoding="utf-8"), encoding="utf-8"
        )
        (self.checkout / "tests/tooling/CMakeLists.txt").write_text(
            (ROOT / "tests/tooling/CMakeLists.txt").read_text(encoding="utf-8"), encoding="utf-8"
        )
        self.lease = execution.OutputLease(self.root, self.build)
        self.lease.__enter__()
        self.addCleanup(self.lease.__exit__, None, None, None)
        self.native_run = execution.NativeRun(self.root, self.directory)

    def finish_fixture(self):
        outcome = record_fixture(self.root, self)
        owners = list((self.root / "out/verification-locks").glob("*/owner.json"))
        if outcome == "passed" and not owners and not os.environ.get("JUICER_TEST_RETAIN_FIXTURES"):
            try:
                self.temporary.cleanup()
            except BaseException:
                record_fixture(self.root, self, "failed")
                print("Registration cleanup failed; retained:", self.root, file=sys.stderr)
                raise
        else:
            print("Retained registration evidence:", self.root, file=sys.stderr)

    @property
    def cmake(self):
        return shutil.which("cmake")

    @property
    def ctest(self):
        return shutil.which("ctest")

    def configure(self, *, source=None, build=None, artifacts=None):
        source = source or self.checkout / "tests/tooling"
        build = build or self.build
        artifacts = artifacts or self.artifacts
        python = Path(sys.executable)
        cargo = shutil.which("cargo")
        ninja = shutil.which("ninja")
        compiler = (r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin\clang-cl.exe"
                    if os.name == "nt" else shutil.which("gcc-13"))
        self.assertTrue(self.cmake and self.ctest and cargo and ninja and compiler,
                        "shared tooling configure requires CMake, CTest, Cargo, Ninja and the native C compiler")
        environment = dict(os.environ)
        self.native_run.command("configure", [
            self.cmake, "-S", str(source), "-B", str(build), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Debug", "-DPython3_EXECUTABLE=" + str(python),
            "-DJUICER_TOOLING_CARGO=" + cargo,
            "-DJUICER_TOOLING_ROOT=" + str(self.checkout),
            "-DJUICER_TOOLING_ARTIFACT_ROOT=" + str(artifacts),
            "-DCMAKE_C_COMPILER=" + compiler,
        ], env=environment, timeout=120,
            vctip_executable=execution.windows_vctip(environment))
        listing = self.directory / "registration.json"
        self.native_run.command("discovery", [self.ctest, "--test-dir", str(build), "--show-only=json-v1"],
                         timeout=30, stdout_only=True)
        # CTest's JSON is also retained as the direct stdout of the discovery command.
        log = self.directory / "discovery.log"
        listing.write_bytes(log.read_bytes())
        return json.loads(listing.read_text(encoding="utf-8"))

    def test_shared_tooling_registration_membership_and_artifact_routing(self):
        product_entry = ROOT / "tests/CMakeLists.txt"
        standalone_entry = ROOT / "tests/tooling/CMakeLists.txt"
        self.assertIn('include("${CMAKE_CURRENT_SOURCE_DIR}/ToolingTests.cmake")', product_entry.read_text())
        self.assertIn('include("${JUICER_TOOLING_ROOT}/tests/ToolingTests.cmake")', standalone_entry.read_text())
        self.assertIn("juicer_register_tooling_tests(", product_entry.read_text())
        self.assertIn("juicer_register_tooling_tests(", standalone_entry.read_text())

        inventory = self.configure()
        tests = inventory["tests"]
        names = [test["name"] for test in tests]
        self.assertCountEqual(names, TOOLING_TESTS)
        self.assertEqual(len(names), len(set(names)))
        labels_by_name = {}
        for test in tests:
            properties = {row["name"]: row["value"] for row in test.get("properties", [])}
            labels = properties.get("LABELS", [])
            labels_by_name[test["name"]] = set(labels)
            self.assertIn("host", labels, test["name"])
            self.assertIn("tooling", labels, test["name"])
            timeout = properties.get("TIMEOUT")
            self.assertIsInstance(timeout, (int, float), test["name"])
            self.assertTrue(math.isfinite(timeout) and timeout > 0, test["name"])
            expected_timeout = 120 if test["name"] in {
                "Quality.NativeResults", "Quality.HostResults", "Quality.GpuResults"
            } else 600
            self.assertEqual(timeout, expected_timeout, test["name"])
            tooling_labels = {label for label in labels if label.startswith("tooling-")}
            self.assertEqual(len(tooling_labels), 1, test["name"])
            if test["name"].startswith("Quality."):
                environment = properties.get("ENVIRONMENT", [])
                artifact_values = [value.partition("=")[2] for value in environment
                                   if value.startswith("JUICER_TEST_ARTIFACT_DIR=")]
                if artifact_values:
                    self.assertEqual(len(artifact_values), 1, test["name"])
                    self.assertTrue(Path(artifact_values[0]).is_relative_to(self.artifacts), test["name"])
        rust_build = next(test for test in tests if test["name"] == "Ffi.Host.RustBuild")
        self.assertEqual(Path(rust_build["command"][rust_build["command"].index("--artifacts") + 1]),
                         self.artifacts / "ffi/rust-build")

        for shard, expected_names in WINDOWS_TOOLING_SHARDS.items():
            selected = sorted(name for name in names if f"tooling-{shard}" in labels_by_name[name])
            self.assertCountEqual(selected, expected_names, f"Windows tooling shard {shard}")

        self.native_run.command("selected-discovery", [self.ctest, "--test-dir", str(self.build),
                           "--show-only=json-v1", "-R", "^Quality.BuildRegistration$"],
                         stdout_only=True)
        selected = json.loads((self.directory / "selected-discovery.log").read_text())
        self.assertEqual([test["name"] for test in selected["tests"]], ["Quality.BuildRegistration"])
        self.native_run.publish({"schema": "native-run-v1", "registration": names})

    def test_actual_shared_cmake_omission_fails_independent_membership(self):
        source = self.checkout / "tests/ToolingTests.cmake"
        original = source.read_text(encoding="utf-8")
        omission = re.sub(
            r"(?ms)^\s*juicer_add_python_test\(NAME Ctest\.Host\.Report\b.*?^\s*list\(APPEND tooling_tests Ctest\.Host\.Report\)\s*$",
            "", original, count=1,
        )
        self.assertNotEqual(omission, original, "the expected Python test wrapper must have one owning definition")
        source.write_text(omission, encoding="utf-8")
        inventory = self.configure(build=self.root / "omitted-build")
        names = [test["name"] for test in inventory["tests"]]
        with self.assertRaises(AssertionError):
            self.assertCountEqual(names, TOOLING_TESTS)
        self.assertNotIn("Ctest.Host.Report", names)


class MembershipRecordTests(unittest.TestCase):
    def test_required_group_and_runtime_membership_are_checked_as_values(self):
        inventory = {"tests": [
            {"name": "product.case", "command": ["python", "--scratch-root", "/tmp/product"],
             "properties": [{"name": "LABELS", "value": ["host", "product"]}]},
            {"name": "tool.case", "command": ["python"],
             "properties": [{"name": "LABELS", "value": ["host", "tooling"]}]},
            {"name": "device.case", "command": ["python"],
             "properties": [{"name": "LABELS", "value": ["gpu", "product"]}]},
        ]}
        host = qualification.membership(inventory, ["product"], "host")
        all_runtime = qualification.membership(inventory, ["product"], "all")
        self.assertEqual(host, {"product": ["product.case"]})
        self.assertEqual(all_runtime, {"product": ["device.case", "product.case"]})

    def test_duplicate_disabled_and_colliding_scratch_registrations_reject(self):
        def row(name, command, labels=("host", "product"), disabled=False):
            properties = [{"name": "LABELS", "value": list(labels)}]
            if disabled:
                properties.append({"name": "DISABLED", "value": True})
            return {"name": name, "command": command, "properties": properties}

        cases = (
            [row("same", ["python"]), row("same", ["python"])],
            [row("disabled", ["python"], disabled=True)],
            [row("first", ["python", "--scratch-root", "/tmp/shared"]),
             row("second", ["python", "--scratch-root", "/tmp/shared"], labels=("host", "boundary"))],
        )
        self.subtest_cases = ("duplicate", "disabled", "scratch-collision")
        for case_id, inventory in zip(self.subtest_cases, cases):
            with self.subTest(case=case_id), self.assertRaises(ValueError):
                qualification.membership({"tests": inventory}, ["product", "boundary"], "host")


MembershipRecordTests.test_duplicate_disabled_and_colliding_scratch_registrations_reject.required_subtests = True
MembershipRecordTests.test_duplicate_disabled_and_colliding_scratch_registrations_reject.subtest_cases = (
    "duplicate", "disabled", "scratch-collision"
)


if __name__ == "__main__":
    unittest.main()
