"""Shared tooling registration and native CTest discovery contracts."""

from __future__ import annotations

import contextlib
import io
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

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
        compiler = shutil.which("clang-cl" if os.name == "nt" else "gcc-13")
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
            self.assertIn(type(properties.get("TIMEOUT")), (int, float), test["name"])
            self.assertEqual(properties["TIMEOUT"], 0, test["name"])
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

    def resource_fixture(self):
        source = self.checkout / "resource staging ü"
        source.mkdir()
        inputs = source / "inputs"
        payloads = {
            "profiles/film.json": b'{"film": "original"}\n',
            "spectral/table.bin": bytes(range(256)),
            "obsolete.txt": b"removed runtime member\n",
        }
        for name, data in payloads.items():
            path = inputs / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        membership = source / "membership.txt"
        membership.write_text("\n".join(payloads) + "\n", encoding="utf-8")
        stage = self.build / "Resources"
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 4.3)\nproject(ResourceStaging NONE)\n'
            f'include("{(ROOT / "tests/ResourceStaging.cmake").as_posix()}")\n'
            'set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS '
            '"${CMAKE_CURRENT_SOURCE_DIR}/membership.txt")\n'
            'file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/membership.txt" JUICER_RUNTIME_RESOURCES ENCODING UTF-8)\n'
            'juicer_stage_test_resources("${CMAKE_CURRENT_SOURCE_DIR}/inputs" '
            '"${CMAKE_BINARY_DIR}/Resources" "${CMAKE_BINARY_DIR}/resources.stamp")\n',
            encoding="utf-8",
        )
        ninja = shutil.which("ninja")
        self.assertTrue(self.cmake and ninja, "resource staging requires CMake and Ninja")
        self.native_run.command("resource-configure", [self.cmake, "-S", source, "-B", self.build,
                                "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=" + ninja], timeout=60)
        return source, inputs, membership, stage, payloads

    def resource_snapshot(self, stage):
        return {path.relative_to(stage).as_posix():
                (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns)
                for path in stage.rglob("*") if path.is_file()}

    def resource_build(self, name):
        started = time.perf_counter()
        log = self.native_run.command(name, [self.cmake, "--build", self.build,
                                      "--target", "juicer_test_resources"], timeout=60)
        return log.read_text(encoding="utf-8"), time.perf_counter() - started

    def test_resource_staging_updates_only_selected_members(self):
        self.subtest_cases = self.test_resource_staging_updates_only_selected_members.subtest_cases
        source, inputs, membership, stage, payloads = self.resource_fixture()
        observations = {}
        with self.subTest(case="initial"):
            _, seconds = self.resource_build("resource-initial")
            before = self.resource_snapshot(stage)
            self.assertEqual({name: data[0] for name, data in before.items()}, payloads)
            observations["initial"] = {"payload_files": len(before),
                                       "payload_bytes": sum(len(data) for data in payloads.values()),
                                       "seconds": seconds}
        with self.subTest(case="noop"):
            log, seconds = self.resource_build("resource-noop")
            self.assertEqual(self.resource_snapshot(stage), before)
            self.assertNotIn("Copying runtime test resource", log)
            observations["noop"] = {"payload_files": 0, "payload_bytes": 0, "seconds": seconds}
        with self.subTest(case="changed"):
            payloads["profiles/film.json"] = b'{"film": "changed"}\n'
            (inputs / "profiles/film.json").write_bytes(payloads["profiles/film.json"])
            self.assertGreater((inputs / "profiles/film.json").stat().st_mtime_ns,
                               before["profiles/film.json"][2])
            log, seconds = self.resource_build("resource-changed")
            after = self.resource_snapshot(stage)
            self.assertEqual([name for name in before if before[name] != after[name]], ["profiles/film.json"])
            self.assertEqual(after["profiles/film.json"][0], payloads["profiles/film.json"])
            self.assertEqual(log.count("Copying runtime test resource"), 1)
            observations["changed"] = {"payload_files": 1, "payload_bytes": len(payloads["profiles/film.json"]),
                                       "seconds": seconds}
            before = after
        with self.subTest(case="removed"):
            (inputs / "obsolete.txt").unlink()
            del payloads["obsolete.txt"]
            membership.write_text("\n".join(payloads) + "\n", encoding="utf-8")
            log, seconds = self.resource_build("resource-removed")
            self.assertFalse((stage / "obsolete.txt").exists())
            self.assertEqual(self.resource_snapshot(stage), {name: before[name] for name in payloads})
            self.assertNotIn("Copying runtime test resource", log)
            observations["removed"] = {"payload_files": 0, "payload_bytes": 0, "seconds": seconds}
            before = self.resource_snapshot(stage)
        with self.subTest(case="added"):
            payloads["filters/new.bin"] = b"new selected member\n"
            (inputs / "filters").mkdir()
            (inputs / "filters/new.bin").write_bytes(payloads["filters/new.bin"])
            membership.write_text("\n".join(payloads) + "\n", encoding="utf-8")
            log, seconds = self.resource_build("resource-added")
            after = self.resource_snapshot(stage)
            self.assertEqual({name: after[name] for name in before}, before)
            self.assertEqual(after["filters/new.bin"][0], payloads["filters/new.bin"])
            self.assertEqual(log.count("Copying runtime test resource"), 1)
            observations["added"] = {"payload_files": 1, "payload_bytes": len(payloads["filters/new.bin"]),
                                     "seconds": seconds}
            before = after
        with self.subTest(case="configuration"):
            with (source / "CMakeLists.txt").open("a", encoding="utf-8") as stream:
                stream.write("# A configuration change does not invalidate resource payloads.\n")
            log, seconds = self.resource_build("resource-configuration")
            self.assertEqual(self.resource_snapshot(stage), before)
            self.assertNotIn("Copying runtime test resource", log)
            observations["configuration"] = {"payload_files": 0, "payload_bytes": 0, "seconds": seconds}
        with self.subTest(case="mutation"):
            original = (inputs / "profiles/film.json").read_bytes()
            self.assertFalse((stage / "profiles/film.json").samefile(inputs / "profiles/film.json"))
            (stage / "profiles/film.json").write_bytes(b"mutated isolated package fixture\n")
            self.assertEqual((inputs / "profiles/film.json").read_bytes(), original)
            self.assertEqual((inputs / "spectral/table.bin").read_bytes(), payloads["spectral/table.bin"])
        with self.subTest(case="missing-output"):
            (stage / "profiles/film.json").unlink()
            log, seconds = self.resource_build("resource-recreate")
            after = self.resource_snapshot(stage)
            self.assertEqual({name: value[0] for name, value in after.items()}, payloads)
            self.assertEqual({name: after[name] for name in before if name != "profiles/film.json"},
                             {name: value for name, value in before.items() if name != "profiles/film.json"})
            self.assertEqual(log.count("Copying runtime test resource"), 1)
            observations["missing-output"] = {"payload_files": 1, "payload_bytes": len(original), "seconds": seconds}
        print("Resource staging observations (logical payload changes):", json.dumps(observations, sort_keys=True))

    def test_resource_staging_missing_input_preserves_existing_payloads(self):
        _, inputs, _, stage, _ = self.resource_fixture()
        self.resource_build("resource-prepare-failure")
        before = self.resource_snapshot(stage)
        (inputs / "profiles/film.json").unlink()
        with self.assertRaises(subprocess.CalledProcessError):
            self.resource_build("resource-missing-input")
        self.assertEqual(self.resource_snapshot(stage), before)


RegistrationTests.test_resource_staging_updates_only_selected_members.required_subtests = True
RegistrationTests.test_resource_staging_updates_only_selected_members.subtest_cases = (
    "initial", "noop", "changed", "removed", "added", "configuration", "mutation", "missing-output"
)


class WorkflowResultTests(unittest.TestCase):
    """Execute the actual required-status Python guards against child results."""

    PLATFORMS = ("linux", "windows")
    ROLES = ("product-debug", "product-release", "tooling")
    CASES = ("layout", "success", "unexpected", "malformed") + tuple(
        role + ":" + outcome for role in ROLES
        for outcome in ("failure", "cancelled", "skipped", "missing")
    )

    def guard(self, platform):
        path = ROOT / f".github/workflows/tests-{platform}-host.yml"
        source = path.read_text(encoding="utf-8")
        # Only extract this source-owned Python step; actionlint validates YAML.
        job = re.search(rf"(?ms)^  {platform}-debug-host:\n(.*?)(?=^  [\w-]+:|\Z)", source)
        self.assertIsNotNone(job, "the established required host status must remain")
        body = job.group(1)
        needs = re.search(r"(?m)^    needs:\n((?:      - [\w-]+\n)+)", body)
        self.assertIsNotNone(needs)
        script = re.search(r"(?m)^        run: \|\n((?:          [^\n]*\n|\n)+)", body)
        self.assertIsNotNone(script)
        self.assertRegex(body, r"(?m)^    if: always\(\)$")
        self.assertIn("${{ toJSON(needs) }}", body)
        return path, {line.strip()[2:] for line in needs.group(1).splitlines()}, textwrap.dedent(script.group(1))

    def test_required_host_status_propagates_every_child_outcome(self):
        self.subtest_cases = self.test_required_host_status_propagates_every_child_outcome.subtest_cases
        for platform in self.PLATFORMS:
            path, needs, script = self.guard(platform)
            expected = {f"{platform}-{role}" for role in self.ROLES}
            code = compile(script, str(path), "exec")
            for case in self.CASES:
                with self.subTest(case=platform + ":" + case):
                    if case == "layout":
                        self.assertEqual(needs, expected)
                        continue
                    results = {name: {"result": "success"} for name in expected}
                    if case == "unexpected":
                        results[f"{platform}-analysis"] = {"result": "success"}
                    elif ":" in case:
                        role, outcome = case.split(":")
                        if outcome == "missing":
                            del results[f"{platform}-{role}"]
                        else:
                            results[f"{platform}-{role}"]["result"] = outcome
                    value = "{" if case == "malformed" else json.dumps(results)
                    status = 0
                    with patch.dict(os.environ, {"REQUIRED_RESULTS": value}), \
                            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        try:
                            exec(code, {"__name__": "__main__"})
                        except SystemExit as error:
                            status = error.code
                    self.assertEqual(status, 0 if case == "success" else 1)


WorkflowResultTests.test_required_host_status_propagates_every_child_outcome.required_subtests = True
WorkflowResultTests.test_required_host_status_propagates_every_child_outcome.subtest_cases = tuple(
    platform + ":" + case for platform in WorkflowResultTests.PLATFORMS for case in WorkflowResultTests.CASES
)


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


@unittest.skipUnless(os.environ.get("JUICER_CUDA_FREE_BUILD"),
                     "requires the configured preset CUDA-free host test tree")
class CudaFreeBuildRegistrationTests(unittest.TestCase):
    """Actual selected closures and owner guards for the shared preset Cargo tree."""

    CUDA_FREE_TARGETS = ("JuicerHashContractTests", "JuicerRenderAssertionTests")
    PRODUCT_RUST_TARGETS = (
        "JuicerCudaAbiTests", "JuicerProfileOwnerTests", "JuicerSpectralOwnerTests",
        "JuicerRouteBridgeTests",
    )
    PRODUCT_RUST_SOURCES = (
        "cuda_abi_c.c", "cuda_abi_test.cpp", "profile_abi_c.c", "profile_owner_test.cpp",
        "spectral_abi_c.c", "spectral_owner_test.cpp", "route_bridge_test.cpp", "ScanRoute.cpp",
    )
    PRODUCT_RUST_VARIANTS = {
        "JuicerCudaAbiTests": ("cuda_abi_c.c", "test-support"),
        "JuicerProfileOwnerTests": ("profile_abi_c.c", "test-support"),
        "JuicerSpectralOwnerTests": ("spectral_abi_c.c", "test-support"),
        "JuicerRouteBridgeTests": ("route_bridge_test.cpp", "production"),
    }
    GPU_ONLY_TARGETS = (
        "JuicerPinnedUploadTests", "JuicerGaussianKernelTests", "JuicerScannerLutResourceTests",
        "JuicerDeferredDestroyTests", "JuicerCudaInspectGpuTests", "JuicerProfileCudaSafetyTests",
        "JuicerScatterHalationGpuTests", "JuicerGrainDeltaTests", "JuicerScannerOutputTests",
    )
    CARGO_FILES = (
        "production/features.txt", "production/build.stamp", "production/CACHEDIR.TAG",
        "test-support/features.txt", "test-support/build.stamp", "test-support/CACHEDIR.TAG",
        "test-support/metadata/binaries.json", "test-support/metadata/cargo.json",
        "test-support/metadata/preparation-timings.json",
    )

    def setUp(self):
        host_build = Path(os.environ["JUICER_CUDA_FREE_BUILD"]).resolve()
        self.preset = host_build.parent.name
        self.assertIn(self.preset, qualification.PRESETS)
        self.preset_build = (ROOT / "out/build" / self.preset).resolve()
        self.host_build = self.preset_build / "host-tests"
        self.assertEqual(host_build, self.host_build.resolve())
        self.assertEqual(host_build.parent, self.preset_build)
        self.lease = execution.OutputLease(ROOT, self.preset_build, timeout=0)
        self.lease.__enter__()
        self.addCleanup(self.lease.__exit__, None, None, None)
        self.assertIn(self.lease.role, ("delegated", "nested"),
                      "the verifier must already hold this preset's canonical owner")
        artifact_root = Path(os.environ.get(
            "JUICER_TEST_ARTIFACT_DIR", ROOT / "out/validation" / self.preset
        )).resolve() / "cuda-free-build-controls"
        artifact_root.mkdir(parents=True, exist_ok=True)
        self.scratch = Path(tempfile.mkdtemp(prefix="selected closures ", dir=artifact_root))
        self.addCleanup(self.finish_scratch)
        self.logs = self.scratch / "commands"
        self.logs.mkdir()
        self.native_run = execution.NativeRun(ROOT, self.logs)
        self.cmake = shutil.which("cmake")
        self.python = Path(sys.executable)
        self.verify = ROOT / "scripts/verify.py"
        self.assertTrue(self.cmake, "CMake is required for selected build evidence")
        self.assertTrue(self.verify.is_file())

    def finish_scratch(self):
        outcome = record_fixture(self.scratch, self)
        owners = list((self.scratch / "out/verification-locks").glob("*/owner.json"))
        if outcome == "passed" and not owners and not os.environ.get("JUICER_TEST_RETAIN_FIXTURES"):
            try:
                shutil.rmtree(self.scratch)
            except BaseException:
                record_fixture(self.scratch, self, "failed")
                print("CUDA-free build evidence cleanup failed; retained:", self.scratch, file=sys.stderr)
                raise
        else:
            print("Retained CUDA-free build evidence:", self.scratch, file=sys.stderr)

    def assert_inventory_timeouts(self, build, expected_names, expected_labels):
        ctest = shutil.which("ctest")
        self.assertTrue(ctest, "CTest is required for registration discovery")
        self.native_run.command("inventory-discovery", [
            ctest, "--test-dir", str(build), "--show-only=json-v1"
        ], stdout_only=True)
        inventory = json.loads((self.logs / "inventory-discovery.log").read_text(encoding="utf-8"))
        tests = inventory["tests"]
        self.assertTrue(tests, "configured inventory must not be empty")
        by_name = {test["name"]: test for test in tests}
        self.assertEqual(len(by_name), len(tests))
        self.assertTrue(set(expected_names).issubset(by_name), expected_names)
        observed_labels = set()
        for test in tests:
            properties = {row["name"]: row["value"] for row in test.get("properties", [])}
            labels = set(properties.get("LABELS", []))
            observed_labels.update(labels)
            self.assertIn(type(properties.get("TIMEOUT")), (int, float), test["name"])
            self.assertEqual(properties["TIMEOUT"], 0, test["name"])
            if "gpu" in labels:
                self.assertIn("juicer_gpu", properties.get("RESOURCE_LOCK", []), test["name"])
        self.assertTrue(set(expected_labels).issubset(observed_labels), observed_labels)

    @unittest.skipIf(os.environ.get("JUICER_PRODUCT_INVENTORY_BUILD"),
                     "CUDA-free inventory is checked in its own configured consumer")
    def test_cuda_free_inventory_has_zero_timeouts(self):
        self.assert_inventory_timeouts(
            self.host_build, ("Quality.BuildRegistration", "Ofx.Host.RenderAssertions"), ("host",)
        )

    @unittest.skipUnless(os.environ.get("JUICER_PRODUCT_INVENTORY_BUILD"),
                         "requires the configured product consumer")
    def test_product_inventory_has_zero_timeouts(self):
        self.assertEqual(Path(os.environ["JUICER_PRODUCT_INVENTORY_BUILD"]).resolve(), self.preset_build)
        self.assert_inventory_timeouts(
            self.preset_build,
            ("Quality.BuildRegistration", "Ofx.Host.RenderAssertions", "Resource.Gpu.ScannerLut",
             "Ffi.Gpu.CudaInspect.MultiDevice", "Ffi.Gpu.ProductionPreparedRender",
             "ScatterHalation.Gpu.HostContracts"),
            ("host", "gpu"),
        )

    def cargo_snapshot(self):
        cargo = self.preset_build / "cargo"
        profile = "release" if self.preset.endswith("release") else "debug"
        if os.name == "nt":
            target, archive = "x86_64-pc-windows-msvc", "film_juicer_plugin.lib"
        else:
            target, archive = "x86_64-unknown-linux-gnu", "libfilm_juicer_plugin.a"
        files = set(self.CARGO_FILES)
        for variant in ("production", "test-support"):
            files.add(f"{variant}/{target}/{profile}/{archive}")
        if profile == "release":
            files = {name.replace("/debug/", "/release/") for name in files}
        snapshot = {}
        for relative in files:
            path = cargo / relative
            snapshot[relative] = ((path.stat().st_size, path.stat().st_mtime_ns)
                                  if path.is_file() else None)
        return snapshot

    def query_commands(self, name, build, *targets):
        cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
        configured = re.search(r"^CMAKE_MAKE_PROGRAM:[^=]+=(.+)$", cache, re.MULTILINE)
        self.assertIsNotNone(configured, "the configuration must select Ninja")
        ninja = configured.group(1)
        self.assertTrue(Path(ninja).is_file())
        return self.native_run.command(
            name, [ninja, "-C", str(build), "-t", "commands", *targets],
            timeout=60, cwd=ROOT,
        ).read_text(encoding="utf-8", errors="replace").replace("\\", "/")

    def query_targets(self, name, build):
        cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
        configured = re.search(r"^CMAKE_MAKE_PROGRAM:[^=]+=(.+)$", cache, re.MULTILINE)
        self.assertIsNotNone(configured, "the configuration must select Ninja")
        ninja = configured.group(1)
        self.assertTrue(Path(ninja).is_file())
        return self.native_run.command(
            name, [ninja, "-C", str(build), "-t", "targets", "all"],
            timeout=60, cwd=ROOT,
        ).read_text(encoding="utf-8", errors="replace").replace("\\", "/")

    def assert_no_test_metadata(self, commands):
        lower = commands.casefold().replace("\\", "/")
        for forbidden in ("nextest list", "--workspace --tests", "metadata/binaries.json",
                          "metadata/cargo.json", "preparation-timings.json"):
            self.assertNotIn(forbidden, lower)


    def test_check_owner_borrows_only_the_inherited_canonical_preset_token(self):
        before = self.cargo_snapshot()
        environment = dict(os.environ)
        inherited = json.loads(environment.get("JUICER_OUTPUT_LEASES", "{}"))
        lease = execution.OutputLease(ROOT, self.preset_build)
        key = hashlib.sha256(os.path.normcase(str(lease.path)).encode()).hexdigest()
        self.assertIn(key, inherited, "this test must run under the canonical preset owner")

        self.native_run.command("owner-delegated", [self.python, "-B", self.verify,
                                "check-owner", "--preset", self.preset],
                                env=environment, timeout=10)

        missing = dict(environment)
        missing.pop("JUICER_OUTPUT_LEASES", None)
        with self.assertRaises(subprocess.CalledProcessError):
            self.native_run.command("owner-missing-token", [self.python, "-B", self.verify,
                                         "check-owner", "--preset", self.preset],
                                         env=missing, timeout=10)

        wrong = dict(environment)
        wrong_tokens = dict(inherited)
        wrong_tokens[key] = "wrong-owner-token"
        wrong["JUICER_OUTPUT_LEASES"] = json.dumps(wrong_tokens, sort_keys=True)
        with self.assertRaises(subprocess.CalledProcessError):
            self.native_run.command("owner-wrong-token", [self.python, "-B", self.verify,
                                       "check-owner", "--preset", self.preset],
                                       env=wrong, timeout=10)
        self.assertEqual(self.cargo_snapshot(), before)

    def test_unowned_rust_test_target_fails_before_touching_shared_cargo(self):
        before = self.cargo_snapshot()
        environment = dict(os.environ)
        environment.pop("JUICER_OUTPUT_LEASES", None)
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            self.native_run.command("unowned-rust-tests", [self.cmake, "--build", str(self.host_build),
                                        "--target", "juicer_rust_tests"],
                                        env=environment, timeout=30, cwd=ROOT)
        self.assertNotEqual(failure.exception.returncode, 0)
        self.assertEqual(self.cargo_snapshot(), before)
        log = (self.logs / "unowned-rust-tests.log").read_text(encoding="utf-8", errors="replace")
        self.assertRegex(log.casefold(), "owner|lease")

    def test_unowned_configuration_rejects_before_touching_shared_cargo(self):
        before = self.cargo_snapshot()
        environment = dict(os.environ)
        environment.pop("JUICER_OUTPUT_LEASES", None)
        try:
            with self.assertRaises(subprocess.CalledProcessError):
                self.native_run.command("unowned-configuration",
                    [self.cmake, "-S", str(ROOT / "tests/host"), "-B", str(self.host_build)],
                    env=environment, timeout=30)
            self.assertEqual(self.cargo_snapshot(), before)
        finally:
            self.native_run.command("restore-configuration",
                [self.cmake, "--build", str(self.host_build), "--target", "help"], timeout=60)

    def test_shared_resource_stage_requires_owner_and_preserves_a_noop(self):
        command = [self.cmake, "--build", str(self.host_build), "--target", "juicer_test_resources"]
        self.native_run.command("resource-stage-prepare", command, timeout=120)
        stage = self.preset_build / "tests/Resources"
        stamp = self.preset_build / "tests/resources.stamp"
        members = (self.host_build / "tests/resource-members.txt").read_text(encoding="utf-8").splitlines()
        self.assertCountEqual([path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()],
                              members)

        def identities():
            return {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                    for path in (stamp, *stage.rglob("*")) if path.is_file()}

        before = identities()
        environment = dict(os.environ)
        environment.pop("JUICER_OUTPUT_LEASES", None)
        # Force both shared producers to need work. A warm rejected target alone
        # would not prove that membership pruning and payload copies are guarded.
        missing = stage / "illuminants/D50.csv"
        missing.unlink()
        stamp.unlink()
        try:
            with self.assertRaises(subprocess.CalledProcessError):
                self.native_run.command("resource-stage-unowned", command, env=environment, timeout=30)
            self.assertFalse(missing.exists())
            self.assertFalse(stamp.exists())
            self.assertEqual(identities(), {name: value for name, value in before.items()
                                           if name not in {str(missing), str(stamp)}})
        finally:
            self.native_run.command("resource-stage-restore", command, timeout=60)
        self.assertEqual(missing.read_bytes(), (ROOT / "Resources/illuminants/D50.csv").read_bytes())
        before = identities()
        self.native_run.command("resource-stage-noop", command, timeout=60)
        self.assertEqual(identities(), before, "owner checks must not force resource payload writes")
        self.assertNotIn("Copying runtime test resource", (self.logs / "resource-stage-noop.log").read_text())
        commands = self.query_commands("resource-stage-commands", self.host_build, "juicer_test_resources")
        self.assertIn("check-owner", commands)
        print("Shared resource stage no-op: zero payload commands and writes; files=", len(before) - 1)

    def test_host_configuration_and_selected_two_target_closure_are_cuda_free(self):
        cache_path = self.host_build / "CMakeCache.txt"
        self.assertTrue(cache_path.is_file())
        cache = cache_path.read_text(encoding="utf-8", errors="replace")
        cache_rows = {match.group(1): match.group(2) for match in
                      re.finditer(r"^([^#/:]+):[^=]*=(.*)$", cache, re.MULTILINE)}
        self.assertEqual(cache_rows.get("BUILD_TESTING"), "ON")
        self.assertEqual(cache_rows.get("CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit"), "TRUE")
        self.assertIn(cache_rows.get("CMAKE_CUDA_COMPILER"), (None, "", "CMAKE_CUDA_COMPILER-NOTFOUND"))
        self.assertFalse(list((self.host_build / "CMakeFiles").rglob("CMakeCUDACompiler.cmake")))

        compile_db = self.host_build / "compile_commands.json"
        self.assertTrue(compile_db.is_file())
        compile_rows = json.loads(compile_db.read_bytes())
        self.assertTrue(compile_rows)
        self.assertTrue(all(not Path(row["file"]).suffix.casefold() == ".cu" for row in compile_rows))

        selected = self.query_commands("host-selected-commands", self.host_build, *self.CUDA_FREE_TARGETS)
        for source in ("hash_contract_test.cpp", "render_assertions_test.cpp"):
            self.assertIn(source, selected)
        for source in self.PRODUCT_RUST_SOURCES:
            self.assertNotIn(source, selected)
        self.assertNotIn("/out/build/" + self.preset + "/cargo/test-support", selected)
        self.assertNotIn("/out/build/" + self.preset + "/cargo/production", selected)
        self.assertNotIn("/host-tests/cargo/", selected)
        self.assertNotIn("nvcc", selected.casefold())
        self.assertNotIn("cufft", selected.casefold())
        self.assertNotIn("/src/Cuda/", selected)
        self.assertNotIn("ResourceStaging.cmake", selected,
                         "immutable explicit-root consumers must not prepare the staged package tree")
        self.assert_no_test_metadata(selected)

        targets = self.query_targets("host-target-inventory", self.host_build)
        for target in self.PRODUCT_RUST_TARGETS:
            self.assertNotRegex(targets, rf"(?m)^{re.escape(target)}(?:$|:)")

        all_commands = self.query_commands("host-all-commands", self.host_build)
        self.assertEqual(all_commands.strip(), "", "the CUDA-free default target must not compile tests")
        for source in (*self.PRODUCT_RUST_SOURCES, "hash_contract_test.cpp", "render_assertions_test.cpp"):
            self.assertNotIn(source, all_commands)
        self.assert_no_test_metadata(all_commands)

    @unittest.skipUnless(os.environ.get("JUICER_PRODUCT_BUILD"),
                         "requires an explicitly configured testing-ON product graph")
    def test_product_analysis_inputs_have_their_ordinary_producers(self):
        commands = self.query_commands("product-analysis-commands", self.preset_build, "juicer_analysis_inputs")
        for source in ("JuicerCudaResources.cpp", "JuicerState.cpp", "JuicerEffect.cpp", "JuicerCudaFilmPipeline.cu"):
            self.assertIn(source, commands)
        self.assertIn("/cargo/production", commands)
        self.assertNotIn("/cargo/test-support", commands)
        self.assertIn("generate_host_cases.py", commands)
        self.assertNotIn("profile_owner_test.cpp", commands)
        self.assert_no_test_metadata(commands)

    @unittest.skipUnless(os.environ.get("JUICER_PRODUCT_BUILD"),
                         "requires an explicitly configured testing-ON product graph")
    def test_product_failure_consumer_and_default_all_have_separate_closures(self):
        self.assertEqual(Path(os.environ["JUICER_PRODUCT_BUILD"]).resolve(), self.preset_build)
        cache = (self.preset_build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
        self.assertIn("BUILD_TESTING:BOOL=ON", cache)
        selected = self.query_commands("product-failure-commands", self.preset_build,
                                       "JuicerCudaFailureTests")
        self.assertIn("cuda_failure_test.cpp", selected)
        self.assertIn("JuicerCudaExecutor.cpp", selected)
        self.assertIn("JuicerCudaFilmPipeline.cu", selected)
        self.assertIn("cufft", selected.casefold())
        for target in (*self.GPU_ONLY_TARGETS, *self.PRODUCT_RUST_TARGETS):
            self.assertNotIn(target, selected)
        # ScanRoute.cpp is also an ordinary production core source.
        test_sources = tuple(source for source in self.PRODUCT_RUST_SOURCES if source != "ScanRoute.cpp")
        for source in test_sources:
            self.assertNotIn(source, selected)

        for target, (source, variant) in self.PRODUCT_RUST_VARIANTS.items():
            commands = self.query_commands("product-rust-consumer-" + target.lower(), self.preset_build, target)
            self.assertIn(source, commands)
            self.assertIn("/cargo/" + variant + "/", commands)
            self.assert_no_test_metadata(commands)

        all_commands = self.query_commands("product-all-commands", self.preset_build)
        for source in ("cuda_failure_test.cpp", "film_recipe_test.cpp", "pinned_upload_test.cpp",
                       *test_sources, "hash_contract_test.cpp", "render_assertions_test.cpp"):
            self.assertNotIn(source, all_commands)
        lower = all_commands.casefold().replace("\\", "/")
        self.assertIn("/cargo/production", lower)
        self.assertNotIn("/cargo/test-support", lower)
        self.assert_no_test_metadata(all_commands)


if __name__ == "__main__":
    unittest.main()
