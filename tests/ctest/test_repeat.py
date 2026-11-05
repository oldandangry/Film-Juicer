from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/ctest"))
from control_artifacts import MARKER, current_fixture_blockers, record_fixture, selected_paths
import execution

ROOT = Path(__file__).resolve().parents[2]


class CtestRepetitionTests(unittest.TestCase):
    def run(self, result=None):
        try:
            return super().run(result)
        except KeyboardInterrupt:
            self.interrupted = True
            if hasattr(self, "lease") and self.lease.path in execution.OutputLease.held:
                self.lease.blocked = True
            while self._cleanups:
                try:
                    self.doCleanups()
                except BaseException:
                    import traceback
                    traceback.print_exc(file=sys.stderr)
            raise

    def setUp(self):
        self.preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        artifact_root = os.environ.get("JUICER_TEST_ARTIFACT_DIR")
        parent = (Path(artifact_root) if artifact_root else
                  ROOT / "out/validation" / self.preset / "ctest-repetition-controls")
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="repeat control ü ", dir=parent, delete=False)
        self.directory = Path(self.temporary.name).resolve()
        self.build = self.directory / "out/build" / self.preset
        self.build.mkdir(parents=True)
        self.result_directory = self.directory / "out/validation" / self.preset / "results/repetition"
        self.result_directory.mkdir(parents=True)
        self.retain = bool(os.environ.get("JUICER_TEST_RETAIN_FIXTURES"))
        self.addCleanup(self.finish_fixture)

        self.lease = execution.OutputLease(self.directory, self.build)
        self.lease.__enter__()
        self.addCleanup(self.lease.__exit__, None, None, None)
        self.native = execution.NativeRun(self.directory, self.result_directory)
        self.ctest = os.environ.get("JUICER_CTEST_EXECUTABLE") or shutil.which("ctest")
        self.assertTrue(self.ctest, "CTest is required for the repetition contract")
        self.events = self.directory / "events.jsonl"
        self.registration_file = self.build / "CTestTestfile.cmake"
        self.registration_file.write_text(self.registration(), encoding="utf-8")

    def registration(self):
        script = self.directory / "control.py"
        script.write_text(
            "import json, os, sys, time\n"
            "from pathlib import Path\n"
            "local_count = 0\n"
            "local_count += 1\n"
            "events = Path(sys.argv[2])\n"
            "attempt = len(events.read_text().splitlines()) + 1 if events.exists() else 1\n"
            "row = {'attempt': attempt, 'pid': os.getpid(), 'started': time.monotonic_ns(),\n"
            "       'local_count': local_count, 'operands': sys.argv[3:]}\n"
            "with events.open('a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps(row) + '\\n')\n"
            "print('CONTROL_ATTEMPT', attempt, flush=True)\n"
            "if sys.argv[1] == 'fail-second' and attempt == 2:\n"
            "    print('INTENTIONAL_SECOND_ATTEMPT_FAILURE', flush=True)\n"
            "    raise SystemExit(7)\n",
            encoding="utf-8",
        )
        arguments = [sys.executable, "-B", str(script), "@MODE@", str(self.events), "fixed operands", "W1-W7"]
        # CTest's bracket arguments preserve spaces in the Python, script and operand paths.
        value = "add_test(RepeatControl " + " ".join(f"[=[{argument}]=]" for argument in arguments) + ")\n"
        return value + "set_tests_properties(RepeatControl PROPERTIES TIMEOUT 5)\n"

    def finish_fixture(self):
        outcome = record_fixture(self.directory, self, owner_root=self.directory)
        owners = list((self.directory / "out/verification-locks").glob("*/owner.json"))
        if outcome != "passed" or self.retain or owners:
            print(f"Retained CTest repetition fixture: {self.directory} test={self.id()}", file=sys.stderr, flush=True)
            return
        try:
            self.temporary.cleanup()
        except BaseException:
            record_fixture(self.directory, self, "failed", owner_root=self.directory)
            print(f"CTest repetition cleanup failed; remaining fixture: {self.directory}", file=sys.stderr, flush=True)
            raise

    def run_control(self, mode="success", selection="^RepeatControl$"):
        self.registration_file.write_text(self.registration().replace("@MODE@", mode), encoding="utf-8")
        command = [self.ctest, "--test-dir", str(self.build), "--parallel", "1", "--no-tests=error",
                   "-R", selection, "--repeat", "until-fail:5", "--verbose", "--output-on-failure",
                   "--stop-on-failure", "--output-log", str(self.directory / "ctest.log"),
                   "--output-junit", str(self.directory / "junit.xml")]
        try:
            self.native.command("ctest", command, timeout=20, stdout_only=True)
        except subprocess.CalledProcessError:
            # The CTest process record and output remain in the owned run even when
            # its child test is an expected negative control.
            pass
        process = json.loads((self.result_directory / "ctest.process.json").read_text(encoding="utf-8"))
        completed = self.native.commands[-1]
        self.native.publish({"schema": "native-run-v1", "status": "passed" if completed["returncode"] == 0 else "failed",
                             "commands": self.native.commands})
        stdout = (self.result_directory / "ctest.log").read_bytes()
        stderr_path = self.result_directory / "ctest.stderr.log"
        stderr = stderr_path.read_bytes() if stderr_path.exists() else b""
        return subprocess.CompletedProcess(command, completed["returncode"], stdout, stderr), process

    def rows(self):
        return [json.loads(line) for line in self.events.read_text(encoding="utf-8").splitlines()]

    def test_success_repeats_in_fresh_processes_with_exact_operands(self):
        result, process = self.run_control()
        expected_attempts = (1, 2, 3, 4, 5)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(process["drained"])
        rows = self.rows()
        self.assertEqual(tuple(row["attempt"] for row in rows), expected_attempts)
        self.assertEqual(len({(row["pid"], row["started"]) for row in rows}), 5)
        self.assertTrue(all(row["local_count"] == 1 and row["operands"] == ["fixed operands", "W1-W7"]
                            for row in rows))
        self.assertEqual(len(list(ET.parse(self.directory / "junit.xml").iter("testcase"))), 1)

    def test_first_failure_stops_repetition_and_keeps_failed_attempt(self):
        result, process = self.run_control("fail-second")
        expected_attempts = (1, 2)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(process["drained"])
        self.assertEqual(tuple(row["attempt"] for row in self.rows()), expected_attempts)
        self.assertIn(b"INTENTIONAL_SECOND_ATTEMPT_FAILURE", result.stdout)
        self.assertNotIn(b"CONTROL_ATTEMPT 3", result.stdout)
        self.assertIsNotNone(ET.parse(self.directory / "junit.xml").find(".//failure"))
        self.assertIn("INTENTIONAL_SECOND_ATTEMPT_FAILURE", (self.directory / "ctest.log").read_text())
        command = json.loads((self.result_directory / "result.json").read_text())
        self.assertEqual(command["status"], "failed")
        self.assertNotEqual(command["commands"][0]["returncode"], 0)

    def test_empty_selection_is_a_nonzero_ctest_result(self):
        result, process = self.run_control("success", "^MissingControl$")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(process["drained"])
        self.assertFalse(self.events.exists())
        self.assertIn(b"No tests were found", result.stderr)


class FixtureLifetimeTests(unittest.TestCase):
    def test_successful_repetition_cleans_owned_scratch(self):
        case = CtestRepetitionTests("test_success_repeats_in_fresh_processes_with_exact_operands")
        result = case.run(unittest.TestResult())
        self.assertTrue(result.wasSuccessful(), result.errors)
        self.assertFalse(case.directory.exists())
        self.assertNotIn(case.lease.path, execution.OutputLease.held)

    def test_outer_failure_retains_drained_ctest_failure_bundle(self):
        case = CtestRepetitionTests("test_first_failure_stops_repetition_and_keeps_failed_attempt")

        def fail_after_control():
            result, process = case.run_control("fail-second")
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(process["drained"])
            self.fail("controlled outer assertion failure")

        case.test_first_failure_stops_repetition_and_keeps_failed_attempt = fail_after_control
        result = case.run(unittest.TestResult())
        self.assertEqual(len(result.failures), 1)
        self.assertTrue(case.directory.exists())
        marker = json.loads((case.directory / MARKER).read_text())
        self.assertEqual(marker["outcome"], "failed")
        self.assertTrue((case.result_directory / "result.json").is_file())
        self.assertTrue(json.loads((case.result_directory / "ctest.process.json").read_text())["drained"])
        # Selection is covered with isolated families below; this control owns
        # only its fixture's outcome and drainage, not the surrounding preset.
        case.temporary.cleanup()

    def test_interruption_retains_blocked_owner_after_process_drainage(self):
        case = CtestRepetitionTests("test_success_repeats_in_fresh_processes_with_exact_operands")
        original_wait = subprocess.Popen.wait
        interrupted = False

        def interrupt_after_test_starts(process, *args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                deadline = time.monotonic() + 10
                while not case.events.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(case.events.exists())
                interrupted = True
                raise KeyboardInterrupt()
            return original_wait(process, *args, **kwargs)

        with patch.object(subprocess.Popen, "wait", interrupt_after_test_starts), self.assertRaises(KeyboardInterrupt):
            case.run(unittest.TestResult())
        self.assertTrue(case.directory.exists())
        marker = json.loads((case.directory / MARKER).read_text())
        self.assertEqual(marker["outcome"], "interrupted")
        owner = json.loads((case.lease.path / "owner.json").read_text())
        self.assertEqual(owner["status"], "blocked")
        self.assertNotIn(case.lease.path, execution.OutputLease.held)
        process_path = case.result_directory / "ctest.process.json"
        if process_path.exists():
            self.assertTrue(json.loads(process_path.read_text())["drained"])
        case.temporary.cleanup()


class ArtifactSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="artifact selector ü ")
        self.addCleanup(self.temporary.cleanup)
        self.preset_root = Path(self.temporary.name).resolve() / "out/validation/linux-debug"
        self.family = self.preset_root / "ctest-repetition-controls"
        self.family.mkdir(parents=True)
        self.case = unittest.FunctionTestCase(lambda: None)

    def test_selector_omits_absent_empty_and_successful_families(self):
        self.subtest_cases = ("absent", "empty", "passed")
        self.family.rmdir()
        for case_id in self.subtest_cases:
            with self.subTest(case=case_id):
                if case_id == "empty":
                    self.family.mkdir()
                elif case_id == "passed":
                    record_fixture(self.family / "completed", self.case, "passed")
                self.assertEqual(selected_paths(self.preset_root), [])

    def test_selector_includes_every_unproved_fixture_and_required_sample(self):
        failed = self.family / "failed"
        interrupted = self.family / "interrupted"
        missing = self.family / "missing-marker"
        malformed = self.family / "malformed-marker"
        passed = self.family / "passed"
        passed_blocked = self.family / "passed-with-blocked-owner"
        for path, outcome in ((failed, "failed"), (interrupted, "interrupted"), (passed, "passed")):
            record_fixture(path, self.case, outcome)
        owner = passed_blocked / "out/verification-locks/control/owner.json"
        owner.parent.mkdir(parents=True)
        owner.write_text('{"status":"blocked"}')
        record_fixture(passed_blocked, self.case, "passed")
        missing.mkdir()
        (malformed / MARKER).parent.mkdir(parents=True)
        (malformed / MARKER).write_text("[]")
        sample = self.preset_root / "rust-core/npy-0081-0000.npy"
        sample.parent.mkdir()
        sample.write_bytes(b"sample")
        unrelated = Path(self.temporary.name) / "out/build/cache/failed.json"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text('{"status":"failed"}')

        expected_rows = (
            ("failed-marker", failed, True), ("interrupted-marker", interrupted, True),
            ("missing-marker", missing, True), ("malformed-marker", malformed, True),
            ("passed-marker", passed, False), ("passed-blocked-owner", passed_blocked, True),
            ("npy-sample", sample, True), ("build-cache", unrelated, False),
        )
        self.subtest_cases = tuple(row[0] for row in expected_rows)
        selected = set(selected_paths(self.preset_root))
        for case_id, path, included in expected_rows:
            with self.subTest(case=case_id):
                self.assertEqual(path in selected, included)

    def test_selector_fails_closed_on_ambiguous_or_unbound_marker_metadata(self):
        cases = (
            "duplicate-outcome", "fixture-mismatch", "owner-root-missing",
            "owner-root-path-missing", "owner-root-outside", "descendant-owner-with-live-root",
            "blocked-not-a-list", "unknown-field",
            "passed-with-failure", "passed-with-error", "cleanup-exception",
        )
        self.subtest_cases = cases
        for case_id in cases:
            with self.subTest(case=case_id):
                fixture = self.family / ("untrusted-" + case_id)
                fixture.mkdir()
                record_fixture(fixture, self.case, "passed")
                marker = fixture / MARKER
                record = json.loads(marker.read_text(encoding="utf-8"))
                if case_id == "duplicate-outcome":
                    content = marker.read_text(encoding="utf-8").replace(
                        '"outcome": "passed"', '"outcome": "failed", "outcome": "passed"', 1)
                    marker.write_text(content, encoding="utf-8")
                else:
                    if case_id == "fixture-mismatch":
                        record["fixture"] = str(self.family)
                    elif case_id == "owner-root-missing":
                        del record["owner_root"]
                    elif case_id == "owner-root-path-missing":
                        record["owner_root"] = str(fixture / "missing-owner")
                    elif case_id == "owner-root-outside":
                        outside = self.preset_root / "outside-owner"
                        outside.mkdir()
                        record["owner_root"] = str(outside)
                    elif case_id == "descendant-owner-with-live-root":
                        child = fixture / "unused-child"
                        child.mkdir()
                        record["owner_root"] = str(child)
                        (fixture / "out/verification-locks/live-root-owner").mkdir(parents=True)
                    elif case_id == "blocked-not-a-list":
                        record["blocked_owners"] = ""
                    elif case_id == "unknown-field":
                        record["legacy_status"] = "passed"
                    elif case_id == "passed-with-failure":
                        record["failures"] = ["failure trace"]
                    elif case_id == "passed-with-error":
                        record["errors"] = ["error trace"]
                    elif case_id == "cleanup-exception":
                        record["cleanup_exception"] = "OSError: cleanup did not finish"
                    marker.write_text(json.dumps(record), encoding="utf-8")
                self.assertIn(fixture, selected_paths(self.preset_root))

    def test_selector_rescans_live_owner_and_named_command_locks(self):
        cases = ("owner-record-after-marker", "empty-lease-directory", "named-command-lock", "dependency-command-lock")
        self.subtest_cases = cases
        for case_id in cases:
            with self.subTest(case=case_id):
                fixture = self.family / ("live-" + case_id)
                fixture.mkdir()
                record_fixture(fixture, self.case, "passed", owner_root=fixture)
                if case_id == "owner-record-after-marker":
                    owner = fixture / "out/verification-locks/late-owner/owner.json"
                    owner.parent.mkdir(parents=True)
                    owner.write_text('{"status":"owned"}', encoding="utf-8")
                elif case_id == "empty-lease-directory":
                    (fixture / "out/verification-locks/acquiring-owner").mkdir(parents=True)
                else:
                    running = (fixture / "logs/native.running.lock" if case_id == "dependency-command-lock" else
                               fixture / "out/validation/linux-debug/results/campaign/native.running.lock")
                    running.parent.mkdir(parents=True)
                    running.write_text("running\n", encoding="utf-8")
                self.assertIn(fixture, selected_paths(self.preset_root))

    @unittest.skipUnless(os.name == "posix", "native POSIX directory-link control")
    def test_selector_retains_linked_output_directories(self):
        locations = ("out", "out/validation/linux-debug", "out/validation/linux-debug/results/run-one")
        self.subtest_cases = locations
        target = Path(self.temporary.name) / "outside-fixture"
        target.mkdir()
        for location in locations:
            with self.subTest(case=location):
                fixture = self.family / ("linked-" + location.replace("/", "-"))
                record_fixture(fixture, self.case, "passed")
                link = fixture / location
                link.parent.mkdir(parents=True, exist_ok=True)
                destination = target / location.replace("/", "-")
                if location == "out":
                    link.rename(destination)
                else:
                    destination.mkdir()
                link.symlink_to(destination, target_is_directory=True)
                self.assertTrue(current_fixture_blockers(fixture))
                self.assertIn(fixture, selected_paths(self.preset_root))

    def test_native_workflows_upload_selected_paths_and_keep_reports(self):
        native_targets = ("linux", "windows")
        self.subtest_cases = tuple(f"{native}-workflow" for native in native_targets)
        for case_id, native in zip(self.subtest_cases, native_targets):
            with self.subTest(case=case_id):
                text = (ROOT / f".github/workflows/tests-{native}-host.yml").read_text()
                self.assertIn("tests/ctest/control_artifacts.py --preset", text)
                self.assertIn("path: ${{ steps.failed-controls.outputs.paths }}", text)
                self.assertIn("include-hidden-files: true", text)
                self.assertIn("if: always() && steps.failed-controls.outputs.paths != ''", text)
                self.assertEqual(text.count("Testing/Temporary/LastTest.log"), 3)
                artifacts = re.findall(r"^          name: (.+)$", text, re.M)
                self.assertEqual(len(artifacts), len(set(artifacts)))
                tooling = f"{native}-tooling"
                expanded = artifacts
                if native == "windows":
                    tooling += "-${{ matrix.shard }}"
                    job = re.search(r"^  windows-tooling:\n((?:    [^\n]*\n|\n)+)", text, re.M)
                    self.assertIsNotNone(job)
                    matrix = re.findall(r"^        shard: \[([^\]\n]+)\]$", job[1], re.M)
                    self.assertEqual(len(matrix), 1)
                    shards = [value.strip() for value in matrix[0].split(",")]
                    self.assertCountEqual(shards, ["core", "BuildEvidence", "Checkpoint"])
                    self.assertIn('$env:GITHUB_RUN_ATTEMPT".ToLowerInvariant()', job[1])
                    self.assertIn('"run_id=$runId" >> $env:GITHUB_OUTPUT', job[1])
                    self.assertIn('--run-id "$runId"', job[1])
                    self.assertIn("steps.tooling-run.outputs.run_id || 'unstarted'", job[1])
                    tooling_artifacts = re.findall(r"^          name: (.+)$", job[1], re.M)
                    self.assertTrue(tooling_artifacts)
                    self.assertTrue(all("${{ matrix.shard }}" in artifact for artifact in tooling_artifacts))
                    expanded = [artifact.replace("${{ matrix.shard }}", shard)
                                for artifact in artifacts
                                for shard in (shards if artifact in tooling_artifacts else [""])]
                    for shard in shards:
                        self.assertIn(f"windows-tooling-{shard}-results", expanded)
                        self.assertIn(f"windows-tooling-{shard}-failed-controls", expanded)
                self.assertEqual(len(expanded), len(set(expanded)), "artifact names must be unique across matrix jobs")
                self.assertIn(tooling + "-results", artifacts)
                self.assertIn(tooling + "-failed-controls", artifacts)
                for lane in ("product-debug", "product-release"):
                    self.assertIn(f"{native}-{lane}-failed-controls", artifacts)
                self.assertNotIn("out/validation/**", text)


ArtifactSelectionTests.test_selector_omits_absent_empty_and_successful_families.required_subtests = True
ArtifactSelectionTests.test_selector_includes_every_unproved_fixture_and_required_sample.required_subtests = True
ArtifactSelectionTests.test_selector_includes_every_unproved_fixture_and_required_sample.subtest_cases = (
    "failed-marker", "interrupted-marker", "missing-marker", "malformed-marker",
    "passed-marker", "passed-blocked-owner", "npy-sample", "build-cache",
)
ArtifactSelectionTests.test_selector_fails_closed_on_ambiguous_or_unbound_marker_metadata.required_subtests = True
ArtifactSelectionTests.test_selector_rescans_live_owner_and_named_command_locks.required_subtests = True
ArtifactSelectionTests.test_selector_retains_linked_output_directories.required_subtests = True
ArtifactSelectionTests.test_native_workflows_upload_selected_paths_and_keep_reports.required_subtests = True
ArtifactSelectionTests.test_native_workflows_upload_selected_paths_and_keep_reports.subtest_cases = (
    "linux-workflow", "windows-workflow"
)


if __name__ == "__main__":
    unittest.main()
