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
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/ctest"))
from control_artifacts import MARKER, record_fixture, selected_paths
import execution

ROOT = Path(__file__).resolve().parents[2]


class CtestRepetitionTests(unittest.TestCase):
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

    def setUp(self) -> None:
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        artifact_root = os.environ.get("JUICER_TEST_ARTIFACT_DIR")
        parent = Path(artifact_root) if artifact_root else ROOT / "out/validation" / preset / "ctest-repetition-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="repeat control ", dir=parent, delete=False)
        self.directory = Path(self.temporary.name).resolve()
        self.retain = bool(os.environ.get("JUICER_TEST_RETAIN_FIXTURES"))

        def finish_fixture():
            outcome = record_fixture(self.directory, self)
            failed = outcome != "passed"
            owners = list(self.directory.glob("out/verification-locks/*/owner.json"))
            if failed or self.retain or owners:
                print(f"Retained repetition fixture: {self.directory} test={self.id()}", file=sys.stderr, flush=True)
            else:
                try:
                    self.temporary.cleanup()
                except BaseException:
                    record_fixture(self.directory, self, "failed")
                    print(f"Repetition cleanup failed; remaining fixture: {self.directory}", file=sys.stderr, flush=True)
                    raise

        self.addCleanup(finish_fixture)
        record_fixture(self.directory, self, "running")
        if self.retain:
            print("Repetition fixture:", self.directory, file=sys.stderr, flush=True)
        self.campaign = execution.Campaign(self.directory, preset, "repetition", ["control"])
        self.lease = execution.OutputLease(self.directory, self.directory)
        self.lease.__enter__()
        self.addCleanup(self.lease.__exit__, None, None, None)
        self.ctest = os.environ.get("JUICER_CTEST_EXECUTABLE") or shutil.which("ctest")
        self.assertTrue(self.ctest, "CTest is required for the repetition contract")
        self.events = self.directory / "events.jsonl"
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
        # Bracket arguments retain executable/script paths containing spaces.
        arguments = [sys.executable, "-B", str(script), "@MODE@", str(self.events), "fixed operands", "W1-W7"]
        self.registration = "add_test(RepeatControl " + " ".join(f"[=[{value}]=]" for value in arguments) + ")\n"
        self.registration += 'set_tests_properties(RepeatControl PROPERTIES TIMEOUT 5)\n'

    def run_control(self, mode: str, selection: str = "^RepeatControl$") -> subprocess.CompletedProcess:
        (self.directory / "CTestTestfile.cmake").write_text(self.registration.replace("@MODE@", mode), encoding="utf-8")
        command = [self.ctest, "--test-dir", str(self.directory), "--parallel", "1", "--no-tests=error",
             "-R", selection, "--repeat", "until-fail:5", "--verbose", "--output-on-failure",
             "--stop-on-failure", "--output-log", str(self.directory / "ctest.log"),
             "--output-junit", str(self.directory / "junit.xml")]
        try:
            path = self.campaign.run("control", command, inputs=["control.py", "CTestTestfile.cmake"], timeout=20)
            record = json.loads(path.read_text())
        except subprocess.CalledProcessError:
            # A failing CTest child can be an expected, passed outer control.
            record = self.campaign.attempts("control")[-1]
        output = (self.directory / record["output"]).read_bytes()
        return subprocess.CompletedProcess(command, record["returncode"], output, b"")

    def rows(self) -> list[dict]:
        return [json.loads(line) for line in self.events.read_text(encoding="utf-8").splitlines()]

    def test_success_repeats_in_fresh_processes_with_exact_operands(self) -> None:
        result = self.run_control("success")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = self.rows()
        self.assertEqual([row["attempt"] for row in rows], list(range(1, 6)))
        self.assertEqual(len({(row["pid"], row["started"]) for row in rows}), 5)
        self.assertTrue(all(row["local_count"] == 1 and row["operands"] == ["fixed operands", "W1-W7"] for row in rows))
        # JUnit contains the final case only: verbose output/events prove all runs.
        self.assertEqual(len(list(ET.parse(self.directory / "junit.xml").iter("testcase"))), 1)

    def test_first_failure_stops_repetition_and_remains_in_evidence(self) -> None:
        result = self.run_control("fail-second")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([row["attempt"] for row in self.rows()], [1, 2])
        output = result.stdout
        self.assertIn(b"INTENTIONAL_SECOND_ATTEMPT_FAILURE", output)
        self.assertNotIn(b"CONTROL_ATTEMPT 3", output)
        self.assertIsNotNone(ET.parse(self.directory / "junit.xml").find(".//failure"))
        self.assertIn("INTENTIONAL_SECOND_ATTEMPT_FAILURE", (self.directory / "ctest.log").read_text())

    def test_empty_selection_cannot_qualify(self) -> None:
        result = self.run_control("success", "^MissingControl$")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.events.exists())


class FixtureLifetimeTests(unittest.TestCase):
    def run_fixture(self, body, *, retain=False):
        case = CtestRepetitionTests("test_empty_selection_cannot_qualify")
        setattr(case, case._testMethodName, lambda: body(case))
        result = unittest.TestResult()
        interrupted = False
        with patch.dict(os.environ, {"JUICER_TEST_RETAIN_FIXTURES": "1" if retain else ""}):
            try:
                case.run(result)
            except KeyboardInterrupt:
                interrupted = True
        return case, result, interrupted

    def test_routing_cleans_consumed_negative_control_and_opt_in_retains(self):
        for retain in (False, True):
            with self.subTest(retain=retain):
                case, result, interrupted = self.run_fixture(
                    CtestRepetitionTests.test_first_failure_stops_repetition_and_remains_in_evidence,
                    retain=retain,
                )
                self.assertTrue(result.wasSuccessful(), result.errors)
                self.assertFalse(interrupted)
                self.assertEqual(case.directory.exists(), retain)
                if retain:
                    self.assertEqual(json.loads((case.directory / MARKER).read_text())["outcome"], "passed")
                    receipt = case.campaign.attempts("control")[-1]
                    self.assertTrue(receipt["process_tree"]["drained"])
                    self.assertNotEqual(receipt["returncode"], 0)
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
                self.assertTrue(case.directory.exists())
                self.assertEqual(json.loads((case.directory / MARKER).read_text())["outcome"], "failed")
                case.temporary.cleanup()
        case = CtestRepetitionTests("test_empty_selection_cannot_qualify")
        case.setUp()
        case.doCleanups()
        self.assertTrue(case.directory.exists())
        case.temporary.cleanup()

    def test_interruption_and_unproved_timeout_keep_owner(self):
        previous = os.environ.get("JUICER_OUTPUT_LEASES")
        for mode in ("before-command", "after-command", "unproved-timeout", "cleanup-error"):
            with self.subTest(mode=mode):
                def body(case):
                    if mode == "cleanup-error":
                        def error():
                            raise OSError("cancelled fixture cleanup error")
                        case.addCleanup(error)
                    if mode == "after-command":
                        case.run_control("success")
                    if mode == "unproved-timeout":
                        # No child is launched by this injection; drainage is unknown to the owner.
                        def unproved(*args, **kwargs):
                            args[7].update(drained=False, status="running")
                            raise subprocess.TimeoutExpired(args[0], args[4])
                        with patch.object(execution, "run_process_tree", side_effect=unproved), case.assertRaises(subprocess.TimeoutExpired):
                            case.run_control("success")
                    else:
                        raise KeyboardInterrupt
                case, result, interrupted = self.run_fixture(body)
                self.assertEqual(interrupted, mode != "unproved-timeout")
                self.assertTrue(result.wasSuccessful(), result.errors)
                self.assertTrue(case.directory.exists())
                self.assertEqual(json.loads((case.lease.path / "owner.json").read_text())["status"], "blocked")
                self.assertNotIn(case.lease.path, execution.OutputLease.held)
                self.assertEqual(os.environ.get("JUICER_OUTPUT_LEASES"), previous)
                # These consumed controls either launched no child or completed a supervised command.
                case.temporary.cleanup()


class ArtifactSelectionTests(unittest.TestCase):
    def test_failed_outer_controls_keep_command_process_and_reproduction(self):
        with patch.object(sys, "path", [str(ROOT / "tests/quality"), *sys.path]):
            from test_checkpoint import CheckpointTests
            from test_build_evidence import BuildEvidenceTests
            from test_build_registration import BuildTests
        classes = ((CheckpointTests, "test_new_campaign_reuses_original_success_without_execution"),
                   (BuildEvidenceTests, "test_unchanged_reconfigure_preserves_registry_and_artifacts"),
                   (BuildTests, "test_selected_membership_executes_without_recursive_runtime_inventory"),
                   (CtestRepetitionTests, "test_empty_selection_cannot_qualify"))
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        with tempfile.TemporaryDirectory(prefix="CI control evidence ") as temporary:
            preset_root = Path(temporary) / "out/validation" / preset
            for cls, method in classes:
                with self.subTest(family=cls.__name__):
                    case = cls(method)
                    artifact_root = preset_root / ("ctest-repetition-controls" if cls is CtestRepetitionTests else "verification-controls")
                    fixture = None
                    def fail_command():
                        nonlocal fixture
                        fixture = case.directory if cls is CtestRepetitionTests else case.root
                        (fixture / "failure.py").write_text("raise SystemExit(17)\n")
                        campaign = execution.Campaign(fixture, preset, "ci-failure-control", ["control"])
                        campaign.run("control", [sys.executable, "failure.py"], inputs=["failure.py"], timeout=10)
                    setattr(case, method, fail_command)
                    with patch.dict(os.environ, {"JUICER_TEST_ARTIFACT_DIR": str(artifact_root), "JUICER_TEST_RETAIN_FIXTURES": ""}):
                        result = case.run(unittest.TestResult())
                    self.assertEqual(len(result.errors), 1, result.errors)
                    record = json.loads((fixture / MARKER).read_bytes())
                    self.assertEqual(record["outcome"], "failed")
                    self.assertEqual(record["test"], case.id())
                    self.assertTrue(record["errors"])
                    self.assertEqual(selected_paths(preset_root), [fixture])
                    receipts = list((fixture / "out/validation" / preset / "ci-failure-control/attempts").glob("*/result.json"))
                    self.assertEqual(len(receipts), 1)
                    command = json.loads(receipts[0].read_bytes())
                    self.assertEqual(command["returncode"], 17)
                    self.assertTrue(command["process_tree"]["drained"])
                    self.assertTrue((receipts[0].parent / "process.json").is_file())
                    self.assertTrue((receipts[0].parent / "output.log").is_file())
                    self.assertEqual((fixture / "failure.py").read_text(), "raise SystemExit(17)\n")
                    # The controlled failure was consumed and its process tree drained.
                    case.temporary.cleanup()

    def test_passing_negative_child_and_successful_retention_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            preset_root = Path(temporary)
            case = CtestRepetitionTests("test_first_failure_stops_repetition_and_remains_in_evidence")
            with patch.dict(os.environ, {"JUICER_TEST_ARTIFACT_DIR": str(preset_root / "ctest-repetition-controls"), "JUICER_TEST_RETAIN_FIXTURES": "1"}):
                result = case.run(unittest.TestResult())
            self.assertTrue(result.wasSuccessful(), result.errors)
            self.assertTrue(case.directory.exists())
            self.assertEqual(json.loads((case.directory / MARKER).read_bytes())["outcome"], "passed")
            self.assertEqual(selected_paths(preset_root), [])
            self.assertNotEqual(case.campaign.attempts("control")[-1]["returncode"], 0)
            case.temporary.cleanup()

    def test_cli_selects_only_failed_families_and_npy_samples(self):
        family = "windows-clang" if os.name == "nt" else "linux"
        with tempfile.TemporaryDirectory(prefix="artifact paths ü ") as temporary:
            workspace = Path(temporary)
            script = workspace / "tests/ctest/control_artifacts.py"
            script.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / "tests/ctest/control_artifacts.py", script)
            debug = workspace / "out/validation" / (family + "-debug")
            failed = debug / "verification-controls/registration-controls/failed control"
            unproved = debug / "checkpoint-controls/interrupted control"
            missing = debug / "build-evidence-controls/missing marker"
            missing.mkdir(parents=True)
            malformed = debug / "registration-controls/malformed marker"
            (malformed / MARKER).parent.mkdir(parents=True)
            (malformed / MARKER).write_text("[]")
            passed = debug / "ctest-repetition-controls/passed control"
            case = unittest.FunctionTestCase(lambda: None)
            record_fixture(failed, case, "failed")
            record_fixture(unproved, case, "running")
            record_fixture(passed, case, "passed")
            blocked = passed / "out/verification-locks/control/owner.json"
            blocked.parent.mkdir(parents=True)
            blocked.write_text('{"status":"blocked"}')
            sample = workspace / "out/validation" / (family + "-release") / "rust-core/npy-0081-0000.npy"
            sample.parent.mkdir(parents=True)
            sample.write_bytes(b"tiny sample")
            unrelated = workspace / "out/build/cache/retained-failure.json"
            unrelated.parent.mkdir(parents=True)
            unrelated.write_text('{"outcome":"failed"}')
            output = workspace / "github output.txt"
            command = [sys.executable, str(script), "--preset", family + "-debug", family + "-release", "--github-output", str(output)]
            subprocess.run(command, check=True, capture_output=True, timeout=10)
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "paths<<" + lines[-1])
            self.assertEqual(set(lines[1:-1]), {p.as_posix() + "/**" for p in
                                              (failed, unproved, missing, malformed)} | {sample.as_posix()})

    def test_native_workflows_upload_selected_paths_and_keep_reports(self):
        for native in ("linux", "windows"):
            text = (ROOT / f".github/workflows/tests-{native}-host.yml").read_text()
            self.assertIn("tests/ctest/control_artifacts.py --preset", text)
            self.assertIn("path: ${{ steps.failed-controls.outputs.paths }}", text)
            self.assertIn("include-hidden-files: true", text)
            self.assertIn("if: always() && steps.failed-controls.outputs.paths != ''", text)
            self.assertEqual(text.count("Testing/Temporary/LastTest.log"), 2)
            self.assertNotIn("out/validation/**", text)


if __name__ == "__main__":
    unittest.main()
