from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


class CtestRepetitionTests(unittest.TestCase):
    def setUp(self) -> None:
        artifact_root = os.environ.get("JUICER_TEST_ARTIFACT_DIR")
        if artifact_root:
            Path(artifact_root).mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix="repeat control ", dir=artifact_root))
        else:
            temporary = tempfile.TemporaryDirectory(prefix="repeat control ")
            self.addCleanup(temporary.cleanup)
            self.directory = Path(temporary.name)
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
            result = subprocess.run(command, capture_output=True, timeout=20, check=False)
        except subprocess.TimeoutExpired as error:
            (self.directory / "output.log").write_bytes((error.stdout or b"") + (error.stderr or b""))
            raise
        (self.directory / "output.log").write_bytes(result.stdout + result.stderr)
        return result

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
        output = (self.directory / "output.log").read_bytes()
        self.assertIn(b"INTENTIONAL_SECOND_ATTEMPT_FAILURE", output)
        self.assertNotIn(b"CONTROL_ATTEMPT 3", output)
        self.assertIsNotNone(ET.parse(self.directory / "junit.xml").find(".//failure"))
        self.assertIn("INTENTIONAL_SECOND_ATTEMPT_FAILURE", (self.directory / "ctest.log").read_text())

    def test_empty_selection_cannot_qualify(self) -> None:
        result = self.run_control("success", "^MissingControl$")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.events.exists())


if __name__ == "__main__":
    unittest.main()
