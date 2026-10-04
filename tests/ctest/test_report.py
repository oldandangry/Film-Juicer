from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import report


class CtestReportTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.junit = root / "full.xml"
        self.inventory = root / "focused.json"
        self.inventory.write_text(json.dumps({"tests": [{"name": "owner"}, {"name": "lifetime"}]}))

    def write_cases(self, cases: str) -> None:
        self.junit.write_text(f"<testsuite>{cases}</testsuite>")

    def test_required_selection_is_reported_from_full_suite(self) -> None:
        self.write_cases('<testcase name="unrelated" status="run"/>'
                         '<testcase name="owner" status="run"/>'
                         '<testcase name="lifetime" status="run"/>')
        summary = report.summarize(self.junit, self.inventory)
        self.assertEqual((summary["tests"], summary["passed"]), (2, 2))
        self.assertIn("no additional execution", summary["source"])

    def test_missing_failed_and_skipped_cases_cannot_pass(self) -> None:
        for case, category in (
            ('', "missing"),
            ('<testcase name="lifetime" status="fail"><failure/></testcase>', "failed"),
            ('<testcase name="lifetime" status="run"><error/></testcase>', "failed"),
            ('<testcase name="lifetime" status="notrun"><skipped/></testcase>', "skipped"),
        ):
            with self.subTest(category=category):
                self.write_cases('<testcase name="owner" status="run"/>' + case)
                summary = report.summarize(self.junit, self.inventory)
                self.assertEqual(summary["passed"], 1)
                self.assertEqual(summary[category], ["lifetime"])
                with patch("sys.argv", ["report.py", "--junit", str(self.junit),
                                        "--inventory", str(self.inventory)]), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(report.main(), 1)

    def test_empty_invalid_and_duplicate_selections_are_rejected(self) -> None:
        self.write_cases('<testcase name="owner" status="run"/>')
        for selection in ({}, {"tests": []}, {"tests": [None]},
                          {"tests": [{"name": ""}]}, {"tests": [{"name": "owner"}] * 2}):
            with self.subTest(selection=selection):
                self.inventory.write_text(json.dumps(selection))
                with self.assertRaises(ValueError):
                    report.summarize(self.junit, self.inventory)

    def test_duplicate_cases_and_unknown_execution_status_are_rejected(self) -> None:
        for cases in ('<testcase name="owner" status="run"/>' * 2,
                      '<testcase name="owner"/>', '<testcase status="run"/>'):
            with self.subTest(cases=cases):
                self.write_cases(cases)
                with self.assertRaises(ValueError):
                    report.summarize(self.junit, self.inventory)

    def test_nested_junit_suites_are_supported(self) -> None:
        self.junit.write_text('<testsuites><testsuite><testcase name="owner" status="run"/>'
                              '</testsuite><testsuite><testcase name="lifetime" status="run"/>'
                              '</testsuite></testsuites>')
        self.assertEqual(report.summarize(self.junit, self.inventory)["passed"], 2)

    def test_malformed_report_fails_at_the_command_boundary(self) -> None:
        self.junit.write_text('<testsuite>')
        with patch("sys.argv", ["report.py", "--junit", str(self.junit),
                                "--inventory", str(self.inventory)]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:
                report.main()
        self.assertEqual(failure.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
