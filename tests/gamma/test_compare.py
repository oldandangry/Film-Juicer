from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import compare as comparator


class PythonRuntimeTests(unittest.TestCase):
    def test_ordinary_suite_uses_python_3_13(self) -> None:
        self.assertEqual(sys.version_info[:2], (3, 13), sys.version)


class FusedEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = [{"path": "/frozen/capture.json", "sha256": "a" * 64}]
        self.frozen = {
            "case": "paired-case",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": 0.5,
        }

    def compare(self, candidate: dict):
        return comparator.compare_fused_case(
            candidate,
            self.frozen,
            "paired-case",
            self.source,
            "paired-case",
        )

    def test_equal_bound_passes_and_next_double_fails(self) -> None:
        passing = {
            "case": "paired-case",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": 0.5,
        }
        result, failures = self.compare(passing)
        self.assertEqual([], failures)
        self.assertEqual("passed", result["status"])

        failing = dict(passing)
        failing["fused_maximum_absolute_difference"] = math.nextafter(
            0.5, math.inf
        )
        result, failures = self.compare(failing)
        self.assertTrue(failures)
        self.assertEqual("failed", result["status"])

    def test_zero_bound_rejects_positive_difference(self) -> None:
        frozen = dict(self.frozen)
        frozen["fused_maximum_absolute_difference"] = 0.0
        candidate = {
            "case": "paired-case",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": math.nextafter(0.0, math.inf),
        }
        result, failures = comparator.compare_fused_case(
            candidate,
            frozen,
            "paired-case",
            self.source,
            "paired-case",
        )
        self.assertTrue(failures)
        self.assertEqual(0.0, result["frozen_bound"])

    def test_malformed_and_changed_candidates_fail(self) -> None:
        valid = {
            "case": "paired-case",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": 0.25,
        }
        malformed = []
        for value in (math.nan, math.inf):
            row = dict(valid)
            row["fused_maximum_absolute_difference"] = value
            malformed.append(row)
        missing_difference = dict(valid)
        del missing_difference["fused_maximum_absolute_difference"]
        malformed.append(missing_difference)
        missing_applicability = dict(valid)
        del missing_applicability["fused_path_applicable"]
        malformed.append(missing_applicability)
        wrong_case = dict(valid)
        wrong_case["case"] = "wrong-case"
        malformed.append(wrong_case)
        changed_applicability = dict(valid)
        changed_applicability["fused_path_applicable"] = False
        changed_applicability["fused_maximum_absolute_difference"] = None
        malformed.append(changed_applicability)

        for index, candidate in enumerate(malformed):
            with self.subTest(index=index):
                result, failures = self.compare(candidate)
                self.assertTrue(failures)
                self.assertEqual("failed", result["status"])

    def test_non_applicable_diffusion_is_explicit(self) -> None:
        candidate = {
            "case": "enlarger-diffusion-black-pro-mist-half",
            "fused_path_applicable": False,
            "fused_maximum_absolute_difference": None,
        }
        frozen = {
            "case": "enlarger-diffusion-black-pro-mist-half",
            "fused_path_applicable": False,
            "fused_maximum_absolute_difference": 0.0,
        }
        result, failures = comparator.compare_fused_case(
            candidate,
            frozen,
            candidate["case"],
            self.source,
            "diffusion",
        )
        self.assertEqual([], failures)
        self.assertEqual("not-applicable", result["status"])
        self.assertIsNone(result["candidate_difference"])

    def test_legacy_applicability_is_limited_to_known_direct_cases(self) -> None:
        candidate = {
            "case": "negative-direct",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": 0.0,
        }
        frozen = {
            "case": "negative-direct",
            "fused_maximum_absolute_difference": 0.0,
        }
        result, failures = comparator.compare_fused_case(
            candidate,
            frozen,
            "negative-direct",
            self.source,
            "negative-direct",
        )
        self.assertEqual([], failures)
        self.assertEqual("passed", result["status"])

        frozen["case"] = "unknown-direct"
        candidate["case"] = "unknown-direct"
        result, failures = comparator.compare_fused_case(
            candidate,
            frozen,
            "unknown-direct",
            self.source,
            "unknown-direct",
        )
        self.assertTrue(failures)
        self.assertEqual("failed", result["status"])


class CommandLineFailureTests(unittest.TestCase):
    def run_compare(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(Path(__file__).with_name("compare.py")), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_missing_input_is_nonzero_and_named(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "frozen.json"
            frozen.write_text(json.dumps({"case": "x"}), encoding="utf-8")
            result = self.run_compare(
                "--candidate",
                str(root / "missing.json"),
                "--frozen",
                str(frozen),
                "--case",
                "x",
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing comparison input", result.stderr)

    def test_deliberate_numerical_mismatch_is_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.json"
            frozen = root / "frozen.json"
            candidate.write_text(
                json.dumps(
                    {
                        "case": "x",
                        "fused_path_applicable": True,
                        "fused_maximum_absolute_difference": 0.5000001,
                    }
                ),
                encoding="utf-8",
            )
            frozen.write_text(
                json.dumps(
                    {
                        "case": "x",
                        "fused_path_applicable": True,
                        "fused_maximum_absolute_difference": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            result = self.run_compare(
                "--candidate",
                str(candidate),
                "--frozen",
                str(frozen),
                "--case",
                "x",
            )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "failed")
        self.assertIn("fused difference", report["failures"][0])


if __name__ == "__main__":
    unittest.main()
