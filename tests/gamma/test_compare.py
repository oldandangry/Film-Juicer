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

    def test_numeric_zero_bound_passes(self) -> None:
        cases = (("integer-zero", 0), ("float-zero", 0.0))
        self.subtest_cases = tuple(case_id for case_id, _ in cases)
        for case_id, zero in cases:
            with self.subTest(case=case_id):
                record = dict(self.frozen)
                record["fused_maximum_absolute_difference"] = zero
                result, failures = comparator.compare_fused_case(
                    record, record, "paired-case", self.source, "paired-case"
                )
                self.assertEqual([], failures)
                self.assertEqual("passed", result["status"])
                self.assertEqual(0.0, result["candidate_difference"])
                self.assertEqual(0.0, result["frozen_bound"])

    def test_boolean_applicable_measurements_fail(self) -> None:
        cases = tuple((side, measurement) for side in ("candidate", "frozen") for measurement in (False, True))
        self.subtest_cases = tuple(f"{side}-boolean-{str(measurement).lower()}" for side, measurement in cases)
        for case_id, (side, measurement) in zip(self.subtest_cases, cases):
            with self.subTest(case=case_id):
                candidate = dict(self.frozen)
                frozen = dict(self.frozen)
                candidate["fused_maximum_absolute_difference"] = 0.0
                frozen["fused_maximum_absolute_difference"] = 1.0
                record = candidate if side == "candidate" else frozen
                record["fused_maximum_absolute_difference"] = measurement
                result, failures = comparator.compare_fused_case(
                    candidate, frozen, "paired-case", self.source, "paired-case"
                )
                self.assertEqual("failed", result["status"])
                self.assertEqual(
                    [f"paired-case.{side}: fused maximum difference must not be Boolean"],
                    failures,
                )

    def test_boolean_legacy_applicable_bound_fails(self) -> None:
        candidate = {
            "case": "negative-direct",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": 0.0,
        }
        measurements = (False, True)
        self.subtest_cases = ("false", "true")
        for case_id, measurement in zip(self.subtest_cases, measurements):
            with self.subTest(case=case_id):
                frozen = {
                    "case": "negative-direct",
                    "fused_maximum_absolute_difference": measurement,
                }
                result, failures = comparator.compare_fused_case(
                    candidate, frozen, "negative-direct", self.source, "negative-direct"
                )
                self.assertEqual("failed", result["status"])
                self.assertEqual(
                    ["negative-direct.frozen: fused maximum difference must not be Boolean"],
                    failures,
                )

    def test_malformed_and_changed_candidates_fail(self) -> None:
        valid = {
            "case": "paired-case",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": 0.25,
        }
        malformed_cases = []
        for case_id, value in (
            ("nan", math.nan), ("positive-infinity", math.inf),
            ("negative-infinity", -math.inf), ("negative-bound", -0.25),
        ):
            row = dict(valid)
            row["fused_maximum_absolute_difference"] = value
            malformed_cases.append((case_id, row))
        missing_difference = dict(valid)
        del missing_difference["fused_maximum_absolute_difference"]
        malformed_cases.append(("missing-difference", missing_difference))
        missing_applicability = dict(valid)
        del missing_applicability["fused_path_applicable"]
        malformed_cases.append(("missing-applicability", missing_applicability))
        wrong_case = dict(valid)
        wrong_case["case"] = "wrong-case"
        malformed_cases.append(("wrong-case", wrong_case))
        changed_applicability = dict(valid)
        changed_applicability["fused_path_applicable"] = False
        changed_applicability["fused_maximum_absolute_difference"] = None
        malformed_cases.append(("changed-applicability", changed_applicability))

        self.subtest_cases = (
            "nan", "positive-infinity", "negative-infinity", "negative-bound",
            "missing-difference", "missing-applicability", "wrong-case", "changed-applicability",
        )
        for case_id, (_, candidate) in zip(self.subtest_cases, malformed_cases):
            with self.subTest(case=case_id):
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

    def test_boolean_measurements_exit_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.json"
            frozen = root / "frozen.json"
            cases = tuple((side, measurement) for side in ("candidate", "frozen") for measurement in (False, True))
            self.subtest_cases = tuple(f"{side}-boolean-{str(measurement).lower()}" for side, measurement in cases)
            for case_id, (side, measurement) in zip(self.subtest_cases, cases):
                with self.subTest(case=case_id):
                    records = {
                        name: {
                            "case": "x",
                            "fused_path_applicable": True,
                            "fused_maximum_absolute_difference": difference,
                        }
                        for name, difference in (("candidate", 0.0), ("frozen", 1.0))
                    }
                    records[side]["fused_maximum_absolute_difference"] = measurement
                    candidate.write_text(json.dumps(records["candidate"]), encoding="utf-8")
                    frozen.write_text(json.dumps(records["frozen"]), encoding="utf-8")
                    result = self.run_compare(
                        "--candidate", str(candidate), "--frozen", str(frozen), "--case", "x"
                    )
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual("failed", report["status"])
                    self.assertEqual(
                        [f"x.{side}: fused maximum difference must not be Boolean"],
                        report["failures"],
                    )

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


FusedEquivalenceTests.test_numeric_zero_bound_passes.required_subtests = True
FusedEquivalenceTests.test_numeric_zero_bound_passes.subtest_cases = ("integer-zero", "float-zero")
FusedEquivalenceTests.test_boolean_applicable_measurements_fail.required_subtests = True
FusedEquivalenceTests.test_boolean_applicable_measurements_fail.subtest_cases = (
    "candidate-boolean-false", "candidate-boolean-true", "frozen-boolean-false", "frozen-boolean-true"
)
FusedEquivalenceTests.test_boolean_legacy_applicable_bound_fails.required_subtests = True
FusedEquivalenceTests.test_boolean_legacy_applicable_bound_fails.subtest_cases = ("false", "true")
FusedEquivalenceTests.test_malformed_and_changed_candidates_fail.required_subtests = True
FusedEquivalenceTests.test_malformed_and_changed_candidates_fail.subtest_cases = (
    "nan", "positive-infinity", "negative-infinity", "negative-bound", "missing-difference",
    "missing-applicability", "wrong-case", "changed-applicability"
)
CommandLineFailureTests.test_boolean_measurements_exit_failed.required_subtests = True
CommandLineFailureTests.test_boolean_measurements_exit_failed.subtest_cases = (
    "candidate-boolean-false", "candidate-boolean-true", "frozen-boolean-false", "frozen-boolean-true"
)


if __name__ == "__main__":
    unittest.main()
