from __future__ import annotations

import io
import json
import platform
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.diffusion.reference.tests import test_reference_contract


class MaintenanceTests(unittest.TestCase):
    def run_case(self, case: unittest.TestCase) -> unittest.TestResult:
        result = unittest.TestResult()
        case.run(result)
        return result

    def test_committed_contract_needs_no_reference_environment(self) -> None:
        # A different checkout and absent historical environment cannot invalidate bytes.
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_reference_contract)
        with mock.patch.object(test_reference_contract.subprocess, "run", side_effect=AssertionError("unexpected Git/environment command")), \
             mock.patch("builtins.__import__", wraps=__import__) as imports:
            result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertEqual(result.testsRun, 9)
        self.assertFalse(any(call.args[0].startswith(("spektrafilm", "tools.diffusion_reference"))
                             or call.args[0] == "tests.diffusion.reference" for call in imports.call_args_list))

    def test_committed_provenance_edit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "fixture.json"
            data = json.loads(test_reference_contract.MANIFEST.read_text(encoding="utf-8"))
            data["reference_git"]["commit"] = "0" * 40
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with mock.patch.object(test_reference_contract, "MANIFEST", manifest):
                result = self.run_case(test_reference_contract.PromotedFixtureIntegrityTests("test_committed_bytes"))
        self.assertEqual(len(result.failures), 1)
        self.assertFalse(result.errors)

    def test_payload_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = Path(temporary) / "fixture.npz"
            with np.load(test_reference_contract.ARRAYS, allow_pickle=False) as arrays:
                changed = {name: arrays[name] for name in arrays.files}
            first = next(iter(changed))
            changed[first] = changed[first].copy()
            changed[first].flat[0] += 1
            np.savez_compressed(payload, **changed)
            case = test_reference_contract.ReferenceContractTests("test_array_manifest_matches_promoted_payload")
            case.arrays = payload
            result = self.run_case(case)
        self.assertEqual(len(result.failures), 1)
        self.assertFalse(result.errors)

    def test_capture_rejects_another_environment_with_the_same_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = json.loads(test_reference_contract.MANIFEST.read_text(encoding="utf-8"))
            data["python"].update(
                version=platform.python_version(), implementation=platform.python_implementation(),
                platform=platform.platform(), machine=platform.machine(), executable=sys.executable,
            )
            (directory / "diffusion_reference_v1.json").write_text(json.dumps(data), encoding="utf-8")
            cases = {case._testMethodName: case for case in test_reference_contract.capture_suite(directory)}
            with mock.patch.object(test_reference_contract.sys, "prefix", str(directory / "another-environment")):
                result = self.run_case(cases["test_current_environment_provenance"])
        self.assertEqual(len(result.failures), 1)
        self.assertFalse(result.errors)
        self.assertIn("capture environment prefix", result.failures[0][1])

    def test_capture_checks_are_explicit_and_reject_historical_environment(self) -> None:
        suite = test_reference_contract.capture_suite(test_reference_contract.MANIFEST.parent)
        cases = {case._testMethodName: case for case in suite}
        self.assertEqual(len(cases), 8)
        self.assertIn("test_current_imported_source_provenance", cases)
        with mock.patch.object(test_reference_contract.platform, "python_version", return_value="3.13.stale"):
            result = self.run_case(cases["test_current_environment_provenance"])
        self.assertEqual(len(result.failures), 1)
        self.assertFalse(result.errors)


if __name__ == "__main__":
    unittest.main()
