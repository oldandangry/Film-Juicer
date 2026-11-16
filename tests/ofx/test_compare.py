"""Failure boundaries for the production halation comparator (no GPU required)."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

import compare


class ComparisonTests(unittest.TestCase):
    def test_float_comparison(self):
        self.subtest_cases = ("signed-zero", "one-ulp", "nonfinite", "alpha")
        with tempfile.TemporaryDirectory() as directory:
            left, right = (Path(directory) / name for name in ("left.f32", "right.f32"))
            pixels = np.array([[[0.0, 0.5, 1.0, 0.25], [0.1, 0.2, 0.3, 0.75]]], dtype="<f4")
            pixels.tofile(left)
            pixels.tofile(right)
            self.assertTrue(compare.compare_pixels(left, right, pixels.shape)["bit_identical"])
            for case, changed_value in (("signed-zero", -0.0), ("one-ulp", np.nextafter(np.float32(0), np.float32(1)))):
                with self.subTest(case=case):
                    changed = pixels.copy()
                    changed[0, 0, 0] = changed_value
                    changed.tofile(right)
                    result = compare.compare_pixels(left, right, pixels.shape)
                    self.assertFalse(result["bit_identical"])
                    self.assertEqual(result["changed_pixels"], 1)
                    self.assertEqual(result["changed_rgb_samples"], 1)
            for location, value, error in (((0, 0, 1), np.nan, "nonfinite"), ((0, 0, 3), 0.5, "alpha")):
                with self.subTest(case=error):
                    changed = pixels.copy()
                    changed[location] = value
                    changed.tofile(right)
                    with self.assertRaisesRegex(ValueError, error):
                        compare.compare_pixels(left, right, pixels.shape)
            right.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "size"):
                compare.compare_pixels(left, right, pixels.shape)

    @staticmethod
    def capture():
        rows = []
        parameters = {}
        for case in compare.CASES:
            parameters[case] = {
                "HalationActive": [1],
                "HalationScatterAmount": [1.0],
                "HalationScatterSpatialScale": [1.0],
                "HalationAmount": [1.0],
                "HalationSpatialScale": [1.0],
                "SpectralUpsampling": [0],
                "InputColorSpace": [1],
                "CameraAutoExposure": [int(case.endswith("-on"))],
                "CameraFilmFormatMm": [35.0],
                "FilmProfileKey": ["kodak_portra_400" if case.startswith("negative") else "fujifilm_provia_100f"],
                "ScanRoute": ["NegativePrintScan" if "-print-" in case else "NegativeDirectScan"],
                "GrainActive": [0],
                "CameraDiffusionEnabled": [0],
                "PrintDiffusionEnabled": [0],
                **copy.deepcopy(compare.ADVANCED_DEFAULTS),
            }
            for phase in ("cold", "warm"):
                rows.append(
                    {
                        "case_id": f"{case}/{phase}",
                        "finite_output": True,
                        "alpha_preserved": True,
                        "padding_preserved": True,
                        "source_preserved": True,
                        "stream_completed": True,
                        "readback_complete": True,
                        "context_preserved": True,
                        "context_restored": True,
                        "cold_warm_identical": True,
                        "ofx_status": 0,
                        "messages": [],
                        "input_shape": list(compare.SHAPE),
                        "output_shape": list(compare.SHAPE),
                        "samples_compared": int(np.prod(compare.SHAPE)),
                        "source_image_releases": 1,
                        "output_image_releases": 1,
                    }
                )
        return {
            "schema": "ofx-halation-capture-v1",
            "rows": rows,
            "parameters": parameters,
            "lifecycle": {
                "effects_created": 8,
                "effects_destroyed": 8,
                "destroy_statuses": [0] * 8,
                "module_load_status": 0,
                "module_unload_status": 0,
                "context_restored": True,
            },
        }

    def test_incomplete_and_wrong_scope_capture(self):
        report = self.capture()
        compare.validate_capture(report, "candidate")
        self.subtest_cases = ("missing-row", "duplicate-row", "warm", "release", "halation", "ae", "boost", "unload")
        for defect in self.subtest_cases:
            with self.subTest(case=defect):
                invalid = copy.deepcopy(report)
                if defect == "missing-row":
                    invalid["rows"].pop()
                elif defect == "duplicate-row":
                    invalid["rows"][1] = invalid["rows"][0]
                elif defect == "warm":
                    invalid["rows"][1]["cold_warm_identical"] = False
                elif defect == "release":
                    invalid["rows"][0]["source_image_releases"] = 0
                elif defect == "unload":
                    invalid["lifecycle"]["module_unload_status"] = 1
                else:
                    name = {
                        "halation": "HalationActive",
                        "ae": "CameraAutoExposure",
                        "boost": "HalationHighlightBoostEv",
                    }[defect]
                    invalid["parameters"]["negative-print-ae-on"][name] = [0 if defect != "boost" else 1.0]
                with self.assertRaises(ValueError):
                    compare.validate_capture(invalid, "candidate")

    test_float_comparison.required_subtests = True
    test_incomplete_and_wrong_scope_capture.required_subtests = True


if __name__ == "__main__":
    unittest.main()
