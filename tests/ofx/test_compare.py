"""Failure boundaries for the production halation comparator (no GPU required)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import compare


class ComparisonTests(unittest.TestCase):
    def test_complete_matched_field_requires_both_live_draws_and_the_terminal_readback(self):
        self.subtest_cases = ("complete", "missing-xyz", "paper", "glare", "color", "readback")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "frame.json"
            bounds = [0, 0, 1, 1]
            frame.write_text(
                json.dumps(
                    {
                        "output": {"bounds": bounds, "components": "OfxImageComponentRGB"},
                        "action": {"render_window": bounds, "time": 0.0},
                        "observe_print": True,
                        "print_glare": "constant",
                        "parameters": {"ScanRoute": {"values": ["PositivePrintScan"]}},
                    }
                ),
                encoding="utf-8",
            )
            for defect in self.subtest_cases:
                with self.subTest(case=defect):
                    for side in compare.SIDES:
                        path = root / side
                        path.mkdir(exist_ok=True)
                        cubic = side == "reference"
                        change = defect if not cubic else "complete"
                        owner = {"resources": 11, "context": 12, "device": 0, "epoch": 13}
                        host_rows, observed = [], []
                        for index, phase in enumerate(("cold", "warm", "reporter")):
                            tail = index == 2
                            draw = {
                                "pass": phase,
                                "status": 0,
                                "time": float(tail),
                                "output_file": None if tail else f"{phase}.f32",
                                **dict.fromkeys(
                                    (
                                        "source_preserved",
                                        "stream_completed",
                                        "context_preserved",
                                        "destination_guard_checked",
                                    ),
                                    True,
                                ),
                                "source_releases": 1,
                                "output_releases": 1,
                            }
                            held = {
                                **owner,
                                "event": 20 + index,
                                "host": 15,
                                "epoch_at_production": 13,
                                "producer": [1 + int(tail), int(tail)],
                                "success": True,
                                "handoff": True,
                            }
                            observed_row = {
                                **owner,
                                "index": index,
                                "route": 3,
                                "thread": 6,
                                "instance": 7,
                                "time_bits": 0x3FF0000000000000 if tail else 0,
                                "token": int(tail),
                                "submission": 1 + int(tail),
                                "stream": 16,
                                "outcome": [0, 0, 0, 0],
                                "prepare": {"stream": 16, "outcome": 0},
                                "retained": held,
                                "poll": {
                                    **owner,
                                    "stream": 16,
                                    "before": [101, 102],
                                    "after": [101, 101],
                                    "entry": [15, 19 + index],
                                    "queries": [
                                        {
                                            "event": 19 + index,
                                            "status": 0,
                                            "completed_flag": int(tail and change == "readback"),
                                        }
                                    ],
                                    "destroyed": [19 + index],
                                    "success": True,
                                    "status": 0,
                                },
                            }
                            if not tail:
                                (path / draw["output_file"]).write_bytes(struct.pack("<fff", 0, 0, 0))
                                density = path / f"{phase}.paper.f32"
                                density.write_bytes(struct.pack("<fff", float(change == "paper"), 0, 0))
                                xyz = path / f"{phase}.xyz.f64"
                                xyz.write_bytes(struct.pack("<ddd", 0, 0, 0))
                                glare = path / f"{phase}.glare.f32"
                                glare.write_bytes(struct.pack("<f", 2 if change == "glare" else 1))
                                draw["scanner_evidence"] = {
                                    "xyz": xyz.name,
                                    "glare_file": glare.name,
                                    "pixels": 1,
                                    "materialized": True,
                                    "glare": True,
                                    "held_cmy": False,
                                    "dust": False,
                                    "correction": 0,
                                    "slope": 1.0,
                                    "offset": 0.0,
                                    "white": [float(change == "color"), 1.0, 1.0],
                                    "cat02": [0.0] * 9,
                                    "xyz_to_rgb": [0.0] * 9,
                                }
                                if change == "missing-xyz" and phase == "warm":
                                    del draw["scanner_evidence"]
                                observed_row.update(
                                    scan_medium="Print",
                                    launches=["juicer_cuda_print_focused_scan_linear_density_rgb"],
                                    density={
                                        "width": 1,
                                        "height": 1,
                                        "file": density.name,
                                        "sha256": compare.execution.digest(density),
                                    },
                                    correction_words={
                                        "correctionActive": 0,
                                        "correctionSlope": 0x3F800000,
                                        "correctionOffset": 0,
                                    },
                                    scan_color={
                                        "cat02": [0] * 9,
                                        "xyzToRgb": [0] * 9,
                                        "outputGamutActive": [0],
                                        "encoding": [0] * (18 + int(cubic)),
                                    },
                                )
                            host_rows.append(draw)
                            observed.append(observed_row)
                        compare.execution.write_json(
                            path / "replay.json",
                            {
                                "schema": "juicer-ofx-replay-v1",
                                "context_restored": True,
                                "cubic_grid": 17 if cubic else None,
                                "parameters": {},
                                "rows": host_rows,
                            },
                        )
                        compare.execution.write_json(
                            path / "print-scanner.json", {"status": "passed", "exit": 0, "frames": observed}
                        )
                    if defect == "complete":
                        proof = compare.compare_scanner_frame(root, frame, scanner_xyz=True)["print_scanner"]
                        self.assertEqual(proof["claim"], "matched-field")
                        self.assertIn("pre_glare_xyz", proof)
                    else:
                        errors = {
                            "missing-xyz": "missing live pre-glare XYZ",
                            "paper": "live Print paper C/M/Y differs",
                            "glare": "expected exact uniform one-percent glare",
                            "color": "matched-field Print correction or color inputs differ",
                            "readback": "exact preceding owner event and zero flag",
                        }
                        with self.assertRaisesRegex(ValueError, errors[defect]):
                            compare.compare_scanner_frame(root, frame, scanner_xyz=True)

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
    test_complete_matched_field_requires_both_live_draws_and_the_terminal_readback.required_subtests = True


if __name__ == "__main__":
    unittest.main()
