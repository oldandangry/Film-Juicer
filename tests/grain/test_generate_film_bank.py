#!/usr/bin/env python3
"""Check Film v1 bank calibration admission through the authoring CLI."""

import io
import json
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate-film-bank.py"
GOLD_CALIBRATION = ROOT / "tests/grain/calibration/gold-200.json"


class FilmBankCalibrationTests(unittest.TestCase):
    def test_rejects_invalid_sublayer_shares_before_capture(self):
        calibration = json.loads(GOLD_CALIBRATION.read_bytes())
        float32_range_error = "sublayer_shares must each be strictly between 0 and 1 after Float32 conversion"
        for shares, diagnostic in (
            ([0.99999999, 0.000000005, 0.000000003, 0.000000002], float32_range_error),
            ([1e-50, 0.3, 0.3, 0.4], float32_range_error),
            ([0.25] * 3, "sublayer_shares must have 4 entries"),
            ([0.2] * 4, "sublayer_shares must sum to 1"),
        ):
            with self.subTest(shares=shares), tempfile.TemporaryDirectory() as directory:
                scratch = Path(directory)
                calibration_path = scratch / "calibration.json"
                calibration_path.write_text(json.dumps({**calibration, "sublayer_shares": shares}), encoding="utf-8")
                output = scratch / "bank"
                argv = [
                    str(GENERATOR),
                    "generate",
                    "--tool",
                    str(scratch / "swift-tool"),
                    "--stock",
                    "kodak_gold_200",
                    "--calibration",
                    str(calibration_path),
                    "--output",
                    str(output),
                ]
                stderr = io.StringIO()
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch("subprocess.Popen", side_effect=AssertionError("unexpected Cargo or Swift launch")),
                    redirect_stderr(stderr),
                    self.assertRaises(SystemExit) as result,
                ):
                    runpy.run_path(str(GENERATOR), run_name="__main__")

                self.assertEqual(result.exception.code, 1)
                self.assertEqual(stderr.getvalue(), f"Film v1 bank: {diagnostic}\n")
                self.assertEqual(set(scratch.iterdir()), {calibration_path})


if __name__ == "__main__":
    unittest.main()
