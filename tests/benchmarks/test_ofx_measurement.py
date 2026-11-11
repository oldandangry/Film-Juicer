"""Bounded current-module compatibility, not a performance measurement."""

import os
from pathlib import Path
import unittest

import ofx


class OfxMeasurementTests(unittest.TestCase):
    def test_workloads(self):
        driver = os.environ["FJ_MEASUREMENT_DRIVER"]
        resources = os.environ["FJ_MEASUREMENT_RESOURCES"]
        directory = Path(os.environ.get("FJ_HOST_BUNDLE", os.environ["FJ_MEASUREMENT_OUTPUT"])) / "ofx-measurement"
        self.subtest_cases = ofx.ROWS
        directory.mkdir(parents=True, exist_ok=True)
        for row in ofx.ROWS:
            with self.subTest(case=row):
                ofx.check_row(
                    driver,
                    os.environ["FJ_MEASUREMENT_SHIPPING"],
                    os.environ["FJ_MEASUREMENT_OBSERVED"],
                    resources,
                    directory,
                    row,
                )


OfxMeasurementTests.test_workloads.required_subtests = True
OfxMeasurementTests.test_workloads.subtest_cases = ofx.ROWS


if __name__ == "__main__":
    unittest.main()
