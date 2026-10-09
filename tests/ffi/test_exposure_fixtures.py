"""Check A5 fixture membership, immutable bits and lossless result sharing."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from compact_exposure_fixtures import compact_products


FIXTURES = Path(__file__).parent / "fixtures/exposure"


class ExposureFixtureTests(unittest.TestCase):
    def test_fixture_hashes_and_case_membership(self) -> None:
        manifest = json.loads((FIXTURES / "manifest.json").read_bytes())
        self.assertEqual(manifest["schema_version"], 2)
        for name, expected in manifest["files"].items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), expected)
        numerical = json.loads((FIXTURES / "numerical.json").read_bytes())
        values = json.loads((FIXTURES / "product-values.json").read_bytes())
        self.assertEqual(values["schema_version"], 1)
        for row in numerical["sensitivity"]:
            if not row["expected"]["built"]:
                self.assertIsInstance(row["expected"]["failure"], str)
                self.assertTrue(row["expected"]["failure"])
        for cohort in manifest["records"]:
            with self.subTest(preset=cohort["preset"]):
                self.assertEqual(len(numerical["reference"]), cohort["reference_cases"])
                self.assertEqual(len(numerical["sensitivity"]), cohort["sensitivity_cases"])
                products = json.loads((FIXTURES / f"products-{cohort['preset']}.json").read_bytes())
                self.assertEqual(len(products["products"]), cohort["product_attempts"])
                for row in products["products"]:
                    if row["built"]:
                        index = row["expected_record"]
                        self.assertIs(type(index), int)
                        self.assertGreaterEqual(index, 0)
                        self.assertLess(index, len(values["expected_records"]))
                        self.assertIsInstance(values["expected_records"][index], dict)

    def test_result_sharing_preserves_bits_controls_and_failures(self) -> None:
        expected = {"bits": [0, 0x80000000, 0x7FC00123], "hash": 0xFFFFFFFFFFFFFFFF}
        originals = {
            "debug": {"products": [{"built": True, "controls": {"amount": 0}, "expected": expected}, {"built": False, "diagnostic": "debug failure"}]},
            "release": {"products": [{"built": True, "controls": {"amount": 1}, "expected": expected}, {"built": False, "diagnostic": "release failure"}]},
        }
        candidates, records = compact_products(originals)
        self.assertEqual(records, [expected])
        for preset, fixture in candidates.items():
            expanded = json.loads(json.dumps(fixture))
            for row in expanded["products"]:
                if "expected_record" in row:
                    row["expected"] = records[row.pop("expected_record")]
            self.assertEqual(expanded, originals[preset])


if __name__ == "__main__":
    unittest.main()
