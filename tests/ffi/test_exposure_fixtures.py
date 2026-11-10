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
        numerical = json.loads((FIXTURES / "numerical.json").read_bytes())
        values = json.loads((FIXTURES / "product-values.json").read_bytes())
        products_by_preset = {
            cohort["preset"]: json.loads((FIXTURES / f"products-{cohort['preset']}.json").read_bytes())["products"]
            for cohort in manifest["records"]
        }
        self.subtest_cases = tuple(
            [f"file/{name}" for name in manifest["files"]]
            + [f"sensitivity/{index}-{row['input']['id']}" for index, row in enumerate(numerical["sensitivity"])]
            + [f"cohort/{cohort['preset']}" for cohort in manifest["records"]]
            + [f"product/{preset}/{index}" for preset, rows in products_by_preset.items()
               for index, _ in enumerate(rows)]
        )
        self.assertEqual(manifest["schema_version"], 2)
        for name, expected in manifest["files"].items():
            with self.subTest(case=f"file/{name}", name=name):
                self.assertEqual(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(values["schema_version"], 1)
        for index, row in enumerate(numerical["sensitivity"]):
            with self.subTest(case=f"sensitivity/{index}-{row['input']['id']}"):
                if not row["expected"]["built"]:
                    self.assertIsInstance(row["expected"]["failure"], str)
                    self.assertTrue(row["expected"]["failure"])
        for cohort in manifest["records"]:
            preset = cohort["preset"]
            with self.subTest(case=f"cohort/{preset}"):
                self.assertEqual(len(numerical["reference"]), cohort["reference_cases"])
                self.assertEqual(len(numerical["sensitivity"]), cohort["sensitivity_cases"])
                products = products_by_preset[preset]
                self.assertEqual(len(products), cohort["product_attempts"])
                for index, row in enumerate(products):
                    with self.subTest(case=f"product/{preset}/{index}"):
                        if not row["built"]:
                            continue
                        index = row["expected_record"]
                        self.assertIs(type(index), int)
                        self.assertGreaterEqual(index, 0)
                        self.assertLess(index, len(values["expected_records"]))
                        self.assertIsInstance(values["expected_records"][index], dict)

    test_fixture_hashes_and_case_membership.required_subtests = True

    def test_result_sharing_preserves_bits_controls_and_failures(self) -> None:
        expected = {"bits": [0, 0x80000000, 0x7FC00123], "hash": 0xFFFFFFFFFFFFFFFF}
        originals = {
            "debug": {"products": [{"built": True, "controls": {"amount": 0}, "expected": expected}, {"built": False, "diagnostic": "debug failure"}]},
            "release": {"products": [{"built": True, "controls": {"amount": 1}, "expected": expected}, {"built": False, "diagnostic": "release failure"}]},
        }
        self.subtest_cases = tuple(
            f"{preset}/product/{index}"
            for preset, fixture in originals.items()
            for index, _ in enumerate(fixture["products"])
        )
        candidates, records = compact_products(originals)
        self.assertEqual(records, [expected])
        for preset, fixture in candidates.items():
            expanded = json.loads(json.dumps(fixture))
            for index, row in enumerate(expanded["products"]):
                with self.subTest(case=f"{preset}/product/{index}"):
                    if "expected_record" in row:
                        row["expected"] = records[row.pop("expected_record")]
                    self.assertEqual(row, originals[preset]["products"][index])
            self.assertEqual(expanded, originals[preset])

    test_result_sharing_preserves_bits_controls_and_failures.required_subtests = True


if __name__ == "__main__":
    unittest.main()
