"""Immutable native A8 capture membership and complete byte sharing."""

import hashlib
import json
from pathlib import Path
import unittest

FIXTURES = Path(__file__).parent / "fixtures/input_hull"


class InputHullFixtures(unittest.TestCase):
    def test_provenance_membership_bits_and_sharing(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        payloads = {name: (FIXTURES / name).read_bytes() for name in manifest["sha256"]}
        records_by_name = {
            member["records"]: json.loads(payloads[member["records"]])
            for member in manifest["preset_membership"].values()
        }
        self.subtest_cases = tuple(
            [f"preset/{preset}" for preset in manifest["preset_membership"]]
            + [f"sha256/{name}" for name in manifest["sha256"]]
            + [
                f"math/{name}/{index}"
                for name, records in records_by_name.items()
                for index, _ in enumerate(records["math"])
            ]
            + [
                (
                    f"product/{name}/control/{index}-{row['method']}-{row['compression']}"
                    if "polarity" not in row
                    else f"product/{name}/{row['polarity']}-{row['route']}-{row['method']}-{row['compression']}"
                )
                for name, records in records_by_name.items()
                for index, row in enumerate(records["products"])
            ]
            + [f"tables/{name}" for name in {member["tables"] for member in manifest["preset_membership"].values()}]
            + [
                f"primitives/{name}"
                for name in {member["primitives"] for member in manifest["preset_membership"].values()}
            ]
        )
        self.assertEqual(manifest["parent"], "e9b4abbb8b146ffdd0f4fbc5f8c61e7e87134c2a")
        self.assertEqual(
            set(manifest["preset_membership"]),
            {"linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"},
        )
        for name, digest in manifest["sha256"].items():
            with self.subTest(case=f"sha256/{name}"):
                self.assertEqual(hashlib.sha256(payloads[name]).hexdigest(), digest)
        for preset, member in manifest["preset_membership"].items():
            with self.subTest(case=f"preset/{preset}"):
                for kind, name in member.items():
                    self.assertEqual(manifest["sha256"][name], manifest["provenance"][preset][kind]["native_sha256"])
        for name in {member["records"] for member in manifest["preset_membership"].values()}:
            records = records_by_name[name]
            self.assertEqual(len(records["math"]), 14)
            self.assertEqual(len(records["products"]), 18)
            self.assertEqual(len(records["weights"]), 1024)
            expected_products = {(p, r, m, c) for p in range(2) for r in range(2) for m in (0, 2) for c in range(2)}
            actual_products = {
                (row["polarity"], row["route"], row["method"], row["compression"])
                for row in records["products"]
                if row["method"] != 1
            }
            self.assertEqual(actual_products, expected_products)
            for index, row in enumerate(records["math"]):
                with self.subTest(case=f"math/{name}/{index}"):
                    if row["built"]:
                        self.assertEqual(len(row["trace"]["locus"]), 66)
                        for key in ("reach", "forward", "coefficients", "inverse", "smoothed", "floored"):
                            self.assertEqual(len(row["trace"][key]), 1024)
                        self.assertEqual(len(row["expected"]["xy"]), 1025)
                        self.assertEqual(row["expected"]["xy"][0], row["expected"]["xy"][-1])
            for index, row in enumerate(records["products"]):
                case = (
                    f"product/{name}/control/{index}-{row['method']}-{row['compression']}"
                    if "polarity" not in row
                    else f"product/{name}/{row['polarity']}-{row['route']}-{row['method']}-{row['compression']}"
                )
                with self.subTest(case=case):
                    if row["method"] != 1:
                        self.assertIn(
                            (row["polarity"], row["route"], row["method"], row["compression"]), expected_products
                        )
        for name in {member["tables"] for member in manifest["preset_membership"].values()}:
            data = payloads[name]
            with self.subTest(case=f"tables/{name}"):
                self.assertEqual(len(data), 16 * 147456 * 4)
                self.assertTrue(all(not any(data[offset::16]) for offset in range(12, 16)))
        for name in {member["primitives"] for member in manifest["preset_membership"].values()}:
            with self.subTest(case=f"primitives/{name}"):
                self.assertEqual(len(payloads[name]), 519536 * 32)

    test_provenance_membership_bits_and_sharing.required_subtests = True


if __name__ == "__main__":
    unittest.main()
