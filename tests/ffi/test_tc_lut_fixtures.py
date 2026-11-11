import hashlib
import json
from pathlib import Path
import unittest


FIXTURES = Path(__file__).parent / "fixtures/tc_lut"


class TcLutFixtures(unittest.TestCase):
    def test_native_provenance_complete_membership_and_frozen_bytes(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        payloads = {name: (FIXTURES / name).read_bytes() for name in manifest["sha256"]}
        records_by_name = {
            membership["records"]: json.loads(payloads[membership["records"]])
            for membership in manifest["preset_membership"].values()
        }
        self.subtest_cases = tuple(
            [f"preset/{preset}" for preset in manifest["preset_membership"]]
            + [f"sha256/{name}" for name in manifest["sha256"]]
            + [f"table/{name}" for name in {m["tables"] for m in manifest["preset_membership"].values()}]
            + [
                f"leaf/{name}/{index}"
                for name, records in records_by_name.items()
                for index, _ in enumerate(records["leaves"]["cases"])
            ]
            + [
                f"product/{name}/{row['polarity']}-{row['route']}-{row['method']}-{row['variant']}"
                for name, records in records_by_name.items()
                for row in records["products"]
            ]
        )
        self.assertEqual(manifest["parent"], "5b134fb1cb731e16db85159e37c70eda38122874")
        presets = {"linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"}
        self.assertEqual(set(manifest["preset_membership"]), presets)
        for name, expected in manifest["sha256"].items():
            with self.subTest(case=f"sha256/{name}"):
                self.assertEqual(hashlib.sha256(payloads[name]).hexdigest(), expected)
        for preset, membership in manifest["preset_membership"].items():
            with self.subTest(case=f"preset/{preset}"):
                self.assertEqual(
                    manifest["sha256"][membership["tables"]], manifest["provenance"][preset]["table_sha256"]
                )
        for name in {member["tables"] for member in manifest["preset_membership"].values()}:
            data = payloads[name]
            with self.subTest(case=f"table/{name}"):
                self.assertEqual(len(data), 9 * 147456 * 4)
                self.assertTrue(all(not any(data[offset::16]) for offset in range(12, 16)))
        for name in {member["records"] for member in manifest["preset_membership"].values()}:
            records = records_by_name[name]
            self.assertEqual(len(records["leaves"]["cases"]), 23)
            self.assertEqual(len(records["products"]), 36)
            self.assertEqual(
                {(row["polarity"], row["route"], row["method"], row["variant"]) for row in records["products"]},
                {(p, r, m, v) for p in range(2) for r in range(2) for m in range(3) for v in range(3)},
            )
            for index, row in enumerate(records["leaves"]["cases"]):
                with self.subTest(case=f"leaf/{name}/{index}"):
                    self.assertIsInstance(row, dict)
            self.assertTrue(
                any(
                    sample["classification"].startswith("native_undefined")
                    for row in records["leaves"]["cases"]
                    if row["built"]
                    for sample in row["samples"]
                )
            )
            for row in records["products"]:
                case = f"product/{name}/{row['polarity']}-{row['route']}-{row['method']}-{row['variant']}"
                with self.subTest(case=case):
                    self.assertEqual((row["expected"]["tc_table"] is None), row["method"] == 1)

    test_native_provenance_complete_membership_and_frozen_bytes.required_subtests = True


if __name__ == "__main__":
    unittest.main()
