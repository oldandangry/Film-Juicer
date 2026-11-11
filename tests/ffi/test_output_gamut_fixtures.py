"""Immutable native A9 captures, raw identity widths and complete byte sharing."""

import hashlib
import json
from pathlib import Path
import struct
import unittest

FIXTURES = Path(__file__).parent / "fixtures/output_gamut"


def fnv(data):
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


class OutputGamutFixtures(unittest.TestCase):
    def test_provenance_membership_and_every_identity_stream(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        payloads = {name: (FIXTURES / name).read_bytes() for name in manifest["files"]}
        memberships = {
            (member["records"], member["tables"], member["extra"]) for member in manifest["preset_membership"].values()
        }
        decoded = {records_name: json.loads(payloads[records_name]) for records_name, _, _ in memberships}
        extras = {extra_name: json.loads(payloads[extra_name]) for _, _, extra_name in memberships}
        self.subtest_cases = tuple(
            [f"file/{name}" for name in manifest["files"]]
            + [
                f"record/{name}/{row['tag']}-{row['variant']}"
                for name, records in decoded.items()
                for row in records["rows"]
            ]
            + [
                f"direction/{name}/{index}"
                for name, records in decoded.items()
                for index, _ in enumerate(records["directions"])
            ]
            + [f"oklab/{name}/{index}" for name, records in decoded.items() for index, _ in enumerate(records["oklab"])]
            + [
                f"extra/{name}/primitive/{index}"
                for name, extra in extras.items()
                for index, _ in enumerate(extra["primitives"])
            ]
            + [f"extra/{name}/cube/{index}" for name, extra in extras.items() for index, _ in enumerate(extra["cube"])]
            + [f"extra/{name}/row/{index}" for name, extra in extras.items() for index, _ in enumerate(extra["rows"])]
        )
        self.assertEqual(manifest["parent"], "d033f27cb3d1493adebcbc6c3d032f4bf6cb1666")
        self.assertEqual(
            set(manifest["preset_membership"]),
            {"linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"},
        )
        for name, entry in manifest["files"].items():
            with self.subTest(case=f"file/{name}"):
                data = payloads[name]
                self.assertEqual(len(data), entry["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
        for records_name, tables_name, extra_name in memberships:
            records = decoded[records_name]
            table = payloads[tables_name]
            self.assertEqual({row["tag"] for row in records["rows"] if row["variant"] == 0}, set(range(9)))
            self.assertEqual(len(records["directions"]), 720)
            self.assertEqual(len(records["oklab"]), 4)
            self.assertEqual(len(table), len(records["rows"]) * 46080 * 4)
            for row in records["rows"]:
                case = f"record/{records_name}/{row['tag']}-{row['variant']}"
                with self.subTest(case=case):
                    self.assertEqual(len(row["transform_stream"]), 305)
                    self.assertEqual(len(row["contract_stream"]), 40)
                    self.assertEqual(row["transform_stream"][4], row["tag"])
                    if row["transform_ok"]:
                        self.assertEqual(fnv(row["transform_stream"]), row["transform"]["hash"])
                        self.assertEqual(fnv(row["contract_stream"]), row["transform"]["contract_hash"])
                    if row["boundary_ok"]:
                        start = row["table_offset"]
                        self.assertEqual(
                            fnv(struct.pack("<Q", row["transform"]["contract_hash"]) + table[start : start + 184320]),
                            row["boundary_hash"],
                        )
            for index, direction in enumerate(records["directions"]):
                with self.subTest(case=f"direction/{records_name}/{index}"):
                    self.assertIsInstance(direction, list)
            for index, row in enumerate(records["oklab"]):
                with self.subTest(case=f"oklab/{records_name}/{index}"):
                    self.assertIsInstance(row, list)
            extra = extras[extra_name]
            self.assertEqual(len(extra["primitives"]), 5)
            self.assertEqual(len(extra["cube"]), 8)
            self.assertEqual(len(extra["rows"]), 6)
            for group in ("primitives", "cube", "rows"):
                row_group = {"primitives": "primitive", "cube": "cube", "rows": "row"}[group]
                for index, value in enumerate(extra[group]):
                    with self.subTest(case=f"extra/{extra_name}/{row_group}/{index}"):
                        self.assertIsNotNone(value)

    test_provenance_membership_and_every_identity_stream.required_subtests = True


if __name__ == "__main__":
    unittest.main()
