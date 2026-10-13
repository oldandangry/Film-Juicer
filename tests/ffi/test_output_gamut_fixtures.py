"""Immutable native A9 captures, raw identity widths and complete byte sharing."""
import hashlib
import json
from pathlib import Path
import struct
import unittest

FIXTURES = Path(__file__).parent / "fixtures/output_gamut"


def fnv(data):
    value = 0xcbf29ce484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    return value


class OutputGamutFixtures(unittest.TestCase):
    def test_provenance_membership_and_every_identity_stream(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        self.assertEqual(manifest["parent"], "d033f27cb3d1493adebcbc6c3d032f4bf6cb1666")
        self.assertEqual(set(manifest["preset_membership"]), {"linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"})
        for name, entry in manifest["files"].items():
            data = (FIXTURES / name).read_bytes()
            self.assertEqual(len(data), entry["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
        for member in manifest["preset_membership"].values():
            records = json.loads((FIXTURES / member["records"]).read_text())
            table = (FIXTURES / member["tables"]).read_bytes()
            self.assertEqual({row["tag"] for row in records["rows"] if row["variant"] == 0}, set(range(9)))
            self.assertEqual(len(records["directions"]), 720)
            self.assertEqual(len(records["oklab"]), 4)
            self.assertEqual(len(table), len(records["rows"]) * 46080 * 4)
            for row in records["rows"]:
                self.assertEqual(len(row["transform_stream"]), 305)
                self.assertEqual(len(row["contract_stream"]), 40)
                self.assertEqual(row["transform_stream"][4], row["tag"])
                if row["transform_ok"]:
                    self.assertEqual(fnv(row["transform_stream"]), row["transform"]["hash"])
                    self.assertEqual(fnv(row["contract_stream"]), row["transform"]["contract_hash"])
                if row["boundary_ok"]:
                    start = row["table_offset"]
                    self.assertEqual(fnv(struct.pack("<Q", row["transform"]["contract_hash"]) + table[start:start + 184320]), row["boundary_hash"])
            extra = json.loads((FIXTURES / member["extra"]).read_text())
            self.assertEqual(len(extra["primitives"]), 5)
            self.assertEqual(len(extra["cube"]), 8)
            self.assertEqual(len(extra["rows"]), 6)


if __name__ == "__main__":
    unittest.main()
