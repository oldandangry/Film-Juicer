"""Immutable native A8 capture membership and complete byte sharing."""
import hashlib
import json
from pathlib import Path
import struct
import unittest

FIXTURES = Path(__file__).parent / "fixtures/input_hull"


class InputHullFixtures(unittest.TestCase):
    def test_provenance_membership_bits_and_sharing(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        self.assertEqual(manifest["parent"], "e9b4abbb8b146ffdd0f4fbc5f8c61e7e87134c2a")
        self.assertEqual(set(manifest["preset_membership"]), {"linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"})
        for name, digest in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), digest)
        for preset, member in manifest["preset_membership"].items():
            for kind, name in member.items():
                self.assertEqual(manifest["sha256"][name], manifest["provenance"][preset][kind]["native_sha256"])
            records = json.loads((FIXTURES / member["records"]).read_text())
            self.assertEqual(len(records["math"]), 14)
            self.assertEqual(len(records["products"]), 18)
            self.assertEqual(len(records["weights"]), 1024)
            self.assertEqual({(row["polarity"], row["route"], row["method"], row["compression"]) for row in records["products"] if row["method"] != 1}, {(p, r, m, c) for p in range(2) for r in range(2) for m in (0, 2) for c in range(2)})
            for row in records["math"]:
                if row["built"]:
                    self.assertEqual(len(row["trace"]["locus"]), 66)
                    for key in ("reach", "forward", "coefficients", "inverse", "smoothed", "floored"):
                        self.assertEqual(len(row["trace"][key]), 1024)
                    self.assertEqual(len(row["expected"]["xy"]), 1025)
                    self.assertEqual(row["expected"]["xy"][0], row["expected"]["xy"][-1])
            data = (FIXTURES / member["tables"]).read_bytes()
            self.assertEqual(len(data), 16 * 147456 * 4)
            self.assertTrue(all(struct.unpack_from("<I", data, i)[0] == 0 for i in range(12, len(data), 16)))
            self.assertEqual((FIXTURES / member["primitives"]).stat().st_size, 519536 * 32)


if __name__ == "__main__":
    unittest.main()
