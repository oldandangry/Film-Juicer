import hashlib
import json
from pathlib import Path
import struct
import unittest


FIXTURES = Path(__file__).parent / "fixtures/tc_lut"


class TcLutFixtures(unittest.TestCase):
    def test_native_provenance_complete_membership_and_frozen_bytes(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        self.assertEqual(manifest["parent"], "5b134fb1cb731e16db85159e37c70eda38122874")
        presets = {"linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"}
        self.assertEqual(set(manifest["preset_membership"]), presets)
        for name, expected in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), expected)
        for preset, membership in manifest["preset_membership"].items():
            records = json.loads((FIXTURES / membership["records"]).read_text())
            data = (FIXTURES / membership["tables"]).read_bytes()
            self.assertEqual(len(data), 9 * 147456 * 4)
            self.assertEqual(hashlib.sha256(data).hexdigest(), manifest["provenance"][preset]["table_sha256"])
            words = [value[0] for value in struct.iter_unpack("<I", data)]
            self.assertTrue(all(value == 0 for value in words[3::4]))
            self.assertEqual(len(records["leaves"]["cases"]), 23)
            self.assertEqual(len(records["products"]), 36)
            self.assertEqual({(row["polarity"], row["route"], row["method"], row["variant"]) for row in records["products"]},
                             {(p, r, m, v) for p in range(2) for r in range(2) for m in range(3) for v in range(3)})
            self.assertTrue(all((row["expected"]["tc_table"] is None) == (row["method"] == 1) for row in records["products"]))
            self.assertTrue(any(sample["classification"].startswith("native_undefined")
                                for row in records["leaves"]["cases"] if row["built"] for sample in row["samples"]))


if __name__ == "__main__":
    unittest.main()
