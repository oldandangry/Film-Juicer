"""Frozen logical captures share storage without sharing mutable patch state."""

import hashlib
import json
from pathlib import Path
import unittest

from compact_exposure_fixtures import compact_expected_blocks, expand_expected_blocks


FIXTURES = Path(__file__).parent / "fixtures"
PRESETS = ("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release")


class SharedFixtureTests(unittest.TestCase):
    def test_unique_payload_integrity_and_native_membership(self):
        families = (("color", "", "manifest.json", "recipes", 15),
                    ("color", "cat02-", "cat02-manifest.json", "recipes", 56),
                    ("color", "input-color-", "input-color-manifest.json", "recipes", 140),
                    ("illuminants", "construction-", "construction-manifest.json", "products", 270))
        for directory, prefix, name, group, count in families:
            root = FIXTURES / directory
            manifest = json.loads((root / name).read_bytes())
            self.assertEqual(set(manifest["payloads"]), {prefix + preset + ".json" for preset in PRESETS})
            physical = set(manifest["payloads"].values())
            referenced = set(physical)
            inputs = None
            for payload in physical:
                fixture = json.loads((root / payload).read_bytes())
                if "expected_blocks" in fixture:
                    referenced.add(fixture["expected_blocks"])
                    blocks = json.loads((root / fixture["expected_blocks"]).read_bytes())
                    fixture = expand_expected_blocks(fixture, blocks)
                self.assertEqual(len(fixture[group]), count)
                if prefix in ("cat02-", "input-color-"):
                    current = [(row["id"], row["controls"]) for row in fixture[group]]
                    if inputs is not None:
                        self.assertEqual(current, inputs, "all native captures retain identical ordered recipe inputs")
                    inputs = current
            if "preset_witnesses" in manifest:
                referenced.add(manifest["preset_witnesses"])
                witnesses = json.loads((root / manifest["preset_witnesses"]).read_bytes())
                self.assertEqual(set(witnesses), set(manifest["payloads"]))
                for logical, witness in witnesses.items():
                    self.assertIn(witness["foreign_capture"], manifest["payloads"])
                    self.assertNotEqual(witness["foreign_capture"], logical)
                    self.assertNotEqual(witness["expected"], witness["rejected"])
            self.assertEqual(referenced, set(manifest["storage_files"]))
            hashes = []
            for payload, expected in manifest["storage_files"].items():
                data = (root / payload).read_bytes()
                actual = hashlib.sha256(data).hexdigest()
                self.assertEqual(actual, expected, payload)
                hashes.append(actual)
            self.assertEqual(len(hashes), len(set(hashes)), "duplicate new payloads need logical aliases")

    def test_expanded_cases_are_independent_and_preserve_exact_words(self):
        words = [0, 0x80000000, 0x7FC00123, 0xFFFFFFFF]
        original = {"recipes": [{"expected": {"sensitivity": words}}, {"expected": {"sensitivity": words}}]}
        compact, blocks = compact_expected_blocks({"native": original}, "blocks.json")
        self.assertEqual(len(blocks["sensitivity"]), 1)
        expanded = expand_expected_blocks(compact["native"], blocks)
        self.assertEqual(expanded, original)
        expanded["recipes"][0]["expected"]["sensitivity"][0] = 17
        self.assertEqual(expanded["recipes"][1]["expected"]["sensitivity"], words)
        self.assertEqual(blocks["sensitivity"][0], words)
        compact["native"]["recipes"][0]["expected"]["sensitivity_record"] = -1
        with self.assertRaisesRegex(ValueError, "invalid expected block"):
            expand_expected_blocks(compact["native"], blocks)


if __name__ == "__main__":
    unittest.main()
