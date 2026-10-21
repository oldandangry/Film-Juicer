"""Frozen logical captures share storage without sharing mutable patch state."""

import hashlib
import json
from pathlib import Path
import unittest

from compact_exposure_fixtures import compact_expected_blocks, expand_expected_blocks


FIXTURES = Path(__file__).parent / "fixtures"
PRESETS = ("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release")


class SharedFixtureTests(unittest.TestCase):
    def test_preflash_native_integrity_inputs_and_caller_membership(self):
        root = Path(__file__).parents[1] / "print/preflash-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        inputs = json.loads((root / "cases.json").read_bytes())["cases"]
        self.assertEqual(len(inputs), 19)
        for name, record in manifest["files"].items():
            data = (root / name).read_bytes()
            self.assertEqual(len(data), record["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"], name)
        for preset in PRESETS:
            capture = json.loads((root / manifest["presets"][preset]["native"]).read_bytes())
            self.assertEqual(capture["preset"], preset)
            self.assertEqual(len(capture["cases"]), 19)
            self.assertEqual(len(capture["products"]), 72)
            for source, row in zip(inputs, capture["cases"], strict=True):
                self.assertEqual({key: row[key] for key in source}, source)
                for key, extent in (("film_base_density", 81), ("filtered_illuminant", 81), ("print_sensitivity_cmy", 243), ("raw", 3)):
                    self.assertEqual(len(row[key]), extent)
                    self.assertTrue(all(type(word) is int and 0 <= word <= 0xffffffff for word in row[key]))
                self.assertEqual(row["success"], row["id"] not in manifest["failure_channels"])
                self.assertEqual(row["diagnostic"], "" if row["success"] else "MalformedRequiredResource phase=4B field=preflash_raw")
            self.assertEqual({(r["route"], r["mode"], r["variant"]) for r in capture["products"]},
                             {(route, mode, variant) for route in range(4) for mode in range(3) for variant in range(6)})
            for row in capture["products"]:
                if row["route"] % 2:
                    self.assertEqual(len(row["descriptors"]), 5)
                    self.assertEqual(len(row["raw"]), 3)
                    self.assertEqual(row["descriptors"][3] == 0, row["variant"] == 0)
                else:
                    self.assertNotIn("raw", row)

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

    def test_film_input_compound_capture_integrity_and_membership(self):
        root = Path(__file__).parents[1] / "film/input-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(manifest["cases"], {"base": 96, "controls": 44})
        self.assertEqual(set(manifest["presets"].values()), set(manifest["payloads"]))
        for name, record in manifest["payloads"].items():
            data = (root / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
            rows = json.loads(data)["rows"]
            self.assertEqual(len(rows), 140)
            base = {(row["route"], row["space"], row["method"], row["decode"]) for row in rows[:96]}
            self.assertEqual(base, {(route, space, method, decode) for route in range(4)
                                   for space in range(4) for method in range(3) for decode in range(2)})
            self.assertEqual([row["control"] for row in rows[96:]], list(range(44)))
            for row in rows:
                raw = row["raw"]
                self.assertEqual([len(raw[field]) for field in ("rgb_to_xyz", "xyz_adapt", "xyz_to_linear_srgb")], [9, 9, 9])
                self.assertEqual(len(raw["sensitivity"]), 81)
                self.assertTrue(all(len(sample) == 3 for sample in raw["sensitivity"]))
                self.assertEqual(len(raw["surface"]), 3)
                self.assertTrue(all(len(channel) == 15 for channel in raw["surface"]))
                self.assertNotEqual(raw["hash"], 0)
                self.assertNotEqual(raw["sensitivity_hash"], 0)
                self.assertEqual(raw["tc_hash"] == 0, row["method"] == 1)

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
