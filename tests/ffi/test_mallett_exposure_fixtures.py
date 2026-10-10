"""Integrity and membership of independently frozen A7 characterization."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures/exposure"


class MallettFixtureTests(unittest.TestCase):
    def test_independent_records_and_preset_sharing(self) -> None:
        manifest = json.loads((FIXTURES / "mallett-manifest.json").read_bytes())
        self.assertEqual(set(manifest["presets"]), {
            "linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release",
        })
        referenced = set()
        for preset, selected in manifest["presets"].items():
            with self.subTest(preset=preset):
                referenced.update(selected.values())
                numerical = json.loads((FIXTURES / selected["numerical"]).read_bytes())
                products = json.loads((FIXTURES / selected["products"]).read_bytes())
                self.assertEqual(len(numerical["focused"]), 96)
                self.assertEqual(len(numerical["normalizations"]), 12)
                self.assertEqual(len(numerical["sources"]), 6736)
                self.assertEqual(len(numerical["references"]), 30)
                self.assertEqual(len(products["products"]), 56)
                self.assertEqual(len(products["metrics"]), 56)
                self.assertEqual(len(products["gpu"]), 12)
                for row in numerical["focused"]:
                    self.assertEqual(len(row["input"]["basis"]), 81)
                    self.assertEqual(len(row["input"]["sensitivity"]), 81)
                    self.assertEqual(len(row["input"]["illuminant"]), 81)
                    self.assertEqual(len(row["expected"]["raw_bgr"]), 3)
        self.assertEqual(referenced, set(manifest["files"]))
        for name, record in manifest["files"].items():
            contents = (FIXTURES / name).read_bytes()
            self.assertEqual(hashlib.sha256(contents).hexdigest(), record["sha256"])
            self.assertEqual(len(contents), record["bytes"])

    def test_corrected_failure_retains_native_provenance(self) -> None:
        contract = json.loads((FIXTURES / "mallett-failure-contract.json").read_bytes())
        self.assertEqual(contract["native_outcome_manifest_sha256"], hashlib.sha256((FIXTURES / "mallett-manifest.json").read_bytes()).hexdigest())
        self.assertEqual(contract["diagnostic"], "MissingRequiredResource component=mallett_midgray requirement=81x3_basis")
        self.assertEqual(contract["selected_basis"], {"available": True, "rows": 81, "cols": 3, "values": 243})
        self.assertIn("shape/data-only", contract["synthetic_reference"])


if __name__ == "__main__":
    unittest.main()
