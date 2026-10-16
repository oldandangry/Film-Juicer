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
        self.assertEqual(referenced, set(manifest["files"]))
        for name, record in manifest["files"].items():
            contents = (FIXTURES / name).read_bytes()
            self.assertEqual(hashlib.sha256(contents).hexdigest(), record["sha256"])
            self.assertEqual(len(contents), record["bytes"])
            value = json.loads(contents)
            if name in {selected["numerical"] for selected in manifest["presets"].values()}:
                self.assertEqual(len(value["focused"]), 96)
                self.assertEqual(len(value["normalizations"]), 12)
                self.assertEqual(len(value["sources"]), 6736)
                self.assertEqual(len(value["references"]), 30)
                for row in value["focused"]:
                    self.assertEqual(len(row["input"]["basis"]), 81)
                    self.assertEqual(len(row["input"]["sensitivity"]), 81)
                    self.assertEqual(len(row["input"]["illuminant"]), 81)
                    self.assertEqual(len(row["expected"]["raw_bgr"]), 3)
            if name in {selected["products"] for selected in manifest["presets"].values()}:
                self.assertEqual(len(value["products"]), 56)
                self.assertEqual(len(value["metrics"]), 56)
                self.assertEqual(len(value["gpu"]), 12)

    def test_corrected_failure_retains_native_provenance(self) -> None:
        contract = json.loads((FIXTURES / "mallett-failure-contract.json").read_bytes())
        self.assertEqual(contract["native_outcome_manifest_sha256"], hashlib.sha256((FIXTURES / "mallett-manifest.json").read_bytes()).hexdigest())
        self.assertEqual(contract["diagnostic"], "MissingRequiredResource component=mallett_midgray requirement=81x3_basis")
        self.assertEqual(contract["selected_basis"], {"available": True, "rows": 81, "cols": 3, "values": 243})
        self.assertIn("shape/data-only", contract["synthetic_reference"])


if __name__ == "__main__":
    unittest.main()
