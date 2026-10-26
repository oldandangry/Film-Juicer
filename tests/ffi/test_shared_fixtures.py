"""Frozen logical captures share storage without sharing mutable patch state."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from compact_exposure_fixtures import compact_expected_blocks, expand_expected_blocks
from shared_fixture import load_shared_fixture, original_capture_bytes


FIXTURES = Path(__file__).parent / "fixtures"
PRESETS = ("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release")


class SharedFixtureTests(unittest.TestCase):
    def test_dir_boundary_exact_native_products_keys_and_production_contracts(self):
        root = Path(__file__).parents[1] / "dir/boundary-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        for name, checksum in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), checksum)
        baseline = None
        ordinary = json.loads((root / manifest["checked"]).read_bytes())["ordinary"]
        for preset in PRESETS:
            payload = (root / manifest["presets"][preset]["native"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest["presets"][preset]["capture"]["sha256"])
            captured = json.loads(payload)
            self.assertEqual(captured["preset"], preset)
            self.assertEqual([len(captured[key]) for key in ("cases", "primitives")], [114, 56])
            if baseline is None:
                baseline = (captured["cases"], captured["primitives"])
            self.assertEqual((captured["cases"], captured["primitives"]), baseline)
            for row in captured["cases"]:
                source = row["input"]
                c = source["component"]
                encoded = c["version"].to_bytes(4, "little")
                for value in [c["feedforward"], *c["feedback"], c["denominator"], c["numerator"], *c["numerators"]]:
                    encoded += value.to_bytes(8, "little")
                for value in [source["width"], source["height"]]:
                    encoded += value.to_bytes(4, "little", signed=True)
                for value in [c["accuracy"], c["tolerance"]]:
                    encoded += value.to_bytes(8, "little")
                self.assertEqual(len(encoded), 100)
                self.assertEqual(list(encoded), row["key_bytes"])
                if ordinary[source["name"]][0] != 2:
                    identity = 0xcbf29ce484222325
                    for byte in encoded:
                        identity = ((identity ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
                    self.assertEqual(identity, row["key"])
                if row["success"]:
                    for name in ("horizontal", "vertical"):
                        axis = row[name]
                        self.assertEqual(len(axis["weights"]), 3 * axis["length"])
                        self.assertEqual(len(axis["terminal"]), 9)
                        self.assertEqual(axis["terminal"][axis["terminal_size"] ** 2:], [0] * (9 - axis["terminal_size"] ** 2))
        self.assertEqual(len(manifest["production_cases"]), 171)
        self.assertEqual({row["case"] for row in manifest["production_cases"]}, set(manifest["production_inputs"]))
        self.assertEqual(len(manifest["production_inputs"]), 171)
        raw = [row for row in manifest["production_cases"] if row["case"].startswith("raw/")]
        self.assertEqual(len(raw), 51)
        for operation in ("describe", "acquire", "view"):
            for mode in (11, 12, 13, 14, 15):
                row = next(row for row in raw if row["case"] == f"raw/{operation}/{mode}")
                message = "spatial DIR reflected-boundary arguments are invalid"
                self.assertEqual(row["diagnostic_length"], min(len(message), max(0, manifest["production_inputs"][row["case"]]["capacity"] - 1)))
                self.assertEqual(row["diagnostic"], message[:row["diagnostic_length"]])
                self.assertEqual(row["envelope"], [2, 0, 0])

    def test_dir_frame_exact_native_bytes_fields_and_production_membership(self):
        root = Path(__file__).parents[1] / "dir/frame-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(len(set(manifest["payloads"].values())), 4)
        for name, checksum in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), checksum)
        baseline = None
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            self.assertEqual(captured["preset"], preset)
            self.assertEqual([len(captured[key]) for key in ("cases", "products")], [74, 24])
            rows = captured["cases"] + captured["products"]
            if baseline is None:
                baseline = rows
            self.assertEqual(rows, baseline)
            for row in rows:
                native = row["native"]
                self.assertEqual(len(native["components"]), 4)
                self.assertEqual(len(native["roles"]), 4)
                self.assertEqual(len(native["target_roles"]), 4)
                for component in native["components"]:
                    self.assertEqual(len(component["feedback"]), 3)
                    self.assertEqual(len(component["numerators"]), 3)
        expected = manifest["production_cases"]
        self.assertEqual(len(expected), 121)
        self.assertEqual(len({row["case"] for row in expected}), 121)
        self.assertEqual({row["case"] for row in expected}, set(manifest["production_inputs"]))
        for row in expected:
            self.assertEqual(len(manifest["production_inputs"][row["case"]]["words"]), 14)
            self.assertTrue(row["output"] is None or len(row["output"]) == 79)
            self.assertTrue(row["failure"] is None or len(row["failure"]) == 82)

    def test_diffusion_frame_original_bytes_and_complete_native_contexts(self):
        root = Path(__file__).parents[1] / "diffusion/frame-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(len(set(manifest["contexts"].values())), 4)
        for name, checksum in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), checksum)
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            provenance = manifest["provenance"][preset]
            self.assertEqual(hashlib.sha256(original_capture_bytes(captured, provenance["line_ending"])).hexdigest(), provenance["original_capture_sha256"])
            self.assertEqual([len(captured[k]) for k in ("samples", "frames", "products", "primitives")], [94, 35, 92, 8])
            edges = load_shared_fixture(root / manifest["presets"][preset]["edges"])
            edge_provenance = manifest["edge_provenance"][preset]
            self.assertEqual(hashlib.sha256(original_capture_bytes(edges, edge_provenance["line_ending"])).hexdigest(), edge_provenance["original_capture_sha256"])
            self.assertEqual(edge_provenance["producer_sha256"], provenance["producer_sha256"])
            self.assertEqual([len(edges[k]) for k in ("samples", "frames", "products", "primitives")], [2, 0, 0, 0])
            samples = captured["samples"] + edges["samples"] + [sample for row in captured["products"] for sample in row["samples"]]
            self.assertEqual(len(samples), 174)
            for row in samples:
                if row["success"]:
                    self.assertEqual(set(row["components"]), {"lambda", "red", "green", "blue"})
                    self.assertTrue(all(len(words) == 9 for words in row["components"].values()))

    def test_halation_frame_original_bytes_and_complete_native_contexts(self):
        root = Path(__file__).parents[1] / "scatter_halation/frame-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(len(set(manifest["contexts"].values())), 4)
        for name, checksum in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), checksum)
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            self.assertEqual(captured["preset"], preset)
            provenance = manifest["provenance"][preset]
            self.assertEqual(hashlib.sha256(original_capture_bytes(captured, provenance["line_ending"])).hexdigest(), provenance["original_capture_sha256"])
            self.assertEqual(len(captured["cases"]), 52)
            self.assertEqual(len(captured["products"]), 20)
            for row in captured["cases"] + captured["products"]:
                if row["present"]:
                    self.assertEqual(len(row["descriptor"]["channels"]), 3)
                    for channel in row["descriptor"]["channels"]:
                        for gaussian in [channel["core"], *channel["tail"], *channel["bounce"]]:
                            self.assertEqual(len(gaussian["weights"]), 19)
                            self.assertEqual(len(gaussian["yvv"]), 4)

    def test_frame_geometry_full_fields_and_distinct_native_contexts(self):
        root = Path(__file__).parents[1] / "frame/geometry-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(len(set(manifest["contexts"].values())), 4)
        for name, expected in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            self.assertEqual(captured["preset"], preset)
            self.assertEqual(len(captured["domains"]), manifest["domain_count"])
            self.assertEqual(len(captured["geometry"]), manifest["geometry_count"])
            adopted = (root / manifest["payloads"][manifest["presets"][preset]["native"]]).read_bytes()
            self.assertNotIn(b"\r", adopted)
            original = adopted.replace(b"\n", b"\r\n") if captured["native_payload_line_ending"] == "CRLF" else adopted
            self.assertEqual(hashlib.sha256(original).hexdigest(), captured["producer"]["files"]["payload.json"]["sha256"])
            self.assertTrue(captured["commands"])
            self.assertEqual(set(captured["producer"]["files"]), {"JuicerAdapterTraceProbe.exe" if "windows" in preset else "JuicerAdapterTraceProbe", "JuicerEffect.cpp.txt", "inputs.json", "payload.json", "context.json"})
            for row in captured["domains"]:
                self.assertEqual(len(row["bounds"]), 4)
                if not row["range"]:
                    self.assertEqual(len(row["domain"]), 4)
            for row in captured["geometry"]:
                self.assertEqual(len(row["domain"]), 4)
                self.assertIsInstance(row["pixel_bits"], int)
                self.assertIsInstance(row["diffusion_bits"], int)

    def test_gaussian_complete_arrays_and_distinct_native_contexts(self):
        root = Path(__file__).parents[1] / "optics/gaussian-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(len(set(manifest["contexts"].values())), 4)
        for name, expected in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            self.assertEqual(captured["preset"], preset)
            self.assertEqual(len(captured["cases"]), 188)
            self.assertEqual(sum(row["count"] for row in captured["cases"]), 22895)
            self.assertEqual([row["name"] for row in captured["cases"]], manifest["production_cases"][:188])
            for row in captured["cases"]:
                self.assertEqual(len(row["weights"]), row["count"])
                self.assertTrue(all(isinstance(word, int) and 0 <= word <= 0xffffffff for word in row["weights"]))
            self.assertTrue(any("fir-dir" in row["name"] and row["count"] > 0 for row in captured["cases"]))
        self.assertEqual(len(manifest["production_cases"]), 201)
        self.assertEqual(len(set(manifest["production_cases"])), 201)

    def test_scanner_post_complete_frozen_contexts(self):
        root = Path(__file__).parents[1] / "scanner/post-effects-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(set(manifest["payloads"].values()), {"native.json"})
        self.assertEqual(len(set(manifest["contexts"].values())), 4)
        for name, expected in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        payloads = []
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            self.assertEqual(captured["preset"], preset)
            self.assertEqual(len(captured["cases"]), 400)
            for row in captured["cases"]:
                self.assertEqual(len(row["input"]["values"]), 6)
                self.assertEqual(len(row["native"]["descriptor"]["radii"]), 3)
                self.assertEqual(row["native"]["descriptor"]["schema"], 2)
            payloads.append(captured["cases"])
        self.assertTrue(all(value == payloads[0] for value in payloads))
        self.assertEqual(len(manifest["production_cases"]), 410)
        self.assertEqual(len({row["case"] for row in manifest["production_cases"]}), 410)

    def test_scanner_descriptor_complete_frozen_contexts(self):
        root = Path(__file__).parents[1] / "scanner/descriptor-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(set(manifest["payloads"].values()), {"native.json"})
        self.assertEqual(len(set(manifest["contexts"].values())), 4)
        for name, expected in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        payloads = []
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            self.assertEqual(captured["preset"], preset)
            self.assertEqual(len(captured["cases"]), 520)
            self.assertEqual(len(captured["products"]), 4)
            self.assertEqual({row["route"] for row in captured["products"]}, {0, 1, 2, 3})
            self.assertEqual(len({row["id"] for row in captured["cases"]}), 520)
            for row in captured["cases"] + captured["products"]:
                self.assertEqual(len(row["input"]["channel_density_cmy"]), 243)
                self.assertEqual(len(row["input"]["base_density"]), 81)
                self.assertEqual(len(row["native"]["descriptor"]), 10)
            payloads.append((captured["cases"], captured["products"]))
        self.assertTrue(all(value == payloads[0] for value in payloads))
        results = manifest["production_cases"] + manifest["production_malformed"]
        self.assertEqual(len(results), 41)
        self.assertEqual(len({row["case"] for row in results}), 41)

    def test_scanner_lut_original_capture_integrity(self):
        root = Path(__file__).parents[1] / "scanner/lut-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        logical_names = {preset + ".json" for preset in PRESETS}
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        for field in ("payloads", "contexts", "original_line_endings"):
            self.assertEqual(set(manifest[field]), logical_names)
        self.assertEqual(set(manifest["payloads"].values()), {"native.json"})
        physical = set(manifest["payloads"].values()) | set(manifest["contexts"].values())
        self.assertEqual(physical, set(manifest["storage_files"]))
        for name, expected in manifest["storage_files"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected, name)
        for preset in PRESETS:
            with self.subTest(preset=preset):
                entry = manifest["presets"][preset]
                logical = entry["native"]
                self.assertEqual(logical, preset + ".json")
                context = json.loads((root / manifest["contexts"][logical]).read_bytes())
                self.assertEqual(set(context), {"preset", "primitives"})
                capture = load_shared_fixture(root / logical)
                self.assertEqual(capture["preset"], preset)
                self.assertEqual({key: len(capture[key]) for key in ("cases", "products", "slopes", "primitives")},
                                 {"cases": 30, "products": 4, "slopes": 15, "primitives": 63})
                raw = original_capture_bytes(capture, manifest["original_line_endings"][logical])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"],
                                 "every original ordered record, integer word and capture field must reconstruct exactly")

    def test_shared_context_is_required_disjoint_and_independently_owned(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"payloads": {"debug.json": "native.json", "release.json": "native.json"},
                        "contexts": {"debug.json": "debug-context.json"}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            words = [0, 0x80000000, 0x7FC00123, 0xFFFFFFFF]
            (root / "native.json").write_text(json.dumps({"words": words}))
            (root / "debug-context.json").write_text(json.dumps({"preset": "debug", "primitives": [1]}))
            first = load_shared_fixture(root / "debug.json")
            self.assertEqual(first, {"words": words, "preset": "debug", "primitives": [1]})
            first["words"][0] = 17
            self.assertEqual(load_shared_fixture(root / "debug.json")["words"], words)
            with self.assertRaises(KeyError):
                load_shared_fixture(root / "release.json")
            (root / "debug-context.json").write_text(json.dumps({"words": words}))
            with self.assertRaisesRegex(ValueError, "context overlaps payload"):
                load_shared_fixture(root / "debug.json")

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
