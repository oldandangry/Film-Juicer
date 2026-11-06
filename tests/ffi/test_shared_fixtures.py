"""Frozen logical captures share storage without sharing mutable patch state."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from compact_exposure_fixtures import compact_expected_blocks, expand_expected_blocks
from shared_fixture import load_shared_fixture, original_capture_bytes
from effects_frame_probe_contract import (
    EFFECTS_FRAME_CASE_IDS, EFFECTS_FRAME_CASE_COUNT,
    EFFECTS_FRAME_FAILURE_WORD_COUNT, EFFECTS_FRAME_INPUT_WORD_COUNT,
    EFFECTS_FRAME_OUTPUT_WORD_COUNT,
)


FIXTURES = Path(__file__).parent / "fixtures"
PRESETS = ("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release")
PREPARED_STATE_CASE_IDS = (
    "NegativeDirectHanatos", "NegativeDirectMallett", "NegativeDirectArctic",
    "NegativePrintHanatos", "NegativePrintMallett", "NegativePrintArctic",
    "PositiveDirectHanatos", "PositiveDirectMallett", "PositiveDirectArctic",
    "PositivePrintHanatos", "PositivePrintMallett", "PositivePrintArctic",
    "DirectIgnoresPrintControls",
    "PreflashAndNormalizationModes[norm=0,compensate=0]",
    "PreflashAndNormalizationModes[norm=0,compensate=1]",
    "PreflashAndNormalizationModes[norm=1,compensate=0]",
    "PreflashAndNormalizationModes[norm=1,compensate=1]",
    "GammaAndGrainLayers", "PhysicalDefectsAndWeave", "CameraAndEnlargerOptics",
    "DirAndCompressionOff", "SignedZeroGlare[+0]", "SignedZeroGlare[-0]",
    "CorrectionsAndScannerControls[print]", "CorrectionsAndScannerControls[direct]",
    "EncodingOnlyReusesMediumIdentities[encoded]", "EncodingOnlyReusesMediumIdentities[linear]",
    "SnapshotIntegerBooleansPreserveNonzeroCasts[direct]",
    "SnapshotIntegerBooleansPreserveNonzeroCasts[print]",
    "DisabledFamiliesExcludeUnusedControls", "GrainDebugSuppressesPhysicalEffects",
)


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _is_git_revision(value):
    return isinstance(value, str) and len(value) == 40 and all(char in "0123456789abcdef" for char in value)


def _safe_relative_path(value):
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return ("\\" not in value and not path.is_absolute() and not PureWindowsPath(value).drive
            and all(part not in ("", ".", "..") for part in value.split("/"))
            and all(part not in ("", ".", "..") for part in path.parts))


class SharedFixtureTests(unittest.TestCase):
    def test_production_header_generation_requires_no_native_execution(self):
        import production_contract

        self.subtest_cases = ("linux-release", "windows-clang-release")
        for preset in self.subtest_cases:
            with self.subTest(case=preset), TemporaryDirectory(prefix="production header ü ") as temporary:
                directory = Path(temporary)
                with patch.object(production_contract, "execute", side_effect=AssertionError("native execution")), \
                     patch.object(production_contract, "arguments", None), \
                     patch.object(production_contract, "build", None):
                    production_contract.main(["--preset", preset, "--artifacts", str(directory),
                                              "--generate-expected-header"])
                header = (directory / "production-film-expected.h").read_text()
                fixture = json.loads((Path(__file__).parents[1] / "film/fixtures" / (preset + ".json")).read_text())
                profile = fixture["profiles"]["kodak_portra_400"]
                self.assertIn("#define FILM_N " + str(len(profile["axis"])) + "U\n", header)
                self.assertIn("#define FILM_NORMALIZED_HASH UINT64_C(" + str(profile["normalized_hash"]) + ")\n", header)
                self.assertIn("static const uint64_t correction_hashes[28]", header)

    test_production_header_generation_requires_no_native_execution.required_subtests = True

    def _assert_prepared_projection_identity(self, payload, preset, entry):
        self.assertEqual(set(payload), {"schema_version", "preset", "producer", "toolchain", "configuration", "inputs", "cases"})
        self.assertIs(type(payload["schema_version"]), int)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["preset"], preset)
        self.assertEqual(payload["producer"], {
            "authority": "Film-Juicer native composition",
            "target": "JuicerPreparedStateTests",
            "manifest_entry": preset,
        })
        producer_executable = PureWindowsPath(entry["producer"]["executable"]).name.removesuffix(".exe")
        self.assertEqual(producer_executable, payload["producer"]["target"])
        for key in ("toolchain", "configuration", "inputs"):
            self.assertEqual(payload[key], entry[key], key)

    def _assert_prepared_projection_cases(self, cases, preset=None):
        self.assertIs(type(cases), list)
        self.assertEqual([row["case_id"] for row in cases], list(PREPARED_STATE_CASE_IDS))
        self.assertEqual(len({row["case_id"] for row in cases}), len(PREPARED_STATE_CASE_IDS))
        row_keys = {"case_id", "input", "status", "api", "native_code", "diagnostic", "expected", "captured_actual", "backing_relations"}
        for row in cases:
            def assert_case():
                self.assertEqual(set(row), row_keys)
                self.assertIsInstance(row["input"], dict)
                self.assertIs(type(row["status"]), int)
                self.assertEqual(row["status"], 0)
                self.assertIs(type(row["api"]), int)
                self.assertEqual(row["api"], 0)
                self.assertIs(type(row["native_code"]), int)
                self.assertEqual(row["native_code"], 0)
                self.assertEqual(row["diagnostic"], "")
                self.assertIsInstance(row["expected"], dict)
                self.assertIsInstance(row["captured_actual"], dict)
                self.assertEqual(row["captured_actual"], row["expected"], row["case_id"])
                self.assertIsInstance(row["backing_relations"], list)
            if preset is None:
                assert_case()
            else:
                with self.subTest(case=f"prepared/{preset}/{row['case_id']}"):
                    assert_case()

    def _assert_prepared_projection_bytes(self, payload_bytes, entry):
        self.assertEqual(len(payload_bytes), entry["bytes"])
        self.assertEqual(hashlib.sha256(payload_bytes).hexdigest(), entry["sha256"])

    def test_prepared_state_projection_four_complete_native_contexts(self):
        for path in ("../escape", "/escape", "C:escape", "C:/escape", "nested\\escape"):
            self.assertFalse(_safe_relative_path(path), path)
        directory = FIXTURES / "prepared-state-projection"
        manifest_bytes = (directory / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        self.subtest_cases = tuple(
            [f"source/{path}" for path in manifest["source_files"]]
            + [f"resource/{path}" for path in manifest["resource_files"]]
            + [f"preset/{preset}" for preset in PRESETS]
            + [f"prepared/{preset}/{case_id}" for preset in PRESETS
               for case_id in PREPARED_STATE_CASE_IDS]
            + ["control/missing-prepared-case", "control/duplicate-prepared-case",
               "control/wrong-preset", "control/wrong-hash"]
        )
        self.assertEqual(set(manifest), {"schema_version", "authority", "source_head", "reference_revision", "source_files", "resource_files", "case_ids", "presets"})
        self.assertIs(type(manifest["schema_version"]), int)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["authority"], "Film-Juicer native composition")
        self.assertTrue(_is_git_revision(manifest["source_head"]))
        self.assertTrue(_is_git_revision(manifest["reference_revision"]))
        self.assertEqual(manifest["case_ids"], list(PREPARED_STATE_CASE_IDS))
        self.assertEqual(set(manifest["presets"]), set(PRESETS))

        self.assertIsInstance(manifest["source_files"], dict)
        self.assertTrue(manifest["source_files"])
        for path, checksum in manifest["source_files"].items():
            with self.subTest(case=f"source/{path}"):
                self.assertTrue(_safe_relative_path(path), path)
                self.assertTrue(_is_sha256(checksum), path)

        resource_root = Path(__file__).parents[2] / "Resources"
        self.assertIsInstance(manifest["resource_files"], dict)
        self.assertTrue(manifest["resource_files"])
        for relative_path, checksum in manifest["resource_files"].items():
            with self.subTest(case=f"resource/{relative_path}", resource=relative_path):
                self.assertTrue(_safe_relative_path(relative_path), relative_path)
                self.assertTrue(_is_sha256(checksum), relative_path)
                resource = resource_root / relative_path
                self.assertTrue(resource.is_file(), relative_path)
                self.assertEqual(hashlib.sha256(resource.read_bytes()).hexdigest(), checksum)

        producer_keys = {"executable", "sha256", "build_log", "build_log_sha256", "capture_log", "capture_log_sha256", "preparation"}
        for preset in PRESETS:
            with self.subTest(case=f"preset/{preset}", preset=preset):
                entry = manifest["presets"][preset]
                self.assertEqual(set(entry), {"file", "bytes", "sha256", "producer", "toolchain", "configuration", "inputs"})
                self.assertEqual(entry["file"], f"{preset}.json")
                self.assertIs(type(entry["bytes"]), int)
                self.assertGreater(entry["bytes"], 0)
                self.assertTrue(_is_sha256(entry["sha256"]))

                producer = entry["producer"]
                self.assertEqual(set(producer), producer_keys)
                for path_key in ("executable", "build_log", "capture_log", "preparation"):
                    self.assertIsInstance(producer[path_key], str)
                    self.assertTrue(producer[path_key].strip(), path_key)
                    self.assertTrue(PureWindowsPath(producer[path_key]).name, path_key)
                self.assertEqual(PureWindowsPath(producer["preparation"]).name, "result.json")
                for digest_key in ("sha256", "build_log_sha256", "capture_log_sha256"):
                    self.assertTrue(_is_sha256(producer[digest_key]), digest_key)

                self.assertEqual(set(entry["toolchain"]), {"cxx", "rust_toolchain", "pointer_bits"})
                self.assertIsInstance(entry["toolchain"]["cxx"], str)
                self.assertTrue(entry["toolchain"]["cxx"].strip())
                self.assertEqual(entry["toolchain"]["rust_toolchain"], "1.98.1")
                self.assertIs(type(entry["toolchain"]["pointer_bits"]), int)
                self.assertEqual(entry["toolchain"]["pointer_bits"], 64)
                build_type = "Debug" if preset.endswith("debug") else "Release"
                self.assertIs(type(entry["configuration"].get("testing")), bool)
                self.assertEqual(entry["configuration"], {"build_type": build_type, "testing": True})
                self.assertEqual(entry["inputs"], {
                    "resource_directory": f"out/build/{preset}/tests/Resources",
                    "parameter_type": "FjPreparationParams",
                })

                payload_bytes = (directory / entry["file"]).read_bytes()
                self._assert_prepared_projection_bytes(payload_bytes, entry)
                payload = json.loads(payload_bytes)
                self._assert_prepared_projection_identity(payload, preset, entry)
                self._assert_prepared_projection_cases(payload["cases"], preset)

                if preset == PRESETS[0]:
                    wrong_preset = dict(payload, preset=PRESETS[1])
                    with self.subTest(case="control/wrong-preset"), self.assertRaises(AssertionError):
                        self._assert_prepared_projection_identity(wrong_preset, preset, entry)
                    wrong_hash = dict(entry, sha256="0" * 64)
                    with self.subTest(case="control/wrong-hash"), self.assertRaises(AssertionError):
                        self._assert_prepared_projection_bytes(payload_bytes, wrong_hash)
                    with self.subTest(case="control/missing-prepared-case"), self.assertRaises(AssertionError):
                        self._assert_prepared_projection_cases(payload["cases"][:-1])
                    duplicate = list(payload["cases"])
                    duplicate[-1] = dict(duplicate[-1], case_id=duplicate[-2]["case_id"])
                    with self.subTest(case="control/duplicate-prepared-case"), self.assertRaises(AssertionError):
                        self._assert_prepared_projection_cases(duplicate)

    test_prepared_state_projection_four_complete_native_contexts.required_subtests = True

    def test_effects_frame_exact_native_contexts_and_complete_production_cases(self):
        directory = Path(__file__).parents[1] / "film/effects-frame-fixtures"
        manifest = json.loads((directory / "manifest.json").read_bytes())
        checked = json.loads((directory / "checked.json").read_bytes())
        production_ids = tuple(
            f"{preset}/{case_id}"
            for preset in PRESETS for case_id in EFFECTS_FRAME_CASE_IDS
        )
        native_ids = tuple(
            f"capture/{preset}/{group}/{case_id}"
            for preset in PRESETS
            for group, rows in (("literal", checked["case_ids"]), ("product", checked["product_ids"]))
            for case_id in rows
        )
        diagnostic_ids = tuple(
            f"diagnostic/{preset}/{row['case']}"
            for preset in PRESETS
            for row in manifest["production_cases"][preset]
            if row["case"].startswith(("geometry-diagnostic/", "inactive-diagnostic/", "raw-diagnostic/"))
        )
        self.subtest_cases = tuple(
            [f"preset/{preset}" for preset in PRESETS]
            + [f"sha256/{name}" for name in manifest["sha256"]]
            + list(native_ids)
            + [f"production/{case_id}" for case_id in production_ids]
            + list(diagnostic_ids)
            + [f"control/{control}/{preset}"
               for preset in PRESETS for control in ("damaged-output", "missing-membership")]
        )
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        for name, checksum in manifest["sha256"].items():
            with self.subTest(case=f"sha256/{name}"):
                self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), checksum)
        self.assertEqual((EFFECTS_FRAME_CASE_COUNT, EFFECTS_FRAME_INPUT_WORD_COUNT, EFFECTS_FRAME_OUTPUT_WORD_COUNT), (453, 180, 40))

        def output(row):
            if not row["success"] or not row["input"]["active"]:
                return [0] * EFFECTS_FRAME_OUTPUT_WORD_COUNT
            native = row["native"]
            flags = sum(int(native[name]) << index for index, name in enumerate(
                ("weave_active", "film_active", "gate_transmittance_active", "gate_output_active", "requires_full_frame")))
            words = [1, flags] + native["render_extent"] + native["full_frame_extent"]
            words += [word for origin in native["origins"] for word in origin]
            words += [native[name] for name in (
                "sample_step_x_mm", "sample_step_y_mm", "roi_offset_x", "roi_offset_y", "gate_width", "gate_height",
                "session_seed", "clip_token", "weave_dx_px", "weave_dy_px", "weave_cos_rot", "weave_sin_rot", "recipe_hash", "hash")]
            self.assertEqual(len(words), EFFECTS_FRAME_OUTPUT_WORD_COUNT)
            return [word & ((1 << 64) - 1) for word in words]

        for preset in PRESETS:
            entry = manifest["presets"][preset]
            with self.subTest(case=f"preset/{preset}", preset=preset):
                captured = load_shared_fixture(directory / entry["native"])
                self.assertEqual(captured["preset"], preset)
                self.assertEqual(hashlib.sha256(original_capture_bytes(captured, "LF")).hexdigest(), entry["original_capture_sha256"])
                self.assertEqual([len(captured[key]) for key in ("cases", "products", "primitives")], [347, 72, 57])
                self.assertEqual([row["input"]["id"] for row in captured["cases"]], checked["case_ids"])
                self.assertEqual(len(set(checked["case_ids"])), 347)
                self.assertEqual([row["input"]["id"] for row in captured["products"]], checked["product_ids"])
            rows = manifest["production_cases"][preset]
            def membership(values):
                self.assertEqual(tuple(row["case"] for row in values), EFFECTS_FRAME_CASE_IDS)
                self.assertEqual(len({row["case"] for row in values}), EFFECTS_FRAME_CASE_COUNT)
            membership(rows)
            with self.subTest(case=f"control/missing-membership/{preset}"), self.assertRaises(AssertionError):
                membership(rows[:-1])
            by_id = {row["case"]: row for row in rows}
            for cohort, key in (("literal", "cases"), ("product", "products")):
                for native in captured[key]:
                    name = cohort + "/" + native["input"]["id"]
                    with self.subTest(case=f"capture/{preset}/{name}"):
                        if native["input"].get("missing_recipe"):
                            continue
                        expected = by_id[name]
                        self.assertEqual(expected["output"], output(native))
                        self.assertEqual(expected["diagnostic"], native["diagnostic"])
                        self.assertEqual(len(manifest["production_inputs"][preset][name]["words"]), EFFECTS_FRAME_INPUT_WORD_COUNT)
            for row in rows:
                with self.subTest(case=f"production/{preset}/{row['case']}"):
                    raw_null_cases = {"raw/4", "raw/5", "raw/6"}
                    is_raw_null = row["case"] in raw_null_cases
                    self.assertEqual(row["output"] is None, is_raw_null)
                    if is_raw_null:
                        source = manifest["production_inputs"][preset][row["case"]]
                        diagnostic = "physical effects frame arguments are invalid"
                        self.assertEqual(source["mode"], int(row["case"].rsplit("/", 1)[1]))
                        self.assertEqual(source["capacity"], 256)
                        self.assertEqual(row["failure"], [1])
                        self.assertEqual(row["diagnostic"], diagnostic)
                        self.assertEqual(row["diagnostic_length"], len(diagnostic))
                    else:
                        self.assertEqual(len(row["output"]), EFFECTS_FRAME_OUTPUT_WORD_COUNT)
                        self.assertTrue(row["failure"] is None or
                                        len(row["failure"]) == EFFECTS_FRAME_FAILURE_WORD_COUNT)
            damaged = list(by_id["literal/mixed"]["output"])
            damaged[-1] ^= 1
            with self.subTest(case=f"control/damaged-output/{preset}"), self.assertRaises(AssertionError):
                self.assertEqual(damaged, output(captured["cases"][0]))
            for prefix, message, category, reason in (
                    ("geometry-diagnostic/", checked["geometry_failure"]["diagnostic"], 3, 3),
                    ("inactive-diagnostic/", "", 3, 2),
                    ("raw-diagnostic/", checked["raw_failure"]["diagnostic"], 2, 1)):
                selected = [row for row in rows if row["case"].startswith(prefix)]
                self.assertEqual(len(selected), 5)
                self.assertEqual({manifest["production_inputs"][preset][row["case"]]["capacity"] for row in selected},
                                 {0, 1, 7, len(message), len(message) + 1})
                for row in selected:
                    with self.subTest(case=f"diagnostic/{preset}/{row['case']}"):
                        capacity = manifest["production_inputs"][preset][row["case"]]["capacity"]
                        self.assertEqual(row["envelope"], [category, 0, 0])
                        self.assertEqual(row["failure"], [reason])
                        self.assertEqual(row["output"], [0] * EFFECTS_FRAME_OUTPUT_WORD_COUNT)
                        self.assertEqual(row["diagnostic"], message[:max(0, capacity - 1)])
                        self.assertEqual(row["diagnostic_length"], len(row["diagnostic"]))

    test_effects_frame_exact_native_contexts_and_complete_production_cases.required_subtests = True

    def test_production_grain_fixture_records_keep_both_authorities(self):
        root = Path(__file__).parents[2]
        tree = ast.parse((root / "tests/ffi/production_contract.py").read_text())
        record = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                      and {"grain_fixture", "grain_frame_fixture"} <= {
                          key.value for key in node.keys if isinstance(key, ast.Constant)})
        fields = {key.value: value for key, value in zip(record.keys, record.values)
                  if isinstance(key, ast.Constant)}
        projection = ast.Dict(keys=[ast.Constant("grain_fixture"), ast.Constant("grain_frame_fixture")],
                              values=[deepcopy(fields["grain_fixture"]), deepcopy(fields["grain_frame_fixture"])])
        self.subtest_cases = tuple(f"preset/{preset}" for preset in PRESETS)
        for preset in PRESETS:
            grain_manifest = root / "tests/film/grain-fixtures/manifest.json"
            grain_payload = grain_manifest.parent / json.loads(grain_manifest.read_bytes())["presets"][preset]["values"]
            frame_manifest = root / "tests/grain/frame-fixtures/manifest.json"
            frame_spec = json.loads(frame_manifest.read_bytes())
            frame_payload = frame_manifest.parent / frame_spec["presets"][preset]["native"]
            bindings = {"str": str, "arguments": SimpleNamespace(preset=preset),
                        "grain_manifest": grain_manifest, "grain_payload": grain_payload,
                        "grain_frame_manifest": frame_manifest, "grain_frame_payload": frame_payload,
                        "grain_frame_checked": frame_manifest.parent / frame_spec["checked"]}

            def render(expression):
                code = compile(ast.fix_missing_locations(ast.Expression(expression)), "production fixture records", "eval")
                return eval(code, {"__builtins__": {}}, bindings)

            with self.subTest(case=f"preset/{preset}"):
                actual = render(deepcopy(projection))
                self.assertEqual(actual["grain_fixture"], {"manifest": str(grain_manifest), "payload": str(grain_payload), "logical_preset": preset})
                self.assertEqual(actual["grain_frame_fixture"], {"manifest": str(frame_manifest), "payload": str(frame_payload),
                    "normative": str(bindings["grain_frame_checked"]), "input_spec": str(frame_manifest.parent / "inputs.json"), "logical_preset": preset})
                damaged = deepcopy(projection)
                for node in ast.walk(damaged):
                    if isinstance(node, ast.Name) and node.id in ("grain_frame_manifest", "grain_frame_payload"):
                        node.id = node.id.replace("grain_frame_", "grain_")
                self.assertNotEqual(render(damaged)["grain_frame_fixture"], actual["grain_frame_fixture"])

    test_production_grain_fixture_records_keep_both_authorities.required_subtests = True

    def test_grain_frame_complete_native_contexts_and_r1_production_contract(self):
        root = Path(__file__).parents[1] / "grain/frame-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.subtest_cases = tuple(
            [f"sha256/{name}" for name in manifest["sha256"]]
            + [f"preset/{preset}" for preset in PRESETS]
            + [f"capture/{preset}/{group}/{index}" for preset in PRESETS
               for group, count in (("case", 153), ("product", 44)) for index in range(count)]
            + [f"production/{row['case']}" for row in manifest["production_cases"]]
            + [f"diagnostic/{row['case']}" for prefix in ("raw-diagnostic/", "r1-diagnostic/")
               for row in manifest["production_cases"] if row["case"].startswith(prefix)]
            + ["control/r1-failure", "control/r1-output"]
        )
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        for name, checksum in manifest["sha256"].items():
            with self.subTest(case=f"sha256/{name}"):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), checksum)
        baseline = None
        for preset in PRESETS:
            captured = json.loads((root / manifest["presets"][preset]["native"]).read_bytes())
            with self.subTest(case=f"preset/{preset}"):
                self.assertEqual(captured["preset"], preset)
                self.assertEqual([len(captured[key]) for key in ("cases", "products", "primitives")], [153, 44, 50])
            semantic = [captured[key] for key in ("cases", "products", "primitives")]
            if baseline is None:
                baseline = semantic
            self.assertEqual(semantic, baseline)
            for group in ("cases", "products"):
                for index, row in enumerate(captured[group]):
                    with self.subTest(case=f"capture/{preset}/{group[:-1]}/{index}"):
                        native = row["native"]
                        self.assertEqual(len(native["correlation"]), 3)
                        self.assertEqual([len(layer) for layer in native["dye_cloud"]], [3, 3, 3])
                        self.assertTrue(all(len(g) == 3 for g in native["correlation"]))
                        self.assertTrue(all(len(g) == 3 for layer in native["dye_cloud"] for g in layer))
            r1 = next(row for row in captured["cases"] if row["input"]["name"] == "S4D6-R1")
            self.assertFalse(r1["success"])
            self.assertEqual(r1["diagnostic"], "ResourceDescriptorMismatch phase=grain_descriptor field=layer_particle_count")
            self.assertEqual(r1["input"]["layer_curves"], [[[0], [0], [1065353216]], [[1065353216], [0], [1065353216]], [[1065353216], [0], [1065353216]]])
        rows = manifest["production_cases"]
        self.assertEqual(len(rows), 188)
        self.assertEqual(len({row["case"] for row in rows}), 188)
        self.assertEqual({row["case"] for row in rows}, set(manifest["production_inputs"]))
        for row in rows:
            with self.subTest(case=f"production/{row['case']}"):
                source = manifest["production_inputs"][row["case"]]
                self.assertEqual(len(source["words"]), 42)
                self.assertEqual(len(source["curves"]), 9)
                self.assertTrue(row["output"] is None or len(row["output"]) == 115)
                self.assertTrue(row["failure"] is None or len(row["failure"]) == 4)
        r1 = next(row for row in rows if row["case"] == "literal/S4D6-R1")
        def require_r1(value):
            self.assertEqual(value["envelope"], [3, 0, 0])
            self.assertEqual(value["failure"], [2, 12, 0, 0])
            self.assertEqual(value["output"], [0] * 115)
        require_r1(r1)
        for key, replacement in (("failure", [2, 10, 3, 1]), ("output", [1] + [0] * 114)):
            damaged = dict(r1, **{key: replacement})
            control = "control/r1-failure" if key == "failure" else "control/r1-output"
            with self.subTest(case=control), self.assertRaises(AssertionError):
                require_r1(damaged)
        for prefix, message, metadata in (("raw-diagnostic/", "visual grain frame arguments are invalid", [1, 0, 3, 3]),
                                          ("r1-diagnostic/", r1["diagnostic"], [2, 12, 0, 0])):
            selected = [row for row in rows if row["case"].startswith(prefix)]
            self.assertEqual(len(selected), 5)
            self.assertEqual({manifest["production_inputs"][row["case"]]["capacity"] for row in selected}, {0, 1, 7, len(message), len(message) + 1})
            for row in selected:
                with self.subTest(case=f"diagnostic/{row['case']}"):
                    capacity = manifest["production_inputs"][row["case"]]["capacity"]
                    self.assertEqual(row["failure"], metadata)
                    self.assertEqual(row["output"], [0] * 115)
                    self.assertEqual(row["diagnostic"], message[:max(0, capacity - 1)])
                    self.assertEqual(row["diagnostic_length"], min(len(message), max(0, capacity - 1)))

    test_grain_frame_complete_native_contexts_and_r1_production_contract.required_subtests = True


    def test_dir_boundary_exact_native_products_keys_and_production_contracts(self):
        root = Path(__file__).parents[1] / "dir/boundary-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.subtest_cases = tuple(
            [f"sha256/{name}" for name in manifest["sha256"]]
            + [f"preset/{preset}" for preset in PRESETS]
            + [f"capture/{preset}/case/{index}" for preset in PRESETS for index in range(114)]
            + [f"production/{row['case']}" for row in manifest["production_cases"]]
            + [f"raw/{operation}/{mode}" for operation in ("describe", "acquire", "view")
               for mode in (11, 12, 13, 14, 15)]
        )
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        for name, checksum in manifest["sha256"].items():
            with self.subTest(case=f"sha256/{name}"):
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
            with self.subTest(case=f"preset/{preset}"):
                self.assertEqual(captured["preset"], preset)
            for index, row in enumerate(captured["cases"]):
                with self.subTest(case=f"capture/{preset}/case/{index}"):
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
        input_by_case = manifest["production_inputs"]
        cases_by_kind = {
            kind: {case for case, source in input_by_case.items() if source["kind"] == kind}
            for kind in ("values", "raw", "lifetime")
        }
        self.assertEqual({kind: len(cases) for kind, cases in cases_by_kind.items()},
                         {"values": 117, "raw": 51, "lifetime": 3})
        self.assertEqual(sum(map(len, cases_by_kind.values())), len(manifest["production_cases"]))
        self.assertEqual(len({row["case"] for row in manifest["production_cases"]}), 171)
        self.assertEqual(len({case for case in cases_by_kind["values"] if case.startswith("values/")}), 114)
        self.assertEqual({case for case in cases_by_kind["values"] if case.startswith("safety/")},
                         {"safety/horizontal-overflow", "safety/vertical-overflow",
                          "safety/horizontal-policy-before-vertical-overflow"})
        self.assertEqual({case for case in cases_by_kind["lifetime"]},
                         {"lifetime/independent-expiry", "lifetime/release-malformed", "lifetime/null-release"})
        self.assertEqual({(input_by_case[case]["operation"], input_by_case[case]["mode"])
                          for case in cases_by_kind["raw"]},
                         {(operation, mode) for operation in range(3) for mode in range(17)})
        raw = [row for row in manifest["production_cases"] if row["case"] in cases_by_kind["raw"]]
        self.assertEqual(len(raw), 51)
        for row in manifest["production_cases"]:
            with self.subTest(case=f"production/{row['case']}"):
                case = row["case"]
                source = input_by_case[case]
                if source["kind"] == "values":
                    self.assertTrue(case.startswith(("values/", "safety/")))
                    expected_source_fields = {"kind", "words"} if case.startswith("values/") else {
                        "kind", "words", "authority"}
                    self.assertEqual(set(source), expected_source_fields)
                    self.assertEqual(len(source["words"]), 18)
                    self.assertTrue(all(type(word) is int and 0 <= word <= 0xFFFFFFFFFFFFFFFF
                                        for word in source["words"]))
                    self.assertEqual(set(row), {"case", "envelope", "key", "failure", "shape", "weights",
                                                "terminal", "diagnostic", "diagnostic_length", "owner"})
                    self.assertEqual(len(row["envelope"]), 3)
                    self.assertEqual(len(row["failure"]), 7)
                    self.assertIs(type(row["key"]), int)
                    self.assertIs(type(row["owner"]), bool)
                    self.assertTrue(row["shape"] is None or len(row["shape"]) == 6)
                    self.assertTrue(row["terminal"] is None or len(row["terminal"]) == 18)
                    self.assertTrue(row["weights"] is None or
                                    all(type(word) is int and 0 <= word <= 0xFFFFFFFFFFFFFFFF
                                        for word in row["weights"]))
                    self.assertIs(type(row["diagnostic"]), str)
                    self.assertEqual(row["diagnostic_length"], len(row["diagnostic"]))
                    if case.startswith("safety/"):
                        self.assertEqual(source["authority"],
                                         "authored representation/order contract; independent FNV encoding; never run through undefined native period overflow")
                elif source["kind"] == "raw":
                    self.assertTrue(case.startswith("raw/"))
                    _, operation, mode = case.split("/")
                    self.assertEqual(set(source), {"kind", "operation", "mode", "output_valid", "failure_valid",
                                                   "error_record", "error_span", "capacity"})
                    self.assertEqual(source["operation"], {"describe": 0, "acquire": 1, "view": 2}[operation])
                    self.assertEqual(source["mode"], int(mode))
                    self.assertTrue(all(type(source[name]) is bool for name in (
                        "output_valid", "failure_valid", "error_record", "error_span")))
                    self.assertIs(type(source["capacity"]), int)
                    self.assertGreaterEqual(source["capacity"], 0)
                    self.assertEqual(set(row), {"case", "envelope", "key", "failure", "shape", "weights",
                                                "terminal", "diagnostic", "diagnostic_length", "owner"})
                    self.assertEqual(len(row["envelope"]), 3)
                    self.assertIn(row["shape"], (None, [0, 0, 0, 0, 0, 0]))
                    self.assertTrue(row["failure"] is None or len(row["failure"]) == 7)
                    self.assertTrue(row["weights"] is None or row["weights"] == [])
                    self.assertTrue(row["terminal"] is None or len(row["terminal"]) == 18)
                    self.assertTrue(row["owner"] is None or type(row["owner"]) is bool)
                    self.assertTrue(row["key"] is None or type(row["key"]) is int)
                    if row["diagnostic"] is None:
                        self.assertIn(row["diagnostic_length"], (None, 0))
                    else:
                        self.assertIs(type(row["diagnostic"]), str)
                        self.assertEqual(row["diagnostic_length"], len(row["diagnostic"]))
                else:
                    self.assertEqual(source, {"kind": "lifetime"})
                    if case == "lifetime/independent-expiry":
                        self.assertEqual(set(row), {"case", "envelope", "key", "count"})
                        self.assertEqual(row["count"], 45)
                        self.assertIs(type(row["key"]), int)
                    elif case == "lifetime/release-malformed":
                        self.assertEqual(set(row), {"case", "envelope", "diagnostic_length"})
                        self.assertEqual(row["diagnostic_length"], 0)
                    else:
                        self.assertEqual(case, "lifetime/null-release")
                        self.assertEqual(set(row), {"case", "envelope"})
                    self.assertEqual(len(row["envelope"]), 3)
        for operation in ("describe", "acquire", "view"):
            for mode in (11, 12, 13, 14, 15):
                row = next(row for row in raw if row["case"] == f"raw/{operation}/{mode}")
                with self.subTest(case=f"raw/{operation}/{mode}"):
                    message = "spatial DIR reflected-boundary arguments are invalid"
                    self.assertEqual(row["diagnostic_length"], min(len(message), max(0, manifest["production_inputs"][row["case"]]["capacity"] - 1)))
                    self.assertEqual(row["diagnostic"], message[:row["diagnostic_length"]])
                    self.assertEqual(row["envelope"], [2, 0, 0])

    test_dir_boundary_exact_native_products_keys_and_production_contracts.required_subtests = True

    def test_dir_frame_exact_native_bytes_fields_and_production_membership(self):
        root = Path(__file__).parents[1] / "dir/frame-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.subtest_cases = tuple(
            [f"sha256/{name}" for name in manifest["sha256"]]
            + [f"preset/{preset}" for preset in PRESETS]
            + [f"capture/{preset}/{group}/{index}" for preset in PRESETS
               for group, count in (("case", 74), ("product", 24)) for index in range(count)]
            + [f"production/{row['case']}" for row in manifest["production_cases"]]
        )
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(len(set(manifest["payloads"].values())), 4)
        for name, checksum in manifest["sha256"].items():
            with self.subTest(case=f"sha256/{name}"):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), checksum)
        baseline = None
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            with self.subTest(case=f"preset/{preset}"):
                self.assertEqual(captured["preset"], preset)
                self.assertEqual([len(captured[key]) for key in ("cases", "products")], [74, 24])
            rows = captured["cases"] + captured["products"]
            if baseline is None:
                baseline = rows
            self.assertEqual(rows, baseline)
            for index, row in enumerate(rows):
                group = "case" if index < 74 else "product"
                row_index = index if group == "case" else index - 74
                with self.subTest(case=f"capture/{preset}/{group}/{row_index}"):
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
            with self.subTest(case=f"production/{row['case']}"):
                self.assertEqual(len(manifest["production_inputs"][row["case"]]["words"]), 14)
                self.assertTrue(row["output"] is None or len(row["output"]) == 79)
                self.assertTrue(row["failure"] is None or len(row["failure"]) == 82)

    test_dir_frame_exact_native_bytes_fields_and_production_membership.required_subtests = True

    def test_diffusion_frame_original_bytes_and_complete_native_contexts(self):
        root = Path(__file__).parents[1] / "diffusion/frame-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.subtest_cases = tuple(
            [f"sha256/{name}" for name in manifest["sha256"]]
            + [f"preset/{preset}" for preset in PRESETS]
            + [f"sample/{preset}/{index}" for preset in PRESETS for index in range(174)]
        )
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(len(set(manifest["contexts"].values())), 4)
        for name, checksum in manifest["sha256"].items():
            with self.subTest(case=f"sha256/{name}"):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), checksum)
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            provenance = manifest["provenance"][preset]
            with self.subTest(case=f"preset/{preset}"):
                self.assertEqual(hashlib.sha256(original_capture_bytes(captured, provenance["line_ending"])).hexdigest(), provenance["original_capture_sha256"])
                self.assertEqual([len(captured[k]) for k in ("samples", "frames", "products", "primitives")], [94, 35, 92, 8])
            edges = load_shared_fixture(root / manifest["presets"][preset]["edges"])
            edge_provenance = manifest["edge_provenance"][preset]
            self.assertEqual(hashlib.sha256(original_capture_bytes(edges, edge_provenance["line_ending"])).hexdigest(), edge_provenance["original_capture_sha256"])
            self.assertEqual(edge_provenance["producer_sha256"], provenance["producer_sha256"])
            self.assertEqual([len(edges[k]) for k in ("samples", "frames", "products", "primitives")], [2, 0, 0, 0])
            samples = captured["samples"] + edges["samples"] + [sample for row in captured["products"] for sample in row["samples"]]
            self.assertEqual(len(samples), 174)
            for index, row in enumerate(samples):
                with self.subTest(case=f"sample/{preset}/{index}"):
                    if row["success"]:
                        self.assertEqual(set(row["components"]), {"lambda", "red", "green", "blue"})
                        self.assertTrue(all(len(words) == 9 for words in row["components"].values()))

    test_diffusion_frame_original_bytes_and_complete_native_contexts.required_subtests = True

    def test_halation_frame_original_bytes_and_complete_native_contexts(self):
        root = Path(__file__).parents[1] / "scatter_halation/frame-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.subtest_cases = tuple(
            [f"sha256/{name}" for name in manifest["sha256"]]
            + [f"preset/{preset}" for preset in PRESETS]
            + [f"capture/{preset}/{group}/{index}" for preset in PRESETS
               for group, count in (("case", 52), ("product", 20)) for index in range(count)]
        )
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(len(set(manifest["contexts"].values())), 4)
        for name, checksum in manifest["sha256"].items():
            with self.subTest(case=f"sha256/{name}"):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), checksum)
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            with self.subTest(case=f"preset/{preset}"):
                self.assertEqual(captured["preset"], preset)
                provenance = manifest["provenance"][preset]
                self.assertEqual(hashlib.sha256(original_capture_bytes(captured, provenance["line_ending"])).hexdigest(), provenance["original_capture_sha256"])
                self.assertEqual(len(captured["cases"]), 52)
                self.assertEqual(len(captured["products"]), 20)
            for group in ("cases", "products"):
                for index, row in enumerate(captured[group]):
                    with self.subTest(case=f"capture/{preset}/{group[:-1]}/{index}"):
                        if row["present"]:
                            self.assertEqual(len(row["descriptor"]["channels"]), 3)
                            for channel in row["descriptor"]["channels"]:
                                for gaussian in [channel["core"], *channel["tail"], *channel["bounce"]]:
                                    self.assertEqual(len(gaussian["weights"]), 19)
                                    self.assertEqual(len(gaussian["yvv"]), 4)

    test_halation_frame_original_bytes_and_complete_native_contexts.required_subtests = True

    def test_frame_geometry_full_fields_and_distinct_native_contexts(self):
        root = Path(__file__).parents[1] / "frame/geometry-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.subtest_cases = tuple(
            [f"sha256/{name}" for name in manifest["sha256"]]
            + [f"preset/{preset}" for preset in PRESETS]
            + [f"domain/{preset}/{index}" for preset in PRESETS for index in range(manifest["domain_count"])]
            + [f"geometry/{preset}/{index}" for preset in PRESETS for index in range(manifest["geometry_count"])]
        )
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(len(set(manifest["contexts"].values())), 4)
        for name, expected in manifest["sha256"].items():
            with self.subTest(case=f"sha256/{name}"):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            with self.subTest(case=f"preset/{preset}"):
                self.assertEqual(captured["preset"], preset)
                self.assertEqual(len(captured["domains"]), manifest["domain_count"])
                self.assertEqual(len(captured["geometry"]), manifest["geometry_count"])
            adopted = (root / manifest["payloads"][manifest["presets"][preset]["native"]]).read_bytes()
            self.assertNotIn(b"\r", adopted)
            original = adopted.replace(b"\n", b"\r\n") if captured["native_payload_line_ending"] == "CRLF" else adopted
            self.assertEqual(hashlib.sha256(original).hexdigest(), captured["producer"]["files"]["payload.json"]["sha256"])
            self.assertTrue(captured["commands"])
            self.assertEqual(set(captured["producer"]["files"]), {"JuicerAdapterTraceProbe.exe" if "windows" in preset else "JuicerAdapterTraceProbe", "JuicerEffect.cpp.txt", "inputs.json", "payload.json", "context.json"})
            for index, row in enumerate(captured["domains"]):
                with self.subTest(case=f"domain/{preset}/{index}"):
                    self.assertEqual(len(row["bounds"]), 4)
                    if not row["range"]:
                        self.assertEqual(len(row["domain"]), 4)
            for index, row in enumerate(captured["geometry"]):
                with self.subTest(case=f"geometry/{preset}/{index}"):
                    self.assertEqual(len(row["domain"]), 4)
                    self.assertIsInstance(row["pixel_bits"], int)
                    self.assertIsInstance(row["diffusion_bits"], int)

    test_frame_geometry_full_fields_and_distinct_native_contexts.required_subtests = True

    def test_gaussian_complete_arrays_and_distinct_native_contexts(self):
        root = Path(__file__).parents[1] / "optics/gaussian-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.subtest_cases = tuple(
            [f"sha256/{name}" for name in manifest["sha256"]]
            + [f"preset/{preset}" for preset in PRESETS]
            + [f"capture/{preset}/{index}" for preset in PRESETS for index in range(188)]
            + [f"sharing/{native}" for native in ("linux", "windows-clang")]
        )
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(len(set(manifest["contexts"].values())), 4)
        self.assertEqual(set(manifest["payloads"].values()), {"linux-payload.json", "windows-clang-payload.json"})
        for name, expected in manifest["sha256"].items():
            with self.subTest(case=f"sha256/{name}"):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            with self.subTest(case=f"preset/{preset}"):
                self.assertEqual(captured["preset"], preset)
                self.assertEqual(len(captured["cases"]), 188)
                self.assertEqual(sum(row["count"] for row in captured["cases"]), 22895)
                self.assertEqual([row["name"] for row in captured["cases"]], manifest["production_cases"][:188])
            for index, row in enumerate(captured["cases"]):
                with self.subTest(case=f"capture/{preset}/{index}"):
                    self.assertEqual(len(row["weights"]), row["count"])
                    self.assertTrue(all(isinstance(word, int) and 0 <= word <= 0xffffffff for word in row["weights"]))
            self.assertTrue(any("fir-dir" in row["name"] and row["count"] > 0 for row in captured["cases"]))
            original = deepcopy(captured)
            provenance = original.pop("provenance")
            ending = "LF" if preset.startswith("linux-") else "CRLF"
            self.assertEqual(hashlib.sha256(original_capture_bytes(original, ending)).hexdigest(), provenance["capture_sha256"])
        self.assertEqual(len(manifest["production_cases"]), 201)
        self.assertEqual(len(set(manifest["production_cases"])), 201)
        for native in ("linux", "windows-clang"):
            debug = load_shared_fixture(root / manifest["presets"][native + "-debug"]["native"])
            release = load_shared_fixture(root / manifest["presets"][native + "-release"]["native"])
            with self.subTest(case=f"sharing/{native}"):
                self.assertEqual(debug["cases"], release["cases"])
                debug["cases"][0]["weights"].append(123)
                debug["provenance"].clear()
                self.assertNotEqual(debug["cases"], release["cases"])
                self.assertTrue(release["provenance"])
                self.assertEqual(load_shared_fixture(root / manifest["presets"][native + "-release"]["native"]), release)

    test_gaussian_complete_arrays_and_distinct_native_contexts.required_subtests = True

    def test_scanner_post_complete_frozen_contexts(self):
        root = Path(__file__).parents[1] / "scanner/post-effects-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.subtest_cases = tuple(
            [f"sha256/{name}" for name in manifest["sha256"]]
            + [f"preset/{preset}" for preset in PRESETS]
            + [f"capture/{preset}/{index}" for preset in PRESETS for index in range(400)]
            + [f"production/{row['case']}" for row in manifest["production_cases"]]
        )
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(set(manifest["payloads"].values()), {"native.json"})
        self.assertEqual(len(set(manifest["contexts"].values())), 4)
        for name, expected in manifest["sha256"].items():
            with self.subTest(case=f"sha256/{name}"):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        payloads = []
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            with self.subTest(case=f"preset/{preset}"):
                self.assertEqual(captured["preset"], preset)
                self.assertEqual(len(captured["cases"]), 400)
            for index, row in enumerate(captured["cases"]):
                with self.subTest(case=f"capture/{preset}/{index}"):
                    self.assertEqual(len(row["input"]["values"]), 6)
                    self.assertEqual(len(row["native"]["descriptor"]["radii"]), 3)
                    self.assertEqual(row["native"]["descriptor"]["schema"], 2)
            payloads.append(captured["cases"])
        self.assertTrue(all(value == payloads[0] for value in payloads))
        self.assertEqual(len(manifest["production_cases"]), 410)
        self.assertEqual(len({row["case"] for row in manifest["production_cases"]}), 410)
        for row in manifest["production_cases"]:
            with self.subTest(case=f"production/{row['case']}"):
                self.assertIsInstance(row["case"], str)

    test_scanner_post_complete_frozen_contexts.required_subtests = True

    def test_scanner_descriptor_complete_frozen_contexts(self):
        root = Path(__file__).parents[1] / "scanner/descriptor-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        result_rows = manifest["production_cases"] + manifest["production_malformed"]
        self.subtest_cases = tuple(
            [f"sha256/{name}" for name in manifest["sha256"]]
            + [f"preset/{preset}" for preset in PRESETS]
            + [f"capture/{preset}/{group}/{index}" for preset in PRESETS
               for group, count in (("case", 520), ("product", 4)) for index in range(count)]
            + [f"production/{row['case']}" for row in result_rows]
        )
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(set(manifest["payloads"].values()), {"native.json"})
        self.assertEqual(len(set(manifest["contexts"].values())), 4)
        for name, expected in manifest["sha256"].items():
            with self.subTest(case=f"sha256/{name}"):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        payloads = []
        for preset in PRESETS:
            captured = load_shared_fixture(root / manifest["presets"][preset]["native"])
            with self.subTest(case=f"preset/{preset}"):
                self.assertEqual(captured["preset"], preset)
                self.assertEqual(len(captured["cases"]), 520)
                self.assertEqual(len(captured["products"]), 4)
                self.assertEqual({row["route"] for row in captured["products"]}, {0, 1, 2, 3})
                self.assertEqual(len({row["id"] for row in captured["cases"]}), 520)
            for group in ("cases", "products"):
                for index, row in enumerate(captured[group]):
                    with self.subTest(case=f"capture/{preset}/{group[:-1]}/{index}"):
                        self.assertEqual(len(row["input"]["channel_density_cmy"]), 243)
                        self.assertEqual(len(row["input"]["base_density"]), 81)
                        self.assertEqual(len(row["native"]["descriptor"]), 10)
            payloads.append((captured["cases"], captured["products"]))
        self.assertTrue(all(value == payloads[0] for value in payloads))
        self.assertEqual(len(result_rows), 41)
        self.assertEqual(len({row["case"] for row in result_rows}), 41)
        for row in result_rows:
            with self.subTest(case=f"production/{row['case']}"):
                self.assertIsInstance(row["case"], str)

    test_scanner_descriptor_complete_frozen_contexts.required_subtests = True

    def test_scanner_lut_original_capture_integrity(self):
        root = Path(__file__).parents[1] / "scanner/lut-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.subtest_cases = tuple(
            [f"storage/{name}" for name in manifest["storage_files"]]
            + [f"preset/{preset}" for preset in PRESETS]
            + [f"capture/{preset}/{group}/{index}" for preset in PRESETS
               for group, count in (("case", 30), ("product", 4), ("slope", 15), ("primitive", 63))
               for index in range(count)]
        )
        logical_names = {preset + ".json" for preset in PRESETS}
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        for field in ("payloads", "contexts", "original_line_endings"):
            self.assertEqual(set(manifest[field]), logical_names)
        self.assertEqual(set(manifest["payloads"].values()), {"native.json"})
        physical = set(manifest["payloads"].values()) | set(manifest["contexts"].values())
        self.assertEqual(physical, set(manifest["storage_files"]))
        for name, expected in manifest["storage_files"].items():
            with self.subTest(case=f"storage/{name}"):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected, name)
        for preset in PRESETS:
            with self.subTest(case=f"preset/{preset}", preset=preset):
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
                for group, expected_count in (("cases", 30), ("products", 4), ("slopes", 15), ("primitives", 63)):
                    row_group = {"cases": "case", "products": "product", "slopes": "slope", "primitives": "primitive"}[group]
                    for index, row in enumerate(capture[group]):
                        with self.subTest(case=f"capture/{preset}/{row_group}/{index}"):
                            self.assertLess(index, expected_count)

    test_scanner_lut_original_capture_integrity.required_subtests = True

    def test_shared_context_is_required_disjoint_and_independently_owned(self):
        self.subtest_cases = ("control/debug-context-load", "control/copy-isolation",
                              "control/missing-release-context", "control/overlapping-context")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"payloads": {"debug.json": "native.json", "release.json": "native.json"},
                        "contexts": {"debug.json": "debug-context.json"}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            words = [0, 0x80000000, 0x7FC00123, 0xFFFFFFFF]
            (root / "native.json").write_text(json.dumps({"words": words}))
            (root / "debug-context.json").write_text(json.dumps({"preset": "debug", "primitives": [1]}))
            with self.subTest(case="control/debug-context-load"):
                first = load_shared_fixture(root / "debug.json")
                self.assertEqual(first, {"words": words, "preset": "debug", "primitives": [1]})
            with self.subTest(case="control/copy-isolation"):
                first["words"][0] = 17
                self.assertEqual(load_shared_fixture(root / "debug.json")["words"], words)
            with self.subTest(case="control/missing-release-context"), self.assertRaises(KeyError):
                load_shared_fixture(root / "release.json")
            (root / "debug-context.json").write_text(json.dumps({"words": words}))
            with self.subTest(case="control/overlapping-context"), self.assertRaisesRegex(ValueError, "context overlaps payload"):
                load_shared_fixture(root / "debug.json")

    test_shared_context_is_required_disjoint_and_independently_owned.required_subtests = True

    def test_preflash_native_integrity_inputs_and_caller_membership(self):
        root = Path(__file__).parents[1] / "print/preflash-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        self.subtest_cases = tuple(
            [f"file/{name}" for name in manifest["files"]]
            + [f"preset/{preset}" for preset in PRESETS]
            + [f"input/{preset}/{index}" for preset in PRESETS for index in range(19)]
            + [f"product/{preset}/{route}-{mode}-{variant}" for preset in PRESETS
               for route in range(4) for mode in range(3) for variant in range(6)]
        )
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        inputs = json.loads((root / "cases.json").read_bytes())["cases"]
        self.assertEqual(len(inputs), 19)
        for name, record in manifest["files"].items():
            data = (root / name).read_bytes()
            with self.subTest(case=f"file/{name}"):
                self.assertEqual(len(data), record["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"], name)
        for preset in PRESETS:
            capture = json.loads((root / manifest["presets"][preset]["native"]).read_bytes())
            with self.subTest(case=f"preset/{preset}"):
                self.assertEqual(capture["preset"], preset)
                self.assertEqual(len(capture["cases"]), 19)
                self.assertEqual(len(capture["products"]), 72)
            for index, (source, row) in enumerate(zip(inputs, capture["cases"], strict=True)):
                with self.subTest(case=f"input/{preset}/{index}"):
                    self.assertEqual({key: row[key] for key in source}, source)
                    for key, extent in (("film_base_density", 81), ("filtered_illuminant", 81), ("print_sensitivity_cmy", 243), ("raw", 3)):
                        self.assertEqual(len(row[key]), extent)
                        self.assertTrue(all(type(word) is int and 0 <= word <= 0xffffffff for word in row[key]))
                    self.assertEqual(row["success"], row["id"] not in manifest["failure_channels"])
                    self.assertEqual(row["diagnostic"], "" if row["success"] else "MalformedRequiredResource phase=4B field=preflash_raw")
            self.assertEqual({(r["route"], r["mode"], r["variant"]) for r in capture["products"]},
                             {(route, mode, variant) for route in range(4) for mode in range(3) for variant in range(6)})
            for row in capture["products"]:
                case = f"product/{preset}/{row['route']}-{row['mode']}-{row['variant']}"
                with self.subTest(case=case):
                    if row["route"] % 2:
                        self.assertEqual(len(row["descriptors"]), 5)
                        self.assertEqual(len(row["raw"]), 3)
                        self.assertEqual(row["descriptors"][3] == 0, row["variant"] == 0)
                    else:
                        self.assertNotIn("raw", row)

    test_preflash_native_integrity_inputs_and_caller_membership.required_subtests = True

    def test_unique_payload_integrity_and_native_membership(self):
        families = (("color", "", "manifest.json", "recipes", 15),
                    ("color", "cat02-", "cat02-manifest.json", "recipes", 56),
                    ("color", "input-color-", "input-color-manifest.json", "recipes", 140),
                    ("illuminants", "construction-", "construction-manifest.json", "products", 270))
        manifests = {name: json.loads((FIXTURES / directory / name).read_bytes())
                     for directory, _, name, _, _ in families}
        self.subtest_cases = tuple(
            [f"family/{name}" for _, _, name, _, _ in families]
            + [f"payload/{name}/{payload}" for _, _, name, _, _ in families
               for payload in set(manifests[name]["payloads"].values())]
            + [f"row/{name}/{payload}/{index}" for _, _, name, group, count in families
               for payload in set(manifests[name]["payloads"].values()) for index in range(count)]
            + [f"witness/{logical}" for _, _, name, _, _ in families
               if "preset_witnesses" in manifests[name]
               for logical in manifests[name]["payloads"]]
            + [f"storage/{name}/{payload}" for _, _, name, _, _ in families
               for payload in manifests[name]["storage_files"]]
        )
        for directory, prefix, name, group, count in families:
            root = FIXTURES / directory
            manifest = manifests[name]
            with self.subTest(case=f"family/{name}"):
                self.assertTrue(manifest["payloads"])
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
                with self.subTest(case=f"payload/{name}/{payload}"):
                    self.assertEqual(len(fixture[group]), count)
                if prefix in ("cat02-", "input-color-"):
                    current = []
                    for index, row in enumerate(fixture[group]):
                        with self.subTest(case=f"row/{name}/{payload}/{index}"):
                            current.append((row["id"], row["controls"]))
                            self.assertIsInstance(row["controls"], dict)
                    if inputs is not None:
                        self.assertEqual(current, inputs, "all native captures retain identical ordered recipe inputs")
                    inputs = current
                else:
                    for index, row in enumerate(fixture[group]):
                        with self.subTest(case=f"row/{name}/{payload}/{index}"):
                            self.assertIsInstance(row, dict)
            if "preset_witnesses" in manifest:
                referenced.add(manifest["preset_witnesses"])
                witnesses = json.loads((root / manifest["preset_witnesses"]).read_bytes())
                self.assertEqual(set(witnesses), set(manifest["payloads"]))
                for logical, witness in witnesses.items():
                    with self.subTest(case=f"witness/{logical}"):
                        self.assertIn(witness["foreign_capture"], manifest["payloads"])
                        self.assertNotEqual(witness["foreign_capture"], logical)
                        self.assertNotEqual(witness["expected"], witness["rejected"])
            self.assertEqual(referenced, set(manifest["storage_files"]))
            hashes = []
            for payload, expected in manifest["storage_files"].items():
                data = (root / payload).read_bytes()
                actual = hashlib.sha256(data).hexdigest()
                with self.subTest(case=f"storage/{name}/{payload}"):
                    self.assertEqual(actual, expected, payload)
                hashes.append(actual)
            self.assertEqual(len(hashes), len(set(hashes)), "duplicate new payloads need logical aliases")

    test_unique_payload_integrity_and_native_membership.required_subtests = True

    def test_film_input_compound_capture_integrity_and_membership(self):
        root = Path(__file__).parents[1] / "film/input-fixtures"
        manifest = json.loads((root / "manifest.json").read_bytes())
        payload_names = tuple(manifest["payloads"])
        self.subtest_cases = tuple(
            [f"preset/{preset}" for preset in PRESETS]
            + [f"payload/{name}" for name in payload_names]
            + [f"input/{name}/{index}" for name in payload_names for index in range(140)]
        )
        self.assertEqual(set(manifest["presets"]), set(PRESETS))
        self.assertEqual(manifest["cases"], {"base": 96, "controls": 44})
        self.assertEqual(set(manifest["presets"].values()), set(manifest["payloads"]))
        for preset, payload in manifest["presets"].items():
            with self.subTest(case=f"preset/{preset}"):
                self.assertIn(payload, manifest["payloads"])
        for name, record in manifest["payloads"].items():
            data = (root / name).read_bytes()
            with self.subTest(case=f"payload/{name}"):
                self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
            rows = json.loads(data)["rows"]
            self.assertEqual(len(rows), 140)
            base = {(row["route"], row["space"], row["method"], row["decode"]) for row in rows[:96]}
            self.assertEqual(base, {(route, space, method, decode) for route in range(4)
                                   for space in range(4) for method in range(3) for decode in range(2)})
            self.assertEqual([row["control"] for row in rows[96:]], list(range(44)))
            for index, row in enumerate(rows):
                with self.subTest(case=f"input/{name}/{index}"):
                    raw = row["raw"]
                    self.assertEqual([len(raw[field]) for field in ("rgb_to_xyz", "xyz_adapt", "xyz_to_linear_srgb")], [9, 9, 9])
                    self.assertEqual(len(raw["sensitivity"]), 81)
                    self.assertTrue(all(len(sample) == 3 for sample in raw["sensitivity"]))
                    self.assertEqual(len(raw["surface"]), 3)
                    self.assertTrue(all(len(channel) == 15 for channel in raw["surface"]))
                    self.assertNotEqual(raw["hash"], 0)
                    self.assertNotEqual(raw["sensitivity_hash"], 0)
                    self.assertEqual(raw["tc_hash"] == 0, row["method"] == 1)

    test_film_input_compound_capture_integrity_and_membership.required_subtests = True

    def test_expanded_cases_are_independent_and_preserve_exact_words(self):
        self.subtest_cases = ("control/compaction", "control/independent-expansion", "control/invalid-reference")
        words = [0, 0x80000000, 0x7FC00123, 0xFFFFFFFF]
        original = {"recipes": [{"expected": {"sensitivity": words}}, {"expected": {"sensitivity": words}}]}
        compact, blocks = compact_expected_blocks({"native": original}, "blocks.json")
        with self.subTest(case="control/compaction"):
            self.assertEqual(len(blocks["sensitivity"]), 1)
        with self.subTest(case="control/independent-expansion"):
            expanded = expand_expected_blocks(compact["native"], blocks)
            self.assertEqual(expanded, original)
            expanded["recipes"][0]["expected"]["sensitivity"][0] = 17
            self.assertEqual(expanded["recipes"][1]["expected"]["sensitivity"], words)
            self.assertEqual(blocks["sensitivity"][0], words)
        compact["native"]["recipes"][0]["expected"]["sensitivity_record"] = -1
        with self.subTest(case="control/invalid-reference"), self.assertRaisesRegex(ValueError, "invalid expected block"):
            expand_expected_blocks(compact["native"], blocks)

    test_expanded_cases_are_independent_and_preserve_exact_words.required_subtests = True


if __name__ == "__main__":
    unittest.main()
