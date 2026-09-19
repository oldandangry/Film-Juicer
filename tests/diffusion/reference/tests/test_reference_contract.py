from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
TOOL = ROOT / "tests/diffusion/reference"
BOOTSTRAP = TOOL / "bootstrap_reference_env.sh"
MANIFEST = TOOL / "fixtures/diffusion_reference_v1.json"
ARRAYS = TOOL / "fixtures/diffusion_reference_v1.npz"


class BootstrapContractTests(unittest.TestCase):
    def test_bootstrap_uses_managed_python_and_normal_import(self) -> None:
        text = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn("uv python install 3.13", text)
        self.assertIn("uv venv --clear --managed-python --python 3.13", text)
        self.assertIn("uv pip sync", text)
        self.assertIn("uv pip install", text)
        self.assertIn("from spektrafilm.model.diffusion import", text)
        for forbidden in ("importlib.util", "sys.modules", "ModuleType", "monkeypatch"):
            self.assertNotIn(forbidden, text)

    def test_lock_is_hashed_and_contains_required_dependency(self) -> None:
        text = (TOOL / "requirements.lock").read_text(encoding="utf-8")
        self.assertIn("exiv2==", text)
        self.assertIn("--hash=sha256:", text)


class PromotedReferenceContractTests(unittest.TestCase):
    def test_promoted_fixture_contract(self) -> None:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            set(data),
            {
                "schema",
                "array_schema",
                "family_order",
                "semantic_channel_order",
                "python",
                "uv",
                "dependency_lock",
                "python_freeze",
                "repository_git",
                "reference_git",
                "source_files",
                "strength_cases",
                "family_cases",
                "warmth_cases",
                "advanced_cases",
                "psf_cases",
                "image_cases",
                "radius_cases",
                "hdr_cases",
                "arrays",
            },
        )
        self.assertEqual(data["schema"], "film-juicer.diffusion-reference.v1")
        self.assertEqual(
            data["array_schema"],
            "film-juicer.diffusion-reference-arrays.v1",
        )
        self.assertEqual(
            data["family_order"],
            ["glimmerglass", "black_pro_mist", "pro_mist", "cinebloom"],
        )
        self.assertEqual(data["semantic_channel_order"], ["R", "G", "B"])
        self.assertEqual(data["python"]["major_minor"], "3.13")
        self.assertTrue(data["source_files"])
        self.assertTrue(ARRAYS.exists())

    def test_environment_provenance_matches_current_evidence(self) -> None:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            set(data["python"]),
            {"major_minor", "version", "implementation", "executable", "platform", "machine"},
        )
        self.assertEqual(data["python"]["major_minor"], "3.13")
        self.assertTrue(data["python"]["version"].startswith("3.13."))
        self.assertEqual(data["python"]["implementation"], "CPython")
        self.assertEqual(data["python"]["executable"], ".tmp/diffusion/reference-venv/bin/python")
        self.assertTrue(data["python"]["platform"])
        self.assertTrue(data["python"]["machine"])

        for key in ("uv", "dependency_lock", "python_freeze"):
            row = data[key]
            path = ROOT / row["path" if key != "uv" else "evidence_path"]
            self.assertTrue(path.is_file(), row)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
        self.assertEqual(set(data["uv"]), {"version", "evidence_path", "sha256"})
        self.assertTrue(data["uv"]["version"].startswith("uv 0.11."))
        self.assertEqual(set(data["dependency_lock"]), {"path", "sha256"})
        self.assertEqual(
            set(data["python_freeze"]),
            {"path", "sha256", "line_count"},
        )
        freeze = (ROOT / data["python_freeze"]["path"]).read_text(encoding="utf-8")
        self.assertEqual(len(freeze.splitlines()), data["python_freeze"]["line_count"])
        self.assertIn(f"-e file://{ROOT / 'external/spektrafilm'}", freeze.splitlines())

        repository_commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        reference_commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT / "external/spektrafilm",
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(data["repository_git"], {"commit": repository_commit})
        self.assertEqual(
            data["reference_git"],
            {"commit": reference_commit, "status_porcelain": ""},
        )

    def test_control_expansions_have_closed_consumed_schemas(self) -> None:
        import numpy as np

        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        families = data["family_order"]
        self.assertEqual([row["family"] for row in data["family_cases"]], families)

        for row in data["strength_cases"]:
            self.assertEqual(set(row), {"family", "strength", "scatter_fraction"})
            self.assertIn(row["family"], families)
            self.assertTrue(np.isfinite(row["scatter_fraction"]))
            self.assertGreaterEqual(row["scatter_fraction"], 0.0)
            self.assertLess(row["scatter_fraction"], 1.0)

        for row in data["family_cases"]:
            self.assertEqual(
                set(row),
                {"family", "w_c", "w_h", "w_b", "halo_warmth_base", "core", "halo", "bloom"},
            )
            self.assertAlmostEqual(row["w_c"] + row["w_h"] + row["w_b"], 1.0, places=14)
            for kind in ("core", "halo", "bloom"):
                group = row[kind]
                expected = {
                    "lambda_um",
                    "spread",
                    "n_components",
                    "expanded_lambdas_um",
                    "expanded_weights",
                }
                if kind == "bloom":
                    expected.add("alpha")
                self.assertEqual(set(group), expected)
                self.assertEqual(len(group["expanded_lambdas_um"]), group["n_components"])
                self.assertEqual(len(group["expanded_weights"]), group["n_components"])
                self.assertAlmostEqual(sum(group["expanded_weights"]), 1.0, places=14)

        expected_warmth = {
            (family, value)
            for family in families
            for value in (-3.0, -1.5, 0.0, 1.5, 3.0)
        }
        self.assertEqual(
            {(row["family"], row["halo_warmth"]) for row in data["warmth_cases"]},
            expected_warmth,
        )
        for row in data["warmth_cases"]:
            self.assertEqual(
                set(row),
                {
                    "family",
                    "halo_warmth",
                    "family_base",
                    "effective_unclamped",
                    "effective_clamped",
                    "channel_weights_rgb",
                },
            )
            self.assertAlmostEqual(
                row["effective_unclamped"],
                row["family_base"] + row["halo_warmth"],
                places=14,
            )
            self.assertGreaterEqual(row["effective_clamped"], -1.5)
            self.assertLessEqual(row["effective_clamped"], 1.5)
            self.assertEqual(len(row["channel_weights_rgb"]), 3)
            for channel in row["channel_weights_rgb"]:
                self.assertAlmostEqual(sum(channel), 1.0, places=14)

        expected_advanced = {
            (family, name)
            for family in families
            for name in (
                "default",
                "proportional",
                "all_zero_nondefault_sizes",
                "runtime_outside_widget",
            )
        }
        self.assertEqual(
            {(row["family"], row["name"]) for row in data["advanced_cases"]},
            expected_advanced,
        )
        for row in data["advanced_cases"]:
            self.assertEqual(
                set(row),
                {
                    "family",
                    "name",
                    "overrides",
                    "resolved_w_c",
                    "resolved_w_h",
                    "resolved_w_b",
                    "core",
                    "halo",
                    "bloom",
                },
            )
            self.assertEqual(
                set(row["overrides"]),
                {
                    "core_intensity",
                    "halo_intensity",
                    "bloom_intensity",
                    "core_size",
                    "halo_size",
                    "bloom_size",
                },
            )
            self.assertAlmostEqual(
                row["resolved_w_c"] + row["resolved_w_h"] + row["resolved_w_b"],
                1.0,
                places=14,
            )
            for kind in ("core", "halo", "bloom"):
                self.assertTrue(row[kind]["expanded_lambdas_um"])
                self.assertAlmostEqual(sum(row[kind]["expanded_weights"]), 1.0, places=14)

        base_by_family = {row["family"]: row for row in data["family_cases"]}
        advanced_by_key = {
            (row["family"], row["name"]): row for row in data["advanced_cases"]
        }
        for family in families:
            base = base_by_family[family]
            for fallback_name in ("default", "all_zero_nondefault_sizes"):
                fallback = advanced_by_key[(family, fallback_name)]
                self.assertEqual(
                    (fallback["resolved_w_c"], fallback["resolved_w_h"], fallback["resolved_w_b"]),
                    (base["w_c"], base["w_h"], base["w_b"]),
                )
                for kind in ("core", "halo", "bloom"):
                    self.assertEqual(fallback[kind], base[kind])
            outside = advanced_by_key[(family, "runtime_outside_widget")]
            self.assertAlmostEqual(
                outside["core"]["lambda_um"],
                base["core"]["lambda_um"] * 1.0e-6,
                places=14,
            )
            self.assertAlmostEqual(
                outside["halo"]["lambda_um"],
                base["halo"]["lambda_um"] * 6.0,
                places=14,
            )
            self.assertAlmostEqual(
                outside["bloom"]["lambda_um"],
                base["bloom"]["lambda_um"] * 0.05,
                places=14,
            )
        self.assertTrue(
            any(
                value == 0.0
                for row in data["warmth_cases"]
                for channel in row["channel_weights_rgb"]
                for value in channel
            )
        )

    def test_reference_cases_cover_families_controls_stages_and_edges(self) -> None:
        import numpy as np

        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        strength = {(row["family"], row["strength"]) for row in data["strength_cases"]}
        for family in data["family_order"]:
            for value in (-1.0, 0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 1.0, 2.0, 4.0):
                self.assertIn((family, value), strength)

        self.assertEqual({row["stage"] for row in data["image_cases"]}, {"camera", "enlarger"})
        self.assertEqual({row["family"] for row in data["image_cases"]}, set(data["family_order"]))
        self.assertTrue(
            {"center", "asymmetric", "constant", "edge", "corner", "random"}
            <= {row["pattern"] for row in data["image_cases"]}
        )
        self.assertEqual(
            {tuple(row["extent"]) for row in data["radius_cases"]},
            {(16, 18), (17, 19), (23, 11), (11, 23)},
        )
        self.assertEqual({row["stage"] for row in data["hdr_cases"]}, {"camera", "enlarger"})
        self.assertEqual(
            {
                (row["family"], row["halo_warmth"], row["advanced_case"])
                for row in data["psf_cases"]
            },
            {
                (family, warmth, advanced)
                for family in data["family_order"]
                for warmth in (-3.0, -1.5, 0.0, 1.5, 3.0)
                for advanced in (
                    "default",
                    "proportional",
                    "all_zero_nondefault_sizes",
                    "runtime_outside_widget",
                )
            },
        )

        with np.load(ARRAYS) as arrays:
            for row in data["psf_cases"]:
                self.assertEqual(
                    set(row),
                    {
                        "family",
                        "halo_warmth",
                        "advanced_case",
                        "overrides",
                        "extent",
                        "pixel_size_um",
                        "spatial_scale",
                        "array",
                        "semantic_channel_order",
                    },
                )
                self.assertEqual(row["extent"], [17, 19])
                self.assertEqual(row["pixel_size_um"], 1000.0)
                self.assertEqual(row["spatial_scale"], 1.0)
                self.assertEqual(row["semantic_channel_order"], ["R", "G", "B"])
                psf = arrays[row["array"]]
                self.assertEqual(psf.dtype, np.dtype("float64"))
                np.testing.assert_allclose(psf.sum(axis=(0, 1)), np.ones(3), atol=5.0e-12)
            for row in data["image_cases"] + data["radius_cases"] + data["hdr_cases"]:
                base_keys = {
                    "stage",
                    "family",
                    "pattern",
                    "extent",
                    "pixel_size_um",
                    "params",
                    "input_array",
                    "output_array",
                    "input_dtype",
                    "output_dtype",
                    "psf_radius",
                    "semantic_channel_order",
                }
                expected_keys = base_keys | (
                    {"extent_class", "image_radius_cap"} if row in data["radius_cases"] else set()
                )
                self.assertEqual(set(row), expected_keys)
                self.assertEqual(row["pixel_size_um"], 1000.0)
                self.assertEqual(row["semantic_channel_order"], ["R", "G", "B"])
                self.assertEqual(
                    set(row["params"]),
                    {
                        "active",
                        "filter_family",
                        "strength",
                        "spatial_scale",
                        "halo_warmth",
                        "core_intensity",
                        "core_size",
                        "halo_intensity",
                        "halo_size",
                        "bloom_intensity",
                        "bloom_size",
                    },
                )
                self.assertEqual(
                    row["params"],
                    {
                        "active": True,
                        "filter_family": row["family"],
                        "strength": 0.5,
                        "spatial_scale": 1.0,
                        "halo_warmth": 0.0,
                        "core_intensity": 1.0,
                        "core_size": 1.0,
                        "halo_intensity": 1.0,
                        "halo_size": 1.0,
                        "bloom_intensity": 1.0,
                        "bloom_size": 1.0,
                    },
                )
                self.assertEqual(row["input_dtype"], "float32")
                self.assertEqual(row["output_dtype"], "float32")
                input_array = arrays[row["input_array"]]
                output_array = arrays[row["output_array"]]
                self.assertEqual(input_array.dtype, np.dtype("float32"))
                self.assertEqual(output_array.dtype, np.dtype("float32"))
                self.assertEqual(list(input_array.shape[:2]), row["extent"])
                self.assertEqual(input_array.shape, output_array.shape)
                self.assertGreaterEqual(row["psf_radius"], 1)
                if "image_radius_cap" in row:
                    self.assertLessEqual(row["psf_radius"], row["image_radius_cap"])
                    expected_extent_class = {
                        (16, 18): "even",
                        (17, 19): "odd",
                        (23, 11): "portrait",
                        (11, 23): "landscape",
                    }[tuple(row["extent"])]
                    self.assertEqual(row["extent_class"], expected_extent_class)
            even_odd_radii = {
                row["psf_radius"]
                for row in data["radius_cases"]
                if tuple(row["extent"]) in {(16, 18), (17, 19)}
            }
            self.assertEqual(even_odd_radii, {6, 7})
            for row in data["radius_cases"]:
                if tuple(row["extent"]) in {(23, 11), (11, 23)}:
                    self.assertEqual(row["psf_radius"], row["image_radius_cap"])
            hdr = arrays[data["hdr_cases"][0]["input_array"]].reshape(-1)
            np.testing.assert_array_equal(
                hdr[:5],
                np.array(
                    [
                        0.0,
                        np.nextafter(np.float32(0.0), np.float32(1.0)),
                        0.5e-10,
                        1.0e-10,
                        2.0e-10,
                    ],
                    dtype=np.float32,
                ),
            )
            self.assertGreaterEqual(float(hdr.min()), 0.0)
            self.assertLessEqual(float(hdr.max()), 16.0)
            self.assertTrue(all(np.isfinite(arrays[name]).all() for name in arrays.files))

    def test_fixture_provenance_uses_real_imported_files(self) -> None:
        import sys

        from tools.diffusion_reference import generate_reference

        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source_paths = {row["path"] for row in data["source_files"]}
        source_root = (ROOT / "external/spektrafilm").resolve()
        imported_source_modules = {
            name
            for name, module in sys.modules.items()
            if (name == "spektrafilm" or name.startswith("spektrafilm."))
            and getattr(module, "__file__", None)
            and source_root in Path(module.__file__).resolve().parents
            and Path(module.__file__).suffix == ".py"
        }
        recorded_modules = {row["module"] for row in data["source_files"]}
        self.assertTrue(imported_source_modules <= recorded_modules)
        self.assertIn(
            "external/spektrafilm/src/spektrafilm/model/diffusion.py",
            source_paths,
        )
        self.assertIn(
            "external/spektrafilm/src/spektrafilm/runtime/params_schema.py",
            source_paths,
        )
        for row in data["source_files"]:
            self.assertEqual(set(row), {"module", "path", "sha256"})
            path = ROOT / row["path"]
            self.assertTrue(path.is_file(), row)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
        self.assertNotIn("scripts/spektrafilm_validate_exact_optics.py", source_paths)
        self.assertEqual(len(data["reference_git"]["commit"]), 40)
        self.assertEqual(data["reference_git"]["status_porcelain"], "")
        self.assertEqual(len(data["repository_git"]["commit"]), 40)

    def test_array_manifest_matches_promoted_payload(self) -> None:
        import numpy as np

        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        with np.load(ARRAYS) as arrays:
            self.assertEqual(set(arrays.files), {row["name"] for row in data["arrays"]})
            for row in data["arrays"]:
                self.assertEqual(set(row), {"name", "shape", "dtype", "sha256"})
                array = arrays[row["name"]]
                self.assertEqual(list(array.shape), row["shape"])
                self.assertEqual(array.dtype.name, row["dtype"])
                self.assertEqual(
                    hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                    row["sha256"],
                )


if __name__ == "__main__":
    unittest.main()
