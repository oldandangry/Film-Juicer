"""Compile real consumer examples against a disposable copy of the Rust workspace."""

from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures/rust_boundaries"
SPEC = importlib.util.spec_from_file_location("boundary_quality", ROOT / "scripts/check-quality.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the tracked quality contract")
quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)


class RustBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cargo = os.environ["JUICER_CARGO"]
        cls.target = os.environ["JUICER_RUST_TARGET"]
        cls.target_dir = Path(os.environ["JUICER_RUST_PROBE_DIR"]).resolve()
        cls.profiles = {"debug": (False,), "release": (True,), "both": (False, True)}[
            os.environ.get("JUICER_RUST_PROBE_PROFILE", "both")
        ]
        artifact_root = Path(os.environ["JUICER_TEST_ARTIFACT_DIR"]).resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        cls.artifacts = Path(tempfile.mkdtemp(prefix="run-", dir=artifact_root))
        directory = tempfile.TemporaryDirectory(prefix="probe-", dir=cls.artifacts)
        cls.addClassCleanup(directory.cleanup)
        cls.workspace = Path(directory.name)
        for name in ("Cargo.toml", "Cargo.lock", "clippy.toml", "rust-toolchain.toml"):
            shutil.copy2(ROOT / name, cls.workspace / name)
        shutil.copytree(ROOT / "rust", cls.workspace / "rust")
        cls.environment = quality.cargo_environment()
        cls.invocation = 0
        cls.invocation_lock = threading.Lock()

    def command(self, arguments: list[str], label: str) -> subprocess.CompletedProcess:
        with self.invocation_lock:
            type(self).invocation += 1
            invocation = self.invocation
        command = [self.cargo, *arguments]
        quality.check_probe_configuration(ROOT, self.workspace)
        result = subprocess.run(
            command, cwd=self.workspace, env=self.environment,
            capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
        )
        log = self.artifacts / f"{invocation:02d}-{self._testMethodName}-{label}.log"
        log.write_text(
            subprocess.list2cmdline(command) + "\n" + result.stdout + result.stderr,
            encoding="utf-8",
        )
        quality.check_probe_configuration(ROOT, self.workspace)
        return result

    def attach(self, owner: str, fixture: str) -> None:
        source = self.workspace / "rust" / owner
        original = source.read_text(encoding="utf-8")
        self.addCleanup(source.write_text, original, encoding="utf-8")
        probe = source.parent / "quality_boundary_probe.rs"
        shutil.copy2(FIXTURES / f"{fixture}.rs", probe)
        self.addCleanup(probe.unlink)
        # Absolute paths let the same fixture enter a crate root or a protected
        # child module. The owner retains its real lint and visibility context.
        source.write_text(
            original + '\n#[cfg(test)]\n'
            '#[allow(dead_code, reason = "compiler-only boundary examples")]\n'
            f'#[path = {json.dumps(probe.as_posix())}]\nmod quality_boundary_probe;\n',
            encoding="utf-8",
        )

    def check(self, package: str, release: bool, test_support: bool = False) -> tuple[int, list[dict]]:
        arguments = [
            "check", "--locked", "--offline", "--package", package, "--all-targets",
            "--target", self.target, "--target-dir",
            str(self.target_dir / ("test-support" if test_support else "default")),
            "--message-format=json",
        ]
        if release:
            arguments.append("--release")
        if test_support:
            arguments += ["--features", "test-support"]
        label = f"{package}-{'release' if release else 'dev'}-{'test-support' if test_support else 'default'}"
        result = self.command(arguments, label)
        diagnostics = [
            message["message"]
            for line in result.stdout.splitlines()
            if (message := json.loads(line)).get("reason") == "compiler-message"
        ]
        return result.returncode, diagnostics

    def check_features(self, package: str, release: bool) -> list[tuple[int, list[dict]]]:
        # Both commands read one fixed fixture, with separate Cargo output owners.
        # Join both before assertions or cleanup, including when either command fails.
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.check, package, release, support) for support in (False, True)]
            errors = [future.exception() for future in futures]
            for error in errors:
                if error is not None:
                    raise error
            return [future.result() for future in futures]

    def assert_rejected(self, status: int, diagnostics: list[dict], expected: dict[str, tuple[str, ...]]) -> None:
        self.assertNotEqual(status, 0, "prohibited consumer compiled successfully")
        for code, markers in expected.items():
            source_lines = [
                line["text"]
                for diagnostic in diagnostics
                if diagnostic["level"] == "error" and (diagnostic.get("code") or {}).get("code") == code
                for span in diagnostic["spans"]
                if span["is_primary"] and span["file_name"].endswith("quality_boundary_probe.rs")
                for line in span["text"]
            ]
            for marker in markers:
                self.assertTrue(
                    any(marker in line for line in source_lines),
                    f"missing intended {code} at {marker}: {diagnostics}",
                )

    def test_workspace_dependencies_and_build_hooks(self) -> None:
        quality.check_workspace_contract(self.workspace)
        result = self.command(
            ["metadata", "--locked", "--offline", "--format-version", "1"],
            "metadata",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        quality.check_dependency_contract(self.workspace, json.loads(result.stdout))

    def test_valid_profile_consumers_compile(self) -> None:
        self.attach("film-juicer-plugin/src/lib.rs", "accepted")
        for release in self.profiles:
            for support, (status, diagnostics) in zip((False, True), self.check_features("film-juicer-plugin", release), strict=True):
                with self.subTest(release=release, test_support=support):
                    self.assertEqual(status, 0, diagnostics)

    def test_profile_views_cannot_outlive_their_owners(self) -> None:
        self.attach("film-juicer-plugin/src/lib.rs", "borrowed_views")
        for release in self.profiles:
            status, diagnostics = self.check("film-juicer-plugin", release)
            self.assert_rejected(status, diagnostics, {
                "E0515": ("film_owner.view()", "print_owner.view()", "spectra_owner.view()", "mallett_owner.samples()", "cmf_owner.rows()", "csv_owner.rows()", "noise_owner.view()"),
                "E0505": ("drop(film_owner)", "drop(print_owner)", "drop(spectra_owner)", "drop(mallett_owner)", "drop(cmf_owner)", "drop(csv_owner)", "drop(noise_owner)"),
            })

    def test_film_input_color_is_complete_and_immutable(self) -> None:
        for fixture, expected in (
            ("input_color_complete", None),
            ("input_color_default", {"E0599": ("InputColor::default()", "Recipe::default()")}),
            ("input_color_private", {"E0616": ("color.rgb_to_xyz", "recipe.color")}),
            ("input_color_mutation", {"E0594": ("color.rgb_to_xyz()[0] = 1.0",)}),
            ("input_color_construction", {"E0624": ("InputColor::from_matrices",)}),
            ("input_color_literal", {"E0451": ("InputColor { ..color }",)}),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    for status, diagnostics in self.check_features("film-juicer-plugin", release):
                        if expected is None:
                            self.assertEqual(status, 0, diagnostics)
                        else:
                            self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_output_color_is_complete_and_immutable(self) -> None:
        for fixture, expected in (
            ("output_color_complete", None),
            ("output_color_private", {"E0616": ("color.hash",)}),
            ("output_color_default", {"E0599": ("OutputColor::default()",)}),
            ("output_color_construction", {"E0451": ("OutputColor { ..*output_color",)}),
            ("output_color_mutation", {"E0594": ("color.rgb_to_xyz()[0] = 1.0",)}),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    for status, diagnostics in self.check_features("film-juicer-plugin", release):
                        if expected is None:
                            self.assertEqual(status, 0, diagnostics)
                        else:
                            self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_film_development_is_complete_private_and_borrowed(self) -> None:
        for fixture, expected in (
            ("film_complete", None),
            ("film_private", {"E0616": ("development.hash", "development.profile")}),
            ("film_default", {"E0599": ("Development::default()",)}),
            ("film_construction", {"E0451": ("Development { ..development }",)}),
            ("film_readonly", {"E0594": ("development.gamma_rgb()[0] = 1.0", "development.normalized_curves_cmy()[0][0] = 0.0")}),
            ("film_borrowed", {"E0515": ("development.log_exposure()",), "E0505": ("drop(development)",)}),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    for status, diagnostics in self.check_features("film-juicer-plugin", release):
                        if expected is None:
                            self.assertEqual(status, 0, diagnostics)
                        else:
                            self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_output_gamut_completed_api_is_private_and_borrowed(self) -> None:
        for fixture, expected in (
            ("output_gamut_private", {"E0616": ("transform.hash", "boundary.cmax[0]")}),
            ("output_gamut_default", {"E0599": ("OutputTransform::default()", "OutputBoundary::default()")}),
            ("output_gamut_construction", {"E0451": ("OutputTransform { ..transform }", "OutputBoundary { ..boundary }")}),
            ("output_gamut_borrowed", {"E0515": ("boundary.cmax()",), "E0505": ("drop(boundary)", "drop(transform)")}),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    status, diagnostics = self.check("film-juicer-plugin", release)
                    self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_print_illuminant_is_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("print_illuminant_complete", None),
            ("print_illuminant_construction", {"E0451": ("print::Illuminant { ..illuminant }",)}),
            ("print_illuminant_default", {"E0599": ("print::Illuminant::default()",)}),
            ("print_illuminant_private", {"E0616": ("illuminant.samples",)}),
            ("print_illuminant_readonly", {"E0594": ("illuminant.samples()[0] = 0.0",)}),
            ("print_illuminant_owned", None),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    for status, diagnostics in self.check_features("film-juicer-plugin", release):
                        if expected is None:
                            self.assertEqual(status, 0, diagnostics)
                        else:
                            self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_optics_are_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("optics_complete", None),
            ("optics_default", {"E0599": ("optics::Recipe::default()", "optics::Diffusion::default()", "optics::LensBlur::default()", "optics::Halation::default()")}),
            ("optics_private", {
                "E0616": ("recipe.camera_diffusion", "recipe.camera_lens_blur().sigma_um", "recipe.scatter_halation().first_sigma_um"),
                "E0624": ("optics::Diffusion::inactive()", "optics::LensBlur::new", "optics::Halation::new"),
            }),
            ("optics_construction", {"E0451": ("optics::Recipe { ..recipe }", "optics::Diffusion { ..diffusion }", "optics::LensBlur { ..lens }", "optics::Halation { ..halation }")}),
            ("optics_readonly", {"E0594": ("*recipe.camera_diffusion() = diffusion", "*recipe.camera_lens_blur() = lens", "*recipe.scatter_halation() = halation")}),
            ("optics_owned", None),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    for status, diagnostics in self.check_features("film-juicer-plugin", release):
                        if expected is None:
                            self.assertEqual(status, 0, diagnostics)
                        else:
                            self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_effects_are_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("effects_complete", None),
            ("effects_default", {"E0599": ("effects::Recipe::default()", "effects::Dust::default()", "effects::Scratch::default()")}),
            ("effects_private", {
                "E0616": ("recipe.film_dust", "dust.cell_width_mm", "scratch.length_min_mm"),
                "E0624": ("effects::Recipe::zero()", "effects::Dust::new", "effects::Scratch::new"),
            }),
            ("effects_construction", {"E0451": ("effects::Recipe { ..recipe }", "effects::Dust { ..dust }", "effects::Scratch { ..scratch }")}),
            ("effects_readonly", {"E0594": ("*recipe.film_dust() = dust", "*recipe.gate_scratch() = scratch")}),
            ("effects_owned", None),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    for status, diagnostics in self.check_features("film-juicer-plugin", release):
                        if expected is None:
                            self.assertEqual(status, 0, diagnostics)
                        else:
                            self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_film_grain_is_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("film_grain_complete", None),
            ("film_grain_default", {"E0599": ("grain::Recipe::default()",)}),
            ("film_grain_private", {"E0616": ("recipe.particle_density_min_cmy", "recipe.axis_block_prefix_max_cmy")}),
            ("film_grain_construction", {"E0451": ("grain::Recipe { ..recipe }",)}),
            ("film_grain_readonly", {"E0594": ("recipe.particle_scale_cmy()[0] = 0.0", "recipe.axis_block_prefix_max_cmy()[0][0] = 0.0")}),
            ("film_grain_owned", None),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    for status, diagnostics in self.check_features("film-juicer-plugin", release):
                        if expected is None:
                            self.assertEqual(status, 0, diagnostics)
                        else:
                            self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_film_dir_density_is_complete_private_and_readonly(self) -> None:
        for fixture, expected in (
            ("film_dir_complete", None),
            ("film_dir_default", {"E0599": ("Recipe::default()", "Minimum::default()", "Bounds::default()")}),
            ("film_dir_private", {"E0616": ("recipe.compensated_axes_rgb", "minimum.cmy", "bounds.hash")}),
            ("film_dir_construction", {"E0451": ("Recipe { ..recipe }", "Minimum { ..minimum }", "Bounds { ..bounds }")}),
            ("film_dir_readonly", {"E0594": ("recipe.matrix_rgb()[0][0] = 0.0", "minimum.cmy()[0] = 0.0", "bounds.minimum_cmy()[0] = 0.0")}),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    for status, diagnostics in self.check_features("film-juicer-plugin", release):
                        if expected is None:
                            self.assertEqual(status, 0, diagnostics)
                        else:
                            self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_completed_profile_storage_is_private_to_its_owner(self) -> None:
        # The plugin is a real external consumer of the core, and a sibling of
        # asset_profile; inserting the probe inside the owner would bypass privacy.
        self.attach("film-juicer-plugin/src/lib.rs", "private_storage")
        for release in self.profiles:
            status, diagnostics = self.check("film-juicer-plugin", release)
            self.assert_rejected(status, diagnostics, {
                "E0616": ("film.processing_defaults", "tables.interpolation_log_exposure", "film_owner.profile", "print_owner.profile", "spectra_owner.lut", "mallett_owner.basis", "cmf_owner.rows", "csv_owner.rows", "noise_owner.bundle"),
            })

    def test_noise_borrows_and_storage_under_test_support(self) -> None:
        for fixture, expected in (
            ("noise_borrowed", {"E0515": ("noise_owner.view()",), "E0505": ("drop(noise_owner)",)}),
            ("noise_private", {"E0616": ("noise_owner.bundle",)}),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    status, diagnostics = self.check("film-juicer-plugin", release, True)
                    self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_completed_profile_construction_cannot_be_bypassed(self) -> None:
        self.attach("film-juicer-plugin/src/lib.rs", "private_construction")
        for release in self.profiles:
            status, diagnostics = self.check("film-juicer-plugin", release)
            self.assert_rejected(status, diagnostics, {
                "E0451": ("FilmProfile { ..film }", "PrintProfile { ..print }"),
            })

    def test_lens_construction_and_storage_are_private(self) -> None:
        # Type-check errors can suppress later field-privacy diagnostics, so
        # each rejected construction/access is its own compiler consumer.
        for fixture, expected in (
            ("illuminant_private", {"E0451": ("LensInput {",)}),
            ("illuminant_default", {"E0599": ("LensInput::default()",)}),
            ("illuminant_mutation", {"E0616": ("input.kg3[0]",)}),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    status, diagnostics = self.check("film-juicer-plugin", release)
                    self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_deleted_film_development_definitions_and_foundation_policy(self) -> None:
        import re
        recipe = (ROOT / "src/RenderRecipe.cpp").read_text(encoding="utf-8-sig")
        def definition(name: str, source: str) -> bool:
            # Declarations and names in diagnostic strings are not definitions.
            return re.search(r"\b" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{", source) is not None
        for name in ("hash_film_develop_recipe", "hash_density_curves_layers", "visual_grain_requires_density_layers"):
            self.assertFalse(definition(name, recipe), name)
            self.assertTrue(definition(name, "std::uint64_t " + name + "(const FilmDevelopRecipe& recipe) { return 1; }"))
            self.assertFalse(definition(name, "std::uint64_t " + name + "(const FilmDevelopRecipe& recipe);"))
        start = recipe.index("        bool build_film_foundation(")
        end = recipe.index("    } // namespace", start)
        foundation = recipe[start:end]
        for name in ("normalize_density_curves", "hash_nan_preserving_floats", "resolve_scan_route"):
            call = r"\b" + name + r"\s*\("
            self.assertNotRegex(foundation, call)
            self.assertRegex("auto value = " + name + "(source);", call)
        self.assertIn("recipe.filmDevelop = std::move(input.filmDevelop)", foundation)
        self.assertTrue(definition("normalize_density_curves", recipe))
        self.assertFalse(definition("hash_nan_preserving_floats", recipe))
        # Print-medium bounds retain native normalization; DIR span hashing has moved.
        self.assertGreaterEqual(len(re.findall(r"\bnormalize_density_curves\s*\(", recipe)), 2)
        state = (ROOT / "src/JuicerState.cpp").read_text(encoding="utf-8-sig")
        self.assertIn("params.grainControls.active && params.grainControls.sublayersActive", state)
        self.assertEqual(state.count("film_foundation_input_from_snapshot(params, *selected.filmSource, assets, input.film, outError)"), 2)

    def test_deleted_dir_density_producers_and_stage_order(self) -> None:
        import re
        recipe = (ROOT / "src/RenderRecipe.cpp").read_text(encoding="utf-8-sig")
        def definition(name: str, source: str) -> bool:
            return re.search(r"\b" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{", source) is not None
        for name in ("build_grain_contract", "build_dir_couplers_recipe", "evaluate_dir_langmuir",
                     "hash_dir_couplers_recipe", "build_film_density_bounds", "hash_nan_preserving_floats",
                     "build_visual_grain_recipe", "hash_visual_grain_recipe",
                     "derive_grain_layer_axis_search_metadata", "finite_nonnegative_grain_triplet",
                     "finite_positive_triplet", "finite_nonnegative_pair",
                     "normalize_effect_amount", "hash_defect_policy",
                     "hash_film_juicer_effects_recipe", "build_film_juicer_effects_recipe"):
            self.assertFalse(definition(name, recipe), name)
            self.assertTrue(definition(name, "bool " + name + "(const Recipe& recipe) { return true; }"))
            self.assertFalse(definition(name, "bool " + name + "(const Recipe& recipe);"))
        for name in ("hash_density_bounds_recipe", "build_print_density_bounds"):
            self.assertTrue(definition(name, recipe), name)
            self.assertGreaterEqual(len(re.findall(r"\b" + name + r"\s*\(", recipe)), 2)
        start = recipe.index("        bool build_film_foundation(")
        end = recipe.index("    } // namespace", start)
        foundation = recipe[start:end]
        stages = [foundation.index(name) for name in ("prepare_spatial_optics(",
                  "prepare_film_density_minimum(", "prepare_visual_grain(",
                  "prepare_physical_effects(", "prepare_film_dir(", "prepare_film_density_bounds(")]
        self.assertEqual(stages, sorted(stages))
        self.assertEqual(recipe.count("JuicerAssets::prepare_physical_effects("), 1)
        self.assertLess(foundation.index("recipe.filmDevelop = std::move(input.filmDevelop)"), foundation.index("prepare_visual_grain("))
        self.assertIn("input.grainContract, recipe.grainContract", foundation)
        bounds = foundation[stages[-1]:]
        self.assertIn("input.grainContract", bounds)
        self.assertNotIn("recipe.grainContract", bounds)

    def test_deleted_spatial_optics_producers_and_retained_consumers(self) -> None:
        import re
        sources = {name: (ROOT / ("src/" + name)).read_text(encoding="utf-8-sig")
                   for name in ["RenderRecipe.cpp", "DiffusionHostBehavior.cpp", "ScatterHalation.cpp"]}
        def definition(name: str, text: str) -> bool:
            return re.search(r"\b" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{", text) is not None
        for file, names in [("RenderRecipe.cpp", ["build_spatial_optics_recipe", "hash_diffusion_filter_optics", "hash_camera_lens_blur_optics"]),
                            ("DiffusionHostBehavior.cpp", ["resolve_diffusion_filter", "resolved_hash", "strength_to_scatter", "validate_authored_finite"]),
                            ("ScatterHalation.cpp", ["resolve_scatter_halation_recipe", "hash_scatter_halation_recipe", "any_positive"])]:
            for name in names:
                self.assertFalse(definition(name, sources[file]), name)
                self.assertTrue(definition(name, "bool " + name + "(const Params& params) { return true; }"))
                self.assertFalse(definition(name, "bool " + name + "(const Params& params);"))
        for name in ["hash_diffusion_authored_controls", "build_diffusion_psf_sample_descriptor", "expand_diffusion_psf_components"]:
            self.assertTrue(definition(name, sources["DiffusionHostBehavior.cpp"]), name)
        for name in ["build_scatter_halation_controls", "build_scatter_halation_frame_descriptor", "build_gaussian_descriptor"]:
            self.assertTrue(definition(name, sources["ScatterHalation.cpp"]), name)
        recipe = sources["RenderRecipe.cpp"]
        self.assertEqual(recipe.count("JuicerAssets::prepare_spatial_optics("), 1)
        header = (ROOT / "src/RenderRecipe.h").read_text(encoding="utf-8-sig")
        self.assertRegex(
            header,
            r'\bkExactOpticsNotImplementedForPhase6\[\]\s*=\s*"ExactOpticsNotImplementedForPhase6";',
        )
        preflight = recipe[recipe.index("    bool preflight_camera_lens_blur("):]
        self.assertRegex(
            preflight,
            r'outDiagnostic\s*=\s*std::string\(kExactOpticsNotImplementedForPhase6\)'
            r'\s*\+\s*" component=camera_lens_blur";',
        )
        foundation = recipe[recipe.index("        bool build_film_foundation("):]
        self.assertLess(foundation.index("prepare_input("), foundation.index("prepare_spatial_optics("))
        self.assertLess(foundation.index("prepare_spatial_optics("), foundation.index("prepare_film_density_minimum("))
        for file in ["diffusion/host_reference_test.cpp", "scatter_halation/recipe_descriptor_test.cpp", "scatter_halation/integration_test.cpp", "scatter_halation/integration_test_cases.cpp"]:
            text = (ROOT / "tests" / file).read_text(encoding="utf-8-sig")
            self.assertNotRegex(text, r"\bresolve_(?:diffusion_filter|scatter_halation_recipe)\s*\(")
        fixture = (ROOT / "tests/ffi/spatial_optics_fixture.cpp").read_text(encoding="utf-8-sig")
        self.assertIn("fj_test_spatial_optics_prepare(", fixture)
        self.assertNotIn("fj_legacy_spatial_optics_prepare(", fixture)
        self.assertNotRegex(fixture, r"\b(?:log2|clamp|hash_value|strength_to_scatter)\s*\(")

    def test_deleted_spectral_producers_and_white_loop(self) -> None:
        import re
        self.assertFalse((ROOT / "src/SpectralProcessing.h").exists())
        processing = "\n".join(
            path.read_text(encoding="utf-8-sig")
            for directory in (ROOT / "src", ROOT / "native")
            for path in directory.rglob("*")
            if path.suffix in {".h", ".cpp", ".cu", ".cuh"}
        )
        state = (ROOT / "src/JuicerState.cpp").read_text(encoding="utf-8")
        removed = (
            "build_tables_from_curves_non_global", "compute_S_inverse_from_tables",
            "hash_float_span_digest_sp", "hash_float_vector_digest_sp",
            "hash_float_scalar_digest_sp", "hash_float_triplet_digest_sp",
            "set_identity_3x3", "store_3x3_rowmajor", "determinant_near_zero",
            "compute_film_raw_midgray", "mallett2019_exposures_from_linear_srgb",
            "mallett_basis_ready_for_tables", "select_spectral_reconstruction_path",
            "compute_layer_exposures_from_reconstruction_path",
            "reconstruct_Ee_from_DWG_RGB_hanatos", "reconstruct_Ee_from_DWG_RGB_with_tables",
            "hanatos_linear_spectrum", "layerExposures_from_sceneSPD_with_curves",
            "rgbDWG_to_layerExposures_from_tables_with_curves", "sanitize_raw_midgray_green_or_one",
        )
        for name in removed:
            pattern = rf"\b(?:void|bool|float|std::uint64_t|SpectralReconstructionPath)\s+{name}\s*\("
            self.assertIsNone(re.search(pattern, processing), name)
            self.assertIsNotNone(re.search(pattern, processing + f"\ninline void {name}() {{}}"), name)
        self.assertNotIn("struct RowMajor3x3d", processing)
        tls = r"\bthread_local\s+std::vector<float>"
        self.assertIsNone(re.search(tls, processing))
        self.assertIsNotNone(re.search(tls, processing + "\nthread_local std::vector<float> Ee_scene;"))
        legacy = (ROOT / "rust/film-juicer-plugin/src/legacy_bridge.rs").read_text(encoding="utf-8")
        for name in ("fj_legacy_adapt_cat02", "fj_legacy_input_to_dwg", "fj_legacy_input_to_linear_srgb", "fj_legacy_dwg_to_xyz"):
            pattern = rf"\bfn\s+{name}\s*\("
            self.assertIsNone(re.search(pattern, legacy))
            self.assertIsNotNone(re.search(pattern, legacy + f"\nfn {name}() {{}}"))
        self.assertEqual(state.count("JuicerSpectral::build_tables("), 3)
        self.assertIn("JuicerSpectral::integrate_white(curve, label, out)", state)
        loop = r"double\s+sum[XYZ]\s*=\s*0\.0"
        self.assertIsNone(re.search(loop, state))
        self.assertIsNotNone(re.search(loop, state + "\ndouble sumX = 0.0;"))
        fixture = (ROOT / "tests/ffi/color_preparation_test.cpp").read_text(encoding="utf-8")
        start = fixture.index("Scanner::ScannerIlluminant scanner_illuminant(")
        end = fixture.index("std::vector<Json> helper_inputs()", start)
        helper = fixture[start:end]
        self.assertIn("fj_test_spectral_white", helper)
        self.assertIsNone(re.search(loop, helper))

    def test_completed_spectral_results_are_private_and_readonly(self) -> None:
        for fixture, expected in (
            ("spectral_default", {"E0599": ("Tables::default()", "White::default()")}),
            ("spectral_private", {"E0451": ("Tables { ..tables }", "White { ..white }")}),
            ("spectral_mutation", {"E0616": ("tables.white_xyz[0]", "white.xyz[0]")}),
            ("spectral_readonly", {"E0594": ("tables.white_xyz()[0]", "white.xyz()[0]")}),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    for status, diagnostics in self.check_features("film-juicer-plugin", release):
                        self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_safe_modules_cannot_relax_unsafe_prohibition(self) -> None:
        owners = (
            ("film-juicer-core", "lib.rs"),
            ("film-juicer-plugin", "asset_catalog.rs"),
            ("film-juicer-plugin", "asset_profile.rs"),
            ("film-juicer-plugin", "asset_spectral.rs"),
            ("film-juicer-plugin", "asset_noise.rs"),
            ("film-juicer-plugin", "asset_calibration.rs"),
        )
        for package, source in owners:
            with self.subTest(package=package, source=source):
                self.attach(f"{package}/src/{source}", "unsafe_override")
                try:
                    for release in self.profiles:
                        status, diagnostics = self.check(package, release)
                        self.assert_rejected(status, diagnostics, {"E0453": ("unsafe_code",)})
                finally:
                    self.doCleanups()

    def test_completed_exposure_results_are_private_and_readonly(self) -> None:
        for fixture, expected in (
            ("exposure_default", {"E0599": ("Sensitivity::default()", "ReferenceWhite::default()")}),
            ("exposure_private", {"E0451": ("Sensitivity { ..sensitivity }", "ReferenceWhite { ..white }")}),
            ("exposure_mutation", {"E0616": ("sensitivity.values_rgb[0][0]", "white.samples[0]")}),
            ("exposure_readonly", {"E0594": ("sensitivity.values_rgb()[0][0]", "white.samples()[0]")}),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    for status, diagnostics in self.check_features("film-juicer-plugin", release):
                        self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()

    def test_completed_exposure_results_do_not_retain_sources(self) -> None:
        self.attach("film-juicer-plugin/src/lib.rs", "exposure_borrow")
        for release in self.profiles:
            status, diagnostics = self.check("film-juicer-plugin", release)
            self.assertEqual(status, 0, diagnostics)

    def test_complete_mallett_results_and_input_borrows(self) -> None:
        for fixture, expected in (
            ("mallett_default", {"E0599": ("MallettMidgray::default()", "MidgrayNormalization::default()", "ReferenceSource::default()", "ReferenceRaw::default()")}),
            ("mallett_private", {"E0451": ("MallettMidgray { ..midgray }", "MidgrayNormalization { ..normalization }", "ReferenceSource { ..source }", "ReferenceRaw { ..raw }")}),
            ("mallett_mutation", {"E0616": ("midgray.raw_midgray_bgr[0]", "normalization.raw_green", "source.value", "raw.rgb[0]")}),
            ("mallett_readonly", {"E0594": ("midgray.raw_midgray_bgr()[0]", "midgray.midgray_dwg_rgb()[0]", "raw.rgb()[0]")}),
            ("mallett_borrow", {"E0515": ("basis_rgb: &basis", "illuminant: &illuminant", "sensitivity_rgb: &sensitivity")}),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    for status, diagnostics in self.check_features("film-juicer-plugin", release):
                        self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()
        self.attach("film-juicer-plugin/src/lib.rs", "mallett_complete")
        for release in self.profiles:
            for status, diagnostics in self.check_features("film-juicer-plugin", release):
                self.assertEqual(status, 0, diagnostics)

    def test_complete_tc_lut_privacy_and_borrowing(self) -> None:
        for fixture, expected in (
            ("tc_default", {"E0599": ("FilmTcLut::default()",)}),
            ("tc_private", {"E0451": ("FilmTcLut { ..lut }",)}),
            ("tc_readonly", {"E0594": ("lut.samples()[0]",)}),
            ("tc_borrow", {"E0515": ("lut.samples()",), "E0505": ("drop(lut)",)}),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    status, diagnostics = self.check("film-juicer-plugin", release)
                    self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()
        self.attach("film-juicer-plugin/src/lib.rs", "tc_complete")
        for release in self.profiles:
            status, diagnostics = self.check("film-juicer-plugin", release)
            self.assertEqual(status, 0, diagnostics)


    def test_complete_input_hull_privacy_and_borrows(self) -> None:
        for fixture, expected in (
            ("hull_default", {"E0599": ("Hull::default()",)}),
            ("hull_private", {"E0451": ("Hull { ..hull }",)}),
            ("hull_mutation", {"E0616": ("hull.xy", "hull.hash", "hull.center")}),
            ("hull_readonly", {"E0594": ("hull.xy()", "hull.center()")}),
            ("hull_borrow", {"E0515": ("HullInput { x: &samples",), "E0505": ("drop(hull)",)}),
        ):
            self.attach("film-juicer-plugin/src/lib.rs", fixture)
            try:
                for release in self.profiles:
                    status, diagnostics = self.check("film-juicer-plugin", release)
                    self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()
        self.attach("film-juicer-plugin/src/lib.rs", "hull_complete")
        for release in self.profiles:
            status, diagnostics = self.check("film-juicer-plugin", release)
            self.assertEqual(status, 0, diagnostics)


if __name__ == "__main__":
    unittest.main()
