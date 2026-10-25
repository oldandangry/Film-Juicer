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

    def test_print_preflash_is_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("print_preflash_complete", None),
            ("print_preflash_construction", {"E0451": ("print::PreflashRaw { ..raw }",)}),
            ("print_preflash_default", {"E0599": ("print::PreflashRaw::default()",)}),
            ("print_preflash_private", {"E0616": ("raw.cmy",)}),
            ("print_preflash_readonly", {"E0594": ("raw.cmy()[0] = 0.0",)}),
            ("print_preflash_owned", None),
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

    def test_film_reference_is_complete_private_readonly_and_borrowed(self) -> None:
        for fixture, expected in (
            ("film_reference_complete", None),
            ("film_reference_construction", {"E0451": ("reference::Reference { ..value }",)}),
            ("film_reference_default", {"E0599": ("reference::Reference::default()",)}),
            ("film_reference_private", {"E0616": ("value.baseline_raw_rgb", "value.hash")}),
            ("film_reference_readonly", {"E0594": ("value.baseline_density_cmy()[0] = 0.0",)}),
            ("film_reference_owned", None),
            ("film_reference_borrow", {"E0515": ("basis_rgb: &basis", "illuminant: &light", "sensitivity_rgb: &sensitivity")}),
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

    def test_print_balance_is_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("print_balance_complete", None),
            ("print_balance_construction", {"E0451": ("print::Balance { ..balance }",)}),
            ("print_balance_default", {"E0599": ("print::Balance::default()",)}),
            ("print_balance_private", {"E0616": ("balance.baseline_raw_rgb",)}),
            ("print_balance_readonly", {"E0594": ("balance.baseline_raw_rgb()[0] = 0.0",)}),
            ("print_balance_owned", None),
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

    def test_frame_geometry_is_complete_private_and_readonly(self) -> None:
        for fixture, expected in (
            ("frame_complete", None),
            ("frame_construction", {"E0451": ("Geometry { ..geometry }",)}),
            ("frame_private", {"E0616": ("geometry.domain", "geometry.pixel_size_um")}),
            ("frame_default", {"E0599": ("Geometry::default()",)}),
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

    def test_dir_frame_is_complete_private_readonly_and_failure_trace_cannot_execute(self) -> None:
        for fixture, expected in (
            ("dir_frame_complete", None),
            ("dir_frame_construction", {"E0451": ("Plan { ..plan }", "Component { ..component }", "FailureTrace { ..trace }")}),
            ("dir_frame_private", {"E0616": ("plan.progress", "component.sigma_px", "error.trace")}),
            ("dir_frame_default", {"E0599": ("Plan::default()", "Component::default()", "FailureTrace::default()", "Error::default()")}),
            ("dir_frame_readonly", {"E0594": ("coefficients()[0] = 0.0",)}),
            ("dir_frame_failure_plan", {"E0308": ("consume(error.trace().unwrap())",)}),
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

    def test_diffusion_frame_is_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("diffusion_frame_complete", None),
            ("diffusion_frame_construction", {"E0451": ("Sample { ..sample }", "Components { ..components }", "FrameSet { ..frame }")}),
            ("diffusion_frame_private", {"E0616": ("sample.fields", "components.lambda_px", "frame.hash", "frame.camera")}),
            ("diffusion_frame_readonly", {"E0594": ("blue_weights()[8] = 0.0",)}),
            ("diffusion_frame_default", {"E0599": ("Sample::default()", "Components::default()", "FrameSet::default()", "Stage::default()")}),
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

    def test_halation_frame_is_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("halation_frame_complete", None),
            ("halation_frame_construction", {"E0451": ("HalationFrame { ..frame }",)}),
            ("halation_frame_private", {"E0616": ("frame.recipe_hash", "frame.channels", "core().radius")}),
            ("halation_frame_readonly", {"E0594": ("fir_weights()[0] = 0.0",)}),
            ("halation_frame_default", {"E0599": ("HalationFrame::default()",)}),
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

    def test_gaussian_is_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("gaussian_complete", None),
            ("gaussian_construction", {"E0451": ("GaussianKernel { ..kernel }",)}),
            ("gaussian_default", {"E0599": ("GaussianKernel::default()",)}),
            ("gaussian_private", {"E0616": ("kernel.radius", "kernel.weights")}),
            ("gaussian_readonly", {"E0594": ("kernel.weights()[0] = 0.0",)}),
            ("gaussian_owned", None),
            ("gaussian_borrow", {"E0515": ("kernel.weights()",)}),
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

    def test_scanner_post_is_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("scanner_post_complete", None),
            ("scanner_post_construction", {"E0451": ("scanner::PostEffects { ..post }",)}),
            ("scanner_post_default", {"E0599": ("scanner::PostEffects::default()",)}),
            ("scanner_post_private", {"E0616": ("post.fields", "post.hash")}),
            ("scanner_post_readonly", {"E0594": ("post.fields().values()[0] = 0.0",)}),
            ("scanner_post_owned", None),
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

    def test_scanner_descriptor_is_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("scanner_descriptor_complete", None),
            ("scanner_descriptor_construction", {"E0451": ("scanner::LutDescriptor { ..descriptor }",)}),
            ("scanner_descriptor_default", {"E0599": ("scanner::LutDescriptor::default()",)}),
            ("scanner_descriptor_private", {"E0616": ("descriptor.route", "descriptor.hash")}),
            ("scanner_descriptor_readonly", {"E0308": ("let _: &mut u64 = descriptor.hash()",)}),
            ("scanner_descriptor_owned", None),
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

    def test_scanner_color_is_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("scanner_color_complete", None),
            ("scanner_color_construction", {"E0451": ("scanner::Color { ..color }",)}),
            ("scanner_color_default", {"E0599": ("scanner::Color::default()",)}),
            ("scanner_color_private", {"E0616": ("color.cat02", "color.hash")}),
            ("scanner_color_readonly", {"E0594": ("color.cat02()[0] = 0.0",)}),
            ("scanner_color_owned", None),
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

    def test_scanner_lut_is_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("scanner_lut_complete", None),
            ("scanner_lut_construction", {"E0451": ("scanner::Lut { ..lut }",)}),
            ("scanner_lut_default", {"E0599": ("scanner::Lut::default()",)}),
            ("scanner_lut_private", {"E0616": ("lut.resolution", "lut.log2_xyz")}),
            ("scanner_lut_readonly", {"E0594": ("lut.log2_xyz()[0] = 0.0",)}),
            ("scanner_lut_owned", None),
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

    def test_scanner_correction_is_complete_private_readonly_and_owned(self) -> None:
        for fixture, expected in (
            ("scanner_correction_complete", None),
            ("scanner_correction_construction", {"E0451": ("scanner::Correction { ..correction }",)}),
            ("scanner_correction_default", {"E0599": ("scanner::Correction::default()",)}),
            ("scanner_correction_private", {"E0616": ("correction.route", "correction.storage")}),
            ("scanner_correction_readonly", {"E0070": ("correction.exposure_scale() = 0.0",)}),
            ("scanner_correction_owned", None),
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
                            ("ScatterHalation.cpp", ["resolve_scatter_halation_recipe", "hash_scatter_halation_recipe", "any_positive", "build_gaussian_descriptor", "fail_descriptor", "canonical_float", "hash_value"])]:
            for name in names:
                self.assertFalse(definition(name, sources[file]), name)
                self.assertTrue(definition(name, "bool " + name + "(const Params& params) { return true; }"))
                self.assertFalse(definition(name, "bool " + name + "(const Params& params);"))
        for name in ["hash_diffusion_authored_controls", "build_diffusion_psf_sample_descriptor", "expand_diffusion_psf_components"]:
            self.assertTrue(definition(name, sources["DiffusionHostBehavior.cpp"]), name)
        for name in ["build_scatter_halation_controls", "build_scatter_halation_frame_descriptor"]:
            self.assertTrue(definition(name, sources["ScatterHalation.cpp"]), name)
        halation = sources["ScatterHalation.cpp"]
        self.assertEqual(halation.count("JuicerHalation::prepare_frame("), 1)
        for token in ("kScatterCoreUm", "kScatterTailUm", "kExponentialSigmaRatios", "kBounceSigmaMultipliers", "kMinimumActiveSigmaPixels", "kFirDispatchSigmaPixels", "std::exp", "0.98711", '"Hash.h"'):
            self.assertNotIn(token, halation)
            with self.assertRaises(AssertionError):
                self.assertNotIn(token, halation + token)
        fixtures = "\n".join((ROOT / "tests/ffi" / name).read_text() for name in ("film_recipe_test.cpp", "halation_frame_test.inc"))
        for token in ("capture-halation-frames", "nativeCapture", "Json capture(", "std::exp"):
            self.assertNotIn(token, fixtures)
        self.assertIn("SF_TEMP_BRIDGE_HALATION_FRAME", (ROOT / "src/RustHalationBridge.h").read_text())
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

    def test_deleted_dir_frame_mathematics_and_retained_native_policy(self) -> None:
        import re
        recipe = (ROOT / "src/RenderRecipe.cpp").read_text(encoding="utf-8-sig")
        header = (ROOT / "src/RenderRecipe.h").read_text(encoding="utf-8-sig")
        for name in ("finite_nonnegative", "dir_extent_valid", "dir_extents_match",
                     "dir_reference_operator_for_sigma", "build_dir_iir_coefficients", "add_dir_component"):
            pattern = r"\b" + name + r"\s*\([^;{}]*\)\s*\{"
            self.assertIsNone(re.search(pattern, recipe), name)
            self.assertIsNotNone(re.search(pattern, "bool " + name + "() { return true; }"))
            self.assertIsNone(re.search(pattern, "bool " + name + "();"))
        for token in ("DirComponentBuildInput", "coreSigmaPixels", "tailSigmaPixels", "0.98711", "2.44413", "0.422205"):
            self.assertNotIn(token, recipe)
            with self.assertRaises(AssertionError):
                self.assertNotIn(token, recipe + token)
        for token in ("kExponentialSigmaRatios", "kExponentialAmplitudes"):
            self.assertNotIn(token, recipe + header)
        for retained in ("dir_target_scratch_tier_for_operator", "dir_target_plane_roles_for_tier",
                         "strict_yvv_channels_aliased_forward_plane_roles", "configure_cached_log_raw_descriptor",
                         "resolve_spatial_dir_scratch", "hash_dir_extent", "hash_dir_plane_roles",
                         "finish_spatial_dir_descriptor", "compose_spatial_dir_descriptor"):
            self.assertIn(retained + "(", recipe)
        self.assertEqual(recipe.count("JuicerDir::prepare_math("), 1)
        resources = (ROOT / "src/Cuda/JuicerCudaResources.cpp").read_text()
        self.assertIn("build_spatial_dir_boundary_axis_cpu(", resources)
        executor = (ROOT / "src/Cuda/JuicerCudaExecutor.cpp").read_text()
        self.assertEqual(executor.count("Spektrafilm::build_spatial_dir_descriptor("), 2)
        for route, phase in (("direct", "3D-3"), ("print", "4C")):
            start = executor.index("const bool dirFrameBuilt", executor.index("RouteDescriptors build_" + route + "_descriptors"))
            end = executor.index("std::string " + route + "GrainDescriptorDiagnostic", start)
            block = executor[start:end]
            self.assertLess(block.index("build_spatial_dir_descriptor("), block.index("observe_dir_frame("))
            self.assertLess(block.index("observe_dir_frame("), block.index("if (!dirFrameBuilt)"))
            self.assertIn("ResourceDescriptorMismatch phase=" + phase + " field=spatial_dir_descriptor", block)
            schedule = executor[executor.index("static void execute_" + route + "_schedule"):]
            self.assertLess(schedule.index("check_abort(abortCallback)"), schedule.index("prepare_cuda_frame("))
        fixtures = "\n".join((ROOT / "tests/ffi" / name).read_text() for name in ("film_recipe_test.cpp", "dir_frame_test.inc"))
        for token in ("capture-dir-frames", "native_case(", "native_products(", "DirFrameTest::capture("):
            self.assertNotIn(token, fixtures)
        for name in ("dir/exposure_cache_test.cpp", "grain/scratch_reuse_test.cpp", "ffi/gaussian_kernel_test.inc"):
            fixture = (ROOT / "tests" / name).read_text()
            self.assertNotIn("Spektrafilm::build_spatial_dir_descriptor(", fixture)
            self.assertIn("JuicerTestSupport::prepare_dir_frame(", fixture)
        self.assertIn("fj_test_dir_frame_prepare(", (ROOT / "tests/ffi/spatial_optics_fixture.cpp").read_text())
        self.assertIn("SF_TEMP_BRIDGE_DIR_FRAME", (ROOT / "src/RustDirBridge.h").read_text())

    def test_deleted_diffusion_construction_and_retained_native_consumers(self) -> None:
        import re
        diffusion = (ROOT / "src/DiffusionHostBehavior.cpp").read_text(encoding="utf-8-sig")
        recipe = (ROOT / "src/RenderRecipe.cpp").read_text(encoding="utf-8-sig")
        for token in ("GroupConfig", "FamilyConfig", "ExpandedGroup", "kFamilyConfigs",
                      "kMinimumScale", "kHaloWarmthAxis", "sample_descriptor_hash(",
                      "validate_sample_descriptor(", "expand_group(", "halo_channel_weights(",
                      "finite_component_output(", "std::ceil", "std::log", "std::exp", "std::pow"):
            self.assertNotIn(token, diffusion)
            with self.assertRaises(AssertionError):
                self.assertNotIn(token, diffusion + token)
        for name in ("fail_diffusion_frame_descriptor", "hash_diffusion_frame_byte",
                     "hash_diffusion_frame_u32_le", "hash_diffusion_frame_u64_le",
                     "hash_diffusion_frame_tag", "hash_diffusion_frame_double",
                     "hash_diffusion_frame_domain", "valid_diffusion_frame_route",
                     "build_diffusion_stage_frame_descriptor"):
            pattern = r"\b" + name + r"\s*\([^;{}]*\)\s*\{"
            self.assertIsNone(re.search(pattern, recipe), name)
            self.assertIsNotNone(re.search(pattern, "bool " + name + "() { return true; }"))
            self.assertIsNone(re.search(pattern, "bool " + name + "();"))
        self.assertNotIn("kDiffusionFrameFnvPrime", recipe)
        self.assertEqual(diffusion.count("JuicerDiffusion::prepare_sample("), 1)
        self.assertEqual(diffusion.count("JuicerDiffusion::prepare_components("), 1)
        self.assertEqual(recipe.count("JuicerDiffusion::prepare_frame("), 1)
        self.assertIn("std::uint64_t hash_diffusion_authored_controls(", diffusion)
        self.assertIn("hash_diffusion_authored_controls(", (ROOT / "src/JuicerState.cpp").read_text())
        resources = (ROOT / "src/Cuda/Diffusion/JuicerCudaDiffusionResources.cpp").read_text()
        self.assertIn("Spektrafilm::expand_diffusion_psf_components(", resources)
        fixtures = "\n".join((ROOT / "tests/ffi" / name).read_text()
                             for name in ("film_recipe_test.cpp", "diffusion_frame_test.inc"))
        for token in ("capture-diffusion-frames", "nativeCapture", "Json capture(", "std::log", "std::exp", "std::pow"):
            self.assertNotIn(token, fixtures)
        for name in ("diffusion/host_reference_test.cpp", "scatter_halation/integration_test.cpp",
                     "ofx/processor_reference_test.cpp", "grain/scratch_reuse_test.cpp"):
            self.assertNotRegex((ROOT / "tests" / name).read_text(),
                                r"\b(?:build_diffusion_psf_sample_descriptor|expand_diffusion_psf_components|build_diffusion_frame_set_descriptor)\s*\(")
        fixture = (ROOT / "tests/ffi/spatial_optics_fixture.cpp").read_text()
        for operation in ("sample", "components", "frame"):
            self.assertIn("fj_test_diffusion_" + operation + "_prepare(", fixture)
        self.assertIn("SF_TEMP_BRIDGE_DIFFUSION_FRAME", (ROOT / "src/RustDiffusionBridge.h").read_text())

    def test_deleted_scanner_lut_producer_and_retained_binding(self) -> None:
        import re
        processing = "\n".join(
            path.read_text(encoding="utf-8-sig")
            for directory in (ROOT / "src", ROOT / "native")
            for path in directory.rglob("*")
            if path.suffix in {".h", ".cpp", ".cu", ".cuh"}
        )
        for name in (
            "build_canonical_scan_lut_cpu", "fill_monotone_slopes_1d",
            "scan_lut_index", "scan_lut_cell_index", "spectral_to_log_xyz",
            "normalize_density", "capture_scanner_lut", "capture_scanner_lut_slopes",
        ):
            pattern = rf"\b{name}\s*\("
            self.assertIsNone(re.search(pattern, processing), name)
            self.assertIsNotNone(re.search(pattern, processing + f"\nvoid {name}() {{}}"), name)
        for name in ("CanonicalScanLutCpu", "ScannerMediumRuntime", "JUICER_SCANNER_LUT_CAPTURE"):
            self.assertNotIn(name, processing)
        self.assertFalse((ROOT / "tests/ffi/scanner_lut_capture.h").exists())
        cmake = (ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
        for name in ("juicer_scanner_lut_capture_objects", "Scanner.Host.LutNativeCapture"):
            self.assertNotIn(name, cmake)
        fixture = (ROOT / "tests/ffi/scanner_lut_test.inc").read_text(encoding="utf-8")
        self.assertNotIn("capture_scanner_lut", fixture)
        self.assertIn("compare(project(borrowed), expected)", fixture)
        resources = (ROOT / "src/Cuda/JuicerCudaResources.cpp").read_text(encoding="utf-8")
        self.assertEqual(resources.count("JuicerScanner::Lut::acquire("), 1)
        self.assertIn("SF_TEMP_BRIDGE_SCANNER_LUT", resources)
        scanner = (ROOT / "src/Scanner.cpp").read_text(encoding="utf-8")
        self.assertIn("SpectralTablesView spectral_tables_view(", scanner)
        self.assertNotIn("ColorRuntime build_color_runtime(", scanner)
        bridge = (ROOT / "src/RustScannerBridge.cpp").read_text(encoding="utf-8")
        self.assertIn("ColorRuntime build_color_runtime(", bridge)
        self.assertIn("SF_TEMP_BRIDGE_SCANNER_COLOR", bridge)

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
