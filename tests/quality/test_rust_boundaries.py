"""Compile real consumer examples against a disposable copy of the Rust workspace."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
        cls.artifacts = Path(os.environ["JUICER_TEST_ARTIFACT_DIR"]).resolve()
        cls.artifacts.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(prefix="probe-", dir=cls.artifacts)
        cls.addClassCleanup(directory.cleanup)
        cls.workspace = Path(directory.name)
        for name in ("Cargo.toml", "Cargo.lock", "clippy.toml", "rust-toolchain.toml"):
            shutil.copy2(ROOT / name, cls.workspace / name)
        shutil.copytree(ROOT / "rust", cls.workspace / "rust")
        cls.environment = quality.cargo_environment()
        cls.environment.pop("CLIPPY_CONF_DIR", None)
        cls.invocation = 0

    def command(self, arguments: list[str], label: str) -> subprocess.CompletedProcess:
        type(self).invocation += 1
        command = [self.cargo, *arguments]
        result = subprocess.run(
            command, cwd=self.workspace, env=self.environment,
            capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
        )
        log = self.artifacts / f"{self.invocation:02d}-{self._testMethodName}-{label}.log"
        log.write_text(
            subprocess.list2cmdline(command) + "\n" + result.stdout + result.stderr,
            encoding="utf-8",
        )
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
            "--target", self.target, "--target-dir", str(self.target_dir),
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
            for test_support in (False, True):
                with self.subTest(release=release, test_support=test_support):
                    status, diagnostics = self.check("film-juicer-plugin", release, test_support)
                    self.assertEqual(status, 0, diagnostics)

    def test_profile_views_cannot_outlive_their_owners(self) -> None:
        self.attach("film-juicer-plugin/src/lib.rs", "borrowed_views")
        for release in self.profiles:
            status, diagnostics = self.check("film-juicer-plugin", release)
            self.assert_rejected(status, diagnostics, {
                "E0515": ("film_owner.view()", "print_owner.view()", "spectra_owner.view()", "mallett_owner.samples()", "cmf_owner.rows()", "csv_owner.rows()", "noise_owner.view()"),
                "E0505": ("drop(film_owner)", "drop(print_owner)", "drop(spectra_owner)", "drop(mallett_owner)", "drop(cmf_owner)", "drop(csv_owner)", "drop(noise_owner)"),
            })

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
                    for test_support in (False, True):
                        status, diagnostics = self.check("film-juicer-plugin", release, test_support)
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
                    for test_support in (False, True):
                        status, diagnostics = self.check("film-juicer-plugin", release, test_support)
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
                    for test_support in (False, True):
                        status, diagnostics = self.check("film-juicer-plugin", release, test_support)
                        self.assert_rejected(status, diagnostics, expected)
            finally:
                self.doCleanups()
        self.attach("film-juicer-plugin/src/lib.rs", "mallett_complete")
        for release in self.profiles:
            for test_support in (False, True):
                status, diagnostics = self.check("film-juicer-plugin", release, test_support)
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


if __name__ == "__main__":
    unittest.main()
