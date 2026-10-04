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
            "--target", self.target, "--target-dir", str(self.workspace / "target"),
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
        for release in (False, True):
            for test_support in (False, True):
                with self.subTest(release=release, test_support=test_support):
                    status, diagnostics = self.check("film-juicer-plugin", release, test_support)
                    self.assertEqual(status, 0, diagnostics)

    def test_profile_views_cannot_outlive_their_owners(self) -> None:
        self.attach("film-juicer-plugin/src/lib.rs", "borrowed_views")
        for release in (False, True):
            status, diagnostics = self.check("film-juicer-plugin", release)
            self.assert_rejected(status, diagnostics, {
                "E0515": ("film_owner.view()", "print_owner.view()", "spectra_owner.view()", "mallett_owner.samples()", "cmf_owner.rows()", "csv_owner.rows()"),
                "E0505": ("drop(film_owner)", "drop(print_owner)", "drop(spectra_owner)", "drop(mallett_owner)", "drop(cmf_owner)", "drop(csv_owner)"),
            })

    def test_completed_profile_storage_is_private_to_its_owner(self) -> None:
        # The plugin is a real external consumer of the core, and a sibling of
        # asset_profile; inserting the probe inside the owner would bypass privacy.
        self.attach("film-juicer-plugin/src/lib.rs", "private_storage")
        for release in (False, True):
            status, diagnostics = self.check("film-juicer-plugin", release)
            self.assert_rejected(status, diagnostics, {
                "E0616": ("film.processing_defaults", "tables.interpolation_log_exposure", "film_owner.profile", "print_owner.profile", "spectra_owner.lut", "mallett_owner.basis", "cmf_owner.rows", "csv_owner.rows"),
            })

    def test_completed_profile_construction_cannot_be_bypassed(self) -> None:
        self.attach("film-juicer-plugin/src/lib.rs", "private_construction")
        for release in (False, True):
            status, diagnostics = self.check("film-juicer-plugin", release)
            self.assert_rejected(status, diagnostics, {
                "E0451": ("FilmProfile { ..film }", "PrintProfile { ..print }"),
            })

    def test_safe_modules_cannot_relax_unsafe_prohibition(self) -> None:
        owners = (
            ("film-juicer-core", "lib.rs"),
            ("film-juicer-plugin", "asset_catalog.rs"),
            ("film-juicer-plugin", "asset_profile.rs"),
            ("film-juicer-plugin", "asset_spectral.rs"),
            ("film-juicer-plugin", "asset_calibration.rs"),
        )
        for package, source in owners:
            with self.subTest(package=package, source=source):
                self.attach(f"{package}/src/{source}", "unsafe_override")
                try:
                    for release in (False, True):
                        status, diagnostics = self.check(package, release)
                        self.assert_rejected(status, diagnostics, {"E0453": ("unsafe_code",)})
                finally:
                    self.doCleanups()


if __name__ == "__main__":
    unittest.main()
