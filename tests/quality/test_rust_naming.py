"""Compile naming probes against the real workspace policy, never production files."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures/rust_naming"
PACKAGES = ("film-juicer-core", "film-juicer-plugin")


class RustNamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cargo = os.environ["JUICER_CARGO"]
        cls.target = os.environ["JUICER_RUST_TARGET"]
        cls.artifacts = Path(os.environ["JUICER_TEST_ARTIFACT_DIR"]).resolve()
        cls.artifacts.mkdir(parents=True, exist_ok=True)
        cls.environment = os.environ.copy()
        for name in ("RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
            cls.environment.pop(name, None)
        # Select the copied configuration even if a developer overrides Clippy locally.
        cls.environment.pop("CLIPPY_CONF_DIR", None)

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="probe-", dir=self.artifacts)
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name)
        for name in ("Cargo.toml", "Cargo.lock", "clippy.toml", "rust-toolchain.toml"):
            shutil.copy2(ROOT / name, self.workspace / name)
        shutil.copytree(ROOT / "rust", self.workspace / "rust")

    def add_probe(self, package: str, fixture: str) -> None:
        source = self.workspace / "rust" / package / "src"
        shutil.copy2(FIXTURES / f"{fixture}.rs", source / "naming_probe.rs")
        with (source / "lib.rs").open("a", encoding="utf-8") as library:
            library.write("\npub mod naming_probe;\n")

    def clippy(self, package: str, release: bool) -> tuple[int, list[dict]]:
        profile = "release" if release else "dev"
        command = [
            self.cargo, "clippy", "--locked", "--offline", "--package", package,
            "--all-targets", "--target", self.target,
            "--target-dir", str(self.workspace / "target"), "--message-format=json",
        ]
        if release:
            command.append("--release")
        # Do not add -D warnings here: prove the checked-in manifest denies violations.
        result = subprocess.run(
            command, cwd=self.workspace, env=self.environment,
            capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
        )
        log = self.artifacts / f"{self._testMethodName}-{package}-{profile}.log"
        log.write_text(
            subprocess.list2cmdline(command) + "\n" + result.stdout + result.stderr,
            encoding="utf-8",
        )
        diagnostics = []
        for line in result.stdout.splitlines():
            message = json.loads(line)
            if message.get("reason") == "compiler-message":
                diagnostics.append(message["message"])
        return result.returncode, diagnostics

    def test_current_code_and_valid_names_pass(self) -> None:
        for package in PACKAGES:
            self.add_probe(package, "accepted")
        for package in PACKAGES:
            for release in (False, True):
                with self.subTest(package=package, release=release):
                    status, diagnostics = self.clippy(package, release)
                    self.assertEqual(status, 0, diagnostics)

    def test_each_invalid_name_is_a_blocking_diagnostic(self) -> None:
        expected = {
            "non_camel_case_types": ("bad_type", "bad_trait", "negative_direct"),
            "non_snake_case": (
                "badModule", "renderFrame", "rowBytes", "cameraEV", "filmDensity",
                "fromRGB", "testRenderFrame",
            ),
            "non_upper_case_globals": ("kFilmFormat", "currentFrame", "kChannels"),
            "clippy::upper_case_acronyms": ("RGB", "RGBToXYZ", "CMY", "PrintRGB", "CUDAContext"),
            "clippy::allow_attributes_without_reason": (
                "#[allow(non_snake_case)]", "#[expect(non_snake_case)]",
            ),
            "unfulfilled_lint_expectations": ("non_snake_case",),
        }
        for package in PACKAGES:
            with self.subTest(package=package):
                self.add_probe(package, "rejected")
                for release in (False, True):
                    with self.subTest(release=release):
                        status, diagnostics = self.clippy(package, release)
                        self.assertNotEqual(status, 0)
                        for lint, names in expected.items():
                            spans = [
                                span
                                for diagnostic in diagnostics
                                if (diagnostic.get("code") or {}).get("code") == lint
                                and diagnostic["level"] == "error"
                                for span in diagnostic["spans"]
                                if span["is_primary"]
                                and span["file_name"].endswith("naming_probe.rs")
                            ]
                            identifiers = [
                                text["text"][text["highlight_start"] - 1:text["highlight_end"] - 1]
                                for span in spans for text in span["text"]
                            ]
                            for name in names:
                                self.assertIn(
                                    name, identifiers,
                                    f"missing blocking {lint} for {name}: {diagnostics}",
                                )
                # The plug-in's dependency must be clean when testing its own lints.
                shutil.copy2(FIXTURES / "accepted.rs", self.workspace / "rust" / package / "src/naming_probe.rs")

    def test_narrow_foreign_boundary_exception_passes(self) -> None:
        self.add_probe("film-juicer-plugin", "boundary")
        for release in (False, True):
            with self.subTest(release=release):
                status, diagnostics = self.clippy("film-juicer-plugin", release)
                self.assertEqual(status, 0, diagnostics)


if __name__ == "__main__":
    unittest.main()
