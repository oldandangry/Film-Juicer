"""Compile naming probes against the real workspace policy, never production files."""

from __future__ import annotations

import importlib.util
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
sys.path.insert(0, str(ROOT / "scripts"))
FIXTURES = Path(__file__).resolve().parent / "fixtures/rust_naming"
PACKAGES = ("film-juicer-core", "film-juicer-plugin")
SCRATCH_LIFETIME_CASES = ("success", "failure", "hold", "cleanup-error", "incomplete", "undrained")
SPEC = importlib.util.spec_from_file_location("naming_quality", ROOT / "scripts/check-quality.py")
quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)


def preset_build_for_probe(target_dir: Path, family: str) -> Path:
    build_root = (ROOT / "out/build").resolve()
    try:
        relative = target_dir.relative_to(build_root)
    except ValueError as error:
        raise ValueError("Rust naming probe cache must be under the standard preset Cargo tree") from error
    if (
        len(relative.parts) != 4
        or relative.parts[0] not in quality.PRESET_TARGETS
        or relative.parts[1:] != ("cargo", "quality-probes", family)
    ):
        raise ValueError("Rust naming probe cache must be <root>/out/build/<preset>/cargo/quality-probes/naming")
    return build_root / relative.parts[0]


class RustNamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cargo = os.environ["JUICER_CARGO"]
        cls.target = os.environ["JUICER_RUST_TARGET"]
        cls.target_dir = Path(os.environ["JUICER_RUST_PROBE_DIR"]).resolve()
        cls.preset_build = preset_build_for_probe(cls.target_dir, "naming")
        if quality.PRESET_TARGETS[cls.preset_build.name] != cls.target:
            raise ValueError("Rust naming target does not match its standard preset")
        cls.profiles = {"debug": (False,), "release": (True,), "both": (False, True)}[
            os.environ.get("JUICER_RUST_PROBE_PROFILE", "both")
        ]
        from execution import NativeRun, OutputLease

        cls.lease = OutputLease(ROOT, cls.preset_build)
        cls.lease.__enter__()
        cls.addClassCleanup(cls.release_lease)
        artifact_root = Path(os.environ["JUICER_TEST_ARTIFACT_DIR"]).resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        cls.artifacts = Path(tempfile.mkdtemp(prefix="run-", dir=artifact_root))
        cls.native_run = NativeRun(ROOT, cls.artifacts)
        cls.expected_tests = set(unittest.defaultTestLoader.getTestCaseNames(cls))
        cls.completed_tests = set()
        cls.retain_artifacts = False
        cls.invocation = 0
        cls.invocation_lock = threading.Lock()
        cls.addClassCleanup(cls.cleanup_artifacts)
        cls.environment = quality.cargo_environment()
        cls.environment.setdefault("CARGO_INCREMENTAL", "0")

    @classmethod
    def release_lease(cls) -> None:
        if cls.lease is not None:
            cls.lease.__exit__(None, None, None)

    @classmethod
    def cleanup_artifacts(cls) -> None:
        if not cls.artifacts.exists():
            return
        if (
            cls.retain_artifacts
            or cls.completed_tests != cls.expected_tests
            or os.environ.get("JUICER_TEST_RETAIN_FIXTURES")
        ):
            print("Retained Rust naming fixture evidence:", cls.artifacts, file=sys.stderr)
            return
        if cls.lease.blocked or not cls.commands_drained():
            print("Retained Rust naming fixtures without a drained preset owner:", cls.artifacts, file=sys.stderr)
            return
        try:
            cls.lease.check_admission()
            cls.report_command_costs()
            shutil.rmtree(cls.artifacts)
        except BaseException:
            print("Rust naming fixture cleanup failed; retained:", cls.artifacts, file=sys.stderr)
            raise

    @classmethod
    def commands_drained(cls) -> bool:
        return all(
            row.get("drained", False) and row.get("process_status") in {"completed", "failed-parent"}
            for row in cls.native_run.commands
        )

    @classmethod
    def report_command_costs(cls) -> None:
        records = cls.native_run.commands if cls.native_run is not None else []
        if not records:
            return
        print(
            "Rust naming NativeRun cost: "
            f"count={len(records)} supervised_seconds={sum(row.get('seconds', 0.0) for row in records):.3f} "
            f"command_seconds={sum(row.get('command_seconds') or 0.0 for row in records):.3f} "
            f"drain_seconds={sum(row.get('drain_seconds') or 0.0 for row in records):.3f} "
            f"stdout_bytes={sum(row.get('output_bytes', 0) for row in records)} "
            f"stderr_bytes={sum(row.get('stderr_bytes', 0) for row in records)}",
            flush=True,
        )

    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="probe-", dir=self.artifacts))
        self.addCleanup(self.finalize_test, self.workspace)
        for name in ("Cargo.toml", "Cargo.lock", "clippy.toml", "rust-toolchain.toml"):
            shutil.copy2(ROOT / name, self.workspace / name)
        shutil.copytree(ROOT / "rust", self.workspace / "rust")

    def finalize_test(self, workspace: Path) -> None:
        cls = type(self)
        outcome = self._outcome
        result = outcome.result if outcome else None
        failed = (
            result is None
            or getattr(self, "interrupted", False)
            or cls.lease.blocked
            or not cls.commands_drained()
            or any(
                owner is self or getattr(owner, "test_case", None) is self
                for owner, _ in [*result.failures, *result.errors]
            )
        )
        if failed:
            cls.retain_artifacts = True
        cls.completed_tests.add(self._testMethodName)
        if cls.retain_artifacts or os.environ.get("JUICER_TEST_RETAIN_FIXTURES"):
            print("Retained Rust naming probe workspace:", workspace, file=sys.stderr)
            return
        try:
            shutil.rmtree(workspace)
        except BaseException:
            cls.retain_artifacts = True
            print("Rust naming probe cleanup failed; retained:", workspace, file=sys.stderr)
            raise

    def command(self, arguments: list[str], package: str, profile: str) -> subprocess.CompletedProcess:
        cls = type(self)
        with cls.invocation_lock:
            cls.invocation += 1
            invocation = cls.invocation
        name = f"{invocation:02d}-{self._testMethodName}-{package}-{profile}".lower().replace("_", "-")
        try:
            log = cls.native_run.command(name, arguments, env=self.environment, stdout_only=True, cwd=self.workspace)
            returncode = next(record["returncode"] for record in cls.native_run.commands if record["name"] == name)
        except subprocess.CalledProcessError as error:
            log = cls.artifacts / f"{name}.log"
            returncode = error.returncode
        stdout = log.read_text(encoding="utf-8") if log.exists() else ""
        stderr_path = cls.artifacts / f"{name}.stderr.log"
        stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        return subprocess.CompletedProcess(arguments, returncode, stdout, stderr)

    def add_probe(self, package: str, fixture: str) -> None:
        source = self.workspace / "rust" / package / "src"
        shutil.copy2(FIXTURES / f"{fixture}.rs", source / "naming_probe.rs")
        with (source / "lib.rs").open("a", encoding="utf-8") as library:
            library.write("\npub mod naming_probe;\n")

    def clippy(self, package: str, release: bool) -> tuple[int, list[dict]]:
        profile = "release" if release else "dev"
        command = [
            self.cargo,
            "clippy",
            "--locked",
            "--offline",
            "--package",
            package,
            "--all-targets",
            "--target",
            self.target,
            "--target-dir",
            str(self.target_dir),
            "--message-format=json",
        ]
        if release:
            command.append("--release")
        # Do not add -D warnings here: prove the checked-in manifest denies violations.
        quality.check_probe_configuration(ROOT, self.workspace)
        result = self.command(command, package, profile)
        quality.check_probe_configuration(ROOT, self.workspace)
        diagnostics = []
        for line in result.stdout.splitlines():
            message = json.loads(line)
            if message.get("reason") == "compiler-message":
                diagnostics.append(message["message"])
        return result.returncode, diagnostics

    def test_current_code_and_valid_names_pass(self) -> None:
        for package in PACKAGES:
            self.add_probe(package, "accepted")
        cases = tuple((package, release) for package in PACKAGES for release in self.profiles)
        self.subtest_cases = tuple(f"{package}:{'release' if release else 'debug'}" for package, release in cases)
        for case_id, (package, release) in zip(self.subtest_cases, cases):
            with self.subTest(case=case_id):
                status, diagnostics = self.clippy(package, release)
                self.assertEqual(status, 0, diagnostics)

    def test_each_invalid_name_is_a_blocking_diagnostic(self) -> None:
        expected = {
            "non_camel_case_types": ("bad_type", "bad_trait", "negative_direct"),
            "non_snake_case": (
                "badModule",
                "renderFrame",
                "rowBytes",
                "cameraEV",
                "filmDensity",
                "fromRGB",
                "testRenderFrame",
            ),
            "non_upper_case_globals": ("kFilmFormat", "currentFrame", "kChannels"),
            "clippy::upper_case_acronyms": ("RGB", "RGBToXYZ", "CMY", "PrintRGB", "CUDAContext"),
            "clippy::allow_attributes_without_reason": (
                "#[allow(non_snake_case)]",
                "#[expect(non_snake_case)]",
            ),
            "unfulfilled_lint_expectations": ("non_snake_case",),
        }
        cases = tuple((package, release) for package in PACKAGES for release in self.profiles)
        self.subtest_cases = tuple(f"{package}:{'release' if release else 'debug'}" for package, release in cases)
        for package in PACKAGES:
            self.add_probe(package, "rejected")
        for case_id, (package, release) in zip(self.subtest_cases, cases):
            with self.subTest(case=case_id):
                status, diagnostics = self.clippy(package, release)
                self.assertNotEqual(status, 0)
                for lint, names in expected.items():
                    spans = [
                        span
                        for diagnostic in diagnostics
                        if (diagnostic.get("code") or {}).get("code") == lint and diagnostic["level"] == "error"
                        for span in diagnostic["spans"]
                        if span["is_primary"] and span["file_name"].endswith("naming_probe.rs")
                    ]
                    identifiers = [
                        text["text"][text["highlight_start"] - 1 : text["highlight_end"] - 1]
                        for span in spans
                        for text in span["text"]
                    ]
                    for name in names:
                        self.assertIn(
                            name,
                            identifiers,
                            f"missing blocking {lint} for {name}: {diagnostics}",
                        )
                if release == self.profiles[-1]:
                    # The plug-in's dependency must be clean when testing its own lints.
                    shutil.copy2(FIXTURES / "accepted.rs", self.workspace / "rust" / package / "src/naming_probe.rs")

    def test_narrow_foreign_boundary_exception_passes(self) -> None:
        self.add_probe("film-juicer-plugin", "boundary")
        compile_cases = tuple(f"plugin:{'release' if release else 'debug'}" for release in self.profiles)
        self.subtest_cases = tuple(f"scratch-lifetime:{case}" for case in SCRATCH_LIFETIME_CASES) + compile_cases
        exercise_scratch_lifetime_controls(self, RustNamingTests)
        for case_id, release in zip(compile_cases, self.profiles):
            with self.subTest(case=case_id):
                status, diagnostics = self.clippy("film-juicer-plugin", release)
                self.assertEqual(status, 0, diagnostics)


def exercise_scratch_lifetime_controls(test_case, owner):
    original_hold = os.environ.get("JUICER_TEST_RETAIN_FIXTURES")
    try:
        scenarios = (
            ("success", False, False, False, False, False),
            ("failure", True, False, False, False, False),
            ("hold", False, True, False, False, False),
            ("cleanup-error", False, False, True, False, False),
            ("incomplete", False, False, False, True, False),
            ("undrained", False, False, False, False, True),
        )
        for name, failed, hold, cleanup_error, incomplete, undrained in scenarios:
            with (
                test_case.subTest(case="scratch-lifetime:" + name),
                tempfile.TemporaryDirectory(prefix="naming scratch lifetime ") as parent,
            ):
                artifacts = Path(parent) / "run-fixture"

                class ControlLease:
                    blocked = undrained

                    def check_admission(self):
                        if self.blocked:
                            raise RuntimeError("controlled undrained owner")

                class ControlRun:
                    commands = []

                class ScratchLifecycleControl(owner):
                    @classmethod
                    def setUpClass(cls):
                        cls.artifacts = artifacts
                        cls.artifacts.mkdir()
                        cls.expected_tests = {"test_control"}
                        if incomplete:
                            cls.expected_tests.add("test_unrun")
                        cls.completed_tests = set()
                        cls.retain_artifacts = False
                        cls.lease = ControlLease()
                        cls.native_run = ControlRun()
                        cls.addClassCleanup(cls.cleanup_artifacts)

                    def setUp(self):
                        self.workspace = self.artifacts / "probe-fixture"
                        self.workspace.mkdir()
                        (self.workspace / "mutated-source.rs").write_text(
                            "synthetic expected-negative probe\n", encoding="utf-8"
                        )
                        self.addCleanup(self.finalize_test, self.workspace)
                        if cleanup_error:
                            self.addCleanup(self.fail_cleanup)

                    def fail_cleanup(self):
                        raise OSError("controlled cleanup failure")

                    def test_control(self):
                        child_status = 1  # Expected synthetic compiler rejection.
                        self.assertEqual(child_status, 1)
                        if failed:
                            self.fail("controlled outer assertion failure")

                case = ScratchLifecycleControl("test_control")
                result = unittest.TestResult()
                if hold:
                    os.environ["JUICER_TEST_RETAIN_FIXTURES"] = "1"
                else:
                    os.environ.pop("JUICER_TEST_RETAIN_FIXTURES", None)
                unittest.TestSuite([case]).run(result)
                retained = failed or hold or cleanup_error or incomplete or undrained
                if result.wasSuccessful() == (failed or cleanup_error):
                    raise AssertionError(f"Rust naming control {name} produced the wrong outer outcome")
                if artifacts.exists() != retained:
                    raise AssertionError(f"Rust naming scratch lifetime was wrong for {name}")
                if failed and not (artifacts / "probe-fixture" / "mutated-source.rs").is_file():
                    raise AssertionError("Rust naming outer failure lost its mutated probe workspace")
    finally:
        if original_hold is None:
            os.environ.pop("JUICER_TEST_RETAIN_FIXTURES", None)
        else:
            os.environ["JUICER_TEST_RETAIN_FIXTURES"] = original_hold


RustNamingTests.test_current_code_and_valid_names_pass.required_subtests = True
RustNamingTests.test_each_invalid_name_is_a_blocking_diagnostic.required_subtests = True
RustNamingTests.test_narrow_foreign_boundary_exception_passes.required_subtests = True


if __name__ == "__main__":
    unittest.main()
