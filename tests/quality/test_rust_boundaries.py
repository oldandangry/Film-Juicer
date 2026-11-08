"""Compile real consumer examples against a disposable copy of the Rust workspace."""

from __future__ import annotations

import importlib.util
from contextlib import ExitStack
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
sys.path.insert(0, str(ROOT / "scripts"))
FIXTURES = Path(__file__).parent / "fixtures/rust_boundaries"
SCRATCH_LIFETIME_CASES = ("success", "failure", "hold", "cleanup-error", "incomplete", "undrained")
SPEC = importlib.util.spec_from_file_location("boundary_quality", ROOT / "scripts/check-quality.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the tracked quality contract")
quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)


def preset_build_for_probe(target_dir: Path, family: str) -> Path:
    build_root = (ROOT / "out/build").resolve()
    try:
        relative = target_dir.relative_to(build_root)
    except ValueError as error:
        raise ValueError("Rust boundary probe cache must be under the standard preset Cargo tree") from error
    if (len(relative.parts) != 4 or relative.parts[0] not in quality.PRESET_TARGETS
            or relative.parts[1:] != ("cargo", "quality-probes", family)):
        raise ValueError("Rust boundary probe cache must be <root>/out/build/<preset>/cargo/quality-probes/boundaries")
    return build_root / relative.parts[0]


class RustBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cargo = os.environ["JUICER_CARGO"]
        cls.target = os.environ["JUICER_RUST_TARGET"]
        cls.target_dir = Path(os.environ["JUICER_RUST_PROBE_DIR"]).resolve()
        cls.preset_build = preset_build_for_probe(cls.target_dir, "boundaries")
        if quality.PRESET_TARGETS[cls.preset_build.name] != cls.target:
            raise ValueError("Rust boundary target does not match its standard preset")
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
        cls.failure_snapshots = set()
        cls.addClassCleanup(cls.cleanup_artifacts)
        cls.workspace = Path(tempfile.mkdtemp(prefix="probe-", dir=cls.artifacts))
        for name in ("Cargo.toml", "Cargo.lock", "clippy.toml", "rust-toolchain.toml"):
            shutil.copy2(ROOT / name, cls.workspace / name)
        shutil.copytree(ROOT / "rust", cls.workspace / "rust")
        cls.environment = quality.cargo_environment()
        cls.environment.setdefault("CARGO_INCREMENTAL", "0")
        cls.invocation = 0
        cls.invocation_lock = threading.Lock()

    @classmethod
    def release_lease(cls) -> None:
        if cls.lease is not None:
            cls.lease.__exit__(None, None, None)

    @classmethod
    def cleanup_artifacts(cls) -> None:
        if not cls.artifacts.exists():
            return
        if (cls.retain_artifacts or cls.completed_tests != cls.expected_tests
                or os.environ.get("JUICER_TEST_RETAIN_FIXTURES")):
            print("Retained Rust boundary fixture evidence:", cls.artifacts, file=sys.stderr)
            return
        if cls.lease.blocked or not cls.commands_drained():
            print("Retained Rust boundary fixtures without a drained preset owner:", cls.artifacts, file=sys.stderr)
            return
        try:
            cls.lease.check_admission()
            cls.report_command_costs()
            shutil.rmtree(cls.artifacts)
        except BaseException:
            print("Rust boundary fixture cleanup failed; retained:", cls.artifacts, file=sys.stderr)
            raise

    @classmethod
    def commands_drained(cls) -> bool:
        return all(row.get("drained", False)
                   and row.get("process_status") in {"completed", "failed-parent"}
                   for row in cls.native_run.commands)

    @classmethod
    def report_command_costs(cls) -> None:
        records = cls.native_run.commands if cls.native_run is not None else []
        if not records:
            return
        print(
            "Rust boundary NativeRun cost: "
            f"count={len(records)} supervised_seconds={sum(row.get('seconds', 0.0) for row in records):.3f} "
            f"command_seconds={sum(row.get('command_seconds') or 0.0 for row in records):.3f} "
            f"drain_seconds={sum(row.get('drain_seconds') or 0.0 for row in records):.3f} "
            f"stdout_bytes={sum(row.get('output_bytes', 0) for row in records)} "
            f"stderr_bytes={sum(row.get('stderr_bytes', 0) for row in records)}",
            flush=True,
        )

    def setUp(self) -> None:
        # Run after later fixture restorations so this observes cleanup errors.
        self.addCleanup(self.finalize_test)

    def finalize_test(self) -> None:
        cls = type(self)
        outcome = self._outcome
        result = outcome.result if outcome else None
        if (result is None or getattr(self, "interrupted", False)
                or cls.lease.blocked
                or not cls.commands_drained()
                or any(owner is self or getattr(owner, "test_case", None) is self
                       for owner, _ in [*result.failures, *result.errors])):
            cls.retain_artifacts = True
        cls.completed_tests.add(self._testMethodName)

    def preserve_failed_workspace(self) -> None:
        result = self._outcome.result if self._outcome else None
        if (sys.exception() is None and (result is None or not any(
                owner is self or getattr(owner, "test_case", None) is self
                for owner, _ in [*result.failures, *result.errors]))):
            return
        cls = type(self)
        cls.retain_artifacts = True
        if cls.lease.blocked or not cls.commands_drained():
            return  # Keep the original workspace while writers remain unproved.
        if self._testMethodName in cls.failure_snapshots:
            return
        snapshot = cls.artifacts / ("failure-" + self._testMethodName)
        shutil.copytree(cls.workspace, snapshot)
        cls.failure_snapshots.add(self._testMethodName)

    def tearDown(self) -> None:
        # Save the mutated probe before unittest runs source/probe restorations.
        self.preserve_failed_workspace()

    def command(self, arguments: list[str], label: str) -> subprocess.CompletedProcess:
        with self.invocation_lock:
            type(self).invocation += 1
            invocation = self.invocation
        command = [self.cargo, *arguments]
        quality.check_probe_configuration(ROOT, self.workspace)
        cls = type(self)
        name = f"{invocation:02d}-{self._testMethodName}-{label}".lower().replace("_", "-")
        try:
            log = cls.native_run.command(name, command, env=self.environment,
                                         stdout_only=True, cwd=self.workspace)
            returncode = next(record["returncode"] for record in cls.native_run.commands
                              if record["name"] == name)
        except subprocess.CalledProcessError as error:
            log = cls.artifacts / f"{name}.log"
            returncode = error.returncode
        stdout = log.read_text(encoding="utf-8") if log.exists() else ""
        stderr_path = cls.artifacts / f"{name}.stderr.log"
        stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        result = subprocess.CompletedProcess(command, returncode, stdout, stderr)
        quality.check_probe_configuration(ROOT, self.workspace)
        return result

    def restore_probe(self, action, *args, **kwargs) -> None:
        cls = type(self)
        if cls.lease.blocked or not cls.commands_drained():
            cls.retain_artifacts = True
            return
        cls.lease.check_admission()
        action(*args, **kwargs)

    def attach(self, owner: str, fixture: str, cleanup=None) -> None:
        register_cleanup = cleanup.callback if cleanup is not None else self.addCleanup
        source = self.workspace / "rust" / owner
        original = source.read_text(encoding="utf-8")
        register_cleanup(self.restore_probe, source.write_text, original, encoding="utf-8")
        probe = source.parent / "quality_boundary_probe.rs"
        shutil.copy2(FIXTURES / f"{fixture}.rs", probe)
        register_cleanup(self.restore_probe, probe.unlink)
        # Absolute paths let the same fixture enter a crate root or a protected
        # child module. The owner retains its real lint and visibility context.
        source.write_text(
            original + '\n#[cfg(test)]\n'
            '#[allow(dead_code, reason = "compiler-only boundary examples")]\n'
            f'#[path = {json.dumps(probe.as_posix())}]\nmod quality_boundary_probe;\n',
            encoding="utf-8",
        )

    def attach_many(self, owner: str, fixtures: tuple[str, ...], cleanup=None) -> dict[str, str]:
        register_cleanup = cleanup.callback if cleanup is not None else self.addCleanup
        source = self.workspace / "rust" / owner
        original = source.read_text(encoding="utf-8")
        register_cleanup(self.restore_probe, source.write_text, original, encoding="utf-8")
        declarations = []
        identities = {}
        for fixture in fixtures:
            probe = source.parent / f"quality_boundary_probe_{fixture}.rs"
            shutil.copy2(FIXTURES / f"{fixture}.rs", probe)
            register_cleanup(self.restore_probe, probe.unlink)
            module = f"quality_boundary_probe_{fixture}"
            declarations.extend((
                "#[cfg(test)]",
                '#[allow(dead_code, reason = "compiler-only boundary examples")]',
                f"#[path = {json.dumps(probe.as_posix())}]",
                f"mod {module};",
            ))
            identities[fixture] = probe.name
        source.write_text(original + "\n" + "\n".join(declarations) + "\n", encoding="utf-8")
        return identities

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

    def assert_rejected(
        self,
        status: int,
        diagnostics: list[dict],
        expected: dict[str, tuple[str, ...]],
        fixture_name: str = "quality_boundary_probe.rs",
    ) -> None:
        self.assertNotEqual(status, 0, f"prohibited consumer compiled successfully: {fixture_name}")
        primary_errors = [
            diagnostic
            for diagnostic in diagnostics
            if diagnostic["level"] == "error"
            and any(
                span["is_primary"] and span["file_name"].endswith(fixture_name)
                for span in diagnostic["spans"]
            )
        ]
        actual_codes = {
            (diagnostic.get("code") or {}).get("code") or "uncoded"
            for diagnostic in primary_errors
        }
        missing_codes = set(expected) - actual_codes
        if missing_codes:
            self.fail(
                f"missing required primary error codes in {fixture_name}: "
                f"required={sorted(expected)}, missing={sorted(missing_codes)}, "
                f"actual={sorted(actual_codes)}"
            )
        for code, markers in expected.items():
            spans = [
                line["text"]
                for diagnostic in primary_errors
                if (diagnostic.get("code") or {}).get("code") == code
                for span in diagnostic["spans"]
                if span["is_primary"] and span["file_name"].endswith(fixture_name)
                for line in span["text"]
            ]
            for marker in markers:
                if not any(marker in line for line in spans):
                    self.fail(
                        f"missing intended {code} marker {marker!r} in {fixture_name}; "
                        f"primary error codes there: {sorted(actual_codes)}"
                    )

    def assert_accepted(self, status: int, diagnostics: list[dict], fixtures: tuple[str, ...]) -> None:
        if status == 0:
            return
        failures = []
        for diagnostic in diagnostics:
            if diagnostic["level"] != "error":
                continue
            code = (diagnostic.get("code") or {}).get("code") or "uncoded"
            for span in diagnostic["spans"]:
                if not span["is_primary"]:
                    continue
                source = Path(span["file_name"]).name
                if source in fixtures:
                    lines = [row["text"].strip()[:160] for row in span["text"][:2]]
                    failures.append(f"{source} {code}: {' | '.join(lines)}")
        summary = "; ".join(failures[:12]) or "no fixture primary span reported"
        self.fail(f"accepted fixture batch failed ({', '.join(fixtures)}): {summary}; see raw Cargo log")

    def check_fixture_cases(
        self,
        owner: str,
        package: str,
        cases: tuple[tuple[str, dict[str, tuple[str, ...]] | None], ...],
        feature_forms: tuple[bool, ...] = (False, True),
    ) -> None:
        """Check one family while keeping each batched fixture's source identity."""
        expected_cases = set(self.subtest_cases)
        for accepted in (True, False):
            candidates = tuple(fixture for fixture, expected in cases if (expected is None) == accepted)
            if accepted:
                batches = [candidates] if candidates else []
            else:
                independent = tuple(
                    fixture for fixture in candidates if "E0451" not in next(
                        expected for name, expected in cases if name == fixture
                    )
                )
                construction = tuple(
                    fixture for fixture in candidates if "E0451" in next(
                        expected for name, expected in cases if name == fixture
                    )
                )
                batches = ([independent] if independent else []) + [(fixture,) for fixture in construction]
            for selected in batches:
                print("Rust boundary fixtures: " + ", ".join(selected), flush=True)
                with ExitStack() as cleanup:
                    try:
                        identities = self.attach_many(owner, selected, cleanup)
                        for release in self.profiles:
                            results = (
                                self.check_features(package, release)
                                if feature_forms == (False, True)
                                else [self.check(package, release, feature_forms[0])]
                            )
                            for test_support, (status, diagnostics) in zip(feature_forms, results, strict=True):
                                for fixture in selected:
                                    expected = next(value for name, value in cases if name == fixture)
                                    case_id = (
                                        f"{fixture}|{'release' if release else 'debug'}|"
                                        f"{'test-support' if test_support else 'default'}"
                                    )
                                    self.assertIn(case_id, expected_cases)
                                    with self.subTest(case=case_id):
                                        if expected is None:
                                            self.assert_accepted(status, diagnostics, tuple(identities.values()))
                                        else:
                                            self.assert_rejected(status, diagnostics, expected, identities[fixture])
                    finally:
                        self.preserve_failed_workspace()

    def test_workspace_dependencies_and_build_hooks(self) -> None:
        quality.check_workspace_contract(self.workspace)
        result = self.command(
            ["metadata", "--locked", "--offline", "--format-version", "1"],
            "metadata",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        quality.check_dependency_contract(self.workspace, json.loads(result.stdout))

    def test_valid_profile_consumers_compile(self) -> None:
        compile_cases = tuple(
            f"accepted|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for release in self.profiles for support in (False, True)
        )
        self.subtest_cases = tuple(f"scratch-lifetime:{case}" for case in SCRATCH_LIFETIME_CASES) + compile_cases
        exercise_scratch_lifetime_controls(self, RustBoundaryTests)
        self.attach("film-juicer-plugin/src/lib.rs", "accepted")
        for release in self.profiles:
            for support, (status, diagnostics) in zip((False, True), self.check_features("film-juicer-plugin", release), strict=True):
                case_id = f"accepted|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
                with self.subTest(case=case_id):
                    self.assertEqual(status, 0, diagnostics)

    def test_profile_views_cannot_outlive_their_owners(self) -> None:
        self.subtest_cases = tuple(
            f"borrowed-views|{'release' if release else 'debug'}" for release in self.profiles
        )
        self.attach("film-juicer-plugin/src/lib.rs", "borrowed_views")
        for release in self.profiles:
            status, diagnostics = self.check("film-juicer-plugin", release)
            case_id = f"borrowed-views|{'release' if release else 'debug'}"
            with self.subTest(case=case_id):
                self.assert_rejected(status, diagnostics, {
                    "E0515": ("film_owner.view()", "print_owner.view()", "spectra_owner.view()", "mallett_owner.samples()", "cmf_owner.rows()", "csv_owner.rows()", "noise_owner.view()"),
                    "E0505": ("drop(film_owner)", "drop(print_owner)", "drop(spectra_owner)", "drop(mallett_owner)", "drop(cmf_owner)", "drop(csv_owner)", "drop(noise_owner)"),
                })

    def test_film_input_color_is_complete_and_immutable(self) -> None:
        cases = (('input_color_complete', None),
         ('input_color_default', {'E0599': ('InputColor::default()', 'Recipe::default()')}),
         ('input_color_private', {'E0616': ('color.rgb_to_xyz', 'recipe.color')}),
         ('input_color_mutation', {'E0594': ('color.rgb_to_xyz()[0] = 1.0',)}),
         ('input_color_construction', {'E0624': ('InputColor::from_matrices',)}),
         ('input_color_literal', {'E0451': ('InputColor { ..color }',)}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_output_color_is_complete_and_immutable(self) -> None:
        cases = (('output_color_complete', None),
         ('output_color_private', {'E0616': ('color.hash',)}),
         ('output_color_default', {'E0599': ('OutputColor::default()',)}),
         ('output_color_construction', {'E0451': ('OutputColor { ..*output_color',)}),
         ('output_color_mutation', {'E0594': ('color.rgb_to_xyz()[0] = 1.0',)}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_film_development_is_complete_private_and_borrowed(self) -> None:
        cases = (('film_complete', None),
         ('film_private', {'E0616': ('development.hash', 'development.profile')}),
         ('film_default', {'E0599': ('Development::default()',)}),
         ('film_construction', {'E0451': ('Development { ..development }',)}),
         ('film_readonly',
          {'E0594': ('development.gamma_rgb()[0] = 1.0', 'development.normalized_curves_cmy()[0][0] = 0.0')}),
         ('film_borrowed', {'E0515': ('development.log_exposure()',), 'E0505': ('drop(development)',)}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_output_gamut_completed_api_is_private_and_borrowed(self) -> None:
        cases = (('output_gamut_private', {'E0616': ('transform.hash', 'boundary.cmax[0]')}),
         ('output_gamut_default', {'E0599': ('OutputTransform::default()', 'OutputBoundary::default()')}),
         ('output_gamut_construction',
          {'E0451': ('OutputTransform { ..transform }', 'OutputBoundary { ..boundary }')}),
         ('output_gamut_borrowed',
          {'E0515': ('boundary.cmax()',), 'E0505': ('drop(boundary)', 'drop(transform)')}))
        feature_forms = (False,)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False,),
        )

    def test_print_illuminant_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('print_illuminant_complete', None),
         ('print_illuminant_construction', {'E0451': ('print::Illuminant { ..illuminant }',)}),
         ('print_illuminant_default', {'E0599': ('print::Illuminant::default()',)}),
         ('print_illuminant_private', {'E0616': ('illuminant.samples',)}),
         ('print_illuminant_readonly', {'E0594': ('illuminant.samples()[0] = 0.0',)}),
         ('print_illuminant_owned', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_print_preflash_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('print_preflash_complete', None),
         ('print_preflash_construction', {'E0451': ('print::PreflashRaw { ..raw }',)}),
         ('print_preflash_default', {'E0599': ('print::PreflashRaw::default()',)}),
         ('print_preflash_private', {'E0616': ('raw.cmy',)}),
         ('print_preflash_readonly', {'E0594': ('raw.cmy()[0] = 0.0',)}),
         ('print_preflash_owned', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_film_reference_is_complete_private_readonly_and_borrowed(self) -> None:
        cases = (('film_reference_complete', None),
         ('film_reference_construction', {'E0451': ('reference::Reference { ..value }',)}),
         ('film_reference_default', {'E0599': ('reference::Reference::default()',)}),
         ('film_reference_private', {'E0616': ('value.baseline_raw_rgb', 'value.hash')}),
         ('film_reference_readonly', {'E0594': ('value.baseline_density_cmy()[0] = 0.0',)}),
         ('film_reference_owned', None),
         ('film_reference_borrow',
          {'E0515': ('basis_rgb: &basis', 'illuminant: &light', 'sensitivity_rgb: &sensitivity')}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_print_balance_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('print_balance_complete', None),
         ('print_balance_construction', {'E0451': ('print::Balance { ..balance }',)}),
         ('print_balance_default', {'E0599': ('print::Balance::default()',)}),
         ('print_balance_private', {'E0616': ('balance.baseline_raw_rgb',)}),
         ('print_balance_readonly', {'E0594': ('balance.baseline_raw_rgb()[0] = 0.0',)}),
         ('print_balance_owned', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_frame_geometry_is_complete_private_and_readonly(self) -> None:
        cases = (('frame_complete', None),
         ('frame_construction', {'E0451': ('Geometry { ..geometry }',)}),
         ('frame_private', {'E0616': ('geometry.domain', 'geometry.pixel_size_um')}),
         ('frame_default', {'E0599': ('Geometry::default()',)}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_dir_boundary_is_complete_private_readonly_and_borrowed(self) -> None:
        cases = (('dir_boundary_complete', None),
         ('dir_boundary_construction',
          {'E0451': ('Input { ..input }', 'Boundary { ..boundary }', 'Axis { ..axis }', 'Error { ..error }')}),
         ('dir_boundary_key', {'E0423': ('Key(0)',)}),
         ('dir_boundary_private',
          {'E0616': ('input.coefficients', 'boundary.weights', 'axis.terminal', 'error.reason', 'key.0')}),
         ('dir_boundary_default',
          {'E0599': ('Input::default()',
                     'Boundary::default()',
                     'Axis::default()',
                     'Error::default()',
                     'Key::default()')}),
         ('dir_boundary_readonly', {'E0594': ('weights()[0] = 0.0', 'terminal()[0] = 0.0')}),
         ('dir_boundary_borrow', {'E0515': ('boundary.weights()',)}),
         ('dir_boundary_drop', {'E0505': ('drop(boundary)',)}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_effects_frame_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('effects_frame_complete', None),
         ('effects_frame_private', {'E0616': ('frame.hash', 'frame.origins')}),
         ('effects_frame_construction', {'E0451': ('Frame { ..frame }',)}),
         ('effects_frame_default', {'E0599': ('Frame::default()',)}),
         ('effects_frame_mutation', {'E0594': ('weave()[0] = 0.0', 'origins()[0].cell_x = 0')}),
         ('effects_frame_owned', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_grain_frame_is_complete_private_readonly_and_owns_no_curve_borrow(self) -> None:
        cases = (('grain_frame_complete', None),
         ('grain_frame_private', {'E0616': ('plan.frame0', 'gaussian.sigma_px', 'error.field')}),
         ('grain_frame_construction',
          {'E0451': ('Plan { ..plan }', 'Gaussian { ..gaussian }', 'Error { ..error }')}),
         ('grain_frame_default', {'E0599': ('Plan::default()', 'Gaussian::default()', 'Error::default()')}),
         ('grain_frame_mutation', {'E0594': ('density_max_cmy()[0] = 0.0', 'density_min_layers()[0][0] = 0.0')}),
         ('grain_frame_borrow', {'E0515': ('plan.density_max_cmy()',), 'E0505': ('drop(plan)',)}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_dir_frame_is_complete_private_readonly_and_failure_trace_cannot_execute(self) -> None:
        cases = (('dir_frame_complete', None),
         ('dir_frame_construction',
          {'E0451': ('Plan { ..plan }', 'Component { ..component }', 'FailureTrace { ..trace }')}),
         ('dir_frame_private', {'E0616': ('plan.progress', 'component.sigma_px', 'error.trace')}),
         ('dir_frame_default',
          {'E0599': ('Plan::default()', 'Component::default()', 'FailureTrace::default()', 'Error::default()')}),
         ('dir_frame_readonly', {'E0594': ('coefficients()[0] = 0.0',)}),
         ('dir_frame_failure_plan', {'E0308': ('consume(error.trace().unwrap())',)}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_diffusion_frame_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('diffusion_frame_complete', None),
         ('diffusion_frame_construction',
          {'E0451': ('Sample { ..sample }', 'Components { ..components }', 'FrameSet { ..frame }')}),
         ('diffusion_frame_private',
          {'E0616': ('sample.fields', 'components.lambda_px', 'frame.hash', 'frame.camera')}),
         ('diffusion_frame_readonly', {'E0594': ('blue_weights()[8] = 0.0',)}),
         ('diffusion_frame_default',
          {'E0599': ('Sample::default()', 'Components::default()', 'FrameSet::default()', 'Stage::default()')}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_halation_frame_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('halation_frame_complete', None),
         ('halation_frame_construction', {'E0451': ('HalationFrame { ..frame }',)}),
         ('halation_frame_private', {'E0616': ('frame.recipe_hash', 'frame.channels', 'core().radius')}),
         ('halation_frame_readonly', {'E0594': ('fir_weights()[0] = 0.0',)}),
         ('halation_frame_default', {'E0599': ('HalationFrame::default()',)}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_gaussian_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('gaussian_complete', None),
         ('gaussian_construction', {'E0451': ('GaussianKernel { ..kernel }',)}),
         ('gaussian_default', {'E0599': ('GaussianKernel::default()',)}),
         ('gaussian_private', {'E0616': ('kernel.radius', 'kernel.weights')}),
         ('gaussian_readonly', {'E0594': ('kernel.weights()[0] = 0.0',)}),
         ('gaussian_owned', None),
         ('gaussian_borrow', {'E0515': ('kernel.weights()',)}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_scanner_post_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('scanner_post_complete', None),
         ('scanner_post_construction', {'E0451': ('scanner::PostEffects { ..post }',)}),
         ('scanner_post_default', {'E0599': ('scanner::PostEffects::default()',)}),
         ('scanner_post_private', {'E0616': ('post.fields', 'post.hash')}),
         ('scanner_post_readonly', {'E0594': ('post.fields().values()[0] = 0.0',)}),
         ('scanner_post_owned', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_scanner_descriptor_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('scanner_descriptor_complete', None),
         ('scanner_descriptor_construction', {'E0451': ('scanner::LutDescriptor { ..descriptor }',)}),
         ('scanner_descriptor_default', {'E0599': ('scanner::LutDescriptor::default()',)}),
         ('scanner_descriptor_private', {'E0616': ('descriptor.route', 'descriptor.hash')}),
         ('scanner_descriptor_readonly', {'E0308': ('let _: &mut u64 = descriptor.hash()',)}),
         ('scanner_descriptor_owned', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_scanner_color_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('scanner_color_complete', None),
         ('scanner_color_construction', {'E0451': ('scanner::Color { ..color }',)}),
         ('scanner_color_default', {'E0599': ('scanner::Color::default()',)}),
         ('scanner_color_private', {'E0616': ('color.cat02', 'color.hash')}),
         ('scanner_color_readonly', {'E0594': ('color.cat02()[0] = 0.0',)}),
         ('scanner_color_owned', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_scanner_lut_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('scanner_lut_complete', None),
         ('scanner_lut_construction', {'E0451': ('scanner::Lut { ..lut }',)}),
         ('scanner_lut_default', {'E0599': ('scanner::Lut::default()',)}),
         ('scanner_lut_private', {'E0616': ('lut.resolution', 'lut.log2_xyz')}),
         ('scanner_lut_readonly', {'E0594': ('lut.log2_xyz()[0] = 0.0',)}),
         ('scanner_lut_owned', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_scanner_correction_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('scanner_correction_complete', None),
         ('scanner_correction_construction', {'E0451': ('scanner::Correction { ..correction }',)}),
         ('scanner_correction_default', {'E0599': ('scanner::Correction::default()',)}),
         ('scanner_correction_private', {'E0616': ('correction.route', 'correction.storage')}),
         ('scanner_correction_readonly', {'E0070': ('correction.exposure_scale() = 0.0',)}),
         ('scanner_correction_owned', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_optics_are_complete_private_readonly_and_owned(self) -> None:
        cases = (('optics_complete', None),
         ('optics_default',
          {'E0599': ('optics::Recipe::default()',
                     'optics::Diffusion::default()',
                     'optics::LensBlur::default()',
                     'optics::Halation::default()')}),
         ('optics_private',
          {'E0616': ('recipe.camera_diffusion',
                     'recipe.camera_lens_blur().sigma_um',
                     'recipe.scatter_halation().first_sigma_um'),
           'E0624': ('optics::Diffusion::inactive()', 'optics::LensBlur::new', 'optics::Halation::new')}),
         ('optics_construction',
          {'E0451': ('optics::Recipe { ..recipe }',
                     'optics::Diffusion { ..diffusion }',
                     'optics::LensBlur { ..lens }',
                     'optics::Halation { ..halation }')}),
         ('optics_readonly',
          {'E0594': ('*recipe.camera_diffusion() = diffusion',
                     '*recipe.camera_lens_blur() = lens',
                     '*recipe.scatter_halation() = halation')}),
         ('optics_owned', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_effects_are_complete_private_readonly_and_owned(self) -> None:
        cases = (('effects_complete', None),
         ('effects_default',
          {'E0599': ('effects::Recipe::default()', 'effects::Dust::default()', 'effects::Scratch::default()')}),
         ('effects_private',
          {'E0616': ('recipe.film_dust', 'dust.cell_width_mm', 'scratch.length_min_mm'),
           'E0624': ('effects::Recipe::zero()', 'effects::Dust::new', 'effects::Scratch::new')}),
         ('effects_construction',
          {'E0451': ('effects::Recipe { ..recipe }',
                     'effects::Dust { ..dust }',
                     'effects::Scratch { ..scratch }')}),
         ('effects_readonly', {'E0594': ('*recipe.film_dust() = dust', '*recipe.gate_scratch() = scratch')}),
         ('effects_owned', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_film_grain_is_complete_private_readonly_and_owned(self) -> None:
        cases = (('film_grain_complete', None),
         ('film_grain_default', {'E0599': ('grain::Recipe::default()',)}),
         ('film_grain_private',
          {'E0616': ('recipe.particle_density_min_cmy', 'recipe.axis_block_prefix_max_cmy')}),
         ('film_grain_construction', {'E0451': ('grain::Recipe { ..recipe }',)}),
         ('film_grain_readonly',
          {'E0594': ('recipe.particle_scale_cmy()[0] = 0.0', 'recipe.axis_block_prefix_max_cmy()[0][0] = 0.0')}),
         ('film_grain_owned', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_film_dir_density_is_complete_private_and_readonly(self) -> None:
        cases = (('film_dir_complete', None),
         ('film_dir_default', {'E0599': ('Recipe::default()', 'Minimum::default()', 'Bounds::default()')}),
         ('film_dir_private', {'E0616': ('recipe.compensated_axes_rgb', 'minimum.cmy', 'bounds.hash')}),
         ('film_dir_construction',
          {'E0451': ('Recipe { ..recipe }', 'Minimum { ..minimum }', 'Bounds { ..bounds }')}),
         ('film_dir_readonly',
          {'E0594': ('recipe.matrix_rgb()[0][0] = 0.0',
                     'minimum.cmy()[0] = 0.0',
                     'bounds.minimum_cmy()[0] = 0.0')}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_completed_profile_storage_is_private_to_its_owner(self) -> None:
        # The plugin is a real external consumer of the core, and a sibling of
        # asset_profile; inserting the probe inside the owner would bypass privacy.
        self.subtest_cases = tuple(
            f"private-storage|{'release' if release else 'debug'}" for release in self.profiles
        )
        self.attach("film-juicer-plugin/src/lib.rs", "private_storage")
        for release in self.profiles:
            status, diagnostics = self.check("film-juicer-plugin", release)
            case_id = f"private-storage|{'release' if release else 'debug'}"
            with self.subTest(case=case_id):
                self.assert_rejected(status, diagnostics, {
                    "E0616": ("film.processing_defaults", "tables.interpolation_log_exposure", "film_owner.profile", "print_owner.profile", "spectra_owner.lut", "mallett_owner.basis", "cmf_owner.rows", "csv_owner.rows", "noise_owner.bundle"),
                })

    def test_noise_borrows_and_storage_under_test_support(self) -> None:
        cases = (('noise_borrowed', {'E0515': ('noise_owner.view()',), 'E0505': ('drop(noise_owner)',)}),
         ('noise_private', {'E0616': ('noise_owner.bundle',)}))
        feature_forms = (True,)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(True,),
        )

    def test_completed_profile_construction_cannot_be_bypassed(self) -> None:
        self.subtest_cases = tuple(
            f"private-construction|{'release' if release else 'debug'}" for release in self.profiles
        )
        self.attach("film-juicer-plugin/src/lib.rs", "private_construction")
        for release in self.profiles:
            status, diagnostics = self.check("film-juicer-plugin", release)
            case_id = f"private-construction|{'release' if release else 'debug'}"
            with self.subTest(case=case_id):
                self.assert_rejected(status, diagnostics, {
                    "E0451": ("FilmProfile { ..film }", "PrintProfile { ..print }"),
                })

    def test_lens_construction_and_storage_are_private(self) -> None:
        # Type-check errors can suppress later field-privacy diagnostics, so
        # each rejected construction/access is its own compiler consumer.
        cases = (('illuminant_private', {'E0451': ('LensInput {',)}),
         ('illuminant_default', {'E0599': ('LensInput::default()',)}),
         ('illuminant_mutation', {'E0616': ('input.kg3[0]',)}))
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}"
            for fixture, _ in cases for release in self.profiles
        )
        for fixture, expected in cases:
            with ExitStack() as cleanup:
                try:
                    self.attach("film-juicer-plugin/src/lib.rs", fixture, cleanup)
                    for release in self.profiles:
                        status, diagnostics = self.check("film-juicer-plugin", release)
                        case_id = f"{fixture}|{'release' if release else 'debug'}"
                        with self.subTest(case=case_id):
                            self.assert_rejected(status, diagnostics, expected)
                finally:
                    self.preserve_failed_workspace()

    def test_deleted_film_development_definitions_and_foundation_policy(self) -> None:
        import re
        recipe = (ROOT / "src/RenderValues.cpp").read_text(encoding="utf-8-sig")
        def definition(name: str, source: str) -> bool:
            # Declarations and names in diagnostic strings are not definitions.
            return re.search(r"\b" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{", source) is not None
        for name in ("hash_film_develop_recipe", "hash_density_curves_layers", "visual_grain_requires_density_layers", "normalize_density_curves"):
            self.assertFalse(definition(name, recipe), name)
            self.assertTrue(definition(name, "std::uint64_t " + name + "(const FilmDevelopRecipe& recipe) { return 1; }"))
            self.assertFalse(definition(name, "std::uint64_t " + name + "(const FilmDevelopRecipe& recipe);"))
        for name in ("normalize_density_curves", "hash_nan_preserving_floats", "resolve_scan_route"):
            call = r"\b" + name + r"\s*\("
            self.assertNotRegex(recipe, call)
            self.assertRegex("auto value = " + name + "(source);", call)
        header = (ROOT / "src/RenderValues.h").read_text(encoding="utf-8-sig")
        retired = (r"\b(?:struct|class)\s+(?:RenderRecipe|ProfileRoute|ScannerOutputRecipe|PrintRecipe|"
                   r"PrintFilterRecipe|PrintExposureRecipe|PrintIlluminantRecipe|PrintDevelopRecipe|FilmFoundationBuildInput|FilmDevelopRecipe|FilmDevelopment|DevelopmentHold)\b|"
                   r"\busing\s+(?:::)?(?:RenderRecipe|ProfileRoute|ScannerOutputRecipe|PrintRecipe|"
                   r"PrintFilterRecipe|PrintExposureRecipe|PrintIlluminantRecipe|PrintDevelopRecipe)\b|"
                   r"\b(?:Direct|Print)RecipeBuild(?:Input|Result)\b|"
                   r"\b(?:make_render_recipe|build_film_foundation|build_direct_render_recipe|"
                   r"build_print_render_recipe|finish_synthetic_reference_recipes)\s*\(")
        self.assertNotRegex(recipe + header, retired)
        for text in ("struct RenderRecipe {};", "struct ProfileRoute {};", "using ::RenderRecipe;", "using ::ProfileRoute;",
                     "struct ScannerOutputRecipe {};", "struct PrintRecipe {};", "struct PrintFilterRecipe {};",
                     "struct PrintExposureRecipe {};", "struct PrintIlluminantRecipe {};", "struct PrintDevelopRecipe {};",
                     "using ::ScannerOutputRecipe;", "using ::PrintRecipe;", "using ::PrintFilterRecipe;",
                     "using ::PrintExposureRecipe;", "using ::PrintIlluminantRecipe;", "using ::PrintDevelopRecipe;",
                     "DirectRecipeBuildInput input;", "DirectRecipeBuildResult result;",
                     "PrintRecipeBuildInput input;", "PrintRecipeBuildResult result;", "make_render_recipe()",
                     "build_film_foundation()", "build_direct_render_recipe()", "build_print_render_recipe()",
                     "finish_synthetic_reference_recipes()"):
            self.assertRegex(text, retired)
        for text in ("struct PrintBalanceRecipe {};", "struct DichroicFilterRecipe {};", "struct CmyCcTriplet {};"):
            self.assertNotRegex(text, retired)
        for text in ("struct FilmFoundationBuildInput {};", "struct FilmDevelopRecipe {};",
                     "class DevelopmentHold {};", "struct FilmDevelopment {};"):
            self.assertRegex(text, retired)
        self.assertFalse((ROOT / "src/ProfileAssets.h").exists())
        self.assertFalse((ROOT / "src/RustFilmInputBridge.cpp").exists())
        bridge = (ROOT / "tests/ffi/test_asset_bridge.cpp").read_text(encoding="utf-8")
        for text in ("ConversionSlots", "FilmSource::", "PrintProfileSource::", "copy_tables("):
            self.assertNotIn(text, bridge)
        self.assertIn("AssetBridge::film_defaults(", bridge)
        self.assertIn("fj_legacy_film_profile_acquire(", bridge)
        self.assertIn("fj_legacy_film_profile_release(owner", bridge)
        complete = (ROOT / "rust/film-juicer-core/src/recipe.rs").read_text(encoding="utf-8")
        self.assertIn("let development = film::Development::new(", complete)
        self.assertIn("layers_required: params.grain.active && params.grain.sublayers_active", complete)
        self.assertIn("pub fn prepare(assets: &Assets, params: &Params) -> Result<RenderState, PrepareError>", complete)
        self.assertIn("grain::Recipe::new(&acquired.development, params.grain)", complete)
        preparation = (ROOT / "rust/film-juicer-plugin/src/preparation.rs").read_text(encoding="utf-8")
        self.assertIn("let completed = recipe::prepare(assets, params)", preparation)
        self.assertIn("FjPreparedState::from_state(Arc::new(completed))", preparation)
        owner = complete.split("pub struct RenderState {", 1)[1].split("\n}", 1)[0]
        self.assertIn("contents: Vec<Contents>", owner)
        contents = complete.split("struct Contents {", 1)[1].split("\n}", 1)[0]
        self.assertIn("development: film::Development", contents)
        self.assertIn("backing.push(contents)", complete)
        public_fields = r"\bpub(?:\([^)]*\))?\s+\w+\s*:"
        self.assertNotRegex(owner, public_fields)
        self.assertNotRegex(contents, public_fields)
        self.assertRegex("pub development: film::Development", public_fields)
        self.assertRegex("pub(crate) development: film::Development", public_fields)

    def test_deleted_dir_density_producers_and_stage_order(self) -> None:
        import re
        recipe = (ROOT / "src/RenderValues.cpp").read_text(encoding="utf-8-sig")
        def definition(name: str, source: str) -> bool:
            return re.search(r"\b" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{", source) is not None
        for name in ("build_grain_contract", "build_dir_couplers_recipe", "evaluate_dir_langmuir",
                     "hash_dir_couplers_recipe", "build_film_density_bounds", "hash_nan_preserving_floats",
                     "build_visual_grain_recipe", "hash_visual_grain_recipe",
                     "derive_grain_layer_axis_search_metadata", "finite_nonnegative_grain_triplet",
                     "finite_positive_triplet", "finite_nonnegative_pair",
                     "normalize_effect_amount", "hash_defect_policy",
                     "hash_film_juicer_effects_recipe", "build_film_juicer_effects_recipe",
                     "hash_density_bounds_recipe", "build_print_density_bounds"):
            self.assertFalse(definition(name, recipe), name)
            self.assertTrue(definition(name, "bool " + name + "(const Recipe& recipe) { return true; }"))
            self.assertFalse(definition(name, "bool " + name + "(const Recipe& recipe);"))
        self.assertNotIn("build_film_foundation(", recipe)
        self.assertNotIn("JuicerAssets::prepare_physical_effects(", recipe)
        complete = (ROOT / "rust/film-juicer-core/src/recipe.rs").read_text(encoding="utf-8")
        complete = complete.split("pub fn prepare(", 1)[1].split("\n}\n", 1)[0]
        stages = [complete.index(name) for name in ("optics::Recipe::new(",
                  "density::Minimum::new(", "grain::Recipe::new(",
                  "effects::Recipe::new(", "dir::Recipe::new(", "density::Bounds::new(")]
        self.assertEqual(stages, sorted(stages))
        self.assertLess(complete.index("let acquired = acquire_foundation("), complete.index("grain::Recipe::new(&acquired.development"))
        self.assertIn("dir::Recipe::new(&acquired.development, params.dir_couplers)", complete)
        bounds = complete[stages[-1]:complete.index("let profile_route_hash", stages[-1])]
        self.assertIn("&acquired.development, &minimum", bounds)
        self.assertNotIn("&grain", bounds)

    def test_deleted_spatial_optics_producers_and_retained_consumers(self) -> None:
        import re
        sources = {name: (ROOT / ("src/" + name)).read_text(encoding="utf-8-sig")
                   for name in ["RenderValues.cpp", "DiffusionHostBehavior.cpp", "ScatterHalation.cpp"]}
        def definition(name: str, text: str) -> bool:
            return re.search(r"\b" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{", text) is not None
        for file, names in [("RenderValues.cpp", ["build_spatial_optics_recipe", "hash_diffusion_filter_optics", "hash_camera_lens_blur_optics"]),
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
        recipe = sources["RenderValues.cpp"]
        self.assertNotIn("JuicerAssets::prepare_spatial_optics(", recipe)
        self.assertNotIn("build_film_foundation(", recipe)
        header = (ROOT / "src/RenderValues.h").read_text(encoding="utf-8-sig")
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
        complete = (ROOT / "rust/film-juicer-core/src/recipe.rs").read_text(encoding="utf-8")
        complete = complete.split("pub fn prepare(", 1)[1].split("\n}\n", 1)[0]
        self.assertLess(complete.index("input::Recipe::new("), complete.index("optics::Recipe::new("))
        self.assertLess(complete.index("optics::Recipe::new("), complete.index("density::Minimum::new("))
        for file in ["diffusion/host_reference_test.cpp", "scatter_halation/recipe_descriptor_test.cpp", "scatter_halation/integration_test.cpp", "scatter_halation/integration_test_cases.cpp"]:
            text = (ROOT / "tests" / file).read_text(encoding="utf-8-sig")
            self.assertNotRegex(text, r"\bresolve_(?:diffusion_filter|scatter_halation_recipe)\s*\(")
        fixture = (ROOT / "tests/ffi/spatial_optics_fixture.cpp").read_text(encoding="utf-8-sig")
        self.assertIn("fj_test_spatial_optics_prepare(", fixture)
        self.assertNotIn("fj_legacy_spatial_optics_prepare(", fixture)
        self.assertNotRegex(fixture, r"\b(?:log2|clamp|hash_value|strength_to_scatter)\s*\(")

    def test_deleted_dir_frame_mathematics_and_retained_native_policy(self) -> None:
        import re
        recipe = (ROOT / "src/RenderValues.cpp").read_text(encoding="utf-8-sig")
        header = (ROOT / "src/RenderValues.h").read_text(encoding="utf-8-sig")
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
        self.assertNotIn("JuicerDir::prepare_math(", recipe)
        resources = (ROOT / "src/Cuda/JuicerCudaResources.cpp").read_text()
        self.assertNotIn("build_spatial_dir_boundary_axis_cpu(", resources)
        self.assertIn("bool ensure_spatial_dir_boundary(", resources)
        self.assertIn("JuicerDirBoundary::Boundary::acquire(", resources)
        executor = (ROOT / "src/Cuda/JuicerCudaExecutor.cpp").read_text()
        retired = r"\b(?:build_direct_descriptors|build_print_descriptors|Spektrafilm::build_spatial_dir_descriptor)\s*\("
        self.assertNotRegex(executor, retired)
        for text in ("build_direct_descriptors()", "build_print_descriptors()", "Spektrafilm::build_spatial_dir_descriptor()"):
            self.assertRegex(text, retired)
        scope = (ROOT / "native/juicer_cuda_call_scope.cpp").read_text(encoding="utf-8")
        block = scope.split("const bool dirComposed =", 1)[1].split("Spektrafilm::VisualGrainFrameDescriptor grain", 1)[0]
        stages = [block.index(token) for token in ("compose_spatial_dir_descriptor(", "observe_dir_frame(", "if (!dirComposed)")]
        self.assertEqual(stages, sorted(stages))
        self.assertIn("ResourceDescriptorMismatch phase=dir_descriptor field=rust_result", block)
        frame = (ROOT / "rust/film-juicer-plugin/src/cuda/frame.rs").read_text(encoding="utf-8")
        self.assertIn("let source = state.dir();", frame)
        self.assertIn("crate::dir_frame_bridge::checked_plan(input, fault)", frame)
        for route in ("direct", "print"):
            schedule = executor[executor.index("static void execute_" + route + "_schedule"):]
            self.assertLess(schedule.index("check_abort(abortCallback)"), schedule.index("prepare_cuda_frame("))
        fixtures = "\n".join((ROOT / "tests/ffi" / name).read_text() for name in ("film_recipe_test.cpp", "dir_frame_test.inc"))
        for token in ("capture-dir-frames", "native_case(", "native_products(", "DirFrameTest::capture("):
            self.assertNotIn(token, fixtures)
        for name in ("dir/exposure_cache_test.cpp", "grain/scratch_reuse_test.cpp"):
            fixture = (ROOT / "tests" / name).read_text()
            self.assertNotIn("Spektrafilm::build_spatial_dir_descriptor(", fixture)
            self.assertIn("FrameFixture::with_prepared_frame_inputs(", fixture)
            self.assertIn("prepared.descriptors.spatialDir", fixture)
        prepared_frame_fixture = (ROOT / "tests/ffi/prepared_frame_fixture.cpp").read_text()
        self.assertIn("fj_test_dir_frame_prepare(", prepared_frame_fixture)
        self.assertIn("Spektrafilm::compose_spatial_dir_descriptor(", prepared_frame_fixture)

        gaussian_fixture = (ROOT / "tests/ffi/gaussian_kernel_test.inc").read_text()
        self.assertNotIn("Spektrafilm::build_spatial_dir_descriptor(", gaussian_fixture)
        self.assertIn("fj_test_dir_frame_prepare(", gaussian_fixture)
        self.assertIn("DirFrameTest::raw_input(", gaussian_fixture)
        self.assertIn("FJ_DIR_OPERATOR_FIR_REFLECT", gaussian_fixture)
        self.assertIn("component.radius", gaussian_fixture)
        self.assertIn("component.sigma_px", gaussian_fixture)
        self.assertIn("require(rows.size() == 188", gaussian_fixture)
        self.assertIn("fj_test_dir_frame_prepare(", (ROOT / "tests/ffi/spatial_optics_fixture.cpp").read_text())
        bridge = (ROOT / "src/RustDirBridge.h").read_text()
        self.assertNotIn("prepare_math(", bridge)
        self.assertIn("static MathOutcome admit(", bridge)
        self.assertIn("FjDirFrameInput", bridge)
        self.assertIn("MathOutcome::admit(", (ROOT / "src/RustDirBridge.cpp").read_text())

    def test_deleted_diffusion_construction_and_retained_native_consumers(self) -> None:
        import re
        diffusion = (ROOT / "src/DiffusionHostBehavior.cpp").read_text(encoding="utf-8-sig")
        recipe = (ROOT / "src/RenderValues.cpp").read_text(encoding="utf-8-sig")
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
        self.assertNotIn("JuicerDiffusion::prepare_frame(", recipe)
        bridge = (ROOT / "src/RustDiffusionBridge.cpp").read_text()
        self.assertIn("fj_legacy_diffusion_frame_prepare(&input, &wire, &rejected, &error)", bridge)
        self.assertIn("JuicerDiffusion::prepare_frame(", (ROOT / "tests/ffi/diffusion_frame_test.inc").read_text())
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
        retired = r"\b(?:JuicerScanner::Lut|SF_TEMP_BRIDGE_SCANNER_LUT|fj_legacy_scanner_lut_(?:acquire|view|release))\b"
        self.assertNotRegex(processing, retired)
        for token in ("JuicerScanner::Lut", "SF_TEMP_BRIDGE_SCANNER_LUT", "fj_legacy_scanner_lut_acquire",
                      "fj_legacy_scanner_lut_view", "fj_legacy_scanner_lut_release"):
            self.assertRegex(token, retired)
        self.assertNotRegex("SF_TEMP_BRIDGE_SCANNER_COLOR ScannerSpectralLutDescriptor", retired)
        resources = (ROOT / "src/Cuda/JuicerCudaResources.cpp").read_text(encoding="utf-8")
        body = resources.split("        if (coreAlreadyPrepared)", 1)[1]
        stages = [body.index(token) for token in (
            "resources.routeDensityBoundsHash = 0;", "resources.routeScannerDescriptorHash = 0;",
            "if (!routeLutAlreadyPrepared)", "const FjHostPreparation emptyPreparation{};",
            "const auto& preparation = hostPreparation ? *hostPreparation : emptyPreparation;",
            "prepare_scanner_lut(preparation,", "struct NextCanonicalScanLut", "alloc_and_upload_bytes(")]
        self.assertEqual(stages, sorted(stages))
        self.assertIn("return prepare_output_gamut_table_locked(", body[:stages[0]])
        dispatch = (ROOT / "native/juicer_host_preparation.cpp").read_text(encoding="utf-8")
        lut = dispatch.split("    bool prepare_scanner_lut(", 1)[1].split("    bool prepare_gaussian_kernel(", 1)[0]
        self.assertIn("!preparation.user || !preparation.scanner_lut || !preparation.gaussian", lut)
        self.assertIn("preparation.scanner_lut(preparation.user, &input, &view, &failure, &error)", lut)
        self.assertIn('{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "Scanner LUT preparation: invalid transport contract"', lut)
        callback = (ROOT / "rust/film-juicer-plugin/src/cuda/host_preparation.rs").read_text(encoding="utf-8")
        for token in ("unsafe extern \"C\" fn scanner_lut(", "scanner_bridge::prepare_host_lut(", "&mut backing.scanner_luts"):
            self.assertIn(token, callback)
        owner = (ROOT / "rust/film-juicer-plugin/src/scanner_bridge.rs").read_text(encoding="utf-8")
        facades = (ROOT / "rust/film-juicer-plugin/src/test_support.rs").read_text(encoding="utf-8")
        c11 = (ROOT / "tests/ffi/film_recipe_abi_c.c").read_text(encoding="utf-8")
        values = fixture.split("    void values(", 1)[1].split("    void callback_failure_diagnostics(", 1)[0]
        for operation in ("acquire", "view", "release"):
            name = f"fj_test_scanner_lut_c_{operation}"
            self.assertRegex(c11, rf"\bFjStatus\s+{name}\s*\([^;{{}}]*\)\s*\{{")
            consumer = c11.split(f"FjStatus {name}(", 1)[1].split("\n}", 1)[0]
            self.assertIn(f"return fj_test_scanner_lut_{operation}(", consumer)
            self.assertIn(f"{name}(", values)
            self.assertIn(f"pub(crate) unsafe fn lut_{operation}(", owner)
            self.assertIn(f"fn fj_test_scanner_lut_{operation}(", facades)
            self.assertIn(f"scanner_bridge::lut_{operation}(", facades)
        self.assertIn("with_host_preparation(", fixture)
        self.assertIn("prepare_scanner_lut(", fixture)
        cases = (ROOT / "tests/resource/scanner_lut_test.cpp").read_text(encoding="utf-8")
        for token in (
            "require(freshRejected,", "error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE",
            "error.failure.status.api == FJ_API_NONE", 'error.failure.diagnostic == "Scanner LUT preparation: invalid transport contract"',
            "warm scanner hit succeeds without host preparation", "warmed.resources->retireQueue.empty()",
            "warmed.resources->pendingFrameUseEvents.empty()", "const auto beforeLedger = warmed.ledger->snapshot();",
            "const auto beforeAllocations = allocation_bytes(*warmed.resources);",
            "allocation_bytes(*warmed.resources) == beforeAllocations", "afterFailure.values == old.values",
            "afterFailure.hash == old.hash", "afterFailure.resolution == old.resolution",
            "require_route_metadata_cleared(*warmed.resources)", "valid scanner preparation recovers the same changed key",
        ):
            self.assertIn(token, cases)
        self.assertFalse((ROOT / "src/Scanner.cpp").exists())
        scanner = (ROOT / "src/Scanner.h").read_text(encoding="utf-8")
        self.assertIn("struct SpectralTablesView", scanner)
        definition = r"\bColorRuntime\s+build_color_runtime\s*\([^;{}]*\)\s*\{"
        self.assertNotRegex(scanner, definition)
        self.assertRegex("ColorRuntime build_color_runtime() {}", definition)
        self.assertNotRegex("ColorRuntime build_color_runtime();", definition)
        adapter = r"\bspectral_tables_view\s*\("
        self.assertNotRegex(processing, adapter)
        self.assertRegex("SpectralTablesView spectral_tables_view(const SpectralTables& tables)", adapter)
        bridge = (ROOT / "src/RustScannerBridge.cpp").read_text(encoding="utf-8")
        self.assertIn("ColorRuntime build_color_runtime(", bridge)
        self.assertIn("SF_TEMP_BRIDGE_SCANNER_COLOR", bridge)
        self.assertNotRegex((ROOT / "src/RustScannerBridge.h").read_text(encoding="utf-8"), r"\bclass\s+Lut\b")
        self.assertIn("add_library(juicer_scanner_test_objects OBJECT ../src/RustScannerBridge.cpp)", cmake)
        self.assertIn("target_compile_definitions(juicer_scanner_test_objects PRIVATE JUICER_SCANNER_TEST_HOOK=1)", cmake)

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
        self.assertNotIn("JuicerSpectral::build_tables(", state)
        self.assertNotIn("JuicerSpectral::integrate_white(", state)
        complete = (ROOT / "rust/film-juicer-core/src/recipe.rs").read_text(encoding="utf-8")
        self.assertIn("spectral::checked_build_tables(spectral::Input {", complete)
        self.assertIn("let exposure_tables = build_tables(", complete)
        self.assertIn("let scanner_tables = build_tables(", complete)
        self.assertIn("spectral::checked_integrate_white(", complete)
        self.assertIn("spectral::checked_s_inverse(", complete)
        preparation = (ROOT / "rust/film-juicer-plugin/src/preparation.rs").read_text(encoding="utf-8")
        self.assertIn("let completed = recipe::prepare(assets, params)", preparation)
        self.assertIn("out.exposure_tables =", preparation)
        self.assertIn("out.scanner_tables =", preparation)
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
        cases = (('spectral_default', {'E0599': ('Tables::default()', 'White::default()')}),
         ('spectral_private', {'E0451': ('Tables { ..tables }', 'White { ..white }')}),
         ('spectral_mutation', {'E0616': ('tables.white_xyz[0]', 'white.xyz[0]')}),
         ('spectral_readonly', {'E0594': ('tables.white_xyz()[0]', 'white.xyz()[0]')}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_safe_modules_cannot_relax_unsafe_prohibition(self) -> None:
        owners = (
            ("film-juicer-core", "lib.rs"),
            ("film-juicer-plugin", "asset_catalog.rs"),
            ("film-juicer-plugin", "asset_profile.rs"),
            ("film-juicer-plugin", "asset_spectral.rs"),
            ("film-juicer-plugin", "asset_noise.rs"),
            ("film-juicer-plugin", "asset_calibration.rs"),
            ("film-juicer-plugin", "instance.rs"),
            ("film-juicer-plugin", "submission.rs"),
        )
        self.subtest_cases = tuple(
            f"{package}/{source}|{'release' if release else 'debug'}"
            for package, source in owners for release in self.profiles
        )
        for package, source in owners:
            for release in self.profiles:
                case_id = f"{package}/{source}|{'release' if release else 'debug'}"
                with ExitStack() as cleanup:
                    try:
                        self.attach(f"{package}/src/{source}", "unsafe_override", cleanup)
                        with self.subTest(case=case_id):
                            status, diagnostics = self.check(package, release)
                            self.assert_rejected(status, diagnostics, {"E0453": ("unsafe_code",)})
                    finally:
                        self.preserve_failed_workspace()

    def test_completed_exposure_results_are_private_and_readonly(self) -> None:
        cases = (('exposure_default', {'E0599': ('Sensitivity::default()', 'ReferenceWhite::default()')}),
         ('exposure_private', {'E0451': ('Sensitivity { ..sensitivity }', 'ReferenceWhite { ..white }')}),
         ('exposure_mutation', {'E0616': ('sensitivity.values_rgb[0][0]', 'white.samples[0]')}),
         ('exposure_readonly', {'E0594': ('sensitivity.values_rgb()[0][0]', 'white.samples()[0]')}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_completed_exposure_results_do_not_retain_sources(self) -> None:
        self.subtest_cases = tuple(
            f"exposure-borrow|{'release' if release else 'debug'}" for release in self.profiles
        )
        self.attach("film-juicer-plugin/src/lib.rs", "exposure_borrow")
        for release in self.profiles:
            status, diagnostics = self.check("film-juicer-plugin", release)
            case_id = f"exposure-borrow|{'release' if release else 'debug'}"
            with self.subTest(case=case_id):
                self.assertEqual(status, 0, diagnostics)

    def test_complete_mallett_results_and_input_borrows(self) -> None:
        cases = (('mallett_default',
          {'E0599': ('MallettMidgray::default()',
                     'MidgrayNormalization::default()',
                     'ReferenceSource::default()',
                     'ReferenceRaw::default()')}),
         ('mallett_private',
          {'E0451': ('MallettMidgray { ..midgray }',
                     'MidgrayNormalization { ..normalization }',
                     'ReferenceSource { ..source }',
                     'ReferenceRaw { ..raw }')}),
         ('mallett_mutation',
          {'E0616': ('midgray.raw_midgray_bgr[0]', 'normalization.raw_green', 'source.value', 'raw.rgb[0]')}),
         ('mallett_readonly',
          {'E0594': ('midgray.raw_midgray_bgr()[0]', 'midgray.midgray_dwg_rgb()[0]', 'raw.rgb()[0]')}),
         ('mallett_borrow',
          {'E0515': ('basis_rgb: &basis', 'illuminant: &illuminant', 'sensitivity_rgb: &sensitivity')}),
         ('mallett_complete', None))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_complete_tc_lut_privacy_and_borrowing(self) -> None:
        cases = (('tc_default', {'E0599': ('FilmTcLut::default()',)}),
         ('tc_private', {'E0451': ('FilmTcLut { ..lut }',)}),
         ('tc_readonly', {'E0594': ('lut.samples()[0]',)}),
         ('tc_borrow', {'E0515': ('lut.samples()',), 'E0505': ('drop(lut)',)}),
         ('tc_complete', None))
        feature_forms = (False,)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False,),
        )


    def test_complete_input_hull_privacy_and_borrows(self) -> None:
        cases = (('hull_default', {'E0599': ('Hull::default()',)}),
         ('hull_private', {'E0451': ('Hull { ..hull }',)}),
         ('hull_mutation', {'E0616': ('hull.xy', 'hull.hash', 'hull.center')}),
         ('hull_readonly', {'E0594': ('hull.xy()', 'hull.center()')}),
         ('hull_borrow', {'E0515': ('HullInput { x: &samples',), 'E0505': ('drop(hull)',)}),
         ('hull_complete', None))
        feature_forms = (False,)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False,),
        )

    def test_prepared_state_is_complete_private_and_borrowed(self) -> None:
        cases = (('prepared_state_complete', None),
         ('prepared_state_private', {'E0616': ('state.contents',), 'E0603': ('crate::preparation::project_state',)}),
         ('prepared_state_construction', {'E0451': ('contents: Vec::new()',)}),
         ('prepared_state_default', {'E0599': ('RenderState::default()',)}),
         ('prepared_state_mutation', {'E0594': ('state.s_inverse()[0]',)}),
         ('prepared_state_borrow', {'E0515': ('state.s_inverse()',), 'E0505': ('drop(state)',)}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_cuda_render_borrows_complete_owners_and_hides_projection(self) -> None:
        cases = (('cuda_render_complete', None),
         ('cuda_render_raw_records', {'E0308': ('raw_frame', 'raw_abort')}),
         ('cuda_render_capability_construction', {'E0451': ('FramePolicy { ..policy }',)}),
         ('cuda_render_projection_private', {'E0603': ('crate::cuda::projection::host_data',)}),
         ('cuda_render_state_borrow', {'E0505': ('drop(state)',)}),
         ('cuda_render_instance_borrow', {'E0505': ('drop(instance)',)}),
         ('cuda_render_owner_thread', {'E0277': ('std::thread::spawn(move || owner)',)}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )

    def test_ofx_callback_borrows(self) -> None:
        cases = (
            ("ofx_callback_scoped", None),
            ("ofx_callback_private", {
                "E0624": ("Callback::new(admission", "Parameters::new(callback", "callback.destroy()"),
            }),
            ("ofx_callback_send", {
                "E0277": ("scope.spawn(move || callback)", "scope.spawn(move || parameters)", "scope.spawn(move || instance_parameters)", "scope.spawn(move || owner)"),
            }),
            ("ofx_callback_escape", {
                "E0515": ("callback.descriptor_properties()", "callback.context_parameters()", "callback.parameter_access(&handles)", "callback.instance()?"),
            }),
        )
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=feature_forms,
        )

    def test_runtime_owner_borrows_and_terminal_consumption(self) -> None:
        cases = (('runtime_complete', None),
         ('runtime_private', {'E0616': ('runtime.assets', 'runtime.native', 'owner.handle')}),
         ('runtime_escape', {'E0515': ('runtime.assets()',)}),
         ('runtime_close_borrow', {'E0505': ('runtime.close()',)}),
         ('runtime_native_reuse', {'E0382': ('owner.raw()',)}),
         ('runtime_owner_borrow_send', {'E0277': ('std::thread::spawn(move || owner)',)}))
        feature_forms = (False, True)
        self.subtest_cases = tuple(
            f"{fixture}|{'release' if release else 'debug'}|{'test-support' if support else 'default'}"
            for fixture, _ in cases for release in self.profiles for support in feature_forms
        )
        self.check_fixture_cases(
            "film-juicer-plugin/src/lib.rs",
            "film-juicer-plugin",
            cases=cases,
            feature_forms=(False, True),
        )


RustBoundaryTests.test_film_input_color_is_complete_and_immutable.required_subtests = True
RustBoundaryTests.test_valid_profile_consumers_compile.required_subtests = True
RustBoundaryTests.test_profile_views_cannot_outlive_their_owners.required_subtests = True
RustBoundaryTests.test_output_color_is_complete_and_immutable.required_subtests = True
RustBoundaryTests.test_film_development_is_complete_private_and_borrowed.required_subtests = True
RustBoundaryTests.test_output_gamut_completed_api_is_private_and_borrowed.required_subtests = True
RustBoundaryTests.test_print_illuminant_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_print_preflash_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_film_reference_is_complete_private_readonly_and_borrowed.required_subtests = True
RustBoundaryTests.test_print_balance_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_frame_geometry_is_complete_private_and_readonly.required_subtests = True
RustBoundaryTests.test_dir_boundary_is_complete_private_readonly_and_borrowed.required_subtests = True
RustBoundaryTests.test_effects_frame_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_grain_frame_is_complete_private_readonly_and_owns_no_curve_borrow.required_subtests = True
RustBoundaryTests.test_dir_frame_is_complete_private_readonly_and_failure_trace_cannot_execute.required_subtests = True
RustBoundaryTests.test_diffusion_frame_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_halation_frame_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_gaussian_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_scanner_post_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_scanner_descriptor_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_scanner_color_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_scanner_lut_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_scanner_correction_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_optics_are_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_effects_are_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_film_grain_is_complete_private_readonly_and_owned.required_subtests = True
RustBoundaryTests.test_film_dir_density_is_complete_private_and_readonly.required_subtests = True
RustBoundaryTests.test_completed_profile_storage_is_private_to_its_owner.required_subtests = True
RustBoundaryTests.test_noise_borrows_and_storage_under_test_support.required_subtests = True
RustBoundaryTests.test_completed_profile_construction_cannot_be_bypassed.required_subtests = True
RustBoundaryTests.test_lens_construction_and_storage_are_private.required_subtests = True
RustBoundaryTests.test_safe_modules_cannot_relax_unsafe_prohibition.required_subtests = True
RustBoundaryTests.test_completed_spectral_results_are_private_and_readonly.required_subtests = True
RustBoundaryTests.test_completed_exposure_results_are_private_and_readonly.required_subtests = True
RustBoundaryTests.test_completed_exposure_results_do_not_retain_sources.required_subtests = True
RustBoundaryTests.test_complete_mallett_results_and_input_borrows.required_subtests = True
RustBoundaryTests.test_complete_tc_lut_privacy_and_borrowing.required_subtests = True
RustBoundaryTests.test_complete_input_hull_privacy_and_borrows.required_subtests = True
RustBoundaryTests.test_prepared_state_is_complete_private_and_borrowed.required_subtests = True
RustBoundaryTests.test_cuda_render_borrows_complete_owners_and_hides_projection.required_subtests = True
RustBoundaryTests.test_ofx_callback_borrows.required_subtests = True
RustBoundaryTests.test_runtime_owner_borrows_and_terminal_consumption.required_subtests = True


def exercise_scratch_lifetime_controls(test_case, owner):
    original_hold = os.environ.get("JUICER_TEST_RETAIN_FIXTURES")
    try:
        scenarios = (("success", False, False, False, False, False),
                     ("failure", True, False, False, False, False),
                     ("hold", False, True, False, False, False),
                     ("cleanup-error", False, False, True, False, False),
                     ("incomplete", False, False, False, True, False),
                     ("undrained", False, False, False, False, True))
        for name, failed, hold, cleanup_error, incomplete, undrained in scenarios:
            with test_case.subTest(case="scratch-lifetime:" + name), tempfile.TemporaryDirectory(prefix="boundary scratch lifetime ") as parent:
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
                        cls.workspace = cls.artifacts / "probe-fixture"
                        cls.expected_tests = {"test_control"}
                        if incomplete:
                            cls.expected_tests.add("test_unrun")
                        cls.completed_tests = set()
                        cls.retain_artifacts = False
                        cls.failure_snapshots = set()
                        cls.lease = ControlLease()
                        cls.native_run = ControlRun()
                        cls.addClassCleanup(cls.cleanup_artifacts)

                    def setUp(self):
                        self.workspace = self.artifacts / "probe-fixture"
                        self.workspace.mkdir()
                        (self.workspace / "mutated-source.rs").write_text(
                            "synthetic expected-negative probe\n", encoding="utf-8")
                        type(self).workspace = self.workspace
                        self.addCleanup(self.finalize_test)
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
                    raise AssertionError(f"Rust boundary control {name} produced the wrong outer outcome")
                if artifacts.exists() != retained:
                    raise AssertionError(f"Rust boundary scratch lifetime was wrong for {name}")
                if failed and not (artifacts / "failure-test_control" / "mutated-source.rs").is_file():
                    raise AssertionError("Rust boundary outer failure lost its mutated probe workspace")
    finally:
        if original_hold is None:
            os.environ.pop("JUICER_TEST_RETAIN_FIXTURES", None)
        else:
            os.environ["JUICER_TEST_RETAIN_FIXTURES"] = original_hold

if __name__ == "__main__":
    unittest.main()
