"""Checkpoint ownership, cross-caller reuse and exact CTest membership controls."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import textwrap
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/ctest"))
from control_artifacts import MARKER, record_fixture, selected_paths
import qualification
from execution import Campaign, OutputLease, digest, snapshot, tool_snapshot, write_json

ROOT = Path(__file__).resolve().parents[2]
SUITE_STARTED = time.monotonic()


class CheckpointTests(unittest.TestCase):
    def run(self, result=None):
        started = time.monotonic()
        print(f"Checkpoint begin {self.id()} suite_seconds={started - SUITE_STARTED:.3f}", file=sys.stderr, flush=True)
        try:
            return super().run(result)
        except KeyboardInterrupt:
            self.interrupted = True
            # unittest skips its cleanup callbacks when cancellation escapes.
            self.retain = True
            if hasattr(self, "lease") and self.lease.path in OutputLease.held:
                # Completed earlier commands cannot prove an interrupted fixture drained.
                self.lease.blocked = True
            self.doCleanups()
            raise
        finally:
            print(f"Checkpoint end {self.id()} case_seconds={time.monotonic() - started:.3f} "
                  f"suite_seconds={time.monotonic() - SUITE_STARTED:.3f}", file=sys.stderr, flush=True)

    def setUp(self):
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        artifact_root = os.environ.get("JUICER_TEST_ARTIFACT_DIR")
        parent = (Path(artifact_root) if artifact_root else ROOT / "out/validation" / preset) / "checkpoint-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, delete=False)
        self.directory = Path(self.temporary.name).resolve()
        self.root = self.directory / "project"
        self.retain = bool(os.environ.get("JUICER_TEST_RETAIN_FIXTURES"))
        if self.retain:
            print("Checkpoint fixture:", self.directory, file=sys.stderr, flush=True)
        def finish():
            if hasattr(self, "lease") and self.lease.path in OutputLease.held:
                self.lease.__exit__(None, None, None)
            outcome = record_fixture(self.directory, self, owner_root=self.root)
            failed = outcome != "passed"
            owners = list(self.root.glob("out/verification-locks/*/owner.json"))
            if owners or self.retain or failed:
                print("Retained checkpoint fixture:", self.directory, file=sys.stderr)
            else:
                try:
                    self.temporary.cleanup()
                except BaseException:
                    record_fixture(self.directory, self, "failed", owner_root=self.root)
                    raise
        self.addCleanup(finish)
        record_fixture(self.directory, self, "running", owner_root=self.root)

        self.root.mkdir()
        self.build = self.root / "out/build" / preset
        self.build.mkdir(parents=True)
        self.state_path = self.build / "verification-checkpoint.json"
        self.state = {}
        self.preset = preset
        self.environment = dict(os.environ)
        self.python = str(Path(sys.base_prefix) / "python.exe") if os.name == "nt" else sys.executable
        self.source = self.root / "input.txt"
        self.source.write_text("first")
        self.other = self.root / "other.txt"
        self.other.write_text("unrelated")
        self.lease = OutputLease(self.root, self.build)
        self.lease.__enter__()

    def campaign(self, name):
        return Campaign(self.root, self.preset, name, ["boundary", "tooling", "prepare", "corpus", "production"])

    def configure_fixture(self, preset, testing, name, *, timeout=30):
        from build_evidence import windows_vctip
        environment = dict(os.environ)
        campaign = Campaign(self.root, preset, name, ["prepare"])
        return campaign.run("prepare", ["cmake", "--preset", preset, "-DBUILD_TESTING=" + ("ON" if testing else "OFF")],
            inputs=[p for p in ("CMakeLists.txt", "CMakePresets.json", "binding.h.in") if (self.root / p).is_file()],
            env=environment, timeout=timeout, vctip_executable=windows_vctip(environment))

    def run_stage(self, name, *, campaign="first", inputs=None, command=None, tools=None, **kwargs):
        return qualification.run_stage(self.campaign(campaign), self.state_path, self.state, name,
            command or [self.python, "-c", "print('checked')"], inputs=inputs or ["input.txt"],
            tools=tools or [self.python], environment=self.environment, timeout=10, **kwargs)

    def test_new_campaign_reuses_original_success_without_execution(self):
        first = self.run_stage("boundary")
        result = self.run_stage("boundary", campaign="renamed")
        self.assertEqual(result["reused_from"], first["receipt"])
        self.assertEqual(result["sha256"], digest(Path(first["receipt"])))
        self.assertFalse((self.campaign("renamed").directory / "attempts").exists())

    def test_invalid_request_preserves_pass_before_reuse_or_input_change(self):
        first = self.run_stage("boundary")
        original_state = self.state_path.read_bytes()
        producer = Path(first["receipt"])
        evidence = {path: path.read_bytes() for path in producer.parent.iterdir() if path.is_file()}
        original_source = self.source.read_bytes()
        command = [self.python, "-c", "print('checked')"]
        cases = [
            ("undeclared", "boundary", command, ["input.txt"], 10, ["tooling"], "declared stage"),
            ("unsafe", "../boundary", command, ["input.txt"], 10, ["boundary"], "lowercase directory name"),
            ("command", "boundary", [], ["input.txt"], 10, ["boundary"], "command"),
            ("inputs", "boundary", command, [], 10, ["boundary"], "input files"),
            *[(f"timeout-{index}", "boundary", command, ["input.txt"], timeout, ["boundary"], "finite positive timeout")
              for index, timeout in enumerate((0, -1, float("nan"), float("inf"), -float("inf")))],
        ]
        for changed in (False, True):
            self.source.write_bytes(b"changed" if changed else original_source)
            for case, stage, selected_command, inputs, timeout, required, message in cases:
                with self.subTest(changed=changed, case=case):
                    campaign = Campaign(self.root, self.preset, f"rejected-{int(changed)}-{case}", required)
                    with patch.object(campaign, "run", side_effect=AssertionError("invalid request dispatched")) as run:
                        with self.assertRaisesRegex(ValueError, message):
                            qualification.run_stage(campaign, self.state_path, self.state, stage, selected_command,
                                inputs=inputs, tools=[self.python], environment=self.environment, timeout=timeout,
                                semantic_command=command, action=lambda: self.fail("invalid request invoked action"))
                        run.assert_not_called()
                    self.assertEqual(self.state_path.read_bytes(), original_state)
                    self.assertEqual(self.state, json.loads(original_state))
                    self.assertFalse((campaign.directory / "attempts").exists())
                    self.assertTrue(all(path.read_bytes() == content for path, content in evidence.items()))
        self.source.write_bytes(original_source)
        campaign = self.campaign("valid-after-rejection")
        with patch.object(campaign, "run", side_effect=AssertionError("reusable pass dispatched")):
            reused = qualification.run_stage(campaign, self.state_path, self.state, "boundary", command,
                inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10,
                action=lambda: self.fail("reusable pass invoked action"))
        self.assertEqual(reused["reused_from"], first["receipt"])
        self.assertEqual(self.state_path.read_bytes(), original_state)
        self.assertTrue(all(path.read_bytes() == content for path, content in evidence.items()))

    def test_invalid_request_preserves_cold_and_failed_checkpoint(self):
        command = [self.python, "-c", "raise SystemExit(7)"]
        rejected = Campaign(self.root, self.preset, "undeclared", ["tooling"])
        with self.assertRaisesRegex(ValueError, "declared stage"):
            qualification.run_stage(rejected, self.state_path, self.state, "boundary", command,
                inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10,
                action=lambda: self.fail("cold invalid request invoked action"))
        self.assertEqual(self.state, {})
        self.assertFalse(self.state_path.exists())
        self.assertFalse((rejected.directory / "attempts").exists())

        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("boundary", command=command)
        original_state = self.state_path.read_bytes()
        self.assertEqual(self.state["boundary"]["status"], "failed")
        attempts = self.campaign("first").directory / "attempts"
        evidence = {path: path.read_bytes() for path in attempts.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "declared stage"):
            qualification.run_stage(rejected, self.state_path, self.state, "boundary", command,
                inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10,
                action=lambda: self.fail("invalid request replaced failed gate"))
        self.assertEqual(self.state_path.read_bytes(), original_state)
        self.assertEqual(self.state, json.loads(original_state))
        self.assertFalse((rejected.directory / "attempts").exists())
        self.assertTrue(all(path.read_bytes() == content for path, content in evidence.items()))
        with self.assertRaisesRegex(ValueError, "pending checkpoint"):
            self.run_stage("boundary", campaign="valid-still-pending", command=command)

    def test_known_output_destinations_do_not_change_gate_identity(self):
        first = self.run_stage("boundary", semantic_command=[self.python, "same", "<junit>"])
        result = self.run_stage("boundary", campaign="renamed", command=[self.python, "unused-output-path"],
                                semantic_command=[self.python, "same", "<junit>"])
        self.assertEqual(result["reused_from"], first["receipt"])

    def test_unrelated_repair_preserves_success_and_pending_failure(self):
        first = self.run_stage("boundary")
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("tooling", inputs=["other.txt"], command=[self.python, "-c", "raise SystemExit(1)"])
        self.other.write_text("repair")
        result = self.run_stage("boundary", campaign="new-caller")
        self.assertEqual(result["reused_from"], first["receipt"])
        self.assertEqual(json.loads(self.state_path.read_text())["tooling"]["status"], "failed")
        self.run_stage("prepare", campaign="preparation", inputs=["other.txt"])
        self.assertEqual(self.state["tooling"]["status"], "failed")

    def test_dependency_content_with_restored_timestamp_invalidates_reuse(self):
        first = self.run_stage("boundary")
        stamp = self.source.stat()
        self.source.write_text("other")
        os.utime(self.source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        result = self.run_stage("boundary", campaign="changed")
        self.assertNotIn("reused_from", result)
        self.assertNotEqual(result["receipt"], first["receipt"])

    def test_failed_gate_cannot_be_reset_by_changing_caller(self):
        command = [self.python, "-c", "raise SystemExit(1)"]
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("boundary", command=command)
        with self.assertRaisesRegex(ValueError, "pending checkpoint"):
            self.run_stage("boundary", campaign="renamed", command=command)
        self.assertEqual(self.state["boundary"]["status"], "failed")

    def test_reuse_validation_failure_stays_pending_across_other_gates(self):
        first = self.run_stage("prepare")
        original = Path(first["receipt"]).read_bytes()
        def reject(path):
            raise ValueError("required output missing")
        with self.assertRaisesRegex(ValueError, "required output"):
            self.run_stage("prepare", campaign="missing-output", validate=reject)
        self.run_stage("tooling", campaign="different-caller", inputs=["other.txt"])
        self.assertEqual(self.state["prepare"]["status"], "failed")
        self.assertEqual(Path(first["receipt"]).read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "pending checkpoint"):
            self.run_stage("prepare", campaign="another-caller")

    def test_input_mutation_during_execution_cannot_publish_success(self):
        command = [self.python, "-c", "from pathlib import Path;Path('input.txt').write_text('mutated')"]
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_stage("boundary", command=command)
        self.assertEqual(self.state["boundary"]["status"], "failed")

    def test_old_pass_cannot_bypass_new_failure_during_collection(self):
        first = self.run_stage("boundary")
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "retained",
                  "preset": self.preset, "selection": ["boundary"], "required": ["boundary"], "stages": {"boundary": first}}
        path = self.root / "old-checkpoint.json"
        path.write_text(json.dumps(record))
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("boundary", campaign="failure", command=[self.python, "-c", "raise SystemExit(1)"])
        with self.assertRaisesRegex(ValueError, "pending"):
            qualification.collect_qualification(self.root, [path])

    def test_other_pending_gate_cannot_be_hidden_by_collecting_old_selection(self):
        first = self.run_stage("boundary")
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "retained",
                  "preset": self.preset, "selection": ["boundary"], "required": ["boundary"], "stages": {"boundary": first}}
        path = self.root / "old-checkpoint.json"
        path.write_text(json.dumps(record))
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("tooling", command=[self.python, "-c", "raise SystemExit(1)"])
        with self.assertRaisesRegex(ValueError, "pending checkpoint work"):
            qualification.collect_qualification(self.root, [path])

    def test_preparation_binds_its_owned_post_configuration_without_a_second_run(self):
        config = self.build / "CMakeCache.txt"
        config.write_text("before")
        inputs = ["input.txt", str(config.relative_to(self.root))]
        def prepare():
            path = self.campaign("first").run("prepare", [self.python, "-c", "print('configured')"], inputs=["input.txt"], timeout=10)
            config.write_text("after")
            return path
        first = self.run_stage("prepare", inputs=inputs, action=prepare)
        reused = self.run_stage("prepare", campaign="renamed", inputs=inputs, action=prepare)
        self.assertEqual(first["receipt"], reused["reused_from"])

    def test_boundary_inputs_bind_native_membership_and_foreign_fixture(self):
        for name in ("src/RenderRecipe.cpp", "src/JuicerState.cpp", "tests/ffi/color_preparation_test.cpp",
                     "tests/ffi/print_illuminant_test.inc", "tests/ffi/print_illuminant_fixture.h",
                     "tests/print/illuminant-fixtures/rust-manifest.json", "tests/ffi/print_preflash_test.inc",
                     "tests/print/preflash-fixtures/manifest.json", "tests/print/preflash-fixtures/linux-debug.json",
                     "tests/ffi/print_balance_test.inc", "tests/print/balance-fixtures/manifest.json",
                     "tests/print/balance-fixtures/linux-debug.json", "tests/ffi/scanner_correction_test.inc",
                     "tests/scanner/correction-fixtures/manifest.json", "tests/scanner/correction-fixtures/checked-expectations.json",
                     "tests/scanner/correction-fixtures/linux-debug.json", "tests/ffi/scanner_lut_test.inc", "tests/ffi/scanner_lut_descriptor_test.inc",
                     "tests/scanner/descriptor-fixtures/manifest.json", "tests/scanner/descriptor-fixtures/native.json",
                     "tests/scanner/descriptor-fixtures/context-linux-debug.json",
                     "tests/scanner/post-effects-fixtures/manifest.json", "tests/scanner/post-effects-fixtures/native.json",
                     "tests/scanner/post-effects-fixtures/context-linux-debug.json", "tests/ffi/scanner_post_effects_test.inc",
                     "tests/scanner/lut-fixtures/manifest.json", "tests/scanner/lut-fixtures/checked.json",
                     "tests/scanner/lut-fixtures/native.json", "tests/scanner/lut-fixtures/context-linux-debug.json",
                     "tests/ffi/exposure_fixture.h", "tests/ffi/scanner_color_test.inc", "tests/ffi/gaussian_kernel_test.inc", "tests/ffi/frame_geometry_test.inc",
                     "tests/frame/geometry-fixtures/manifest.json", "tests/frame/geometry-fixtures/native.json", "tests/frame/geometry-fixtures/context.json", "tests/frame/geometry-fixtures/inputs.json",
                     "tests/optics/gaussian-fixtures/manifest.json", "tests/optics/gaussian-fixtures/native.json", "tests/optics/gaussian-fixtures/context.json",
                     "tests/ffi/diffusion_frame_test.inc", "tests/diffusion/frame-fixtures/manifest.json",
                     "tests/diffusion/frame-fixtures/native.json", "tests/diffusion/frame-fixtures/context.json",
                     "tests/diffusion/frame-fixtures/checked.json", "tests/diffusion/frame-fixtures/inputs.json", "tests/diffusion/frame-fixtures/edge-inputs.json",
                     "tests/ffi/halation_frame_test.inc", "tests/scatter_halation/frame-fixtures/manifest.json",
                     "tests/scatter_halation/frame-fixtures/native.json", "tests/scatter_halation/frame-fixtures/context.json",
                     "tests/scatter_halation/frame-fixtures/checked.json", "tests/scatter_halation/frame-fixtures/inputs.json",
                     "tests/scanner/color-fixtures/manifest.json", "tests/scanner/color-fixtures/native.json",
                     "tests/scanner/color-fixtures/checked.json", "tests/ffi/film_reference_test.inc",
                     "tests/film/reference-fixtures/manifest.json", "tests/film/reference-fixtures/linux-debug.json"):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("source")
        (self.build / "CMakeCache.txt").write_text("configured")
        (self.build / "CTestTestfile.cmake").write_text("")
        first = qualification.checkpoint_inputs(self.root, self.build, "boundary")
        self.assertIn("src/RenderRecipe.cpp", first)
        self.assertIn("tests/ffi/film_reference_test.inc", first)
        self.assertIn("tests/film/reference-fixtures/manifest.json", first)
        self.assertIn("tests/film/reference-fixtures/linux-debug.json", first)
        self.assertIn("tests/ffi/color_preparation_test.cpp", first)
        self.assertIn("tests/ffi/print_illuminant_test.inc", first)
        self.assertIn("tests/ffi/print_illuminant_fixture.h", first)
        self.assertIn("tests/print/illuminant-fixtures/rust-manifest.json", first)
        self.assertIn("tests/ffi/print_preflash_test.inc", first)
        self.assertIn("tests/print/preflash-fixtures/manifest.json", first)
        self.assertIn("tests/print/preflash-fixtures/linux-debug.json", first)
        self.assertIn("tests/ffi/print_balance_test.inc", first)
        self.assertIn("tests/print/balance-fixtures/manifest.json", first)
        self.assertIn("tests/print/balance-fixtures/linux-debug.json", first)
        self.assertIn("tests/ffi/scanner_correction_test.inc", first)
        self.assertIn("tests/scanner/correction-fixtures/manifest.json", first)
        self.assertIn("tests/scanner/correction-fixtures/checked-expectations.json", first)
        self.assertIn("tests/scanner/correction-fixtures/linux-debug.json", first)
        self.assertIn("tests/ffi/scanner_color_test.inc", first)
        self.assertIn("tests/scanner/color-fixtures/manifest.json", first)
        self.assertIn("tests/scanner/color-fixtures/native.json", first)
        self.assertIn("tests/scanner/color-fixtures/checked.json", first)
        self.assertIn("tests/ffi/scanner_lut_descriptor_test.inc", first)
        self.assertIn("tests/scanner/descriptor-fixtures/manifest.json", first)
        self.assertIn("tests/scanner/descriptor-fixtures/native.json", first)
        self.assertIn("tests/scanner/descriptor-fixtures/context-linux-debug.json", first)
        self.assertIn("tests/scanner/post-effects-fixtures/manifest.json", first)
        self.assertIn("tests/scanner/post-effects-fixtures/native.json", first)
        self.assertIn("tests/scanner/post-effects-fixtures/context-linux-debug.json", first)
        self.assertIn("tests/ffi/scanner_lut_test.inc", first)
        self.assertIn("tests/scanner/lut-fixtures/manifest.json", first)
        self.assertIn("tests/scanner/lut-fixtures/checked.json", first)
        self.assertIn("tests/scanner/lut-fixtures/native.json", first)
        self.assertIn("tests/scanner/lut-fixtures/context-linux-debug.json", first)
        self.assertIn("tests/ffi/exposure_fixture.h", first)
        added = self.root / "src/SpectralProcessing.h"
        added.write_text("new member")
        self.assertIn("src/SpectralProcessing.h", qualification.checkpoint_inputs(self.root, self.build, "boundary"))
        self.assertNotIn("src/SpectralProcessing.h", first)
        fixture = self.root / "tests/print/illuminant-fixtures/product-updates.json"
        fixture.write_text("new selected expectation member")
        self.assertIn(str(fixture.relative_to(self.root)).replace("\\", "/"),
                      qualification.checkpoint_inputs(self.root, self.build, "boundary"))
        self.assertNotIn("tests/print/illuminant-fixtures/product-updates.json", first)

    def test_cargo_configuration_content_and_membership_are_tool_inputs(self):
        (self.build / "CMakeCache.txt").write_text("")
        home = self.root / "cargo-home"
        home.mkdir()
        config = home / "config.toml"
        environment = {"PATH": "", "CARGO_HOME": str(home), "RUSTUP_HOME": str(self.root / "rustup")}
        before = qualification.checkpoint_tools(self.build, environment)
        absent = self.run_stage("boundary", tools=before)
        config.write_text('[build]\n')
        after = qualification.checkpoint_tools(self.build, environment)
        self.assertIn(str(config), before)
        self.assertIn(str(config), after)
        first = self.run_stage("boundary", tools=after)
        self.assertNotEqual(absent["receipt"], first["receipt"])
        config.write_text('[build]\nrustflags=["--cfg", "changed"]\n')
        changed = self.run_stage("boundary", campaign="config-changed", tools=qualification.checkpoint_tools(self.build, environment))
        self.assertNotEqual(first["receipt"], changed["receipt"])
        self.assertNotIn("reused_from", changed)

    def native_project(self):
        (self.root / ".gitignore").write_text("out/\n")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True, timeout=10)
        (self.root / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.24)\nproject(Checkpoint NONE)\ninclude(CTest)\n'
            'add_custom_target(material ALL COMMAND "${CMAKE_COMMAND}" -E copy_if_different '
            '"${CMAKE_SOURCE_DIR}/input.txt" "${CMAKE_BINARY_DIR}/material.txt" '
            'DEPENDS "${CMAKE_SOURCE_DIR}/input.txt" BYPRODUCTS "${CMAKE_BINARY_DIR}/material.txt")\n'
            'if(BUILD_TESTING)\nadd_test(NAME boundary COMMAND "${CMAKE_COMMAND}" -E true)\n'
            'set_tests_properties(boundary PROPERTIES LABELS "host;boundary")\nendif()\n')
        (self.root / "CMakePresets.json").write_text(json.dumps({"version": 6, "configurePresets": [{
            "name": self.preset, "generator": "Ninja", "binaryDir": str(self.build)}]}))
        self.configure_fixture(self.preset, False, "fixture-base")

    def test_real_native_dispatcher_executes_and_reuses_exact_ctest_membership(self):
        self.native_project()
        self.configure_fixture(self.preset, True, "fixture-on")
        args = SimpleNamespace(reason="native caller regression", selection=["boundary"], preset=self.preset,
                               runtime="host", campaign="caller-one", repeat_reason="", base=None)
        import verify
        argv = ["verify.py", "qualification", "--preset", self.preset, "--reason", args.reason,
                "--selection", "boundary", "--runtime", "host", "--campaign", args.campaign]
        with patch.object(verify, "ROOT", self.root), patch.object(sys, "argv", argv):
            self.assertEqual(verify.main(), 0)
        first = json.loads((self.root / "out/validation" / self.preset / "caller-one/qualification.json").read_text())
        args.campaign = "caller-two"
        second = json.loads(qualification.qualify(self.root, args).read_text())
        self.assertEqual(first["membership"], {"boundary": ["boundary"]})
        self.assertEqual(second["status"], "passed")
        self.assertEqual(second["stages"]["boundary"]["reused_from"], first["stages"]["boundary"]["receipt"])
        self.assertFalse((self.root / "out/validation" / self.preset / "caller-two/attempts").exists())

    def test_documentation_changes_reuse_preparation_and_ctest_after_commit(self):
        from verify import source_snapshot
        self.native_project()
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text().replace('"host;boundary"', '"host;corpus"'))
        documents = [self.root / name for name in (
            "README.md", "CONTRIBUTING.md", "docs/workflow.md", "tests/README.md",
            "tests/film/fixtures/README.md")]
        for path in documents:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Initial documentation.\n")

        def commit(message):
            subprocess.run(["git", "add", "--all"], cwd=self.root, check=True, capture_output=True, timeout=10)
            subprocess.run(["git", "-c", "user.name=Checkpoint fixture", "-c", "user.email=checkpoint@example.invalid",
                            "-c", "commit.gpgsign=false", "commit", "-qm", message],
                           cwd=self.root, check=True, capture_output=True, timeout=10)

        commit("Initial miniature source and documentation")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, timeout=10)
        prepared = qualification.prepare_checkpoint(self.campaign("documentation-preparation"), self.build,
            self.state_path, self.state, self.environment, timeout=60)
        args = SimpleNamespace(reason="documentation discovery and retained native evidence", selection=["corpus"],
                               preset=self.preset, runtime="host", campaign="documentation-baseline", repeat_reason="", base=None)
        baseline = qualification.qualify(self.root, args)
        record = json.loads(baseline.read_text())
        stage = record["stages"]["corpus"]
        self.assertEqual(record["membership"], {"corpus": ["boundary"]})
        protected = {path: digest(path) for path in (baseline, Path(prepared["receipt"]), Path(stage["receipt"]))}
        state_bytes = self.state_path.read_bytes()
        source = source_snapshot(self.root)

        for operation in ("edit", "add", "delete", "commit"):
            with self.subTest(operation=operation):
                if operation == "edit":
                    for path in documents:
                        path.write_text("Revised documentation.\n")
                elif operation == "add":
                    for name in ("docs/new-guide.md", "tests/new-domain/README.md"):
                        path = self.root / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("New documentation.\n")
                elif operation == "delete":
                    for path in documents:
                        path.unlink()
                else:
                    commit("Documentation only")
                    self.assertNotEqual(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, timeout=10), head)
                self.assertEqual(source_snapshot(self.root), source)
                with patch.object(Campaign, "run", side_effect=AssertionError("documentation reran a gate")), \
                        patch.object(qualification, "prepare_build", side_effect=AssertionError("documentation rebuilt the fixture")):
                    current_state = json.loads(self.state_path.read_text())
                    reused = qualification.prepare_checkpoint(self.campaign("documentation-prepare-" + operation),
                        self.build, self.state_path, current_state, self.environment, timeout=60)
                    self.assertEqual(reused["reused_from"], prepared["receipt"])
                    args.campaign = "documentation-" + operation
                    current = json.loads(qualification.qualify(self.root, args).read_text())
                    self.assertEqual(current["stages"]["corpus"]["reused_from"], stage["receipt"])
                    with self.assertRaisesRegex(ValueError, "full checkpoint membership is incomplete"):
                        qualification.collect_qualification(self.root, [baseline])
                self.assertEqual(self.state_path.read_bytes(), state_bytes)
                self.assertEqual(protected, {path: digest(path) for path in protected})
                self.assertEqual(prepared["payloads"], {name: digest(Path(name)) for name in prepared["payloads"]})
                self.assertEqual(stage["payloads"], {name: digest(Path(name)) for name in stage["payloads"]})

    def runtime_project(self):
        self.native_project()
        fixture = self.root / "tests/fixtures/runtime.json"
        fixture.parent.mkdir(parents=True)
        fixture.write_text('{"value": 1}\n')
        loader = self.root / "tests/runtime_loader.py"
        loader.write_text("import json, pathlib, sys\n"
                          "value = json.loads(pathlib.Path(sys.argv[1]).read_text())['value']\n"
                          "pathlib.Path(sys.argv[2]).write_text(str(value))\n"
                          "assert value > 0\n")
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
            f'add_test(NAME runtime COMMAND "{Path(self.python).as_posix()}" '
            '"${CMAKE_SOURCE_DIR}/tests/runtime_loader.py" "${CMAKE_SOURCE_DIR}/tests/fixtures/runtime.json" '
            '"${CMAKE_BINARY_DIR}/observed.txt")\n'
            'set_tests_properties(runtime PROPERTIES LABELS "host;corpus")\n')
        self.configure_fixture(self.preset, True, "runtime-on")
        return fixture, loader

    def test_runtime_edit_reuses_preparation_but_executes_changed_test_input(self):
        import build_evidence
        fixture, loader = self.runtime_project()
        first = qualification.prepare_checkpoint(self.campaign("runtime-prepare"), self.build,
            self.state_path, self.state, self.environment, timeout=60)
        args = SimpleNamespace(reason="runtime preparation boundary", selection=["corpus"], preset=self.preset,
                               runtime="host", campaign="runtime-first", repeat_reason="", base=None)
        original = json.loads(qualification.qualify(self.root, args).read_text())["stages"]["corpus"]
        protected = {p: p.read_bytes() for p in (
            Path(first["receipt"]), *build_evidence.preparation_paths(self.build, ["all"])[:1],
            *(Path(name) for name in first["payloads"]))}
        for index, path in enumerate((fixture, loader), start=2):
            with self.subTest(path=path.name):
                if path == fixture:
                    path.write_text('{"value": 2}\n')
                else:
                    path.write_text(path.read_text().replace("str(value)", "str(value + 1)"))
                with patch.object(qualification, "prepare_build", side_effect=AssertionError("preparation redispatched")), \
                        patch.object(Campaign, "run", side_effect=AssertionError("configure/build/clean dispatched")):
                    reused = qualification.prepare_checkpoint(self.campaign("runtime-reuse-" + str(index)), self.build,
                        self.state_path, json.loads(self.state_path.read_text()), self.environment, timeout=60)
                self.assertEqual(reused["reused_from"], first["receipt"])
                self.assertTrue(all(p.read_bytes() == value for p, value in protected.items()))
                args.campaign = "runtime-changed-" + str(index)
                record_path = qualification.qualify(self.root, args)
                current = json.loads(record_path.read_text())["stages"]["corpus"]
                self.assertNotIn("reused_from", current)
                self.assertNotEqual(current["receipt"], original["receipt"])
                self.assertEqual((self.build / "observed.txt").read_text(), str(index))
                original = current

    def test_runtime_variable_configure_copy_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'set(stem runtime)\n'
            'execute_process(COMMAND "${CMAKE_COMMAND}" -E copy '
            '"${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" "${CMAKE_BINARY_DIR}/configured.json" '
            'COMMAND_ERROR_IS_FATAL ANY)\n')

    def test_runtime_variable_file_read_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'set(operation READ)\nset(stem runtime)\n'
            'file(${operation} "${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" payload)\n'
            'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/configured.json" CONTENT "${payload}")\n')

    def test_runtime_literal_file_read_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'file(READ "${CMAKE_SOURCE_DIR}/tests/fixtures/runtime.json" payload)\n'
            'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/configured.json" CONTENT "${payload}")\n')

    def test_runtime_build_tree_module_read_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'set(stem runtime)\n'
            'file(READ "${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" payload)\n'
            'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/configured.json" CONTENT "${payload}")\n',
            module=self.build / "consumer.cmake")

    def test_runtime_external_module_copy_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'set(stem runtime)\n'
            'execute_process(COMMAND "${CMAKE_COMMAND}" -E copy '
            '"${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" "${CMAKE_BINARY_DIR}/configured.json" '
            'COMMAND_ERROR_IS_FATAL ANY)\n',
            module=self.directory / "external/consumer.cmake")

    def test_runtime_bundled_module_copy_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'include(CPackIFWConfigureFile)\nset(stem runtime)\n'
            'cpack_ifw_configure_file("${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" '
            '"${CMAKE_BINARY_DIR}/configured.json")\n',
            bundled_module="CPackIFWConfigureFile.cmake")

    def test_runtime_included_listfile_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'set(stem runtime)\n'
            'file(READ "${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" payload)\n'
            'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/configured.json" CONTENT "${payload}")\n',
            module=self.build / "consumer.inc")

    def test_bootstrap_exceptions_require_exact_location_and_bytes(self):
        self.native_project()
        cache = self.build / "CMakeCache.txt"
        cmake_root = Path(next(line.partition("=")[2] for line in cache.read_text().splitlines()
                              if line.startswith("CMAKE_ROOT:INTERNAL=")))
        tool_root = self.root / "tool"
        bootstrap = tool_root / "Modules/CTestTargets.cmake"
        bootstrap.parent.mkdir(parents=True)
        content = (cmake_root / "Modules/CTestTargets.cmake").read_bytes()
        bootstrap.write_bytes(content)
        cache.write_text(f"CMAKE_ROOT:INTERNAL={tool_root.as_posix()}\n")

        def opaque(path):
            return qualification.has_opaque_configure_commands([], build=self.build,
                configuration={"regeneration_mtimes": {str(path): path.stat().st_mtime_ns}})

        self.assertFalse(opaque(bootstrap))
        self.assertTrue(qualification.has_opaque_configure_commands([bootstrap], build=self.build,
            configuration={"regeneration_mtimes": {str(bootstrap): bootstrap.stat().st_mtime_ns}}))
        # A known name/location alone never establishes the exception.
        bootstrap.write_bytes(content + b"\n# changed installation\n")
        self.assertTrue(opaque(bootstrap))
        shadow = self.build / "bootstrap.inc"
        shadow.write_bytes(content)
        self.assertTrue(opaque(shadow))
        shadow.write_bytes(b"\xff")
        self.assertTrue(opaque(shadow))

    def test_runtime_new_configure_membership_rejects_mid_preparation_edit(self):
        import build_evidence
        fixture, _ = self.runtime_project()
        module = self.build / "consumer.inc"
        module.write_text('set(stem runtime)\n'
            'file(READ "${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" payload)\n'
            'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/configured.json" CONTENT "${payload}")\n')
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + f'include("{module.as_posix()}")\n'
            'add_custom_target(staged ALL COMMAND "${CMAKE_COMMAND}" -E copy_if_different '
            '"${CMAKE_BINARY_DIR}/configured.json" "${CMAKE_BINARY_DIR}/staged.json" '
            'DEPENDS "${CMAKE_BINARY_DIR}/configured.json" BYPRODUCTS "${CMAKE_BINARY_DIR}/staged.json")\n')
        self.assertNotIn("tests/fixtures/runtime.json", qualification.checkpoint_inputs(self.root, self.build, "prepare"))
        run = Campaign.run
        edited = False

        def edit_after_configure(campaign, stage, command, **kwargs):
            nonlocal edited
            result = run(campaign, stage, command, **kwargs)
            if not edited and ("--preset" in command or "-S" in command):
                fixture.write_text('{"value": 2}\n')
                edited = True
            return result

        campaign = self.campaign("new-membership-edit")
        with patch.object(Campaign, "run", new=edit_after_configure):
            with self.assertRaisesRegex(ValueError, "newly discovered preparation inputs changed"):
                qualification.prepare_checkpoint(campaign, self.build, self.state_path, self.state, self.environment, timeout=60)
        self.assertTrue(edited)
        self.assertEqual(self.state["prepare"]["status"], "failed")
        write_json(campaign.directory / "failed-checkpoint.json", self.state)
        for name in ("configured.json", "staged.json"):
            self.assertEqual(json.loads((self.build / name).read_text())["value"], 1)
        baseline = build_evidence.preparation_paths(self.build, ["all"])[0]
        original = Path(json.loads(baseline.read_text())["producer"])
        original_receipt = original.read_bytes()
        renewed = qualification.prepare_checkpoint(self.campaign("new-membership-repaired"), self.build,
            self.state_path, self.state, self.environment, timeout=60)
        self.assertNotIn("reused_from", renewed)
        for name in ("configured.json", "staged.json"):
            self.assertEqual(json.loads((self.build / name).read_text())["value"], 2)
        self.assertEqual(original.read_bytes(), original_receipt)

    def assert_configure_copy_reprepared(self, configure_commands, *, module=None, bundled_module=None):
        started = time.monotonic()
        def progress(stage):
            print(f"Checkpoint {self._testMethodName} stage={stage} seconds={time.monotonic() - started:.3f}",
                  file=sys.stderr, flush=True)
        progress("runtime-project")
        fixture, _ = self.runtime_project()
        cmake = self.root / "CMakeLists.txt"
        if module:
            module.parent.mkdir(parents=True, exist_ok=True)
            module.write_text(configure_commands)
            configure_commands = f'include("{module.as_posix()}")\n'
        cmake.write_text(cmake.read_text() + configure_commands +
            'add_custom_target(staged ALL COMMAND "${CMAKE_COMMAND}" -E copy_if_different '
            '"${CMAKE_BINARY_DIR}/configured.json" "${CMAKE_BINARY_DIR}/staged.json" '
            'DEPENDS "${CMAKE_BINARY_DIR}/configured.json" BYPRODUCTS "${CMAKE_BINARY_DIR}/staged.json")\n')
        if module or bundled_module:
            from verify import source_snapshot
            self.assertFalse(qualification.has_opaque_configure_commands(self.root / name for name in source_snapshot(self.root)))
        progress("first-preparation")
        first = qualification.prepare_checkpoint(self.campaign("variable-copy-first"), self.build,
            self.state_path, self.state, self.environment, timeout=60)
        progress("first-preparation-complete")
        producer = Path(first["receipt"])
        original_receipt = producer.read_bytes()
        if bundled_module:
            configuration = json.loads(original_receipt)["scoped_preparation"]["context"]["configuration"]
            module = next(Path(name) for name in configuration["regeneration_mtimes"] if Path(name).name == bundled_module)
        if module:
            configuration = json.loads(original_receipt)["scoped_preparation"]["context"]["configuration"]
            self.assertIn(str(module), configuration["regeneration_mtimes"])
            identities = configuration["inputs"] if module.is_relative_to(self.root) else configuration["tools"]
            name = module.relative_to(self.root).as_posix() if module.is_relative_to(self.root) else str(module)
            self.assertEqual(identities[name], digest(module))
            self.assertNotIn(str(fixture), configuration["regeneration_mtimes"])
            original_module = module.read_bytes()
        self.assertIn("tests/fixtures/runtime.json", first["identity"]["inputs"])
        configured, staged = (self.build / name for name in ("configured.json", "staged.json"))
        self.assertEqual(json.loads(staged.read_text())["value"], 1)
        self.assertIn(str(staged), first["payloads"])
        (producer.parent / "original-staged.json").write_bytes(staged.read_bytes())
        progress("unchanged-admission")
        with patch.object(qualification, "prepare_build", side_effect=AssertionError("unchanged opaque scope redispatched")), \
                patch.object(Campaign, "run", side_effect=AssertionError("unchanged opaque scope executed")):
            unchanged = qualification.prepare_checkpoint(self.campaign("variable-copy-unchanged"), self.build,
                self.state_path, self.state, self.environment, timeout=60)
        self.assertEqual(unchanged["reused_from"], first["receipt"])
        progress("unchanged-admission-complete")
        fixture.write_text('{"value": 2}\n')
        commands = []
        run = Campaign.run

        def observe(campaign, stage, command, **kwargs):
            commands.append(list(command))
            return run(campaign, stage, command, **kwargs)

        progress("changed-preparation")
        with patch.object(Campaign, "run", new=observe):
            second = qualification.prepare_checkpoint(self.campaign("variable-copy-changed"), self.build,
                self.state_path, self.state, self.environment, timeout=60)
        progress("changed-preparation-complete")
        write_json(self.build / "variable-copy-control.json", {
            "first_receipt": first["receipt"], "second_receipt": second["receipt"],
            "commands": commands, "runtime": json.loads(fixture.read_text()),
            "configured": json.loads(configured.read_text()), "staged": json.loads(staged.read_text())})
        self.assertEqual(json.loads(configured.read_text())["value"], 2)
        self.assertEqual(json.loads(staged.read_text())["value"], 2)
        if module:
            self.assertEqual(module.read_bytes(), original_module)
        self.assertNotIn("reused_from", second)
        self.assertNotEqual(second["receipt"], first["receipt"])
        self.assertIn("tests/fixtures/runtime.json", second["identity"]["inputs"])
        self.assertTrue(any("--preset" in command or "-S" in command for command in commands))
        self.assertEqual(producer.read_bytes(), original_receipt)

    def test_runtime_preparation_keeps_build_consumers_and_unknown_inputs(self):
        import build_evidence
        fixture, loader = self.runtime_project()
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
            'configure_file(tests/fixtures/runtime.json generated.json COPYONLY)\n'
            'add_custom_target(staged ALL COMMAND "${CMAKE_COMMAND}" -E copy '
            '"${CMAKE_SOURCE_DIR}/tests/runtime_loader.py" "${CMAKE_BINARY_DIR}/staged.py" '
            'DEPENDS "${CMAKE_SOURCE_DIR}/tests/runtime_loader.py" BYPRODUCTS "${CMAKE_BINARY_DIR}/staged.py")\n')
        unknown = self.root / "tests/fixtures/unknown.json"
        unknown.write_text("unknown dependency")
        resources = self.root / "Resources/source.json"
        resources.parent.mkdir()
        resources.write_text("resource")
        tool = self.root / "generator.py"
        tool.write_text("# build generator\n")
        cmake.write_text(cmake.read_text() +
            'add_custom_target(generator ALL COMMAND "${CMAKE_COMMAND}" -E true '
            'DEPENDS "${CMAKE_SOURCE_DIR}/generator.py")\n')
        first = qualification.prepare_checkpoint(self.campaign("build-consumers"), self.build,
            self.state_path, self.state, self.environment, timeout=60)
        preparation = qualification.checkpoint_inputs(self.root, self.build, "prepare")
        for path in (fixture, loader, unknown, resources, cmake, tool):
            self.assertIn(path.relative_to(self.root).as_posix(), preparation)
            before = path.read_bytes()
            path.write_bytes(before + b" ")
            self.assertNotEqual(snapshot(self.root, preparation), first["identity"]["inputs"])
            with self.assertRaises(ValueError):
                build_evidence.check_scoped_preparation(self.campaign("build-consumers"), self.build, first["receipt"])
            path.write_bytes(before)

    def test_runtime_preparation_retains_configure_copies_and_opaque_cargo_inputs(self):
        fixture, _ = self.runtime_project()
        cmake = self.root / "CMakeLists.txt"
        original = cmake.read_text()
        for operation in (
            'file(COPY "${CMAKE_SOURCE_DIR}/tests/fixtures/runtime.json" DESTINATION "${CMAKE_BINARY_DIR}/copied")\n',
            'file(READ "${CMAKE_SOURCE_DIR}/tests/fixtures/runtime.json" fixture_text)\n',
            'set(fixture "${CMAKE_SOURCE_DIR}/tests/fixtures/runtime.json")\nfile(READ "${fixture}" fixture_text)\n',
            'execute_process(COMMAND "${CMAKE_COMMAND}" -E cat "${CMAKE_SOURCE_DIR}/tests/fixtures/runtime.json")\n',
            'set(operation execute_process)\ncmake_language(CALL ${operation} COMMAND "${CMAKE_COMMAND}" -E true)\n',
        ):
            with self.subTest(operation=operation):
                cmake.write_text(original + operation)
                self.assertIn("tests/fixtures/runtime.json", qualification.checkpoint_inputs(self.root, self.build, "prepare"))
        cmake.write_text(original)
        cmake.write_text(original +
            'add_custom_target(copy_directory ALL COMMAND "${CMAKE_COMMAND}" -E copy_directory '
            '"${CMAKE_SOURCE_DIR}/tests/fixtures" "${CMAKE_BINARY_DIR}/staged-fixtures")\n')
        self.configure_fixture(self.preset, True, "runtime-directory-copy")
        self.assertIn("tests/fixtures/runtime.json", qualification.checkpoint_inputs(self.root, self.build, "prepare"))
        cmake.write_text(original)
        self.configure_fixture(self.preset, True, "runtime-directory-copy-restored")
        for name, content in (("rust/src/lib.rs", 'const DATA: &str = include_str!("data.json");\n'),
                              ("rust/src/lib.rs", 'const DATA: &[u8] = include_bytes!("data.json");\n'),
                              ("rust/src/lib.rs", 'include!("data.json");\n'),
                              ("Cargo.toml", '[package]\nbuild = "custom_build.rs"\n'),
                              ("rust/build.rs", "fn main() {}\n")):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            self.assertIn(fixture.relative_to(self.root).as_posix(),
                          qualification.checkpoint_inputs(self.root, self.build, "prepare"))
            path.unlink()

    def test_runtime_preparation_preserves_explicit_and_older_broad_requirements(self):
        import build_evidence
        fixture, _ = self.runtime_project()
        with patch.object(qualification, "preparation_inputs", side_effect=lambda root, build, paths, **kwargs: paths):
            first = qualification.prepare_checkpoint(self.campaign("broad"), self.build,
                self.state_path, self.state, self.environment, timeout=60)
        receipt = Path(first["receipt"])
        original = receipt.read_bytes()
        fixture.write_text('{"value": 2}\n')
        with self.assertRaisesRegex(ValueError, "dependency contents or membership changed"):
            build_evidence.check_scoped_preparation(self.campaign("broad"), self.build, receipt)
        renewed = qualification.prepare_checkpoint(self.campaign("narrow"), self.build,
            self.state_path, self.state, self.environment, timeout=60)
        self.assertNotEqual(renewed["receipt"], first["receipt"])
        self.assertEqual(receipt.read_bytes(), original)
        # Direct prepare_build inputs remain authoritative even for a file the
        # checkpoint can prove is runtime-only.
        explicit = build_evidence.prepare_build(self.campaign("explicit"), "prepare", source=self.root,
            build=self.build, inputs=["input.txt", "tests/fixtures/runtime.json"], timeout=60,
            targets=["all"], configure_args=["-DBUILD_TESTING=ON"], checkpoint=True)[1]
        original_explicit = explicit.read_bytes()
        fixture.write_text('{"value": 3}\n')
        with self.assertRaisesRegex(ValueError, "dependency contents or membership changed"):
            build_evidence.check_scoped_preparation(self.campaign("explicit"), self.build, explicit)
        self.assertEqual(explicit.read_bytes(), original_explicit)

    def test_runtime_preparation_does_not_admit_tampered_or_unproved_owners(self):
        import build_evidence
        fixture, _ = self.runtime_project()
        first = qualification.prepare_checkpoint(self.campaign("owned"), self.build,
            self.state_path, self.state, self.environment, timeout=60)
        baseline, pending = build_evidence.preparation_paths(self.build, ["all"])
        original_owner = baseline.read_bytes()
        original_receipt = Path(first["receipt"]).read_bytes()
        artifact = next(Path(name) for name in first["payloads"] if Path(name).name == "material.txt")
        original_artifact = artifact.read_bytes()
        fixture.write_text('{"value": 2}\n')
        for fault in ("artifact", "pending", "unproved"):
            with self.subTest(fault=fault):
                if fault == "artifact":
                    artifact.write_bytes(b"tampered")
                elif fault == "pending":
                    pending.write_text('{"status": "unproved"}')
                else:
                    baseline.unlink()
                with patch.object(Campaign, "run", side_effect=AssertionError("unsafe owner dispatched")):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        qualification.prepare_checkpoint(self.campaign("rejected-" + fault), self.build,
                            self.state_path, json.loads(self.state_path.read_text()), self.environment, timeout=60)
                self.assertEqual(Path(first["receipt"]).read_bytes(), original_receipt)
                # Faults are isolated in this miniature, never shared owners.
                artifact.write_bytes(original_artifact)
                if pending.exists():
                    pending.rename(self.build / "unproved-pending-retained.json")
                if not baseline.exists():
                    baseline.write_bytes(original_owner)
                write_json(self.state_path, {"prepare": first})

    def test_runtime_preparation_collection_uses_its_own_membership(self):
        self.assert_runtime_preparation_collection()

    def test_runtime_module_preparation_collection_keeps_configure_inputs(self):
        self.assert_runtime_preparation_collection(module=self.build / "consumer.cmake")

    def assert_runtime_preparation_collection(self, *, module=None):
        fixture, _ = self.runtime_project()
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text().replace('"host;corpus"', '"host;product"'))
        if module:
            module.write_text('set(stem runtime)\n'
                'file(READ "${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" payload)\n'
                'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/configured.json" CONTENT "${payload}")\n')
            cmake.write_text(cmake.read_text() + f'include("{module.as_posix()}")\n')
        quality = self.root / "scripts/check-quality.py"
        quality.parent.mkdir()
        quality.write_text("print('miniature quality scheduling witness')\n")
        args = SimpleNamespace(reason="miniature preparation collection", selection=["product"], preset=self.preset,
                               runtime="all", campaign="collection-first", repeat_reason="", base="fixture")
        path = qualification.qualify(self.root, args)
        record = json.loads(path.read_text())
        self.assertEqual("tests/fixtures/runtime.json" in record["stages"]["prepare"]["identity"]["inputs"], bool(module))
        self.assertIn("tests/fixtures/runtime.json", record["stages"]["product"]["identity"]["inputs"])
        original = path.read_bytes()
        with patch.object(Campaign, "run", side_effect=AssertionError("collector dispatched")):
            with self.assertRaisesRegex(ValueError, "full checkpoint membership is incomplete"):
                qualification.collect_qualification(self.root, [path])
        fixture.write_text('{"value": 2}\n')
        if module:
            with patch.object(Campaign, "run", side_effect=AssertionError("collector dispatched")):
                with self.assertRaisesRegex(ValueError, "checkpoint declared source/data inputs changed"):
                    qualification.collect_qualification(self.root, [path])
        args.campaign = "collection-changed"
        refreshed = qualification.qualify(self.root, args)
        result = json.loads(refreshed.read_text())
        if module:
            self.assertNotIn("reused_from", result["stages"]["prepare"])
            self.assertEqual(json.loads((self.build / "configured.json").read_text())["value"], 2)
        else:
            self.assertEqual(result["stages"]["prepare"]["reused_from"], record["stages"]["prepare"]["receipt"])
        self.assertNotIn("reused_from", result["stages"]["product"])
        with self.assertRaisesRegex(ValueError, "full checkpoint membership is incomplete"):
            qualification.collect_qualification(self.root, [refreshed])
        self.assertEqual(path.read_bytes(), original)

    def test_runtime_preparation_cold_discovery_preserves_producer_requirements(self):
        self.native_project()
        self.assertFalse((self.build / "CTestTestfile.cmake").exists())
        fixture = self.root / "tests/runtime.txt"
        fixture.parent.mkdir()
        fixture.write_text("runtime input")
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
            'add_test(NAME runtime COMMAND "${CMAKE_COMMAND}" -E cat "${CMAKE_SOURCE_DIR}/tests/runtime.txt")\n')
        first = qualification.prepare_checkpoint(self.campaign("cold-discovery"), self.build,
            self.state_path, self.state, self.environment, timeout=60)
        proof = json.loads(Path(first["receipt"]).read_text())["scoped_preparation"]
        declared = {name for name in first["identity"]["inputs"] if not name.startswith("out/build/")}
        self.assertEqual(declared, set(proof["inputs"]))
        self.assertIn("tests/runtime.txt", declared)
        self.assertIn("tests/runtime.txt", qualification.checkpoint_inputs(self.root, self.build, "prepare",
            preparation=proof["context"]))
        self.assertNotIn("tests/runtime.txt", qualification.checkpoint_inputs(self.root, self.build, "prepare"))

    def test_explicitly_consumed_markdown_remains_a_freshness_input(self):
        name = "docs/runtime-contract.md"
        document = self.root / name
        document.parent.mkdir(parents=True)
        document.write_text("first")
        self.assertIn(name, qualification.checkpoint_inputs(self.root, self.build, "corpus", source_files=[name]))
        command = [self.python, "-c", "from pathlib import Path; assert Path('docs/runtime-contract.md').read_text() == 'first'"]
        first = self.run_stage("boundary", command=command, inputs=[name])
        receipt = Path(first["receipt"])
        original = receipt.read_bytes()
        document.write_text("changed consumed content")
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("boundary", campaign="changed-markdown", command=command, inputs=[name])
        self.assertEqual(self.state["boundary"]["status"], "failed")
        self.assertEqual(receipt.read_bytes(), original)

    def production_fixture(self):
        """Native tiny probes exercise retained proof, not product mathematics."""
        for name in ("grain_frame_abi_assertions.inc", "grain_probe_contract.py"):
            target = self.root / "tests/ffi" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / "tests/ffi" / name).read_bytes())
        directory = self.root / "tests/film/reference-fixtures"
        directory.mkdir(parents=True, exist_ok=True)
        payload = directory / "mini.json"
        write_json(payload, {"scope": "fixture membership protocol only", "products": [
            {"method": method, "input": {}, "reference": {"hash": 10}} for method in (0, 1)]})
        write_json(directory / "manifest.json", {"presets": {
            preset: payload.name for preset in qualification.PRESETS},
            "payload_sha256": {payload.name: digest(payload)}})
        directory = self.root / "tests/scanner/descriptor-fixtures"
        directory.mkdir(parents=True, exist_ok=True)
        payload = directory / "native.json"
        context = directory / "context.json"
        write_json(payload, {"scope": "fixture membership protocol only"})
        write_json(context, {"scope": "distinct native context protocol"})
        write_json(directory / "manifest.json", {"presets": {preset: {"native": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": payload.name}, "contexts": {"logical.json": context.name},
            "sha256": {payload.name: digest(payload), context.name: digest(context)},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0, 0, 0], "hash": 10, "fields": "10-pass"} for i in range(24)],
            "production_malformed": [{"case": str(i), "envelope": [2, 0, 0, 1, 0], "hash": 0, "fields": "10-pass"} for i in range(17)]})
        post_directory = self.root / "tests/scanner/post-effects-fixtures"
        post_directory.mkdir(parents=True, exist_ok=True)
        post_payload = post_directory / "native.json"
        post_context = post_directory / "context.json"
        write_json(post_payload, {"scope": "post-effects retained membership protocol only"})
        write_json(post_context, {"scope": "distinct post-effects native context protocol"})
        write_json(post_directory / "manifest.json", {"presets": {preset: {"native": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": post_payload.name}, "contexts": {"logical.json": post_context.name},
            "sha256": {post_payload.name: digest(post_payload), post_context.name: digest(post_context)},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0, 0, 0], "hash": 10, "fields": "13-pass"} for i in range(410)]})
        gaussian_directory = self.root / "tests/optics/gaussian-fixtures"
        gaussian_directory.mkdir(parents=True, exist_ok=True)
        gaussian_payload = gaussian_directory / "native.json"
        gaussian_context = gaussian_directory / "context.json"
        write_json(gaussian_payload, {"cases": [{"name": str(i), "radius": 1, "sigma_bits": 1065353216, "weights": [1, 2, 3]} for i in range(188)]})
        write_json(gaussian_context, {"scope": "Gaussian retained-array protocol only"})
        names = [str(i) for i in range(188)] + ["radius-upper-reject", "null-input", "unaligned-input", "null-owner-output", "null-failure", "malformed-diagnostic", "null-view-owner", "null-view-output", "null-view-failure", "view-malformed-diagnostic", "zero-capacity", "null-release", "release-malformed-diagnostic"]
        write_json(gaussian_directory / "manifest.json", {"presets": {preset: {"native": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": gaussian_payload.name}, "contexts": {"logical.json": gaussian_context.name},
            "sha256": {gaussian_payload.name: digest(gaussian_payload), gaussian_context.name: digest(gaussian_context)}, "production_cases": names})
        frame_directory = self.root / "tests/frame/geometry-fixtures"
        frame_directory.mkdir(parents=True, exist_ok=True)
        frame_payload = frame_directory / "native.json"
        frame_context = frame_directory / "context.json"
        frame_inputs = frame_directory / "inputs.json"
        write_json(frame_payload, {"scope": "frame full-field retained protocol only"})
        write_json(frame_context, {"scope": "distinct frame native context protocol"})
        write_json(frame_inputs, {"scope": "frame consumed specification protocol"})
        write_json(frame_directory / "manifest.json", {
            "presets": {preset: {"native": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": frame_payload.name}, "contexts": {"logical.json": frame_context.name},
            "sha256": {p.name: digest(p) for p in (frame_payload, frame_context, frame_inputs)},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0, 0], "domain": [1, 2, 3, 4], "pixel_bits": 1, "diffusion_bits": 2, "diagnostic": ""} for i in range(69)]})
        halation_directory = self.root / "tests/scatter_halation/frame-fixtures"
        halation_directory.mkdir(parents=True, exist_ok=True)
        halation_files = {name: halation_directory / name for name in ("native.json", "context.json", "checked.json", "inputs.json")}
        for name, path in halation_files.items():
            write_json(path, {"scope": "halation retained full-field protocol only", "file": name})
        write_json(halation_directory / "manifest.json", {
            "presets": {preset: {"native": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": "native.json"}, "contexts": {"logical.json": "context.json"}, "checked": "checked.json",
            "sha256": {name: digest(path) for name, path in halation_files.items()},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0], "failure": [0, 0, 3, 3], "sigma_bits": 0, "sigma_nan": False,
                                  "output": {"recipe_hash": i + 1, "words": [0] * 528 + [i]}, "diagnostic": ""} for i in range(86)]})
        diffusion_directory = self.root / "tests/diffusion/frame-fixtures"
        diffusion_directory.mkdir(parents=True, exist_ok=True)
        diffusion_files = {name: diffusion_directory / name for name in ("native.json", "context.json", "checked.json", "inputs.json", "edge-inputs.json")}
        for name, path in diffusion_files.items():
            write_json(path, {"scope": "diffusion retained full-field protocol only", "file": name})
        write_json(diffusion_directory / "manifest.json", {
            "presets": {preset: {"native": "logical.json", "edges": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": "native.json"}, "contexts": {"logical.json": "context.json"}, "checked": "checked.json",
            "sha256": {name: digest(path) for name, path in diffusion_files.items()},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0], "failure": [0, 0, 0],
                                  "output": [0] * 35 + [i], "diagnostic": ""} for i in range(507)]})
        dir_directory = self.root / "tests/dir/frame-fixtures"
        dir_directory.mkdir(parents=True, exist_ok=True)
        dir_files = {name: dir_directory / name for name in ("native.json", "checked.json", "inputs.json")}
        for name, path in dir_files.items():
            write_json(path, {"scope": "DIR retained full-field/trace protocol only", "file": name})
        write_json(dir_directory / "manifest.json", {
            "presets": {preset: {"native": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": "native.json"}, "checked": "checked.json",
            "sha256": {name: digest(path) for name, path in dir_files.items()},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0], "failure": [0, 0, 4] + [0] * 79,
                                  "output": [0] * 78 + [i], "diagnostic": ""} for i in range(121)]})
        grain_directory = self.root / "tests/grain/frame-fixtures"
        grain_directory.mkdir(parents=True, exist_ok=True)
        grain_files = {name: grain_directory / name for name in ("native.json", "checked.json", "inputs.json")}
        for name, path in grain_files.items():
            write_json(path, {"scope": "grain frame retained full-value/metadata protocol only", "file": name})
        write_json(grain_directory / "manifest.json", {
            "presets": {preset: {"native": "native.json"} for preset in qualification.PRESETS},
            "checked": "checked.json", "sha256": {name: digest(path) for name, path in grain_files.items()},
            "production_cases": [{"case": case, "envelope": [0, 0, 0], "failure": [0, 0, 3, 3],
                                  "output": [0] * 114 + [i], "diagnostic": "", "diagnostic_length": 0}
                                 for i, case in enumerate(qualification.GRAIN_FRAME_CASE_IDS)]})
        boundary_directory = self.root / "tests/dir/boundary-fixtures"
        boundary_directory.mkdir(parents=True, exist_ok=True)
        boundary_files = {name: boundary_directory / name for name in ("native.json", "checked.json", "inputs.json")}
        for name, path in boundary_files.items():
            write_json(path, {"scope": "DIR boundary retained full-values protocol only", "file": name})
        write_json(boundary_directory / "manifest.json", {
            "presets": {preset: {"native": "native.json"} for preset in qualification.PRESETS},
            "checked": "checked.json", "sha256": {name: digest(path) for name, path in boundary_files.items()},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0], "key": i, "failure": [0, 0, 3, 0, 0, 0, 0],
                                  "shape": [1, 2, 3, 4, 1, 2], "weights": [i], "terminal": [0] * 17 + [i],
                                  "diagnostic": "", "diagnostic_length": 0, "owner": True} for i in range(171)]})
        return r'''
import json, sys, subprocess, hashlib, shutil
preset=sys.argv[sys.argv.index("--preset")+1]
out=out.resolve()
def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
build=(Path('out/build')/preset).resolve()
build_sources=[build/name for name in ('CMakeCache.txt','build.ninja','compile_commands.json','cargo-features.txt',
                                     'libjuicer.a','juicer.lib','juicer.ofx','juicer.pdb') if (build/name).is_file()]
assert 'BUILD_TESTING:BOOL=OFF' in (build/'CMakeCache.txt').read_text().splitlines()
build_inputs={}
for source in build_sources:
    before=checksum(source)
    target=out/'build'/source.relative_to(build)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    assert checksum(source)==checksum(target)==before
    build_inputs[str(source)]={'file':str(target),'sha256':before}
commands=[]
def execute(name, command):
    result=subprocess.run(command,capture_output=True,timeout=30,check=False)
    for extension,data in [('stdout',result.stdout),('stderr',result.stderr)]:
        (out/(name+'.'+extension)).write_bytes(data)
    commands.append({'name':name,'command':command,'returncode':result.returncode,
                     'stdout_sha256':checksum(out/(name+'.stdout')),
                     'stderr_sha256':checksum(out/(name+'.stderr'))})
    result.check_returncode()
header=out/'production-film-expected.h'
header.write_text('#define EXPECTED 10\n')
consumed={str(header):checksum(header)}
for name in ('grain_frame_abi_assertions.inc', 'grain_probe_contract.py'):
    helper=(Path('tests/ffi')/name).resolve()
    consumed[str(helper)]=checksum(helper)
reference_manifest=(Path('tests/film/reference-fixtures')/'manifest.json').resolve()
reference_spec=json.loads(reference_manifest.read_text())
reference_payload=reference_manifest.parent/reference_spec['presets'][preset]
consumed.update({str(path):checksum(path) for path in (reference_manifest,reference_payload)})
descriptor_manifest=(Path('tests/scanner/descriptor-fixtures')/'manifest.json').resolve()
descriptor_spec=json.loads(descriptor_manifest.read_text())
descriptor_payload=descriptor_manifest.parent/'native.json'
descriptor_context=descriptor_manifest.parent/'context.json'
consumed.update({str(path):checksum(path) for path in (descriptor_manifest,descriptor_payload,descriptor_context)})
descriptor_results=descriptor_spec['production_cases']+descriptor_spec['production_malformed']
post_manifest=(Path('tests/scanner/post-effects-fixtures')/'manifest.json').resolve()
post_spec=json.loads(post_manifest.read_text())
post_payload=post_manifest.parent/'native.json'
post_context=post_manifest.parent/'context.json'
consumed.update({str(path):checksum(path) for path in (post_manifest,post_payload,post_context)})
post_results=post_spec['production_cases']
frame_manifest=(Path('tests/frame/geometry-fixtures')/'manifest.json').resolve()
frame_spec=json.loads(frame_manifest.read_text())
frame_payload=frame_manifest.parent/'native.json'
frame_context=frame_manifest.parent/'context.json'
consumed.update({str(path):checksum(path) for path in (frame_manifest,frame_payload,frame_context,frame_manifest.parent/'inputs.json')})
frame_results=frame_spec['production_cases']
halation_manifest=(Path('tests/scatter_halation/frame-fixtures')/'manifest.json').resolve()
halation_spec=json.loads(halation_manifest.read_text())
halation_payload=halation_manifest.parent/'native.json'
halation_context=halation_manifest.parent/'context.json'
halation_checked=halation_manifest.parent/'checked.json'
halation_inputs=halation_manifest.parent/'inputs.json'
consumed.update({str(path):checksum(path) for path in (halation_manifest,halation_payload,halation_context,halation_checked,halation_inputs)})
halation_results=halation_spec['production_cases']
diffusion_manifest=(Path('tests/diffusion/frame-fixtures')/'manifest.json').resolve()
diffusion_spec=json.loads(diffusion_manifest.read_text())
diffusion_payload=diffusion_manifest.parent/'native.json'
diffusion_context=diffusion_manifest.parent/'context.json'
diffusion_checked=diffusion_manifest.parent/'checked.json'
diffusion_inputs=diffusion_manifest.parent/'inputs.json'
diffusion_edges_inputs=diffusion_manifest.parent/'edge-inputs.json'
consumed.update({str(path):checksum(path) for path in (diffusion_manifest,diffusion_payload,diffusion_context,diffusion_checked,diffusion_inputs,diffusion_edges_inputs)})
diffusion_results=diffusion_spec['production_cases']
dir_manifest=(Path('tests/dir/frame-fixtures')/'manifest.json').resolve()
dir_spec=json.loads(dir_manifest.read_text())
dir_payload=dir_manifest.parent/'native.json'
dir_checked=dir_manifest.parent/'checked.json'
dir_inputs=dir_manifest.parent/'inputs.json'
consumed.update({str(path):checksum(path) for path in (dir_manifest,dir_payload,dir_checked,dir_inputs)})
dir_results=dir_spec['production_cases']
grain_manifest=(Path('tests/grain/frame-fixtures')/'manifest.json').resolve()
grain_spec=json.loads(grain_manifest.read_text())
grain_payload=grain_manifest.parent/'native.json'
grain_checked=grain_manifest.parent/'checked.json'
grain_inputs=grain_manifest.parent/'inputs.json'
consumed.update({str(path):checksum(path) for path in (grain_manifest,grain_payload,grain_checked,grain_inputs)})
grain_results=grain_spec['production_cases']
boundary_manifest=(Path('tests/dir/boundary-fixtures')/'manifest.json').resolve()
boundary_spec=json.loads(boundary_manifest.read_text())
boundary_payload=boundary_manifest.parent/'native.json'
boundary_checked=boundary_manifest.parent/'checked.json'
boundary_inputs=boundary_manifest.parent/'inputs.json'
consumed.update({str(path):checksum(path) for path in (boundary_manifest,boundary_payload,boundary_checked,boundary_inputs)})
boundary_results=boundary_spec['production_cases']

reference_results=[]
for row in json.loads(reference_payload.read_text())['products']:
    for case in range(8 if row['method']==1 else 7):
        envelope=([0,0,0,0,0,0,0] if case in (0,5) else [2,0,0,1,0,0,0] if case in (1,4) else
                  [3,0,0,2,1,0,0] if case==2 else [3,0,0,2,2,1,0] if case==3 else
                  [3,0,0,3,0,7,0] if case==6 else [3,0,0,4,1,8,1])
        reference_results.append({'case':f"reference-{row['method']}-{case}",'envelope':envelope,
                                  'hash':row['reference']['hash'] if case in (0,5) else 0,'fields':'13-pass'})
gaussian_manifest=(Path('tests/optics/gaussian-fixtures')/'manifest.json').resolve()
gaussian_spec=json.loads(gaussian_manifest.read_text())
gaussian_logical=gaussian_spec['presets'][preset]['native']
gaussian_payload=gaussian_manifest.parent/gaussian_spec['payloads'][gaussian_logical]
gaussian_context=gaussian_manifest.parent/gaussian_spec['contexts'][gaussian_logical]
consumed.update({str(path):checksum(path) for path in (gaussian_manifest,gaussian_payload,gaussian_context)})
gaussian_results=[{'case':row['name'],'envelope':[0,0,0,0],'radius':row['radius'],'sigma_bits':row['sigma_bits'],'weights':row['weights']}
                  for row in json.loads(gaussian_payload.read_bytes())['cases']]
for name in gaussian_spec['production_cases'][188:]:
    reason=2 if name=='radius-upper-reject' else 0 if name in {'zero-capacity','null-release'} else 1
    gaussian_results.append({'case':name,'envelope':[3 if reason==2 else 2 if reason else 0,0,0,reason],'radius':0,'sigma_bits':0,'weights':[]})
probes=[]
for name in ['film','accepted-output-color']:
    source=out/(name+'.c')
    result_lines = ''.join('puts('+json.dumps('DIFFUSION_FRAME_RESULT '+json.dumps(row))+');' for row in diffusion_results) if name=='film' else ''
    if name=='film':result_lines += ''.join('puts('+json.dumps('GRAIN_FRAME_RESULT '+json.dumps(row))+');' for row in grain_results)
    if name=='film':result_lines += ''.join('puts('+json.dumps('DIR_FRAME_RESULT '+json.dumps(row))+');' for row in dir_results)
    if name=='film':result_lines += ''.join('puts('+json.dumps('DIR_BOUNDARY_RESULT '+json.dumps(row))+');' for row in boundary_results)
    source.write_text('#include "production-film-expected.h"\n#include <stdio.h>\n'
                      'typedef struct Pair {int first; int second;} Pair;\n'
                      '_Static_assert(sizeof(Pair)==8,"ABI");\n'
                      'int main(void){printf("value=%d abi=%zu\\n",EXPECTED,sizeof(Pair));'+result_lines+'return EXPECTED!=10;}\n')
    executable=out/(name+('.exe' if sys.platform=='win32' else ''))
    if sys.platform=='win32':
        command=[shutil.which('clang-cl.exe'),'/nologo','/TC','/std:c11','/MD','/W4','/WX',
                 '/clang:-fuse-ld=lld',str(source),'/Fe:'+str(executable),'/Fo:'+str(out/(name+'.obj'))]
    else:
        command=['gcc-13','-std=c11','-Wall','-Wextra','-Werror',str(source),'-o',str(executable)]
    source_hash=checksum(source)
    execute(name+'-compile',command)
    linked=checksum(executable)
    before=checksum(executable)
    execute(name+'-values',[str(executable)])
    after=checksum(executable)
    assert linked==before==after and checksum(source)==source_hash
    probes.append({'name':name,'executable':str(executable),'linked_sha256':linked,
                   'before_run_sha256':before,'after_run_sha256':after,
                   'source':str(source),'source_sha256':source_hash})
for name,value in consumed.items():assert checksum(Path(name))==value
for name,row in build_inputs.items():assert checksum(Path(name))==checksum(Path(row['file']))==row['sha256']
record=out/'commands.json'
record.write_text(json.dumps(commands))
(out/'production.json').write_text(json.dumps({'scope':'native miniature retained-payload protocol only',
    'preset':preset,'build_testing':'OFF','build_inputs':build_inputs,'probes':probes,'consumed_inputs':consumed,
    'reference_fixture':{'manifest':str(reference_manifest),'payload':str(reference_payload),'logical_preset':preset},
    'scanner_descriptor_fixture':{'manifest':str(descriptor_manifest),'payload':str(descriptor_payload),'context':str(descriptor_context),'logical_preset':preset},
    'scanner_descriptor_results':descriptor_results,
    'scanner_post_fixture':{'manifest':str(post_manifest),'payload':str(post_payload),'context':str(post_context),'logical_preset':preset},
    'scanner_post_results':post_results,
    'gaussian_fixture':{'manifest':str(gaussian_manifest),'payload':str(gaussian_payload),'context':str(gaussian_context),'logical_preset':preset},
    'frame_fixture':{'manifest':str(frame_manifest),'payload':str(frame_payload),'context':str(frame_context),'logical_preset':preset},
    'frame_results':frame_results,
    'halation_frame_fixture':{'manifest':str(halation_manifest),'payload':str(halation_payload),'context':str(halation_context),
                              'normative':str(halation_checked),'input_spec':str(halation_inputs),'logical_preset':preset},
    'halation_frame_results':halation_results,
    'diffusion_frame_fixture':{'edges_payload':str(diffusion_payload),'edges_context':str(diffusion_context),'edges_input_spec':str(diffusion_edges_inputs),'manifest':str(diffusion_manifest),'payload':str(diffusion_payload),'context':str(diffusion_context),
                              'normative':str(diffusion_checked),'input_spec':str(diffusion_inputs),'logical_preset':preset},
    'diffusion_frame_results':diffusion_results,
    'dir_frame_fixture':{'manifest':str(dir_manifest),'payload':str(dir_payload),
                         'normative':str(dir_checked),'input_spec':str(dir_inputs),'logical_preset':preset},
    'dir_frame_results':dir_results,
    'grain_frame_fixture':{'manifest':str(grain_manifest),'payload':str(grain_payload),
                          'normative':str(grain_checked),'input_spec':str(grain_inputs),'logical_preset':preset},
    'grain_frame_results':grain_results,
    'dir_boundary_fixture':{'manifest':str(boundary_manifest),'payload':str(boundary_payload),
                            'normative':str(boundary_checked),'input_spec':str(boundary_inputs),'logical_preset':preset},
    'dir_boundary_results':boundary_results,
    'gaussian_results':gaussian_results,
    'reference_results':reference_results,
    'commands':{'file':str(record),'sha256':checksum(record)},
    'command_outputs':{str(out/(row['name']+'.'+extension)):row[extension+'_sha256']
                       for row in commands for extension in ['stdout','stderr']}}))
print("PRODUCTION_MANIFEST_SHA256",checksum(out/"production.json"),flush=True)
'''

    def production_protected(self, stage):
        receipt = Path(stage["receipt"])
        command = json.loads(receipt.read_text())
        manifest = qualification.production_manifest_identity(receipt, root=self.root)[0]
        paths = [receipt, self.root / command["output"], Path(command["process_tree"]["receipt"]), manifest]
        paths += [self.root / name for name in stage["identity"]["inputs"]
                  if Path(name).suffix.lower() in {".a", ".lib", ".ofx", ".pdb"}]
        return {str(path): digest(path) for path in paths}

    def test_retained_grain_helpers_resolve_foreign_paths(self):
        """Exercise shared-input admission before the independent reference gate."""
        preset = "windows-clang-release" if os.name == "nt" else "linux-release"
        build = self.root / "out/build" / preset
        out = self.directory / "retained"
        (out / "build").mkdir(parents=True)
        build.mkdir(parents=True)
        inputs, copies, probes = {}, {}, []
        for name, content in (("CMakeCache.txt", b"BUILD_TESTING:BOOL=OFF\n"), ("libjuicer.a", b"archive")):
            source, retained = build / name, out / "build" / name
            source.write_bytes(content)
            retained.write_bytes(content)
            checksum = digest(source)
            inputs[source.relative_to(self.root).as_posix()] = checksum
            copies[str(source)] = {"file": str(retained), "sha256": checksum}
        for name in ("film", "accepted-output-color"):
            source, executable = out / (name + ".c"), out / (name + (".exe" if os.name == "nt" else ""))
            source.write_text("/* retained source identity */\n")
            executable.write_bytes(b"retained executable identity")
            probes.append({"name": name, "source": str(source), "source_sha256": digest(source),
                           "executable": str(executable), "linked_sha256": digest(executable),
                           "before_run_sha256": digest(executable), "after_run_sha256": digest(executable)})
        helpers = [self.root / "tests/ffi" / name for name in
                   ("grain_frame_abi_assertions.inc", "grain_probe_contract.py")]
        for helper in helpers:
            helper.parent.mkdir(parents=True, exist_ok=True)
            helper.write_bytes((ROOT / helper.relative_to(self.root)).read_bytes())
            inputs[helper.relative_to(self.root).as_posix()] = digest(helper)
        reference = self.root / "tests/film/reference-fixtures/manifest.json"
        reference.parent.mkdir(parents=True)
        write_json(reference, {"presets": {preset: "native.json"}})
        manifest = out / "production.json"

        for prefix, separator in (("Z:\\Film-Juicer\\", "\\"), ("/foreign/Film-Juicer/", "/")):
            def foreign(path):
                return prefix + path.relative_to(self.root).as_posix().replace("/", separator)
            def resolve(name):
                if name.startswith(prefix):
                    return self.root / name[len(prefix):].replace(separator, "/")
                return Path(name)
            document = {"preset": preset, "build_testing": "OFF", "build_inputs": copies,
                        "probes": probes, "consumed_inputs": {foreign(p): digest(p) for p in helpers}}
            write_json(manifest, document)
            # This intentionally incomplete tiny record must reach the next gate;
            # it does not claim real archive execution or full payload admission.
            with self.assertRaisesRegex(ValueError, "film-reference fixture membership"):
                qualification.production_payloads(manifest, digest(manifest), root=self.root, inputs=inputs, resolve=resolve)
            for helper in helpers:
                for mode in ("missing", "hash", "content"):
                    with self.subTest(prefix=prefix, helper=helper.name, mode=mode):
                        damaged = json.loads(json.dumps(document))
                        original = helper.read_bytes()
                        try:
                            if mode == "missing":
                                del damaged["consumed_inputs"][foreign(helper)]
                            elif mode == "hash":
                                damaged["consumed_inputs"][foreign(helper)] = "0" * 64
                            else:
                                helper.write_bytes(original + b"\n")
                            write_json(manifest, damaged)
                            with self.assertRaisesRegex(ValueError, "grain C11 assertion input identity changed"):
                                qualification.production_payloads(manifest, digest(manifest), root=self.root, inputs=inputs, resolve=resolve)
                        finally:
                            helper.write_bytes(original)

    def production_mutations(self, stage):
        manifest = qualification.production_manifest_identity(stage["receipt"], root=self.root)[0]
        document = json.loads(manifest.read_text())
        artifact = next(Path(row["file"]) for name, row in document["build_inputs"].items()
                        if Path(name).suffix.lower() in {".a", ".lib", ".ofx"})
        return [Path(row[key]) for row in document["probes"] for key in ("executable", "source")] + [
            manifest.parent / "production-film-expected.h", manifest.parent / "film-values.stdout", artifact]

    def assert_production_reuse_controls(self, path, checkpoint, calls):
        record = json.loads(path.read_text())
        stage = record["stages"]["production"]
        owner = self.root / "out/build" / record["preset"] / "verification-checkpoint.json"
        owner_bytes = owner.read_bytes()
        protected = self.production_protected(stage)
        before = len(calls)
        reused = json.loads(checkpoint("retained-unchanged").read_text())["stages"]["production"]
        self.assertEqual(reused["reused_from"], stage["receipt"])
        self.assertEqual(len(calls), before, "unchanged retained evidence executed probes again")
        outcomes = [{"operation": "reuse", "case": "unchanged", "status": "passed", "commands": 0}]
        for target in self.production_mutations(stage):
            data, stamp = target.read_bytes(), target.stat()
            for mutation in ("replace", "remove"):
                with self.subTest(operation="reuse", file=target.name, mutation=mutation):
                    try:
                        target.write_bytes(b"changed after execution") if mutation == "replace" else target.unlink()
                        with self.assertRaisesRegex(ValueError, "production execution payload missing or changed"):
                            checkpoint("retained-" + target.name.replace(".", "-") + "-" + mutation)
                        self.assertEqual(json.loads(owner.read_text())["production"]["status"], "failed")
                        self.assertEqual(len(calls), before, "rejected reuse executed probes")
                        self.assertEqual(protected, {name: digest(Path(name)) for name in protected})
                        outcomes.append({"operation": "reuse", "case": target.name, "mutation": mutation,
                                         "rejected": "production execution payload missing or changed"})
                    finally:
                        target.write_bytes(data)
                        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                        owner.write_bytes(owner_bytes)
        try:
            state = json.loads(owner_bytes)
            header = next(name for name in stage["payloads"] if Path(name).name == "production-film-expected.h")
            state["production"]["payloads"].pop(header)
            write_json(owner, state)
            with self.assertRaisesRegex(ValueError, "production retained payload membership"):
                checkpoint("retained-omitted-member")
            self.assertEqual(len(calls), before)
            outcomes.append({"operation": "reuse", "case": "omitted-member", "rejected": "production retained payload membership"})
        finally:
            owner.write_bytes(owner_bytes)
        for target_kind, cause in [("probe", "production execution payload missing or changed"),
                                   ("manifest", "production manifest execution identity changed")]:
            prior = Campaign.run
            def drift(campaign, name, command, **kwargs):
                receipt = prior(campaign, name, command, **kwargs)
                if name == "production":
                    raw = json.loads(receipt.read_text())
                    self.assertTrue(raw["process_tree"]["drained"])
                    manifest = qualification.production_manifest_identity(receipt, root=self.root)[0]
                    document = json.loads(manifest.read_text())
                    if target_kind == "probe":
                        Path(document["probes"][0]["executable"]).write_bytes(b"drift after drained execution")
                    else:
                        document["scope"] = "manifest replaced after drained execution"
                        manifest.write_text(json.dumps(document))
                return receipt
            try:
                with patch.object(Campaign, "run", new=drift):
                    with self.assertRaisesRegex(ValueError, cause):
                        checkpoint("retained-registration-drift-" + target_kind,
                                   "intentional post-execution retained-payload drift control")
                self.assertEqual(json.loads(owner.read_text())["production"]["status"], "failed")
                self.assertEqual(protected, {name: digest(Path(name)) for name in protected})
                outcomes.append({"operation": "registration", "case": "post-drain-drift-" + target_kind,
                                 "rejected": cause})
            finally:
                owner.write_bytes(owner_bytes)
        return outcomes

    def assert_production_collection_controls(self, paths):
        preset = "windows-clang-release" if os.name == "nt" else "linux-release"
        path = next(path for path in paths if json.loads(path.read_text())["preset"] == preset
                    and json.loads(path.read_text())["selection"] == ["production"])
        record = json.loads(path.read_text())
        stage = record["stages"]["production"]
        owner = self.root / "out/build" / preset / "verification-checkpoint.json"
        owner_bytes = owner.read_bytes()
        protected = self.production_protected(stage)
        protected[str(path)] = digest(path)
        outcomes = [{"operation": "collection", "case": "unchanged", "status": "passed"}]
        for target in self.production_mutations(stage):
            data, stamp = target.read_bytes(), target.stat()
            for mutation in ("replace", "remove"):
                with self.subTest(operation="collection", file=target.name, mutation=mutation):
                    try:
                        target.write_bytes(b"changed after execution") if mutation == "replace" else target.unlink()
                        with self.assertRaisesRegex(ValueError, "production execution payload missing or changed"):
                            qualification.collect_qualification(self.root, paths)
                        self.assertEqual(protected, {name: digest(Path(name)) for name in protected})
                        outcomes.append({"operation": "collection", "case": target.name, "mutation": mutation,
                                         "rejected": "production execution payload missing or changed"})
                    finally:
                        target.write_bytes(data)
                        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        try:
            header = next(name for name in stage["payloads"] if Path(name).name == "production-film-expected.h")
            state = json.loads(owner_bytes)
            state["production"]["payloads"].pop(header)
            record["stages"]["production"]["payloads"].pop(header)
            write_json(owner, state)
            omitted = path.parent / "omitted-membership.json"
            write_json(omitted, record)
            with self.assertRaisesRegex(ValueError, "production retained payload membership"):
                qualification.collect_qualification(self.root, [omitted if other == path else other for other in paths])
            self.assertEqual(protected, {name: digest(Path(name)) for name in protected})
            outcomes.append({"operation": "collection", "case": "omitted-member",
                             "rejected": "production retained payload membership"})
        finally:
            owner.write_bytes(owner_bytes)
        self.assertEqual(qualification.collect_qualification(self.root, paths)["status"], "passed")
        return outcomes

    def test_collection_rejects_prepare_without_scoped_proof(self):
        self.native_project()
        from verify import source_snapshot
        required = ["prepare", "quality", "product"]
        campaign = Campaign(self.root, self.preset, "unproved-preparation", required)
        stages = {name: qualification.run_stage(campaign, self.state_path, self.state, name,
            [self.python, "-c", "print('ordinary command without preparation proof')"],
            inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10)
            for name in required}
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed",
                  "reason": "ordinary command cannot supply scoped preparation", "preset": self.preset,
                  "runtime": "all", "selection": ["product"], "required": required,
                  "stages": stages, "source": source_snapshot(self.root)}
        path = campaign.directory / "qualification.json"
        write_json(path, record)
        protected = {p: p.read_bytes() for p in
                     [path, self.state_path, *(Path(stage["receipt"]) for stage in stages.values())]}
        with patch.object(Campaign, "run", side_effect=AssertionError("collection dispatched")):
            with self.assertRaisesRegex(ValueError, "legacy sealed-build preparation"):
                qualification.collect_qualification(self.root, [path])
        self.assertEqual(protected, {p: p.read_bytes() for p in protected})

    def test_production_retained_execution_payloads(self):
        """Real OFF owners and preparations; unrelated execution uses unit scaffolds."""
        started = time.monotonic()
        files = self.compiled_project()
        from verify import source_snapshot
        paths, outcomes, commands = [], [], []
        actual_run = Campaign.run
        def counted(campaign, stage, command, **kwargs):
            commands.append({"stage": stage, "command": list(command)})
            return actual_run(campaign, stage, command, **kwargs)
        def checkpoint(preset, name, repeat_reason=""):
            args = SimpleNamespace(reason="native retained OFF payload controls", selection=["production"],
                                   preset=preset, runtime="all", campaign=name,
                                   repeat_reason=repeat_reason, base=None)
            if preset == ("windows-clang-release" if os.name == "nt" else "linux-release"):
                return qualification.qualify(self.root, args)
            with OutputLease(self.root, self.root / "out/build" / preset):
                return qualification._qualify_owned(self.root, self.root / "out/build" / preset, args)
        with patch.object(Campaign, "run", autospec=True, side_effect=counted):
            for preset in ("linux-release", "windows-clang-release"):
                build = self.root / "out/build" / preset
                campaign = Campaign(self.root, preset, "retained-off-prepare", ["prepare"])
                qualification.prepare_build(campaign, "prepare", source=self.root, build=build,
                    inputs=list(files), timeout=60, targets=["all"],
                    configure_args=["-DBUILD_TESTING=OFF"], checkpoint=True)
                path = checkpoint(preset, "retained-off")
                paths.append(path)
                if preset == ("windows-clang-release" if os.name == "nt" else "linux-release"):
                    outcomes += self.assert_production_reuse_controls(path,
                        lambda name, repeat_reason="": checkpoint(preset, name, repeat_reason), commands)
            # Real miniature preparations supply the collector's context and
            # artifact identity. Only unrelated relevance execution is scaffolded;
            # no preparation or production validation is mocked.
            for preset in qualification.PRESETS:
                build = self.root / "out/build" / preset
                self.configure_fixture(preset, True, "retained-prerequisite-on", timeout=60)
                state_path = build / "verification-checkpoint.json"
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                groups = ["boundary", "product"]
                if preset.endswith("debug"):
                    groups.append("tooling")
                if preset == "linux-debug":
                    groups.append("corpus")
                required = ["prepare", "quality", *groups]
                campaign = Campaign(self.root, preset, "retained-unrelated-scaffold", required)
                stages = {"prepare": qualification.prepare_checkpoint(campaign, build, state_path, state,
                    self.environment, timeout=60)}
                dependency = qualification.preparation_identity(build, stages["prepare"])
                for name in required[1:]:
                    stages[name] = qualification.run_stage(campaign, state_path, state, name,
                        [self.python, "-c", "print('unrelated collection prerequisite scaffold')"],
                        inputs=qualification.checkpoint_inputs(self.root, build, name if name in qualification.GROUPS else "product"),
                        tools=[self.python], environment=self.environment, timeout=10,
                        dependencies=[dependency] if name in {"quality", "product"} else [])
                record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed",
                          "reason": "synthetic unrelated prerequisites for isolated production collector controls",
                          "preset": preset, "runtime": "all", "selection": groups, "required": required,
                          "stages": stages, "source": source_snapshot(self.root)}
                path = campaign.directory / "qualification.json"
                write_json(path, record)
                paths.append(path)
            self.assertEqual(qualification.collect_qualification(self.root, paths)["status"], "passed")
            outcomes += self.assert_production_collection_controls(paths)
        destination = self.root / "out/validation" / self.preset / "retained-production-controls.json"
        write_json(destination, {"scope": "real production checkpoint/reuse/collector controls with explicit unrelated product scaffolds; no product qualification",
            "native": os.name, "controls": outcomes, "commands": commands,
            "seconds": time.monotonic() - started})

    def test_production_evidence_survives_on_restoration_and_unrelated_failure(self):
        files = self.compiled_project()
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
                         "target_compile_definitions(juicer PRIVATE MINI_TESTING=$<BOOL:${BUILD_TESTING}>)\n")
        native = self.root / "src/native.cpp"
        native.write_text(native.read_text() + "int testing_feature(){return MINI_TESTING;}\n")
        preset = "windows-clang-release" if os.name == "nt" else "linux-release"
        build = self.root / "out/build" / preset
        state_path = build / "verification-checkpoint.json"
        campaign = Campaign(self.root, preset, "off-preparation", ["prepare"])
        qualification.prepare_build(campaign, "prepare", source=self.root, build=build,
            inputs=list(files), timeout=60, targets=["all"], configure_args=["-DBUILD_TESTING=OFF"], checkpoint=True)
        state = {}
        with OutputLease(self.root, build):
            with self.assertRaises(subprocess.CalledProcessError):
                qualification.run_stage(Campaign(self.root, preset, "unrelated-failure", ["boundary"]),
                    state_path, state, "boundary", [self.python, "-c", "raise SystemExit(7)"],
                    inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10)
        args = SimpleNamespace(reason="retained OFF evidence across ON restoration", selection=["production"],
                               preset=preset, runtime="all", campaign="original-off", repeat_reason="", base=None)
        path = qualification.qualify(self.root, args)
        original_record = path.read_bytes()
        record = json.loads(original_record)
        self.assertEqual(record["status"], "incomplete")
        self.assertEqual(record["scope_status"], "passed")
        self.assertEqual(record["pending"], ["boundary"])
        stage = record["stages"]["production"]
        manifest = qualification.production_manifest_identity(stage["receipt"], root=self.root)[0]
        document = json.loads(manifest.read_text())
        self.assertEqual(len(document["reference_results"]), 15)
        self.assertEqual(len(document["scanner_post_results"]), 410)
        self.assertEqual(len(document["gaussian_results"]), 201)
        self.assertEqual(len(document["frame_results"]), 69)
        self.assertEqual(len(document["halation_frame_results"]), 86)
        self.assertEqual(len(document["diffusion_frame_results"]), 507)
        self.assertEqual(len(document["dir_frame_results"]), 121)
        self.assertEqual(len(document["grain_frame_results"]), 188)
        self.assertEqual(len(document["dir_boundary_results"]), 171)
        stdout = manifest.parent / "film-values.stdout"
        commands_path = manifest.parent / "commands.json"
        original_stdout, original_commands = stdout.read_bytes(), commands_path.read_bytes()
        try:
            line = next(line for line in original_stdout.decode().splitlines() if line.startswith("DIR_FRAME_RESULT "))
            original_row = json.loads(line.removeprefix("DIR_FRAME_RESULT "))
            changed_row = json.loads(json.dumps(original_row))
            changed_row["failure"][-1] ^= 1
            changed_stdout = original_stdout.replace(json.dumps(original_row).encode(), json.dumps(changed_row).encode(), 1)
            self.assertNotEqual(changed_stdout, original_stdout)
            stdout.write_bytes(changed_stdout)
            commands = json.loads(original_commands)
            next(row for row in commands if row["name"] == "film-values")["stdout_sha256"] = digest(stdout)
            write_json(commands_path, commands)
            damaged = json.loads(json.dumps(document))
            damaged["commands"]["sha256"] = digest(commands_path)
            damaged["command_outputs"][str(stdout)] = digest(stdout)
            damaged_path = manifest.with_name("dir-executed-trace.json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "DIR frame executed stdout"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        finally:
            stdout.write_bytes(original_stdout)
            commands_path.write_bytes(original_commands)

        try:
            line = next(line for line in original_stdout.decode().splitlines() if line.startswith("DIR_BOUNDARY_RESULT "))
            original_row = json.loads(line.removeprefix("DIR_BOUNDARY_RESULT "))
            changed_row = json.loads(json.dumps(original_row))
            changed_row["weights"][-1] ^= 1
            changed_stdout = original_stdout.replace(json.dumps(original_row).encode(), json.dumps(changed_row).encode(), 1)
            self.assertNotEqual(changed_stdout, original_stdout)
            stdout.write_bytes(changed_stdout)
            commands = json.loads(original_commands)
            next(row for row in commands if row["name"] == "film-values")["stdout_sha256"] = digest(stdout)
            write_json(commands_path, commands)
            damaged = json.loads(json.dumps(document))
            damaged["commands"]["sha256"] = digest(commands_path)
            damaged["command_outputs"][str(stdout)] = digest(stdout)
            damaged_path = manifest.with_name("dir-boundary-executed-word.json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "DIR boundary executed stdout"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        finally:
            stdout.write_bytes(original_stdout)
            commands_path.write_bytes(original_commands)

        for mode in ("missing", "duplicate", "word", "key", "envelope", "failure", "terminal", "shape", "diagnostic_length", "owner"):
            damaged = json.loads(json.dumps(document))
            rows = damaged["dir_boundary_results"]
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = rows[0]
            elif mode == "word":
                rows[0]["weights"][-1] ^= 1
            elif mode == "key":
                rows[0]["key"] ^= 1
            elif mode == "envelope":
                rows[0]["envelope"][0] = 3
            elif mode == "failure":
                rows[0]["failure"][2] = 0
            elif mode == "terminal":
                rows[0]["terminal"][-1] ^= 1
            elif mode == "shape":
                rows[0]["shape"][0] ^= 1
            elif mode == "diagnostic_length":
                rows[0]["diagnostic_length"] += 1
            else:
                rows[0]["owner"] = False
            damaged_path = manifest.with_name("dir-boundary-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "DIR boundary"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])

        for mode in ("missing", "duplicate", "case", "word", "envelope", "failure", "diagnostic", "length"):
            damaged = json.loads(json.dumps(document))
            rows = damaged["grain_frame_results"]
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = rows[0]
            elif mode == "case":
                rows[0]["case"] = "unknown-grain-case"
            elif mode == "word":
                rows[0]["output"][-1] ^= 1
            elif mode == "envelope":
                rows[0]["envelope"][0] = 3
            elif mode == "failure":
                rows[0]["failure"][2] = 0
            elif mode == "diagnostic":
                rows[0]["diagnostic"] = "wrong"
            else:
                rows[0]["diagnostic_length"] += 1
            damaged_path = manifest.with_name("grain-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "grain frame"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        for name in ("grain_frame_abi_assertions.inc", "grain_probe_contract.py"):
            target = self.root / "tests/ffi" / name
            original = target.read_bytes()
            try:
                target.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "grain C11 assertion input identity changed"):
                    qualification.production_payloads(manifest, digest(manifest), root=self.root, inputs=stage["identity"]["inputs"])
            finally:
                target.write_bytes(original)

        # Even an aligned, rehashed expected/result substitution cannot redefine
        # the independently reviewed complete case set.
        target = Path(document["grain_frame_fixture"]["manifest"])
        original = target.read_bytes()
        try:
            spec = json.loads(original)
            spec["production_cases"][0]["case"] = "unknown-grain-case"
            write_json(target, spec)
            damaged = json.loads(json.dumps(document))
            damaged["grain_frame_results"][0]["case"] = "unknown-grain-case"
            damaged["consumed_inputs"][str(target)] = digest(target)
            inputs = dict(stage["identity"]["inputs"])
            inputs[target.relative_to(self.root).as_posix()] = digest(target)
            damaged_path = manifest.with_name("grain-aligned-case-substitution.json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "grain frame full values"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=inputs)
        finally:
            target.write_bytes(original)
        for member in ("manifest", "payload", "normative", "input_spec"):
            target = Path(document["grain_frame_fixture"][member])
            original = target.read_bytes()
            try:
                target.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "grain frame consumed fixture"):
                    qualification.production_payloads(manifest, digest(manifest), root=self.root, inputs=stage["identity"]["inputs"])
            finally:
                target.write_bytes(original)
        try:
            line = next(line for line in original_stdout.decode().splitlines() if line.startswith("GRAIN_FRAME_RESULT "))
            original_row = json.loads(line.removeprefix("GRAIN_FRAME_RESULT "))
            changed_row = json.loads(json.dumps(original_row))
            changed_row["output"][-1] ^= 1
            changed_stdout = original_stdout.replace(json.dumps(original_row).encode(), json.dumps(changed_row).encode(), 1)
            self.assertNotEqual(changed_stdout, original_stdout)
            stdout.write_bytes(changed_stdout)
            commands = json.loads(original_commands)
            next(row for row in commands if row["name"] == "film-values")["stdout_sha256"] = digest(stdout)
            write_json(commands_path, commands)
            damaged = json.loads(json.dumps(document))
            damaged["commands"]["sha256"] = digest(commands_path)
            damaged["command_outputs"][str(stdout)] = digest(stdout)
            damaged_path = manifest.with_name("grain-executed-values.json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "grain frame executed stdout"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        finally:
            stdout.write_bytes(original_stdout)
            commands_path.write_bytes(original_commands)

        for mode in ("missing", "duplicate", "word", "envelope", "failure", "trace", "diagnostic"):
            damaged = json.loads(json.dumps(document))
            rows = damaged["dir_frame_results"]
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = rows[0]
            elif mode == "word":
                rows[0]["output"][-1] ^= 1
            elif mode == "envelope":
                rows[0]["envelope"][0] = 3
            elif mode == "failure":
                rows[0]["failure"][2] = 1
            elif mode == "trace":
                rows[0]["failure"][-1] ^= 1
            else:
                rows[0]["diagnostic"] = "wrong"
            damaged_path = manifest.with_name("dir-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "DIR frame"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])

        try:
            line = next(line for line in original_stdout.decode().splitlines() if line.startswith("DIFFUSION_FRAME_RESULT "))
            original_row = json.loads(line.removeprefix("DIFFUSION_FRAME_RESULT "))
            changed_row = json.loads(json.dumps(original_row))
            changed_row["output"][-1] ^= 1
            changed_stdout = original_stdout.replace(json.dumps(original_row).encode(), json.dumps(changed_row).encode(), 1)
            self.assertNotEqual(changed_stdout, original_stdout)
            stdout.write_bytes(changed_stdout)
            commands = json.loads(original_commands)
            next(row for row in commands if row["name"] == "film-values")["stdout_sha256"] = digest(stdout)
            write_json(commands_path, commands)
            damaged = json.loads(json.dumps(document))
            damaged["commands"]["sha256"] = digest(commands_path)
            damaged["command_outputs"][str(stdout)] = digest(stdout)
            damaged_path = manifest.with_name("diffusion-executed-word.json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "diffusion frame executed stdout"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        finally:
            stdout.write_bytes(original_stdout)
            commands_path.write_bytes(original_commands)

        for mode in ("missing", "duplicate", "word", "envelope", "failure", "diagnostic"):
            damaged = json.loads(json.dumps(document))
            rows = damaged["diffusion_frame_results"]
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = rows[0]
            elif mode == "word":
                rows[0]["output"][-1] ^= 1
            elif mode == "envelope":
                rows[0]["envelope"][0] = 3
            elif mode == "failure":
                rows[0]["failure"][2] = 1
            else:
                rows[0]["diagnostic"] = "wrong"
            damaged_path = manifest.with_name("diffusion-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "diffusion frame"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])

        for mode in ("missing", "duplicate", "word", "hash", "envelope", "failure", "sigma", "diagnostic"):
            damaged = json.loads(json.dumps(document))
            rows = damaged["halation_frame_results"]
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = rows[0]
            elif mode == "word":
                rows[0]["output"]["words"][-1] ^= 1
            elif mode == "hash":
                rows[0]["output"]["recipe_hash"] ^= 1
            elif mode == "envelope":
                rows[0]["envelope"][0] = 3
            elif mode == "failure":
                rows[0]["failure"][3] = 0
            elif mode == "sigma":
                rows[0]["sigma_bits"] ^= 1
            else:
                rows[0]["diagnostic"] = "wrong"
            damaged_path = manifest.with_name("halation-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "halation frame"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        for mode in ("missing", "duplicate", "changed-domain", "changed-pixel", "changed-diffusion", "changed-envelope", "changed-diagnostic"):
            damaged = json.loads(json.dumps(document))
            if mode == "missing":
                damaged["frame_results"].pop()
            elif mode == "duplicate":
                damaged["frame_results"][-1] = damaged["frame_results"][0]
            elif mode == "changed-domain":
                damaged["frame_results"][0]["domain"][3] ^= 1
            elif mode == "changed-pixel":
                damaged["frame_results"][0]["pixel_bits"] ^= 1
            elif mode == "changed-diffusion":
                damaged["frame_results"][0]["diffusion_bits"] ^= 1
            elif mode == "changed-envelope":
                damaged["frame_results"][0]["envelope"][0] = 3
            else:
                damaged["frame_results"][0]["diagnostic"] = "wrong"
            damaged_path = manifest.with_name("frame-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "frame geometry"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])

        for mode in ("missing", "duplicate", "changed-weight", "changed-metadata", "changed-envelope"):
            damaged = json.loads(json.dumps(document))
            if mode == "missing":
                damaged["gaussian_results"].pop()
            elif mode == "duplicate":
                damaged["gaussian_results"][-1] = damaged["gaussian_results"][0]
            elif mode == "changed-weight":
                damaged["gaussian_results"][0]["weights"][1] ^= 1
            elif mode == "changed-metadata":
                damaged["gaussian_results"][0]["sigma_bits"] ^= 1
            else:
                damaged["gaussian_results"][-1]["envelope"][0] = 0
            damaged_path = manifest.with_name("gaussian-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "Gaussian full arrays/metadata/failure"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])

        for mode in ("missing", "duplicate", "changed"):
            damaged = json.loads(json.dumps(document))
            if mode == "missing":
                damaged["scanner_post_results"].pop()
            elif mode == "duplicate":
                damaged["scanner_post_results"][-1] = damaged["scanner_post_results"][0]
            else:
                damaged["scanner_post_results"][0]["hash"] ^= 1
            damaged_path = manifest.with_name("post-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "post-effects value/failure"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        for field in ("grain_frame_results", "grain_frame_fixture", "dir_boundary_results", "dir_boundary_fixture", "dir_frame_results", "dir_frame_fixture", "diffusion_frame_results", "diffusion_frame_fixture", "halation_frame_results", "halation_frame_fixture", "frame_results", "frame_fixture", "reference_results", "reference_fixture", "scanner_descriptor_results", "scanner_descriptor_fixture", "scanner_post_results", "scanner_post_fixture", "gaussian_results", "gaussian_fixture"):
            missing = manifest.with_name("without-" + field + ".json")
            write_json(missing, {key: value for key, value in document.items() if key != field})
            expected = "grain frame" if field.startswith("grain_frame") else "DIR boundary" if field.startswith("dir_boundary") else "DIR frame" if field.startswith("dir_frame") else "diffusion frame" if field.startswith("diffusion") else "halation frame" if field.startswith("halation") else "frame geometry" if field.startswith("frame") else "Gaussian" if field.startswith("gaussian") else "scanner post-effects" if field.startswith("scanner_post") else "scanner descriptor" if field.startswith("scanner_descriptor") else "film-reference"
            with self.assertRaisesRegex(ValueError, expected):
                qualification.production_payloads(missing, digest(missing), root=self.root,
                    inputs=stage["identity"]["inputs"])
        artifact = build / ("juicer.lib" if os.name == "nt" else "libjuicer.a")
        retained = Path(document["build_inputs"][str(artifact)]["file"])
        off_hash = stage["identity"]["inputs"][artifact.relative_to(self.root).as_posix()]
        self.assertEqual(digest(artifact), off_hash)
        self.assertEqual(digest(retained), off_hash)
        evidence = {name: checksum for name, checksum in self.production_protected(stage).items()
                    if not Path(name).is_relative_to(build)}
        payloads = {name: digest(Path(name)) for name in stage["payloads"]}
        legacy = manifest.with_name("without-retained-build-inputs.json")
        write_json(legacy, {key: value for key, value in document.items() if key != "build_inputs"})
        with self.assertRaisesRegex(ValueError, "fresh OFF execution required"):
            qualification.production_payloads(legacy, digest(legacy), root=self.root,
                inputs=stage["identity"]["inputs"])

        qualification.prepare_build(Campaign(self.root, preset, "restore-on", ["prepare"]), "prepare",
            source=self.root, build=build, inputs=list(files), timeout=60, targets=["all"],
            configure_args=["-DBUILD_TESTING=ON"], checkpoint=True)
        self.assertIn("BUILD_TESTING:BOOL=ON", (build / "CMakeCache.txt").read_text())
        self.assertNotEqual(digest(artifact), off_hash)
        self.assertEqual(payloads, {name: digest(Path(name)) for name in payloads})

        def rejected_collection(message):
            with patch.object(Campaign, "run", side_effect=AssertionError("collection executed a command")):
                with self.assertRaisesRegex(ValueError, message):
                    qualification.collect_qualification(self.root, [path])

        rejected_collection("pending checkpoint work")
        state = json.loads(state_path.read_text())
        with OutputLease(self.root, build):
            qualification.run_stage(Campaign(self.root, preset, "unrelated-repair", ["boundary"]),
                state_path, state, "boundary", [self.python, "-c", "print('repaired')"],
                inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10)
        rejected_collection("full checkpoint membership is incomplete")
        data, stamp = retained.read_bytes(), retained.stat()
        on_hash = digest(artifact)
        for mutation in ("replace", "remove"):
            with self.subTest(retained_artifact=mutation):
                try:
                    retained.write_bytes(b"changed retained OFF archive") if mutation == "replace" else retained.unlink()
                    rejected_collection("production execution payload missing or changed")
                    self.assertEqual(digest(artifact), on_hash)
                finally:
                    retained.write_bytes(data)
                    os.utime(retained, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        rejected_collection("full checkpoint membership is incomplete")
        self.assertEqual(path.read_bytes(), original_record)
        self.assertEqual(evidence, {name: digest(Path(name)) for name in evidence})
        self.assertEqual(payloads, {name: digest(Path(name)) for name in payloads})

        with OutputLease(self.root, build):
            with self.assertRaises(subprocess.CalledProcessError):
                qualification.run_stage(Campaign(self.root, preset, "later-production-failure", ["production"]),
                    state_path, state, "production", [self.python, "-c", "raise SystemExit(9)"],
                    inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10)
        rejected_collection("pending checkpoint work")
        self.assertEqual(json.loads(state_path.read_text())["production"]["status"], "failed")
        self.assertEqual(path.read_bytes(), original_record)
        self.assertEqual(evidence, {name: digest(Path(name)) for name in evidence})

    def compiled_project(self):
        self.native_project()
        files = {
            "CMakeLists.txt": 'cmake_minimum_required(VERSION 3.24)\nproject(Checkpoint LANGUAGES C CXX)\n'
                'set(CMAKE_C_STANDARD 11)\nset(CMAKE_CXX_STANDARD 20)\n'
                'set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\ninclude(CTest)\n'
                'configure_file(binding.h.in generated/binding.h COPYONLY)\n'
                'add_library(juicer STATIC src/native.cpp src/abi.c)\n'
                'target_include_directories(juicer PUBLIC "${CMAKE_BINARY_DIR}/generated")\n'
                'add_executable(behavior src/main.cpp)\ntarget_link_libraries(behavior PRIVATE juicer)\n'
                'if(BUILD_TESTING)\nforeach(group product boundary tooling corpus)\n'
                'add_test(NAME mini.${group} COMMAND behavior 10)\n'
                'set_tests_properties(mini.${group} PROPERTIES LABELS "host;${group}")\nendforeach()\nendif()\n',
            "binding.h.in": "#define BINDING 4\n",
            "src/abi.h": 'typedef struct Witness { int first; int second; } Witness;\n',
            "src/abi.c": '#include <stddef.h>\n#include "abi.h"\n'
                '_Static_assert(sizeof(Witness)==8,"C11 ABI");\n'
                'int c_size(void){return sizeof(Witness);}\nint c_offset(void){return offsetof(Witness,second);}\n',
            "src/native.cpp": '#include "binding.h"\nint native(){return 6+BINDING;}\n',
            "src/main.cpp": '#include <cstdio>\n#include <cstdlib>\n#include <cstddef>\n#include "abi.h"\n'
                'static_assert(sizeof(Witness)==8);\nint native();\nextern "C" int c_size(void);\nextern "C" int c_offset(void);\n'
                'int main(int argc,char** argv){int v=native();const char* offset=std::getenv("JUICER_P0_A_RUNTIME_OFFSET");'
                'if(offset)v+=std::atoi(offset);std::printf("value=%d abi=%d offset=%d\\n",v,c_size(),c_offset());'
                'return argc!=2 || v!=std::atoi(argv[1]) || c_size()!=sizeof(Witness) || c_offset()!=offsetof(Witness,second);}\n',
            "scripts/check-quality.py": 'import sys\nfrom pathlib import Path\n'
                'preset=sys.argv[sys.argv.index("--preset")+1]\np=Path("out")/(preset+"-quality-count.txt")\n'
                'p.write_text(str(int(p.read_text())+1) if p.exists() else "1")\nprint("miniature quality scheduling witness")\n',
            "tests/ffi/production_contract.py": 'import json,sys,subprocess\nfrom pathlib import Path\n'
                'preset=sys.argv[sys.argv.index("--preset")+1]\nout=Path(sys.argv[sys.argv.index("--artifacts")+1])\n'
                'build=Path("out/build")/preset\nexe=build/("behavior.exe" if sys.platform=="win32" else "behavior")\n'
                'value=subprocess.check_output([str(exe.resolve()),"10"],text=True,timeout=10)\n'
                'assert "value=10 abi=8 offset=4" in value\nout.mkdir(parents=True)\n'
                + self.production_fixture(),
        }
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        for name in ("grain_frame_abi_assertions.inc", "grain_probe_contract.py"):
            relative = "tests/ffi/" + name
            content = (ROOT / relative).read_text()
            (self.root / relative).write_text(content)
            files[relative] = content
        compiler = ({"CMAKE_C_COMPILER": "clang-cl.exe", "CMAKE_CXX_COMPILER": "clang-cl.exe", "CMAKE_LINKER_TYPE": "LLD",
                     "CMAKE_C_FLAGS": "/Brepro", "CMAKE_CXX_FLAGS": "/Brepro"}
                    if os.name == "nt" else {"CMAKE_C_COMPILER": "/usr/bin/gcc-13", "CMAKE_CXX_COMPILER": "/usr/bin/g++-13"})
        (self.root / "CMakePresets.json").write_text(json.dumps({"version": 6, "configurePresets": [
            {"name": p, "generator": "Ninja", "binaryDir": str(self.root / "out/build" / p),
             "cacheVariables": {**compiler, "CMAKE_BUILD_TYPE": "Release" if p.endswith("release") else "Debug"}}
            for p in qualification.PRESETS]}))
        return files

    def replay_preparation_renewal(self):
        """Four protocol lanes use this OS's tiny binaries, never product qualification."""
        calls, deletions, artifact_reads, input_checks, tool_checks = [], [], [], [], []
        started = time.monotonic()
        progress = {"scope": "native miniature protocol only", "complete": False, "phases": []}
        progress_path = self.root / "out/validation" / self.preset / "replay-progress.json"
        progress_path.parent.mkdir(parents=True)
        def record_phase(name, action, *args, **kwargs):
            row = {"name": name, "status": "running"}
            progress["phases"].append(row)
            write_json(progress_path, progress)
            start = time.monotonic()
            try:
                result = action(*args, **kwargs)
                row["status"] = "passed"
                return result
            except BaseException as error:
                row.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "raised",
                           error=type(error).__name__)
                raise
            finally:
                row["seconds"] = time.monotonic() - start
                progress.update(seconds=time.monotonic() - started, commands=calls, deleted=deletions,
                                preparation_input_checks=len(input_checks),
                                preparation_input_check_seconds=sum(input_checks),
                                tool_checks=len(tool_checks), tool_check_seconds=sum(tool_checks),
                                artifact_hash_reads=len(artifact_reads),
                                artifact_hash_seconds=sum(value[0] for value in artifact_reads))
                write_json(progress_path, progress)
        files = record_phase("fixture.configure", self.compiled_project)
        artifact_paths = set()
        actual_digest = qualification.digest
        def measured_digest(path):
            path = Path(path).resolve()
            started = time.monotonic()
            result = actual_digest(path)
            if path in artifact_paths:
                artifact_reads.append((time.monotonic() - started, path.stat().st_size))
            return result
        actual_run = Campaign.run
        from build_dependencies import NinjaGraph
        actual_invalidate = NinjaGraph.invalidate
        def counted(campaign, stage, command, **kwargs):
            started = time.monotonic()
            try:
                return actual_run(campaign, stage, command, **kwargs)
            finally:
                calls.append({"stage": stage, "command": list(command), "seconds": time.monotonic() - started})
        def invalidate(graph, names):
            removed = actual_invalidate(graph, names)
            deletions.extend(removed)
            return removed
        actual_check = qualification.check_preparation_inputs
        def measured_inputs(*args, **kwargs):
            started = time.monotonic()
            try:
                return actual_check(*args, **kwargs)
            finally:
                input_checks.append(time.monotonic() - started)
        actual_tools = qualification.tool_snapshot
        def measured_tools(*args, **kwargs):
            started = time.monotonic()
            try:
                return actual_tools(*args, **kwargs)
            finally:
                tool_checks.append(time.monotonic() - started)
        def checkpoint(preset, campaign, selection):
            args = SimpleNamespace(reason="miniature complete protocol replay", selection=selection, preset=preset,
                                   runtime="all", campaign=campaign, repeat_reason="", base="fixture-baseline")
            # Other names model collector membership with native binaries on this OS.
            # The public native-platform admission remains exercised for the native lane.
            if preset == self.preset:
                return record_phase(f"checkpoint.{preset}.{campaign}", qualification.qualify, self.root, args)
            with OutputLease(self.root, self.root / "out/build" / preset):
                return record_phase(f"checkpoint.{preset}.{campaign}", qualification._qualify_owned,
                                    self.root, self.root / "out/build" / preset, args)
        paths = []
        with patch.object(Campaign, "run", autospec=True, side_effect=counted), \
                patch.object(NinjaGraph, "invalidate", autospec=True, side_effect=invalidate), \
                patch.object(qualification, "digest", side_effect=measured_digest), \
                patch.object(qualification, "check_preparation_inputs", side_effect=measured_inputs), \
                patch.object(qualification, "tool_snapshot", side_effect=measured_tools), \
                patch("execution.tool_snapshot", side_effect=measured_tools):
            for preset in qualification.PRESETS:
                build = self.root / "out/build" / preset
                campaign = Campaign(self.root, preset, "cold", ["prepare"])
                if preset.endswith("release"):
                    record_phase(f"prepare.{preset}.cold", qualification.prepare_build, campaign, "prepare", source=self.root, build=build,
                        inputs=list(files), timeout=60, targets=["all"], configure_args=["-DBUILD_TESTING=OFF"], checkpoint=True)
                    paths.append(checkpoint(preset, "off", ["production"]))
                else:
                    record_phase(f"configure.{preset}.cold", self.configure_fixture, preset, True, "fixture-on", timeout=60)
                groups = ["product", "boundary"]
                if preset.endswith("debug"):
                    groups += ["tooling"]
                if preset == "linux-debug":
                    groups += ["corpus"]
                paths.append(checkpoint(preset, "initial", groups))
            initial = self.root / "out/validation" / self.preset / "initial/qualification.json"
            selection = json.loads(initial.read_text())["selection"]
            # CMake may resolve preset compiler spellings during cold setup.
            # Admit the resulting tool inventory before the proof-only comparison.
            stable = checkpoint(self.preset, "stable", selection)
            paths = [stable if p == initial else p for p in paths]
            initial = stable
            initial_collection = record_phase("collect.initial", qualification.collect_qualification, self.root, paths)
            self.assertEqual(initial_collection["status"], "passed")
            first = json.loads(initial.read_text())
            proof = json.loads(Path(first["stages"]["prepare"]["receipt"]).read_text())["scoped_preparation"]
            artifacts = {p: digest(Path(p)) for p in proof["outputs"]}
            before_source = snapshot(self.root, list(files))
            originals = {str(p): digest(p) for p in (self.root / "out/validation").rglob("*")
                         if p.is_file() and p != progress_path}
            for path in paths:
                artifact_paths.update(Path(p).resolve() for p in json.loads(path.read_text())["stages"].get("prepare", {}).get("payloads", {}))
            warm = []
            for trial in range(3):
                start = time.monotonic()
                call_start, delete_start, hash_start = len(calls), len(deletions), len(artifact_reads)
                check_start = len(input_checks)
                tool_start = len(tool_checks)
                with patch.object(qualification, "check_scoped_preparation", side_effect=ValueError("injected checkpoint proof validation failure")):
                    with self.assertRaisesRegex(ValueError, "injected checkpoint"):
                        checkpoint(self.preset, f"failed-{trial}", first["selection"])
                with self.assertRaisesRegex(ValueError, "pending checkpoint work"):
                    qualification.collect_qualification(self.root, paths)
                self.assertEqual(artifacts, {p: digest(Path(p)) for p in artifacts})
                state = json.loads(self.state_path.read_text())
                repair = self.campaign(f"repair-{trial}")
                repair_start = time.monotonic()
                renewed = record_phase(f"prepare.repair-{trial}", qualification.prepare_checkpoint,
                    repair, self.build, self.state_path, state, dict(os.environ),
                    timeout=60, repeat_reason="diagnosed checkpoint validation failure; completed artifact producer intact")
                repair_seconds = time.monotonic() - repair_start
                self.assertNotEqual(renewed["sha256"], first["stages"]["prepare"]["sha256"])
                current = checkpoint(self.preset, f"continued-{trial}", first["selection"])
                paths = [current if p == initial or p.parent.name.startswith("continued-") and p.parent.parent.name == self.preset else p for p in paths]
                collection_start = time.monotonic()
                collected = record_phase(f"collect.continued-{trial}", qualification.collect_qualification, self.root, paths)
                collection_seconds = time.monotonic() - collection_start
                self.assertEqual(collected["status"], "passed")
                self.assertEqual(before_source, snapshot(self.root, list(files)))
                self.assertEqual(artifacts, {p: digest(Path(p)) for p in artifacts})
                value = subprocess.check_output([str(self.build / ("behavior.exe" if os.name == "nt" else "behavior")), "10"], text=True, timeout=10)
                self.assertIn("value=10 abi=8 offset=4", value)
                logs = [self.root / json.loads(p.read_text())["output"] for p in repair.directory.glob("attempts/*/result.json")]
                self.assertTrue(any("no work to do" in p.read_text() for p in logs))
                warm.append({"seconds": time.monotonic() - start, "commands": calls[call_start:], "deleted": deletions[delete_start:],
                             "repair_seconds": repair_seconds, "collection_seconds": collection_seconds,
                             "artifact_hash_reads": len(artifact_reads) - hash_start,
                             "artifact_hash_seconds": sum(row[0] for row in artifact_reads[hash_start:]),
                             "artifact_bytes_hashed": sum(row[1] for row in artifact_reads[hash_start:]),
                             "preparation_input_checks": len(input_checks) - check_start,
                             "preparation_input_check_seconds": sum(input_checks[check_start:]),
                             "tool_checks": len(tool_checks) - tool_start,
                             "tool_check_seconds": sum(tool_checks[tool_start:]),
                             "renewed": renewed, "checkpoint": str(current), "collection": collected})
            self.assertEqual(originals, {p: digest(Path(p)) for p in originals})
            summary = {"scope": "native miniature protocol; four synthetic lane names, no Film-Juicer/GPU/OFX qualification",
                       "fixture": str(self.root), "native": os.name, "artifacts": artifacts, "inputs": before_source,
                       "initial_collection": initial_collection, "warm": warm, "seconds": time.monotonic() - started,
                       "tool_checks": len(tool_checks), "tool_check_seconds": sum(tool_checks),
                       "rust_recovery": "not applicable; no Rust fixture", "cold_and_warm_commands": calls}
            for trial in warm:
                self.assertFalse(trial["deleted"])
                # Compiler/linker modules keep configure opaque. Require that
                # refresh plus one no-op build, with no downstream work.
                self.assertEqual([c["stage"] for c in trial["commands"]], ["prepare", "prepare"],
                                 "proof renewal repeated unchanged quality/product work")
                configure, build = [c["command"] for c in trial["commands"]]
                self.assertEqual(configure[:4], ["cmake", "--preset", self.preset, "-DBUILD_TESTING=ON"])
                self.assertIn("--trace-expand", configure[4:])
                self.assertEqual(build, ["cmake", "--build", str(self.build), "--parallel", "2", "--target", "all"])
            write_json(self.root / "out/validation" / self.preset / "replay.json", summary)
            progress.update(complete=True, seconds=time.monotonic() - started)
            write_json(progress_path, progress)

    def prepare_compiled_project(self):
        self.compiled_project()
        self.configure_fixture(self.preset, True, "fixture-on", timeout=60)
        return qualification.prepare_checkpoint(self.campaign("compiled"), self.build, self.state_path, self.state,
                                                dict(os.environ), timeout=60)

    def test_preparation_output_bytes_are_required_for_reuse(self):
        self.assert_output_mutation("behavior.exe" if os.name == "nt" else "behavior")

    def test_collection_rechecks_current_preparation_inputs(self):
        self.assert_collection_input_mutations([
            ("generated/binding.h", lambda b: b.replace(b"BINDING 4", b"BINDING 5"), "dependency contents"),
            ("CTestTestfile.cmake", lambda b: b + b'\nadd_test(unrun "missing")\n', "test registration"),
            ("generated/new-shadow.h", lambda b: b"#define SHADOW 1\n", "include membership"),
        ])

    def test_collection_rechecks_changed_cache(self):
        # Retain this mutation in its own fixture. DrvFS can round timestamps
        # when restoring them, so it cannot be another case's fresh baseline.
        self.assert_collection_input_mutations([
            ("CMakeCache.txt", lambda b: b.replace(b"CMAKE_CXX_FLAGS:STRING=", b"CMAKE_CXX_FLAGS:STRING=-DABI_CHANGED=1 "), "configuration inputs"),
        ], retain_mutation=True)

    def assert_collection_input_mutations(self, mutations, *, retain_mutation=False):
        self.prepare_compiled_project()
        args = SimpleNamespace(reason="collection input drift control", selection=["product"], preset=self.preset,
                               runtime="all", campaign="admitted", repeat_reason="", base="fixture-baseline")
        path = qualification.qualify(self.root, args)
        record = json.loads(path.read_text())
        stage = record["stages"]["prepare"]
        identity = qualification.preparation_identity(self.build, stage)
        # Other fixture lanes are deliberately absent; the complete replay owns
        # positive aggregation. This control must reach that completeness check.
        with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
            qualification.collect_qualification(self.root, [path])
        executable = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        outcomes = []
        for name, mutate, diagnostic in mutations:
            with self.subTest(input=name):
                target = self.build / name
                content = target.read_bytes() if target.exists() else None
                stamp = target.stat() if target.exists() else None
                try:
                    changed = mutate(content or b"")
                    self.assertNotEqual(content, changed)
                    target.write_bytes(changed)
                    if stamp:
                        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        qualification.preparation_identity(self.build, stage)
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        qualification.collect_qualification(self.root, [path])
                    value = subprocess.check_output([str(executable), "10"], text=True, timeout=10)
                    self.assertIn("value=10 abi=8 offset=4", value)
                    outcomes.append({"input": name, "rejected": diagnostic,
                                     "mtime_restoration_exact": bool(stamp) and target.stat().st_mtime_ns == stamp.st_mtime_ns,
                                     "behavior": value.strip(), "mutation_retained": retain_mutation})
                finally:
                    if retain_mutation:
                        pass
                    elif content is None:
                        target.unlink()
                    else:
                        target.write_bytes(content)
                        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            if not retain_mutation:
                self.assertEqual(identity, qualification.preparation_identity(self.build, stage))
        write_json(path.parent / "collection-controls.json", outcomes)

    def test_consumed_environment_change_cannot_reuse_product(self):
        files = self.compiled_project()
        self.configure_fixture(self.preset, True, "fixture-on", timeout=60)
        args = SimpleNamespace(reason="consumed runtime environment control", selection=["product"], preset=self.preset,
                               runtime="all", campaign="environment-baseline", repeat_reason="", base="fixture-baseline")
        outcomes = {}
        with patch.dict(os.environ, {"JUICER_P0_A_RUNTIME_OFFSET": "0"}):
            first = qualification.qualify(self.root, args)
            original = first.read_bytes()
            source = snapshot(self.root, list(files))
            args.campaign = "environment-changed"
            with patch.dict(os.environ, {"JUICER_P0_A_RUNTIME_OFFSET": "1"}):
                executable = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
                stale = subprocess.run([str(executable), "10"], capture_output=True, text=True, timeout=10)
                self.assertNotEqual(stale.returncode, 0)
                self.assertIn("value=11 abi=8 offset=4", stale.stdout)
                with self.assertRaises(subprocess.CalledProcessError):
                    qualification.qualify(self.root, args)
                with self.assertRaisesRegex(ValueError, "pending checkpoint work"):
                    qualification.collect_qualification(self.root, [first])
                self.assertEqual((self.root / "out" / (self.preset + "-quality-count.txt")).read_text(), "2")
                failed = self.root / "out/validation" / self.preset / args.campaign / "qualification.json"
                outcomes.update(changed_checkpoint=str(failed), witness=stale.stdout.strip(), variable="JUICER_P0_A_RUNTIME_OFFSET")
            self.assertEqual(original, first.read_bytes())
            self.assertEqual(source, snapshot(self.root, list(source)))
            args.campaign = "environment-restored"
            current = qualification.qualify(self.root, args)
            self.assertEqual(json.loads(current.read_text())["status"], "passed")
            outcomes.update(baseline=str(first), restored=str(current), status="changed environment executed quality and failed real product; restored environment passed")
            write_json(current.parent / "environment-control.json", outcomes)

    def test_preparation_archive_bytes_are_required_for_reuse(self):
        self.assert_output_mutation("juicer.lib" if os.name == "nt" else "libjuicer.a")

    def assert_output_mutation(self, name):
        first = self.prepare_compiled_project()
        original = Path(first["receipt"]).read_bytes()
        artifact = self.build / name
        content, stamp = artifact.read_bytes(), artifact.stat()
        artifact.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        os.utime(artifact, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "payload changed|artifacts changed"):
            qualification.prepare_checkpoint(self.campaign("mutated"), self.build, self.state_path, self.state,
                                             dict(os.environ), timeout=60)
        self.assertEqual(self.state["prepare"]["status"], "failed")
        self.assertEqual(original, Path(first["receipt"]).read_bytes())
        for attempt in range(2):
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                qualification.prepare_checkpoint(self.campaign(f"repair-mutated-{attempt}"), self.build, self.state_path,
                    self.state, dict(os.environ), timeout=60, repeat_reason="diagnosed artifact admission failure")
            self.assertEqual(self.state["prepare"]["status"], "failed")
            self.assertEqual(first["payloads"], self.state["prepare"]["payloads"])

    def test_preparation_receipt_and_owner_permutations_preserve_one_baseline(self):
        first = self.prepare_compiled_project()
        record = json.loads(Path(first["receipt"]).read_text())
        process = Path(record["process_tree"]["receipt"])
        original_process = process.read_bytes()
        document = json.loads(original_process)
        document["drained"] = False
        process.write_text(json.dumps(document))
        with patch.object(Campaign, "run", side_effect=AssertionError("dispatched with unproved drainage")):
            for trial in range(2):
                with self.assertRaisesRegex(ValueError, "process evidence"):
                    qualification.prepare_checkpoint(self.campaign(f"undrained-{trial}"), self.build, self.state_path,
                        self.state, dict(os.environ), timeout=60, repeat_reason="diagnosed process proof failure")
                self.assertEqual(self.state["prepare"]["status"], "failed")
                self.assertEqual(first["payloads"], self.state["prepare"]["payloads"])
        process.write_bytes(original_process)

        receipt = Path(first["receipt"]).read_bytes()
        expected = qualification.preparation_identity(self.build, first)
        candidate = {**first, "payloads": dict(first["payloads"])}
        candidate["payloads"].pop(next(iter(candidate["payloads"])))
        with self.assertRaisesRegex(ValueError, "exact artifact identities"):
            qualification.preparation_identity(self.build, candidate)
        from build_evidence import preparation_paths
        baseline, pending = preparation_paths(self.build, ["all"])
        pending.write_text('{"test": "unproved artifact writer"}')
        with self.assertRaisesRegex(ValueError, "pending preparation"):
            qualification.preparation_identity(self.build, first)
        # Only this fixture's injected pending marker is removed; no writer ran.
        pending.unlink()
        self.assertEqual(expected, qualification.preparation_identity(self.build, first))
        original = baseline.read_bytes()
        changed = json.loads(original)
        changed["producer"] = "superseded"
        baseline.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "latest completed preparation"):
            qualification.preparation_identity(self.build, first)
        baseline.write_bytes(original)
        self.assertEqual(Path(first["receipt"]).read_bytes(), receipt)

    def test_restored_time_edit_exposes_stale_behavior_before_preparation(self):
        first = self.prepare_compiled_project()
        executable = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        subprocess.run([str(executable), "10"], check=True, capture_output=True, timeout=10)
        source = self.root / "src/native.cpp"
        content, stamp = source.read_bytes(), source.stat()
        source.write_bytes(content.replace(b"6+BINDING", b"7+BINDING"))
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        old = subprocess.run([str(executable), "11"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(old.returncode, 0)
        self.assertIn("value=10 abi=8 offset=4", old.stdout)
        with self.assertRaisesRegex(ValueError, "dependency contents"):
            qualification.check_scoped_preparation(self.campaign("stale"), self.build, Path(first["receipt"]))
        qualification.prepare_checkpoint(self.campaign("rebuilt"), self.build, self.state_path, self.state,
                                         dict(os.environ), timeout=60)
        new = subprocess.run([str(executable), "11"], check=True, capture_output=True, text=True, timeout=10)
        self.assertIn("value=11 abi=8 offset=4", new.stdout)

    def mixed_project(self):
        self.native_project()
        probe = self.root / "probe.py"
        probe.write_text(
            "import json,sys,time\nfrom pathlib import Path\n"
            "root=Path(__file__).parent\nrole=sys.argv[1]\n"
            f"state=json.loads((root/'out/build/{self.preset}/verification-checkpoint.json').read_text())\n"
            "assert state['boundary']['status']=='pending'\n"
            "leases=list((root/'out/verification-locks').glob('*/owner.json'))\n"
            "assert any(p.parent.name.startswith('build-') for p in leases)\n"
            "assert (root/'out/verification-locks/gpu/owner.json').is_file()==(role=='gpu')\n"
            "with (root/'out/order.txt').open('a') as f:f.write(role+'\\n')\n"
            "mode=(root/'failure.txt').read_text() if (root/'failure.txt').exists() else ''\n"
            "if mode=='timeout' and role=='gpu':time.sleep(30)\n"
            "raise SystemExit(1 if mode==role else 0)\n")
        cmake = self.root / "CMakeLists.txt"
        text = cmake.read_text()
        text += '\n'
        for role in ("host", "gpu"):
            text += (f'add_test(NAME {role} COMMAND "{Path(self.python).as_posix()}" "{probe.as_posix()}" {role})\n'
                     f'set_tests_properties({role} PROPERTIES LABELS "{role};boundary")\n')
        cmake.write_text(text)
        self.configure_fixture(self.preset, True, "fixture-on")
        return SimpleNamespace(reason="mixed runtime regression", selection=["boundary"], preset=self.preset,
                               runtime="all", campaign="mixed", repeat_reason="", base=None)

    def test_mixed_ctest_owns_partitions_and_reuses_without_gpu_lease(self):
        args = self.mixed_project()
        actual_lease = qualification.OutputLease
        def bounded_lease(root, build=None, **kwargs):
            if build is None:
                self.assertGreater(kwargs["timeout"], 30, "GPU wait must use the remaining stage budget")
                self.assertLessEqual(kwargs["timeout"], 1800)
                self.assertEqual(json.loads(self.state_path.read_text())["boundary"]["status"], "pending")
            return actual_lease(root, build, **kwargs)
        with patch.object(qualification, "OutputLease", side_effect=bounded_lease):
            first = json.loads(qualification.qualify(self.root, args).read_text())
        self.assertEqual((self.root / "out/order.txt").read_text().splitlines(), ["host", "gpu"])
        stage = first["stages"]["boundary"]
        self.assertEqual(len(stage["additional_receipts"]), 1)
        self.assertEqual(first["runtime_membership"]["boundary"], {"host": ["boundary", "host"], "gpu": ["gpu"]})
        self.assertFalse((OutputLease(self.root).path / "owner.json").exists())
        args.campaign = "reused-mixed"
        actual_lease = qualification.OutputLease
        def build_lease(root, build=None, **kwargs):
            self.assertIsNotNone(build, "unchanged reuse attempted to acquire the GPU lease")
            return actual_lease(root, build, **kwargs)
        with patch.object(qualification, "run_ctest_parts", side_effect=AssertionError("executed reused stage")), \
                patch.object(qualification, "OutputLease", side_effect=build_lease):
            reused = json.loads(qualification.qualify(self.root, args).read_text())
        self.assertEqual(reused["stages"]["boundary"]["reused_from"], stage["receipt"])
        self.assertEqual((self.root / "out/order.txt").read_text().splitlines(), ["host", "gpu"])
        secondary = Path(next(iter(stage["additional_receipts"])))
        original = secondary.read_bytes()
        secondary.write_bytes(original + b" ")
        args.campaign = "changed-secondary"
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            qualification.qualify(self.root, args)
        self.assertEqual(json.loads(self.state_path.read_text())["boundary"]["status"], "failed")
        secondary.write_bytes(original)
        args.campaign = "cannot-hide-failure"
        with self.assertRaisesRegex(ValueError, "pending checkpoint"):
            qualification.qualify(self.root, args)

    def test_mixed_ctest_host_failure_stops_gpu_and_gpu_failure_stays_pending(self):
        args = self.mixed_project()
        for role in ("host", "gpu"):
            (self.root / "failure.txt").write_text(role)
            args.campaign = "fail-" + role
            args.repeat_reason = "exercise distinct partition failure"
            (self.root / "out/order.txt").unlink(missing_ok=True)
            with self.assertRaises(subprocess.CalledProcessError):
                qualification.qualify(self.root, args)
            self.assertEqual((self.root / "out/order.txt").read_text().splitlines(),
                             ["host"] if role == "host" else ["host", "gpu"])
            self.assertEqual(json.loads(self.state_path.read_text())["boundary"]["status"], "failed")
            self.assertFalse((OutputLease(self.root).path / "owner.json").exists())
            args.campaign = "new-caller-" + role
            args.repeat_reason = ""
            with self.assertRaisesRegex(ValueError, "pending checkpoint"):
                qualification.qualify(self.root, args)

    def test_zero_exit_host_with_unrun_case_never_schedules_gpu(self):
        args = self.mixed_project()
        actual = Campaign.run
        def incomplete(campaign, stage, command, **kwargs):
            path = actual(campaign, stage, command, **kwargs)
            if "-LE" in command:
                import xml.etree.ElementTree as ET
                report = Path(command[command.index("--output-junit") + 1])
                document = ET.parse(report)
                next(document.getroot().iter("testcase")).set("status", "notrun")
                document.write(report)
            return path
        with patch.object(Campaign, "run", autospec=True, side_effect=incomplete):
            with self.assertRaisesRegex(ValueError, "did not execute"):
                qualification.qualify(self.root, args)
        self.assertEqual((self.root / "out/order.txt").read_text().splitlines(), ["host"])
        self.assertEqual(json.loads(self.state_path.read_text())["boundary"]["status"], "failed")
        self.assertEqual(len(Campaign(self.root, self.preset, "mixed", ["boundary"]).attempts("boundary")), 1)

    def test_mixed_ctest_timeout_drains_gpu_and_keeps_whole_stage_failed(self):
        args = self.mixed_project()
        (self.root / "failure.txt").write_text("timeout")
        actual = qualification.run_ctest_parts
        def bounded(*args, **kwargs):
            kwargs["timeout"] = 15
            return actual(*args, **kwargs)
        with patch.object(qualification, "run_ctest_parts", side_effect=bounded):
            with self.assertRaises(subprocess.TimeoutExpired):
                qualification.qualify(self.root, args)
        self.assertEqual(json.loads(self.state_path.read_text())["boundary"]["status"], "failed")
        raw = Campaign(self.root, self.preset, "mixed", ["boundary"]).attempts("boundary")
        self.assertEqual([r["status"] for r in raw], ["passed", "failed"])
        self.assertTrue(all(r["process_tree"]["drained"] for r in raw))
        self.assertFalse((OutputLease(self.root).path / "owner.json").exists())

    def test_collection_binds_all_partition_receipts_and_aggregate_membership(self):
        args = self.mixed_project()
        path = qualification.qualify(self.root, args)
        record = json.loads(path.read_text())
        # The valid partial checkpoint reaches only the final full-matrix gate.
        with self.assertRaisesRegex(ValueError, "membership is incomplete"):
            qualification.collect_qualification(self.root, [path])
        for change in ("secondary", "overlap", "missing", "swapped"):
            candidate = json.loads(json.dumps(record))
            if change == "secondary":
                candidate["stages"]["boundary"]["additional_receipts"] = {}
            elif change == "overlap":
                candidate["runtime_membership"]["boundary"]["host"].append("gpu")
            elif change == "missing":
                del candidate["runtime_membership"]["boundary"]["gpu"]
            else:
                parts = candidate["runtime_membership"]["boundary"]
                parts["host"], parts["gpu"] = parts["gpu"], parts["host"]
            path.write_text(json.dumps(candidate))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "ownership|partition"):
                qualification.collect_qualification(self.root, [path])

    def test_real_off_to_on_preparation_owns_generated_configuration(self):
        self.native_project()
        inputs = qualification.checkpoint_inputs(self.root, self.build, "product")
        source_inputs = [p for p in inputs if not p.startswith("out/build/")]
        def prepare():
            return qualification.prepare_build(self.campaign("first"), "prepare", source=self.root,
                build=self.build, inputs=source_inputs, timeout=30, targets=["all"],
                configure_args=["-DBUILD_TESTING=ON"], checkpoint=True)[1]
        first = self.run_stage("prepare", inputs=inputs, action=prepare,
            refresh_inputs=lambda: qualification.checkpoint_inputs(self.root, self.build, "product"),
            validate=lambda p: qualification.check_scoped_preparation(self.campaign("first"), self.build, p))
        self.assertIn("BUILD_TESTING:BOOL=ON", (self.build / "CMakeCache.txt").read_text())
        self.assertTrue((self.build / "CTestTestfile.cmake").is_file())
        refreshed = qualification.checkpoint_inputs(self.root, self.build, "product")
        second = self.run_stage("prepare", campaign="next-caller", inputs=refreshed, action=prepare,
            validate=lambda p: qualification.check_scoped_preparation(self.campaign("next-caller"), self.build, p))
        self.assertEqual(second["reused_from"], first["receipt"])

    def test_production_inputs_do_not_require_test_registration(self):
        self.native_project()
        helpers = ("tests/ffi/grain_frame_abi_assertions.inc", "tests/ffi/grain_probe_contract.py")
        for name in helpers:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        self.assertFalse((self.build / "CTestTestfile.cmake").exists())
        inputs = qualification.checkpoint_inputs(self.root, self.build, "production")
        self.assertIn("CMakeLists.txt", inputs)
        self.assertIn((self.build / "CMakeCache.txt").relative_to(self.root).as_posix(), inputs)
        for name in helpers:
            self.assertIn(name, inputs)

    @unittest.skipUnless(os.name == "nt", "Windows production PDB")
    def test_production_pdb_changes_and_missing_repair_cannot_reuse_old_pass(self):
        self.native_project()
        self.preset = "windows-clang-release"
        self.build = self.root / "out/build" / self.preset
        self.build.mkdir()
        (self.build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={self.root.as_posix()}\nBUILD_TESTING:BOOL=OFF\n")
        module = self.build / "juicer.ofx"
        module.write_text("module")
        pdb = self.build / "juicer.pdb"
        pdb.write_text("first")
        driver = self.root / "tests/ffi/production_contract.py"
        driver.parent.mkdir(parents=True)
        driver.write_text(
            "import argparse\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser();p.add_argument('--preset');p.add_argument('--artifacts')\n"
            "a=p.parse_args();root=Path.cwd();out=Path(a.artifacts);out.mkdir(parents=True,exist_ok=True)\n"
            "count=root/'out/production-count.txt'\n"
            "count.write_text(str(int(count.read_text())+1) if count.exists() else '1')\n"
            "assert (root/'out/build'/a.preset/'juicer.pdb').is_file()\n"
            + self.production_fixture())
        args = SimpleNamespace(reason="production PDB identity regression", selection=["production"],
                               preset=self.preset, runtime="host", campaign="first", repeat_reason="", base=None)
        count = self.root / "out/production-count.txt"
        with patch("build_dependencies.NinjaGraph", return_value=SimpleNamespace(outputs={str(module)})), \
                patch.object(qualification, "checkpoint_tools", return_value=[self.python]):
            first = json.loads(qualification.qualify(self.root, args).read_text())["stages"]["production"]
            self.assertEqual(first["identity"]["inputs"][pdb.relative_to(self.root).as_posix()], digest(pdb))
            args.campaign = "unchanged"
            reused = json.loads(qualification.qualify(self.root, args).read_text())["stages"]["production"]
            self.assertEqual(reused["reused_from"], first["receipt"])
            self.assertEqual(count.read_text(), "1")
            stamp = pdb.stat()
            pdb.write_text("other")
            os.utime(pdb, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            args.campaign = "pdb-changed"
            changed = json.loads(qualification.qualify(self.root, args).read_text())["stages"]["production"]
            self.assertNotIn("reused_from", changed)
            self.assertEqual(count.read_text(), "2")
            pdb.unlink()
            args.campaign = "pdb-missing"
            with self.assertRaises(subprocess.CalledProcessError):
                qualification.qualify(self.root, args)
            state = self.build / "verification-checkpoint.json"
            self.assertEqual(json.loads(state.read_text())["production"]["status"], "failed")
            self.assertEqual(count.read_text(), "3")
            args.campaign = "missing-still-pending"
            with self.assertRaisesRegex(ValueError, "pending checkpoint"):
                qualification.qualify(self.root, args)
            self.assertEqual(count.read_text(), "3")
            pdb.write_text("other")
            args.campaign = "pdb-repaired"
            repaired = json.loads(qualification.qualify(self.root, args).read_text())["stages"]["production"]
            self.assertNotIn("reused_from", repaired)
            self.assertEqual(count.read_text(), "4")

    @unittest.skipUnless(os.name == "nt", "Windows production symbol tools")
    def test_production_symbol_tool_changes_invalidate_reuse(self):
        (self.build / "CMakeCache.txt").write_text("")
        llvm = self.root / "llvm"
        llvm.mkdir()
        for name in ("llvm-nm.exe", "clang-cl.exe", "llvm-readobj.exe", "llvm-pdbutil.exe"):
            (llvm / name).write_text("first")
        environment = {"PATH": "", "CARGO_HOME": str(self.root / "cargo-home"),
                       "RUSTUP_HOME": str(self.root / "rustup")}
        def local_path(value, *parts):
            if value == "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin":
                return llvm
            return Path(value, *parts)
        with patch.object(qualification, "Path", side_effect=local_path) as factory:
            factory.home.return_value = Path.home()
            tools = qualification.checkpoint_tools(self.build, environment, "production")
        for name in ("llvm-readobj.exe", "llvm-pdbutil.exe"):
            self.assertIn(str(llvm / name), tools)
        previous = self.run_stage("production", tools=tools)
        for name in ("llvm-readobj.exe", "llvm-pdbutil.exe"):
            tool = llvm / name
            stamp = tool.stat()
            tool.write_text("other")
            os.utime(tool, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            changed = self.run_stage("production", campaign=name.removesuffix(".exe"), tools=tools)
            self.assertNotIn("reused_from", changed)
            self.assertNotEqual(changed["receipt"], previous["receipt"])
            previous = changed

    def retained_record(self, stage):
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "retained",
                  "preset": self.preset, "selection": ["corpus"], "required": ["corpus"], "stages": {"corpus": stage}}
        path = self.root / "checkpoint.json"
        path.write_text(json.dumps(record))
        return path

    def test_collection_rechecks_authored_data_and_input_membership(self):
        data = self.root / "Resources/data.txt"
        data.parent.mkdir()
        data.write_text("authored")
        inputs = ["Resources/data.txt"]
        stage = self.run_stage("corpus", inputs=inputs)
        path = self.retained_record(stage)
        with patch.object(qualification, "checkpoint_inputs", return_value=inputs):
            with self.assertRaisesRegex(ValueError, "membership is incomplete"):
                qualification.collect_qualification(self.root, [path])
            data.write_text("changed")
            with self.assertRaisesRegex(ValueError, "source/data inputs changed"):
                qualification.collect_qualification(self.root, [path])
            data.write_text("authored")
        with patch.object(qualification, "checkpoint_inputs", return_value=inputs + ["Resources/new.txt"]):
            with self.assertRaisesRegex(ValueError, "source/data membership changed"):
                qualification.collect_qualification(self.root, [path])

    def test_collection_rejects_previously_absent_tool_configuration(self):
        config = self.root / "cargo-config.toml"
        stage = self.run_stage("corpus", tools=[self.python, str(config)])
        path = self.retained_record(stage)
        config.write_text('[build]\n')
        with patch.object(qualification, "checkpoint_inputs", return_value=["input.txt"]):
            with self.assertRaisesRegex(ValueError, "toolchain/configuration changed"):
                qualification.collect_qualification(self.root, [path])

    def test_collection_rechecks_tool_bytes_at_completion(self):
        self.assert_collection_tool_seal("bytes")

    def test_collection_rechecks_source_data_at_completion(self):
        data = self.root / "Resources/data.txt"
        data.parent.mkdir()
        data.write_text("authored")
        inputs = [data.relative_to(self.root).as_posix()]
        stage = self.run_stage("corpus", inputs=inputs)
        path = self.retained_record(stage)
        actual_digest = qualification.digest
        def mutate_during_read(target):
            value = actual_digest(target)
            if str(target) == stage["receipt"]:
                stamp = data.stat()
                data.write_text("modified")
                os.utime(data, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            return value
        with patch.object(qualification, "checkpoint_inputs", return_value=inputs), \
                patch.object(qualification, "digest", side_effect=mutate_during_read):
            with self.assertRaisesRegex(ValueError, "source/data inputs changed during collection"):
                qualification.collect_qualification(self.root, [path])

    def test_collection_rechecks_source_membership_at_completion(self):
        import verify
        self.native_project()
        inputs = qualification.checkpoint_inputs(self.root, self.build, "corpus")
        stage = self.run_stage("corpus", inputs=inputs)
        path = self.retained_record(stage)
        record = json.loads(path.read_text())
        target = self.root / "out/validation" / self.preset / "corpus-record.json"
        path.rename(target)
        record["source"] = verify.source_snapshot(self.root)
        target.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
            qualification.collect_qualification(self.root, [target])
        actual_digest = qualification.digest
        def mutate_during_read(path):
            value = actual_digest(path)
            if str(path) == stage["receipt"]:
                (self.root / "new-consumed.cpp").write_text("int added();\n")
            return value
        with patch.object(qualification, "digest", side_effect=mutate_during_read):
            with self.assertRaisesRegex(ValueError, "source/data membership changed during collection"):
                qualification.collect_qualification(self.root, [target])
        self.assertEqual(digest(Path(stage["receipt"])), stage["sha256"])

    def test_collection_rechecks_ignored_input_membership_at_completion(self):
        import verify
        self.native_project()
        added = self.root / "tests/ffi/out/added.bin"
        added.parent.mkdir(parents=True)
        inputs = qualification.checkpoint_inputs(self.root, self.build, "corpus")
        stage = self.run_stage("corpus", inputs=inputs)
        path = self.retained_record(stage)
        record = json.loads(path.read_text())
        target = self.root / "out/validation" / self.preset / "ignored-membership-record.json"
        path.rename(target)
        record["source"] = verify.source_snapshot(self.root)
        target.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
            qualification.collect_qualification(self.root, [target])
        actual_inputs = qualification.checkpoint_inputs
        changed = []
        def mutate_after_discovery(*args, **kwargs):
            result = actual_inputs(*args, **kwargs)
            if not changed:
                added.write_bytes(b"new declared test input")
                changed.append(added.relative_to(self.root).as_posix())
            return result
        with patch.object(qualification, "checkpoint_inputs", side_effect=mutate_after_discovery):
            with self.assertRaisesRegex(ValueError, "source/data membership changed during collection"):
                qualification.collect_qualification(self.root, [target])
        self.assertEqual(changed, [added.relative_to(self.root).as_posix()])
        subprocess.run(["git", "check-ignore", "--quiet", "--", changed[0]], cwd=self.root,
                       check=True, capture_output=True, timeout=10)
        self.assertEqual(record["source"], verify.source_snapshot(self.root))
        self.assertIn(changed[0], actual_inputs(self.root, self.build, "corpus"))
        self.assertEqual(digest(Path(stage["receipt"])), stage["sha256"])
        write_json(target.parent / "ignored-membership-control.json",
                   {"added": changed[0], "git_ignored": True, "candidate_unchanged": True,
                    "rejected": "source/data membership changed during collection",
                    "receipt": stage["receipt"], "sha256": stage["sha256"]})

    def test_collection_rechecks_absent_tool_configuration_at_completion(self):
        self.assert_collection_tool_seal("absent")

    def test_collection_rechecks_tool_path_binding_at_completion(self):
        self.assert_collection_tool_seal("binding")

    def assert_collection_tool_seal(self, change):
        directory = self.root / "out/validation" / self.preset / "tool-seal"
        first, second, alias = (directory / name for name in ("first", "second", "alias"))
        for target in (first, second):
            target.mkdir(parents=True)
            (target / "tool").write_text("first")
        def bind(target):
            if os.name == "nt":
                subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(target)],
                               check=True, capture_output=True, timeout=10)
            else:
                alias.symlink_to(target, target_is_directory=True)
        def unbind():
            if os.name == "nt":
                alias.rmdir()
            else:
                alias.unlink()
        bind(first)
        config = directory / "absent.toml"
        tools = [self.python, str(first / "tool"), str(second / "tool"), str(alias / "tool"), str(config)]
        stages = {name: self.run_stage(name, tools=tools) for name in ("boundary", "corpus", "tooling")}
        path = self.retained_record(stages["corpus"])
        record = json.loads(path.read_text())
        record.update(selection=list(stages), required=list(stages), stages=stages)
        path.write_text(json.dumps(record))
        originals = {stage["receipt"]: digest(Path(stage["receipt"])) for stage in stages.values()}
        actual_digest = qualification.digest
        changed = []
        def mutate_during_read(path):
            value = actual_digest(path)
            if not changed and str(path) in originals:
                if change == "bytes":
                    tool = first / "tool"
                    stamp = tool.stat()
                    tool.write_text("other")
                    os.utime(tool, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                elif change == "absent":
                    config.write_text("[build]\n")
                else:
                    unbind()
                    bind(second)
                changed.append(change)
            return value
        with patch.object(qualification, "checkpoint_inputs", return_value=["input.txt"]):
            with patch.object(qualification, "tool_snapshot", wraps=qualification.tool_snapshot) as reads:
                with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
                    qualification.collect_qualification(self.root, [path])
                self.assertEqual(reads.call_count, 2, "one opening and one closing tool scan for all stages")
            with patch.object(qualification, "digest", side_effect=mutate_during_read):
                with self.assertRaisesRegex(ValueError, "toolchain/configuration changed during collection"):
                    qualification.collect_qualification(self.root, [path])
            self.assertEqual(changed, [change])
            if change != "binding":
                # A later collection must observe the retained mutation afresh.
                with self.assertRaisesRegex(ValueError, "toolchain/configuration changed"):
                    qualification.collect_qualification(self.root, [path])
        self.assertEqual(originals, {p: digest(Path(p)) for p in originals})
        write_json(directory / "control.json", {"change": change, "rejected": True,
                   "scope": "real collector; incomplete lane membership remains rejected",
                   "original_receipts": originals, "tool_reads_per_collection": 2})

    def test_collection_rejects_production_without_execution_bound_manifest(self):
        artifact = self.build / "plugin.a"
        artifact.write_bytes(b"inspected final artifact")
        inputs = ["input.txt", artifact.relative_to(self.root).as_posix()]
        stage = self.run_stage("production", inputs=inputs)
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "production",
                  "preset": self.preset, "selection": ["production"], "required": ["production"],
                  "stages": {"production": stage}}
        path = self.root / "production-checkpoint.json"
        path.write_text(json.dumps(record))
        with patch.object(qualification, "checkpoint_inputs", return_value=inputs):
            with self.assertRaisesRegex(ValueError, "production command does not declare retained execution evidence"):
                qualification.collect_qualification(self.root, [path])

    def test_probe_artifact_ancestor_configuration_invalidates_checkpoint(self):
        (self.build / "CMakeCache.txt").write_text("")
        environment = {"PATH": "", "CARGO_HOME": str(self.root / "cargo-home")}
        tools = qualification.checkpoint_tools(self.build, environment)
        config = self.root / "out/.cargo/config.toml"
        first = self.run_stage("boundary", tools=tools)
        self.assertIn(str(config), tools)
        config.parent.mkdir(parents=True)
        config.write_text('[build]\n')
        changed = self.run_stage("boundary", campaign="artifact-config", tools=tools)
        self.assertNotEqual(first["receipt"], changed["receipt"])
        import importlib.util
        spec = importlib.util.spec_from_file_location("checkpoint_quality", ROOT / "scripts/check-quality.py")
        quality = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = quality
        spec.loader.exec_module(quality)
        with self.assertRaisesRegex(quality.QualityError, "different Cargo configuration"):
            quality.check_probe_configuration(self.root, self.root / "out/probe")
        config.unlink()
        config.mkdir()
        with self.assertRaisesRegex(quality.QualityError, "not a file"):
            quality.check_probe_configuration(self.root, self.root / "out/probe")

    def test_tool_snapshot_distinguishes_absent_files_from_invalid_directories(self):
        config = self.root / "config.toml"
        with self.assertRaises(FileNotFoundError):
            tool_snapshot([str(config)])
        self.assertEqual(tool_snapshot([str(config)], allow_missing=True), {str(config): None})
        config.mkdir()
        for allow_missing in (False, True):
            with self.subTest(allow_missing=allow_missing), self.assertRaisesRegex(ValueError, "not a file"):
                tool_snapshot([str(config)], allow_missing=allow_missing)


class FixtureLifetimeTests(unittest.TestCase):
    def run_checkpoint_fixture(self, environment, body):
        case = CheckpointTests("test_new_campaign_reuses_original_success_without_execution")
        case.test_new_campaign_reuses_original_success_without_execution = lambda: body(case)
        with patch.dict(os.environ, environment):
            result = case.run(unittest.TestResult())
        self.addCleanup(case.temporary.cleanup)
        return case, result

    def test_artifact_routing_cleans_successful_checkpoint_fixture(self):
        with tempfile.TemporaryDirectory() as artifact_root:
            artifact_root = Path(artifact_root).resolve()
            case, result = self.run_checkpoint_fixture({
                "JUICER_TEST_ARTIFACT_DIR": str(artifact_root),
                "JUICER_TEST_RETAIN_FIXTURES": "",
            }, lambda case: None)
        self.assertTrue(result.wasSuccessful())
        self.assertFalse(case.directory.exists())
        self.assertTrue(case.directory.is_relative_to(artifact_root))

    def test_external_inputs_follow_checkpoint_fixture_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary).resolve()
            for outcome in ("passed", "failed", "retained"):
                with self.subTest(outcome=outcome):
                    def body(case):
                        module = case.directory / "external/consumer.cmake"
                        module.parent.mkdir()
                        module.write_bytes(b"# original external input\n")
                        self.assertFalse(module.is_relative_to(case.root))
                        if outcome == "failed":
                            case.fail("external input retention control")
                    case, result = self.run_checkpoint_fixture({
                        "JUICER_TEST_ARTIFACT_DIR": str(artifact_root),
                        "JUICER_TEST_RETAIN_FIXTURES": "1" if outcome == "retained" else "",
                    }, body)
                    self.assertEqual(result.wasSuccessful(), outcome != "failed")
                    self.assertEqual(case.directory.exists(), outcome != "passed")
                    self.assertEqual(selected_paths(artifact_root), [case.directory] if outcome == "failed" else [])
                    if outcome != "passed":
                        self.assertEqual((case.directory / "external/consumer.cmake").read_bytes(), b"# original external input\n")
                        record = json.loads((case.directory / MARKER).read_text())
                        self.assertEqual(record["outcome"], "failed" if outcome == "failed" else "passed")
                        self.assertEqual(record["blocked_owners"], [])
                    case.temporary.cleanup()

    def test_diagnostic_opt_in_and_test_failures_retain_checkpoint_fixture(self):
        with tempfile.TemporaryDirectory() as artifact_root:
            case, result = self.run_checkpoint_fixture({
                "JUICER_TEST_ARTIFACT_DIR": artifact_root,
                "JUICER_TEST_RETAIN_FIXTURES": "1",
            }, lambda case: None)
            self.assertTrue(result.wasSuccessful())
            self.assertTrue(case.root.exists())
            self.assertEqual(json.loads((case.directory / MARKER).read_text())["outcome"], "passed")
            self.assertFalse((case.lease.path / "owner.json").exists())
            case.temporary.cleanup()

            def fail_test(case):
                case.fail("fixture lifecycle control")

            failed_case, failed_result = self.run_checkpoint_fixture({
                "JUICER_TEST_ARTIFACT_DIR": artifact_root,
                "JUICER_TEST_RETAIN_FIXTURES": "",
            }, fail_test)
            self.assertEqual(len(failed_result.failures), 1)
            self.assertTrue(failed_case.root.exists())
            self.assertEqual(json.loads((failed_case.directory / MARKER).read_text())["outcome"], "failed")
            failed_case.temporary.cleanup()

            def fail_subtest(case):
                with case.subTest(control="failure"):
                    case.fail("fixture lifecycle control")

            failed_case, failed_result = self.run_checkpoint_fixture({
                "JUICER_TEST_ARTIFACT_DIR": artifact_root,
                "JUICER_TEST_RETAIN_FIXTURES": "",
            }, fail_subtest)
            self.assertEqual(len(failed_result.failures), 1)
            self.assertTrue(failed_case.root.exists())
            self.assertFalse((failed_case.lease.path / "owner.json").exists())
            failed_case.temporary.cleanup()

    def test_missing_result_retains_checkpoint_fixture(self):
        case = CheckpointTests("test_new_campaign_reuses_original_success_without_execution")
        case.setUp()
        self.addCleanup(case.temporary.cleanup)
        case.doCleanups()
        self.assertTrue(case.root.exists())

    def test_cancellation_keeps_fixture_owner_after_completed_or_unproved_command(self):
        for drained in (True, False):
            with self.subTest(drained=drained):
                case = CheckpointTests("test_new_campaign_reuses_original_success_without_execution")
                previous = os.environ.get("JUICER_OUTPUT_LEASES")
                def cancel():
                    case.retain = True
                    module = case.directory / "external/consumer.cmake"
                    module.parent.mkdir()
                    module.write_bytes(b"# interrupted external input\n")
                    if drained:
                        case.run_stage("boundary")
                        raise KeyboardInterrupt
                    # No process is launched: cancellation has no terminal proof.
                    def unproved_launch(*args, **kwargs):
                        args[7].update(drained=False, status="running")
                        raise KeyboardInterrupt
                    with patch("execution.run_process_tree", side_effect=unproved_launch):
                        case.run_stage("boundary")
                setattr(case, case._testMethodName, cancel)
                with self.assertRaises(KeyboardInterrupt):
                    case.run(unittest.TestResult())
                self.addCleanup(case.temporary.cleanup)
                self.assertNotIn(case.lease.path, OutputLease.held)
                self.assertEqual(os.environ.get("JUICER_OUTPUT_LEASES"), previous)
                receipt = next((case.root / "out/validation").glob("*/first/attempts/*/result.json"))
                record = json.loads(receipt.read_text())
                owner = case.lease.path / "owner.json"
                self.assertEqual(record["process_tree"]["drained"], drained)
                self.assertTrue(owner.exists())
                self.assertEqual(json.loads(owner.read_text())["status"], "blocked")
                self.assertEqual((case.directory / "external/consumer.cmake").read_bytes(), b"# interrupted external input\n")
                marker = json.loads((case.directory / MARKER).read_text())
                self.assertEqual(marker["outcome"], "interrupted")
                self.assertEqual(marker["blocked_owners"], [str(owner)])
                self.assertIn(case.directory, selected_paths(case.directory.parent.parent))
                if drained:
                    self.assertEqual(record["status"], "passed")
                    process = json.loads(Path(record["process_tree"]["receipt"]).read_text())
                    self.assertTrue(process["drained"])
                    self.assertEqual(process["status"], "completed")
                else:
                    self.assertEqual(record["status"], "blocked")
                    self.assertTrue((case.root / "out/validation" / case.preset / "first/running.lock").exists())
                write_json(case.root / "out/validation" / case.preset / "cancellation.json",
                           {"scope": "cancelled fixture remains blocked; unproved branch launches no child",
                            "command_drained": drained, "receipt": str(receipt), "sha256": digest(receipt),
                            "owner_retained": owner.exists()})


class MembershipTests(unittest.TestCase):
    @staticmethod
    def inventory(*rows):
        return {"tests": [{"name": name, "properties": [{"name": "LABELS", "value": labels}]} for name, labels in rows]}

    def test_relevance_is_disjoint_and_runtime_is_orthogonal(self):
        inventory = self.inventory(("behavior", ["host", "product"]), ("boundary", ["host", "boundary"]),
                                   ("runner", ["host", "tooling"]), ("data", ["host", "corpus"]),
                                   ("device", ["gpu", "product"]))
        result = qualification.membership(inventory, ["product", "boundary", "tooling", "corpus"], "all")
        self.assertEqual(sorted(n for names in result.values() for n in names), ["behavior", "boundary", "data", "device", "runner"])
        self.assertNotIn("device", qualification.membership(inventory, ["product"], "host")["product"])

    def test_missing_duplicate_unclassified_and_empty_membership_fail(self):
        for inventory in (self.inventory(), self.inventory(("a", ["host"])),
                          self.inventory(("a", ["host", "product", "tooling"])),
                          self.inventory(("a", ["host", "product"]), ("a", ["host", "product"]))):
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                qualification.membership(inventory, ["product"], "host")

    def test_disabled_required_case_is_rejected(self):
        inventory = self.inventory(("a", ["host", "boundary"]))
        inventory["tests"][0]["properties"].append({"name": "DISABLED", "value": True})
        with self.assertRaisesRegex(ValueError, "disabled"):
            qualification.membership(inventory, ["boundary"], "host")

    def test_global_serialization_requires_a_named_owner(self):
        inventory = self.inventory(("a", ["host", "product"]))
        inventory["tests"][0]["properties"].append({"name": "RUN_SERIAL", "value": True})
        with self.assertRaisesRegex(ValueError, "global serialization"):
            qualification.membership(inventory, ["product"], "host")
        inventory["tests"][0]["properties"].pop()
        inventory["tests"][0]["properties"].append({"name": "RESOURCE_LOCK", "value": ["owned_resource"]})
        self.assertEqual(qualification.membership(inventory, ["product"], "host"), {"product": ["a"]})

    def test_explicit_scratch_roots_cannot_collide_across_selected_groups(self):
        inventory = self.inventory(("a", ["host", "product"]), ("b", ["host", "boundary"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inventory["tests"][0]["command"] = ["case", "--scratch-root", str(root / "scratch/../shared")]
            inventory["tests"][1]["command"] = ["case", "--scratch-root", str(root / "shared")]
            with self.assertRaisesRegex(ValueError, "colliding scratch"):
                qualification.membership(inventory, ["product", "boundary"], "all")
            inventory["tests"][1]["command"][-1] = str(root / "other")
            result = qualification.membership(inventory, ["product", "boundary"], "all")
            self.assertEqual(result, {"product": ["a"], "boundary": ["b"]})
            for arguments in (["--scratch-root", "shared"], ["--scratch-root"],
                              ["--scratch-root", str(root / "other"), "--scratch-root", str(root / "shared")]):
                inventory["tests"][1]["command"] = ["case", *arguments]
                with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "scratch destination"):
                    qualification.membership(inventory, ["product", "boundary"], "all")

    def test_ctest_exit_zero_cannot_hide_missing_skipped_or_unrun_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "tests.xml"
            for body in ('', '<testcase name="required" status="notrun"/>',
                         '<testcase name="required" status="run"><skipped/></testcase>',
                         '<testcase name="required" status="run"><failure/></testcase>'):
                report.write_text('<testsuite>' + body + '</testsuite>')
                with self.subTest(body=body), self.assertRaisesRegex(ValueError, "did not execute"):
                    qualification.validate_ctest_report(report, ["required"])
            report.write_text('<testsuite><testcase name="required" status="run"/></testsuite>')
            qualification.validate_ctest_report(report, ["required"])

    def ci_jobs(self, native):
        text = (ROOT / f".github/workflows/tests-{native}-host.yml").read_text()
        body = text.split("\njobs:\n", 1)[1]
        starts = list(re.finditer(r"^  ([a-z][a-z0-9-]*):$", body, re.M))
        jobs = {match[1]: body[match.end():starts[index + 1].start() if index + 1 < len(starts) else len(body)]
                for index, match in enumerate(starts)}
        return text, jobs

    def test_native_ci_preserves_independent_product_and_tooling_coverage(self):
        for native, family in (("linux", "linux"), ("windows", "windows-clang")):
            text, jobs = self.ci_jobs(native)
            children = {f"{native}-{lane}" for lane in ("product-debug", "product-release", "analysis", "tooling")}
            self.assertEqual(set(jobs), children | {f"{native}-debug-host"})
            self.assertIn("cancel-in-progress: true", text)
            for trigger in ("workflow_dispatch:", "pull_request:", "push:"):
                self.assertIn(trigger, text)
            self.assertNotRegex(text, re.compile(r"^    paths(?:-ignore)?:", re.M))
            for name in children:
                self.assertNotRegex(jobs[name], re.compile(r"^    (?:needs|if|strategy):", re.M))
                self.assertIn("fetch-depth: 0", jobs[name])
                self.assertIn("tests/ctest/control_artifacts.py --preset", jobs[name])
                self.assertIn("include-hidden-files: true", jobs[name])
            for profile in ("debug", "release"):
                block = jobs[f"{native}-product-{profile}"]
                preset = f"{family}-{profile}"
                self.assertRegex(block, re.compile(rf"^          cmake --build --preset {preset}$", re.M))
                labels = "product|boundary|corpus" if (native, profile) == ("linux", "debug") else "product|boundary"
                selections = re.findall(rf'ctest --preset {preset}(?: -N)? -L "([^"\n]+)" -LE gpu', block)
                self.assertEqual(selections, [labels, labels])
                self.assertIn("--no-tests=error", block)
                self.assertIn(f"out/validation/{preset}/quality/rust-naming/**", block)
                self.assertIn(f"out/validation/{preset}/quality/rust-boundaries/**", block)
                if native == "linux":
                    # NativeBoundary inspects real ordinary objects in both product lanes.
                    self.assertIn("llvm-toolchain-noble-22 main", block)
                    self.assertRegex(block, r"sudo apt-get install[^\n]*\bllvm-22\b")
                    self.assertIn("llvm-nm-22 --version", block)
                if native == "windows":
                    self.assertIn(f"out/build/{preset}/juicer.ofx", block)
                    for operation in ("NativeLibrary]::Load", "TryGetExport", "OfxGetNumberOfPlugins", "OfxGetPlugin", "NativeLibrary]::Free"):
                        self.assertIn(operation, block)
            tooling = jobs[f"{native}-tooling"]
            self.assertIn(f"cmake -S tests/tooling -B out/build/tooling-{native}", tooling)
            self.assertIn(f"ctest --test-dir out/build/tooling-{native} -L tooling", tooling)
            self.assertIn("--no-tests=error", tooling)
            self.assertIn(f"out/validation/{family}-debug", tooling)
            self.assertNotIn("Install CUDA", tooling)
            self.assertNotIn("cmake --preset", tooling)
            self.assertIn("Report cumulative verifier scope (skips disabled)", tooling)
            self.assertIn("github.event.pull_request.base.sha || github.event.before", tooling)
            self.assertIn("SCOPE_CANDIDATE: ${{ github.sha }}", tooling)
            self.assertIn("--verifier-scope-output", tooling)
            self.assertIn("subprocess.run(command, check=True)", tooling)
            self.assertIn("verifier-scope.json", tooling)
            for obsolete in ("--probe-manifest", "JUICER_RUST_PROBE_MANIFEST", "probe_evidence.py"):
                self.assertNotIn(obsolete, text)

    def test_native_ci_analysis_owns_cumulative_debug_and_release_quality(self):
        for native, family in (("linux", "linux"), ("windows", "windows-clang")):
            _, jobs = self.ci_jobs(native)
            block = jobs[f"{native}-analysis"]
            self.assertIn(f"cmake --build --preset {family}-debug", block)
            self.assertIn(f"cmake --preset {family}-release -DBUILD_TESTING=ON", block)
            self.assertIn(f"cmake --build --preset {family}-release --target juicer_rust_build", block)
            self.assertIn("github.event.pull_request.base.sha", block)
            self.assertIn("github.event.before", block)
            self.assertIn("git cat-file -e", block)
            self.assertNotIn("HEAD^", block)
            for selection in ("--base", "--all-owned"):
                self.assertRegex(block, rf"scripts/check-quality.py {selection}[^\n]*--preset {family}-debug --jobs 2")
            self.assertIn(f"scripts/check-quality.py --files Cargo.toml --preset {family}-release --jobs 2", block)
            for profile in ("debug", "release"):
                self.assertIn(f"out/validation/{family}-{profile}/quality/**/*.log", block)
                self.assertIn(f"out/build/{family}-{profile}/compile_commands.json", block)

    def test_required_ci_aggregates_reject_every_non_success_and_missing_child(self):
        for native, display in (("linux", "Linux"), ("windows", "Windows")):
            _, jobs = self.ci_jobs(native)
            aggregate = jobs[f"{native}-debug-host"]
            expected = {f"{native}-{lane}" for lane in ("product-debug", "product-release", "analysis", "tooling")}
            self.assertIn(f"name: {display} Debug host and Python checks", aggregate)
            self.assertIn("    if: always()", aggregate)
            needs = set(re.findall(r"^      - ([a-z][a-z0-9-]*)$", aggregate, re.M))
            self.assertEqual(needs, expected)
            self.assertIn("REQUIRED_RESULTS: ${{ toJSON(needs) }}", aggregate)
            self.assertIn("shell: python", aggregate)
            # Execute the actual inline gate, not a separately implemented truth table.
            code = textwrap.dedent(aggregate.split("        run: |\n", 1)[1])
            success = {name: {"result": "success"} for name in expected}
            def run(document):
                return subprocess.run([sys.executable, "-c", code],
                    env={**os.environ, "REQUIRED_RESULTS": json.dumps(document)},
                    capture_output=True, text=True, timeout=10).returncode
            self.assertEqual(run(success), 0)
            # Report-only decisions never authorize a skipped lane, even when
            # the complete same-candidate classification says it is unnecessary.
            for decision in ({"complete": True, "required": False, "candidate": "current"},
                             {"complete": True, "required": True, "candidate": "current"},
                             {"complete": False}, {}, None):
                tooling = {"result": "skipped", "outputs": {"classification": decision}}
                self.assertNotEqual(run({**success, f"{native}-tooling": tooling}), 0)
            for name in sorted(expected):
                for status in ("failure", "cancelled", "timed_out", "skipped", "", None):
                    with self.subTest(native=native, child=name, status=status):
                        self.assertNotEqual(run({**success, name: {"result": status}}), 0)
                self.assertNotEqual(run({key: value for key, value in success.items() if key != name}), 0)
                self.assertNotEqual(run({**success, name: {}}), 0)
            for malformed in ({}, {**success, "unexpected-child": {"result": "success"}}, [], None):
                self.assertNotEqual(run(malformed), 0)

    def test_malformed_checkpoint_selection_and_empty_collection_schedule_no_work(self):
        import verify
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        with patch.object(qualification, "OutputLease") as lease:
            for changes in ({"reason": ""}, {"selection": []}, {"selection": ["unknown"]},
                            {"preset": "unsupported-debug"}, {"runtime": "unsupported"}):
                args = SimpleNamespace(**({"reason": "requested checkpoint", "selection": ["boundary"],
                                          "preset": preset, "runtime": "host"} | changes))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    qualification.qualify(ROOT, args)
            with self.assertRaisesRegex(ValueError, "checkpoint receipts are required"):
                qualification.collect_qualification(ROOT, [])
            argv = ["verify.py", "qualification", "--preset", preset]
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as stopped:
                verify.main()
            self.assertEqual(stopped.exception.code, 2)
            lease.assert_not_called()

    def test_partial_development_and_failed_results_cannot_supply_checkpoint(self):
        for record in ({"mode": "development", "status": "passed"}, {"mode": "checkpoint", "status": "failed"},
                       {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "partial", "required": ["boundary"], "stages": {}}):
            with self.assertRaises(ValueError):
                qualification.require_checkpoint(record)


if __name__ == "__main__":
    unittest.main()
