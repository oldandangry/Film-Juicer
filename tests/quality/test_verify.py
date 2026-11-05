"""Focused source selection and public verification CLI dispatch contracts."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify
import execution


ROOT = Path(__file__).resolve().parents[2]


class DevelopmentSelectionTests(unittest.TestCase):
    def test_route_implementation_selects_its_named_owner(self):
        selection = verify.select_consumers(ROOT, ROOT / "absent-tree", ["src/ScanRoute.cpp"])
        self.assertEqual(selection["targets"], ["JuicerRouteBridgeTests"])
        self.assertEqual(selection["tests"], ["Route.Host.Bridge"])
        self.assertEqual(selection["compiler_consumers"], ["src/ScanRoute.cpp"])

    def test_route_header_selection_uses_the_supplied_compile_consumers(self):
        with patch.object(verify, "header_consumers", return_value=(
                ["CMakeFiles/JuicerRouteBridgeTests.dir/ScanRoute.cpp.o"], ["src/ScanRoute.cpp"])) as consumers:
            selection = verify.select_consumers(ROOT, ROOT / "absent-tree", ["src/ScanRoute.h"])
        consumers.assert_called_once()
        self.assertEqual(selection["targets"], ["CMakeFiles/JuicerRouteBridgeTests.dir/ScanRoute.cpp.o", "JuicerRouteBridgeTests"])
        self.assertEqual(selection["tests"], ["Route.Host.Bridge"])
        self.assertEqual(selection["compiler_consumers"], ["src/ScanRoute.cpp"])

    def test_route_rust_owner_selects_route_fixture_and_boundary_checks(self):
        selection = verify.select_consumers(ROOT, ROOT / "absent-tree",
                                            ["rust/film-juicer-core/src/route.rs"])
        self.assertEqual(selection["targets"], ["JuicerRouteBridgeTests"])
        self.assertEqual(selection["tests"], ["Quality.RustBoundaries", "Route.Host.Bridge"])
        self.assertIn("film-juicer-core", selection["rust_packages"])

    def test_output_color_rust_owner_selects_its_native_and_boundary_consumers(self):
        selection = verify.select_consumers(ROOT, ROOT / "absent-tree",
                                            ["rust/film-juicer-core/src/color/output.rs"])
        self.assertEqual(selection["targets"], ["JuicerOutputColorTests"])
        self.assertEqual(selection["tests"], ["Ffi.Host.OutputColor", "Quality.RustBoundaries"])
        self.assertIn("film-juicer-core", selection["rust_packages"])
        self.assertNotIn("Route.Host.Bridge", selection["tests"])

    def test_quality_command_does_not_forward_removed_rust_test_filters(self):
        selection = {"files": ["rust/film-juicer-core/src/route.rs"],
                     "compiler_consumers": [], "rust_packages": ["film-juicer-core"],
                     "rust_test_filters": ["route::"]}
        command = verify.quality_command("linux-debug", selection)
        self.assertNotIn("--rust-test-filters", command)


class PublicCliTests(unittest.TestCase):
    """The CLI delegates to fresh operation owners without reading old receipts."""

    def test_prepare_dispatches_targets_and_explicit_testing_mode(self):
        with patch.object(sys, "argv", ["verify.py", "prepare", "--preset", "linux-debug",
                                        "--targets", "JuicerRouteBridgeTests", "--testing", "ON",
                                        "--run-id", "prepare-one"]), \
                patch.object(verify, "fresh_prepare", return_value=0) as prepare:
            self.assertEqual(verify.main(), 0)
        args = prepare.call_args.args[0]
        self.assertEqual(args.preset, "linux-debug")
        self.assertEqual(args.targets, ["JuicerRouteBridgeTests"])
        self.assertEqual(args.testing, "ON")
        self.assertEqual(args.run_id, "prepare-one")

    def test_prepare_dispatches_cuda_free_mode(self):
        with patch.object(sys, "argv", ["verify.py", "prepare", "--preset", "linux-debug",
                                        "--targets", "JuicerHashContractTests", "--cuda-free"]), \
                patch.object(verify, "fresh_prepare", return_value=0) as prepare:
            self.assertEqual(verify.main(), 0)
        self.assertTrue(prepare.call_args.args[0].cuda_free)

    def test_development_dispatches_an_explicit_file_selection(self):
        with patch.object(sys, "argv", ["verify.py", "development", "--preset", "linux-debug",
                                        "--files", "src/ScanRoute.cpp"]), \
                patch.object(verify, "development", return_value=0) as development:
            self.assertEqual(verify.main(), 0)
        args = development.call_args.args[0]
        self.assertEqual(args.files, ["src/ScanRoute.cpp"])
        self.assertFalse(hasattr(args, "previous"))

    def test_development_dispatches_explicit_test_selection(self):
        with patch.object(sys, "argv", ["verify.py", "development", "--preset", "linux-debug",
                                        "--tests", "Route.Host.Bridge"]), \
                patch.object(verify, "development", return_value=0) as development:
            self.assertEqual(verify.main(), 0)
        self.assertEqual(development.call_args.args[0].tests, ["Route.Host.Bridge"])

    def test_development_dispatches_explicit_target_selection(self):
        with patch.object(sys, "argv", ["verify.py", "development", "--preset", "linux-debug",
                                        "--targets", "JuicerRouteBridgeTests"]), \
                patch.object(verify, "development", return_value=0) as development:
            self.assertEqual(verify.main(), 0)
        self.assertEqual(development.call_args.args[0].targets, ["JuicerRouteBridgeTests"])

    def test_development_dispatches_cuda_free_test_selection(self):
        with patch.object(sys, "argv", ["verify.py", "development", "--preset", "linux-debug",
                                        "--tests", "Route.Host.Bridge", "--cuda-free"]), \
                patch.object(verify, "development", return_value=0) as development:
            self.assertEqual(verify.main(), 0)
        self.assertTrue(development.call_args.args[0].cuda_free)

    def test_cuda_free_rejects_file_and_build_directory_selections(self):
        with patch.object(sys, "argv", ["verify.py", "development", "--preset", "linux-debug",
                                        "--files", "src/ScanRoute.cpp", "--cuda-free"]):
            with self.assertRaises(SystemExit):
                verify.main()
        with patch.object(sys, "argv", ["verify.py", "development", "--preset", "linux-debug",
                                        "--tests", "Route.Host.Bridge", "--build-dir", "out/build/linux-debug",
                                        "--cuda-free"]):
            with self.assertRaises(SystemExit):
                verify.main()
        with patch.object(sys, "argv", ["verify.py", "prepare", "--preset", "linux-debug",
                                        "--files", "src/ScanRoute.cpp", "--cuda-free"]):
            with self.assertRaises(SystemExit):
                verify.main()

    def test_check_owner_dispatches_canonical_preset(self):
        with patch.object(sys, "argv", ["verify.py", "check-owner", "--preset", "linux-debug"]), \
                patch.object(verify, "check_owner", return_value=0) as check_owner:
            self.assertEqual(verify.main(), 0)
        self.assertEqual(check_owner.call_args.args[0].preset, "linux-debug")

    def test_check_owner_requires_an_already_held_preset_lease(self):
        from execution import OutputLease
        args = type("Args", (), {"preset": "linux-debug"})()
        with patch.object(OutputLease, "check_held", side_effect=RuntimeError("no existing preset owner")):
            with self.assertRaisesRegex(RuntimeError, "no existing preset owner"):
                verify.check_owner(args)

    def test_qualification_dispatches_named_groups_and_runtime(self):
        with patch.object(sys, "argv", ["verify.py", "qualification", "--preset", "linux-debug",
                                        "--reason", "focused contract", "--selection", "boundary", "tooling",
                                        "--runtime", "all", "--build-dir", "out/build/linux-debug",
                                        "--label", "test-label", "--run-id", "run-one"]), \
                patch.object(verify, "native_tests", return_value=0) as native_tests:
            self.assertEqual(verify.main(), 0)
        args = native_tests.call_args.args[0]
        self.assertEqual(args.selection, ["boundary", "tooling"])
        self.assertEqual(args.runtime, "all")
        self.assertEqual(args.build_dir, Path("out/build/linux-debug"))
        self.assertEqual(args.label, "test-label")
        self.assertEqual(args.run_id, "run-one")

    def test_offline_collection_preserves_bundles_and_output_interface(self):
        with patch.object(sys, "argv", ["verify.py", "collect-qualification", "--bundles", "bundle-a", "bundle-b",
                                        "--output", "aggregate.json"]), \
                patch.object(verify, "collect_bundles", return_value=0) as collect:
            with patch.object(verify, "expire_diagnostics", side_effect=AssertionError("offline reader swept native outputs")):
                self.assertEqual(verify.main(), 0)
        args = collect.call_args.args[0]
        self.assertEqual(args.bundles, [Path("bundle-a"), Path("bundle-b")])
        self.assertEqual(args.output, Path("aggregate.json"))

    def test_closeout_dispatch_preserves_explicit_review_and_holds(self):
        with patch.object(sys, "argv", ["verify.py", "closeout", "--preset", "linux-debug", "--run", "run-one",
                                        "--accept", "--review", "closed", "--hold", "review", "reproduction",
                                        "--release-hold", "failure"]), \
                patch.object(verify, "closeout", return_value=0) as close:
            self.assertEqual(verify.main(), 0)
        args = close.call_args.args[0]
        self.assertEqual((args.run_id, args.accept, args.review), ("run-one", True, "closed"))
        self.assertEqual(args.hold, ["review", "reproduction"])
        self.assertEqual(args.release_hold, ["failure"])

    def test_closeout_changes_require_a_named_run(self):
        with patch.object(sys, "argv", ["verify.py", "closeout", "--preset", "linux-debug", "--accept"]), \
                self.assertRaises(SystemExit):
            verify.main()


class CloseoutCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="closeout cli ü ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        self.directory = self.root / "out/validation" / self.preset / "results/run-one"
        self.directory.mkdir(parents=True)
        candidate = json.dumps({"head": "fixture", "preset": self.preset, "inputs": {"source": "fixture"}}).encode()
        payloads = {"candidate.json": candidate, "candidate.patch": b"scoped fixture patch\n"}
        for name, data in payloads.items():
            (self.directory / name).write_bytes(data)
        self.result = {"schema": "native-build-v1", "preset": self.preset, "status": "passed", "scope": "build",
                       "shipping_isolation": False, "candidate_head": "fixture", "candidate_unchanged": True,
                       "commands": [{"name": "fixture", "returncode": 0, "drained": True, "process_status": "completed"}],
                       "members": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                                   for name, data in payloads.items()},
                       "diagnostics": {"path": "diagnostics.gz", "optional": True}}
        self.raw = json.dumps(self.result).encode()
        (self.directory / "result.json").write_bytes(self.raw)
        (self.directory / "diagnostics.gz").write_bytes(b"optional diagnostic fixture")
        self.enterContext(patch.object(verify, "ROOT", self.root))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def cli(self, *arguments):
        with patch.object(sys, "argv", ["verify.py", "closeout", "--preset", self.preset, *arguments]):
            return verify.main()

    def test_accept_hold_release_and_expiry_use_no_tools_and_preserve_required_bytes(self):
        with patch.object(execution.time, "time", return_value=100), \
                patch.object(subprocess, "Popen", side_effect=AssertionError("closeout launched a tool")):
            self.assertEqual(self.cli("--run", "run-one", "--accept", "--review", "closed", "--hold", "reproduction"), 0)
        with patch.object(execution.time, "time", return_value=100 + 7 * 86400):
            self.assertEqual(self.cli(), 0)
            self.assertTrue((self.directory / "diagnostics.gz").exists())
            self.assertEqual(self.cli("--run", "run-one", "--release-hold", "reproduction"), 0)
        self.assertFalse((self.directory / "diagnostics.gz").exists())
        self.assertEqual((self.directory / "result.json").read_bytes(), self.raw)
        from qualification import read_native_result
        self.assertEqual(read_native_result(self.directory)["status"], "passed")

    def test_deletion_error_is_reported_by_cli_and_keeps_the_diagnostic(self):
        with patch.object(execution.time, "time", return_value=100):
            self.assertEqual(self.cli("--run", "run-one", "--accept", "--review", "closed"), 0)
        unlink = Path.unlink
        def fail_diagnostic(path, *args, **kwargs):
            if path.name == "diagnostics.gz":
                raise PermissionError("controlled locked diagnostic")
            return unlink(path, *args, **kwargs)
        with patch.object(execution.time, "time", return_value=100 + 7 * 86400), \
                patch.object(Path, "unlink", fail_diagnostic):
            self.assertEqual(self.cli(), 1)
        self.assertTrue((self.directory / "diagnostics.gz").exists())

    def test_prepare_expires_after_owner_acquisition_before_any_native_command(self):
        events = []
        def expired(root, preset):
            self.assertEqual((root, preset), (self.root, self.preset))
            execution._preset_owner(root, preset)
            events.append("expiry")
            return {"read_bytes": 17}
        native = Mock()
        native.commands = []
        def command(*args, **kwargs):
            events.append(args[0])
            raise ValueError("controlled stop before native execution")
        native.command.side_effect = command
        args = SimpleNamespace(preset=self.preset, run_id="prepare-order", targets=["fixture"],
                               files=None, testing="ON", cuda_free=False)
        environment = dict(os.environ)
        environment.pop("FJ_HOST_BUNDLE", None)
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(verify, "expire_diagnostics", side_effect=expired), \
                patch.object(execution, "NativeRun", return_value=native), \
                patch.object(verify, "capture_native_source", side_effect=lambda *args, **kwargs: (
                    {"head": "fixture"}, b"", {"source_read_bytes": 0})), \
                patch.object(verify.shutil, "which", return_value="fixture-cmake"), \
                patch.object(verify, "windows_vctip", return_value=None), \
                patch.object(verify, "finish_native_run", return_value=1) as finish:
            self.assertEqual(verify.fresh_prepare(args), 1)
        self.assertEqual(events, ["expiry", "configure"])
        self.assertEqual(finish.call_args.args[3]["report_read_bytes"], 17)

    def test_removed_receipt_and_recovery_options_are_rejected(self):
        with patch.object(sys, "argv", ["verify.py", "development", "--preset", "linux-debug",
                                        "--files", "src/ScanRoute.cpp", "--previous", "old.json"]):
            with self.assertRaises(SystemExit):
                verify.main()
        with patch.object(sys, "argv", ["verify.py", "qualification", "--preset", "linux-debug",
                                        "--reason", "focused", "--selection", "boundary", "--runtime", "host",
                                        "--receipts", "old.json"]), \
                patch.object(verify, "native_tests", side_effect=AssertionError("invalid CLI dispatched")):
            with self.assertRaises(SystemExit):
                verify.main()
        with patch.object(sys, "argv", ["verify.py", "qualification", "--preset", "linux-debug",
                                        "--reason", "focused", "--selection", "boundary", "--runtime", "host",
                                        "--prepare"]), \
                patch.object(verify, "native_tests", side_effect=AssertionError("legacy build option dispatched")):
            with self.assertRaises(SystemExit):
                verify.main()
        with patch.object(sys, "argv", ["verify.py", "collect-qualification", "--bundles", "bundle-a",
                                        "--output", "report.json", "--windows-python", "python.exe"]), \
                patch.object(verify, "collect_bundles", side_effect=AssertionError("legacy interpreter option dispatched")):
            with self.assertRaises(SystemExit):
                verify.main()


if __name__ == "__main__":
    unittest.main()
