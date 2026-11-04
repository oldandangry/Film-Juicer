"""Focused source selection and public verification CLI dispatch contracts."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify


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
            self.assertEqual(verify.main(), 0)
        args = collect.call_args.args[0]
        self.assertEqual(args.bundles, [Path("bundle-a"), Path("bundle-b")])
        self.assertEqual(args.output, Path("aggregate.json"))

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
