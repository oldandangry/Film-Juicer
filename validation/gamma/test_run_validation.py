from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_validation as runner


class FusedEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = [{"path": "/frozen/capture.json", "sha256": "a" * 64}]
        self.frozen = {
            "case": "paired-case",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": 0.5,
        }

    def compare(self, candidate: dict):
        return runner.compare_fused_case(
            candidate,
            self.frozen,
            "paired-case",
            self.source,
            "paired-case",
        )

    def test_equal_bound_passes_and_next_double_fails(self) -> None:
        passing = {
            "case": "paired-case",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": 0.5,
        }
        result, failures = self.compare(passing)
        self.assertEqual([], failures)
        self.assertEqual("passed", result["status"])

        failing = dict(passing)
        failing["fused_maximum_absolute_difference"] = math.nextafter(
            0.5, math.inf
        )
        result, failures = self.compare(failing)
        self.assertTrue(failures)
        self.assertEqual("failed", result["status"])

    def test_zero_bound_rejects_positive_difference(self) -> None:
        frozen = dict(self.frozen)
        frozen["fused_maximum_absolute_difference"] = 0.0
        candidate = {
            "case": "paired-case",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": math.nextafter(0.0, math.inf),
        }
        result, failures = runner.compare_fused_case(
            candidate,
            frozen,
            "paired-case",
            self.source,
            "paired-case",
        )
        self.assertTrue(failures)
        self.assertEqual(0.0, result["frozen_bound"])

    def test_malformed_and_changed_candidates_fail(self) -> None:
        valid = {
            "case": "paired-case",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": 0.25,
        }
        malformed = []
        for value in (math.nan, math.inf):
            row = dict(valid)
            row["fused_maximum_absolute_difference"] = value
            malformed.append(row)
        missing_difference = dict(valid)
        del missing_difference["fused_maximum_absolute_difference"]
        malformed.append(missing_difference)
        missing_applicability = dict(valid)
        del missing_applicability["fused_path_applicable"]
        malformed.append(missing_applicability)
        wrong_case = dict(valid)
        wrong_case["case"] = "wrong-case"
        malformed.append(wrong_case)
        changed_applicability = dict(valid)
        changed_applicability["fused_path_applicable"] = False
        changed_applicability["fused_maximum_absolute_difference"] = None
        malformed.append(changed_applicability)

        for index, candidate in enumerate(malformed):
            with self.subTest(index=index):
                result, failures = self.compare(candidate)
                self.assertTrue(failures)
                self.assertEqual("failed", result["status"])

    def test_non_applicable_diffusion_is_explicit(self) -> None:
        candidate = {
            "case": "enlarger-diffusion-black-pro-mist-half",
            "fused_path_applicable": False,
            "fused_maximum_absolute_difference": None,
        }
        frozen = {
            "case": "enlarger-diffusion-black-pro-mist-half",
            "fused_path_applicable": False,
            "fused_maximum_absolute_difference": 0.0,
        }
        result, failures = runner.compare_fused_case(
            candidate,
            frozen,
            candidate["case"],
            self.source,
            "diffusion",
        )
        self.assertEqual([], failures)
        self.assertEqual("not-applicable", result["status"])
        self.assertIsNone(result["candidate_difference"])

    def test_corrupt_fused_result_fails_with_unchanged_reference_metrics(self) -> None:
        candidate = {
            "case": "paired-case",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": 1.0,
            "metrics": {"reference": {"maximum_absolute_error": 0.0}},
        }
        result, failures = self.compare(candidate)
        self.assertTrue(failures)
        self.assertEqual(1.0, result["candidate_difference"])

    def test_legacy_applicability_is_limited_to_known_direct_cases(self) -> None:
        candidate = {
            "case": "negative-direct",
            "fused_path_applicable": True,
            "fused_maximum_absolute_difference": 0.0,
        }
        frozen = {
            "case": "negative-direct",
            "fused_maximum_absolute_difference": 0.0,
        }
        result, failures = runner.compare_fused_case(
            candidate,
            frozen,
            "negative-direct",
            self.source,
            "negative-direct",
        )
        self.assertEqual([], failures)
        self.assertEqual("passed", result["status"])

        frozen["case"] = "unknown-direct"
        candidate["case"] = "unknown-direct"
        result, failures = runner.compare_fused_case(
            candidate,
            frozen,
            "unknown-direct",
            self.source,
            "unknown-direct",
        )
        self.assertTrue(failures)
        self.assertEqual("failed", result["status"])


class FrozenEvidenceReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_root = Path(__file__).resolve().parent / "fixtures"
        cls.baseline, _ = runner.verify_frozen_baseline(cls.fixture_root)
        cls.route_probe, _ = runner.verify_frozen_route_probe(cls.fixture_root)

    def test_actual_historical_capture_layouts_are_verified_and_immutable(self) -> None:
        before = runner.tree_manifest(self.fixture_root)
        baseline_capture, baseline_source = runner.read_verified_baseline_capture(
            self.fixture_root,
            self.baseline,
            "Release-Clang",
            "default-paper-1",
            "default-paper",
        )
        route_capture, route_source = runner.read_verified_route_probe_capture(
            self.fixture_root,
            self.route_probe,
            "Release-Clang",
        )
        self.assertEqual("default-paper", baseline_capture["case"])
        self.assertEqual("route-probe", route_capture["case"])
        self.assertEqual(64, len(baseline_source["sha256"]))
        self.assertEqual(64, len(route_source["sha256"]))
        self.assertEqual(before, runner.tree_manifest(self.fixture_root))

    def test_wrong_configuration_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            runner.read_verified_baseline_capture(
                self.fixture_root,
                self.baseline,
                "Debug",
                "default-paper-1",
                "default-paper",
            )

    def test_output_inside_fixture_tree_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture_root = Path(temporary) / "fixtures"
            fixture_root.mkdir()
            with self.assertRaises(RuntimeError):
                runner.validate_fresh_output_root(
                    fixture_root / "candidate",
                    fixture_root,
                    "candidate",
                )


class RuntimeResourceProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_root = Path(__file__).resolve().parent / "fixtures"
        cls.baseline, _ = runner.verify_frozen_baseline(cls.fixture_root)
        cls.route_probe, _ = runner.verify_frozen_route_probe(cls.fixture_root)

    def test_frozen_ordinary_state_and_route_probe_inventories_agree(self) -> None:
        ordinary = runner.resolve_candidate_runtime_resources(
            self.fixture_root,
            self.baseline,
            "Release-Clang",
            "default-paper",
        )
        state = runner.resolve_candidate_runtime_resources(
            self.fixture_root,
            self.baseline,
            "Release-Clang",
            "state",
        )
        route_probe = runner.resolve_candidate_runtime_resources(
            self.fixture_root,
            self.baseline,
            "Release-Clang",
            "candidate-route-probe",
            self.route_probe,
        )
        self.assertEqual(56, len(ordinary["inventory"]))
        self.assertEqual(ordinary["inventory"], state["inventory"])
        self.assertEqual(ordinary["inventory"], route_probe["inventory"])
        self.assertTrue(ordinary["authoritative_sources"])
        self.assertTrue(route_probe["authoritative_sources"])

    def test_exact_inventory_stages_for_both_candidate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "GammaValidation.exe"
            executable.write_bytes(b"candidate-binary")
            resources = root / "Resources"
            resources.mkdir()
            (resources / "profile.json").write_bytes(b"profile")
            expected = runner.tree_manifest(resources)
            for index, label in enumerate(("compare", "candidate-route-probe")):
                staged = runner.stage_runtime(
                    executable,
                    resources,
                    root / f"runtime-{index}",
                    expected_resources=expected,
                    runtime_label=label,
                )
                self.assertEqual(expected, staged[2])
                self.assertEqual(
                    staged[3]["source_pre_launch_sha256"],
                    staged[3]["staged_pre_launch_sha256"],
                )

    def test_resource_mismatches_fail_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "GammaValidation.exe"
            executable.write_bytes(b"candidate-binary")
            expected_root = root / "expected"
            expected_root.mkdir()
            (expected_root / "profile.json").write_bytes(b"profile")
            (expected_root / "illuminant.csv").write_bytes(b"light")
            expected = runner.tree_manifest(expected_root)

            variants = {
                "changed-same-size": {
                    "profile.json": b"PROFILE",
                    "illuminant.csv": b"light",
                },
                "extra": {
                    "profile.json": b"profile",
                    "illuminant.csv": b"light",
                    "extra.txt": b"extra",
                },
                "missing": {"profile.json": b"profile"},
            }
            wrong_pinned = {
                name: dict(identity) for name, identity in expected.items()
            }
            wrong_pinned["profile.json"]["sha256"] = "0" * 64
            variants["wrong-pinned"] = {
                "profile.json": b"profile",
                "illuminant.csv": b"light",
            }

            for index, (name, files) in enumerate(variants.items()):
                with self.subTest(name=name):
                    resources = root / f"resources-{index}"
                    resources.mkdir()
                    for relative, contents in files.items():
                        (resources / relative).write_bytes(contents)
                    selected_expected = (
                        wrong_pinned if name == "wrong-pinned" else expected
                    )
                    with mock.patch.object(runner.subprocess, "run") as launch:
                        with self.assertRaises(RuntimeError):
                            runner.stage_runtime(
                                executable,
                                resources,
                                root / f"runtime-fail-{index}",
                                expected_resources=selected_expected,
                                runtime_label=(
                                    "candidate-route-probe"
                                    if index % 2
                                    else "compare"
                                ),
                            )
                        launch.assert_not_called()

    def test_inconsistent_frozen_sources_are_rejected(self) -> None:
        inventory = {
            "profile.json": {"sha256": "1" * 64, "bytes": 7}
        }
        changed = {
            "profile.json": {"sha256": "2" * 64, "bytes": 7}
        }
        with self.assertRaises(RuntimeError):
            runner.choose_runtime_resource_inventory(
                [
                    {"path": "/frozen/a.json", "sha256": "a" * 64, "inventory": inventory},
                    {"path": "/frozen/b.json", "sha256": "b" * 64, "inventory": changed},
                ],
                "frozen reports",
            )

    def test_post_stage_executable_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "GammaValidation.exe"
            executable.write_bytes(b"candidate-binary")
            resources = root / "Resources"
            resources.mkdir()
            (resources / "profile.json").write_bytes(b"profile")
            launched, _, _, identity = runner.stage_runtime(
                executable,
                resources,
                root / "runtime",
                expected_resources=runner.tree_manifest(resources),
                runtime_label="compare",
            )
            launched.write_bytes(b"CANDIDATE-BINARY")
            with self.assertRaises(RuntimeError):
                runner.verify_runtime_executable_after_run(
                    executable,
                    launched,
                    identity,
                    "compare",
                )

    def test_ordinary_resource_mismatch_cannot_reach_mocked_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable_root = root / "Release-Clang"
            executable_root.mkdir()
            executable = executable_root / "GammaValidation.exe"
            executable.write_bytes(b"candidate-binary")
            resources = root / "Resources"
            resources.mkdir()
            (resources / "wrong.json").write_bytes(b"wrong")
            arguments = runner.argparse.Namespace(
                exe=executable,
                resource_root=resources,
                fixture_root=self.fixture_root,
                output_root=root / "ordinary-output",
                case_group="default-paper",
            )
            with (
                mock.patch.object(runner.subprocess, "run") as launch,
                mock.patch.object(runner, "compare_metric_tree") as comparison,
            ):
                with self.assertRaises(RuntimeError):
                    runner.run_compare(arguments)
                launch.assert_not_called()
                comparison.assert_not_called()

    def test_route_probe_resource_mismatch_cannot_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable_root = root / "Release-Clang"
            executable_root.mkdir()
            executable = executable_root / "GammaValidation.exe"
            executable.write_bytes(b"candidate-binary")
            resources = root / "Resources"
            resources.mkdir()
            (resources / "wrong.json").write_bytes(b"wrong")
            arguments = runner.argparse.Namespace(
                exe=executable,
                resource_root=resources,
                fixture_root=self.fixture_root,
                output_root=root / "route-output",
                candidate=True,
            )
            with (
                mock.patch.object(runner.subprocess, "run") as launch,
                mock.patch.object(
                    runner,
                    "windows_path",
                    side_effect=lambda path: str(path.resolve()),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    runner.run_route_probe(arguments)
                launch.assert_not_called()


class EffectiveSettingsContractTests(unittest.TestCase):
    @staticmethod
    def valid_capture() -> dict:
        common = {
            "dir_active": True,
            "dir_diffusion_size_um": 0.0,
            "spatial_dir_active": False,
            "scatter_active": False,
            "halation_active": False,
            "camera_diffusion_active": False,
            "enlarger_diffusion_active": False,
            "scanner_black_correction": False,
            "scanner_white_correction": False,
            "scanner_lens_blur_sigma_px": 0.0,
            "scanner_unsharp_sigma_px": 0.0,
            "scanner_unsharp_amount": 0.0,
        }
        correction = {
            **common,
            "scanner_black_correction": True,
            "scanner_white_correction": True,
        }
        diffusion = {
            **common,
            "enlarger_diffusion_active": True,
            "enlarger_diffusion_family": "black_pro_mist",
            "enlarger_diffusion_scatter_fraction": 0.35 * 0.75,
            "enlarger_diffusion_spatial_scale": 1.0,
        }
        spatial_dir = {
            **common,
            "dir_diffusion_size_um": 2187.5,
            "spatial_dir_active": True,
        }
        return {
            "case": "routes",
            "routes": [
                {
                    "case": "scanner-black-white-correction-on",
                    "settings": correction,
                },
                {
                    "case": "enlarger-diffusion-black-pro-mist-half",
                    "settings": diffusion,
                    "enlarger_diffusion_active": True,
                    "print_log_exposure_boundary": "print_development_input",
                    "print_log_exposure_source": "post_diffusion_planes",
                    "print_development_observer_exact": True,
                    "print_development_observer_padded_stride_verified": True,
                    "fused_path_applicable": False,
                    "fused_maximum_absolute_difference": None,
                },
                {
                    "case": "spatial-dir-ramp-edge-direct",
                    "settings": spatial_dir,
                },
            ],
        }

    def test_required_g09_effective_settings_pass(self) -> None:
        self.assertEqual([], runner.contract_failures("routes", self.valid_capture()))

    def test_hardcoded_scanner_corrections_fail(self) -> None:
        capture = self.valid_capture()
        correction = capture["routes"][0]["settings"]
        correction["scanner_black_correction"] = False
        correction["scanner_white_correction"] = False
        failures = runner.contract_failures("routes", capture)
        self.assertTrue(any("scanner_black_correction" in value for value in failures))
        self.assertTrue(any("scanner_white_correction" in value for value in failures))

    def test_spatial_cases_reject_missing_activation_and_unrelated_effects(self) -> None:
        capture = self.valid_capture()
        diffusion = capture["routes"][1]["settings"]
        diffusion["enlarger_diffusion_active"] = False
        spatial_dir = capture["routes"][2]["settings"]
        spatial_dir["spatial_dir_active"] = False
        spatial_dir["halation_active"] = True
        spatial_dir["scanner_unsharp_sigma_px"] = 0.7
        spatial_dir["scanner_unsharp_amount"] = 0.7
        failures = runner.contract_failures("routes", capture)
        self.assertTrue(any("enlarger_diffusion_active" in value for value in failures))
        self.assertTrue(any("spatial_dir_active" in value for value in failures))
        self.assertTrue(any("halation_active" in value for value in failures))
        self.assertTrue(any("scanner_unsharp_sigma_px" in value for value in failures))

    def test_default_paper_rejects_active_corrections_or_unrelated_effects(self) -> None:
        settings = self.valid_capture()["routes"][0]["settings"]
        settings["scanner_black_correction"] = False
        settings["scanner_white_correction"] = False
        default_capture = {"case": "default-paper", "settings": settings}
        self.assertEqual(
            [], runner.contract_failures("default-paper", default_capture)
        )
        settings["scanner_black_correction"] = True
        settings["halation_active"] = True
        failures = runner.contract_failures("default-paper", default_capture)
        self.assertTrue(any("scanner_black_correction" in value for value in failures))
        self.assertTrue(any("halation_active" in value for value in failures))


class HistoricalReferenceValidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_root = Path(__file__).resolve().parent / "fixtures"
        cls.manifest_sha256 = runner.sha256_file(
            cls.fixture_root / "manifest.json"
        )

    @staticmethod
    def zero_metrics(*boundaries: str) -> dict:
        return {
            boundary: {
                channel: {
                    "maximum_absolute_error": 0.0,
                    "rmse": 0.0,
                }
                for channel in ("C", "M", "Y")
            }
            for boundary in boundaries
        }

    def test_all_zero_metrics_cannot_validate_known_bad_g09_cases(self) -> None:
        metrics = {
            "enlarger-diffusion-black-pro-mist-half": self.zero_metrics(
                "film_density",
                "print_density",
                "scanner_linear_rgb",
                "output_rgb",
            ),
            "spatial-dir-ramp-edge-direct": self.zero_metrics(
                "film_density",
                "scanner_linear_rgb",
                "output_rgb",
            ),
        }
        report, failures = runner.historical_reference_validity(
            "routes",
            self.fixture_root,
            self.manifest_sha256,
            metrics,
        )
        self.assertEqual("failed", report["status"])
        self.assertEqual(2, len(report["invalid_cases"]))
        self.assertEqual(2, len(failures))
        self.assertTrue(all("configuration-invalid" in value for value in failures))

    def test_unaffected_case_group_is_not_annotated(self) -> None:
        report, failures = runner.historical_reference_validity(
            "default-routes",
            self.fixture_root,
            self.manifest_sha256,
            {},
        )
        self.assertEqual("not-applicable", report["status"])
        self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main()
