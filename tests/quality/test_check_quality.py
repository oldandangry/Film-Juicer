from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts/check-quality.py"
SPEC = importlib.util.spec_from_file_location("check_quality", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT_PATH}")
check_quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_quality
SPEC.loader.exec_module(check_quality)


class IsolatedOwnerTests(unittest.TestCase):
    """Dispatcher units own temporary output state; process tests cover admission."""

    def setUp(self):
        environment = os.environ.copy()
        environment.pop("FJ_HOST_BUNDLE", None)
        self.enterContext(patch.dict(os.environ, environment, clear=True))
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        import execution
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        lease_root = Path(directory.name).resolve()
        owner = execution.OutputLease

        class FixtureOutputLease(owner):
            def __init__(self, root, build=None, **arguments):
                super().__init__(lease_root, lease_root / "build" if build is not None else None, **arguments)

        replacement = patch.object(execution, "OutputLease", FixtureOutputLease)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.enterContext(patch.object(check_quality, "finish_quality_result", return_value=0))


class CheckQualityTests(IsolatedOwnerTests):
    def test_verifier_scope_is_conservative_for_dependencies_and_uncertainty(self):
        root = SCRIPT_PATH.parents[1]
        policy = check_quality.load_policy(root)
        head = "a" * 40
        path_cases = (
            ("src/Scanner.cpp", False), ("rust/film-juicer-core/src/optics/gaussian.rs", False),
            ("src/ScanRoute.cpp", True), ("src/RenderValues.cpp", True),
            ("rust/film-juicer-core/src/route.rs", True),
            ("rust/film-juicer-core/src/color/output.rs", True),
            ("tests/quality/test_checkpoint.py", True), ("scripts/execution.py", True),
            ("cmake/JuicerRust.cmake", True), ("CMakePresets.json", True),
            ("rust-toolchain.toml", True), ("Cargo.lock", True),
            (".github/workflows/tests-linux-host.yml", True),
            ("tests/ToolingTests.cmake", True), ("tests/ctest/control_artifacts.py", True),
            ("src/GeneratedColorSpaces.cpp", True), ("src/Scanner.h", True),
            ("unknown/input", True),
        )
        status_cases = (
            ("deleted", b"D\0src/Scanner.cpp\0", b"src/Scanner.cpp\0"),
            ("renamed", b"R100\0src/old.cpp\0src/new.cpp\0", b"src/old.cpp\0src/new.cpp\0"),
            ("added", b"A\0src/new.cpp\0", b"src/new.cpp\0"),
            ("malformed", b"M\0", b""),
        )
        base_cases = (
            ("missing-base", None, [head.encode()]),
            ("invalid-base", "absent", [head.encode(), check_quality.QualityError("missing base")]),
        )
        path_case_ids = tuple(f"path:{name}" for name, _ in path_cases)
        self.subtest_cases = (
            *path_case_ids,
            *(f"status:{case_id}" for case_id, _, _ in status_cases),
            *(case_id for case_id, _, _ in base_cases),
            "invalid-candidate",
        )
        for (name, required), case_id in zip(path_cases, path_case_ids, strict=True):
            with self.subTest(case=case_id, name=name), patch.object(check_quality, "run_git", side_effect=[
                    head.encode(), b"base", b"M\0" + name.encode() + b"\0", b""]):
                report = check_quality.verifier_scope(root, "pr-base", head, policy)
                self.assertEqual(report["required"], required)
                self.assertTrue(report["complete"])
                self.assertEqual(report["candidate"], head)
                self.assertEqual(report["mode"], "report-only")
        for case_id, payload, statuses in status_cases:
            with self.subTest(case=f"status:{case_id}"), patch.object(check_quality, "run_git", side_effect=[
                    head.encode(), b"base", payload, statuses]):
                self.assertTrue(check_quality.verifier_scope(root, "base", head, policy)["required"])
        for case_id, base, side_effect in base_cases:
            with self.subTest(case=case_id), patch.object(check_quality, "run_git", side_effect=side_effect):
                self.assertTrue(check_quality.verifier_scope(root, base, head, policy)["required"])
        with self.subTest(case="invalid-candidate"), patch.object(
            check_quality, "run_git", side_effect=check_quality.QualityError("invalid candidate"),
        ):
            with self.assertRaisesRegex(check_quality.QualityError, "invalid candidate"):
                check_quality.verifier_scope(root, "base", head, policy)

    def test_verifier_scope_compares_merge_base_to_candidate(self):
        root = SCRIPT_PATH.parents[1]
        self.subtest_cases = ("merge-base-scope",)
        with self.subTest(case=self.subtest_cases[0]), patch.object(
            check_quality, "run_git", side_effect=[b"head", b"ancestor", b"M\0src/Scanner.cpp\0", b""]
        ) as git:
            report = check_quality.verifier_scope(root, "pr-base", "candidate", check_quality.load_policy(root))
            self.assertFalse(report["required"])
            self.assertIn(unittest.mock.call(root, ["merge-base", "pr-base", "head"]), git.call_args_list)
            for call in git.call_args_list[2:]:
                self.assertEqual(call.args[1][-3:], ["ancestor", "head", "--"])

    def test_bash_source_policy_accepts_native_windows_crlf_rows(self) -> None:
        bash = r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else "/bin/bash"
        helper = SCRIPT_PATH.parent / "lib/style_common.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            python = root / "bin/python3"
            python.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'supported\\t.cpp\\r\\nheader\\t.h\\r\\ntranslation\\t.cpp\\r\\n"
                "ignored\\tdocs/\\r\\ngenerated\\tsrc/generated.cpp\\r\\n'\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            result = subprocess.run(
                [bash, "-c", 'export PATH="$PWD/bin:$PATH"; '
                 f"source '{helper.as_posix()}'; "
                 "style_is_supported_source src/example.cpp && "
                 "style_is_header_like src/example.h && "
                 "style_is_translation_unit src/example.cpp && "
                 "style_is_ignored_path docs/example.cpp && "
                 "style_is_ignored_path src/generated.cpp && "
                 "style_supported_extension_regex"],
                cwd=root, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            expression = result.stdout.strip()
            self.assertIsNotNone(re.fullmatch(expression, "src/example.cpp"))
            self.assertIsNone(re.fullmatch(expression, "src/example.txt"))

    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.enterContext(patch.object(
            check_quality, "create_run_directory",
            side_effect=lambda root, preset: Path(tempfile.mkdtemp(dir=directory.name)),
        ))
        self.enterContext(patch.object(check_quality, "capture_candidate", return_value=(
            {"head": "test-head", "inputs": {}, "new_files_base64": {}}, b"", {
                "source_capture_seconds": 0.0, "git_output_bytes": 0, "source_read_bytes": 0,
            },
        )))

    def test_configured_checkout_path_rejects_a_different_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache = root / "out/build/linux-debug/CMakeCache.txt"
            check_quality.check_build_path(root, "linux-debug")
            cache.parent.mkdir(parents=True)
            cache.write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={root}\n", encoding="utf-8")
            check_quality.check_build_path(root, "linux-debug")
            other = root.parent / "different-checkout"
            cache.write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={other}\n", encoding="utf-8")
            with self.assertRaisesRegex(check_quality.QualityError, "configured checkout path"):
                check_quality.check_build_path(root, "linux-debug")

    def test_name_status_preserves_spaces_and_rename_destination(self) -> None:
        selected, excluded = check_quality.parse_name_status(
            b"M\0native/a file.cpp\0R100\0src/old.cpp\0src/new name.cpp\0"
        )
        self.assertEqual(selected, ["native/a file.cpp", "src/new name.cpp"])
        self.assertEqual(excluded, ["renamed source: src/old.cpp"])

    def test_name_status_excludes_deleted_paths(self) -> None:
        selected, excluded = check_quality.parse_name_status(b"D\0src/gone.cpp\0")
        self.assertEqual(selected, [])
        self.assertEqual(excluded, ["deleted: src/gone.cpp"])

    def test_policy_selects_generated_rust_and_excludes_private_and_generated_native(self) -> None:
        root = SCRIPT_PATH.parent.parent
        policy = check_quality.load_policy(root)
        self.assertEqual(
            check_quality.exclusion_reason(".juicer-local/private.cpp", policy),
            "excluded prefix .juicer-local/",
        )
        self.assertEqual(
            check_quality.exclusion_reason("src/GeneratedColorSpaces.cpp", policy),
            "generated source",
        )
        self.assertIsNone(check_quality.exclusion_reason("native/a file.cpp", policy))
        generated_rust = "rust/film-juicer-plugin/src/cuda/sys.rs"
        self.assertIn(generated_rust, policy.generated_paths)
        self.assertIsNone(check_quality.exclusion_reason(generated_rust, policy))

    def test_generated_rust_dispatches_required_check_families(self) -> None:
        generated_rust = "rust/film-juicer-plugin/src/cuda/sys.rs"
        header = "native/juicer_cuda_api.h"
        cases = (
            ("generated-only", [generated_rust]),
            ("generated-and-doc", [generated_rust, "tests/ffi/README.md"]),
            ("native-header-and-generated", [header, generated_rust]),
        )
        self.subtest_cases = tuple(case_id for case_id, _ in cases)
        for case_id, files in cases:
            with (
                self.subTest(case=case_id, files=files),
                patch.object(sys, "argv", [str(SCRIPT_PATH), "--preset", "linux-debug", "--files", *files]),
                patch.object(check_quality, "Runner"),
                patch.object(check_quality, "check_build_path"),
                patch.object(check_quality, "check_source_hygiene"),
                patch.object(check_quality, "check_rust") as rust,
                patch.object(check_quality, "prepare_native_analysis", return_value=[]) as native,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(check_quality.main(), 0)
                rust.assert_called_once()
                self.assertEqual(rust.call_args.args[3], "x86_64-unknown-linux-gnu")
                if header in files:
                    native.assert_called_once()
                    self.assertEqual(native.call_args.args[3], [header])
                else:
                    native.assert_not_called()
                report = output.getvalue()
                self.assertIn(f"{generated_rust} (generated Rust; checked with workspace)", report)
                self.assertIn("Excluded categories:\n  none", report)
                self.assertNotIn("Documentation/configuration-only selection", report)

    def test_generated_rust_check_failure_stops_dispatch(self) -> None:
        generated_rust = "rust/film-juicer-plugin/src/cuda/sys.rs"
        cases = (
            ("generated-only", [generated_rust]),
            ("generated-and-doc", [generated_rust, "tests/ffi/README.md"]),
            ("native-header-and-generated", ["native/juicer_cuda_api.h", generated_rust]),
        )
        self.subtest_cases = tuple(case_id for case_id, _ in cases)
        for case_id, files in cases:
            with (
                self.subTest(case=case_id, files=files),
                patch.object(sys, "argv", [str(SCRIPT_PATH), "--preset", "linux-debug", "--files", *files]),
                patch.object(check_quality, "Runner"),
                patch.object(check_quality, "check_build_path"),
                patch.object(check_quality, "check_source_hygiene"),
                patch.object(check_quality, "check_rust", side_effect=check_quality.QualityError("Rust check failed")),
                patch.object(check_quality, "prepare_native_analysis", return_value=[]) as native,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                with self.assertRaisesRegex(check_quality.QualityError, "Rust check failed"):
                    check_quality.main()
                self.assertEqual(native.call_count, int("native/juicer_cuda_api.h" in files))
                self.assertNotIn("Quality checks passed", output.getvalue())

    def test_release_cargo_manifest_dispatches_full_rust_only(self) -> None:
        root = SCRIPT_PATH.parent.parent
        policy = check_quality.load_policy(root)
        cases = (
            ("linux-release", "linux-release", "x86_64-unknown-linux-gnu"),
            ("windows-clang-release", "windows-clang-release", "x86_64-pc-windows-msvc"),
        )
        self.subtest_cases = tuple(case_id for case_id, _, _ in cases)
        for case_id, preset, target in cases:
            with (
                self.subTest(case=case_id, preset=preset),
                patch.object(sys, "argv", [str(SCRIPT_PATH), "--preset", preset, "--files", "Cargo.toml"]),
                patch.object(check_quality, "check_source_hygiene"),
                patch.object(check_quality, "check_rust") as rust,
                patch.object(check_quality, "prepare_native_analysis") as native,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                arguments = check_quality.parse_arguments()
                runner = Mock(log_dir=root / "logs", name_prefix="")
                self.assertEqual(check_quality.run_checks(root, arguments, policy, arguments.files, [], runner), 0)
                rust.assert_called_once_with(root, runner, arguments, target)
                self.assertFalse(arguments.development)
                self.assertIsNone(arguments.rust_packages)
                native.assert_not_called()
                runner.run_analysis.assert_not_called()
                self.assertEqual([call.kwargs["label"] for call in runner.run.call_args_list],
                                 ["git-diff-check", "native-boundary-enforcement"])

    def test_policy_changes_cannot_skip_required_checks(self) -> None:
        cases = (
            ("workspace-manifest", "Cargo.toml"),
            ("member-manifest", "rust/film-juicer-core/Cargo.toml"),
            ("rust-boundary-test", "tests/quality/test_rust_boundaries.py"),
            ("quality-runner", "scripts/check-quality.py"),
        )
        self.subtest_cases = tuple(case_id for case_id, _ in cases)
        for case_id, path in cases:
            with (
                self.subTest(case=case_id, path=path),
                patch.object(sys, "argv", [str(SCRIPT_PATH), "--preset", "linux-debug", "--files", path]),
                patch.object(check_quality, "Runner") as runner,
                patch.object(check_quality, "check_build_path"),
                patch.object(check_quality, "check_source_hygiene"),
                patch.object(check_quality, "check_rust") as rust,
                patch.object(check_quality, "prepare_native_analysis", return_value=[]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(check_quality.main(), 0)
                rust.assert_called_once()
                labels = [call.kwargs["label"] for call in runner.return_value.run.call_args_list]
                self.assertIn("native-boundary-enforcement", labels)

    def test_native_boundary_failure_blocks_even_documentation_selection(self) -> None:
        with (
            patch.object(sys, "argv", [str(SCRIPT_PATH), "--preset", "linux-debug", "--files", "CONTRIBUTING.md"]),
            patch.object(check_quality, "Runner") as runner,
            patch.object(check_quality, "check_build_path"),
            patch.object(check_quality, "check_source_hygiene"),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            runner.return_value.run.side_effect = ["", check_quality.QualityError("boundary violation")]
            with self.assertRaisesRegex(check_quality.QualityError, "boundary violation"):
                check_quality.main()
            self.assertNotIn("Quality checks passed", output.getvalue())

    def test_edited_native_failure_blocks_rust_and_header_consumers(self) -> None:
        producer = check_quality.AnalysisCommand(["tool", "producer"], Path("."), "producer", "src/mainProcessing.cpp")
        consumer = check_quality.AnalysisCommand(["tool", "consumer"], Path("."), "consumer", "src/main.cpp")
        with (
            patch.object(sys, "argv", [str(SCRIPT_PATH), "--preset", "linux-debug", "--files",
                "src/mainProcessing.cpp", "native/juicer_legacy_api.h", "rust/film-juicer-core/src/exposure.rs"]),
            patch.object(check_quality, "Runner") as runner,
            patch.object(check_quality, "check_build_path"),
            patch.object(check_quality, "check_source_hygiene"),
            patch.object(check_quality, "prepare_native_analysis", return_value=[consumer, producer]),
            patch.object(check_quality, "check_rust") as rust,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            runner.return_value.run_analysis.side_effect = check_quality.QualityError("producer finding")
            with self.assertRaisesRegex(check_quality.QualityError, "producer finding"):
                check_quality.main()
            runner.return_value.run_analysis.assert_called_once_with([producer])
            rust.assert_not_called()

    def test_success_retains_all_header_consumers_and_cuda_variants(self) -> None:
        producer = check_quality.AnalysisCommand(["tool", "producer"], Path("."), "producer", "src/mainProcessing.cpp")
        consumers = [check_quality.AnalysisCommand(["tool", variant], Path("."), variant,
            "src/Cuda/Film/JuicerCudaFilmPipeline.cu") for variant in ("ordinary", "test")]
        events = []
        with (
            patch.object(sys, "argv", [str(SCRIPT_PATH), "--preset", "linux-debug", "--files",
                "src/mainProcessing.cpp", "native/juicer_legacy_api.h", "rust/film-juicer-core/src/exposure.rs"]),
            patch.object(check_quality, "Runner") as runner,
            patch.object(check_quality, "check_build_path"),
            patch.object(check_quality, "check_source_hygiene"),
            patch.object(check_quality, "prepare_native_analysis", return_value=[*consumers, producer]),
            patch.object(check_quality, "check_rust", side_effect=lambda *args: events.append("rust")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            runner.return_value.run_analysis.side_effect = lambda items: events.append(items)
            self.assertEqual(check_quality.main(), 0)
        self.assertEqual(events, [[producer], "rust", consumers])

    def test_explicit_qualification_keeps_complete_native_and_rust_gates(self):
        with (
            patch.object(sys, 'argv', [str(SCRIPT_PATH), '--preset', 'linux-debug', '--qualification', '--files', 'CONTRIBUTING.md']),
            patch.object(check_quality, 'Runner'), patch.object(check_quality, 'check_build_path'),
            patch.object(check_quality, 'check_source_hygiene'),
            patch.object(check_quality, 'check_rust') as rust,
            patch.object(check_quality, 'prepare_native_analysis', return_value=[]) as native,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(check_quality.main(), 0)
            rust.assert_called_once()
            self.assertEqual(set(native.call_args.args[3]), set(check_quality.REPRESENTATIVE_NATIVE_FILES))

    def test_rust_probe_profiles_select_the_preset_or_complete_matrix(self) -> None:
        profile_cases = (
            ("debug", (False,)),
            ("release", (True,)),
            ("both", (False, True)),
        )
        module_names = ("test_rust_naming", "test_rust_boundaries")
        cases = tuple(
            (f"{module_name}:{profile}", module_name, profile, expected)
            for module_name in module_names for profile, expected in profile_cases
        ) + tuple((f"{module_name}:invalid", module_name, "invalid", None) for module_name in module_names)
        self.subtest_cases = tuple(case_id for case_id, *_ in cases)
        for case_id, module_name, profile, expected in cases:
            module = __import__(module_name)
            suite = module.RustNamingTests if module_name.endswith("naming") else module.RustBoundaryTests
            with tempfile.TemporaryDirectory() as directory:
                settings = {"JUICER_CARGO": "cargo", "JUICER_RUST_TARGET": "x86_64-unknown-linux-gnu",
                    "JUICER_RUST_PROBE_DIR": directory + "/cache", "JUICER_TEST_ARTIFACT_DIR": directory + "/logs"}
                with self.subTest(case=case_id, module=module_name, profile=profile), patch.dict(
                    os.environ, {**settings, "JUICER_RUST_PROBE_PROFILE": profile},
                ):
                    if expected is None:
                        with self.assertRaises(KeyError):
                            suite.setUpClass()
                    else:
                        try:
                            suite.setUpClass()
                            self.assertEqual(suite.profiles, expected)
                            self.assertEqual(suite.target_dir, Path(settings["JUICER_RUST_PROBE_DIR"]).resolve())
                        finally:
                            suite.doClassCleanups()

    def test_boundary_feature_checks_keep_both_compiler_commands_isolated(self) -> None:
        module = __import__("test_rust_boundaries")
        probe = module.RustBoundaryTests("test_valid_profile_consumers_compile")
        with tempfile.TemporaryDirectory() as directory:
            probe.target_dir = Path(directory)
            probe.target = "x86_64-unknown-linux-gnu"
            with patch.object(probe, "command", return_value=subprocess.CompletedProcess([], 0, "", "")) as command:
                self.assertEqual(probe.check_features("film-juicer-plugin", True), [(0, []), (0, [])])
            arguments = [call.args[0] for call in command.call_args_list]
            self.assertEqual(len(arguments), 2)
            outputs = {}
            for args in arguments:
                self.assertIn("--all-targets", args)
                self.assertIn("--locked", args)
                self.assertIn("--offline", args)
                self.assertIn("--release", args)
                self.assertEqual(args[args.index("--target") + 1], probe.target)
                output = Path(args[args.index("--target-dir") + 1])
                self.assertEqual(output.parent, probe.target_dir)
                outputs[output.name] = args
            self.assertEqual(set(outputs), {"default", "test-support"})
            self.assertNotIn("--features", outputs["default"])
            self.assertEqual(outputs["test-support"][-2:], ["--features", "test-support"])

    def test_boundary_negative_checks_bind_code_marker_and_fixture(self) -> None:
        module = __import__("test_rust_boundaries")
        probe = module.RustBoundaryTests("test_valid_profile_consumers_compile")
        expected = {"E0616": ("color.rgb_to_xyz",)}
        valid = [{
            "level": "error",
            "code": {"code": "E0616"},
            "spans": [{
                "is_primary": True,
                "file_name": "/tmp/quality_boundary_probe_input_color_private.rs",
                "text": [{"text": "let _ = color.rgb_to_xyz;"}],
            }],
        }]

        # Compiler lints can add primary errors in the same fixture without
        # replacing the required ownership diagnostic.
        with_lint = valid + [{
            "level": "error",
            "code": {"code": "E0453"},
            "spans": [{
                "is_primary": True,
                "file_name": "/tmp/quality_boundary_probe_input_color_private.rs",
                "text": [{"text": "unsafe attribute used without unsafe"}],
            }],
        }]
        cases = (
            ("private-field", valid, "quality_boundary_probe_input_color_private.rs", True),
            ("additional-lint", with_lint, "quality_boundary_probe_input_color_private.rs", True),
            ("wrong-fixture", valid, "quality_boundary_probe_output_color_private.rs", False),
            ("wrong-code", [{**valid[0], "code": {"code": "E0599"}}],
             "quality_boundary_probe_input_color_private.rs", False),
            ("wrong-expression", [{**valid[0], "spans": [{**valid[0]["spans"][0],
             "text": [{"text": "let _ = color.hash;"}]}]}],
             "quality_boundary_probe_input_color_private.rs", False),
        )
        self.subtest_cases = tuple(case_id for case_id, *_ in cases)
        for case_id, diagnostics, fixture, accepted in cases:
            with self.subTest(case=case_id, fixture=fixture):
                if accepted:
                    probe.assert_rejected(1, diagnostics, expected, fixture)
                else:
                    with self.assertRaises(AssertionError):
                        probe.assert_rejected(1, diagnostics, expected, fixture)

    def test_boundary_fixture_batches_keep_success_and_failure_groups_separate(self) -> None:
        module = __import__("test_rust_boundaries")
        probe = module.RustBoundaryTests("test_valid_profile_consumers_compile")
        probe.profiles = (False,)
        identities = {
            "accepted_a": "quality_boundary_probe_accepted_a.rs",
            "accepted_b": "quality_boundary_probe_accepted_b.rs",
            "private_a": "quality_boundary_probe_private_a.rs",
            "private_b": "quality_boundary_probe_private_b.rs",
            "construct": "quality_boundary_probe_construct.rs",
        }
        expected = {"E0616": ("private_a.field",)}
        rejected = [{
            "level": "error",
            "code": {"code": "E0616"},
            "spans": [{
                "is_primary": True,
                "file_name": "/tmp/quality_boundary_probe_private_a.rs",
                "text": [{"text": "let _ = private_a.field;"}],
            }],
        }, {
            "level": "error",
            "code": {"code": "E0616"},
            "spans": [{
                "is_primary": True,
                "file_name": "/tmp/quality_boundary_probe_private_b.rs",
                "text": [{"text": "let _ = private_b.field;"}],
            }],
        }]
        construction_error = [{
            "level": "error",
            "code": {"code": "E0451"},
            "spans": [{
                "is_primary": True,
                "file_name": "/tmp/quality_boundary_probe_construct.rs",
                "text": [{"text": "Private { value: 1 }"}],
            }],
        }]
        fixtures = ("accepted_a", "accepted_b", "private_a", "private_b", "construct")
        feature_cases = ((False, "default"), (True, "test-support"))
        self.subtest_cases = tuple(
            f"{fixture}|debug|{variant}"
            for fixture in fixtures for _, variant in feature_cases
        )
        probe.subtest_cases = self.subtest_cases
        probe.subTest = lambda msg=None, **parameters: self.subTest(msg=msg, **parameters)
        with (
            patch.object(probe, "attach_many", side_effect=lambda owner, names: {
                name: identities[name] for name in names
            }) as attach,
            patch.object(probe, "check_features", side_effect=[
                [(0, []), (0, [])],
                [(1, rejected), (1, rejected)],
                [(1, construction_error), (1, construction_error)],
            ]) as compile_features,
        ):
            probe.check_fixture_cases(
                "film-juicer-plugin/src/lib.rs",
                "film-juicer-plugin",
                cases=(
                    ("accepted_a", None),
                    ("accepted_b", None),
                    ("private_a", expected),
                    ("private_b", {"E0616": ("private_b.field",)}),
                    ("construct", {"E0451": ("Private { value: 1 }",)}),
                ),
            )

        self.assertEqual(attach.call_count, 3)
        self.assertEqual(attach.call_args_list[0].args[1], ("accepted_a", "accepted_b"))
        self.assertEqual(attach.call_args_list[1].args[1], ("private_a", "private_b"))
        self.assertEqual(attach.call_args_list[2].args[1], ("construct",))
        self.assertEqual(compile_features.call_count, 3)

    def test_boundary_feature_failure_waits_for_the_other_command(self) -> None:
        module = __import__("test_rust_boundaries")
        probe = module.RustBoundaryTests("test_valid_profile_consumers_compile")
        started = threading.Barrier(2, timeout=5)
        failed = threading.Event()
        finished = threading.Event()

        def check(package, release, support):
            self.assertEqual((package, release), ("film-juicer-plugin", False))
            started.wait()
            if not support:
                failed.set()
                raise subprocess.TimeoutExpired("compiler", 60)
            self.assertTrue(failed.wait(timeout=5))
            finished.set()
            return 0, []

        with patch.object(probe, "check", side_effect=check) as command:
            with self.assertRaises(subprocess.TimeoutExpired):
                probe.check_features("film-juicer-plugin", False)
        self.assertTrue(finished.is_set())
        self.assertEqual(sorted(call.args[2] for call in command.call_args_list), [False, True])

    def test_cuda_header_selects_real_c_and_cpp_consumers(self) -> None:
        root = SCRIPT_PATH.parent.parent
        policy = check_quality.load_policy(root)
        entries = [
            check_quality.CompilationEntry("tests/ffi/cuda_abi_c.c", "cc -std=c11 -Inative"),
            check_quality.CompilationEntry("tests/ffi/cuda_abi_test.cpp", "c++ -std=c++20 -Inative"),
        ]
        self.assertEqual(
            check_quality.tidy_translation_units(root, ["native/juicer_cuda_api.h"], entries, policy),
            [entry.path for entry in entries],
        )

    def test_missing_tool_is_a_failure(self) -> None:
        with self.assertRaises(check_quality.QualityError):
            check_quality.resolve_tool(
                "/definitely/missing/juicer-tool",
                "JUICER_TEST_MISSING_TOOL",
                (),
            )

    def test_subprocess_failure_propagates_and_is_logged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = check_quality.Runner(root, root / "logs")
            with self.assertRaises(check_quality.QualityError):
                runner.run(
                    [sys.executable, "-c", "raise SystemExit(7)"],
                    label="expected-failure",
                )
            self.assertTrue((root / "logs/expected-failure.log").is_file())
            record, = runner.native_run.commands
            self.assertEqual(record["returncode"], 7)
            self.assertGreaterEqual(record["seconds"], 0)

    def test_member_lint_inheritance_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for member in ("film-juicer-core", "film-juicer-plugin"):
                member_root = root / "rust" / member
                (member_root / "src").mkdir(parents=True)
                (member_root / "Cargo.toml").write_text(
                    "[package]\nname = \"x\"\nversion = \"0.1.0\"\n",
                    encoding="utf-8",
                )
                (member_root / "src/lib.rs").write_text(
                    "#![forbid(unsafe_code)]\n", encoding="utf-8"
                )
            with self.assertRaises(check_quality.QualityError):
                check_quality.check_workspace_contract(root)


class CommandOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.runner = check_quality.Runner(self.root, self.root / "logs")

    def command(self, stdout: bytes, stderr: bytes, status: int = 0) -> list[str]:
        return [sys.executable, "-c",
                f"import os; os.write(2, {stderr!r}); os.write(1, {stdout!r}); raise SystemExit({status})"]

    def test_metadata_stdout_excludes_diagnostics_and_preserves_raw_bytes(self) -> None:
        stdout = b'{"workspace_members": []}\r\n'
        stderr = b'Blocking waiting for file lock on package cache\r\n\xff\n'
        with contextlib.redirect_stdout(io.StringIO()):
            output = self.runner.run(self.command(stdout, stderr), label="metadata",
                                     stdout_only=True, display_output=False)
        self.assertEqual(json.loads(output), {"workspace_members": []})
        self.assertEqual((self.runner.log_dir / "metadata.log").read_bytes(), stdout)
        self.assertEqual((self.runner.log_dir / "metadata.stderr.log").read_bytes(), stderr)
        self.assertEqual(self.runner.native_run.commands[0]["returncode"], 0)

    def test_empty_or_malformed_stdout_cannot_use_json_from_stderr(self) -> None:
        cases = (("empty", b""), ("not-json", b"not JSON"),
                 ("trailing-junk", b'{"ok": true}\ntrailing junk'))
        self.subtest_cases = tuple(case_id for case_id, _ in cases)
        for case_id, stdout in cases:
            with self.subTest(case=case_id, stdout=stdout), contextlib.redirect_stdout(io.StringIO()):
                output = self.runner.run(self.command(stdout, b'{"ok": true}'),
                                         label="bad-metadata", stdout_only=True)
                with self.assertRaises(json.JSONDecodeError):
                    json.loads(output)

    def test_failed_command_retains_both_streams_and_cannot_return_valid_json(self) -> None:
        stdout, stderr = b'{"ok": true}\n', b"intentional Cargo failure\n"
        with contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(check_quality.QualityError, "exit code 7"):
            self.runner.run(self.command(stdout, stderr, 7), label="failed-metadata", stdout_only=True)
        self.assertEqual((self.runner.log_dir / "failed-metadata.log").read_bytes(), stdout)
        self.assertEqual((self.runner.log_dir / "failed-metadata.stderr.log").read_bytes(), stderr)
        record = self.runner.native_run.commands[0]
        self.assertEqual(record["returncode"], 7)

    def test_large_stderr_is_drained_without_corrupting_stdout(self) -> None:
        stderr = b"package-cache diagnostic\n" * 8192
        command = [sys.executable, "-c",
                   'import os; os.write(2, b"package-cache diagnostic\\n" * 8192); os.write(1, b\'{"ok": true}\')']
        with contextlib.redirect_stdout(io.StringIO()):
            output = self.runner.run(command,
                                     label="large-stderr", stdout_only=True)
        self.assertEqual(json.loads(output), {"ok": True})
        self.assertEqual((self.runner.log_dir / "large-stderr.stderr.log").read_bytes(), stderr)

    def test_default_command_keeps_merged_diagnostic_behavior(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            output = self.runner.run(self.command(b"stdout\n", b"stderr\n"), label="ordinary")
        self.assertEqual(output, "stderr\nstdout")
        self.assertEqual(len(list(self.runner.log_dir.glob("*.log"))), 1)


class ParallelAnalysisTests(unittest.TestCase):
    def test_cli_requires_a_positive_worker_limit(self) -> None:
        base = [str(SCRIPT_PATH), "--preset", "linux-debug", "--files", "CONTRIBUTING.md"]
        explicit_cases = tuple((f"explicit-workers:{count}", count) for count in (1, 2, 4))
        default_cases = (("default-workers:none", None, 1), ("default-workers:one", 1, 1),
                         ("default-workers:two", 2, 2), ("default-workers:four", 4, 4),
                         ("default-workers:cap", 16, 4))
        invalid_cases = tuple((f"invalid-workers:{count}", count) for count in ("0", "-1", "invalid"))
        self.subtest_cases = (*[case_id for case_id, _ in explicit_cases],
                              *[case_id for case_id, _, _ in default_cases],
                              *[case_id for case_id, _ in invalid_cases])
        for case_id, count in explicit_cases:
            with self.subTest(case=case_id, count=count), patch.object(sys, "argv", [*base, "--jobs", str(count)]):
                self.assertEqual(check_quality.parse_arguments().jobs, count)
        for case_id, available, expected in default_cases:
            with self.subTest(case=case_id, available=available), patch.object(sys, "argv", base), \
                    patch.object(check_quality.os, "process_cpu_count", return_value=available):
                self.assertEqual(check_quality.parse_arguments().jobs, expected)
        for case_id, count in invalid_cases:
            with self.subTest(case=case_id, count=count), patch.object(sys, "argv", [*base, "--jobs", count]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                check_quality.parse_arguments()
            self.assertEqual(failure.exception.code, 2)

    def test_development_cannot_stage_probe_or_complete_qualification_receipts(self):
        base = [str(SCRIPT_PATH), '--preset', 'linux-debug', '--files', 'src/ScanRoute.cpp', '--development']
        cases = (
            ("qualification-flag", ["--qualification"]),
            ("removed-probe-flags", ["--probe-manifest", "unused.json", "--qualification-id", "01234567-89ab-4def-8012-3456789abcde"]),
        )
        self.subtest_cases = tuple(case_id for case_id, _ in cases)
        for case_id, extra in cases:
            with self.subTest(case=case_id, extra=extra), patch.object(sys, 'argv', [*base, *extra]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                check_quality.parse_arguments()


    def test_serial_and_parallel_commands_keep_arguments_directories_and_logs(self) -> None:
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        parent = Path(os.environ.get("JUICER_TEST_ARTIFACT_DIR",
            str(SCRIPT_PATH.parent.parent / "out/validation" / preset / "quality/parallel-controls")))
        parent.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=parent, delete=False)
        root = Path(directory.name).resolve()
        try:
            for jobs in (1, 2, 4):
                logs = root / str(jobs)
                runner = check_quality.Runner(root, logs, jobs)
                commands = [check_quality.AnalysisCommand(
                    [sys.executable, "-c", "import os,sys; print(os.getcwd()); print(sys.argv[1])", token],
                    root, "same-label",
                ) for token in ("one space", "two", "three", "four")]
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.run_analysis(commands)
                outputs = [path.read_text(encoding="utf-8").splitlines() for path in logs.glob("*.log")]
                self.assertEqual(len(outputs), 4)
                self.assertEqual({output[1] for output in outputs}, {item.command[-1] for item in commands})
                self.assertTrue(all(Path(output[0]).resolve() == root.resolve() for output in outputs))
        except BaseException:
            print("Retained parallel analysis control:", root, file=sys.stderr)
            raise
        else:
            directory.cleanup()

    def test_runner_failure_keeps_native_diagnostic_and_preserves_cause(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = check_quality.Runner(root, root / "logs", 1)
            commands = [check_quality.AnalysisCommand([sys.executable], root, "launch")]
            with patch.object(runner.native_run, "command", side_effect=RuntimeError("runner witness")), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaisesRegex(check_quality.QualityError, "runner witness"):
                runner.run_analysis(commands)

    def test_worker_limit_allows_overlap_without_exceeding_the_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = check_quality.Runner(root, root / "logs", 2)
            barrier = threading.Barrier(2, timeout=5)
            lock = threading.Lock()
            active = peak = 0

            def command(arguments, **kwargs):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    barrier.wait()
                    return subprocess.CompletedProcess(arguments, 0, stdout=arguments[-1] + "\n")
                finally:
                    with lock:
                        active -= 1

            commands = [check_quality.AnalysisCommand([sys.executable, str(i)], root, f"analysis-{i}") for i in range(4)]
            def native_command(name, arguments, **kwargs):
                output = command(arguments, **kwargs)
                path = root / "logs" / f"{name}.log"
                path.write_text(output.stdout, encoding="utf-8")
                return path

            with patch.object(runner.native_run, "command", side_effect=native_command), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run_analysis(commands)
            self.assertEqual(peak, 2)
            self.assertEqual(active, 0)
            self.assertEqual(len(list(runner.log_dir.glob("*.log"))), 4)

    def test_running_findings_are_observed_without_starting_queued_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = check_quality.Runner(root, root / "logs", 2)
            barrier = threading.Barrier(2, timeout=5)

            def command(arguments, **kwargs):
                status = int(arguments[-1])
                if status:
                    barrier.wait()
                return subprocess.CompletedProcess(arguments, status, stdout=f"status {status}\n")

            commands = [check_quality.AnalysisCommand([sys.executable, str(i)], root, f"analysis-{i}") for i in (7, 9, 0)]
            def native_command(name, arguments, **kwargs):
                completed = command(arguments, **kwargs)
                path = root / "logs" / f"{name}.log"
                path.write_text(completed.stdout, encoding="utf-8")
                if completed.returncode:
                    raise subprocess.CalledProcessError(completed.returncode, arguments)
                return path

            with patch.object(runner.native_run, "command", side_effect=native_command), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaises(check_quality.QualityError) as failure:
                runner.run_analysis(commands)
            self.assertIn("exit code 7", str(failure.exception))
            self.assertIn("exit code 9", str(failure.exception))
            self.assertEqual(len(list(runner.log_dir.glob("*.log"))), 2)

    def test_serial_failure_never_starts_the_next_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = check_quality.Runner(root, root / "logs", 1)
            commands = [check_quality.AnalysisCommand([sys.executable, str(i)], root, f"analysis-{i}") for i in range(4)]
            with patch.object(runner, "run", side_effect=check_quality.QualityError("failed")) as run:
                with self.assertRaisesRegex(check_quality.QualityError, "failed"):
                    runner.run_analysis(commands)
            run.assert_called_once()

    def test_launch_failure_is_logged_and_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = check_quality.Runner(root, root / "logs", 2)
            commands = [check_quality.AnalysisCommand(
                [str(root / "missing-tool")], root, "missing-tool",
            )]
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(check_quality.QualityError):
                runner.run_analysis(commands)
            self.assertIn("FileNotFoundError", (runner.log_dir / "missing-tool.log").read_text())


class DependencyContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = SCRIPT_PATH.parent.parent
        registry = "registry+https://github.com/rust-lang/crates.io-index"
        self.core = {
            "id": "core", "name": "film-juicer-core",
            "manifest_path": str(self.root / "rust/film-juicer-core/Cargo.toml"),
            "links": None, "targets": [{"kind": ["lib"]}],
            "dependencies": [
                {"name": name, "kind": None, "source": registry}
                for name in ("libm", "serde", "serde_json")
            ],
        }
        self.plugin = {
            "id": "plugin", "name": "film-juicer-plugin",
            "manifest_path": str(self.root / "rust/film-juicer-plugin/Cargo.toml"),
            "links": None, "targets": [{"kind": ["staticlib"]}],
            "dependencies": [{
                "name": "film-juicer-core", "kind": None,
                "path": str(self.root / "rust/film-juicer-core"),
            }],
        }
        self.metadata = {
            "packages": [self.core, self.plugin, *[
                {"id": name, "name": name, "source": registry}
                for name in ("libm", "serde", "serde_json")
            ]],
            "workspace_members": ["core", "plugin"],
            "resolve": {"nodes": [
                {"id": "core", "deps": [{"pkg": name} for name in ("libm", "serde", "serde_json")]},
                {"id": "plugin", "deps": [{"pkg": "core"}]},
            ]},
        }

    def test_current_dependencies_pass(self) -> None:
        check_quality.check_dependency_contract(self.root, self.metadata)

    def test_inactive_renamed_dependency_is_not_hidden(self) -> None:
        self.core["dependencies"].append({
            "name": "cuda-sys", "rename": "tables", "kind": None,
            "target": "cfg(windows)", "optional": True,
        })
        with self.assertRaisesRegex(check_quality.QualityError, "direct dependencies"):
            check_quality.check_dependency_contract(self.root, self.metadata)

    def test_native_build_hooks_and_links_are_rejected(self) -> None:
        cases = tuple(
            (f"{member}:{field}", index, field, value)
            for index, member in enumerate(("core", "plugin"))
            for field, value in (("links", "cuda"), ("targets", [{"kind": ["custom-build"]}]))
        )
        self.subtest_cases = tuple(case_id for case_id, *_ in cases)
        for case_id, index, field, value in cases:
            with self.subTest(case=case_id, member=index, field=field):
                metadata = copy.deepcopy(self.metadata)
                metadata["packages"][index][field] = value
                with self.assertRaisesRegex(check_quality.QualityError, "belong to CMake"):
                    check_quality.check_dependency_contract(self.root, metadata)

    def test_build_or_dev_dependency_requires_contract_change(self) -> None:
        cases = (("build-dependency", "build"), ("dev-dependency", "dev"))
        self.subtest_cases = tuple(case_id for case_id, _ in cases)
        for case_id, kind in cases:
            self.core["dependencies"][0]["kind"] = kind
            with self.subTest(case=case_id, kind=kind), self.assertRaisesRegex(check_quality.QualityError, "unapproved"):
                check_quality.check_dependency_contract(self.root, self.metadata)

    def test_plugin_cannot_select_another_core(self) -> None:
        self.plugin["dependencies"][0]["path"] = str(self.root / "alternate-core")
        with self.assertRaisesRegex(check_quality.QualityError, "workspace core by path"):
            check_quality.check_dependency_contract(self.root, self.metadata)

    def test_local_registry_patch_is_not_hidden(self) -> None:
        self.metadata["packages"][2]["source"] = None
        with self.assertRaisesRegex(check_quality.QualityError, "source override: libm"):
            check_quality.check_dependency_contract(self.root, self.metadata)

    def test_new_workspace_member_requires_contract_change(self) -> None:
        self.metadata["workspace_members"].append("libm")
        with self.assertRaisesRegex(check_quality.QualityError, "only the core and plugin"):
            check_quality.check_dependency_contract(self.root, self.metadata)

    def run_metadata(
        self,
        preset: str = "linux-debug",
        target: str = "x86_64-unknown-linux-gnu",
        *,
        development: bool = False,
        rust_packages: list[str] | None = None,
    ) -> Mock:
        runner = Mock(log_dir=self.root / "logs")
        for package in self.metadata["packages"]:
            if package.get("source"):
                package["manifest_path"] = str(self.root / "absent-registry" / package["name"] / "Cargo.toml")
        versions = {
            "cargo-version": "cargo 1.98.1 qualified",
            "rustc-version": "rustc 1.98.1 qualified",
            "rustfmt-version": "rustfmt 1.9.0-stable qualified",
            "clippy-version": "clippy 0.1.98 qualified",
            "rust-dependency-contract": json.dumps(self.metadata),
        }
        runner.run.side_effect = lambda *args, **kwargs: (
            "running 1 test\ntest result: ok" if kwargs["label"].startswith("test-")
            else versions.get(kwargs["label"], "")
        )
        arguments = argparse.Namespace(
            cargo=sys.executable, rustc=sys.executable, preset=preset,
            development=development, rust_packages=rust_packages,
        )
        with patch.object(check_quality, "resolve_tool", return_value=sys.executable), \
                patch.object(check_quality.shutil, "which", return_value=None), \
                patch.object(check_quality, "check_workspace_contract"):
            check_quality.check_rust(self.root, runner, arguments, target)
        return runner

    def test_clippy_matches_preset_and_preserves_quality_checks(self) -> None:
        cases = (
            ("linux-debug", "x86_64-unknown-linux-gnu", "dev", []),
            ("linux-release", "x86_64-unknown-linux-gnu", "release", ["--release"]),
            ("windows-clang-debug", "x86_64-pc-windows-msvc", "dev", []),
            ("windows-clang-release", "x86_64-pc-windows-msvc", "release", ["--release"]),
        )
        self.subtest_cases = tuple(f"preset:{preset}" for preset, *_ in cases)
        for preset, target, profile, profile_flags in cases:
            with self.subTest(case=f"preset:{preset}", preset=preset):
                runner = self.run_metadata(preset, target)
                common = [sys.executable, "clippy", "--locked", "--target", target,
                          "--target-dir", str(self.root / "out/build" / preset / "cargo")]
                expected = [
                    ("cargo-version", [sys.executable, "--version"]),
                    ("rustc-version", [sys.executable, "--version", "--verbose"]),
                    ("rustfmt-version", [sys.executable, "fmt", "--version"]),
                    ("clippy-version", [sys.executable, "clippy", "--version"]),
                    ("rust-dependency-contract", [sys.executable, "metadata", "--locked",
                                                  "--offline", "--format-version", "1"]),
                    ("rustfmt", [sys.executable, "fmt", "--all", "--", "--check"]),
                    (f"clippy-film-juicer-core-{profile}",
                     [*common, "-p", "film-juicer-core", "--all-targets", *profile_flags,
                      "--", "-D", "warnings"]),
                    (f"clippy-film-juicer-plugin-{profile}",
                     [*common, "-p", "film-juicer-plugin", "--all-targets", *profile_flags,
                      "--", "-D", "warnings"]),
                    (f"clippy-film-juicer-plugin-test-support-{profile}",
                     [*common, "-p", "film-juicer-plugin", "--all-targets", "--features",
                      "test-support", *profile_flags, "--", "-D", "warnings"]),
                ]
                calls = runner.run.call_args_list
                self.assertEqual([(call.kwargs["label"], call.args[0]) for call in calls], expected)
                self.assertTrue(calls[4].kwargs["stdout_only"])
                self.assertFalse(calls[4].kwargs["display_output"])
                for call in calls:
                    self.assertEqual(call.kwargs["env"]["CLIPPY_CONF_DIR"], str(self.root))

    def test_development_preserves_packages_and_uses_only_quality_commands(self) -> None:
        preset_cases = (
            ("linux-debug", "dev", []),
            ("linux-release", "release", ["--release"]),
        )
        package_cases = (
            ("core", ["film-juicer-core"]),
            ("plugin", ["film-juicer-plugin"]),
            ("both", ["film-juicer-core", "film-juicer-plugin"]),
        )
        cases = tuple(
            (f"{preset}:{package_case}", preset, profile, profile_flags, packages)
            for preset, profile, profile_flags in preset_cases
            for package_case, packages in package_cases
        )
        self.subtest_cases = tuple(case_id for case_id, *_ in cases)
        for case_id, preset, profile, profile_flags, packages in cases:
                with self.subTest(case=case_id, preset=preset, packages=packages):
                    runner = self.run_metadata(
                        preset, development=True, rust_packages=packages,
                    )
                    target_flags = ["--target", "x86_64-unknown-linux-gnu", "--target-dir",
                                    str(self.root / "out/build" / preset / "cargo")]
                    expected_clippy = []
                    for package in packages:
                        variants = [([], "production")]
                        if package == "film-juicer-plugin":
                            variants.append((["--features", "test-support"], "test-support"))
                        for features, variant in variants:
                            label = f"clippy-{package}-"
                            if variant == "test-support":
                                label += "test-support-"
                            expected_clippy.append((label + profile, [sys.executable, "clippy", "--locked",
                                *target_flags, "-p", package, "--all-targets", *features, *profile_flags,
                                "--", "-D", "warnings"]))
                    calls = runner.run.call_args_list
                    clippy = [(call.kwargs["label"], call.args[0]) for call in calls
                              if call.kwargs["label"].startswith("clippy-film-juicer-")]
                    self.assertEqual(clippy, expected_clippy)
                    self.assertFalse(any(call.args[0][1] == "test" for call in calls))
                    format_calls = [call.args[0] for call in calls if call.kwargs["label"] == "rustfmt"]
                    format_packages = [item for package in packages for item in ("--package", package)]
                    self.assertEqual(format_calls, [[sys.executable, "fmt", *format_packages, "--", "--check"]])

    def test_selected_clippy_failure_stops_dispatch(self) -> None:
        original = check_quality.check_rust
        preset_cases = (("linux-debug", "dev"), ("linux-release", "release"))
        cases = tuple(
            (f"{preset}:{label}", preset, profile, label, index)
            for preset, profile in preset_cases
            for index, label in enumerate((
                f"clippy-film-juicer-core-{profile}", f"clippy-film-juicer-plugin-{profile}",
                f"clippy-film-juicer-plugin-test-support-{profile}",
            ))
        )
        self.subtest_cases = tuple(case_id for case_id, *_ in cases)
        for case_id, preset, profile, failed_label, index in cases:
            labels = [f"clippy-film-juicer-core-{profile}", f"clippy-film-juicer-plugin-{profile}",
                      f"clippy-film-juicer-plugin-test-support-{profile}"]
            with self.subTest(case=case_id, preset=preset, failed_label=failed_label):
                    failure = check_quality.QualityError("selected Clippy failed")
                    captured = []

                    def failing_clippy(root, runner, arguments, target):
                        captured.append(runner)
                        run = runner.run.side_effect

                        def fail_selected(*args, **kwargs):
                            if kwargs["label"] == failed_label:
                                raise failure
                            return run(*args, **kwargs)

                        runner.run.side_effect = fail_selected
                        return original(root, runner, arguments, target)

                    with patch.object(check_quality, "check_rust", side_effect=failing_clippy), \
                            self.assertRaises(check_quality.QualityError) as raised:
                        self.run_metadata(preset, development=True)
                    self.assertIs(raised.exception, failure)
                    calls = captured[0].run.call_args_list
                    self.assertEqual(calls[-1].kwargs["label"], failed_label)
                    self.assertEqual([call.kwargs["label"] for call in calls
                                      if call.kwargs["label"].startswith("clippy-film-juicer-")],
                                     labels[:index + 1])

    def test_metadata_consumer_still_rejects_unapproved_dependency(self) -> None:
        self.core["dependencies"].append({"name": "cuda-sys", "kind": None, "target": "cfg(windows)"})
        with self.assertRaisesRegex(check_quality.QualityError, "direct dependencies"):
            self.run_metadata()


class CudaQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="cuda quality ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.build = self.root / "out/build/linux-debug"
        self.build.mkdir(parents=True)
        self.policy = check_quality.load_policy(SCRIPT_PATH.parent.parent)
        self.nvcc = self.root / "cuda/bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
        self.host = self.root / ("msvc/bin/Hostx64/x64/cl.exe" if os.name == "nt" else "gcc/bin/g++-13")
        self.arguments = [
            str(self.nvcc), "-forward-unknown-to-host-compiler", f"-ccbin={self.host}",
            "-DJUICER_DIAGNOSTICS_COMPILED=1", "-I../../include with spaces",
            "-isystem", "../../third party", '-DLABEL="a b"', "-UOLD", "-DOLD=2",
            "-std=c++20", "--generate-code=arch=compute_75,code=[sm_75]",
            "--generate-code=arch=compute_75,code=[compute_75]",
        ]
        self.arguments += (
            ["--cudart=hybrid", "-Xcompiler= /EHsc", "-Xcompiler= -Ob0 -Od",
             "-Xcompiler=-MD", "-Xcompiler=-Fdobject.dir/,-FS"]
            if os.name == "nt" else ["-Xcompiler=-fPIE", "-Xcompiler=-fvisibility=hidden"]
        )
        self.arguments += ["-x", "cu", "-c", str(self.root / "src/kernel.cu"), "-o", "kernel.o"]

    def entry(self, arguments: list[str] | None = None) -> check_quality.CompilationEntry:
        args = self.arguments if arguments is None else arguments
        command = subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)
        return check_quality.CompilationEntry("src/kernel.cu", command, self.build, tuple(args))

    def write_source(self, name: str, text: str) -> None:
        destination = self.root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")

    def test_platform_command_parser_preserves_attached_quotes_and_backslashes(self) -> None:
        entry = self.entry()
        self.assertEqual(check_quality.split_compiler_command(entry.command), entry.arguments)

    def test_arguments_database_retains_working_directory_and_quoted_values(self) -> None:
        entry = self.entry()
        (self.build / "compile_commands.json").write_text(json.dumps([{
            "directory": str(self.build), "file": "../../../src/kernel.cu",
            "arguments": list(entry.arguments), "command": "this command must not be parsed",
        }]), encoding="utf-8")
        actual = check_quality.compilation_entries(self.root, "linux-debug")
        self.assertEqual(actual[0].arguments, entry.arguments)
        self.assertEqual(actual[0].directory, self.build)
        self.assertEqual(actual[0].path, "src/kernel.cu")

    def test_translation_preserves_ordered_preprocessor_inputs_and_host_semantics(self) -> None:
        nvcc, host, arguments = check_quality.cuda_frontend_arguments(self.root, self.entry())
        self.assertEqual(nvcc, self.nvcc)
        self.assertEqual(host, self.host)
        expected = ["-D", "JUICER_DIAGNOSTICS_COMPILED=1", "-I", "../../include with spaces",
                    "-isystem", "../../third party", "-D", 'LABEL="a b"', "-U", "OLD", "-D", "OLD=2"]
        start = arguments.index("-D")
        self.assertEqual(arguments[start:start + len(expected)], expected)
        self.assertIn("-std=c++20", arguments)
        self.assertIn("-xcuda", arguments)
        self.assertIn("--cuda-gpu-arch=sm_75", arguments)
        self.assertIn(f"--cuda-path={self.nvcc.parent.parent}", arguments)
        if os.name == "nt":
            self.assertIn("-fms-runtime-lib=dll", arguments)
            self.assertIn("-fdelayed-template-parsing", arguments)
        else:
            self.assertIn("-fPIE", arguments)
            self.assertIn("-fvisibility=hidden", arguments)
        self.assertFalse(any(argument.startswith("-Wno-") for argument in arguments))

    def test_release_definition_and_optimization_are_taken_from_database(self) -> None:
        args = [arg.replace("JUICER_DIAGNOSTICS_COMPILED=1", "JUICER_DIAGNOSTICS_COMPILED=0")
                for arg in self.arguments]
        args += ["-O3", "-Xptxas=-O3"]
        if os.name == "nt":
            args += ["-Xcompiler=/EHsc,/O2,/Oi,/Ob2,/Gy,/Gw,/GF"]
        _, _, frontend = check_quality.cuda_frontend_arguments(self.root, self.entry(args))
        self.assertIn("JUICER_DIAGNOSTICS_COMPILED=0", frontend)
        self.assertNotIn("JUICER_DIAGNOSTICS_COMPILED=1", frontend)
        self.assertIn("-O3", frontend)

    def test_unknown_or_incomplete_options_fail_instead_of_weakening_analysis(self) -> None:
        cases = (
            ("unknown-option", self.arguments + ["--use_fast_math"]),
            ("response-file", self.arguments + ["@flags.rsp"]),
            ("host-flag", self.arguments + ["-Xcompiler=-fno-exceptions"]),
            ("incomplete-include", self.arguments + ["-I"]),
            ("wrong-cxx-standard", [arg.replace("c++20", "c++17") for arg in self.arguments]),
            ("wrong-architecture", [arg.replace("compute_75", "compute_86") for arg in self.arguments]),
            ("missing-host-compiler", [arg for arg in self.arguments if not arg.startswith("-ccbin=")]),
            ("wrong-input-name", [arg.replace("kernel.cu", "other.cu") for arg in self.arguments]),
            ("outside-source-root", [str(self.root / "other/src/kernel.cu")
                                      if arg == str(self.root / "src/kernel.cu") else arg
                                      for arg in self.arguments]),
        )
        self.subtest_cases = tuple(case_id for case_id, _ in cases)
        for case_id, args in cases:
            with self.subTest(case=case_id, args=args), self.assertRaises(check_quality.QualityError):
                check_quality.cuda_frontend_arguments(self.root, self.entry(args))

    def test_headers_select_transitive_cuda_and_cpp_owners_without_duplicates(self) -> None:
        self.write_source("src/shared.h", "#pragma once\n")
        self.write_source("src/kernel.cuh", '#include "shared.h"\n')
        self.write_source("src/kernel.cu", '#include "kernel.cuh"\n')
        self.write_source("src/host.cpp", '#include <shared.h>\n')
        entries = [self.entry(), self.entry(), check_quality.CompilationEntry("src/host.cpp", "c++ -Isrc")]
        self.assertEqual(check_quality.tidy_translation_units(
            self.root, ["src/shared.h"], entries, self.policy,
        ), ["src/host.cpp", "src/kernel.cu"])
        self.assertEqual(check_quality.tidy_translation_units(
            self.root, ["src/kernel.cuh", "src/kernel.cu"], entries, self.policy,
        ), ["src/kernel.cu"])

    def test_missing_cuda_source_or_header_owner_fails(self) -> None:
        cases = (("missing-cuda", "src/missing.cu"), ("missing-header", "src/missing.cuh"))
        self.subtest_cases = tuple(case_id for case_id, _ in cases)
        for case_id, source in cases:
            with self.subTest(case=case_id, source=source), self.assertRaises(check_quality.QualityError):
                check_quality.tidy_translation_units(self.root, [source], [self.entry()], self.policy)

    def test_native_dispatch_checks_all_cuda_variants_and_propagates_findings(self) -> None:
        self.write_source("src/kernel.cuh", "#pragma once\n")
        self.write_source("src/kernel.cu", '#include "kernel.cuh"\n')
        entries = [self.entry(), self.entry(self.arguments + ["-DTEST_VARIANT=1"])]
        runner = Mock()
        runner.run.side_effect = ["clang-format version 22.1.8", "LLVM version 22.1.8", "", ""]
        arguments = argparse.Namespace(clang_format=None, clang_tidy=None, preset="linux-debug")
        with patch.object(check_quality, "resolve_tool", return_value="tool"), \
             patch.object(check_quality, "compilation_entries", return_value=entries), \
             patch.object(check_quality, "prepare_cuda_analysis", side_effect=check_quality.QualityError("CUDA finding")) as cuda:
            with self.assertRaisesRegex(check_quality.QualityError, "CUDA finding"):
                check_quality.prepare_native_analysis(self.root, runner, arguments, ["src/kernel.cuh"], self.policy)
        cuda.assert_called_once_with(self.root, runner, "tool", entries)
        runner.run_analysis.assert_not_called()

    def test_overlay_retains_vendor_implementation_and_license(self) -> None:
        root = SCRIPT_PATH.parent.parent
        vendor = (root / "third_party/openrand/util.h").read_text(encoding="utf-8")
        overlay = check_quality.prepare_cuda_overlay(root, self.root / "logs")
        self.assertEqual((overlay / "openrand/util.h").read_text(encoding="utf-8"),
                         vendor.replace("#ifdef __CUDA_ARCH__\n", "#if defined(__CUDACC__)\n"))
        self.assertTrue((overlay / "texture_fetch_functions.h").is_file())
        self.write_source("third_party/openrand/util.h", "// changed attribute guard\n")
        with self.assertRaisesRegex(check_quality.QualityError, "guard changed"):
            check_quality.prepare_cuda_overlay(self.root, self.root / "logs")

    def test_cuda_invocation_keeps_variants_config_warnings_and_compile_directory(self) -> None:
        log_dir = self.root / "logs"
        log_dir.mkdir()
        runner = Mock(log_dir=log_dir)
        library = self.root / "gcc/lib/libstdc++.so"
        self.write_source("gcc/lib/libstdc++.so", "")
        self.write_source("cuda/include/curand_mtgp32_kernel.h", "")
        runner.run.side_effect = lambda command, **kwargs: {
            "cuda-nvcc-version": "Cuda compilation tools, release 13.2, V13.2.86",
            "cuda-host-version": "13.3.0", "cuda-host-library": str(library),
        }.get(kwargs["label"], "")
        entries = [self.entry(), self.entry(self.arguments + ["-DTEST_VARIANT=1"])]
        with patch.object(check_quality, "prepare_cuda_overlay", return_value=log_dir), \
             patch.dict(os.environ, {"VCToolsInstallDir": str(self.root / "msvc")}):
            commands = check_quality.prepare_cuda_analysis(self.root, runner, "clang-tidy", entries)
        self.assertEqual(len(commands), 2)
        self.assertIn("TEST_VARIANT=1", commands[1].command)
        for item in commands:
            self.assertIn("--warnings-as-errors=*", item.command)
            self.assertIn(f"--config-file={self.root / '.clang-tidy'}", item.command)
            self.assertEqual(item.directory, self.build)
        self.assertFalse(any(call.args[0][0] == "clang-tidy" for call in runner.run.call_args_list))
        self.assertEqual(
            [item.label for item in commands],
            ["clang-tidy-cuda-1-kernel.cu", "clang-tidy-cuda-2-kernel.cu"],
        )

    def test_missing_curand_is_an_actionable_failure(self) -> None:
        runner = Mock(log_dir=self.root / "logs")
        runner.run.return_value = "Cuda compilation tools, release 13.2, V13.2.86"
        with patch.object(check_quality, "prepare_cuda_overlay", return_value=self.root):
            with self.assertRaisesRegex(check_quality.QualityError, "cuRAND development headers"):
                check_quality.prepare_cuda_analysis(self.root, runner, "clang-tidy", [self.entry()])
        self.assertEqual(runner.run.call_count, 1)


class SourceHygieneTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.policy = check_quality.load_policy(SCRIPT_PATH.parent.parent)
        self.log_path = self.root / "source-hygiene.log"

    def write_source(self, path: str, source: str) -> None:
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.encode("utf-8"))

    def check(self, paths: list[str]) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            check_quality.check_source_hygiene(
                self.root, paths, self.policy, self.log_path
            )

    def test_each_violation_fails_with_location_and_log(self) -> None:
        cases = (
            ("trailing-whitespace", "src/file.cpp", "int count; \t", "trailing whitespace"),
            ("conflict-start", "src/file.cpp", "<<<<<<< ours", "merge conflict marker"),
            ("conflict-separator", "src/file.cpp", "=======", "merge conflict marker"),
            ("conflict-end", "src/file.cpp", ">>>>>>> theirs", "merge conflict marker"),
            ("conflict-base", "src/file.cpp", "||||||| base", "merge conflict marker"),
            ("retired-test-guard", "native/file.cpp", "#ifdef JUICER_TESTS", "retired JUICER_TESTS"),
            ("retired-validation-guard", "tests/file.cpp", "#ifdef JUICER_BUILD_VALIDATION", "retired JUICER_BUILD_VALIDATION"),
            ("std-endl", "src/file.cu", "stream << std::endl;", "std::endl is prohibited"),
            ("namespace-directive", "src/file.h", "    using namespace std;", "using namespace directives"),
            ("namespace-tab", "src/file.cuh", "using\tnamespace std;", "using namespace directives"),
        )
        self.subtest_cases = tuple(case_id for case_id, *_ in cases)
        for case_id, path, violation, diagnostic in cases:
            with self.subTest(case=case_id, path=path, violation=violation):
                self.write_source(path, "// first line\n" + violation + "\n")
                with self.assertRaises(check_quality.QualityError):
                    self.check([path])
                report = self.log_path.read_text(encoding="utf-8")
                self.assertIn(f"{path}:2:", report)
                self.assertIn(diagnostic, report)

    def test_all_diagnostics_are_reported_before_failure(self) -> None:
        self.write_source("src/a.cpp", "#ifdef JUICER_TESTS\n#endif\n")
        self.write_source("src/b.cpp", "#ifdef JUICER_BUILD_VALIDATION\n#endif\n")
        with self.assertRaises(check_quality.QualityError):
            self.check(["src/a.cpp", "src/b.cpp"])
        report = self.log_path.read_text(encoding="utf-8")
        self.assertIn("src/a.cpp:1:8:", report)
        self.assertIn("src/b.cpp:1:8:", report)

    def test_source_text_policy_includes_comments_and_literals(self) -> None:
        self.write_source("src/file.cpp", '// JUICER_TESTS\n"std::endl";\n')
        with self.assertRaises(check_quality.QualityError):
            self.check(["src/file.cpp"])
        report = self.log_path.read_text(encoding="utf-8")
        self.assertIn("retired JUICER_TESTS", report)
        self.assertIn("std::endl is prohibited", report)

    def test_clean_crlf_source_and_permitted_namespace_use_pass(self) -> None:
        self.write_source("src/file.cpp", "using namespace std;\r\nint count;\r\n")
        self.write_source("src/file.h", "using std::size_t;\nnamespace film {}\n")
        self.check(["src/file.cpp", "src/file.h"])
        self.assertIn("passed for 2", self.log_path.read_text(encoding="utf-8"))

    def test_all_policy_header_extensions_reject_namespace_directives(self) -> None:
        extensions = tuple(sorted(self.policy.header_extensions))
        self.subtest_cases = tuple(f"header-extension:{extension}" for extension in extensions)
        for extension in extensions:
            with self.subTest(case=f"header-extension:{extension}", extension=extension):
                path = f"src/file{extension}"
                self.write_source(path, "using namespace std;\n")
                with self.assertRaises(check_quality.QualityError):
                    self.check([path])

    def test_excluded_and_non_native_files_are_not_scanned(self) -> None:
        paths = [
            ".juicer-local/file.cpp", "external/file.h", "out/file.cu",
            "src/GeneratedColorSpaces.cpp", "tests/fixtures/file.cpp",
            "tests/quality/test_check_quality.py", "CONTRIBUTING.md",
        ]
        for path in paths:
            self.write_source(path, "JUICER_TESTS \n")
        self.check(paths)
        self.assertIn("passed for 0", self.log_path.read_text(encoding="utf-8"))

    def test_unreadable_or_invalid_utf8_source_fails(self) -> None:
        self.write_source("src/file.cpp", "")
        (self.root / "src/file.cpp").write_bytes(b"\xff")
        cases = (("missing-file", "src/missing.cpp"), ("invalid-utf8", "src/file.cpp"))
        self.subtest_cases = tuple(case_id for case_id, _ in cases)
        for case_id, path in cases:
            with self.subTest(case=case_id, path=path):
                with self.assertRaises(check_quality.QualityError):
                    self.check([path])
                self.assertIn("cannot read source", self.log_path.read_text(encoding="utf-8"))

    def test_dispatcher_rejects_violations_for_every_selection_mode(self) -> None:
        scripts = self.root / "scripts"
        scripts.mkdir()
        shutil.copy2(SCRIPT_PATH, scripts / SCRIPT_PATH.name)
        for name in ('execution.py', 'process_tree.py'):
            shutil.copy2(SCRIPT_PATH.parent / name, scripts)
        for name in ('verify.py', 'qualification.py', 'build_dependencies.py'):
            shutil.copy2(SCRIPT_PATH.parent / name, scripts)
        shutil.copy2(SCRIPT_PATH.parent / "source_file_policy.json", scripts)
        path = "native/a file.cuh"
        self.write_source(path, "#pragma once\n")
        for command in (
            ["git", "init", "--quiet"],
            ["git", "add", "--", path],
            ["git", "-c", "user.name=Quality Test", "-c", "user.email=quality@example.invalid",
             "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Baseline"],
        ):
            subprocess.run(command, cwd=self.root, check=True, capture_output=True)
        self.write_source(path, "#ifdef JUICER_BUILD_VALIDATION\n#endif\n")
        self.write_source("src/untracked.cpp", "#ifdef JUICER_TESTS\n#endif\n")
        selections = (
            ("explicit-file", ["--files", path]),
            ("base-selection", ["--base", "HEAD"]),
            ("all-owned", ["--all-owned"]),
        )
        self.subtest_cases = tuple(case_id for case_id, _ in selections)
        for index, (case_id, selection) in enumerate(selections):
            with self.subTest(case=case_id, selection=selection):
                bundle = self.root / f"out/validation/linux-debug/results/quality-boundary-{index}"
                bundle.mkdir(parents=True)
                result = subprocess.run(
                    [sys.executable, str(scripts / SCRIPT_PATH.name),
                     "--preset", "linux-debug", *selection],
                    cwd=self.root, env=dict(os.environ, FJ_HOST_BUNDLE=str(bundle)),
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(f"{path}:1:8:", result.stdout)
                self.assertIn("source hygiene checks failed", result.stderr)
                self.assertNotIn("Quality checks passed", result.stdout)
                report = bundle / "quality-source-hygiene.log"
                self.assertIn("JUICER_BUILD_VALIDATION", report.read_text(encoding="utf-8"))
                receipt = json.loads((bundle / "quality.checks.json").read_text(encoding="utf-8"))
                self.assertEqual(receipt["status"], "failed")
                self.assertFalse((bundle / "result.json").exists())
                self.assertFalse((bundle / "candidate.json").exists())
                if selection[0] == "--base":
                    self.assertIn("src/untracked.cpp:1:8:", result.stdout)


class SelectionAndEvidenceTests(unittest.TestCase):

    def test_run_directories_are_unique_and_never_include_old_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "out/validation/linux-debug/results/old.log"
            old.parent.mkdir(parents=True)
            old.write_text("retained", encoding="utf-8")
            first = check_quality.create_run_directory(root, "linux-debug")
            second = check_quality.create_run_directory(root, "linux-debug")
            self.assertNotEqual(first, second)
            self.assertEqual(list(first.iterdir()), [])
            self.assertEqual(list(second.iterdir()), [])
            self.assertEqual(old.read_text(encoding="utf-8"), "retained")

    def test_include_graph_preserves_cycles_local_resolution_and_all_consumers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "src/a.cpp": '#include "shared.h"\n',
                "src/kernel.cu": '#include "shared.h"\n',
                "src/shared.h": '#include "cycle.h"\n',
                "src/cycle.h": '#include "shared.h"\n',
                "tests/local/a.cpp": '#include "shared.h"\n',
                "tests/local/shared.h": '#pragma once\n',
            }
            for name, content in sources.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            entries = [check_quality.CompilationEntry(name, "compiler") for name in
                       ["src/a.cpp", "src/kernel.cu", "src/kernel.cu", "tests/local/a.cpp"]]
            policy = check_quality.load_policy(SCRIPT_PATH.parent.parent)
            reads = []
            original = Path.read_text

            def read(path, *args, **kwargs):
                reads.append(path)
                return original(path, *args, **kwargs)

            with patch.object(Path, "read_text", read):
                selected = check_quality.tidy_translation_units(root, ["src/shared.h", "src/cycle.h"], entries, policy)
            self.assertEqual(selected, ["src/a.cpp", "src/kernel.cu"])
            self.assertEqual(len(reads), len(set(reads)))
            # The graph is invocation-local; edits cannot leave cached reachability.
            (root / "src/a.cpp").write_text("", encoding="utf-8")
            self.assertEqual(check_quality.tidy_translation_units(root, ["src/shared.h"], entries, policy), ["src/kernel.cu"])

    def test_include_graph_resolves_an_aliased_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "source"
            (root / "src").mkdir(parents=True)
            (root / "src/owner.cpp").write_text('#include "shared.h"\n', encoding="utf-8")
            (root / "src/shared.h").write_text("#pragma once\n", encoding="utf-8")
            alias = parent / "alias"
            if os.name == "nt":
                subprocess.run(
                    ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(root)],
                    check=True, capture_output=True, timeout=30,
                )
            else:
                alias.symlink_to(root, target_is_directory=True)
            try:
                self.assertNotEqual(alias, alias.resolve())
                entries = [check_quality.CompilationEntry("src/owner.cpp", "compiler")]
                policy = check_quality.load_policy(SCRIPT_PATH.parent.parent)
                self.assertEqual(
                    check_quality.tidy_translation_units(alias, ["src/shared.h"], entries, policy),
                    ["src/owner.cpp"],
                )
            finally:
                if os.name == "nt":
                    alias.rmdir()
                else:
                    alias.unlink()

    def test_missing_translation_unit_fails_before_include_or_tool_work(self) -> None:
        root = SCRIPT_PATH.parent.parent
        policy = check_quality.load_policy(root)
        entries = [check_quality.CompilationEntry("src/a.cpp", "compiler")]
        selected = ["src/ColorTransforms.h", "tests/new.cpp"]
        with patch.object(check_quality, "included_paths") as includes:
            with self.assertRaisesRegex(check_quality.QualityError, "tests/new.cpp"):
                check_quality.tidy_translation_units(root, selected, entries, policy)
            includes.assert_not_called()
        with patch.object(check_quality, "compilation_entries", return_value=entries), \
             patch.object(check_quality, "resolve_tool") as tools:
            with self.assertRaisesRegex(check_quality.QualityError, "tests/new.cpp"):
                check_quality.prepare_native_analysis(root, Mock(), argparse.Namespace(preset="linux-debug"), selected, policy)
            tools.assert_not_called()

    def test_unresolved_header_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = check_quality.load_policy(SCRIPT_PATH.parent.parent)
            with self.assertRaisesRegex(check_quality.QualityError, "no consuming translation unit"):
                check_quality.tidy_translation_units(root, ["src/missing.h"], [], policy)

    def test_include_search_contexts_preserve_order_and_quote_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name, content in {
                "src/a.cpp": '#include "shared.h"\n',
                "src/b.cpp": '#include "shared.h"\n',
                "src/shared.h": '#include <choice.h>\n',
                "src/choice.h": "", "first/choice.h": "", "second/choice.h": "",
            }.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            entries = [check_quality.CompilationEntry("src/a.cpp", "cc -Ifirst -Isecond"),
                       check_quality.CompilationEntry("src/b.cpp", "cc -Isecond -Ifirst")]
            policy = check_quality.load_policy(SCRIPT_PATH.parent.parent)
            for header, expected in (("first/choice.h", ["src/a.cpp"]), ("second/choice.h", ["src/b.cpp"])):
                self.assertEqual(check_quality.tidy_translation_units(root, [header], entries, policy), expected)
            with self.assertRaisesRegex(check_quality.QualityError, "no consuming"):
                check_quality.tidy_translation_units(root, ["src/choice.h"], entries, policy)

    def test_asset_lookup_facade_selects_test_consumer(self) -> None:
        root = SCRIPT_PATH.parent.parent
        policy = check_quality.load_policy(root)
        entries = [check_quality.CompilationEntry(
            "tests/ffi/asset_lookup_test.cpp", "c++ -Itests/ffi -Isrc -Inative")]
        for header in ("tests/ffi/juicer_test_api.h", "tests/ffi/asset_lookup_test.h"):
            self.assertEqual(check_quality.tidy_translation_units(root, [header], entries, policy),
                             ["tests/ffi/asset_lookup_test.cpp"])

    def test_include_search_accepts_msvc_nvcc_quote_system_and_forced_paths(self) -> None:
        root = Path.cwd().resolve()
        entry = check_quality.CompilationEntry("src/a.cpp", "", root,
            ("compiler", "/I", "one space", "--include-path=two,three", "-iquote", "quoted",
             "-isystem", "system", "/FIforced.h"))
        with patch.dict(os.environ, {"CPATH": "", "INCLUDE": "", "CPLUS_INCLUDE_PATH": ""}):
            search = check_quality.include_search(root, entry)
        self.assertEqual(search.quoted, (root / "quoted",))
        self.assertEqual(search.ordinary, tuple(root / name for name in ("one space", "two", "three", "system")))
        self.assertEqual(search.forced, ("forced.h",))


class FixtureTransportTests(unittest.TestCase):
    def test_film_capture_crlf_preserves_bytes_and_rejects_trailing_spaces(self) -> None:
        source = SCRIPT_PATH.parent.parent
        fixtures = {
            path.relative_to(source): path.read_bytes()
            for path in (source / "tests/film/fixtures").glob("windows-clang-*-*.json")
        }
        self.assertTrue(fixtures)
        self.assertTrue(all(content.endswith(b"\r\n") for content in fixtures.values()))
        autocrlf_cases = (("autocrlf:false", "false"), ("autocrlf:true", "true"))
        ending_cases = (("space", b" \r\n"), ("tab", b"\t\r\n"))
        self.subtest_cases = (
            *(case_id for case_id, _ in autocrlf_cases),
            *(f"trailing-space:{autocrlf_id.removeprefix('autocrlf:')}:{ending_id}"
              for autocrlf_id, _ in autocrlf_cases for ending_id, _ in ending_cases),
        )
        for autocrlf_case, autocrlf in autocrlf_cases:
            with self.subTest(case=autocrlf_case, autocrlf=autocrlf), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)

                def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
                    return subprocess.run(
                        ["git", *arguments], cwd=root, check=check,
                        capture_output=True, timeout=30,
                    )

                git("init", "-q")
                git("config", "core.autocrlf", autocrlf)
                git("config", "core.whitespace", "blank-at-eol,blank-at-eof,space-before-tab")
                (root / ".gitattributes").write_bytes((source / ".gitattributes").read_bytes())
                for path, content in fixtures.items():
                    target = root / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                git("add", ".")
                result = git("diff", "--cached", "--check", check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for path, content in fixtures.items():
                    self.assertEqual(git("show", f":{path.as_posix()}").stdout, content)
                    (root / path).unlink()
                git("checkout-index", "--all")
                for path, content in fixtures.items():
                    self.assertEqual((root / path).read_bytes(), content)

                path = next(iter(fixtures))
                for ending_id, ending in ending_cases:
                    case_id = f"trailing-space:{autocrlf}:{ending_id}"
                    with self.subTest(case=case_id, ending=ending):
                        (root / path).write_bytes(b"{}" + ending)
                        git("add", "--", path.as_posix())
                        result = git("diff", "--cached", "--check", check=False)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(b"trailing whitespace", result.stdout)


class EmptySelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        environment = os.environ.copy()
        environment.pop("FJ_HOST_BUNDLE", None)
        self.enterContext(patch.dict(os.environ, environment, clear=True))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.policy = check_quality.load_policy(SCRIPT_PATH.parent.parent)
        local_scripts = self.root / "scripts"
        local_scripts.mkdir(parents=True)
        for name in ("check-quality.py", "execution.py", "process_tree.py", "verify.py",
                     "qualification.py", "build_dependencies.py", "source_file_policy.json"):
            shutil.copy2(SCRIPT_PATH.parent / name, local_scripts / name)
        files = {
            ".gitignore": "out/\n__pycache__/\n",
            "README.md": "Documentation\n",
            ".gitattributes": "*.py text\n",
            "Resources/sample.json": "{}\n",
            "src/owner.cpp": "// Boundary owner\n",
            "src/removable.cpp": "// Removable source\n",
            # A small executable guard makes command failure observable without
            # copying the product boundary or requiring compiler tools.
            "tests/ffi/test_native_boundary.py": (
                "import unittest\nfrom pathlib import Path\n"
                "class BoundaryTests(unittest.TestCase):\n"
                "    def test_owner_exists(self):\n"
                "        self.assertTrue(Path('src/owner.cpp').is_file())\n"
            ),
        }
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "baseline")
        self.enterContext(patch.object(check_quality, "__file__", str(self.root / "scripts/check-quality.py")))
        self.enterContext(patch.object(check_quality, "load_policy", return_value=self.policy))
        self.enterContext(patch.object(sys, "argv", ["quality", "--preset", "linux-debug", "--base", "HEAD"]))
        self.output = self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def git(self, *arguments: str) -> None:
        subprocess.run(["git", *arguments], cwd=self.root, check=True,
                       capture_output=True, timeout=30)

    def receipt(self) -> dict:
        path, = (self.root / "out/validation/linux-debug/results").glob("quality-*/result.json")
        self.result_directory = path.parent
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["selected"], [])
        self.assertFalse(record["shipping_isolation"])
        self.assertTrue(record["members"])
        return record

    def checks(self) -> dict:
        return json.loads((self.result_directory / "quality.checks.json").read_text(encoding="utf-8"))

    def test_excluded_only_diff_runs_guards_and_records_exclusions(self) -> None:
        for name in ("README.md", ".gitattributes", "Resources/sample.json"):
            with (self.root / name).open("a", encoding="utf-8") as stream:
                stream.write("Changed\n")
        with (patch.object(check_quality, "prepare_native_analysis") as native,
              patch.object(check_quality, "check_rust") as rust):
            self.assertEqual(check_quality.main(), 0)
        native.assert_not_called()
        rust.assert_not_called()
        record = self.receipt()
        self.assertEqual(record["status"], "passed")
        self.assertTrue(record["candidate_unchanged"])
        self.assertEqual(record["excluded"], [
            ".gitattributes: outside owned quality scope",
            "README.md: outside owned quality scope",
            "Resources/sample.json: excluded prefix Resources/",
        ])
        self.assertIn("git-diff-check", [item["name"] for item in record["commands"]])
        self.assertIn("native-boundary-enforcement", [item["name"] for item in record["commands"]])
        checks = self.checks()
        self.assertEqual(checks["expected_checks"], [
            "source-hygiene", "git-diff-check", "native-boundary-enforcement", "empty-selection",
        ])
        self.assertEqual(checks["completed_checks"], checks["expected_checks"])
        self.assertEqual(set(checks["expected_commands"]), {item["name"] for item in record["commands"]})
        self.assertEqual(checks["python_cases"], ["quality-boundary"])
        self.assertIn("No owned files selected", self.output.getvalue())

    def test_clean_base_still_publishes_a_result(self) -> None:
        self.assertEqual(check_quality.main(), 0)
        record = self.receipt()
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["excluded"], [])

    def test_deleted_owned_file_is_recorded_without_native_analysis(self) -> None:
        (self.root / "src/removable.cpp").unlink()
        with patch.object(check_quality, "prepare_native_analysis") as native:
            self.assertEqual(check_quality.main(), 0)
        native.assert_not_called()
        record = self.receipt()
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["excluded"], ["deleted: src/removable.cpp"])

    def test_whitespace_failure_in_excluded_file_is_not_a_pass(self) -> None:
        (self.root / "README.md").write_text("Trailing space \n", encoding="utf-8")
        with self.assertRaisesRegex(check_quality.QualityError, "git-diff-check"):
            check_quality.main()
        self.assertEqual(self.receipt()["status"], "failed")
        self.assertNotIn("Quality checks passed", self.output.getvalue())

    def test_boundary_failure_after_deletion_is_not_a_pass(self) -> None:
        (self.root / "src/owner.cpp").unlink()
        with self.assertRaisesRegex(check_quality.QualityError, "native-boundary-enforcement"):
            check_quality.main()
        self.assertEqual(self.receipt()["status"], "failed")
        checks = self.checks()
        self.assertIn("native-boundary-enforcement", checks["expected_checks"])
        self.assertNotIn("native-boundary-enforcement", checks["completed_checks"])
        self.assertIn("native-boundary-enforcement", checks["expected_commands"])
        self.assertNotIn("Quality checks passed", self.output.getvalue())

    def test_standalone_candidate_change_cannot_publish_pass(self) -> None:
        first = ({"head": "test-head", "inputs": {"src/a.cpp": {"git_blob": "one"}},
                  "new_files_base64": {}}, b"", {"source_capture_seconds": 0.0})
        changed = ({"head": "test-head", "inputs": {"src/a.cpp": {"git_blob": "two"}},
                    "new_files_base64": {}}, b"", {"source_capture_seconds": 0.0})
        with patch.object(check_quality, "capture_candidate", side_effect=[first, changed]), \
             patch.object(check_quality, "run_checks", return_value=0):
            with self.assertRaisesRegex(check_quality.QualityError, "candidate changed"):
                check_quality.main()
        record = self.receipt()
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["candidate_unchanged"])

    def test_explicit_and_all_owned_empty_selections_remain_errors(self) -> None:
        self.git("rm", "-qr", "src", "tests", "scripts", ".gitignore")
        cases = (
            ("explicit-outside-owned", ["--files", "README.md"]),
            ("explicit-missing", ["--files", "src/missing.cpp"]),
            ("all-owned-empty", ["--all-owned"]),
        )
        self.subtest_cases = tuple(case_id for case_id, _ in cases)
        for case_id, mode in cases:
            with self.subTest(case=case_id, mode=mode):
                arguments = argparse.Namespace(base=None, files=mode[1:] if mode[0] == "--files" else None)
                with self.assertRaisesRegex(check_quality.QualityError, "selection is empty"):
                    check_quality.select_paths(self.root, arguments, self.policy)

    def test_invalid_base_and_discovery_errors_remain_errors(self) -> None:
        arguments = argparse.Namespace(base="refs/heads/does-not-exist", files=None)
        self.subtest_cases = ("invalid-base", "malformed-record", "path-traversal", "missing-path")
        with self.subTest(case="invalid-base"), self.assertRaises(check_quality.QualityError):
            check_quality.select_paths(self.root, arguments, self.policy)
        arguments.base = "HEAD"
        cases = (
            ("malformed-record", b"M\0"),
            ("path-traversal", b"M\0../outside.cpp\0"),
            ("missing-path", b"M\0src/missing.cpp\0"),
        )
        for case_id, payload in cases:
            with self.subTest(case=case_id, payload=payload), patch.object(check_quality, "run_git", side_effect=[payload, b""]):
                with self.assertRaises(check_quality.QualityError):
                    check_quality.select_paths(self.root, arguments, self.policy)

    def test_base_still_selects_mixed_and_untracked_owned_files(self) -> None:
        (self.root / "README.md").write_text("Changed\n", encoding="utf-8")
        (self.root / "src/owner.cpp").write_text("// Changed\n", encoding="utf-8")
        (self.root / "src/new.cpp").write_text("// Untracked\n", encoding="utf-8")
        selected, excluded = check_quality.select_paths(
            self.root, argparse.Namespace(base="HEAD", files=None), self.policy)
        self.assertEqual(selected, ["src/new.cpp", "src/owner.cpp"])
        self.assertEqual(excluded, ["README.md: outside owned quality scope"])


CheckQualityTests.test_verifier_scope_is_conservative_for_dependencies_and_uncertainty.required_subtests = True
CheckQualityTests.test_verifier_scope_compares_merge_base_to_candidate.required_subtests = True
CheckQualityTests.test_generated_rust_dispatches_required_check_families.required_subtests = True
CheckQualityTests.test_generated_rust_check_failure_stops_dispatch.required_subtests = True
CheckQualityTests.test_release_cargo_manifest_dispatches_full_rust_only.required_subtests = True
CheckQualityTests.test_policy_changes_cannot_skip_required_checks.required_subtests = True
CheckQualityTests.test_rust_probe_profiles_select_the_preset_or_complete_matrix.required_subtests = True
CheckQualityTests.test_boundary_negative_checks_bind_code_marker_and_fixture.required_subtests = True
CheckQualityTests.test_boundary_fixture_batches_keep_success_and_failure_groups_separate.required_subtests = True
CommandOutputTests.test_empty_or_malformed_stdout_cannot_use_json_from_stderr.required_subtests = True
ParallelAnalysisTests.test_cli_requires_a_positive_worker_limit.required_subtests = True
ParallelAnalysisTests.test_development_cannot_stage_probe_or_complete_qualification_receipts.required_subtests = True
DependencyContractTests.test_native_build_hooks_and_links_are_rejected.required_subtests = True
DependencyContractTests.test_build_or_dev_dependency_requires_contract_change.required_subtests = True
DependencyContractTests.test_clippy_matches_preset_and_preserves_quality_checks.required_subtests = True
DependencyContractTests.test_development_preserves_packages_and_uses_only_quality_commands.required_subtests = True
DependencyContractTests.test_selected_clippy_failure_stops_dispatch.required_subtests = True
CudaQualityTests.test_unknown_or_incomplete_options_fail_instead_of_weakening_analysis.required_subtests = True
CudaQualityTests.test_missing_cuda_source_or_header_owner_fails.required_subtests = True
SourceHygieneTests.test_each_violation_fails_with_location_and_log.required_subtests = True
SourceHygieneTests.test_all_policy_header_extensions_reject_namespace_directives.required_subtests = True
SourceHygieneTests.test_unreadable_or_invalid_utf8_source_fails.required_subtests = True
SourceHygieneTests.test_dispatcher_rejects_violations_for_every_selection_mode.required_subtests = True
FixtureTransportTests.test_film_capture_crlf_preserves_bytes_and_rejects_trailing_spaces.required_subtests = True
EmptySelectionTests.test_explicit_and_all_owned_empty_selections_remain_errors.required_subtests = True
EmptySelectionTests.test_invalid_base_and_discovery_errors_remain_errors.required_subtests = True


if __name__ == "__main__":
    unittest.main()
