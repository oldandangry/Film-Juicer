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


class CheckQualityTests(unittest.TestCase):
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
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.enterContext(patch.object(check_quality, "create_run_directory", return_value=Path(directory.name)))
        # Dispatch tests mock the checks themselves; receipt stability has its
        # own real temporary-repository witnesses below.
        self.enterContext(patch.object(check_quality, "quality_inputs", return_value={"dispatch_test": True}))

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
        selections = (
            [generated_rust],
            [generated_rust, "tests/ffi/README.md"],
            [header, generated_rust],
        )
        for files in selections:
            with (
                self.subTest(files=files),
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
        for files in (
            [generated_rust],
            [generated_rust, "tests/ffi/README.md"],
            ["native/juicer_cuda_api.h", generated_rust],
        ):
            with (
                self.subTest(files=files),
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

    def test_policy_changes_cannot_skip_required_checks(self) -> None:
        for path in ("Cargo.toml", "rust/film-juicer-core/Cargo.toml", "tests/quality/test_rust_boundaries.py", "tests/quality/probe_evidence.py", "scripts/check-quality.py"):
            with (
                self.subTest(path=path),
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
        producer = check_quality.AnalysisCommand(["tool", "producer"], Path("."), "producer", "src/Illuminants.cpp")
        consumer = check_quality.AnalysisCommand(["tool", "consumer"], Path("."), "consumer", "src/main.cpp")
        with (
            patch.object(sys, "argv", [str(SCRIPT_PATH), "--preset", "linux-debug", "--files",
                "src/Illuminants.cpp", "native/juicer_legacy_api.h", "rust/film-juicer-core/src/exposure.rs"]),
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
        producer = check_quality.AnalysisCommand(["tool", "producer"], Path("."), "producer", "src/Illuminants.cpp")
        consumers = [check_quality.AnalysisCommand(["tool", variant], Path("."), variant,
            "src/Cuda/Film/JuicerCudaFilmPipeline.cu") for variant in ("ordinary", "test")]
        events = []
        with (
            patch.object(sys, "argv", [str(SCRIPT_PATH), "--preset", "linux-debug", "--files",
                "src/Illuminants.cpp", "native/juicer_legacy_api.h", "rust/film-juicer-core/src/exposure.rs"]),
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

    def test_rust_probe_profiles_select_the_preset_or_complete_matrix(self) -> None:
        for module_name in ("test_rust_naming", "test_rust_boundaries"):
            module = __import__(module_name)
            suite = module.RustNamingTests if module_name.endswith("naming") else module.RustBoundaryTests
            with tempfile.TemporaryDirectory() as directory:
                settings = {"JUICER_CARGO": "cargo", "JUICER_RUST_TARGET": "x86_64-unknown-linux-gnu",
                    "JUICER_RUST_PROBE_DIR": directory + "/cache", "JUICER_TEST_ARTIFACT_DIR": directory + "/logs"}
                for profile, expected in (("debug", (False,)), ("release", (True,)), ("both", (False, True))):
                    with self.subTest(module=module_name, profile=profile), patch.dict(os.environ,
                        {**settings, "JUICER_RUST_PROBE_PROFILE": profile}), patch.object(module, "ProbeEvidence"):
                        try:
                            suite.setUpClass()
                            self.assertEqual(suite.profiles, expected)
                            self.assertEqual(suite.target_dir, Path(settings["JUICER_RUST_PROBE_DIR"]).resolve())
                        finally:
                            suite.doClassCleanups()
                with patch.dict(os.environ, {**settings, "JUICER_RUST_PROBE_PROFILE": "invalid"}):
                    with self.assertRaises(KeyError):
                        suite.setUpClass()

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
            self.assertTrue((root / "logs/01-expected-failure.log").is_file())
            record = json.loads((root / "logs/01-expected-failure.command.json").read_text())
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
        self.assertEqual((self.runner.log_dir / "01-metadata.stdout.log").read_bytes(), stdout)
        self.assertEqual((self.runner.log_dir / "01-metadata.stderr.log").read_bytes(), stderr)
        record = json.loads((self.runner.log_dir / "01-metadata.command.json").read_text())
        for stream in ("stdout", "stderr"):
            path = self.runner.log_dir / record[stream]["file"]
            self.assertEqual(record[stream]["sha256"], check_quality.file_digest(path))
        self.assertEqual(record["returncode"], 0)
        self.runner.verify_inputs()

    def test_empty_or_malformed_stdout_cannot_use_json_from_stderr(self) -> None:
        for stdout in (b"", b"not JSON", b'{"ok": true}\ntrailing junk'):
            with self.subTest(stdout=stdout), contextlib.redirect_stdout(io.StringIO()):
                output = self.runner.run(self.command(stdout, b'{"ok": true}'),
                                         label="bad-metadata", stdout_only=True)
                with self.assertRaises(json.JSONDecodeError):
                    json.loads(output)

    def test_failed_command_retains_both_streams_and_cannot_return_valid_json(self) -> None:
        stdout, stderr = b'{"ok": true}\n', b"intentional Cargo failure\n"
        with contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(check_quality.QualityError, "exit code 7"):
            self.runner.run(self.command(stdout, stderr, 7), label="failed-metadata", stdout_only=True)
        self.assertEqual((self.runner.log_dir / "01-failed-metadata.stdout.log").read_bytes(), stdout)
        self.assertEqual((self.runner.log_dir / "01-failed-metadata.stderr.log").read_bytes(), stderr)
        record = json.loads((self.runner.log_dir / "01-failed-metadata.command.json").read_text())
        self.assertEqual(record["returncode"], 7)

    def test_large_stderr_is_drained_without_corrupting_stdout(self) -> None:
        stderr = b"package-cache diagnostic\n" * 8192
        command = [sys.executable, "-c",
                   'import os; os.write(2, b"package-cache diagnostic\\n" * 8192); os.write(1, b\'{"ok": true}\')']
        with contextlib.redirect_stdout(io.StringIO()):
            output = self.runner.run(command,
                                     label="large-stderr", stdout_only=True)
        self.assertEqual(json.loads(output), {"ok": True})
        self.assertEqual((self.runner.log_dir / "01-large-stderr.stderr.log").read_bytes(), stderr)

    def test_raw_stream_mutation_invalidates_consumed_input_identity(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.runner.run(self.command(b'{"ok": true}', b"diagnostic"),
                            label="metadata", stdout_only=True)
        for stream in ("stdout", "stderr"):
            with self.subTest(stream=stream):
                path = self.runner.log_dir / f"01-metadata.{stream}.log"
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(check_quality.QualityError, "input changed"):
                    self.runner.verify_inputs()
                path.write_bytes(original)
        self.runner.verify_inputs()

    def test_default_command_keeps_merged_diagnostic_behavior(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            output = self.runner.run(self.command(b"stdout\n", b"stderr\n"), label="ordinary")
        self.assertEqual(output, "stderr\nstdout")
        self.assertEqual(len(list(self.runner.log_dir.glob("*.log"))), 1)


class ParallelAnalysisTests(unittest.TestCase):
    def test_cli_requires_a_positive_worker_limit(self) -> None:
        base = [str(SCRIPT_PATH), "--preset", "linux-debug", "--files", "CONTRIBUTING.md"]
        for count in (1, 2, 4):
            with self.subTest(count=count), patch.object(sys, "argv", [*base, "--jobs", str(count)]):
                self.assertEqual(check_quality.parse_arguments().jobs, count)
        with patch.object(sys, "argv", base):
            self.assertEqual(check_quality.parse_arguments().jobs, 2)
        for count in ("0", "-1", "invalid"):
            with self.subTest(count=count), patch.object(sys, "argv", [*base, "--jobs", count]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                check_quality.parse_arguments()
            self.assertEqual(failure.exception.code, 2)

    def test_probe_handoff_requires_new_path_and_matching_identity_arguments(self) -> None:
        base = [str(SCRIPT_PATH), "--preset", "linux-debug", "--files", "CONTRIBUTING.md"]
        identity = "01234567-89ab-4def-8012-3456789abcde"
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "probes.json"
            for extra in (["--probe-manifest", str(manifest)], ["--qualification-id", identity],
                          ["--probe-manifest", str(manifest), "--qualification-id", "bad-id"]):
                with self.subTest(extra=extra), patch.object(sys, "argv", [*base, *extra]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    check_quality.parse_arguments()
            arguments = [*base, "--probe-manifest", str(manifest), "--qualification-id", identity]
            with patch.object(sys, "argv", arguments):
                self.assertEqual(check_quality.parse_arguments().qualification_id, identity)
            for path in (manifest, manifest.with_name(manifest.name + ".pending")):
                path.write_text("existing")
                with patch.object(sys, "argv", arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    check_quality.parse_arguments()
                path.unlink()

    def test_serial_and_parallel_commands_keep_arguments_directories_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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

            commands = [check_quality.AnalysisCommand(["tool", str(i)], root, f"analysis-{i}") for i in range(4)]
            with patch.object(check_quality.subprocess, "run", side_effect=command), \
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

            commands = [check_quality.AnalysisCommand(["tool", str(i)], root, f"analysis-{i}") for i in (7, 9, 0)]
            with patch.object(check_quality.subprocess, "run", side_effect=command), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaises(check_quality.QualityError) as failure:
                runner.run_analysis(commands)
            self.assertIn("exit code 7", str(failure.exception))
            self.assertIn("exit code 9", str(failure.exception))
            self.assertEqual(len(list(runner.log_dir.glob("*.log"))), 2)

    def test_serial_failure_never_starts_the_next_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = check_quality.Runner(root, root / "logs", 1)
            commands = [check_quality.AnalysisCommand(["tool", str(i)], root, f"analysis-{i}") for i in range(4)]
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
            self.assertIn("could not start command", (runner.log_dir / "01-missing-tool.log").read_text())


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
        for index in (0, 1):
            for field, value in (("links", "cuda"), ("targets", [{"kind": ["custom-build"]}])):
                with self.subTest(member=index, field=field):
                    metadata = copy.deepcopy(self.metadata)
                    metadata["packages"][index][field] = value
                    with self.assertRaisesRegex(check_quality.QualityError, "belong to CMake"):
                        check_quality.check_dependency_contract(self.root, metadata)

    def test_build_or_dev_dependency_requires_contract_change(self) -> None:
        for kind in ("build", "dev"):
            self.core["dependencies"][0]["kind"] = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(check_quality.QualityError, "unapproved"):
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

    def run_metadata(self) -> Mock:
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
        runner.run.side_effect = lambda *args, **kwargs: versions.get(kwargs["label"], "")
        arguments = argparse.Namespace(cargo=sys.executable, rustc=sys.executable, preset="linux-debug")
        with patch.object(check_quality, "resolve_tool", return_value=sys.executable), \
                patch.object(check_quality.shutil, "which", return_value=None), \
                patch.object(check_quality, "check_workspace_contract"):
            check_quality.check_rust(self.root, runner, arguments, "x86_64-unknown-linux-gnu")
        return runner

    def test_metadata_consumer_requests_stdout_and_keeps_complete_quality_dispatch(self) -> None:
        runner = self.run_metadata()
        calls = {call.kwargs["label"]: call for call in runner.run.call_args_list}
        self.assertTrue(calls["rust-dependency-contract"].kwargs["stdout_only"])
        self.assertEqual(calls["rust-dependency-contract"].args[0][1:],
                         ["metadata", "--locked", "--offline", "--format-version", "1"])
        for label in ("rustfmt", "rust-naming-enforcement", "rust-boundary-enforcement"):
            self.assertIn(label, calls)
        clippy = [label for label in calls if label.startswith("clippy-film-juicer-")]
        self.assertEqual(len(clippy), 6)

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
        cases = [
            self.arguments + ["--use_fast_math"],
            self.arguments + ["@flags.rsp"],
            self.arguments + ["-Xcompiler=-fno-exceptions"],
            self.arguments + ["-I"],
            [arg.replace("c++20", "c++17") for arg in self.arguments],
            [arg.replace("compute_75", "compute_86") for arg in self.arguments],
            [arg for arg in self.arguments if not arg.startswith("-ccbin=")],
            [arg.replace("kernel.cu", "other.cu") for arg in self.arguments],
            [str(self.root / "other/src/kernel.cu") if arg == str(self.root / "src/kernel.cu") else arg
             for arg in self.arguments],
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(check_quality.QualityError):
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
        for source in ["src/missing.cu", "src/missing.cuh"]:
            with self.subTest(source=source), self.assertRaises(check_quality.QualityError):
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
        receipt = json.loads((log_dir / "cuda-command-1.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["nvcc_arguments"], self.arguments)

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
            ("src/file.cpp", "int count; \t", "trailing whitespace"),
            ("src/file.cpp", "<<<<<<< ours", "merge conflict marker"),
            ("src/file.cpp", "=======", "merge conflict marker"),
            ("src/file.cpp", ">>>>>>> theirs", "merge conflict marker"),
            ("src/file.cpp", "||||||| base", "merge conflict marker"),
            ("native/file.cpp", "#ifdef JUICER_TESTS", "retired JUICER_TESTS"),
            ("tests/file.cpp", "#ifdef JUICER_BUILD_VALIDATION", "retired JUICER_BUILD_VALIDATION"),
            ("src/file.cu", "stream << std::endl;", "std::endl is prohibited"),
            ("src/file.h", "    using namespace std;", "using namespace directives"),
            ("src/file.cuh", "using\tnamespace std;", "using namespace directives"),
        )
        for path, violation, diagnostic in cases:
            with self.subTest(path=path, violation=violation):
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
        for extension in self.policy.header_extensions:
            with self.subTest(extension=extension):
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
        for path in ("src/missing.cpp", "src/file.cpp"):
            with self.subTest(path=path):
                with self.assertRaises(check_quality.QualityError):
                    self.check([path])
                self.assertIn("cannot read source", self.log_path.read_text(encoding="utf-8"))

    def test_dispatcher_rejects_violations_for_every_selection_mode(self) -> None:
        scripts = self.root / "scripts"
        scripts.mkdir()
        shutil.copy2(SCRIPT_PATH, scripts / SCRIPT_PATH.name)
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
        for selection in (["--files", path], ["--base", "HEAD"], ["--all-owned"]):
            with self.subTest(selection=selection):
                result = subprocess.run(
                    [sys.executable, str(scripts / SCRIPT_PATH.name),
                     "--preset", "linux-debug", *selection],
                    cwd=self.root, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(f"{path}:1:8:", result.stdout)
                self.assertIn("source hygiene checks failed", result.stderr)
                self.assertNotIn("Quality checks passed", result.stdout)
                reports = sorted((self.root / "out/validation/linux-debug/quality/runs").glob("*/source-hygiene.log"),
                                 key=lambda path: path.stat().st_mtime_ns)
                report = reports[-1]
                self.assertIn("JUICER_BUILD_VALIDATION", report.read_text(encoding="utf-8"))
                receipt = json.loads((report.parent / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(receipt["status"], "failed")
                self.assertIn("source-hygiene.log", receipt["files"])
                if selection[0] == "--base":
                    self.assertIn("src/untracked.cpp:1:8:", result.stdout)


class SelectionAndEvidenceTests(unittest.TestCase):
    def test_manifest_preparation_preserves_an_empty_bytecode_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            scripts = root / "scripts"
            quality = root / "tests/quality"
            scripts.mkdir()
            quality.mkdir(parents=True)
            shutil.copy2(SCRIPT_PATH, scripts)
            shutil.copy2(SCRIPT_PATH.parent / "source_file_policy.json", scripts)
            shutil.copy2(SCRIPT_PATH.parent.parent / "tests/quality/probe_evidence.py", quality)
            # Exercise the real terminal receipt and late manifest import. The
            # compiler work is irrelevant to whether that import creates a cache.
            bootstrap = (
                "import importlib.util, sys; from unittest.mock import patch; "
                "spec = importlib.util.spec_from_file_location('check_quality', sys.argv[1]); "
                "module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; "
                "spec.loader.exec_module(module); sys.argv = sys.argv[1:]; "
                "patch.object(module, 'select_paths', return_value=([], [])).start(); "
                "patch.object(module, 'quality_inputs', return_value={}).start(); "
                "patch.object(module, 'run_checks', return_value=0).start(); "
                "raise SystemExit(module.main())"
            )
            environment = dict(os.environ)
            environment.pop("PYTHONDONTWRITEBYTECODE", None)
            environment.pop("PYTHONPYCACHEPREFIX", None)
            result = subprocess.run(
                [sys.executable, "-c", bootstrap, str(scripts / SCRIPT_PATH.name),
                 "--preset", "linux-debug", "--files", "CONTRIBUTING.md",
                 "--probe-manifest", str(root / "probes.json"),
                 "--qualification-id", "01234567-89ab-4def-8012-3456789abcde"],
                cwd=root, env=environment, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            receipt, = (root / "out/validation/linux-debug/quality/runs").glob("*/run.json")
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8"))["status"], "passed")
            self.assertEqual(list(quality.rglob("*.pyc")), [],
                             "manifest preparation must not change the captured Python context")

    def test_run_directories_are_unique_and_never_include_old_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "out/validation/linux-debug/quality/old.log"
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

    def test_asset_lookup_hook_selects_render_recipe_consumer(self) -> None:
        root = SCRIPT_PATH.parent.parent
        policy = check_quality.load_policy(root)
        entries = [check_quality.CompilationEntry(
            "src/RenderRecipe.cpp", "c++ -DJUICER_ASSET_LOOKUP_TEST_HOOK=1 -Itests/ffi -Isrc -Inative")]
        self.assertEqual(check_quality.tidy_translation_units(root, ["tests/ffi/juicer_test_api.h"], entries, policy),
                         ["src/RenderRecipe.cpp"])

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


class EmptySelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.policy = check_quality.load_policy(SCRIPT_PATH.parent.parent)
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
        path, = (self.root / "out/validation/linux-debug/quality/runs").glob("*/run.json")
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["selected"], [])
        self.assertEqual(record["selected_sha256"], {})
        self.assertIn("git_head", record["inputs"])
        self.assertTrue(record["consumed_inputs"])
        self.assertIn("finished_at", record)
        for name, digest in record["files"].items():
            self.assertEqual(check_quality.file_digest(path.parent / name), digest)
        return record

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
        self.assertEqual(record["excluded"], [
            ".gitattributes: outside owned quality scope",
            "README.md: outside owned quality scope",
            "Resources/sample.json: excluded prefix Resources/",
        ])
        self.assertTrue(any("git-diff-check" in name for name in record["files"]))
        self.assertTrue(any("native-boundary-enforcement" in name for name in record["files"]))
        self.assertIn("No owned files selected", self.output.getvalue())

    def test_clean_base_still_completes_a_receipt(self) -> None:
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
        self.assertNotIn("Quality checks passed", self.output.getvalue())

    def test_empty_selection_still_rechecks_source_and_consumed_inputs(self) -> None:
        original = check_quality.run_checks
        for changed in ("source", "consumed"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                log_dir = Path(directory)

                def check_then_mutate(root, arguments, policy, selected, excluded, runner):
                    result = original(root, arguments, policy, selected, excluded, runner)
                    path = root / "README.md" if changed == "source" else log_dir / "consumed.txt"
                    if changed == "consumed":
                        path.write_text("original", encoding="utf-8")
                        runner.bind_inputs([path])
                    path.write_text("changed", encoding="utf-8")
                    return result

                with (patch.object(check_quality, "create_run_directory", return_value=log_dir),
                      patch.object(check_quality, "run_checks", side_effect=check_then_mutate)):
                    with self.assertRaisesRegex(check_quality.QualityError, "quality input"):
                        check_quality.main()
                record = json.loads((log_dir / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(record["selected"], [])
                self.assertEqual(record["status"], "failed")
                self.assertNotIn("Quality checks passed", self.output.getvalue())

    def test_explicit_and_all_owned_empty_selections_remain_errors(self) -> None:
        self.git("rm", "-qr", "src", "tests", ".gitignore")
        for mode in (["--files", "README.md"], ["--files", "src/missing.cpp"], ["--all-owned"]):
            with self.subTest(mode=mode):
                arguments = argparse.Namespace(base=None, files=mode[1:] if mode[0] == "--files" else None)
                with self.assertRaisesRegex(check_quality.QualityError, "selection is empty"):
                    check_quality.select_paths(self.root, arguments, self.policy)

    def test_invalid_base_and_discovery_errors_remain_errors(self) -> None:
        arguments = argparse.Namespace(base="refs/heads/does-not-exist", files=None)
        with self.assertRaises(check_quality.QualityError):
            check_quality.select_paths(self.root, arguments, self.policy)
        arguments.base = "HEAD"
        for payload in (b"M\0", b"M\0../outside.cpp\0", b"M\0src/missing.cpp\0"):
            with self.subTest(payload=payload), patch.object(check_quality, "run_git", side_effect=[payload, b""]):
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


class QualityReceiptTests(unittest.TestCase):
    def test_generated_cuda_inputs_must_remain_stable_until_receipt_completion(self) -> None:
        headers = ("openrand/util.h", "texture_fetch_functions.h")
        cases = [(None, None), *((name, action) for name in headers for action in ("modify", "delete"))]
        for name, action in cases:
            with self.subTest(header=name, action=action), tempfile.TemporaryDirectory() as directory:
                log_dir = Path(directory).resolve()
                expected = {}

                def check_then_mutate(root, arguments, policy, selected, excluded, runner):
                    # Use real preparation/binding and receipt completion; compiler
                    # execution is outside this input-lifetime regression.
                    check_quality.prepare_cuda_analysis(root, runner, "unused", [])
                    overlay = runner.log_dir / "cuda-include"
                    expected.update({str((overlay / header).resolve()): check_quality.file_digest(overlay / header)
                                     for header in headers})
                    if action == "modify":
                        with (overlay / name).open("a", encoding="utf-8") as stream:
                            stream.write("\n#error generated input changed after analysis\n")
                    elif action == "delete":
                        (overlay / name).unlink()
                    return 0

                with (patch.object(sys, "argv", ["quality", "--preset", "linux-debug", "--files", "scripts/check-quality.py"]),
                      patch.object(check_quality, "check_build_path"),
                      patch.object(check_quality, "create_run_directory", return_value=log_dir),
                      patch.object(check_quality, "quality_inputs", return_value={"unchanged_source_inventory": True}),
                      patch.object(check_quality, "run_checks", side_effect=check_then_mutate),
                      contextlib.redirect_stdout(io.StringIO()) as output):
                    if action is None:
                        self.assertEqual(check_quality.main(), 0)
                    else:
                        with self.assertRaisesRegex(check_quality.QualityError, "quality input"):
                            check_quality.main()
                record = json.loads((log_dir / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(record["consumed_inputs"], expected)
                self.assertEqual(record["status"], "passed" if action is None else "failed")
                self.assertEqual("Quality checks passed" in output.getvalue(), action is None)

    def test_source_and_configuration_mutations_cannot_pass(self) -> None:
        for changed in ("scripts/check-quality.py", "src/transitive.h", "rust/core/src/lib.rs",
                        "out/build/linux-debug/compile_commands.json", "src/new-generated.h"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                files = ["scripts/check-quality.py", "src/transitive.h", "rust/core/src/lib.rs",
                         "out/build/linux-debug/compile_commands.json"]
                for name in files:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("original", encoding="utf-8")
                for command in (["git", "init", "-q"], ["git", "add", "scripts", "src", "rust"],
                                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                                 "-c", "commit.gpgsign=false", "commit", "-qm", "baseline"]):
                    subprocess.run(command, cwd=root, check=True, capture_output=True)
                policy = check_quality.load_policy(SCRIPT_PATH.parent.parent)
                def mutate(*args):
                    (root / changed).write_text("changed", encoding="utf-8")
                    return 0
                with (patch.object(check_quality, "__file__", str(root / "scripts/check-quality.py")),
                      patch.object(sys, "argv", ["check-quality", "--preset", "linux-debug", "--files", "scripts/check-quality.py"]),
                      patch.object(check_quality, "load_policy", return_value=policy),
                      patch.object(check_quality, "run_checks", side_effect=mutate),
                      contextlib.redirect_stdout(io.StringIO()) as output):
                    with self.assertRaisesRegex(check_quality.QualityError, "inputs changed"):
                        check_quality.main()
                receipts = list((root / "out/validation").rglob("run.json"))
                self.assertEqual(len(receipts), 1)
                record = json.loads(receipts[0].read_text())
                self.assertEqual(record["status"], "failed")
                self.assertIn("git_head", record["inputs"])
                self.assertNotIn("Quality checks passed", output.getvalue())

    def test_consumed_external_input_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool = root / "tool"
            tool.write_bytes(b"original")
            runner = check_quality.Runner(root, root / "logs")
            runner.bind_inputs([tool])
            tool.write_bytes(b"changed")
            with self.assertRaisesRegex(check_quality.QualityError, "input changed"):
                runner.verify_inputs()


if __name__ == "__main__":
    unittest.main()
