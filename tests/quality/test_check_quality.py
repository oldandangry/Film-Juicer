from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
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
                patch.object(check_quality, "check_source_hygiene"),
                patch.object(check_quality, "check_rust") as rust,
                patch.object(check_quality, "check_native") as native,
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
                patch.object(check_quality, "check_source_hygiene"),
                patch.object(check_quality, "check_rust", side_effect=check_quality.QualityError("Rust check failed")),
                patch.object(check_quality, "check_native") as native,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                with self.assertRaisesRegex(check_quality.QualityError, "Rust check failed"):
                    check_quality.main()
                native.assert_not_called()
                self.assertNotIn("Quality checks passed", output.getvalue())

    def test_policy_changes_cannot_skip_required_checks(self) -> None:
        for path in ("Cargo.toml", "rust/film-juicer-core/Cargo.toml", "tests/quality/test_rust_boundaries.py", "scripts/check-quality.py"):
            with (
                self.subTest(path=path),
                patch.object(sys, "argv", [str(SCRIPT_PATH), "--preset", "linux-debug", "--files", path]),
                patch.object(check_quality, "Runner") as runner,
                patch.object(check_quality, "check_source_hygiene"),
                patch.object(check_quality, "check_rust") as rust,
                patch.object(check_quality, "check_native"),
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
            patch.object(check_quality, "check_source_hygiene"),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            runner.return_value.run.side_effect = ["", check_quality.QualityError("boundary violation")]
            with self.assertRaisesRegex(check_quality.QualityError, "boundary violation"):
                check_quality.main()
            self.assertNotIn("Quality checks passed", output.getvalue())

    def test_cuda_header_selects_real_c_and_cpp_consumers(self) -> None:
        root = SCRIPT_PATH.parent.parent
        policy = check_quality.load_policy(root)
        entries = [
            check_quality.CompilationEntry("tests/ffi/cuda_abi_c.c", "cc -std=c11"),
            check_quality.CompilationEntry("tests/ffi/cuda_abi_test.cpp", "c++ -std=c++20"),
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

    def test_all_findings_are_observed_before_parallel_analysis_fails(self) -> None:
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
            self.assertEqual(len(list(runner.log_dir.glob("*.log"))), 3)

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
        entries = [self.entry(), self.entry(), check_quality.CompilationEntry("src/host.cpp", "c++")]
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
                check_quality.check_native(self.root, runner, arguments, ["src/kernel.cuh"], self.policy)
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
                report = self.root / "out/validation/linux-debug/quality/source-hygiene.log"
                self.assertIn("JUICER_BUILD_VALIDATION", report.read_text(encoding="utf-8"))
                if selection[0] == "--base":
                    self.assertIn("src/untracked.cpp:1:8:", result.stdout)


if __name__ == "__main__":
    unittest.main()
