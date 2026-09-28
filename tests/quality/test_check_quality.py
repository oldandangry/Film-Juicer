from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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
