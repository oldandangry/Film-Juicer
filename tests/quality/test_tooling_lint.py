from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tooling", ROOT / "scripts/check-tooling.py")
tooling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tooling)


class ToolingLintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True)
        self.write("ruff.toml", (ROOT / "ruff.toml").read_text(encoding="utf-8"))
        self.write("scripts/space name.py", "value = 1\n")
        self.write("scripts/check.sh", "#!/usr/bin/env bash\nprintf '%s\\n' hello\n")
        self.write(".github/workflows/check.yaml", "name: Test\non: push\njobs:\n  check:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hello\n")
        self.tools = {}
        for name in ("ruff", "shellcheck", "actionlint"):
            path = self.write("executables/" + name + (".exe" if sys.platform == "win32" else ""), "tool")
            path.chmod(0o755)
            self.tools[name] = str(path)
        self.real_run = subprocess.run
        self.commands = []
        self.mutation = None
        self.failure = None

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def run_tool(self, argv, **kwargs):
        if argv[0] == "git":
            return self.real_run(argv, **kwargs)
        self.commands.append(argv)
        if argv[1] in {"--version", "-version"}:
            versions = {"ruff": "ruff 0.15.22\n", "shellcheck": "ShellCheck\nversion: 0.11.0\n", "actionlint": "1.7.12\n"}
            return subprocess.CompletedProcess(argv, 0, versions[Path(argv[0]).stem], "")
        if self.mutation:
            mutation, self.mutation = self.mutation, None
            mutation()
        if self.failure:
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0)

    def check(self):
        with mock.patch.object(tooling.subprocess, "run", side_effect=self.run_tool):
            return tooling.check(self.root, **self.tools)

    def test_tracked_untracked_and_excluded_scope(self) -> None:
        subprocess.run(["git", "add", "scripts/space name.py"], cwd=self.root, check=True)
        self.write("tests/new.py", "pass\n")
        for name in ("tests/fixtures/bad.py", "tests/generated/bad.py", "tests/vendor/bad.py", "scripts/vendored/bad.sh", ".juicer-local/private.py", "out/bad.py", "external/bad.py", "Resources/bad.py", "src/bad.py"):
            self.write(name, "invalid !")
        self.write(".gitignore", "scripts/ignored.py\n")
        self.write("scripts/ignored.py", "invalid !")
        files = tooling.source_files(self.root)
        self.assertEqual(files, {"python": ["scripts/space name.py", "tests/new.py"], "shell": ["scripts/check.sh"], "workflow": [".github/workflows/check.yaml"]})

    def test_deleted_tracked_file_is_outside_current_membership(self) -> None:
        path = self.root / "scripts/space name.py"
        subprocess.run(["git", "add", str(path)], cwd=self.root, check=True)
        path.unlink()
        self.assertEqual(tooling.source_files(self.root)["python"], [])
        self.assertEqual(self.check()["python"], 0)

    def test_valid_inputs_pass_unchanged_with_argument_arrays(self) -> None:
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(self.check(), {"python": 1, "shell": 1, "workflow": 1})
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        ruff = next(c for c in self.commands if c[1] == "check")
        self.assertIn("scripts/space name.py", ruff)
        self.assertIn("--no-cache", ruff)
        self.assertNotIn("--fix", ruff)
        shell = next(c for c in self.commands if "--external-sources" in c)
        self.assertIn("--source-path=SCRIPTDIR", shell)
        workflow = next(c for c in self.commands if "-shellcheck" in c)
        self.assertEqual(workflow[workflow.index("-shellcheck") + 1], str(Path(self.tools['shellcheck']).resolve()))
        self.assertEqual(workflow[workflow.index("-pyflakes") + 1], "")

    def test_missing_tools_fail(self) -> None:
        for name in self.tools:
            with self.subTest(tool=name), self.assertRaisesRegex(tooling.ToolingError, "missing"):
                tooling.require_tool(name, str(self.root / "absent"), "--version", "unused")

    def test_wrong_versions_fail(self) -> None:
        for name in self.tools:
            with self.subTest(tool=name), mock.patch.object(tooling.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "wrong 99", "")), self.assertRaisesRegex(tooling.ToolingError, "wrong"):
                tooling.require_tool(name, self.tools[name], "--version", r"^expected$")

    def test_tool_failures_fail_the_gate(self) -> None:
        self.failure = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.check()

    def test_source_edit_during_lint_fails(self) -> None:
        self.mutation = lambda: self.write("scripts/space name.py", "value = 2\n")
        with self.assertRaisesRegex(tooling.ToolingError, "changed"):
            self.check()

    def test_config_edit_during_lint_fails(self) -> None:
        self.mutation = lambda: self.write("ruff.toml", 'target-version = "py312"\n')
        with self.assertRaisesRegex(tooling.ToolingError, "changed"):
            self.check()

    def test_new_eligible_file_during_lint_fails(self) -> None:
        self.mutation = lambda: self.write("tests/added.py", "pass\n")
        with self.assertRaisesRegex(tooling.ToolingError, "changed"):
            self.check()

    def test_tool_edit_during_lint_fails(self) -> None:
        self.mutation = lambda: Path(self.tools['ruff']).write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(tooling.ToolingError, "changed"):
            self.check()

    def test_missing_selected_input_fails(self) -> None:
        self.mutation = lambda: (self.root / "scripts/space name.py").unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_same_byte_retarget_during_final_hash_fails(self) -> None:
        source = self.root / "scripts/space name.py"
        target = self.write("target.txt", source.read_text(encoding="utf-8"))
        digest, resolve = tooling.digest, Path.resolve
        reads = 0
        retargeted = False
        def read(path):
            nonlocal reads, retargeted
            checksum = digest(path)
            if path == source:
                reads += 1
                retargeted = reads == 2
            return checksum
        def identity(path, *args, **kwargs):
            return target if path == source and retargeted else resolve(path, *args, **kwargs)
        with mock.patch.object(tooling, "digest", read), mock.patch.object(Path, "resolve", identity), self.assertRaisesRegex(tooling.ToolingError, "changed"):
            self.check()

    def test_outside_repository_input_fails(self) -> None:
        resolve = Path.resolve
        outside = self.root.parent / "outside.py"
        def redirect(path, *args, **kwargs):
            return outside if path == self.root / "scripts/space name.py" else resolve(path, *args, **kwargs)
        with mock.patch.object(Path, "resolve", redirect), self.assertRaisesRegex(tooling.ToolingError, "outside"):
            tooling.source_files(self.root)

    def test_empty_scope_still_verifies_tools(self) -> None:
        shutil.rmtree(self.root / "scripts")
        shutil.rmtree(self.root / ".github")
        self.assertEqual(self.check(), {"python": 0, "shell": 0, "workflow": 0})
        self.assertEqual(len(self.commands), 3)


if __name__ == "__main__":
    unittest.main()
