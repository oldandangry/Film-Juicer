"""Step 1 completeness is checked as data; OS ownership gets small real witnesses."""
from __future__ import annotations

import copy
import hashlib
import json
import io
import os
from pathlib import Path
import shutil
import signal
import time
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import execution
from qualification import OPTICS_FUNCTION, OPTICS_TESTS, read_native_result, validate_native_result


class NativeResultTests(unittest.TestCase):
    read_bytes = 0
    written_bytes = 0

    @classmethod
    def tearDownClass(cls):
        print(f"Reader mutation I/O: {cls.read_bytes} logical bundle bytes read; {cls.written_bytes} bytes written (Python/OS I/O excluded).")

    @classmethod
    def setUpClass(cls):
        cls.read_bytes = cls.written_bytes = 0
        cls.rows = [f"row-{i}" for i in range(553)]
        cls.operations = [{"caller": "cpp", "id": "layout"}, {"caller": "c11", "id": "raw:null"}]
        native = [{"caller": c, "id": r} for c in ("c11", "cpp") for r in cls.rows] + cls.operations
        def events(cases):
            return b"".join((json.dumps({**r, "event": event, **({"outcome": "passed"} if event == "terminal" else {})}) + "\n").encode()
                            for r in cases for event in ("start", "terminal"))
        candidate = {"head": "source-baseline", "preset": "linux-debug", "configuration": {"CMAKE_BUILD_TYPE": "Debug", "BUILD_TESTING": "ON"}, "inputs": {"source.cpp": "identity"},
                     "fixtures": {"rows": cls.rows}, "original_source": "/unavailable/source", "original_build": "/unavailable/build"}
        cls.payloads = {
            "candidate.json": json.dumps(candidate).encode(), "candidate.patch": b"a reconstructible patch\n",
            "native-list.jsonl": b"".join((json.dumps(r) + "\n").encode() for r in native),
            "discovery.json": json.dumps({"rust-suites": {"film-juicer-core": {"testcases": {
                OPTICS_FUNCTION: {"ignored": False, "filter-match": {"status": "matches"}},
                "outside_selection": {"ignored": False, "filter-match": {"status": "mismatch"}}}}}}).encode(),
            "rust.jsonl": events([{"caller": "rust", "id": r} for r in cls.rows]), "native.jsonl": events(native),
            "nextest.xml": ('<testsuites><testsuite name="film-juicer-core"><testcase name="' + OPTICS_FUNCTION + '"/>'
                            '<testcase name="outside_selection"><skipped/></testcase></testsuite></testsuites>').encode(),
            "ctest.xml": ('<testsuite>' + ''.join(f'<testcase name="{n}" status="run"/>' for n in OPTICS_TESTS) + '</testsuite>').encode(),
            "nextest.toml": b'nextest-version = "0.9.143"\n',
            "rust-execution.json": b'{"commands":[{"returncode":0},{"returncode":0}]}',
        }
        cls.result = {"schema": "native-spatial-optics-v1", "preset": "linux-debug", "status": "passed",
                       "candidate_head": "source-baseline", "shipping_isolation": False, "candidate_unchanged": True,
                       "membership": {"rows": cls.rows, "native_operations": cls.operations,
                                      "ctest": OPTICS_TESTS, "rust_function": OPTICS_FUNCTION},
                       "commands": [{"returncode": 0, "drained": True, "process_status": "completed"}], "members": {}}

        actual = os.environ.get("FJ_OPTICS_COMPLETED_BUNDLE")
        if actual:
            source = Path(actual)
            raw = (source / "result.json").read_bytes()
            cls.result = json.loads(raw)
            cls.payloads = {name: (source / name).read_bytes() for name in cls.result["members"]}
            cls.read_bytes += len(raw) + sum(map(len, cls.payloads.values()))
            cls.rows = cls.result["membership"]["rows"]
            cls.operations = cls.result["membership"]["native_operations"]
        cls.template_result = cls.result
        cls.template_payloads = cls.payloads

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="optics results ü ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        original_read = Path.read_bytes
        original_write = Path.write_bytes
        def read(path):
            data = original_read(path)
            type(self).read_bytes += len(data)
            return data
        def write(path, data):
            size = original_write(path, data)
            type(self).written_bytes += size
            return size
        for method, replacement in (("read_bytes", read), ("write_bytes", write)):
            patcher = patch.object(Path, method, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.bundle = self.root / "relocated bundle ü"
        self.bundle.mkdir()
        self.result = copy.deepcopy(type(self).template_result)
        self.payloads = dict(type(self).template_payloads)  # Bytes remain immutable.
        self.rows = self.result["membership"]["rows"]
        self.operations = self.result["membership"]["native_operations"]

    def save(self):
        self.result["members"] = {}
        for name, data in self.payloads.items():
            (self.bundle / name).write_bytes(data)
            self.result["members"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        (self.bundle / "result.json").write_bytes(json.dumps(self.result).encode())

    def reject(self):
        # Semantic mutations exercise the same pure validator as publication and
        # the reader, without rewriting an entire fixture bundle for every row.
        self.result["members"] = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                                  for name, data in self.payloads.items()}
        with self.assertRaises((ValueError, KeyError, OSError)):
            validate_native_result(self.result, self.payloads)

    def test_complete_results_and_expired_optional_diagnostics(self):
        self.save()
        result = read_native_result(self.bundle)
        self.assertEqual(result["rust_rows"], 553)
        self.assertEqual(result["native_rows"], 1106)
        self.assertFalse(result["shipping_isolation"])
        self.assertEqual(result["bookkeeping_read_bytes"], sum(p.stat().st_size for p in self.bundle.iterdir()))

    def test_damaged_member_rejected_by_offline_reader(self):
        self.save()
        (self.bundle / "rust.jsonl").write_bytes(b"damaged\n")
        with self.assertRaisesRegex(ValueError, "damaged bundle member"):
            read_native_result(self.bundle)

    def test_gpu_witness_cannot_pass_without_shared_exclusion(self):
        from qualification import GPU_WITNESS_TEST
        self.result["gpu_witness"] = {"test": GPU_WITNESS_TEST, "scope": "containment-and-shared-lock-only",
                                      "status": "passed", "contender_excluded": False, "drained_while_locked": True}
        self.payloads["gpu-witness.xml"] = ('<testsuite><testcase name="' + GPU_WITNESS_TEST + '" status="run"/></testsuite>').encode()
        self.reject()

    def test_gpu_witness_requires_its_case_results(self):
        from qualification import GPU_WITNESS_TEST
        self.result["gpu_witness"] = {"test": GPU_WITNESS_TEST, "scope": "containment-and-shared-lock-only",
                                      "status": "passed", "contender_excluded": True, "drained_while_locked": True}
        self.payloads.pop("gpu-witness.xml", None)
        self.reject()

    def test_missing_rust_function(self):
        self.payloads["discovery.json"] = b'{"rust-suites":{}}'
        self.reject()

    def test_missing_rust_row(self):
        self.payloads["rust.jsonl"] = b'\n'.join(self.payloads["rust.jsonl"].splitlines()[:-2]) + b'\n'
        self.reject()

    def test_missing_native_caller(self):
        self.payloads["native.jsonl"] = b'\n'.join(line for line in self.payloads["native.jsonl"].splitlines() if json.loads(line)["caller"] != "c11") + b'\n'
        self.reject()

    def test_missing_native_row(self):
        self.payloads["native.jsonl"] = b'\n'.join(self.payloads["native.jsonl"].splitlines()[2:]) + b'\n'
        self.reject()

    def test_missing_named_operation(self):
        self.payloads["native.jsonl"] = b'\n'.join(self.payloads["native.jsonl"].splitlines()[:-2]) + b'\n'
        self.reject()

    def test_native_listing_omits_row(self):
        self.payloads["native-list.jsonl"] = b'\n'.join(self.payloads["native-list.jsonl"].splitlines()[1:]) + b'\n'
        self.reject()

    def test_duplicate_terminal(self):
        self.payloads["rust.jsonl"] += self.payloads["rust.jsonl"].splitlines()[-1] + b'\n'
        self.reject()

    def test_crash_before_terminal(self):
        self.payloads["rust.jsonl"] = b'\n'.join(self.payloads["rust.jsonl"].splitlines()[:-1]) + b'\n'
        self.reject()

    def test_truncated_output(self):
        self.payloads["native.jsonl"] = self.payloads["native.jsonl"][:-5]
        self.reject()

    def test_nonzero_exit(self):
        self.result["commands"][0]["returncode"] = 9
        self.reject()

    def test_ignored_selected_function(self):
        document = json.loads(self.payloads["discovery.json"])
        for suite in document["rust-suites"].values():
            suite["testcases"][OPTICS_FUNCTION]["ignored"] = True
        self.payloads["discovery.json"] = json.dumps(document).encode()
        self.reject()

    def test_unexpected_skip(self):
        import xml.etree.ElementTree as ET
        tree = ET.fromstring(self.payloads["nextest.xml"])
        for case in tree.iter("testcase"):
            if case.get("name") == OPTICS_FUNCTION:
                ET.SubElement(case, "skipped")
        self.payloads["nextest.xml"] = ET.tostring(tree)
        self.reject()

    def test_failed_row_despite_zero_exit(self):
        self.payloads["native.jsonl"] = self.payloads["native.jsonl"].replace(b'"passed"', b'"failed"', 1)
        self.reject()

    def test_missing_junit(self):
        del self.payloads["nextest.xml"]
        self.reject()

    def test_candidate_mismatch(self):
        self.result["candidate_head"] = "different"
        self.reject()

    def test_actual_build_type_mismatch(self):
        candidate = json.loads(self.payloads["candidate.json"])
        candidate["configuration"]["CMAKE_BUILD_TYPE"] = "Release"
        self.payloads["candidate.json"] = json.dumps(candidate).encode()
        self.reject()

    def test_failed_bundle_collection_exits_nonzero(self):
        import verify
        self.result["status"] = "failed"
        self.save()
        output = self.root / "report.json"
        with patch.object(sys, "argv", ["verify.py", "collect-qualification", "--bundles", str(self.bundle), "--output", str(output)]):
            self.assertEqual(verify.main(), 1)
        self.assertEqual(json.loads(output.read_bytes())["status"], "failed")

    def test_invalid_membership_rejected_before_payload_reads(self):
        self.save()
        self.result["membership"]["rows"] = self.rows[:-1]
        raw = json.dumps(self.result).encode()
        with patch.object(Path, "read_bytes", side_effect=AssertionError("payload read")):
            with self.assertRaisesRegex(ValueError, "membership"):
                read_native_result(self.bundle, result_bytes=raw)

    def test_path_escape_rejected_before_payload_reads(self):
        self.save()
        self.result["members"]["../source"] = {"bytes": 0, "sha256": "0" * 64}
        with patch.object(Path, "read_bytes", side_effect=AssertionError("payload read")):
            with self.assertRaises(ValueError):
                read_native_result(self.bundle, result_bytes=json.dumps(self.result).encode())

    def test_missing_publication_cannot_pass(self):
        self.save()
        (self.bundle / "result.json").unlink()
        with self.assertRaises(FileNotFoundError):
            read_native_result(self.bundle)

    def test_reader_copy_runs_offline_without_tools_or_original_paths(self):
        self.save()
        reader = self.root / "reader.py"
        shutil.copyfile(Path(__file__).resolve().parents[2] / "scripts/qualification.py", reader)
        code = r'''
import sys, os, runpy
from pathlib import Path
root = Path(sys.argv[1]).resolve()
stdlib = Path(os.__file__).resolve().parent
opens = []
def audit(event, args):
    if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec'}:
        raise AssertionError('offline reader launched a tool')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = Path(os.fsdecode(args[0])).resolve()
        if not (p.is_relative_to(root) or p.is_relative_to(stdlib)):
            raise AssertionError('offline reader accessed original machine: ' + str(p))
        if p.suffix.lower() in {'.exe', '.dll', '.a', '.lib', '.ofx'}:
            raise AssertionError('offline reader read executable bytes')
        opens.append(str(p))
sys.addaudithook(audit)
sys.argv = [str(root / 'reader.py'), '--bundle', str(root / 'relocated bundle ü')]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as exc:
    if exc.code:
        raise
for p in (root / 'relocated bundle ü').iterdir():
    assert opens.count(str(p)) == 1, (p, opens.count(str(p)))
'''
        completed = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(self.root)],
                                   capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        type(self).read_bytes += json.loads(completed.stdout)["bookkeeping_read_bytes"]


class VersionProbeTests(unittest.TestCase):
    def test_failure_and_success_are_both_recorded_with_exact_arguments(self):
        import argparse
        import verify
        output = io.StringIO()
        commands = []
        def probe(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 7 if command[0] == "failed tool" else 0, "version output")
        with patch.object(verify.subprocess, "run", side_effect=probe), patch.object(sys, "stdout", output):
            status = verify.tool_versions(argparse.Namespace(tool=[("cargo", "failed tool"), ("rustc", "proxy path/rustc")]))
        records = json.loads(output.getvalue())
        self.assertEqual(status, 1)
        self.assertEqual([r["returncode"] for r in records], [7, 0])
        self.assertIn(["proxy path/rustc", "-Vv"], commands)
        self.assertIn(["failed tool", "--version"], commands)

    def test_missing_tool_is_a_failed_probe_and_keeps_other_results(self):
        import argparse
        import verify
        output = io.StringIO()
        def probe(command, **kwargs):
            if command[0] == "missing":
                raise FileNotFoundError("missing tool")
            return subprocess.CompletedProcess(command, 0, "version output")
        with patch.object(verify.subprocess, "run", side_effect=probe), patch.object(sys, "stdout", output):
            status = verify.tool_versions(argparse.Namespace(tool=[("cargo", "missing"), ("cmake", "present")]))
        records = json.loads(output.getvalue())
        self.assertEqual(status, 1)
        self.assertIn("error", records[0])
        self.assertEqual(records[1]["returncode"], 0)


class OpticsSourceTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="optics source ü ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.literal = "tests/film/optics-fixtures/literal.json"
        self.git("init", "-q")
        policy = Path(__file__).resolve().parents[2] / "scripts/source_file_policy.json"
        (self.root / "scripts").mkdir()
        (self.root / "scripts/source_file_policy.json").write_bytes(policy.read_bytes())
        for name in ("src/producer.h", "rust/producer.rs", "tests/ffi/consumer.inc", self.literal,
                     "tests/route/development.json", "tests/quality/test_native_results.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("baseline\n")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root)

    def capture(self):
        from verify import capture_native_source
        return capture_native_source(self.root, self.literal)

    @unittest.skipUnless(os.name == "posix", "POSIX executable mode control")
    def test_new_input_modes_follow_git_filemode_policy(self):
        source = self.root / "src/new_header.h"
        source.write_text("new source\n")
        self.git("config", "core.filemode", "false")
        source.chmod(0o644)
        first = self.capture()[0]
        source.chmod(0o755)
        second = self.capture()[0]
        self.assertEqual(first, second)
        self.assertEqual(second["inputs"]["src/new_header.h"]["mode"], "100644")
        self.git("config", "core.filemode", "true")
        self.assertEqual(self.capture()[0]["inputs"]["src/new_header.h"]["mode"], "100755")

    def test_tracked_inputs_use_git_without_reopening_each_source(self):
        policy = self.root / "scripts/source_file_policy.json"
        data = policy.read_bytes()
        def read(path):
            if path == policy:
                return data
            raise AssertionError("tracked source reread")
        with patch.object(Path, "read_bytes", read):
            first = self.capture()
            second = self.capture()
        self.assertEqual(first[:2], second[:2])
        self.assertEqual(first[2]["source_read_bytes"], len(data))
        self.assertIn("tests/route/development.json", first[0]["inputs"])

    def test_tracked_dirty_patch_detects_same_length_edit_and_reconstructs(self):
        before = self.capture()
        (self.root / "src/producer.h").write_text("modified\n")
        after = self.capture()
        self.assertNotEqual(before[1], after[1])
        self.assertEqual(after[0]["head"], before[0]["head"])
        subprocess.run(["git", "apply", "--reverse", "--check"], cwd=self.root, input=after[1], check=True)

    def test_rapid_same_size_edit_after_staging_and_capture_is_detected(self):
        path = self.root / "src/producer.h"
        same_second = 0
        for _ in range(8):
            path.write_bytes(b"entry words\n")
            self.git("add", "src/producer.h")
            before = self.capture()
            timestamp = path.stat().st_mtime_ns // 1_000_000_000
            path.write_bytes(b"other words\n")
            same_second += timestamp == path.stat().st_mtime_ns // 1_000_000_000
            self.assertNotEqual(before[1], self.capture()[1])
        self.assertGreater(same_second, 0, "no same-second mutation was exercised")

    def test_new_input_bytes_and_deletion_change_candidate(self):
        import base64
        before = self.capture()
        path = self.root / "rust/new.rs"
        path.write_bytes(b"new source\n")
        after = self.capture()
        self.assertNotEqual(before[0], after[0])
        self.assertEqual(base64.b64decode(after[0]["new_files_base64"]["rust/new.rs"]), path.read_bytes())
        path.write_bytes(b"changed source\n")
        self.assertNotEqual(after[0], self.capture()[0])
        (self.root / "src/producer.h").unlink()
        self.assertNotEqual(before[1], self.capture()[1])

    def test_staged_deletion_remains_in_reconstructible_patch(self):
        self.git("rm", "-q", "src/producer.h")
        source, patch_bytes, _ = self.capture()
        self.assertNotIn("src/producer.h", source["inputs"])
        self.assertIn(b"deleted file mode", patch_bytes)
        subprocess.run(["git", "apply", "--reverse", "--check"], cwd=self.root, input=patch_bytes, check=True)

    def test_unrelated_scientific_payload_is_outside_pilot_inputs(self):
        before = self.capture()
        path = self.root / "tests/unrelated/array.json"
        path.parent.mkdir(parents=True)
        path.write_text("unrelated\n")
        self.assertEqual(before[:2], self.capture()[:2])

    def test_linked_support_in_other_test_domain_is_captured(self):
        before = self.capture()
        path = self.root / "tests/dir/boundary_test_hooks.cpp"
        path.parent.mkdir(parents=True)
        path.write_text("linked support\n")
        self.git("add", str(path.relative_to(self.root)))
        after = self.capture()
        self.assertIn("tests/dir/boundary_test_hooks.cpp", after[0]["inputs"])
        self.assertNotEqual(before[1], after[1])
        path.write_text("changed support\n")
        self.assertNotEqual(after[1], self.capture()[1])

    def test_supported_untracked_headers_are_preserved_and_compared(self):
        import base64
        self.subtest_cases = tuple(f"extension/{extension}" for extension in (".cuh", ".hh", ".hxx", ".inl", ".ixx"))
        for extension in (".cuh", ".hh", ".hxx", ".inl", ".ixx"):
            with self.subTest(case=f"extension/{extension}", extension=extension):
                path = self.root / ("src/new" + extension)
                path.write_bytes(b"new header\n")
                before = self.capture()
                name = path.relative_to(self.root).as_posix()
                self.assertEqual(base64.b64decode(before[0]["new_files_base64"][name]), path.read_bytes())
                path.write_bytes(b"changed header\n")
                self.assertNotEqual(before[0], self.capture()[0])
                path.unlink()

    @unittest.skipUnless(os.name == "posix", "native symlink worktree witness")
    def test_unstaged_regular_file_to_symlink_is_rejected(self):
        target = self.root / "target"
        target.write_text("external bytes\n")
        path = self.root / "src/producer.h"
        path.unlink()
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "worktree source must be a regular file"):
            self.capture()

    @unittest.skipUnless(os.name == "posix", "native linked-directory witness")
    def test_linked_source_directory_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "producer.h").write_text("outside bytes\n")
        shutil.rmtree(self.root / "src")
        (self.root / "src").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "worktree source must be a regular file"):
            self.capture()

    @unittest.skipUnless(os.name == "posix", "native nested linked-directory witness")
    def test_new_linked_test_directory_is_rejected_before_suffix_filter(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "new.h").write_text("outside bytes\n")
        (self.root / "tests/new_support").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "worktree source must be a regular file"):
            self.capture()

    @unittest.skipUnless(os.name == "nt", "Windows junction source witness")
    def test_windows_source_junction_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "producer.h").write_text("outside bytes\n")
        shutil.rmtree(self.root / "src")
        subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(self.root / "src"), str(outside)], check=True, capture_output=True)
        with self.assertRaisesRegex(ValueError, "worktree source must be a regular file"):
            self.capture()

    def test_gpu_witness_records_resource_inputs_without_copying_arrays(self):
        from verify import capture_native_source
        path = self.root / "Resources/profiles/selected.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"resource bytes\n")
        self.git("add", "Resources")
        before = capture_native_source(self.root, self.literal, gpu_witness=True)
        self.assertIn("Resources/profiles/selected.json", before[0]["inputs"])
        path.write_bytes(b"changed resource bytes\n")
        after = capture_native_source(self.root, self.literal, gpu_witness=True)
        self.assertNotEqual(before[1], after[1])

    def test_index_flags_cannot_hide_selected_mutation(self):
        self.subtest_cases = ("flag/assume-unchanged", "flag/skip-worktree")
        for flag in ("assume-unchanged", "skip-worktree"):
            with self.subTest(case=f"flag/{flag}", flag=flag):
                self.git("update-index", "--" + flag, "src/producer.h")
                with self.assertRaisesRegex(ValueError, "ordinary, merged Git tracking"):
                    self.capture()
                self.git("update-index", "--no-" + flag, "src/producer.h")

    test_supported_untracked_headers_are_preserved_and_compared.required_subtests = True
    test_index_flags_cannot_hide_selected_mutation.required_subtests = True


class NativeOwnerTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="native owner ü ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name).resolve()
        self.directory = self.root / "result"
        self.directory.mkdir()

    def command(self, code):
        python = str(Path(sys.base_prefix) / "python.exe") if os.name == "nt" else sys.executable
        return [python, "-c", code]

    @unittest.skipUnless(os.name == "nt", "Windows compiler telemetry job policy")
    def test_native_compiler_cleanup_drains_its_exact_telemetry_child(self):
        vctip = self.root / "vctip.exe"
        shutil.copyfile(Path(os.environ["SystemRoot"]) / "System32/ping.exe", vctip)
        code = f"import subprocess;subprocess.Popen([{str(vctip)!r},'-n','30','127.0.0.1'])"
        with execution.OutputLease(self.root, self.root / "build"):
            run = execution.NativeRun(self.root, self.directory)
            run.command("compile", self.command(code), timeout=5, vctip_executable=vctip)
            self.assertTrue(run.commands[0]["drained"])
            self.assertEqual(len(run.commands[0]["vctip_terminated"]), 1)
            self.assertEqual(Path(run.commands[0]["vctip_terminated"][0]["path"]), vctip)
            self.assertEqual(run.commands[0]["vctip_cleanup"]["grace_seconds"], 1)
            run.command("test", self.command("pass"), timeout=5)
            self.assertNotIn("vctip_cleanup", run.commands[1])

    def test_failure_records_exit_and_drains_without_legacy_admission(self):
        with execution.OutputLease(self.root, self.root / "build"), patch.object(execution, "digest", side_effect=AssertionError("tool hashing")):
            run = execution.NativeRun(self.root, self.directory)
            with self.assertRaises(subprocess.CalledProcessError):
                run.command("failure", self.command("raise SystemExit(7)"))
            self.assertEqual(run.commands[0]["returncode"], 7)
            self.assertTrue(run.commands[0]["drained"])
            self.assertFalse((self.directory / "result.json").exists())

    def test_successful_parent_waits_for_descendant(self):
        child = "import time; time.sleep(.15); print('descendant finished',flush=True)"
        code = "import subprocess,sys;subprocess.Popen([sys.executable,'-c'," + repr(child) + "])"
        with execution.OutputLease(self.root, self.root / "build"):
            run = execution.NativeRun(self.root, self.directory)
            log = run.command("descendant", self.command(code), timeout=10)
            self.assertIn("descendant finished", log.read_text())
            self.assertTrue(run.commands[0]["drained"])

    def test_cancellation_drains_and_does_not_publish(self):
        with execution.OutputLease(self.root, self.root / "build"):
            run = execution.NativeRun(self.root, self.directory)
            with self.assertRaises(subprocess.TimeoutExpired):
                run.command("cancel", self.command("import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(30)"), timeout=.2)
            self.assertTrue(run.commands[0]["drained"])
            self.assertFalse((self.directory / "result.json").exists())

    def test_owner_cancellation_drains_started_descendant_on_both_hosts(self):
        marker = self.root / "descendant started"
        child = "import pathlib,time;pathlib.Path(" + repr(str(marker)) + ").write_text('ready');time.sleep(30)"
        command = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c'," + repr(child) + "]);time.sleep(30)"
        original = execution.subprocess.Popen
        def launch(*args, **kwargs):
            process = original(*args, **kwargs)
            wait = process.wait
            first = True
            def cancel(timeout=None):
                nonlocal first
                if first:
                    first = False
                    deadline = time.monotonic() + 10
                    while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue(marker.exists(), "real descendant did not start")
                    raise KeyboardInterrupt
                return wait(timeout=timeout)
            process.wait = cancel
            return process
        with execution.OutputLease(self.root, self.root / "build"):
            run = execution.NativeRun(self.root, self.directory)
            with patch.object(execution.subprocess, "Popen", side_effect=launch):
                with self.assertRaises(KeyboardInterrupt):
                    run.command("cancel-owner", self.command(command))
            self.assertTrue(run.commands[0]["drained"])
            self.assertEqual(run.commands[0]["process_status"], "interrupted")
            self.assertFalse((self.directory / "result.json").exists())

    @unittest.skipUnless(os.name == "posix", "Step 1 Linux owner interruption witness")
    def test_manual_owner_interruption_drains_descendants(self):
        started = self.root / "started"
        child = "import pathlib,time;pathlib.Path(" + repr(str(started)) + ").write_text('started');time.sleep(30)"
        owner = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from execution import NativeRun,OutputLease
root=Path(sys.argv[2])
with OutputLease(root,root/'build'):
    run=NativeRun(root,root/'result')
    try:
        run.command('interrupt',[sys.executable,'-c',sys.argv[3]])
    except KeyboardInterrupt:
        pass
    (root/'observed.json').write_text(json.dumps(run.commands))
"""
        process = subprocess.Popen([sys.executable, "-B", "-c", owner, str(Path(execution.__file__).parent), str(self.root), child],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while not started.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(started.exists(), "contained child did not start")
            process.send_signal(signal.SIGINT)
            _, error = process.communicate(timeout=25)
            self.assertEqual(process.returncode, 0, error.decode())
            records = json.loads((self.root / "observed.json").read_text())
            self.assertTrue(records[0]["drained"])
            self.assertEqual(records[0]["process_status"], "interrupted")
            self.assertFalse((self.directory / "result.json").exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=20)

    def test_crash_after_case_start_never_publishes(self):
        events = self.directory / "partial.jsonl"
        code = "import pathlib,os;pathlib.Path(" + repr(str(events)) + ").write_text('start\\n');os._exit(9)"
        with execution.OutputLease(self.root, self.root / "build"):
            run = execution.NativeRun(self.root, self.directory)
            with self.assertRaises(subprocess.CalledProcessError):
                run.command("crash", self.command(code))
            self.assertEqual(events.read_text(), "start\n")
            self.assertTrue(run.commands[0]["drained"])
            self.assertFalse((self.directory / "result.json").exists())

    def test_uncertain_drainage_blocks_owner_and_publication(self):
        lease = execution.OutputLease(self.root, self.root / "build")
        with lease:
            run = execution.NativeRun(self.root, self.directory)
            def uncertain(*args):
                args[-1]["drained"] = False
                raise RuntimeError("uncertain child")
            with patch.object(execution, "run_process_tree", side_effect=uncertain):
                with self.assertRaises(RuntimeError):
                    run.command("uncertain", self.command("pass"))
            self.assertTrue(lease.blocked)
            with self.assertRaises(RuntimeError):
                run.publish({"status": "passed"})
        self.assertEqual(json.loads((lease.path / "owner.json").read_text())["status"], "blocked")

    def test_atomic_publication_failure_leaves_no_completed_result(self):
        with execution.OutputLease(self.root, self.root / "build"):
            run = execution.NativeRun(self.root, self.directory)
            with patch.object(Path, "replace", side_effect=OSError("interrupted publication")):
                with self.assertRaises(OSError):
                    run.publish({"status": "passed"})
            self.assertFalse((self.directory / "result.json").exists())
            run.publish({"status": "failed"})
            with self.assertRaises(ValueError):
                run.publish({"status": "passed"})

    def test_gpu_cancellation_drains_before_releasing_both_owners(self):
        with execution.OutputLease(self.root, self.root / "build") as build_owner:
            with execution.OutputLease(self.root) as gpu_owner:
                run = execution.NativeRun(self.root, self.directory)
                with self.assertRaises(subprocess.TimeoutExpired):
                    run.command("gpu-cancel", self.command("import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(30)"), timeout=.2)
                self.assertTrue(run.commands[0]["drained"])
                self.assertTrue((gpu_owner.path / "owner.json").exists())
                self.assertTrue((build_owner.path / "owner.json").exists())
                self.assertFalse((self.directory / "result.json").exists())
            self.assertFalse(gpu_owner.path.exists())
        self.assertFalse(build_owner.path.exists())

    def test_uncertain_gpu_descendants_block_both_owners(self):
        with execution.OutputLease(self.root, self.root / "build") as build_owner:
            with execution.OutputLease(self.root) as gpu_owner:
                run = execution.NativeRun(self.root, self.directory)
                def uncertain(*args):
                    args[-1]["drained"] = False
                    raise RuntimeError("uncertain GPU child")
                with patch.object(execution, "run_process_tree", side_effect=uncertain):
                    with self.assertRaisesRegex(RuntimeError, "uncertain"):
                        run.command("gpu-uncertain", self.command("pass"))
                self.assertTrue(build_owner.blocked and gpu_owner.blocked)
                with self.assertRaises(RuntimeError):
                    run.publish({"status": "passed"})
        for owner in (build_owner, gpu_owner):
            self.assertEqual(json.loads((owner.path / "owner.json").read_text())["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
